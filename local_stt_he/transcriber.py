"""
core/transcriber.py
-------------------
Unified transcription engine. Accepts a config dict from benchmark.get_best_config().
Dispatches to faster-whisper (CPU/CUDA) or openvino_genai backend.
Whisper parameters are resolved per-call via core.params.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

ROOT = Path(__file__).parent.parent
MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
OV_MODEL_DIR = "models_ov/whisper-large-v3-turbo-ov"


def _ov_set(cfg, attr: str, val) -> None:
    """Set a WhisperGenerateConfig attribute only if it exists in this openvino_genai version."""
    if hasattr(cfg, attr):
        setattr(cfg, attr, val)


@dataclass
class TranscriptResult:
    text: str
    segments: list
    audio_duration: float
    elapsed: float
    rtf: float
    config_label: str
    backend: str
    tier_used: str
    whisper_params: dict


class Transcriber:
    def __init__(self, config: dict):
        """
        config dict from benchmark.get_best_config(bucket).
        Keys: device, compute_type, cpu_threads, num_workers.
        For OpenVINO: also openvino_device.
        Optional: venv_path — if present, inference runs inside a venv worker subprocess.

        Applies resource profile before loading the backend.
        """
        from local_stt_he.resource import apply_profile, check_memory_headroom, check_vram_before_load
        config = apply_profile(config)
        config = check_memory_headroom(config)
        config = check_vram_before_load(config)
        self._config = config

        # Load per-session config flags (read once at construction time)
        self._language = "he"
        self._vad_filter = True
        self._vad_min_silence_ms = 300
        self._vad_speech_pad_ms = 200
        try:
            import yaml as _yaml
            _cfg_path = ROOT / "config.yaml"
            if _cfg_path.exists():
                _ucfg = _yaml.safe_load(_cfg_path.read_text()) or {}
                self._language = _ucfg.get("language", "he")
                self._vad_filter = _ucfg.get("vad_filter", True)
                self._vad_min_silence_ms = _ucfg.get("vad_min_silence_ms", 300)
                self._vad_speech_pad_ms = _ucfg.get("vad_speech_pad_ms", 200)
        except Exception:
            pass
        self._backend_type = config["device"]
        self._config_label = self._make_label(config)
        self._first_call = True
        self._worker_proc: Optional[subprocess.Popen] = None

        os.environ["OMP_NUM_THREADS"] = str(config.get("cpu_threads", 4))
        os.environ["MKL_NUM_THREADS"] = str(config.get("cpu_threads", 4))

        venv_path = config.get("venv_path")
        if venv_path and Path(venv_path).exists():
            self._backend = None
            self._worker_proc = self._spawn_worker(config, Path(venv_path))
            if self._worker_proc is None:
                # Worker failed to start — fall back to direct load
                print("[Transcriber] Worker spawn failed, loading directly.", file=sys.stderr)
                self._load_direct(config)
        else:
            self._load_direct(config)

    def _load_direct(self, config: dict) -> None:
        print(f"[Transcriber] Loading model: {MODEL_ID} ({self._config_label})...", file=sys.stderr)
        t0 = time.time()
        try:
            self._backend = self._load_backend(config)
            print(f"[Transcriber] Model ready ({time.time() - t0:.1f}s load)", file=sys.stderr)
            return
        except Exception as e:
            print(f"[Transcriber] Load failed ({self._config_label}): {e}", file=sys.stderr)

        # Walk fallback chain from benchmark_results.json
        from local_stt_he.benchmark import RESULTS_PATH, probe_and_cache_fallback
        fallback_chain: list[dict] = []
        if RESULTS_PATH.exists():
            try:
                data = json.loads(RESULTS_PATH.read_text())
                fallback_chain = data.get("fallback_order", [])
            except Exception:
                pass

        import threading
        for fallback in fallback_chain:
            flabel = self._make_label(fallback)
            print(f"[Transcriber] Trying fallback: {flabel}...", file=sys.stderr)
            try:
                self._backend = self._load_backend(fallback)
                self._config = fallback
                self._backend_type = fallback["device"]
                self._config_label = flabel
                primary_label = self._make_label(config)
                print(
                    f"\n[STT] WARNING: Primary device ({primary_label}) failed to load.\n"
                    f"[STT]          Running on fallback: {flabel}.\n"
                    f"[STT]          Transcription may be slower. "
                    f"Re-run setup.py if this is unexpected.\n",
                    file=sys.stderr,
                )
                print(f"[Transcriber] Fallback ready ({time.time() - t0:.1f}s): {flabel}",
                      file=sys.stderr)
                # Lazy RTF probe in background so accuracy tier is correct next session
                threading.Thread(
                    target=probe_and_cache_fallback, args=(fallback,), daemon=True
                ).start()
                return
            except Exception as fe:
                print(f"[Transcriber] Fallback {flabel} failed: {fe}", file=sys.stderr)

        raise RuntimeError(
            "All devices in the fallback chain failed to load. "
            "Check hardware state and re-run: python run_benchmark.py --force"
        )

    def _spawn_worker(self, config: dict, venv_path: Path) -> Optional[subprocess.Popen]:
        """Spawn worker.py inside the device venv. Returns the Popen handle or None."""
        if sys.platform == "win32":
            py = venv_path / "Scripts" / "python.exe"
        else:
            py = venv_path / "bin" / "python"

        if not py.exists():
            print(f"[Transcriber] venv python not found at {py}", file=sys.stderr)
            return None

        worker_script = Path(__file__).parent / "worker.py"
        worker_config = {**self._config, "model_id": MODEL_ID, "root": str(ROOT)}

        print(f"[Transcriber] Spawning venv worker ({self._config_label})...", file=sys.stderr)
        t0 = time.time()
        try:
            proc = subprocess.Popen(
                [str(py), str(worker_script), json.dumps(worker_config)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,   # worker stderr flows to our terminal
                text=True,
                bufsize=1,     # line-buffered
            )
            ready_line = proc.stdout.readline()
            if not ready_line:
                proc.terminate()
                print("[Transcriber] Worker produced no ready signal.", file=sys.stderr)
                return None
            ready = json.loads(ready_line.strip())
            if ready.get("status") != "ready":
                proc.terminate()
                print(f"[Transcriber] Worker startup failed: {ready}", file=sys.stderr)
                return None
            elapsed = time.time() - t0
            print(f"[Transcriber] Worker ready ({elapsed:.1f}s)", file=sys.stderr)
            return proc
        except Exception as e:
            print(f"[Transcriber] Failed to spawn worker: {e}", file=sys.stderr)
            return None

    @staticmethod
    def _make_label(config: dict) -> str:
        device = config.get("device", "cpu")
        if device == "openvino":
            return f"OpenVINO {config.get('openvino_device', 'CPU')}"
        return f"{device.upper()} {config.get('compute_type', '')}"

    def _load_backend(self, config: dict):
        device = config["device"]
        if device in ("cpu", "cuda"):
            from faster_whisper import WhisperModel
            return WhisperModel(
                MODEL_ID,
                device=device,
                compute_type=config["compute_type"],
                cpu_threads=config.get("cpu_threads", 4),
                num_workers=config.get("num_workers", 1),
            )
        elif device == "openvino":
            try:
                import openvino_genai as ov_genai
            except ImportError:
                raise RuntimeError(
                    "openvino_genai is not installed. "
                    "Install it or re-run the benchmark to select a different config."
                )
            ov_model_dir = ROOT / OV_MODEL_DIR
            if not ov_model_dir.exists():
                raise RuntimeError(
                    f"OpenVINO model not found at {ov_model_dir}. "
                    "Run tests/test_openvino.py to convert the model first."
                )
            return ov_genai.WhisperPipeline(
                str(ov_model_dir),
                device=config.get("openvino_device", "CPU"),
            )
        else:
            raise RuntimeError(f"Unknown device in config: {device!r}")

    @staticmethod
    def _confidence_ok(seg_list: list, threshold: float) -> bool:
        """
        Duration-weighted average log-probability check.
        Returns True if quality is acceptable (avg_logprob >= threshold).
        """
        if not seg_list:
            return True
        total_weight = sum(max(s.end - s.start, 0.01) for s in seg_list)
        weighted = sum(s.avg_logprob * max(s.end - s.start, 0.01) for s in seg_list)
        return (weighted / total_weight) >= threshold

    def transcribe(
        self,
        source: Union[str, Path, np.ndarray],
        bucket: str = "medium",
        on_segment: Optional[Callable[[dict], None]] = None,
        _tier_override=None,
        is_aborted: Optional[Callable[[], bool]] = None,
        language: str = None,
    ) -> TranscriptResult:
        """
        source: file path or float32 numpy array at 16 kHz.
        bucket: duration hint for params selection.
        _tier_override: WhisperParams instance from local_stt_he.params; bypasses auto-selection.
                        Used by LiveStreamer for graceful degradation under queue pressure.
        is_aborted: optional callable returning bool. Checked between segments.
        """
        from local_stt_he.params import get_params

        _lang = language if language is not None else self._language
        params = _tier_override if _tier_override is not None else get_params(bucket, self._config)

        vad_filter = self._vad_filter
        vad_min_silence_ms = self._vad_min_silence_ms
        vad_speech_pad_ms = self._vad_speech_pad_ms

        if self._worker_proc is not None:
            return self._transcribe_via_worker(source, bucket, params,
                                               vad_filter, vad_min_silence_ms,
                                               vad_speech_pad_ms, is_aborted=is_aborted,
                                               language=_lang)

        t0 = time.time()

        if self._backend_type in ("cpu", "cuda"):
            audio = self._resolve_source(source)
            audio_duration = len(audio) / 16000.0 if isinstance(audio, np.ndarray) else self._get_duration(source)

            kwargs = params.as_transcribe_kwargs()
            kwargs["language"] = _lang
            kwargs["vad_filter"] = vad_filter
            kwargs["vad_parameters"] = dict(
                min_silence_duration_ms=vad_min_silence_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )

            segs_gen, info = self._backend.transcribe(audio, **kwargs)
            seg_list = []   # raw Segment objects (needed for avg_logprob)
            segments = []   # dicts for TranscriptResult
            for s in segs_gen:
                if is_aborted and is_aborted():
                    print("[STT] Transcription aborted by caller.", file=sys.stderr)
                    break

                seg_dict = {"start": s.start, "end": s.end, "text": s.text,
                            "confidence": round(float(s.avg_logprob), 3)}
                seg_list.append(s)
                segments.append(seg_dict)
                if on_segment is not None:
                    on_segment(seg_dict)
            text = "".join(s.text for s in seg_list).strip()
            if not isinstance(audio, np.ndarray):
                audio_duration = info.duration

            # Confidence-gated retry: re-run at next tier if quality is low.
            # on_segment is NOT re-called for retry segments — the original callbacks
            # already fired; the final result.text reflects the retry output.
            if (params.confidence_retry_enabled
                    and bucket != "streaming"
                    and seg_list
                    and not self._confidence_ok(seg_list, params.log_prob_threshold)):
                from local_stt_he.params import next_tier, get_params_for_tier
                upgrade = next_tier(params.tier_used)
                if upgrade:
                    print(f"[STT] Low confidence — retrying at '{upgrade}' tier", file=sys.stderr)
                    params = get_params_for_tier(upgrade, bucket, self._config)
                    kwargs2 = params.as_transcribe_kwargs()
                    kwargs2["language"] = _lang
                    kwargs2["vad_filter"] = vad_filter
                    kwargs2["vad_parameters"] = dict(
                        min_silence_duration_ms=vad_min_silence_ms,
                        speech_pad_ms=vad_speech_pad_ms,
                    )
                    segs2, info = self._backend.transcribe(audio, **kwargs2)
                    seg_list = list(segs2)
                    segments = [{"start": s.start, "end": s.end, "text": s.text,
                                 "confidence": round(float(s.avg_logprob), 3)}
                                for s in seg_list]
                    text = "".join(s.text for s in seg_list).strip()

            from local_stt_he.postprocess import normalize_text
            text = normalize_text(text, _lang)

        elif self._backend_type == "openvino":
            import openvino_genai as ov_genai
            audio = self._to_array(source)
            audio_duration = len(audio) / 16000.0

            gen_config = ov_genai.WhisperGenerateConfig()
            gen_config.language = f"<|{_lang}|>"
            gen_config.return_timestamps = not params.without_timestamps
            _ov_set(gen_config, "beam_size", params.beam_size)
            _ov_set(gen_config, "temperature", params.temperature)
            _ov_set(gen_config, "repetition_penalty", params.patience)
            result = self._backend.generate(audio, gen_config)
            text = result.texts[0].strip() if result.texts else ""
            from local_stt_he.postprocess import normalize_text
            text = normalize_text(text, _lang)
            segments = []
            if not params.without_timestamps and hasattr(result, "chunks") and result.chunks:
                segments = [
                    {"start": c.timestamps.begin, "end": c.timestamps.end, "text": c.text}
                    for c in result.chunks
                ]
        else:
            raise RuntimeError(f"Unknown backend: {self._backend_type}")

        elapsed = time.time() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        return TranscriptResult(
            text=text,
            segments=segments,
            audio_duration=audio_duration,
            elapsed=round(elapsed, 3),
            rtf=round(rtf, 4),
            config_label=self._config_label,
            backend="faster-whisper" if self._backend_type in ("cpu", "cuda") else "openvino_genai",
            tier_used=params.tier_used,
            whisper_params=params.as_transcribe_kwargs(),
        )

    def _transcribe_via_worker(
        self, source, bucket: str, params, vad_filter: bool,
        vad_min_silence_ms: int, vad_speech_pad_ms: int,
        is_aborted: Optional[Callable[[], bool]] = None,
        language: str = None,
    ) -> TranscriptResult:
        """Send a transcription request to the venv worker subprocess."""
        t0 = time.time()
        temp_npy: Optional[str] = None

        if isinstance(source, np.ndarray):
            # Write numpy array to a temp file the worker can read
            fd, temp_npy = tempfile.mkstemp(suffix=".npy")
            os.close(fd)
            np.save(temp_npy, source.astype(np.float32))
            audio_path = temp_npy
            fallback_duration = len(source) / 16000.0
        else:
            audio_path = str(source)
            fallback_duration = self._get_duration(source)

        _lang = language if language is not None else self._language
        kwargs = params.as_transcribe_kwargs()
        kwargs["language"] = _lang
        kwargs["language_token"] = f"<|{_lang}|>"
        kwargs["vad_filter"] = vad_filter
        kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
        )

        request = json.dumps({
            "action":     "transcribe",
            "audio_path": audio_path,
            "params":     kwargs,
            "bucket":     bucket,
        })

        try:
            self._worker_proc.stdin.write(request + "\n")
            self._worker_proc.stdin.flush()
            response_line = self._worker_proc.stdout.readline()
        except Exception as e:
            if temp_npy:
                try:
                    Path(temp_npy).unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Worker communication error: {e}") from e

        # Worker deletes the .npy file itself; clean up here only on error path
        if not response_line:
            if temp_npy:
                try:
                    Path(temp_npy).unlink()
                except Exception:
                    pass
            raise RuntimeError("Worker closed stdout unexpectedly")

        try:
            response = json.loads(response_line.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Worker returned invalid JSON: {e}") from e

        if response.get("status") != "ok":
            raise RuntimeError(f"Worker error: {response.get('error', 'unknown')}")

        elapsed = time.time() - t0
        audio_duration = response.get("audio_duration", fallback_duration)
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        return TranscriptResult(
            text=response["text"],
            segments=response.get("segments", []),
            audio_duration=audio_duration,
            elapsed=round(elapsed, 3),
            rtf=round(rtf, 4),
            config_label=self._config_label,
            backend=f"venv-worker/{self._backend_type}",
            tier_used=params.tier_used,
            whisper_params=kwargs,
        )

    def _resolve_source(self, source) -> Union[np.ndarray, str]:
        """For faster-whisper: arrays pass through; paths pass through as strings."""
        if isinstance(source, np.ndarray):
            return source
        return str(source)

    def _to_array(self, source) -> np.ndarray:
        """Convert file path or array to float32 numpy array at 16kHz."""
        if isinstance(source, np.ndarray):
            return source.astype(np.float32)
        import soundfile as sf
        audio, sr = sf.read(str(source), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return audio

    def _get_duration(self, source) -> float:
        try:
            import soundfile as sf
            info = sf.info(str(source))
            return info.duration
        except Exception:
            return 0.0

    def unload(self) -> None:
        """Explicitly unload model and free memory."""
        if self._worker_proc is not None:
            try:
                self._worker_proc.stdin.write(
                    json.dumps({"action": "quit"}) + "\n"
                )
                self._worker_proc.stdin.flush()
                self._worker_proc.wait(timeout=5)
            except Exception:
                self._worker_proc.terminate()
            self._worker_proc = None

        if self._backend is not None:
            del self._backend
            self._backend = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
