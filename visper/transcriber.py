"""
visper/transcriber.py
-------------------
Unified transcription engine. Accepts a config dict from benchmark.get_best_config().
Dispatches to faster-whisper (CPU/CUDA) or openvino_genai backend.
Whisper parameters are resolved per-call via visper/params.py.
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
_FALLBACK_MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"
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
    # Set only for a two-stage Hebrew->English translation: the original Hebrew
    # transcript, so a bilingual view is possible. Empty otherwise.
    he_text: str = ""


class Transcriber:
    def __init__(self, config: dict):
        """
        config dict from benchmark.get_best_config(bucket).
        Keys: device, compute_type, cpu_threads, num_workers.
        For OpenVINO: also openvino_device.
        Optional: venv_path — if present, inference runs inside a venv worker subprocess.

        Applies resource profile before loading the backend.
        """
        from visper.resource import apply_profile, check_memory_headroom, check_vram_before_load
        config = apply_profile(config)
        config = check_memory_headroom(config)
        config = check_vram_before_load(config)
        self._config = config

        # Load per-session config flags (read once at construction time)
        self._language = "he"
        self._vad_filter = True
        self._vad_min_silence_ms = 500
        self._vad_speech_pad_ms = 200
        self._denoise = True
        self._normalize_volume = True
        self._highpass = True
        self._hotwords: str = ""
        try:
            from visper._config import load_config
            _ucfg = load_config()
            self._language = _ucfg.get("language", "he")
            self._vad_filter = _ucfg.get("vad_filter", True)
            # Fallbacks match the shipped config.yaml so a config-load failure
            # degrades to the same behaviour, not a silently different one.
            self._vad_min_silence_ms = _ucfg.get("vad_min_silence_ms", 500)
            self._vad_speech_pad_ms = _ucfg.get("vad_speech_pad_ms", 200)
            self._denoise = _ucfg.get("audio_denoise", True)
            self._normalize_volume = _ucfg.get("audio_normalize", True)
            self._highpass = _ucfg.get("audio_highpass", True)
            self._hotwords = _ucfg.get("hotwords", "") or ""
        except Exception:
            pass
        self._model_id = config.get("model_id", _FALLBACK_MODEL_ID)
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
        print(f"[Transcriber] Loading model: {self._model_id} ({self._config_label})...", file=sys.stderr)
        t0 = time.time()
        try:
            self._backend = self._load_backend(config)
            print(f"[Transcriber] Model ready ({time.time() - t0:.1f}s load)", file=sys.stderr)
            return
        except Exception as e:
            print(f"[Transcriber] Load failed ({self._config_label}): {e}", file=sys.stderr)

        # Walk fallback chain from benchmark_results.json
        from visper.benchmark import RESULTS_PATH, probe_and_cache_fallback
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
        worker_config = {**self._config, "model_id": self._model_id, "root": str(ROOT)}

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
        if device == "mlx":
            return "MLX (Apple Silicon)"
        return f"{device.upper()} {config.get('compute_type', '')}"

    def _load_backend(self, config: dict):
        device = config["device"]
        if device in ("cpu", "cuda"):
            from faster_whisper import WhisperModel
            return WhisperModel(
                self._model_id,
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
        elif device == "mlx":
            try:
                import mlx_whisper  # noqa: F401 — verify installed; model loaded on first call
            except ImportError:
                raise RuntimeError(
                    "mlx-whisper is not installed. "
                    "Install it with: pip install mlx-whisper"
                )
            return None  # mlx-whisper caches models internally; no persistent object needed
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

    @staticmethod
    def _confidence_ok_dicts(segments: list, threshold: float) -> bool:
        """Same check as _confidence_ok but over the worker's segment dicts
        ({'start','end','confidence'} where confidence is the rounded avg_logprob)."""
        if not segments:
            return True
        total_weight = sum(max(s["end"] - s["start"], 0.01) for s in segments)
        weighted = sum(s.get("confidence", 0.0) * max(s["end"] - s["start"], 0.01) for s in segments)
        return (weighted / total_weight) >= threshold

    def transcribe(
        self,
        source: Union[str, Path, np.ndarray],
        bucket: str = "medium",
        on_segment: Optional[Callable[[dict], None]] = None,
        _tier_override=None,
        is_aborted: Optional[Callable[[], bool]] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptResult:
        """
        source: file path or float32 numpy array at 16 kHz.
        bucket: duration hint for params selection.
        _tier_override: WhisperParams instance from visper.params; bypasses auto-selection.
                        Used by LiveStreamer for graceful degradation under queue pressure.
        is_aborted: optional callable returning bool. Checked between segments.
        """
        from visper.params import get_params

        _lang = language if language is not None else self._language

        # Two-stage Hebrew -> English: the ivrit-ai fine-tune is a transcription
        # specialist and translates poorly, so transcribe in Hebrew and run a
        # dedicated he->en MT pass over the segments. Falls through to Whisper's
        # own translate task when the MT model isn't available.
        if task == "translate" and _lang == "he":
            from visper.translate import get_hebrew_english_translator
            _mt = get_hebrew_english_translator()
            if _mt is not None:
                return self._translate_hebrew(
                    source, bucket, _mt, on_segment=on_segment,
                    _tier_override=_tier_override, is_aborted=is_aborted,
                    initial_prompt=initial_prompt,
                )

        params = _tier_override if _tier_override is not None else get_params(bucket, self._config)

        vad_filter = self._vad_filter
        vad_min_silence_ms = self._vad_min_silence_ms
        vad_speech_pad_ms = self._vad_speech_pad_ms

        if self._worker_proc is not None:
            return self._transcribe_via_worker(source, bucket, params,
                                               vad_filter, vad_min_silence_ms,
                                               vad_speech_pad_ms, is_aborted=is_aborted,
                                               language=_lang, initial_prompt=initial_prompt,
                                               task=task)

        t0 = time.time()

        if self._backend_type in ("cpu", "cuda"):
            audio = self._resolve_source(source)
            audio_duration = len(audio) / 16000.0 if isinstance(audio, np.ndarray) else self._get_duration(source)

            if (self._denoise or self._normalize_volume or self._highpass) and bucket != "streaming":
                if not isinstance(audio, np.ndarray):
                    audio = self._to_array(source)
                if self._normalize_volume:
                    audio = self._normalize_audio_volume(audio)
                if self._highpass:
                    audio = self._highpass_filter(audio)
                if self._denoise:
                    audio = self._denoise_audio(audio)
                audio_duration = len(audio) / 16000.0

            kwargs = params.as_transcribe_kwargs()
            kwargs["language"] = _lang
            if task == "translate":
                kwargs["task"] = "translate"
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            if self._hotwords:
                kwargs["hotwords"] = self._hotwords
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
                from visper.params import next_tier, get_params_for_tier
                upgrade = next_tier(params.tier_used)
                if upgrade:
                    print(f"[STT] Low confidence — retrying at '{upgrade}' tier", file=sys.stderr)
                    params = get_params_for_tier(upgrade, bucket, self._config)
                    kwargs2 = params.as_transcribe_kwargs()
                    kwargs2["language"] = _lang
                    if task == "translate":
                        kwargs2["task"] = "translate"
                    if initial_prompt:
                        kwargs2["initial_prompt"] = initial_prompt
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

            from visper.postprocess import normalize_text
            text = normalize_text(text, "en" if task == "translate" else _lang)

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
            from visper.postprocess import normalize_text
            text = normalize_text(text, _lang)
            segments = []
            if not params.without_timestamps and hasattr(result, "chunks") and result.chunks:
                segments = [
                    {"start": c.timestamps.begin, "end": c.timestamps.end, "text": c.text}
                    for c in result.chunks
                ]

        elif self._backend_type == "mlx":
            import mlx_whisper as _mlx
            if isinstance(source, np.ndarray):
                audio_input = source
                audio_duration = len(source) / 16000.0
            else:
                audio_input = str(source)
                audio_duration = self._get_duration(source)

            # Apply audio preprocessing if configured (not for streaming)
            if (self._denoise or self._normalize_volume or self._highpass) and bucket != "streaming":
                arr = audio_input if isinstance(audio_input, np.ndarray) else self._to_array(source)
                if self._normalize_volume:
                    arr = self._normalize_audio_volume(arr)
                if self._highpass:
                    arr = self._highpass_filter(arr)
                if self._denoise:
                    arr = self._denoise_audio(arr)
                audio_input = arr
                audio_duration = len(arr) / 16000.0

            _beam = params.beam_size if params.beam_size > 0 else 1
            _temp = params.temperature
            _temp0 = _temp[0] if hasattr(_temp, "__iter__") else float(_temp)

            mlx_result = _mlx.transcribe(
                audio_input,
                path_or_hf_repo=self._model_id,
                language=_lang,
                task=task,
                beam_size=_beam,
                temperature=_temp0,
                initial_prompt=initial_prompt or None,
            )
            text = (mlx_result.get("text") or "").strip()
            raw_segs = mlx_result.get("segments") or []
            segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"],
                 "confidence": round(float(s.get("avg_logprob", 0.0)), 3)}
                for s in raw_segs
            ]
            if on_segment:
                for seg in segments:
                    on_segment(seg)
            from visper.postprocess import normalize_text
            text = normalize_text(text, "en" if task == "translate" else _lang)

        else:
            raise RuntimeError(f"Unknown backend: {self._backend_type}")

        elapsed = time.time() - t0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        _backend_label = (
            "faster-whisper" if self._backend_type in ("cpu", "cuda")
            else "mlx-whisper" if self._backend_type == "mlx"
            else "openvino_genai"
        )
        return TranscriptResult(
            text=text,
            segments=segments,
            audio_duration=audio_duration,
            elapsed=round(elapsed, 3),
            rtf=round(rtf, 4),
            config_label=self._config_label,
            backend=_backend_label,
            tier_used=params.tier_used,
            whisper_params=params.as_transcribe_kwargs(),
        )

    def _translate_hebrew(self, source, bucket: str, mt, *,
                          on_segment: Optional[Callable[[dict], None]] = None,
                          _tier_override=None,
                          is_aborted: Optional[Callable[[], bool]] = None,
                          initial_prompt: Optional[str] = None) -> TranscriptResult:
        """Stage 2: Hebrew transcription, then a he->en MT pass over the segments.

        Runs stage 1 with ``on_segment`` withheld — the callbacks fire here,
        once per *English* segment, so a caller's streamed segments and the
        final ``text`` are the same language. The stage-1 Whisper decode is
        therefore silent; the MT pass is a small fraction of ASR time.
        """
        t0 = time.time()
        he = self.transcribe(
            source, bucket=bucket, on_segment=None, _tier_override=_tier_override,
            is_aborted=is_aborted, language="he", initial_prompt=initial_prompt,
            task="transcribe",
        )

        from visper.postprocess import normalize_text
        segments = he.segments or []
        en_texts = mt.translate([s.get("text", "") for s in segments]) if segments else []

        new_segments: list = []
        for s, en in zip(segments, en_texts):
            seg = dict(s)
            seg["he_text"] = s.get("text", "")
            seg["text"] = normalize_text(en, "en")
            new_segments.append(seg)
            if on_segment is not None:
                on_segment(seg)

        if new_segments:
            text = " ".join(s["text"] for s in new_segments if s["text"]).strip()
        elif he.text:
            text = normalize_text(mt.translate([he.text])[0], "en")
        else:
            text = ""

        elapsed = time.time() - t0
        rtf = elapsed / he.audio_duration if he.audio_duration > 0 else 0.0
        return TranscriptResult(
            text=text,
            segments=new_segments,
            audio_duration=he.audio_duration,
            elapsed=round(elapsed, 3),
            rtf=round(rtf, 4),
            config_label=he.config_label,
            backend=f"{he.backend}+opus-mt-he-en",
            tier_used=he.tier_used,
            whisper_params=he.whisper_params,
            he_text=he.text,
        )

    def _transcribe_via_worker(
        self, source, bucket: str, params, vad_filter: bool,
        vad_min_silence_ms: int, vad_speech_pad_ms: int,
        is_aborted: Optional[Callable[[], bool]] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        task: str = "transcribe",
    ) -> TranscriptResult:
        """Send a transcription request to the venv worker subprocess.

        Feature parity with the in-process path — these used to be applied only
        when the model ran in-process, so the default (venv-worker) runtime
        silently skipped them: audio pre-processing (denoise / highpass /
        normalize), hotwords, initial_prompt, task=translate, confidence-gated
        retry, and the Hebrew post-normalization pass.
        """
        t0 = time.time()
        _lang = language if language is not None else self._language

        if is_aborted is not None and is_aborted():
            return TranscriptResult(
                text="", segments=[], audio_duration=0.0, elapsed=0.0, rtf=0.0,
                config_label=self._config_label,
                backend=f"venv-worker/{self._backend_type}",
                tier_used=params.tier_used,
                whisper_params=params.as_transcribe_kwargs(),
            )

        # ── Audio pre-processing (array path only, never for streaming) ──
        preprocess = (self._denoise or self._normalize_volume or self._highpass) and bucket != "streaming"
        audio_arr: Optional[np.ndarray] = None
        if isinstance(source, np.ndarray):
            audio_arr = source.astype(np.float32)
        elif preprocess:
            audio_arr = self._to_array(source)
        if preprocess and audio_arr is not None:
            if self._normalize_volume:
                audio_arr = self._normalize_audio_volume(audio_arr)
            if self._highpass:
                audio_arr = self._highpass_filter(audio_arr)
            if self._denoise:
                audio_arr = self._denoise_audio(audio_arr)

        if audio_arr is not None:
            fd, temp_npy = tempfile.mkstemp(suffix=".npy")
            os.close(fd)
            np.save(temp_npy, audio_arr.astype(np.float32))
            audio_path: str = temp_npy
            fallback_duration = len(audio_arr) / 16000.0
        else:
            temp_npy = None
            audio_path = str(source)
            fallback_duration = self._get_duration(source)

        def _build_kwargs(p) -> dict:
            kw = p.as_transcribe_kwargs()
            kw["language"] = _lang
            kw["language_token"] = f"<|{_lang}|>"
            if task == "translate":
                kw["task"] = "translate"
            if initial_prompt:
                kw["initial_prompt"] = initial_prompt
            if self._hotwords:
                kw["hotwords"] = self._hotwords
            kw["vad_filter"] = vad_filter
            kw["vad_parameters"] = dict(
                min_silence_duration_ms=vad_min_silence_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )
            return kw

        kwargs = _build_kwargs(params)
        response = self._worker_roundtrip(audio_path, kwargs, bucket, temp_npy)

        # ── Confidence-gated retry (parent-side: re-send at the next tier) ──
        segs = response.get("segments", [])
        if (params.confidence_retry_enabled and bucket != "streaming" and segs
                and not self._confidence_ok_dicts(segs, params.log_prob_threshold)
                and not (is_aborted is not None and is_aborted())):
            from visper.params import next_tier, get_params_for_tier
            upgrade = next_tier(params.tier_used)
            if upgrade:
                print(f"[STT] Low confidence — retrying at '{upgrade}' tier (worker)", file=sys.stderr)
                params = get_params_for_tier(upgrade, bucket, self._config)
                retry_path, retry_npy = audio_path, None
                if temp_npy is not None:
                    # worker already unlinked the first .npy — write a fresh one
                    src_arr = audio_arr if audio_arr is not None else self._to_array(source)
                    fd, retry_npy = tempfile.mkstemp(suffix=".npy")
                    os.close(fd)
                    np.save(retry_npy, src_arr.astype(np.float32))
                    retry_path = retry_npy
                response = self._worker_roundtrip(retry_path, _build_kwargs(params), bucket, retry_npy)

        elapsed = time.time() - t0
        audio_duration = response.get("audio_duration", fallback_duration) or 0.0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        from visper.postprocess import normalize_text
        text = normalize_text(response.get("text", ""), "en" if task == "translate" else _lang)

        return TranscriptResult(
            text=text,
            segments=response.get("segments", []),
            audio_duration=audio_duration,
            elapsed=round(elapsed, 3),
            rtf=round(rtf, 4),
            config_label=self._config_label,
            backend=f"venv-worker/{self._backend_type}",
            tier_used=params.tier_used,
            whisper_params=kwargs,
        )

    def _worker_roundtrip(self, audio_path: str, kwargs: dict, bucket: str,
                          temp_npy: Optional[str]) -> dict:
        """One request/response with the worker. The worker unlinks a .npy it
        was handed on success; this cleans it up on every error path."""
        def _cleanup():
            if temp_npy:
                try:
                    Path(temp_npy).unlink()
                except Exception:
                    pass

        request = json.dumps({
            "action": "transcribe", "audio_path": audio_path,
            "params": kwargs, "bucket": bucket,
        })
        try:
            self._worker_proc.stdin.write(request + "\n")
            self._worker_proc.stdin.flush()
            response_line = self._worker_proc.stdout.readline()
        except Exception as e:
            _cleanup()
            raise RuntimeError(f"Worker communication error: {e}") from e

        if not response_line:
            _cleanup()
            raise RuntimeError("Worker closed stdout unexpectedly")
        try:
            response = json.loads(response_line.strip())
        except json.JSONDecodeError as e:
            _cleanup()
            raise RuntimeError(f"Worker returned invalid JSON: {e}") from e
        if response.get("status") != "ok":
            _cleanup()
            raise RuntimeError(f"Worker error: {response.get('error', 'unknown')}")
        return response

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

    def _highpass_filter(self, audio: np.ndarray, cutoff_hz: float = 80.0) -> np.ndarray:
        try:
            from scipy.signal import butter, filtfilt
            nyq = 16000 / 2.0
            b, a = butter(4, cutoff_hz / nyq, btype='high')
            return filtfilt(b, a, audio).astype(np.float32)
        except Exception:
            return audio

    def _normalize_audio_volume(self, audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 1e-8:
            return audio
        gain = min(target_rms / rms, 10.0)  # cap at 10× to avoid amplifying pure noise
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    def _denoise_audio(self, audio: np.ndarray) -> np.ndarray:
        try:
            import noisereduce as nr
            return nr.reduce_noise(y=audio, sr=16000).astype(np.float32)
        except Exception:
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
