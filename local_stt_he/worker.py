"""
core/worker.py
--------------
Persistent transcription worker — runs INSIDE a device venv as a subprocess.
Spawned by core/transcriber.py when a venv_path is present in the config.

Protocol
--------
  argv[1] : JSON-encoded config dict (device, compute_type, cpu_threads,
            model_id, root, ...)
  stdout  : one JSON line per response (model ready signal + transcription results)
  stdin   : one JSON line per request
  stderr  : progress / error messages (forwarded to parent terminal)

Startup sequence
----------------
  1. Parent spawns:  python worker.py '<config_json>'
  2. Worker loads model, then writes: {"status": "ready"}
  3. Parent reads "ready" line before sending any requests.

Request
-------
  {"action": "transcribe", "audio_path": "/abs/path.npy|.wav|.mp3", "params": {...}}
  {"action": "quit"}

Response
--------
  {"status": "ok", "text": "...", "segments": [...],
   "elapsed": 1.23, "audio_duration": 5.0}
  {"status": "error", "error": "..."}
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# CUDA DLL registration (Windows)
# ---------------------------------------------------------------------------

def _register_cuda_dlls() -> None:
    import site, pathlib
    for sp in site.getsitepackages():
        nvidia_path = pathlib.Path(sp) / "nvidia"
        if nvidia_path.exists():
            for dll_dir in nvidia_path.rglob("*.dll"):
                folder = str(dll_dir.parent)
                if folder not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = folder + ";" + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(config: dict, root: str):
    omp = str(config.get("omp_threads", config.get("cpu_threads", 4)))
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["MKL_NUM_THREADS"] = omp

    device = config["device"]
    model_id = config["model_id"]

    if device in ("cpu", "cuda"):
        if device == "cuda":
            _register_cuda_dlls()
        from faster_whisper import WhisperModel
        return WhisperModel(
            model_id,
            device=device,
            compute_type=config["compute_type"],
            cpu_threads=config.get("cpu_threads", 4),
            num_workers=config.get("num_workers", 1),
        )

    elif device == "openvino":
        import openvino_genai as ov_genai
        ov_dir = Path(root) / "ov_model"
        if not ov_dir.exists():
            raise RuntimeError(f"OpenVINO model not found at {ov_dir}")
        return ov_genai.WhisperPipeline(
            str(ov_dir), device=config.get("openvino_device", "CPU")
        )

    raise RuntimeError(f"Unknown device: {device!r}")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(audio_path: str):
    """Load audio from .npy, .wav, .mp3, etc. Returns (float32 array, duration)."""
    import numpy as np
    p = Path(audio_path)

    if p.suffix == ".npy":
        audio = np.load(str(p)).astype(np.float32)
        return audio, len(audio) / float(SAMPLE_RATE)

    try:
        from faster_whisper import decode_audio
        audio = decode_audio(str(p), sampling_rate=SAMPLE_RATE)
        return audio, len(audio) / float(SAMPLE_RATE)
    except ImportError:
        pass

    import librosa
    audio, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), len(audio) / float(SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _transcribe(model, audio, config: dict, params: dict):
    """Run inference. Returns (text, segments, elapsed, audio_duration)."""
    audio_duration = len(audio) / float(SAMPLE_RATE)
    device = config["device"]

    t0 = time.time()

    if device in ("cpu", "cuda"):
        transcribe_kwargs = {
            "language":                   params.get("language", "he"),
            "beam_size":                  params.get("beam_size", 1),
            "best_of":                    params.get("best_of", 1),
            "temperature":                params.get("temperature", 0.0),
            "patience":                   params.get("patience", 1.0),
            "condition_on_previous_text": params.get("condition_on_previous_text", False),
            "without_timestamps":         params.get("without_timestamps", True),
            "compression_ratio_threshold":params.get("compression_ratio_threshold", 2.4),
            "log_prob_threshold":         params.get("log_prob_threshold", -1.0),
            "no_speech_threshold":        params.get("no_speech_threshold", 0.6),
            "vad_filter":                 params.get("vad_filter", True),
            "vad_parameters":             params.get("vad_parameters",
                                              dict(min_silence_duration_ms=300,
                                                   speech_pad_ms=200)),
        }
        segs, _ = model.transcribe(audio, **transcribe_kwargs)
        seg_list = list(segs)
        text = "".join(s.text for s in seg_list).strip()
        segments = [{"start": s.start, "end": s.end, "text": s.text,
                     "confidence": round(float(s.avg_logprob), 3)} for s in seg_list]

    elif device == "openvino":
        import openvino_genai as ov_genai
        gen_config = ov_genai.WhisperGenerateConfig()
        gen_config.language = params.get("language_token", "<|he|>")
        gen_config.beam_size = params.get("beam_size", 1)
        gen_config.return_timestamps = not params.get("without_timestamps", True)
        result = model.generate(audio, gen_config)
        text = result.texts[0].strip() if result.texts else ""
        segments = []
        if not params.get("without_timestamps", True) and \
                hasattr(result, "chunks") and result.chunks:
            segments = [
                {"start": c.timestamps.begin, "end": c.timestamps.end, "text": c.text}
                for c in result.chunks
            ]
    else:
        raise RuntimeError(f"Unknown device: {device!r}")

    elapsed = time.time() - t0
    return text, segments, elapsed, audio_duration


# ---------------------------------------------------------------------------
# Main event loop
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "No config provided"}), flush=True)
        sys.exit(1)

    config = json.loads(sys.argv[1])
    root = config.get("root", str(Path(__file__).parent.parent))

    print(f"[Worker] Loading model ({config.get('device', '?')} "
          f"{config.get('compute_type', '')})...", file=sys.stderr, flush=True)

    try:
        model = _load_model(config, root)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}), flush=True)
        sys.exit(1)

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"status": "error", "error": f"Invalid JSON: {e}"}), flush=True)
            continue

        action = req.get("action", "transcribe")

        if action == "quit":
            break

        if action == "transcribe":
            audio_path = req.get("audio_path", "")
            is_temp_npy = audio_path.endswith(".npy")
            try:
                audio, _ = _load_audio(audio_path)
                # Delete temp numpy file immediately after loading
                if is_temp_npy:
                    try:
                        Path(audio_path).unlink()
                    except Exception:
                        pass
                text, segments, elapsed, audio_duration = _transcribe(
                    model, audio, config, req.get("params", {})
                )
                print(json.dumps({
                    "status":         "ok",
                    "text":           text,
                    "segments":       segments,
                    "elapsed":        round(elapsed, 3),
                    "audio_duration": round(audio_duration, 3),
                }), flush=True)
            except Exception as e:
                if is_temp_npy:
                    try:
                        Path(audio_path).unlink()
                    except Exception:
                        pass
                print(json.dumps({"status": "error", "error": str(e)}), flush=True)
        else:
            print(json.dumps({"status": "error",
                               "error": f"Unknown action: {action!r}"}), flush=True)

    try:
        del model
    except Exception:
        pass
    gc.collect()


if __name__ == "__main__":
    main()
