"""
core/benchmark_worker.py
------------------------
Benchmark worker — runs INSIDE a device venv as a subprocess.
Invoked by core/benchmark.py via subprocess.Popen.

Protocol
--------
  stdin  : one JSON object (the full benchmark request)
  stdout : one JSON object (the full result: {bucket: result_dict})
  stderr : progress messages forwarded to the parent terminal

Request schema
--------------
{
  "candidate": {device, compute_type, cpu_threads, ...},
  "audio_files": {
    "short":    {"path": "/abs/path", "target_duration": 5.0},
    "medium":   {...},
    "long":     {...},
    "extended": {...}
  },
  "model_id":   "ivrit-ai/whisper-large-v3-turbo-ct2",
  "root":       "/abs/path/to/repo",
  "sample_rate": 16000
}

Result schema
-------------
{
  "short":     {"status": "ok",        "rtf": 0.23,  "elapsed": 1.15, "audio_duration": 5.0},
  "medium":    {"status": "ok",        "rtf": 0.18,  ...},
  "streaming": {"status": "ok",        "median_rtf": 0.20, "rtf_per_call": [...], ...},
  "long":      {"status": "slow_skip", "rtf": 3.2},
  ...
}
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

SAMPLE_RATE = 16_000
BUCKET_ORDER = ["short", "medium", "long", "extended"]
_WARMUP_SKIP_RTF = 3.0

_TRANSCRIBE_KWARGS = dict(
    language="he",
    beam_size=1,
    temperature=0.0,
    condition_on_previous_text=False,
    without_timestamps=True,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_dur: float):
    """Decode audio and slice to target_dur seconds. Returns (array, actual_dur)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio not found: {p}")

    # faster_whisper is in CPU/CUDA venvs; librosa is in OpenVINO venv
    try:
        from faster_whisper import decode_audio
        full = decode_audio(str(p), sampling_rate=SAMPLE_RATE)
    except ImportError:
        import librosa
        full, _ = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)

    target_samples = int(target_dur * SAMPLE_RATE)
    sliced = full[:target_samples]
    return sliced, len(sliced) / float(SAMPLE_RATE)


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
# Model loading / inference
# ---------------------------------------------------------------------------

def _load_model(candidate: dict, root: str):
    omp = str(candidate.get("omp_threads", candidate.get("cpu_threads", 4)))
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["MKL_NUM_THREADS"] = omp

    device = candidate["device"]
    model_id = candidate["model_id"]

    if device in ("cpu", "cuda"):
        if device == "cuda":
            _register_cuda_dlls()
        from faster_whisper import WhisperModel
        return WhisperModel(
            model_id,
            device=device,
            compute_type=candidate["compute_type"],
            cpu_threads=candidate.get("cpu_threads", 4),
            num_workers=candidate.get("num_workers", 1),
        )

    elif device == "openvino":
        import openvino_genai as ov_genai
        ov_dir = Path(root) / "ov_model"
        if not ov_dir.exists():
            raise RuntimeError(f"OpenVINO model not found at {ov_dir}")
        return ov_genai.WhisperPipeline(
            str(ov_dir), device=candidate.get("openvino_device", "CPU")
        )

    raise RuntimeError(f"Unknown device: {device!r}")


def _infer(model, audio, candidate: dict) -> str:
    device = candidate["device"]
    if device in ("cpu", "cuda"):
        segs, _ = model.transcribe(audio, **_TRANSCRIBE_KWARGS)
        return "".join(s.text for s in segs).strip()
    elif device == "openvino":
        import openvino_genai as ov_genai
        cfg = ov_genai.WhisperGenerateConfig()
        cfg.language = "<|he|>"
        result = model.generate(audio, cfg)
        return result.texts[0].strip() if result.texts else ""
    return ""


# ---------------------------------------------------------------------------
# Main benchmark session
# ---------------------------------------------------------------------------

def run(request: dict) -> dict:
    candidate = request["candidate"]
    audio_files = request["audio_files"]   # {bucket: {path, target_duration}}
    root = request.get("root", ".")

    ordered = [b for b in BUCKET_ORDER if b in audio_files]
    results: dict = {}

    # Load model
    print(f"[Worker] Loading model ({candidate.get('label', candidate['device'])})...",
          file=sys.stderr, flush=True)
    try:
        model = _load_model(candidate, root)
    except Exception as e:
        err = str(e)[:300]
        print(f"[Worker] Load failed: {err}", file=sys.stderr)
        for b in ordered + ["streaming"]:
            results[b] = {"status": "failed", "error": err}
        return results

    print("[Worker] Model loaded — running benchmark...", file=sys.stderr, flush=True)

    try:
        # Pre-load audio slices
        bucket_audio: dict = {}
        for b in ordered:
            info = audio_files[b]
            try:
                audio, dur = _load_audio(info["path"], info["target_duration"])
                bucket_audio[b] = (audio, dur)
            except Exception as e:
                print(f"[Worker] Audio load failed for {b}: {e}", file=sys.stderr)

        if not bucket_audio:
            for b in ordered + ["streaming"]:
                results[b] = {"status": "failed", "error": "no audio loaded"}
            return results

        first_b = next(iter(bucket_audio))
        first_audio, first_dur = bucket_audio[first_b]

        # Warm-up (discarded)
        _infer(model, first_audio, candidate)

        # Timed warm-up for early-skip check
        t0 = time.time()
        _infer(model, first_audio, candidate)
        warmup_rtf = (time.time() - t0) / first_dur

        if warmup_rtf > _WARMUP_SKIP_RTF:
            print(f"[Worker] Warm-up RTF {warmup_rtf:.2f} > {_WARMUP_SKIP_RTF} — slow_skip",
                  file=sys.stderr)
            for b in ordered + ["streaming"]:
                results[b] = {"status": "slow_skip", "rtf": round(warmup_rtf, 4)}
            return results

        # Bucket trials — early exit if RTF > 1.5 from the second bucket onward
        _BUCKET_SLOW_RTF = 1.5
        warmup_elapsed = time.time() - t0
        for idx, b in enumerate(ordered):
            if b not in bucket_audio:
                continue
            audio, dur = bucket_audio[b]
            if b == first_b:
                # Reuse the timed warm-up pass
                elapsed = warmup_elapsed
                rtf = warmup_rtf
            else:
                t0 = time.time()
                _infer(model, audio, candidate)
                elapsed = time.time() - t0
                rtf = elapsed / dur
            results[b] = {
                "status":         "ok",
                "rtf":            round(rtf, 4),
                "elapsed":        round(elapsed, 3),
                "audio_duration": dur,
            }
            print(f"[Worker]   {b}: RTF {rtf:.3f}", file=sys.stderr, flush=True)

            # From the second bucket onward: cut if too slow for real-time use
            if idx > 0 and rtf > _BUCKET_SLOW_RTF:
                remaining = [ob for ob in ordered[idx + 1:] if ob in bucket_audio]
                if remaining:
                    print(f"[Worker]   RTF {rtf:.2f} > {_BUCKET_SLOW_RTF} — skipping "
                          f"{remaining}", file=sys.stderr, flush=True)
                    for rb in remaining:
                        results[rb] = {"status": "slow_skip", "rtf": round(rtf, 4)}
                break

        # Streaming trial (5 calls, median of calls 2-5)
        s_audio, s_dur = bucket_audio[first_b]
        rtfs = []
        for _ in range(5):
            t0 = time.time()
            _infer(model, s_audio, candidate)
            rtfs.append(round((time.time() - t0) / s_dur, 4))
        measured = rtfs[1:]  # skip warm-up call
        median_rtf = sorted(measured)[len(measured) // 2]
        stdev_rtf = statistics.stdev(measured) if len(measured) >= 2 else 0.0
        p95_rtf = sorted(measured)[-1]  # with 4 values, p95 ≈ max
        results["streaming"] = {
            "status":         "ok",
            "median_rtf":     round(median_rtf, 4),
            "rtf_stdev":      round(stdev_rtf, 4),
            "rtf_p95":        round(p95_rtf, 4),
            "rtf_per_call":   rtfs,
            "audio_duration": s_dur,
        }
        print(f"[Worker]   streaming: median RTF {median_rtf:.3f} ±{stdev_rtf:.3f}", file=sys.stderr, flush=True)

    except Exception as e:
        err = str(e)[:300]
        print(f"[Worker] Benchmark error: {err}", file=sys.stderr)
        for b in ordered + ["streaming"]:
            if b not in results:
                results[b] = {"status": "failed", "error": err}
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
        result = run(request)
        print(json.dumps(result), flush=True)
    except Exception as e:
        print(json.dumps({"_error": str(e)}), flush=True)
        sys.exit(1)
