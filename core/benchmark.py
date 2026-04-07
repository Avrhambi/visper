"""
core/benchmark.py
-----------------
Single source of truth for all hardware detection and config selection.
Writes benchmark_results.json and exposes get_best_config(bucket).

Speed design
------------
Each candidate is loaded ONCE, all buckets + streaming trial are run in that
single session, then the model is unloaded.  This reduces model-load cost from
N_candidates × N_buckets down to N_candidates — the dominant saving.

Audio strategy
--------------
school.mp3 is decoded once and sliced in-memory to 5 s / 20 s / 45 s / 90 s
for short / medium / long / extended buckets.  All four buckets are always
populated regardless of what other files exist in records/.
If school.mp3 is missing, falls back to closest native file per bucket.

Benchmark modes
---------------
smart (default): tests only the most promising configs per hardware.
  CPU: 1-2 best compute types (int8 on AVX2+) × 2 smart thread counts.
  Total: ~4-6 CPU candidates instead of 20+.  3-5× faster than full.
full: exhaustive — all compute types × thread counts [2,4,6,8].
  Use --full flag or move to tests/test_benchmark_full.py.
quick: heuristic only — derives config from hardware detection, no inference.
  Writes estimated config instantly.  Use when timing is not needed.

Configuration candidates
------------------------
CPU:  compute_type × cpu_threads  (num_workers fixed at 1 — a second worker
      loads a second model copy and has no effect on single-request latency).
      OMP_NUM_THREADS / MKL_NUM_THREADS set equal to cpu_threads and recorded
      in the candidate dict so the winning config can be replayed exactly.
CUDA: compute_type only (threads/workers fixed at 4/1).
OpenVINO: one candidate per Intel GPU device + OpenVINO-CPU fallback.
"""
from __future__ import annotations

import gc
import json
import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_PATH = ROOT / "benchmark_results.json"
RECORDS_DIR  = ROOT / "records"
MODEL_ID     = "ivrit-ai/whisper-large-v3-turbo-ct2"
SAMPLE_RATE  = 16_000

CANONICAL_FILE = "school.mp3"   # sliced for all buckets

BUCKETS = {
    "short":    (0,   10),
    "medium":   (10,  30),
    "long":     (30,  60),
    "extended": (60,  9999),
}
SLICE_TARGETS_S  = {"short": 5.0, "medium": 20.0, "long": 45.0, "extended": 90.0}
BUCKET_MIDPOINTS = {"short": 5,   "medium": 20,   "long": 45,   "extended": 90}
BUCKET_ORDER     = ["short", "medium", "long", "extended"]

_TRANSCRIBE_KWARGS = dict(
    language="he",
    beam_size=1,
    temperature=0.0,
    condition_on_previous_text=False,
    without_timestamps=True,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
)

# RTF threshold above which a candidate is considered too slow to bother with
_WARMUP_SKIP_RTF = 3.0
# RTF threshold below which we consider the device "elite" and skip slower
# variants of the same (device, compute_type)
_ELITE_RTF = 0.15


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def _audio_duration(path: Path) -> Optional[float]:
    try:
        import soundfile as sf
        return sf.info(str(path)).duration
    except Exception:
        pass
    try:
        from mutagen import File as MutagenFile
        f = MutagenFile(str(path))
        if f is not None and f.info is not None:
            return float(f.info.length)
    except Exception:
        pass
    try:
        from faster_whisper import decode_audio
        audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
        return len(audio) / float(SAMPLE_RATE)
    except Exception:
        pass
    return None


def _bucket_for(duration: float) -> str:
    for name, (lo, hi) in BUCKETS.items():
        if lo <= duration < hi:
            return name
    return "extended"


def _get_records_by_bucket() -> dict[str, Optional[tuple[Path, float]]]:
    """
    Returns {bucket: (path, target_duration)}.

    Primary: school.mp3 sliced to SLICE_TARGETS_S for each bucket.
    Fallback: closest file to each bucket midpoint in records/.
    """
    result: dict[str, Optional[tuple[Path, float]]] = {b: None for b in BUCKETS}

    canonical = RECORDS_DIR / CANONICAL_FILE
    if canonical.exists():
        total_dur = _audio_duration(canonical)
        if total_dur:
            for bucket, target_s in SLICE_TARGETS_S.items():
                if total_dur >= target_s:
                    result[bucket] = (canonical, target_s)
            return result

    # Fallback: one file per bucket
    file_candidates: dict[str, list[tuple[Path, float]]] = {b: [] for b in BUCKETS}
    if RECORDS_DIR.exists():
        for p in RECORDS_DIR.iterdir():
            if p.suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"):
                continue
            dur = _audio_duration(p)
            if dur is None:
                print(f"[Benchmark] Warning: could not read duration of {p.name}, skipping.")
                continue
            b = _bucket_for(dur)
            file_candidates[b].append((p, dur))

    for b, files in file_candidates.items():
        if not files:
            continue
        mid = BUCKET_MIDPOINTS[b]
        best_file = min(files, key=lambda t: abs(t[1] - mid))
        result[b] = best_file

    return result


# ---------------------------------------------------------------------------
# Hardware candidate generation
# ---------------------------------------------------------------------------

def _smart_cpu_compute_types(cpu_types: set, hw_info: dict) -> list[str]:
    """
    Pick the 1-2 most promising CPU compute types based on CPU capabilities.
    On AVX2/AVX512 hardware int8 is dominant; float variants add no benefit.
    """
    if hw_info.get("avx512") or hw_info.get("avx2"):
        preferred = ["int8", "int8_float32"]
    else:
        preferred = ["int8_float32", "int8", "float32"]
    return [t for t in preferred if t in cpu_types]


def _smart_cpu_thread_counts(hw_info: dict) -> list[int]:
    """
    Return 2 thread counts to test.  Estimates physical cores from logical
    (assumes 2-way hyperthreading) and tests physical and physical//2.
    """
    logical = hw_info.get("logical_cores", multiprocessing.cpu_count())
    physical_est = max(2, logical // 2)
    counts = sorted({max(2, physical_est // 2), physical_est})
    return [t for t in counts if t <= logical]


def _get_hardware_candidates(hw_info: dict = None, mode: str = "smart") -> list[dict]:
    """
    Returns (device, compute_type, cpu_threads) combinations to benchmark.

    mode="smart" (default): tests only the most promising configs.
      CPU: 1-2 best compute types × 2 smart thread counts (~4 candidates).
    mode="full": exhaustive — all compute types × [2,4,6,8] threads.

    num_workers is always 1 — a second worker loads a second model copy but has
    no effect on single-request latency (only relevant for concurrent batches).
    """
    if hw_info is None:
        hw_info = {"logical_cores": multiprocessing.cpu_count(),
                   "avx2": False, "avx512": False}

    candidates = []
    logical_cores = hw_info.get("logical_cores", multiprocessing.cpu_count())

    import ctranslate2

    if mode == "full":
        thread_counts = [t for t in [2, 4, 6, 8] if t <= logical_cores]
    else:
        thread_counts = _smart_cpu_thread_counts(hw_info)

    # ── CPU ──────────────────────────────────────────────────────────────────
    try:
        cpu_types = ctranslate2.get_supported_compute_types("cpu")
        if mode == "full":
            cpu_type_order = ["int8", "int8_float32", "int8_float16", "float16", "float32"]
            ranked = [t for t in cpu_type_order if t in cpu_types]
        else:
            ranked = _smart_cpu_compute_types(cpu_types, hw_info)
        for ct in ranked:
            for threads in thread_counts:
                candidates.append({
                    "device":       "cpu",
                    "compute_type": ct,
                    "cpu_threads":  threads,
                    "num_workers":  1,
                    "omp_threads":  threads,
                    "label":        f"CPU {ct} {threads}t",
                })
    except Exception:
        pass

    # ── CUDA ─────────────────────────────────────────────────────────────────
    try:
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        # Exclude float32 (no quantization benefit) and bare int8 (hangs on
        # entry-level 2GB GPUs like MX350 — confirmed by benchmark).
        cuda_types = cuda_types - {"float32", "int8"}
        cuda_order = ["int8_float16", "float16", "int8_float32"]
        ranked_cuda = [t for t in cuda_order if t in cuda_types]
        for ct in ranked_cuda:
            candidates.append({
                "device":       "cuda",
                "compute_type": ct,
                "cpu_threads":  4,
                "num_workers":  1,
                "omp_threads":  4,
                "label":        f"CUDA {ct}",
            })
    except Exception:
        pass

    # ── OpenVINO ─────────────────────────────────────────────────────────────
    try:
        import openvino as ov
        core = ov.Core()
        intel_gpus = [d for d in core.available_devices if d.startswith("GPU")]
        for device_id in intel_gpus:
            try:
                name = core.get_property(device_id, "FULL_DEVICE_NAME").lower()
            except Exception:
                name = ""
            if "nvidia" in name or "geforce" in name or "radeon" in name:
                continue
            candidates.append({
                "device":          "openvino",
                "openvino_device": device_id,
                "compute_type":    "int8",
                "cpu_threads":     4,
                "num_workers":     1,
                "omp_threads":     4,
                "is_igpu":         True,
                "label":           f"OpenVINO {device_id}",
            })
        candidates.append({
            "device":          "openvino",
            "openvino_device": "CPU",
            "compute_type":    "int8",
            "cpu_threads":     4,
            "num_workers":     1,
            "omp_threads":     4,
            "is_igpu":         False,
            "label":           "OpenVINO CPU",
        })
    except ImportError:
        pass

    return candidates


_CUDA_MIN_VRAM_MB = 1800  # int8_float32 works on 2GB (MX350 confirmed); int8 hangs

def _estimate_config_heuristic(hw_info: dict) -> dict[str, Optional[dict]]:
    """
    Derive a reasonable config from hardware info alone — no inference timing.
    Used by quick mode and when skip_benchmark=true in config.yaml.
    All results carry status="estimated" so callers can distinguish from timed runs.

    CUDA is only chosen when VRAM >= _CUDA_MIN_VRAM_MB (3 GB). Entry-level
    cards like MX350 (2 GB) fail or hang loading large-v3-turbo — benchmark
    evidence from i5-1135G7 / MX350 machine confirmed this.

    For AVX2 CPUs, 2 threads outperforms physical core count because the
    model is memory-bandwidth bound; adding threads increases contention.
    """
    logical = hw_info.get("logical_cores", 4)
    physical_est = max(2, logical // 2)
    # Use physical core count — test_cpu.py confirmed 4t (physical) beats 2t on
    # i5-1135G7 AVX2+AVX512. Earlier 2t preference was based on flawed benchmark
    # data where CUDA DLL failures distorted CPU measurements.
    cpu_threads = physical_est

    cuda_ok = (
        hw_info.get("cuda_available")
        and hw_info.get("gpu_vram_mb", 0) >= _CUDA_MIN_VRAM_MB
    )

    if cuda_ok:
        # int8_float16 requires tensor cores (4GB+ GPUs); entry-level cards
        # like MX350 (2GB) use int8_float32. Bare int8 hangs on 2GB VRAM.
        vram_mb = hw_info.get("gpu_vram_mb", 0)
        cuda_compute = "int8_float16" if vram_mb >= 4000 else "int8_float32"
        base: dict = {
            "device":       "cuda",
            "compute_type": cuda_compute,
            "cpu_threads":  4,
            "num_workers":  1,
            "omp_threads":  4,
        }
    else:
        igpu_devices = [d for d in hw_info.get("openvino_devices", [])
                        if d.startswith("GPU")]
        ov_model_ready = (ROOT / "ov_model").exists()
        if hw_info.get("openvino_available") and igpu_devices and ov_model_ready:
            base = {
                "device":          "openvino",
                "openvino_device": igpu_devices[0],
                "compute_type":    "int8",
                "cpu_threads":     4,
                "num_workers":     1,
                "omp_threads":     4,
                "is_igpu":         True,
            }
        else:
            base = {
                "device":       "cpu",
                "compute_type": "int8",
                "cpu_threads":  cpu_threads,
                "num_workers":  1,
                "omp_threads":  cpu_threads,
            }

    result: dict[str, Optional[dict]] = {}
    for b in list(BUCKETS.keys()) + ["streaming"]:
        result[b] = {**base, "auto_accuracy_tier": "balanced",
                     "rtf": None, "status": "estimated"}
    return result


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def _register_cuda_dlls() -> None:
    """Add nvidia package DLL folders to PATH (required on Windows)."""
    import site
    import pathlib
    for sp in site.getsitepackages():
        nvidia_path = pathlib.Path(sp) / "nvidia"
        if nvidia_path.exists():
            for dll_dir in nvidia_path.rglob("*.dll"):
                folder = str(dll_dir.parent)
                if folder not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = folder + ";" + os.environ.get("PATH", "")


def _load_model(candidate: dict):
    omp = str(candidate.get("omp_threads", candidate["cpu_threads"]))
    os.environ["OMP_NUM_THREADS"] = omp
    os.environ["MKL_NUM_THREADS"] = omp

    device = candidate["device"]
    if device == "cuda":
        _register_cuda_dlls()
    if device in ("cpu", "cuda"):
        from faster_whisper import WhisperModel
        return WhisperModel(
            MODEL_ID,
            device=device,
            compute_type=candidate["compute_type"],
            cpu_threads=candidate["cpu_threads"],
            num_workers=candidate["num_workers"],
        )
    elif device == "openvino":
        import openvino_genai as ov_genai
        ov_model_dir = ROOT / "ov_model"
        if not ov_model_dir.exists():
            raise RuntimeError(
                f"OpenVINO model not found at {ov_model_dir}. "
                "Run tests/test_openvino.py to convert the model first."
            )
        return ov_genai.WhisperPipeline(
            str(ov_model_dir), device=candidate.get("openvino_device", "CPU")
        )
    raise RuntimeError(f"Unknown device: {device!r}")


def _transcribe_model(model, audio: np.ndarray, candidate: dict) -> str:
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
# Per-candidate session (one model load, all buckets + streaming)
# ---------------------------------------------------------------------------

def _run_candidate_session(
    candidate: dict,
    bucket_audio: dict[str, tuple[np.ndarray, float]],
) -> dict[str, dict]:
    """
    Load model once, run warm-up, then run every bucket + streaming trial in
    a single session.  Returns {bucket_name: result_dict}.

    Warm-up uses the shortest available audio.  If warm-up RTF exceeds
    _WARMUP_SKIP_RTF the candidate is marked slow_skip for all buckets.
    """
    # Ordered buckets that actually have audio
    ordered = [b for b in BUCKET_ORDER if b in bucket_audio]
    if not ordered:
        return {}

    results: dict[str, dict] = {}

    try:
        model = _load_model(candidate)
    except Exception as e:
        err = str(e)[:120]
        for b in ordered + ["streaming"]:
            results[b] = {**candidate, "status": "failed", "error": err}
        return results

    try:
        # ── Warm-up ──────────────────────────────────────────────────────────
        first_audio, first_dur = bucket_audio[ordered[0]]
        _transcribe_model(model, first_audio, candidate)

        # Quick warmup-RTF check — if clearly too slow, bail out early
        t0 = time.time()
        _transcribe_model(model, first_audio, candidate)
        warmup_rtf = (time.time() - t0) / first_dur

        if warmup_rtf > _WARMUP_SKIP_RTF:
            del model; gc.collect()
            for b in ordered + ["streaming"]:
                results[b] = {**candidate, "status": "slow_skip", "rtf": round(warmup_rtf, 4)}
            return results

        # ── Bucket trials ─────────────────────────────────────────────────────
        for b in ordered:
            audio, dur = bucket_audio[b]
            # First bucket reuses the warmup pass result to avoid an extra call
            if b == ordered[0]:
                elapsed = (time.time() - t0)
                rtf = warmup_rtf
            else:
                t0 = time.time()
                _transcribe_model(model, audio, candidate)
                elapsed = time.time() - t0
                rtf = elapsed / dur
            results[b] = {
                **candidate,
                "audio_duration": dur,
                "elapsed":        round(elapsed, 3),
                "rtf":            round(rtf, 4),
                "status":         "ok",
            }

        # ── Streaming trial ───────────────────────────────────────────────────
        stream_bucket = ordered[0]           # use shortest available audio
        s_audio, s_dur = bucket_audio[stream_bucket]
        rtfs = []
        for _ in range(5):
            t0 = time.time()
            _transcribe_model(model, s_audio, candidate)
            rtfs.append(round((time.time() - t0) / s_dur, 4))
        median_rtf = sorted(rtfs[1:])[len(rtfs[1:]) // 2]
        results["streaming"] = {
            **candidate,
            "audio_duration": s_dur,
            "median_rtf":     round(median_rtf, 4),
            "rtf_per_call":   rtfs,
            "status":         "ok",
        }

    except Exception as e:
        err = str(e)[:120]
        for b in ordered + ["streaming"]:
            if b not in results:
                results[b] = {**candidate, "status": "failed", "error": err}
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Venv-isolated candidate execution
# ---------------------------------------------------------------------------

def _run_candidate_in_venv(
    candidate: dict,
    audio_files_info: dict[str, dict],
) -> dict[str, dict]:
    """
    Run _run_candidate_session equivalent inside the appropriate device venv.
    Returns {bucket_name: result_dict} with candidate fields merged in — same
    shape as _run_candidate_session() so the rest of run_benchmark() is unchanged.

    audio_files_info: {bucket: {"path": str, "target_duration": float}}
    """
    from core.venv_manager import ensure_venv, python_exe as venv_python

    device = candidate["device"]
    ensure_venv(device)

    py = str(venv_python(device))
    worker_script = str(Path(__file__).parent / "benchmark_worker.py")

    request = {
        "candidate":   {**candidate, "model_id": MODEL_ID},
        "audio_files": audio_files_info,
        "root":        str(ROOT),
        "sample_rate": SAMPLE_RATE,
    }

    ordered = [b for b in BUCKET_ORDER if b in audio_files_info]
    all_buckets = ordered + ["streaming"]

    try:
        # stderr is not captured — it flows directly to the terminal for progress.
        proc = subprocess.Popen(
            [py, worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
        )
        stdout_data, _ = proc.communicate(
            input=json.dumps(request),
            timeout=600,
        )

        if proc.returncode != 0:
            err = f"worker exited with code {proc.returncode}"
            return {b: {**candidate, "status": "failed", "error": err}
                    for b in all_buckets}

        raw = json.loads(stdout_data.strip())

        if "_error" in raw:
            err = raw["_error"]
            return {b: {**candidate, "status": "failed", "error": err}
                    for b in all_buckets}

        # Merge candidate fields into each bucket result (mirrors _run_candidate_session)
        return {b: {**candidate, **br} for b, br in raw.items()}

    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return {b: {**candidate, "status": "failed", "error": "timeout (>600s)"}
                for b in all_buckets}
    except Exception as e:
        return {b: {**candidate, "status": "failed", "error": str(e)[:200]}
                for b in all_buckets}


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def _collect_hardware_info() -> dict:
    import ctranslate2

    logical_cores = multiprocessing.cpu_count()
    cpu_name = "unknown"
    avx2 = False
    avx512 = False

    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        cpu_name = info.get("brand_raw") or info.get("brand") or "unknown"
        flags = set(info.get("flags", []))
        avx2   = "avx2"    in flags
        avx512 = "avx512f" in flags
    except Exception:
        try:
            import subprocess
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name", "/value"],
                text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if line.startswith("Name="):
                    cpu_name = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass

    cuda_available = False
    gpu_name = None
    gpu_vram_mb = 0
    try:
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types and cuda_types != {"float32"}:
            cuda_available = True
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        if out:
            first_line = out.split("\n")[0]
            parts = first_line.rsplit(",", 1)
            gpu_name = parts[0].strip()
            if len(parts) == 2:
                try:
                    gpu_vram_mb = int(parts[1].strip())
                except ValueError:
                    pass
    except Exception:
        pass

    openvino_available = False
    openvino_devices: list = []
    openvino_device_names: dict = {}
    try:
        import openvino as ov
        core = ov.Core()
        openvino_available = True
        openvino_devices = list(core.available_devices)
        for d in openvino_devices:
            try:
                openvino_device_names[d] = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                openvino_device_names[d] = d
    except ImportError:
        pass

    return {
        "cpu":                   cpu_name,
        "logical_cores":         logical_cores,
        "avx2":                  avx2,
        "avx512":                avx512,
        "cuda_available":        cuda_available,
        "gpu_name":              gpu_name,
        "gpu_vram_mb":           gpu_vram_mb,
        "openvino_available":    openvino_available,
        "openvino_devices":      openvino_devices,
        "openvino_device_names": openvino_device_names,
    }


# ---------------------------------------------------------------------------
# iGPU preference
# ---------------------------------------------------------------------------

def _apply_igpu_preference(best: dict, results: dict, margin: float) -> dict:
    """Prefer iGPU if its RTF ≤ best_rtf × (1 + margin)."""
    if margin <= 0:
        return best

    updated = dict(best)
    for bucket, cfg in best.items():
        if cfg is None:
            continue
        best_rtf = cfg.get("rtf") or cfg.get("median_rtf")
        if best_rtf is None:
            continue
        threshold = best_rtf * (1 + margin)

        igpu_candidate = None
        igpu_rtf = None
        for r in results.get(bucket, []):
            if r.get("status") != "ok" or not r.get("is_igpu"):
                continue
            r_rtf = r.get("rtf") or r.get("median_rtf")
            if r_rtf is None or r_rtf > threshold:
                continue
            if igpu_rtf is None or r_rtf < igpu_rtf:
                igpu_candidate = r
                igpu_rtf = r_rtf

        if igpu_candidate:
            print(f"  [iGPU preference] {bucket}: {igpu_candidate['label']} "
                  f"RTF {igpu_rtf:.3f} ≤ best {best_rtf:.3f} × {1+margin:.2f} — preferring iGPU")
            new_cfg = {k: v for k, v in igpu_candidate.items()
                       if k not in ("label", "audio_duration",
                                    "elapsed", "rtf_per_call", "note", "status")}
            new_cfg["rtf"] = igpu_rtf
            new_cfg["auto_accuracy_tier"] = cfg.get("auto_accuracy_tier", "fast")
            new_cfg["igpu_preferred"] = True
            updated[bucket] = new_cfg

    return updated


# ---------------------------------------------------------------------------
# Config helpers (used by both run_benchmark and get_best_config)
# ---------------------------------------------------------------------------

def _load_config_yaml() -> dict:
    try:
        import yaml
        path = ROOT / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(force: bool = False, quick: bool = False, full: bool = False) -> None:
    """
    Run benchmark and write benchmark_results.json.

    quick=True  — heuristic only, no inference. Writes estimated config instantly.
    full=True   — exhaustive: all compute types × thread counts [2,4,6,8].
    default     — smart mode: tests only the most promising configs (3-5× faster).

    Each candidate is loaded once; all 4 buckets + streaming trial are run in
    that single session before the model is unloaded.
    """
    if RESULTS_PATH.exists() and not force:
        print("[Benchmark] Results already exist. Use --force to re-run.")
        return

    if full:
        mode = "full"
    else:
        # Respect benchmark_mode from config.yaml unless explicitly overridden
        cfg_mode = _load_config_yaml().get("benchmark_mode", "smart")
        mode = "full" if cfg_mode == "full" else "smart"
    mode_label = "quick (heuristic)" if quick else mode
    print(f"[Benchmark] Starting hardware benchmark ({mode_label} mode)...")
    print(f"[Benchmark] Model: {MODEL_ID}")

    hw = _collect_hardware_info()
    avx_str = ("AVX2" if hw["avx2"] else "") + (" AVX-512" if hw["avx512"] else "")
    print(f"[Benchmark] CPU: {hw['cpu']} "
          f"({hw['logical_cores']} cores{', ' + avx_str if avx_str else ''})")
    if hw["cuda_available"]:
        print(f"[Benchmark] GPU: {hw.get('gpu_name', 'unknown')}")
    if hw["openvino_available"]:
        print(f"[Benchmark] OpenVINO devices: {hw['openvino_devices']}")
        for d, name in hw.get("openvino_device_names", {}).items():
            print(f"  {d}: {name}")

    # ── Quick mode: heuristic only, no inference ──────────────────────────────
    if quick:
        best = _estimate_config_heuristic(hw)
        output = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_id":  MODEL_ID,
            "hardware":  hw,
            "results":   {b: [] for b in list(BUCKETS.keys()) + ["streaming"]},
            "best":      best,
            "mode":      "quick",
        }
        RESULTS_PATH.write_text(json.dumps(output, indent=2))
        print(f"\n[Benchmark] Quick mode: estimated config from hardware detection.")
        device = best["short"]["device"] if best.get("short") else "?"
        print(f"[Benchmark] Estimated best device: {device}")
        print(f"[Benchmark] Results written to {RESULTS_PATH}")
        print("[Benchmark] Re-run without --quick for timed RTF measurements.")
        return

    records = _get_records_by_bucket()
    for bucket, rec in records.items():
        if rec:
            path, target_dur = rec
            print(f"[Benchmark] {bucket}: {path.name} [{target_dur:.0f}s slice]")
        else:
            print(f"[Benchmark] {bucket}: no audio file found in records/")

    candidates = _get_hardware_candidates(hw_info=hw, mode=mode)
    print(f"[Benchmark] {len(candidates)} candidates ({mode} mode) — "
          f"model loaded once per candidate\n")

    # Build audio file info for benchmark workers (workers decode internally)
    audio_files_info: dict[str, dict] = {}
    for bucket in BUCKET_ORDER:
        rec = records.get(bucket)
        if rec is None:
            continue
        path, target_dur = rec
        audio_files_info[bucket] = {
            "path":            str(path.resolve()),
            "target_duration": target_dur,
        }

    results: dict[str, list] = {b: [] for b in list(BUCKETS.keys()) + ["streaming"]}
    best: dict[str, Optional[dict]] = {b: None for b in list(BUCKETS.keys()) + ["streaming"]}

    # Track compute_types that already have an elite result → skip slower thread counts
    elite_compute_types: set[str] = set()

    for i, candidate in enumerate(candidates):
        label = candidate["label"]
        device = candidate["device"]

        # Skip remaining thread-count variants if this compute_type is already elite
        if device == "cpu":
            ct_key = f"cpu_{candidate['compute_type']}"
            if ct_key in elite_compute_types:
                print(f"  [{i+1}/{len(candidates)}] {label:<40} SKIPPED (elite already found)")
                continue

        print(f"  [{i+1}/{len(candidates)}] {label:<40}", flush=True)

        session = _run_candidate_in_venv(candidate, audio_files_info)

        # Collect results
        for b in BUCKET_ORDER:
            r = session.get(b)
            if r is None:
                continue
            results[b].append(r)
            if r["status"] == "ok":
                rtf = r["rtf"]
                if best[b] is None or rtf < best[b]["rtf"]:
                    best[b] = {k: v for k, v in candidate.items() if k != "label"}
                    best[b]["rtf"] = rtf

        # Streaming result
        r_stream = session.get("streaming")
        if r_stream:
            results["streaming"].append(r_stream)
            if r_stream["status"] == "ok":
                mrt = r_stream["median_rtf"]
                if best["streaming"] is None or mrt < best["streaming"].get("rtf", 999):
                    best["streaming"] = {k: v for k, v in candidate.items() if k != "label"}
                    best["streaming"]["rtf"] = mrt

        # Print summary line for this candidate
        bucket_rtfs = " | ".join(
            f"{b[:3]} {session[b]['rtf']:.3f}" if session.get(b, {}).get("status") == "ok"
            else f"{b[:3]} {session.get(b,{}).get('status','?')}"
            for b in BUCKET_ORDER if b in session
        )
        stream_r = session.get("streaming", {})
        if stream_r.get("status") == "ok":
            bucket_rtfs += f" | str {stream_r['median_rtf']:.3f}"
        print(f"  {bucket_rtfs}")

        # Mark elite if best bucket RTF < threshold
        if device == "cpu":
            ct_key = f"cpu_{candidate['compute_type']}"
            long_r = session.get("long", session.get("medium", {}))
            if long_r.get("status") == "ok" and long_r["rtf"] < _ELITE_RTF:
                elite_compute_types.add(ct_key)

    print()

    # Add auto_accuracy_tier
    for bucket, cfg in best.items():
        if cfg:
            rtf = cfg.get("rtf") or cfg.get("median_rtf", 1.0)
            if rtf * 4.5 < 0.85:
                tier = "accurate"
            elif rtf * 1.8 < 0.85:
                tier = "balanced"
            else:
                tier = "fast"
            cfg["auto_accuracy_tier"] = tier

    # Apply iGPU preference
    igpu_margin = _load_igpu_margin()
    if igpu_margin > 0:
        print(f"[Benchmark] Applying iGPU preference (margin={igpu_margin:.0%})...")
        best = _apply_igpu_preference(best, results, igpu_margin)

    # Summary table
    print("\n[Benchmark] Best configs:")
    for bucket in BUCKET_ORDER + ["streaming"]:
        cfg = best.get(bucket)
        if cfg:
            rtf_val = cfg.get("rtf") or cfg.get("median_rtf")
            rtf_str = f"RTF {rtf_val:.3f}" if rtf_val else "RTF ?"
            if cfg["device"] == "openvino":
                hw_str = f"openvino/{cfg.get('openvino_device','')}"
            else:
                hw_str = f"{cfg['device']} {cfg['compute_type']} {cfg['cpu_threads']}t"
            print(f"  {bucket:<12} {hw_str:<30} {rtf_str}  tier={cfg.get('auto_accuracy_tier','?')}")
        else:
            print(f"  {bucket:<12} (no result)")

    # ── Venv cleanup: keep only winning device venvs ─────────────────────────
    from core import venv_manager
    winning_devices = {cfg["device"] for cfg in best.values() if cfg}
    all_devices = set(venv_manager.DEVICE_PACKAGES.keys())
    for dev in all_devices - winning_devices:
        if venv_manager.venv_exists(dev):
            print(f"[Benchmark] Removing unused {dev} venv...")
            venv_manager.delete_venv(dev)

    # Stamp venv_path into each winning config so Transcriber can use it
    for cfg in best.values():
        if cfg is None:
            continue
        vp = venv_manager.venv_path(cfg["device"])
        if vp.exists():
            cfg["venv_path"] = str(vp)

    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id":  MODEL_ID,
        "hardware":  hw,
        "results":   results,
        "best":      best,
        "mode":      mode,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\n[Benchmark] Done. Results written to {RESULTS_PATH}")


def _load_igpu_margin() -> float:
    try:
        return float(_load_config_yaml().get("igpu_preference_margin", 0.0))
    except Exception:
        return 0.0


def force_rebenchmark() -> None:
    run_benchmark(force=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_best_config(bucket: str) -> dict:
    """
    Returns the empirically best config for the given bucket.

    If skip_benchmark=true in config.yaml AND force_device + force_compute_type
    are both set, returns the manual config immediately without touching
    benchmark_results.json.  Otherwise auto-triggers benchmark if missing/empty.

    Falls back to nearest bucket if requested one has no result.
    Respects force_device / force_compute_type / force_cpu_threads from config.yaml.
    """
    user_cfg = _load_config_yaml()

    # Skip benchmark entirely when explicitly requested + manual config is complete
    if user_cfg.get("skip_benchmark") and user_cfg.get("force_device") and \
            user_cfg.get("force_compute_type"):
        logical = multiprocessing.cpu_count()
        threads = user_cfg.get("force_cpu_threads") or max(2, logical // 2)
        cfg: dict = {
            "device":       user_cfg["force_device"],
            "compute_type": user_cfg["force_compute_type"],
            "cpu_threads":  threads,
            "num_workers":  1,
            "omp_threads":  threads,
            "auto_accuracy_tier": "balanced",
            "status": "manual",
        }
        return cfg

    if not RESULTS_PATH.exists():
        run_benchmark()
    else:
        data = json.loads(RESULTS_PATH.read_text())
        if all(v is None for v in data.get("best", {}).values()):
            run_benchmark(force=True)

    data = json.loads(RESULTS_PATH.read_text())
    best = data.get("best", {})

    cfg = best.get(bucket)
    if cfg:
        return _apply_config_overrides(cfg)

    all_buckets = BUCKET_ORDER + ["streaming"]
    if bucket in all_buckets:
        idx = all_buckets.index(bucket)
        for delta in range(1, len(all_buckets)):
            for direction in (-1, 1):
                i = idx + direction * delta
                if 0 <= i < len(all_buckets):
                    cfg = best.get(all_buckets[i])
                    if cfg:
                        return _apply_config_overrides(cfg)

    return {"device": "cpu", "compute_type": "int8", "cpu_threads": 4,
            "num_workers": 1, "omp_threads": 4}


def _apply_config_overrides(cfg: dict) -> dict:
    cfg = dict(cfg)
    try:
        user_cfg = _load_config_yaml()
        if user_cfg.get("force_device"):
            cfg["device"] = user_cfg["force_device"]
        if user_cfg.get("force_compute_type"):
            cfg["compute_type"] = user_cfg["force_compute_type"]
        if user_cfg.get("force_cpu_threads", 0) > 0:
            cfg["cpu_threads"] = user_cfg["force_cpu_threads"]
    except Exception:
        pass
    return cfg
