"""
tests/test_local_config.py
--------------------------
Full hardware sweep: CUDA (MX350) × CPU (i5-1135G7) × iGPU (OpenVINO / Iris Xe).
Measures RTF per bucket per tier, streaming chunk latency, model cold-start time,
WER, CER, hallucination rate, and dropout rate on real Hebrew audio.

Usage:
    python tests/test_local_config.py

Output:
    Console summary + tests/local_config_results.md

Phases
------
Phase 1   — RTF + latency
    CUDA candidates: int8 / int8_float32 / float32  ×  cpu_threads 2/4/6
    CPU  candidates: int8 / int8_float32 / float32  ×  threads×workers
                     (2/1)  (4/1)  (4/2)  (6/1)  (8/1)  (8/2)
    OMP_NUM_THREADS and MKL_NUM_THREADS set for each CPU config.
    Preference ordering applied: int8 > int8_float32 > float32.
    One warm-up pass (short bucket), then two timed passes per bucket per tier.
    Cold-start time recorded per candidate.

Phase 1b  — iGPU (OpenVINO / Intel Iris Xe) RTF
    Skipped gracefully if openvino_genai is not installed or converted model
    is not found. RTF only — no WER (different model format).
    Converted model expected at: models_ov/whisper-large-v3-turbo-ov/
    (run tests/test_openvino.py first to create it).

Phase 1c  — Streaming simulation
    Top-3 fastest candidates process STREAM_CHUNKS consecutive real ~5s chunks
    back-to-back with model loaded once (mirrors LiveStreamer behaviour).
    Reports median / p95 chunk latency (ms).

Phase 2   — WER / CER / hallucination / dropout
    Top-3 fully-feasible candidates × 40 paired files from audios_/ + refs_/.
    Each candidate run twice: VAD on (file mode) and VAD off (streaming mode).
    Hallucination: >30% more words than reference.
    Dropout:       <70% of reference words (content-loss / over-aggressive VAD).

Phase 3   — Recommendation
    Ranks all candidates; prints ready-to-paste config.yaml block.

Runtime estimate: 30–60 min (dominated by CPU candidates).
"""
from __future__ import annotations

import gc
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout/stderr on Windows (default cp1255 can't encode ✓ ✗ → etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add NVIDIA DLL dirs to PATH before any CUDA imports (Windows only).
# ctranslate2 calls LoadLibraryW("cublas64_12.dll") which searches PATH, not the
# Python DLL directory list — so os.environ["PATH"] is the correct fix here.
if sys.platform == "win32":
    import site as _site
    _nvidia_bins = [
        str(_d) for _sp in _site.getsitepackages()
        for _d in Path(_sp).glob("nvidia/*/bin") if _d.is_dir()
    ]
    if _nvidia_bins:
        os.environ["PATH"] = ";".join(_nvidia_bins) + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for _d in _nvidia_bins:
                os.add_dll_directory(_d)

# ── Constants ──────────────────────────────────────────────────────────────────
CHECKPOINT_FILE      = ROOT / "tests" / "local_config_checkpoint.json"
MODEL_ID             = "ivrit-ai/whisper-large-v3-turbo-ct2"
OV_MODEL_DIR         = ROOT / "models_ov" / "whisper-large-v3-turbo-ov"
SAMPLE_RATE          = 16_000
RTF_BUDGET           = 0.85
RTF_SLOW_SKIP        = 5.0    # warm-up RTF > this  → skip entire candidate
                              # (5s slice has high per-call overhead; BUCKET_SKIP handles real filtering)
RTF_BUCKET_SKIP      = 1.5    # per-bucket RTF > this on >=20s audio → skip remaining buckets
LIVE_LATENCY_P95_MS  = 4_000   # max acceptable p95 for a ~5s live chunk
RESULTS_FILE         = ROOT / "tests" / "local_config_results.md"
RECORDS_DIR          = ROOT / "records"
CANONICAL_NAME       = "school.mp3"
RANDOM_SEED          = 42

# Dataset definitions: (audio_dir, ref_dir, max_minutes | None=all)
DATASETS = [
    (ROOT / "audios_1", ROOT / "refs_1", None),   # all pairs  (~14.5 min)
    (ROOT / "audios_2", ROOT / "refs_2", 30.0),   # random up to 30 min
    (ROOT / "audios_3", ROOT / "refs_3", 30.0),   # random up to 30 min
]
STREAM_CHUNKS        = 10
STREAM_CHUNK_S       = 5.0
HALLUC_THRESHOLD     = 0.30   # hypothesis > 30% more words = hallucination
DROPOUT_THRESHOLD    = 0.70   # hypothesis < 70% of reference words = dropout

SLICE_S    = {"short": 5.0, "medium": 20.0, "long": 45.0, "extended": 90.0}
TIER_ORDER = ["accurate", "balanced", "fast"]

# ── Candidate definitions ──────────────────────────────────────────────────────
# Preference: int8 > int8_float32 > float32 (mirrors benchmark.py + test_cpu.py)
COMPUTE_PREFERENCE = ["int8", "int8_float32", "float32"]

# CUDA: compute_type × cpu_threads alongside GPU (pre/post processing)
# int8 excluded: MX350 lacks cublas64_12.dll (CUDA 12 cuBLAS) — matches test_gpu.py
CUDA_COMPUTE_TYPES  = ["int8_float32", "float32"]
CUDA_CPU_THREADS    = [2, 4, 6]     # cpu_threads alongside CUDA

# CPU: compute_type × (cpu_threads, num_workers)
# mirrors test_cpu.py THREAD_CONFIGS
CPU_COMPUTE_TYPES   = ["int8", "int8_float32", "float32"]
CPU_THREAD_CONFIGS  = [
    (2, 1),   # 2 threads / 1 worker
    (4, 1),   # 4 threads / 1 worker  ← typical winner on i5-1135G7
    (4, 2),   # 4 threads / 2 workers
    (6, 1),   # 6 threads / 1 worker
    (8, 1),   # 8 threads / 1 worker
    (8, 2),   # 8 threads / 2 workers
]

# Accuracy tiers (self-contained; mirrors core/params.py)
TIERS = {
    "fast": {
        "beam_size": 1, "best_of": 1, "temperature": 0.0,
        "patience": 1.0, "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0, "no_speech_threshold": 0.6,
    },
    "balanced": {
        "beam_size": 3, "best_of": 1, "temperature": 0.0,
        "patience": 1.0, "compression_ratio_threshold": 2.2,
        "log_prob_threshold": -0.8, "no_speech_threshold": 0.5,
    },
    "accurate": {
        "beam_size": 5, "best_of": 3, "temperature": 0.0,
        "patience": 1.5, "compression_ratio_threshold": 1.8,
        "log_prob_threshold": -0.5, "no_speech_threshold": 0.4,
    },
}

# Candidate key: (device, compute_type, cpu_threads, num_workers)
# For CUDA: num_workers=1 always; cpu_threads = the "alongside CUDA" value.
# For CPU:  cpu_threads and num_workers vary per config.


# ── Hebrew normaliser ──────────────────────────────────────────────────────────
_NIKUD = re.compile(r'[ְ-ׇ]')
_HEB   = re.compile(r'[^א-ת\s]')
_PREF  = re.compile(r'\b([וכבלמהש])\s+(?=[א-ת])')
_DISFL = re.compile(r'\b([א-ת])\s+\1(\s+\1)+\b')

def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = " ".join(text.splitlines())
    text = _NIKUD.sub("", text)
    text = _HEB.sub("", text)
    text = _DISFL.sub(r"\1", text)
    text = _PREF.sub(r"\1", text)
    text = re.sub(r"(?<!\S)[א-ת](?!\S)", "", text)
    return " ".join(text.split()).strip()


def _edit(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, bj in enumerate(b, 1):
            prev, d[j] = d[j], prev if ai == bj else 1 + min(d[j], d[j - 1], prev)
    return d[len(b)]

def _wer(hyp: str, ref: str) -> float:
    r = _norm(ref).split()
    return _edit(_norm(hyp).split(), r) / len(r) if r else 0.0

def _cer(hyp: str, ref: str) -> float:
    r = list(_norm(ref))
    return _edit(list(_norm(hyp)), r) / len(r) if r else 0.0

def _is_hallucination(hyp: str, ref: str) -> bool:
    h, r = _norm(hyp).split(), _norm(ref).split()
    return bool(r) and (len(h) - len(r)) / len(r) > HALLUC_THRESHOLD

def _is_dropout(hyp: str, ref: str) -> bool:
    """Hypothesis < 70% of reference words — model dropped content (CoSIH failure mode)."""
    h, r = _norm(hyp).split(), _norm(ref).split()
    return bool(r) and len(h) / len(r) < DROPOUT_THRESHOLD


# ── Audio loading ──────────────────────────────────────────────────────────────
def _load_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return data.astype(np.float32)

def _make_slices(audio: np.ndarray) -> dict[str, np.ndarray]:
    return {b: audio[:int(s * SAMPLE_RATE)] if len(audio) >= int(s * SAMPLE_RATE) else audio
            for b, s in SLICE_S.items()}


# ── Model helpers ──────────────────────────────────────────────────────────────
def _set_mkl_env(cpu_threads: int):
    """Set MKL / OpenMP thread counts (mirrors test_cpu.py)."""
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

def _load_model(device: str, compute_type: str, cpu_threads: int, num_workers: int):
    from faster_whisper import WhisperModel
    if device == "cpu":
        _set_mkl_env(cpu_threads)
    return WhisperModel(MODEL_ID, device=device, compute_type=compute_type,
                        cpu_threads=cpu_threads, num_workers=num_workers)

def _build_kwargs(tier: str, vad: bool = True) -> dict:
    t = TIERS[tier]
    kw = dict(
        language="he",
        beam_size=t["beam_size"], best_of=t["best_of"],
        temperature=t["temperature"], patience=t["patience"],
        compression_ratio_threshold=t["compression_ratio_threshold"],
        log_prob_threshold=t["log_prob_threshold"],
        no_speech_threshold=t["no_speech_threshold"],
        condition_on_previous_text=False,
        without_timestamps=True,
        vad_filter=vad,
    )
    if vad:
        kw["vad_parameters"] = {"min_silence_duration_ms": 300, "speech_pad_ms": 200}
    return kw

def _transcribe(model, audio: np.ndarray, kwargs: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    segs, _ = model.transcribe(audio, **kwargs)
    text = " ".join(s.text for s in segs)
    return text, time.perf_counter() - t0


# ── Candidate builder ──────────────────────────────────────────────────────────
def _build_candidates() -> list[tuple[str, str, int, int]]:
    """Returns list of (device, compute_type, cpu_threads, num_workers)."""
    import ctranslate2
    cands = []

    # CUDA: all supported compute types × cpu_threads alongside GPU
    if ctranslate2.get_cuda_device_count() > 0:
        supported_cuda = ctranslate2.get_supported_compute_types("cuda")
        for ct in COMPUTE_PREFERENCE:
            if ct in supported_cuda and ct in CUDA_COMPUTE_TYPES:
                for nth in CUDA_CPU_THREADS:
                    cands.append(("cuda", ct, nth, 1))

    # CPU: all supported compute types × thread configs
    supported_cpu = ctranslate2.get_supported_compute_types("cpu")
    for ct in COMPUTE_PREFERENCE:
        if ct in supported_cpu and ct in CPU_COMPUTE_TYPES:
            for nth, nw in CPU_THREAD_CONFIGS:
                cands.append(("cpu", ct, nth, nw))

    return cands

def _cand_label(device: str, ct: str, nth: int, nw: int) -> str:
    if device == "cuda":
        return f"cuda/{ct}/ct{nth}"
    return f"cpu/{ct}/t{nth}w{nw}"

def _cand_key(device: str, ct: str, nth: int, nw: int) -> tuple:
    return (device, ct, nth, nw)


# ── VRAM helper ────────────────────────────────────────────────────────────────
def _vram_used_mb() -> float:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return float(out.strip().split("\n")[0])
    except Exception:
        return float("nan")


# ── WER sample ─────────────────────────────────────────────────────────────────
def _build_wer_sample() -> list[dict]:
    """
    Build WER sample from all three datasets:
      audios_1/refs_1 — all pairs (no cap)
      audios_2/refs_2 — random sample up to 30 min
      audios_3/refs_3 — random sample up to 30 min
    """
    import soundfile as sf
    rng = random.Random(RANDOM_SEED)
    all_pairs = []

    for adir, rdir, max_min in DATASETS:
        if not adir.exists():
            print(f"  [warn] {adir.name} not found — skipping")
            continue

        candidates = []
        for wav in sorted(adir.glob("*.wav")):
            ref = rdir / f"{wav.stem}.txt"
            if ref.exists():
                try:
                    dur = sf.info(str(wav)).duration
                    candidates.append({"audio": wav, "ref": ref, "dur": dur,
                                       "dataset": adir.name})
                except Exception:
                    pass

        if max_min is None:
            all_pairs.extend(candidates)
            print(f"  {adir.name}: {len(candidates)} pairs (all)")
        else:
            rng.shuffle(candidates)
            budget_s, total_s, picked = max_min * 60, 0.0, []
            for c in candidates:
                if total_s + c["dur"] > budget_s:
                    continue
                picked.append(c)
                total_s += c["dur"]
            all_pairs.extend(picked)
            print(f"  {adir.name}: {len(picked)}/{len(candidates)} pairs "
                  f"({total_s/60:.1f} min / {max_min:.0f} min budget)")

    rng.shuffle(all_pairs)
    return all_pairs


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _ckpt_load() -> dict:
    import json
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _ckpt_save(data: dict):
    import json
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _ckpt_clear():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"  Checkpoint cleared: {CHECKPOINT_FILE.name}")


# ── Phase 1: RTF + latency sweep ───────────────────────────────────────────────
def phase1_rtf(slices: dict[str, np.ndarray]) -> list[dict]:
    """
    Returns rows: device, compute_type, cpu_threads, num_workers, tier,
                  bucket, rtf, latency_ms, load_ok, load_time_s, vram_mb

    Early-stop rules (mirrors benchmark.py + test_gpu.py):
      1. SLOW_SKIP  — timed warm-up RTF > RTF_SLOW_SKIP (2.0) on short audio
                      → skip entire candidate across all tiers.
      2. BUCKET_SKIP — per-bucket RTF > RTF_BUCKET_SKIP (1.5) on audio >= 20s
                      → skip remaining longer buckets for this tier.
      3. TIER_SKIP  — if fast tier produced zero passing buckets (all > RTF_BUDGET)
                      → skip balanced and accurate tiers (they can only be slower).
    """
    candidates  = _build_candidates()
    total_cands = len(candidates)
    bucket_list = list(SLICE_S.keys())   # ordered: short, medium, long, extended

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    ckpt = _ckpt_load()
    results: list[dict] = ckpt.get("phase1_results", [])
    done_keys: set = set(tuple(k) for k in ckpt.get("phase1_done", []))
    if done_keys:
        print(f"  Resuming: {len(done_keys)}/{total_cands} candidates already done.")

    print(f"\n  {total_cands} candidates x {len(TIERS)} tiers x {len(SLICE_S)} buckets")
    print(f"  Early-stop: slow_skip RTF>{RTF_SLOW_SKIP}  bucket_skip RTF>{RTF_BUCKET_SKIP}(>=20s)  tier_skip if fast=0 pass")

    def _record_skipped(device, ct, nth, nw, tier, buckets, load_ok,
                        load_time_s=float("nan"), vram_mb=float("nan")):
        for b in buckets:
            results.append(dict(device=device, compute_type=ct,
                                cpu_threads=nth, num_workers=nw,
                                tier=tier, bucket=b,
                                rtf=float("nan"), latency_ms=float("nan"),
                                load_ok=load_ok, load_time_s=load_time_s,
                                vram_mb=vram_mb))

    for ci, (device, ct, nth, nw) in enumerate(candidates, 1):
        label = _cand_label(device, ct, nth, nw)
        cand_key = (device, ct, nth, nw)

        if cand_key in done_keys:
            print(f"[Phase 1  {ci}/{total_cands}] {label}  [CACHED]")
            continue

        print(f"\n{'─'*60}")
        print(f"[Phase 1  {ci}/{total_cands}] {label}")

        vram_before = _vram_used_mb() if device == "cuda" else float("nan")
        t_load      = time.perf_counter()
        try:
            model = _load_model(device, ct, nth, nw)
            load_time_s = time.perf_counter() - t_load
        except Exception as e:
            print(f"  [SKIP] Load failed: {e}")
            for tier in TIERS:
                _record_skipped(device, ct, nth, nw, tier, bucket_list, load_ok=False)
            continue

        vram_after = _vram_used_mb() if device == "cuda" else float("nan")
        vram_mb    = (vram_after - vram_before
                      if not (np.isnan(vram_before) or np.isnan(vram_after))
                      else float("nan"))
        print(f"  Cold-start: {load_time_s:.1f}s" +
              (f"  VRAM: +{vram_mb:.0f} MB" if not np.isnan(vram_mb) else ""))

        # ── Rule 1: untimed warm-up then timed RTF check → SLOW_SKIP ────────
        # Two-pass approach (mirrors test_cpu.py):
        #   Pass 1 (untimed) — absorbs CTranslate2 kernel-init / JIT overhead.
        #   Pass 2 (timed)   — measures real inference RTF for SLOW_SKIP gate.
        # Minimal params to avoid triggering vad/sampling code paths.
        kwargs_warmup = {"language": "he", "beam_size": 1, "without_timestamps": True}
        try:
            _transcribe(model, slices["short"], kwargs_warmup)   # untimed init pass
            _, wu_elapsed = _transcribe(model, slices["short"], kwargs_warmup)
            wu_rtf = wu_elapsed / SLICE_S["short"]
        except Exception as e:
            print(f"  [SKIP] Warm-up failed: {e}")
            for tier in TIERS:
                _record_skipped(device, ct, nth, nw, tier, bucket_list,
                                load_ok=True, load_time_s=load_time_s, vram_mb=vram_mb)
            del model; gc.collect(); continue

        if wu_rtf > RTF_SLOW_SKIP:
            print(f"  [SLOW_SKIP] warm-up RTF={wu_rtf:.2f} > {RTF_SLOW_SKIP} -- skipping all tiers")
            for tier in TIERS:
                _record_skipped(device, ct, nth, nw, tier, bucket_list,
                                load_ok=True, load_time_s=load_time_s, vram_mb=vram_mb)
            del model; gc.collect(); continue

        print(f"  Warm-up RTF={wu_rtf:.3f} -- proceeding")

        # ── Tier loop ─────────────────────────────────────────────────────────
        fast_any_pass = False

        for tier in TIERS:
            # ── Rule 3: skip harder tiers if fast tier had no passing buckets ─
            if tier != "fast" and not fast_any_pass:
                print(f"  [TIER_SKIP] fast tier: 0 passing buckets -- skipping {tier}")
                _record_skipped(device, ct, nth, nw, tier, bucket_list,
                                load_ok=True, load_time_s=load_time_s, vram_mb=vram_mb)
                continue

            kwargs   = _build_kwargs(tier)
            row_base = dict(device=device, compute_type=ct, cpu_threads=nth,
                            num_workers=nw, tier=tier, load_ok=True,
                            load_time_s=load_time_s, vram_mb=vram_mb)

            # Warm-up for non-fast tiers (fast tier already warmed up above)
            # Minimal params — purpose is kernel init, not accuracy measurement.
            if tier != "fast":
                try:
                    _transcribe(model, slices["short"], kwargs_warmup)
                except Exception as e:
                    print(f"  [{tier}] Warm-up failed: {e}")
                    _record_skipped(device, ct, nth, nw, tier, bucket_list,
                                    load_ok=True, load_time_s=load_time_s, vram_mb=vram_mb)
                    continue

            print(f"  {tier} (beam={TIERS[tier]['beam_size']}): ", end="", flush=True)
            tier_any_pass = False
            remaining = list(slices.items())

            for bi, (bucket, audio) in enumerate(remaining):
                dur_s   = len(audio) / SAMPLE_RATE
                timings = []
                for _ in range(2):
                    try:
                        _, elapsed = _transcribe(model, audio, kwargs)
                        timings.append(elapsed)
                    except Exception:
                        pass

                if timings:
                    best = min(timings)
                    rtf  = best / dur_s
                    lat  = best * 1000
                else:
                    rtf = lat = float("nan")

                ok = "OK" if not np.isnan(rtf) and rtf < RTF_BUDGET else "!!"
                print(f"{bucket}={rtf:.3f}{ok}({lat:.0f}ms) ", end="", flush=True)
                results.append({**row_base, "bucket": bucket,
                                "rtf": rtf, "latency_ms": lat})

                if not np.isnan(rtf) and rtf < RTF_BUDGET:
                    tier_any_pass = True
                    if tier == "fast":
                        fast_any_pass = True

                # ── Rule 2: BUCKET_SKIP ───────────────────────────────────────
                if not np.isnan(rtf) and rtf > RTF_BUCKET_SKIP and dur_s >= 20:
                    skipped = [b for b, _ in remaining[bi + 1:]]
                    if skipped:
                        print(f"\n  [BUCKET_SKIP] RTF={rtf:.2f}>{RTF_BUCKET_SKIP} on {bucket} ({dur_s:.0f}s) -- skipping {skipped}")
                        _record_skipped(device, ct, nth, nw, tier, skipped,
                                        load_ok=True, load_time_s=load_time_s, vram_mb=vram_mb)
                    break
            print()

        # Save checkpoint after each candidate completes
        done_keys.add(cand_key)
        ckpt["phase1_results"] = results
        ckpt["phase1_done"] = [list(k) for k in done_keys]
        _ckpt_save(ckpt)

        del model
        gc.collect()

    return results


# ── Phase 1b: OpenVINO iGPU RTF ───────────────────────────────────────────────
def phase1b_openvino(slices: dict[str, np.ndarray]) -> list[dict]:
    """
    RTF-only benchmark on Intel Iris Xe via openvino_genai.
    Requires:
      - openvino_genai installed
      - Converted model at OV_MODEL_DIR (run tests/test_openvino.py first)
      - GPU.0 available (Intel iGPU)
    """
    try:
        import openvino_genai as ov_genai
        import openvino as ov
    except ImportError:
        print("  [SKIP] openvino / openvino_genai not installed.")
        return []

    if not OV_MODEL_DIR.exists() or not any(OV_MODEL_DIR.iterdir()):
        print(f"  OpenVINO model not found — converting now (one-time, ~15-20 min).")
        print(f"  Source : ivrit-ai/whisper-large-v3-turbo")
        print(f"  Output : {OV_MODEL_DIR}")
        OV_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import subprocess as _sp
        import shutil as _shutil

        # Python 3.14 has an optimum/transformers version conflict for OV export.
        # Use an isolated Python 3.12 venv where compatible versions can be installed.
        _venv = ROOT / "tests" / ".venv_ov_conv"
        _pip    = _venv / "Scripts" / "pip"
        _py312  = _venv / "Scripts" / "python"
        _optcli = _venv / "Scripts" / "optimum-cli"

        try:
            # Check if venv is fully set up (optimum-cli present)
            if not _optcli.exists():
                print("  Setting up Python 3.12 venv for conversion (~2 min)...")
                _shutil.rmtree(_venv, ignore_errors=True)
                _sp.check_call(["py", "-3.12", "-m", "venv", str(_venv)])
                _sp.check_call([str(_py312), "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
                _sp.check_call([str(_py312), "-m", "pip", "install", "-q",
                    "fsspec<=2026.2.0",
                    "optimum[openvino,onnx]",
                    "optimum-intel[openvino]>=1.25.2",
                    "transformers>=4.45.0",
                ])
                print("  Venv ready.")

            print("  Running optimum-cli export openvino ...")
            _sp.check_call([
                str(_optcli), "export", "openvino",
                "--model", "ivrit-ai/whisper-large-v3-turbo",
                "--task", "automatic-speech-recognition-with-past",
                "--weight-format", "int8",
                str(OV_MODEL_DIR),
            ])
            print("  Conversion done.")
        except Exception as e:
            print(f"  [SKIP] Model conversion failed: {e}")
            _shutil.rmtree(_venv, ignore_errors=True)
            return []

    core = ov.Core()
    av_devices = core.available_devices

    # Determine which Intel GPU is available
    igpu_device = next((d for d in av_devices if d == "GPU.0"), None)
    if igpu_device is None:
        igpu_device = next((d for d in av_devices if d.startswith("GPU")), None)

    # Build list of OV devices to test
    ov_devices = []
    if igpu_device:
        ov_devices.append((igpu_device, "int8_ov",       "igpu"))
        ov_devices.append((f"HETERO:{igpu_device},CPU", "int8_ov_hetero", "igpu_cpu"))
    ov_devices.append(("CPU", "int8_ov_cpu", "ov_cpu"))

    results = []

    for ov_dev, ct_label, dev_label in ov_devices:
        print(f"\n  [{dev_label}] device={ov_dev}")
        try:
            t_load = time.perf_counter()
            pipe = ov_genai.WhisperPipeline(str(OV_MODEL_DIR), ov_dev)
            load_time_s = time.perf_counter() - t_load
            cfg = pipe.get_generation_config()
            cfg.language          = "<|he|>"
            cfg.task              = "transcribe"
            cfg.return_timestamps = False
            print(f"  Loaded in {load_time_s:.1f}s")
        except Exception as e:
            print(f"  [SKIP] Pipeline load failed: {e}")
            continue

        row_base = dict(device=dev_label, compute_type=ct_label, cpu_threads=0,
                        num_workers=0, load_ok=True, load_time_s=load_time_s,
                        vram_mb=float("nan"))

        try:
            pipe.generate(slices["short"].tolist(), cfg)
        except Exception as e:
            print(f"  [SKIP] Warm-up failed: {e}")
            del pipe
            gc.collect()
            continue

        print("  RTF per bucket: ", end="", flush=True)
        for bucket, audio in slices.items():
            dur_s = len(audio) / SAMPLE_RATE
            timings = []
            for _ in range(2):
                try:
                    t0 = time.perf_counter()
                    pipe.generate(audio.tolist(), cfg)
                    timings.append(time.perf_counter() - t0)
                except Exception:
                    pass
            if timings:
                best = min(timings)
                rtf  = best / dur_s
                lat  = best * 1000
            else:
                rtf = lat = float("nan")

            ok = "✓" if not np.isnan(rtf) and rtf < RTF_BUDGET else "✗"
            print(f"{bucket}={rtf:.3f}{ok}({lat:.0f}ms) ", end="", flush=True)
            results.append({**row_base, "tier": "fast", "bucket": bucket,
                            "rtf": rtf, "latency_ms": lat})
        print()

        del pipe
        gc.collect()

    return results


# ── Phase 1c: Streaming simulation ────────────────────────────────────────────
def phase1c_streaming(rtf_results: list[dict],
                      stream_chunks: list[np.ndarray]) -> list[dict]:
    """
    Load model once; feed STREAM_CHUNKS consecutive chunks back-to-back.
    Selects top-3 candidates by RTF on short/fast.
    """
    short_fast = [r for r in rtf_results
                  if r["bucket"] == "short" and r["tier"] == "fast"
                  and r["load_ok"] and not np.isnan(r["rtf"])]
    short_fast.sort(key=lambda r: r["rtf"])
    seen, top3 = set(), []
    for r in short_fast:
        key = _cand_key(r["device"], r["compute_type"], r["cpu_threads"], r["num_workers"])
        if key not in seen:
            seen.add(key)
            top3.append(key)
        if len(top3) == 3:
            break

    if not top3:
        print("  [SKIP] No feasible candidates for streaming simulation.")
        return []

    stream_results = []
    kwargs = _build_kwargs("fast", vad=False)   # no double-VAD in streaming mode

    for device, ct, nth, nw in top3:
        label = _cand_label(device, ct, nth, nw)
        print(f"\n  Sim: {label}  ({STREAM_CHUNKS} × {STREAM_CHUNK_S:.0f}s chunks, VAD off)")

        try:
            model = _load_model(device, ct, nth, nw)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        latencies_ms = []
        for i, chunk in enumerate(stream_chunks):
            _, elapsed = _transcribe(model, chunk, kwargs)
            lat_ms = elapsed * 1000
            latencies_ms.append(lat_ms)
            status = "✓" if lat_ms < LIVE_LATENCY_P95_MS else "✗"
            print(f"    chunk {i+1:2d}: {lat_ms:6.0f} ms {status}")

        del model
        gc.collect()

        arr   = np.array(latencies_ms)
        p50   = float(np.percentile(arr, 50))
        p95   = float(np.percentile(arr, 95))
        first = float(arr[0])
        ok    = p95 < LIVE_LATENCY_P95_MS
        print(f"  -> median {p50:.0f} ms  p95 {p95:.0f} ms  "
              f"first {first:.0f} ms  live-ok: {'YES ✓' if ok else 'NO ✗'}")

        stream_results.append(dict(
            device=device, compute_type=ct, cpu_threads=nth, num_workers=nw,
            median_ms=p50, p95_ms=p95, first_chunk_ms=first,
            live_ok=ok, latencies=latencies_ms,
        ))

    return stream_results


# ── Phase 2: WER / CER / hallucination / dropout ──────────────────────────────
def phase2_accuracy(rtf_results: list[dict],
                    wer_sample: list[dict]) -> list[dict]:
    """
    Top-3 fully-feasible candidates × wer_sample files.
    Each candidate tested with VAD on AND VAD off.
    """
    # A candidate is feasible if ALL 4 buckets pass budget at a given tier
    bucket_pass = defaultdict(lambda: defaultdict(int))
    for row in rtf_results:
        if row["load_ok"] and not np.isnan(row["rtf"]) and row["rtf"] < RTF_BUDGET:
            key = _cand_key(row["device"], row["compute_type"],
                            row["cpu_threads"], row["num_workers"])
            bucket_pass[key][row["tier"]] += 1

    best_per_cand = {}
    for key, tier_counts in bucket_pass.items():
        for tier in TIER_ORDER:
            if tier_counts.get(tier, 0) == len(SLICE_S):
                best_per_cand[key] = tier
                break

    if not best_per_cand:
        print("\n[Phase 2] No fully feasible candidates — skipping.")
        return []

    def _sort(item):
        key, tier = item
        return (TIER_ORDER.index(tier), 0 if key[0] == "cuda" else 1)

    top3    = sorted(best_per_cand.items(), key=_sort)[:3]

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    ckpt = _ckpt_load()
    results: list[dict] = ckpt.get("phase2_results", [])
    done_p2: set = set(tuple(k) for k in ckpt.get("phase2_done", []))
    if done_p2:
        print(f"  Resuming Phase 2: {len(done_p2)} candidates already done.")

    for (device, ct, nth, nw), tier in top3:
        label    = _cand_label(device, ct, nth, nw)
        cand_key = (device, ct, nth, nw)

        if cand_key in done_p2:
            print(f"[Phase 2] {label}  [CACHED]")
            continue

        print(f"\n{'─'*60}")
        print(f"[Phase 2] {label} @ {tier}  ({len(wer_sample)} files x VAD on/off)")

        try:
            model = _load_model(device, ct, nth, nw)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        for vad_on in [True, False]:
            vad_label = "vad=on" if vad_on else "vad=off"
            kwargs    = _build_kwargs(tier, vad=vad_on)
            wers, cers, halluc, dropout = [], [], 0, 0
            t0 = time.perf_counter()

            for i, pair in enumerate(wer_sample, 1):
                ref_text = _norm(pair["ref"].read_text(encoding="utf-8"))
                try:
                    audio = _load_audio(pair["audio"])
                    hyp, _ = _transcribe(model, audio, kwargs)
                    wers.append(_wer(hyp, ref_text))
                    cers.append(_cer(hyp, ref_text))
                    if _is_hallucination(hyp, ref_text): halluc  += 1
                    if _is_dropout(hyp, ref_text):       dropout += 1
                except Exception as e:
                    print(f"  [warn] {pair['audio'].name}: {e}")

                if i % 10 == 0:
                    eta = (time.perf_counter() - t0) / i * (len(wer_sample) - i)
                    print(f"  [{vad_label}] {i}/{len(wer_sample)}  "
                          f"WER={np.mean(wers):.3f}  CER={np.mean(cers):.3f}  "
                          f"Halluc={halluc}  Dropout={dropout}  ETA {eta:.0f}s")

            mwer = float(np.mean(wers)) if wers else float("nan")
            mcer = float(np.mean(cers)) if cers else float("nan")
            print(f"  [{vad_label}] DONE  WER={mwer:.3f}  CER={mcer:.3f}  "
                  f"Halluc={halluc}  Dropout={dropout}/{len(wers)}")
            results.append(dict(
                device=device, compute_type=ct, cpu_threads=nth, num_workers=nw,
                tier=tier, vad=vad_on, mean_wer=mwer, mean_cer=mcer,
                hallucination_count=halluc, dropout_count=dropout, files=len(wers),
            ))

        done_p2.add(cand_key)
        ckpt["phase2_results"] = results
        ckpt["phase2_done"] = [list(k) for k in done_p2]
        _ckpt_save(ckpt)

        del model
        gc.collect()

    return results


# ── Phase 3: Recommendation ───────────────────────────────────────────────────
def phase3_recommend(rtf_results: list[dict],
                     ov_results:  list[dict],
                     stream_results: list[dict],
                     acc_results: list[dict]) -> str:
    lines = ["\n" + "=" * 70, "RECOMMENDATION", "=" * 70]

    # Best per-bucket tier for each candidate
    by_cand = defaultdict(lambda: defaultdict(list))
    for r in rtf_results + ov_results:
        if r["load_ok"] and not np.isnan(r.get("rtf", float("nan"))):
            key = _cand_key(r["device"], r["compute_type"],
                            r["cpu_threads"], r["num_workers"])
            by_cand[key][r["tier"]].append((r["bucket"], r["rtf"], r["latency_ms"]))

    def best_tier(key):
        for tier in TIER_ORDER:
            buckets = by_cand[key].get(tier, [])
            if len(buckets) == len(SLICE_S) and all(v < RTF_BUDGET for _, v, _ in buckets):
                return tier
        return None

    if acc_results:
        vad_on  = [r for r in acc_results if r["vad"]]
        vad_off = [r for r in acc_results if not r["vad"]]
        best    = min(vad_on or acc_results,
                      key=lambda r: r["mean_wer"] if not np.isnan(r["mean_wer"]) else 99)
        key     = _cand_key(best["device"], best["compute_type"],
                            best["cpu_threads"], best["num_workers"])
        label   = _cand_label(*key)
        lines.append(f"\nBest candidate (file mode / VAD on): {label} @ {best['tier']}")
        lines.append(f"  WER={best['mean_wer']:.3f}  CER={best['mean_cer']:.3f}  "
                     f"Halluc={best['hallucination_count']}  Dropout={best['dropout_count']}/{best['files']}")

        if vad_off:
            best_off = min(vad_off,
                           key=lambda r: r["mean_wer"] if not np.isnan(r["mean_wer"]) else 99)
            ko       = _cand_key(best_off["device"], best_off["compute_type"],
                                  best_off["cpu_threads"], best_off["num_workers"])
            lo       = _cand_label(*ko)
            delta    = best_off["mean_wer"] - best["mean_wer"]
            lines.append(f"\nBest candidate (streaming / VAD off): {lo} @ {best_off['tier']}")
            lines.append(f"  WER={best_off['mean_wer']:.3f}  CER={best_off['mean_cer']:.3f}  "
                         f"VAD-off delta: {delta:+.3f} "
                         f"({'double-VAD hurts' if delta > 0.005 else 'VAD neutral on clean audio'})")
    else:
        key = None
        lines.append("\n[No WER data — RTF-only recommendation]")

    if stream_results:
        best_s = sorted(stream_results, key=lambda r: r["median_ms"])[0]
        sl     = _cand_label(best_s["device"], best_s["compute_type"],
                             best_s["cpu_threads"], best_s["num_workers"])
        lines.append(f"\nBest streaming candidate: {sl}")
        lines.append(f"  Median {best_s['median_ms']:.0f} ms  p95 {best_s['p95_ms']:.0f} ms  "
                     f"Live-OK: {'YES ✓' if best_s['live_ok'] else 'NO ✗'}")

    if ov_results:
        ov_rtfs = {b: r["rtf"] for r in ov_results for b in [r["bucket"]]}
        lines.append(f"\niGPU (OpenVINO Iris Xe) RTF: " +
                     "  ".join(f"{b}={ov_rtfs.get(b, float('nan')):.3f}" for b in SLICE_S))

    if key:
        d, ct, nth, nw = key
        bucket_tiers   = {}
        for tier in TIER_ORDER:
            for r in rtf_results:
                rkey = _cand_key(r["device"], r["compute_type"],
                                 r["cpu_threads"], r["num_workers"])
                if (rkey == key and r["tier"] == tier and r["load_ok"]
                        and not np.isnan(r["rtf"]) and r["rtf"] < RTF_BUDGET):
                    bucket_tiers.setdefault(r["bucket"], (tier, r["rtf"], r["latency_ms"]))

        lines.append("\nPer-bucket best tier (RTF < 0.85):")
        for b in ["short", "medium", "long", "extended"]:
            if b in bucket_tiers:
                tier, rtf, lat = bucket_tiers[b]
                lines.append(f"  {b:<10} -> {tier:<10} RTF={rtf:.3f}  {lat:.0f} ms")
            else:
                lines.append(f"  {b:<10} -> NONE feasible")

        lines += [
            "",
            "── Paste into config.yaml ──────────────────────────────────────",
            f'force_device: "{d}"',
            f'force_compute_type: "{ct}"',
        ]
        if d == "cpu":
            lines.append(f"force_cpu_threads: {nth}")
        lines.append('accuracy_mode: "auto"')
        overrides = {b: t for b, (t, _, _) in bucket_tiers.items()}
        if overrides:
            lines.append("bucket_accuracy_overrides:")
            for b, t in overrides.items():
                lines.append(f'  {b}: "{t}"')

    lines.append("─" * 60)
    return "\n".join(lines)


# ── Markdown report ────────────────────────────────────────────────────────────
def _build_report(rtf_results, ov_results, stream_results, acc_results, rec) -> str:
    import datetime
    lines = [
        "# Local Config Test Results",
        f"**Date:** {datetime.date.today()}  ",
        f"**Model:** `{MODEL_ID}`  ",
        f"**RTF budget:** {RTF_BUDGET}  ",
        f"**Live p95 budget:** {LIVE_LATENCY_P95_MS} ms  ",
        "",
        "Compute preference: `int8 > int8_float32 > float32`  ",
        "CPU candidates: 3 compute types × 6 thread/worker configs + OMP/MKL env vars  ",
        "CUDA candidates: 3 compute types × 3 cpu_thread counts (pre/post-processing)  ",
        "iGPU: OpenVINO Iris Xe (RTF only)",
        "",
        "---",
        "",
        "## Phase 1 — RTF and Latency",
        "",
        "Format: `✓/✗ RTF (latency ms)`. Cold-start = model load time.",
        "",
        "| Candidate | Cold-start | VRAM | Tier | short | medium | long | extended |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    by_cand = defaultdict(lambda: defaultdict(dict))
    load_times, vram_mbs = {}, {}
    for r in rtf_results:
        k = _cand_key(r["device"], r["compute_type"], r["cpu_threads"], r["num_workers"])
        by_cand[k][r["tier"]][r["bucket"]] = r
        if r["load_ok"] and not np.isnan(r.get("load_time_s", float("nan"))):
            load_times[k] = r["load_time_s"]
            vram_mbs[k]   = r.get("vram_mb", float("nan"))

    def fmt(r):
        if r is None or not r.get("load_ok") or np.isnan(r.get("rtf", float("nan"))):
            return "fail"
        ok = "✓" if r["rtf"] < RTF_BUDGET else "✗"
        return f"{ok}{r['rtf']:.2f}({r['latency_ms']:.0f}ms)"

    for k in sorted(by_cand):
        label = _cand_label(*k)
        lt    = f"{load_times[k]:.1f}s" if k in load_times else "fail"
        vm    = (f"{vram_mbs[k]:.0f}MB" if k in vram_mbs and not np.isnan(vram_mbs[k])
                 else "—")
        for tier in ["fast", "balanced", "accurate"]:
            bdata = by_cand[k].get(tier, {})
            row = f"| {label} | {lt} | {vm} | {tier} "
            for b in ["short", "medium", "long", "extended"]:
                row += f"| {fmt(bdata.get(b))} "
            row += "|"
            lines.append(row)

    if ov_results:
        lines += ["", "**iGPU (OpenVINO) RTF** (fast / beam=1 equivalent):", ""]
        ov_row = "| igpu/int8_ov | "
        lt = f"{ov_results[0].get('load_time_s', float('nan')):.1f}s"
        ov_row += f"{lt} | — | fast "
        for b in ["short", "medium", "long", "extended"]:
            match = next((r for r in ov_results if r["bucket"] == b), None)
            ov_row += f"| {fmt(match)} "
        ov_row += "|"
        lines += [
            "| Candidate | Cold-start | VRAM | Tier | short | medium | long | extended |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ov_row,
        ]

    lines += [
        "",
        "---",
        "",
        "## Phase 1c — Streaming Simulation",
        "",
        f"{STREAM_CHUNKS} × {STREAM_CHUNK_S:.0f}s chunks, model loaded once, VAD off "
        f"(matches `LiveStreamer` behaviour). Live-OK = p95 < {LIVE_LATENCY_P95_MS} ms.",
        "",
    ]
    if stream_results:
        lines += [
            "| Candidate | Median (ms) | p95 (ms) | First chunk (ms) | Live-OK |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in sorted(stream_results, key=lambda x: x["median_ms"]):
            label = _cand_label(r["device"], r["compute_type"],
                                r["cpu_threads"], r["num_workers"])
            ok = "✓ YES" if r["live_ok"] else "✗ NO"
            lines.append(f"| {label} | {r['median_ms']:.0f} | {r['p95_ms']:.0f} "
                         f"| {r['first_chunk_ms']:.0f} | {ok} |")
    else:
        lines.append("_No streaming data._")

    n_acc = sum(r["files"] for r in acc_results[:1]) if acc_results else "?"
    lines += [
        "",
        "---",
        "",
        "## Phase 2 — Accuracy (WER / CER / Hallucinations / Dropout)",
        "",
        f"Sample: audios_1 (all) + audios_2 (<=30 min) + audios_3 (<=30 min) = {n_acc} files. "
        f"Hallucination: >{int(HALLUC_THRESHOLD*100)}% word inflation. "
        f"Dropout: <{int(DROPOUT_THRESHOLD*100)}% of reference words (CoSIH failure mode).",
        "",
    ]
    if acc_results:
        lines += [
            "| Candidate | Tier | VAD | WER | CER | WER-CER | Halluc | Dropout | Files |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in sorted(acc_results, key=lambda x: (x["mean_wer"], not x["vad"])):
            label  = _cand_label(r["device"], r["compute_type"],
                                 r["cpu_threads"], r["num_workers"])
            gap    = r["mean_wer"] - r["mean_cer"]
            vad_s  = "on" if r["vad"] else "**off**"
            lines.append(f"| {label} | {r['tier']} | {vad_s} "
                         f"| {r['mean_wer']:.3f} | {r['mean_cer']:.3f} | {gap:.3f} "
                         f"| {r['hallucination_count']} | {r['dropout_count']} | {r['files']} |")
        lines += [
            "",
            "**VAD off** simulates streaming bucket — LiveStreamer already gates audio externally. "
            "CoSIH sweep: double-VAD costs +0.040 WER on spontaneous speech.",
            "**Dropout high + VAD on** → VAD cutting valid speech.",
        ]
    else:
        lines.append("_No accuracy data._")

    lines += [
        "",
        "---",
        "",
        "## Phase 3 — Recommendation",
        "",
        "```",
        rec.strip(),
        "```",
    ]
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import ctranslate2

    if "--clear" in sys.argv:
        _ckpt_clear()

    print("=" * 70)
    print("Local Config Test — Hebrew STT")
    print(f"Model: {MODEL_ID}")
    print(f"RTF budget: {RTF_BUDGET}  Live p95 budget: {LIVE_LATENCY_P95_MS} ms")
    print("Compute preference: int8 > int8_float32 > float32")
    print("=" * 70)

    # Detect hardware
    cuda_ok = ctranslate2.get_cuda_device_count() > 0
    supported_cpu  = ctranslate2.get_supported_compute_types("cpu")
    supported_cuda = ctranslate2.get_supported_compute_types("cuda") if cuda_ok else set()
    print(f"\nCUDA: {'available' if cuda_ok else 'NOT available'}")
    print(f"  CUDA compute types : {sorted(supported_cuda) if cuda_ok else '—'}")
    print(f"  CPU  compute types : {sorted(supported_cpu)}")
    print(f"  Preferred CPU type : " +
          next((t for t in COMPUTE_PREFERENCE if t in supported_cpu), "float32"))

    # Canonical audio + slices
    canonical = RECORDS_DIR / CANONICAL_NAME
    if not canonical.exists():
        fallback = sorted(RECORDS_DIR.glob("*"), key=lambda p: p.stat().st_size, reverse=True)
        if not fallback:
            sys.exit("[ERROR] No audio in records/ — aborting.")
        canonical = fallback[0]
    print(f"\nCanonical audio: {canonical.name}")
    canon_audio = _load_audio(canonical)
    print(f"  {len(canon_audio)/SAMPLE_RATE:.1f}s")
    slices = _make_slices(canon_audio)

    # Streaming chunks — prefer short audio from audios_2 (sentence-level clips)
    print(f"\nBuilding {STREAM_CHUNKS} streaming chunks ({STREAM_CHUNK_S:.0f}s each)...")
    _sc_adir, _sc_rdir = ROOT / "audios_2", ROOT / "refs_2"
    stream_cands = [p for p in sorted(_sc_adir.glob("*.wav"))
                    if (_sc_rdir / f"{p.stem}.txt").exists()]
    random.seed(RANDOM_SEED + 1)
    random.shuffle(stream_cands)
    stream_chunks = []
    for p in stream_cands:
        if len(stream_chunks) >= STREAM_CHUNKS:
            break
        try:
            a = _load_audio(p)
            n = int(STREAM_CHUNK_S * SAMPLE_RATE)
            stream_chunks.append(a[:n] if len(a) >= n else a)
        except Exception:
            pass
    while len(stream_chunks) < STREAM_CHUNKS:
        n = int(STREAM_CHUNK_S * SAMPLE_RATE)
        offset = len(stream_chunks) * n
        stream_chunks.append(canon_audio[offset:offset + n])
    print(f"  {len(stream_chunks)} chunks ready")

    # WER sample
    print("\nBuilding WER sample (audios_1 all + audios_2 <=30min + audios_3 <=30min)...")
    wer_sample = _build_wer_sample()
    total_wer_s = sum(p["dur"] for p in wer_sample)
    print(f"  {len(wer_sample)} paired files  ({total_wer_s/60:.1f} min total)")

    candidates = _build_candidates()
    print(f"\nTotal faster-whisper candidates: {len(candidates)}")
    for d, ct, nth, nw in candidates:
        print(f"  {_cand_label(d, ct, nth, nw)}")

    # Phase 1
    print("\n" + "=" * 70)
    print("PHASE 1 — RTF + Latency  (CUDA × cpu_threads / CPU × threads × workers)")
    print("=" * 70)
    rtf_results = phase1_rtf(slices)

    # Phase 1b — iGPU
    print("\n" + "=" * 70)
    print("PHASE 1b — iGPU (OpenVINO / Intel Iris Xe)")
    print("=" * 70)
    ov_results = phase1b_openvino(slices)

    # Phase 1c — Streaming
    print("\n" + "=" * 70)
    print("PHASE 1c — Streaming Simulation")
    print("=" * 70)
    stream_results = phase1c_streaming(rtf_results + ov_results, stream_chunks)

    # Phase 2
    acc_results = []
    if wer_sample:
        print("\n" + "=" * 70)
        print("PHASE 2 — WER / CER / Hallucination / Dropout (VAD on + VAD off)")
        print("=" * 70)
        acc_results = phase2_accuracy(rtf_results, wer_sample)
    else:
        print("\n[Phase 2] No paired files — skipping.")

    # Phase 3
    rec = phase3_recommend(rtf_results, ov_results, stream_results, acc_results)
    print(rec)

    report = _build_report(rtf_results, ov_results, stream_results, acc_results, rec)
    RESULTS_FILE.write_text(report, encoding="utf-8")
    print(f"\nReport saved: {RESULTS_FILE}")


if __name__ == "__main__":
    if "--igpu-only" in sys.argv:
        canonical = RECORDS_DIR / CANONICAL_NAME
        if not canonical.exists():
            fallback = sorted(RECORDS_DIR.glob("*"), key=lambda p: p.stat().st_size, reverse=True)
            canonical = fallback[0]
        canon_audio = _load_audio(canonical)
        slices = _make_slices(canon_audio)
        print("\n" + "=" * 70)
        print("PHASE 1b — OpenVINO (iGPU / HETERO / CPU)")
        print("=" * 70)
        ov_results = phase1b_openvino(slices)
        # Append OV results to existing report file
        if ov_results and RESULTS_FILE.exists():
            existing = RESULTS_FILE.read_text(encoding="utf-8")
            ov_lines = ["\n---\n", "\n## Phase 1b — OpenVINO Devices\n",
                        "\n| Device | Bucket | RTF | Latency | Pass |\n",
                        "| --- | --- | --- | --- | --- |\n"]
            for r in ov_results:
                ok = "✓" if not np.isnan(r["rtf"]) and r["rtf"] < RTF_BUDGET else "✗"
                ov_lines.append(f"| {r['device']} | {r['bucket']} | {r['rtf']:.3f} | {r['latency_ms']:.0f}ms | {ok} |\n")
            RESULTS_FILE.write_text(existing + "".join(ov_lines), encoding="utf-8")
            print(f"\nAppended OV results to {RESULTS_FILE}")
    else:
        main()
