"""
test_gpu.py
-----------
Standalone test script for CUDA Whisper on NVIDIA GeForce MX350 (2 GB VRAM).
Run this BEFORE migrating your main app.

Usage:
    python test_gpu.py

Steps it performs:
    1. Check / install required packages + CUDA libraries
    2. Detect GPU and validate CUDA compute types
    3. Benchmark compute_type options to find the optimal one for MX350
    4. Load the faster-whisper model with the best config
    5. Run a short transcription test and measure RTF
    6. Bucket benchmark on school.mp3 (short/medium/long/extended)

No existing project files are touched.
"""

import subprocess
import sys
import time
import os
import pathlib
import shutil
import site
from pathlib import Path

# ── Venv bootstrap ────────────────────────────────────────────────────────────
_VENV_DIR = Path(__file__).parent / ".venv_gpu"
_CHILD_ENV = "_TEST_GPU_IN_VENV"

def _bootstrap():
    """Create isolated venv, install deps, re-run inside it, then delete."""
    print("[Setup] Creating isolated venv for GPU test...")
    subprocess.check_call([sys.executable, "-m", "venv", str(_VENV_DIR)])
    pip    = _VENV_DIR / "Scripts" / "pip"
    python = _VENV_DIR / "Scripts" / "python"
    print("[Setup] Installing dependencies...")
    subprocess.check_call([str(pip), "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([str(pip), "install", "-q",
        "faster-whisper", "numpy",
        "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
    ])
    print("[Setup] Running test in isolated environment...\n")
    env = {**os.environ, _CHILD_ENV: "1"}
    ret = subprocess.run([str(python), str(Path(__file__).resolve())], env=env)
    print("\n[Setup] Removing venv...")
    shutil.rmtree(_VENV_DIR, ignore_errors=True)
    sys.exit(ret.returncode)

# ── IMPORTANT ────────────────────────────────────────────────────────────────
# Same CT2 model used by the main app. Already cached after first run.
HF_MODEL_ID  = "ivrit-ai/whisper-large-v3-turbo-ct2"
SAMPLE_RATE  = 16_000

# ── Compute type configurations to benchmark ──────────────────────────────────
# MX350 is an entry-level Pascal/Turing GPU with 2 GB VRAM.
# It does NOT support float16 natively (no tensor cores).
# int8_float32 is typically the sweet spot — quantized weights, fp32 compute.
# We test all viable options and let the numbers decide.
COMPUTE_CONFIGS = [
    {"compute_type": "int8_float32", "label": "int8_float32  (recommended for MX350)"},
    {"compute_type": "float32",      "label": "float32       (baseline, no quantization)"},
]

# ── Transcription settings (mirrors core/constants.py) ────────────────────────
LANGUAGE                = "he"
BEAM_SIZE               = 1
TEMPERATURE             = 0.0
CONDITION_ON_PREV_TEXT  = False
WITHOUT_TIMESTAMPS      = True
VAD_FILTER              = True
VAD_MIN_SILENCE_MS      = 300
VAD_SPEECH_PAD_MS       = 200

# CPU fallback threads (used alongside CUDA for pre/post processing)
CPU_THREADS = 4
NUM_WORKERS = 1


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 - Install packages if missing
# ─────────────────────────────────────────────────────────────────────────────

def _pip(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages]
    )


def _register_cuda_dlls():
    """Add nvidia DLL folders to PATH so CTranslate2 can find cublas/cudnn."""
    registered = []
    for sp in site.getsitepackages():
        nvidia_path = pathlib.Path(sp) / "nvidia"
        if nvidia_path.exists():
            for dll_dir in nvidia_path.rglob("*.dll"):
                folder = str(dll_dir.parent)
                if folder not in os.environ["PATH"]:
                    os.environ["PATH"] += f";{folder}"
                    registered.append(folder)
    if registered:
        print(f"  Registered {len(registered)} CUDA DLL path(s).")
    else:
        print("  No nvidia DLL folders found (may already be on PATH).")


def ensure_packages():
    print("\n[1/6] Checking / installing required packages...")

    base_required = [
        ("faster-whisper", "faster_whisper"),
        ("numpy",          "numpy"),
    ]

    missing = []
    for pkg, import_name in base_required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Installing base: {', '.join(missing)}")
        _pip(*missing)

    # CUDA libraries — nvidia-cublas + nvidia-cudnn are needed by faster-whisper on Windows
    cuda_pkgs = [
        ("nvidia-cublas-cu12", "nvidia.cublas"),
        ("nvidia-cudnn-cu12",  "nvidia.cudnn"),
    ]
    cuda_missing = []
    for pkg, import_name in cuda_pkgs:
        try:
            __import__(import_name)
        except (ImportError, ModuleNotFoundError):
            cuda_missing.append(pkg)

    if cuda_missing:
        print(f"  Installing CUDA libraries: {', '.join(cuda_missing)}")
        _pip(*cuda_missing)
        print("  Done.")
    else:
        print("  All packages already installed.")

    _register_cuda_dlls()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 - Detect GPU and validate CUDA
# ─────────────────────────────────────────────────────────────────────────────

def check_gpu() -> list:
    """Print GPU info and return list of supported compute types."""
    import ctranslate2

    print("\n[2/6] GPU / CUDA Detection:")

    # Try to get GPU name via nvidia-smi
    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            text=True
        ).strip()
        print(f"  GPU            : {name}")
    except Exception:
        print("  GPU            : (nvidia-smi not found — check driver)")

    supported = ctranslate2.get_supported_compute_types("cuda")
    print(f"  CT2 CUDA types : {', '.join(sorted(supported))}")

    if not supported or supported == {"float32"}:
        print("\n  WARNING: CUDA not properly available to CTranslate2.")
        print("  Possible causes:")
        print("    - nvidia-cublas-cu12 / nvidia-cudnn-cu12 not installed")
        print("    - CUDA DLLs not on PATH")
        print("    - Driver mismatch")
        print("  Try re-running — Step 1 will attempt to fix DLL paths.")
        sys.exit(1)

    print()
    if "int8_float32" in supported:
        print("  int8_float32 available  (optimal for MX350)")
    if "float16" in supported:
        print("  float16 available  (tensor cores detected — better than expected!)")
    if "int8_float16" in supported:
        print("  int8_float16 available")

    return list(supported)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 - Benchmark compute_type options
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_compute(supported_types: list) -> dict:
    """
    Run a timed transcription with each viable compute_type on 5 s of audio.
    Returns the config dict with the lowest RTF.
    """
    import numpy as np
    from faster_whisper import WhisperModel

    # Filter configs to only those supported by this GPU
    configs_to_test = [
        c for c in COMPUTE_CONFIGS
        if c["compute_type"] in supported_types
    ]

    # Also test int8_float16 if available (unexpected bonus on MX350)
    if "int8_float16" in supported_types:
        configs_to_test.insert(0, {
            "compute_type": "int8_float16",
            "label": "int8_float16  (tensor cores — best if available)"
        })

    print(f"\n[3/6] Benchmarking {len(configs_to_test)} compute_type(s)...")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Audio : 5 s sine tone\n")

    duration = 5.0
    t_arr = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype="float32")
    audio = (0.05 * np.sin(2 * 3.14159 * 220 * t_arr))

    results = []

    for cfg in configs_to_test:
        ct    = cfg["compute_type"]
        label = cfg["label"]

        try:
            model = WhisperModel(
                HF_MODEL_ID,
                device="cuda",
                compute_type=ct,
                cpu_threads=CPU_THREADS,
                num_workers=NUM_WORKERS,
            )

            # Warm-up pass (CUDA kernel compilation + memory transfer)
            print(f"  Warming up [{label}]...")
            segs, _ = model.transcribe(audio, language=LANGUAGE, beam_size=BEAM_SIZE,
                                       without_timestamps=WITHOUT_TIMESTAMPS,
                                       vad_filter=VAD_FILTER)
            list(segs)

            # Timed pass
            t0 = time.time()
            segs, _ = model.transcribe(audio, language=LANGUAGE, beam_size=BEAM_SIZE,
                                       without_timestamps=WITHOUT_TIMESTAMPS,
                                       vad_filter=VAD_FILTER)
            list(segs)
            elapsed = time.time() - t0
            rtf = elapsed / duration

            tag = ""
            if rtf < 0.3:
                tag = "  [EXCELLENT]"
            elif rtf < 0.6:
                tag = "  [GREAT]"
            elif rtf < 1.0:
                tag = "  [GOOD]"
            else:
                tag = "  [SLOW]"

            print(f"  {label:<50s}  RTF: {rtf:.3f}{tag}")
            results.append({"cfg": cfg, "rtf": rtf, "elapsed": elapsed})
            del model

        except Exception as e:
            print(f"  {label:<50s}  ERROR: {e}")

    if not results:
        print("  All configs failed — check CUDA setup.")
        sys.exit(1)

    best = min(results, key=lambda r: r["rtf"])
    best_cfg = best["cfg"]
    print(f"\n  Winning Config: {best_cfg['label']}  (RTF {best['rtf']:.3f})")

    # Load the winner once — reused for all remaining steps
    print(f"\n[4/6] Loading model with winning config (kept for all remaining steps)...")
    print(f"  device : cuda  compute_type : {best_cfg['compute_type']}")
    t0 = time.time()
    model = WhisperModel(
        HF_MODEL_ID, device="cuda", compute_type=best_cfg["compute_type"],
        cpu_threads=CPU_THREADS, num_workers=NUM_WORKERS,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return best_cfg, model


# ─────────────────────────────────────────────────────────────────────────────
def _transcribe(model, audio):
    segs, _ = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURE,
        condition_on_previous_text=CONDITION_ON_PREV_TEXT,
        without_timestamps=WITHOUT_TIMESTAMPS,
        vad_filter=VAD_FILTER,
        vad_parameters=dict(
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        ),
    )
    return "".join(seg.text for seg in segs).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 - Bucket benchmark on school.mp3
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_DURATIONS = {"short": 5, "medium": 20, "long": 45, "extended": 90}


def benchmark_buckets(model, best_cfg: dict):
    """Transcribe school.mp3 slices for each bucket and report RTF."""
    import numpy as np
    from faster_whisper import decode_audio

    audio_path = "records/school.mp3"
    if not os.path.exists(audio_path):
        print(f"\n[5/5] Skipping bucket benchmark — {audio_path} not found.")
        return

    print(f"\n[5/5] Bucket benchmark on {audio_path}...")
    print(f"  Config : CUDA {best_cfg['compute_type']}")

    full_audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    total_dur = len(full_audio) / SAMPLE_RATE
    print(f"  Total audio: {total_dur:.1f}s\n")

    # Single warm-up pass
    _transcribe(model, full_audio[:SAMPLE_RATE * 2])

    for bucket, target_dur in BUCKET_DURATIONS.items():
        if target_dur > total_dur:
            print(f"  {bucket:<10s} ({target_dur:3d}s) — skipped (audio too short)")
            continue
        audio_slice = full_audio[:int(target_dur * SAMPLE_RATE)]
        t0 = time.time()
        _transcribe(model, audio_slice)
        elapsed = time.time() - t0
        rtf = elapsed / target_dur
        tag = "OK" if rtf < 1.0 else "SLOW"
        print(f"  {bucket:<10s} ({target_dur:3d}s) — RTF {rtf:.3f}  [{tag}]")
        if rtf > 1.5 and target_dur >= 20:
            print("  RTF > 1.5 on longer bucket — skipping remaining.")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CUDA Whisper Test - NVIDIA GeForce MX350 (2 GB)")
    print("=" * 60)

    ensure_packages()
    supported        = check_gpu()
    best_cfg, model  = benchmark_compute(supported)
    benchmark_buckets(model, best_cfg)

    print("\n" + "=" * 60)
    print("  Test complete.")
    print(f"  Best config: CUDA {best_cfg['compute_type']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if not os.environ.get(_CHILD_ENV):
        _bootstrap()
    main()