"""
test_openvino.py
----------------
Standalone test script for OpenVINO Whisper on Intel Iris Xe GPU.
Run this BEFORE migrating your main app.

Usage:
    python test_openvino.py

Steps it performs:
    1. Check / install required packages
    2. List available OpenVINO devices
    3. Convert the Whisper model to OpenVINO IR (one-time, cached)
    4. Load pipeline and warm up
    5. Run a short transcription test and measure RTF
    6. Bucket benchmark on school.mp3 (short/medium/long/extended)

No existing project files are touched.
"""

import subprocess
import sys
import time
import os
import shutil
import pathlib
from pathlib import Path

# ── Venv bootstrap ────────────────────────────────────────────────────────────
_VENV_DIR = Path(__file__).parent / ".venv_ov"
_CHILD_ENV = "_TEST_OV_IN_VENV"

def _bootstrap():
    """Create isolated Python 3.12 venv, install deps, re-run inside it, then delete."""
    print("[Setup] Creating isolated Python 3.12 venv for OpenVINO test...")
    subprocess.check_call(["py", "-3.12", "-m", "venv", str(_VENV_DIR)])
    pip    = _VENV_DIR / "Scripts" / "pip"
    python = _VENV_DIR / "Scripts" / "python"
    print("[Setup] Upgrading pip/setuptools/wheel...")
    subprocess.check_call([str(pip), "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    print("[Setup] Installing dependencies...")
    subprocess.check_call([str(pip), "install", "-q",
        "fsspec<=2026.2.0",
        "optimum[openvino,onnx]",
        "optimum-intel[openvino]>=1.25.2",
        "transformers>=4.45.0",
        "librosa",
        "sounddevice",
        "numpy",
    ])
    print("[Setup] Running test in isolated environment...\n")
    env = {**os.environ, _CHILD_ENV: "1"}
    ret = subprocess.run([str(python), str(Path(__file__).resolve())], env=env)
    print("\n[Setup] Removing venv...")
    shutil.rmtree(_VENV_DIR, ignore_errors=True)
    sys.exit(ret.returncode)

# ── IMPORTANT ────────────────────────────────────────────────────────────────
# We use the original OpenAI PyTorch model for conversion.
HF_SOURCE_MODEL = "ivrit-ai/whisper-large-v3-turbo"
OUTPUT_DIR      = "models_ov_test/whisper-large-v3-turbo-ov"
SAMPLE_RATE     = 16_000

# Which Intel GPU to use:
#   "GPU.0" = Intel Iris Xe (iGPU)  <- we want this
#   "GPU.1" = NVIDIA MX350 (dGPU)   <- skip, covered by faster-whisper
PREFERRED_INTEL_GPU = "GPU.0"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 - Install packages if missing
# ─────────────────────────────────────────────────────────────────────────────

def _pip(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages]
    )


def ensure_packages():
    print("\n[1/6] Checking / installing required packages...")

    missing = []
    for pkg, import_name in [
        ("openvino>=2025.0",       "openvino"),
        ("openvino-genai>=2025.0", "openvino_genai"),
        ("optimum-intel>=1.21",    "optimum"),
        ("transformers>=4.45.0",   "transformers"),
        ("librosa",                "librosa"),
        ("numpy",                  "numpy"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Installing: {', '.join(missing)}")
        _pip(*missing, "--upgrade-strategy", "eager")
        print("  Done.")
    else:
        print("  All packages already installed")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 - List OpenVINO devices and pick the right one
# ─────────────────────────────────────────────────────────────────────────────

def check_devices() -> str:
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices

    print("\n[2/6] OpenVINO available devices:")
    for d in devices:
        name = core.get_property(d, "FULL_DEVICE_NAME")
        tag = ""
        if d == PREFERRED_INTEL_GPU:
            tag = "  <- will use this"
        elif d == "GPU.1":
            tag = "  (NVIDIA - skipping, covered by faster-whisper)"
        print(f"  {d:10s} -> {name}{tag}")

    # Prefer GPU.0 (Intel iGPU), fall back to CPU
    if PREFERRED_INTEL_GPU in devices:
        chosen = PREFERRED_INTEL_GPU
        print(f"\n  Selected: {chosen} (Intel Iris Xe)")
    elif "GPU" in devices:
        chosen = "GPU"
        print(f"\n  Selected: GPU")
    else:
        chosen = "CPU"
        print(f"\n  WARNING: No GPU found - falling back to CPU.")
        print("     Check that your Intel Graphics driver is up to date.")

    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 - Convert model (one-time)
# ─────────────────────────────────────────────────────────────────────────────

def convert_model():
    out = pathlib.Path(OUTPUT_DIR)
    if out.exists() and any(out.iterdir()):
        print(f"\n[3/6] Model already converted at '{OUTPUT_DIR}' - skipping.")
        return

    print(f"\n[3/6] Converting model to OpenVINO IR format...")
    print(f"  Source : {HF_SOURCE_MODEL}  (OpenAI PyTorch model)")
    print(f"  Output : {OUTPUT_DIR}")
    print("  This downloads ~3 GB and may take 5-15 minutes. Runs once only.\n")

    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", HF_SOURCE_MODEL,
        "--task",  "automatic-speech-recognition-with-past",
        "--weight-format", "int8",
        OUTPUT_DIR,
    ]

    print(f"  Running: {' '.join(cmd)}\n")
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        print("  (optimum-cli not on PATH - retrying via python -m ...)")
        cmd2 = [
            sys.executable, "-m", "optimum.exporters.openvino",
            "--model", HF_SOURCE_MODEL,
            "--task",  "automatic-speech-recognition-with-past",
            OUTPUT_DIR,
        ]
        subprocess.check_call(cmd2)

    print(f"\n  Model saved to '{OUTPUT_DIR}'")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 - Load pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(device: str):
    import openvino_genai as ov_genai

    print(f"\n[4/6] Loading WhisperPipeline on {device}...")
    print("  (First load on GPU may take 30-60s while OpenVINO compiles kernels)")
    t0 = time.time()
    pipe = ov_genai.WhisperPipeline(OUTPUT_DIR, device)
    elapsed = time.time() - t0
    print(f"  Pipeline loaded in {elapsed:.1f}s")

    cfg = pipe.get_generation_config()
    cfg.language          = "<|he|>"
    cfg.task              = "transcribe"
    cfg.return_timestamps = False

    return pipe, cfg


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Step 5 - Bucket benchmark on school.mp3
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_DURATIONS = {"short": 5, "medium": 20, "long": 45, "extended": 90}


def benchmark_buckets(pipe, cfg, device: str):
    """Transcribe school.mp3 slices for each bucket and report RTF."""
    import numpy as np

    audio_path = pathlib.Path("records/school.mp3")
    if not audio_path.exists():
        print(f"\n[5/5] Skipping bucket benchmark — {audio_path} not found.")
        return

    print(f"\n[5/5] Bucket benchmark on {audio_path}...")
    print(f"  Device : {device}")

    # Load via librosa to ensure 16kHz mono float32
    import librosa
    full_audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    total_dur = len(full_audio) / SAMPLE_RATE
    print(f"  Total audio: {total_dur:.1f}s\n")

    # Single warm-up pass
    warmup = full_audio[:SAMPLE_RATE * 2].tolist()
    pipe.generate(warmup, cfg)

    for bucket, target_dur in BUCKET_DURATIONS.items():
        if target_dur > total_dur:
            print(f"  {bucket:<10s} ({target_dur:3d}s) — skipped (audio too short)")
            continue
        audio_slice = full_audio[:int(target_dur * SAMPLE_RATE)].tolist()
        t0 = time.time()
        pipe.generate(audio_slice, cfg)
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
    print("  OpenVINO Whisper Test - Intel Iris Xe (GPU.0)")
    print("=" * 60)

    ensure_packages()
    device = check_devices()
    convert_model()
    pipe, cfg = load_pipeline(device)
    benchmark_buckets(pipe, cfg, device)

    # Clean up IGC kernel error log generated during GPU shader compilation
    kernel_errors = pathlib.Path("kernel.errors.txt")
    if kernel_errors.exists():
        kernel_errors.unlink()

    print("\n" + "=" * 60)
    print("  Test complete.")
    if device == PREFERRED_INTEL_GPU:
        print("  Intel Iris Xe GPU is working with OpenVINO.")
    else:
        print("  WARNING: Ran on CPU - Intel GPU was not available.")
        print("  Update your Intel Graphics driver and re-run.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if not os.environ.get(_CHILD_ENV):
        _bootstrap()
    main()