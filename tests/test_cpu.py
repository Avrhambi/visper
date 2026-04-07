"""
test_cpu.py
-----------
Standalone test script for CPU-only Whisper transcription.
Optimised for Intel Core i5-1135G7 (4 cores / 8 threads, AVX2 + AVX-512).

Run this BEFORE migrating your main app.

Usage:
    python test_cpu.py

Steps it performs:
    1. Check / install required packages
    2. Detect CPU capabilities (AVX2, AVX-512, core count)
    3. Benchmark multiple thread configurations to find the optimal one
    4. Load the faster-whisper model with the best config
    5. Run a short transcription test and measure RTF
    6. Bucket benchmark on school.mp3 (short/medium/long/extended)

No existing project files are touched.
"""

import subprocess
import sys
import os
import platform
import shutil
import time
from pathlib import Path

# ── Venv bootstrap ────────────────────────────────────────────────────────────
_VENV_DIR = Path(__file__).parent / ".venv_cpu"
_CHILD_ENV = "_TEST_CPU_IN_VENV"

def _bootstrap():
    """Create isolated venv, install deps, re-run inside it, then delete."""
    print("[Setup] Creating isolated venv for CPU test...")
    subprocess.check_call([sys.executable, "-m", "venv", str(_VENV_DIR)])
    pip    = _VENV_DIR / "Scripts" / "pip"
    python = _VENV_DIR / "Scripts" / "python"
    print("[Setup] Installing dependencies...")
    subprocess.check_call([str(pip), "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    subprocess.check_call([str(pip), "install", "-q", "faster-whisper", "numpy"])
    print("[Setup] Running test in isolated environment...\n")
    env = {**os.environ, _CHILD_ENV: "1"}
    ret = subprocess.run([str(python), str(Path(__file__).resolve())], env=env)
    print("\n[Setup] Removing venv...")
    shutil.rmtree(_VENV_DIR, ignore_errors=True)
    sys.exit(ret.returncode)

# ── IMPORTANT ────────────────────────────────────────────────────────────────
# We reuse the same CT2 model already used by the main app (no re-download).
# If you have not used the main app yet, it will download on first run (~1.5 GB).
HF_MODEL_ID  = "ivrit-ai/whisper-large-v3-turbo-ct2"
SAMPLE_RATE  = 16_000

# ── Thread configurations to benchmark ───────────────────────────────────────
# i5-1135G7 has 4P-cores / 8 logical threads.
# We test a range. Usually 4 or 6 threads wins for inference latency.
THREAD_CONFIGS = [
    {"cpu_threads": 2, "num_workers": 1, "label": "2 threads / 1 worker"},
    {"cpu_threads": 4, "num_workers": 1, "label": "4 threads / 1 worker"},
    {"cpu_threads": 4, "num_workers": 2, "label": "4 threads / 2 workers"},
    {"cpu_threads": 6, "num_workers": 1, "label": "6 threads / 1 worker"},
    {"cpu_threads": 8, "num_workers": 1, "label": "8 threads / 1 worker"},
    {"cpu_threads": 8, "num_workers": 2, "label": "8 threads / 2 workers"},
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 - Install packages if missing
# ─────────────────────────────────────────────────────────────────────────────

def _pip(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages]
    )


def ensure_packages():
    print("\n[1/6] Checking / installing required packages...")

    required = [
        ("faster-whisper",  "faster_whisper"),
        ("numpy",           "numpy"),
    ]

    missing = []
    for pkg, import_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Installing: {', '.join(missing)}")
        _pip(*missing)
        print("  Done.")
    else:
        print("  All packages already installed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 - Detect CPU capabilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_cpu_flags() -> set:
    """Read CPU feature flags from /proc/cpuinfo (Linux) or use CPUID on Windows."""
    flags = set()
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags = set(line.split(":")[1].split())
                        break
        elif platform.system() == "Windows":
            # cpuinfo module is optional – skip gracefully
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                flags = set(info.get("flags", []))
            except ImportError:
                pass
    except Exception:
        pass
    return flags


def check_cpu() -> str:
    """Print CPU info and return best compute_type."""
    import multiprocessing
    logical  = multiprocessing.cpu_count()

    flags = _get_cpu_flags()
    has_avx512 = "avx512f" in flags
    has_avx2   = "avx2"    in flags

    print("\n[2/6] CPU Capabilities:")
    print(f"  Processor      : {platform.processor() or 'Unknown'}")
    print(f"  Logical cores  : {logical}")
    print(f"  AVX2           : {'YES' if has_avx2   else 'no'}")
    print(f"  AVX-512        : {'YES' if has_avx512 else 'no'}")

    # Determine best CTranslate2 compute type for this CPU
    import ctranslate2
    supported = ctranslate2.get_supported_compute_types("cpu")
    print(f"  CT2 supported  : {', '.join(sorted(supported))}")

    # Preference: int8 > int8_float32 > float32
    if "int8" in supported:
        compute_type = "int8"
    elif "int8_float32" in supported:
        compute_type = "int8_float32"
    else:
        compute_type = "float32"

    print(f"\n  Selected compute_type: {compute_type}")
    if compute_type == "int8":
        print("  INT8 is available – expect best speed on your i5-1135G7.")
    elif compute_type == "float32":
        print("  WARNING: INT8 not supported. Inference will be slower.")

    return compute_type


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 - Benchmark thread configurations
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_threads(compute_type: str) -> dict:
    """
    Run a timed transcription with several thread configs on a local file.
    Returns the config dict that gave the lowest RTF.
    """
    from faster_whisper import WhisperModel, decode_audio
    audio_path = "records/school.mp3"

    if not os.path.exists(audio_path):
        print(f"Error: Could not find audio file at {audio_path}")
        return THREAD_CONFIGS[1]  # Fallback

    print(f"\n[3/6] Benchmarking {len(THREAD_CONFIGS)} thread configurations...")
    print(f"  Model : {HF_MODEL_ID}")
    print(f"  Audio : {audio_path}")

    audio = decode_audio(audio_path, sampling_rate=16000)
    duration = len(audio) / SAMPLE_RATE
    # Use a 45s slice to keep each candidate run consistent and fast
    slice_samples = int(min(45.0, duration) * SAMPLE_RATE)
    audio = audio[:slice_samples]
    duration = len(audio) / SAMPLE_RATE
    print(f"  Using first {duration:.1f}s of audio")

    results = []

    for cfg in THREAD_CONFIGS:
        label       = cfg["label"]
        cpu_threads = cfg["cpu_threads"]
        num_workers = cfg["num_workers"]

        # Set environment variables for the math libraries (MKL/OpenMP)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

        try:
            model = WhisperModel(
                HF_MODEL_ID,
                device="cpu",
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )

            # Warm-up pass (ensures model is fully in memory/initialized)
            # We use a small slice of the audio for a quick warm-up
            model.transcribe(audio[:16000*5], language=LANGUAGE) 

            # Timed pass on the 45s slice
            t0 = time.time()
            segments, _ = model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                without_timestamps=WITHOUT_TIMESTAMPS,
                vad_filter=VAD_FILTER,
            )
            list(segments)  # Force execution of the generator
            elapsed = time.time() - t0
            rtf = elapsed / duration

            status_tag = ""
            if rtf < 0.2: status_tag = "  [BLAZING FAST]"
            elif rtf < 0.5: status_tag = "  [EXCELLENT]"
            elif rtf < 1.0: status_tag = "  [GOOD]"

            print(f"  {label:<35s}  RTF: {rtf:.3f}  ({elapsed:.2f}s){status_tag}")
            results.append({"cfg": cfg, "rtf": rtf, "elapsed": elapsed})
            del model

        except Exception as e:
            print(f"  {label:<35s}  ERROR: {e}")

    if not results:
        best_cfg = THREAD_CONFIGS[1]
    else:
        best = min(results, key=lambda r: r["rtf"])
        best_cfg = best["cfg"]
        print(f"\n  Winning Config: {best_cfg['label']} (RTF {best['rtf']:.3f})")

    # Load the winner once — reused for all remaining steps
    cpu_threads = best_cfg["cpu_threads"]
    num_workers = best_cfg["num_workers"]
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    print(f"\n[4/6] Loading model with winning config (kept for all remaining steps)...")
    print(f"  cpu_threads : {cpu_threads}  num_workers : {num_workers}  compute : {compute_type}")
    t0 = time.time()
    model = WhisperModel(
        HF_MODEL_ID, device="cpu", compute_type=compute_type,
        cpu_threads=cpu_threads, num_workers=num_workers,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return best_cfg, model


# ─────────────────────────────────────────────────────────────────────────────
def _transcribe(model, audio):
    """Helper: run transcription and return joined text."""
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


def benchmark_buckets(model, compute_type: str, best_cfg: dict):
    """Transcribe school.mp3 slices for each bucket and report RTF."""
    from faster_whisper import decode_audio
    audio_path = "records/school.mp3"
    if not os.path.exists(audio_path):
        print(f"\n[5/5] Skipping bucket benchmark — {audio_path} not found.")
        return

    print(f"\n[5/5] Bucket benchmark on {audio_path}...")
    print(f"  Config : {best_cfg['label']}  compute={compute_type}")

    full_audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
    total_dur = len(full_audio) / SAMPLE_RATE
    print(f"  Total audio: {total_dur:.1f}s\n")

    # Single warm-up pass with a 2s slice
    warmup = full_audio[:SAMPLE_RATE * 2]
    _transcribe(model, warmup)

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
    print("  CPU-Only Whisper Test - Intel i5-1135G7")
    print("=" * 60)

    ensure_packages()
    compute_type    = check_cpu()
    best_cfg, model = benchmark_threads(compute_type)
    benchmark_buckets(model, compute_type, best_cfg)

    print("\n" + "=" * 60)
    print("  Test complete.")
    print(f"  Best config: {best_cfg['label']}  compute={compute_type}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if not os.environ.get(_CHILD_ENV):
        _bootstrap()
    main()