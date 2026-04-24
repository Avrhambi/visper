# install.py
import subprocess
import sys
import os
import pathlib
import shutil
import site
import time
import threading
from tqdm import tqdm


def _run_with_progress(label: str, cmd: list, estimated_seconds: int = 10):
    """Run a subprocess while showing a tqdm progress bar."""
    bar = tqdm(total=100, desc=label, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s", ncols=70)
    done = threading.Event()

    def _fill():
        steps = estimated_seconds * 10  # update every 0.1s
        for _ in range(steps):
            if done.is_set():
                break
            time.sleep(0.1)
            bar.update(100 // steps)

    filler = threading.Thread(target=_fill, daemon=True)
    filler.start()

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        done.set()
        filler.join()
        bar.n = 100
        bar.refresh()
        bar.close()


def _register_cuda_dlls():
    """Add nvidia DLL folders to PATH."""
    for sp in site.getsitepackages():
        nvidia_path = pathlib.Path(sp) / "nvidia"
        if nvidia_path.exists():
            for dll_dir in nvidia_path.rglob("*.dll"):
                folder = str(dll_dir.parent)
                if folder not in os.environ["PATH"]:
                    os.environ["PATH"] += f";{folder}"
    print("[Setup] CUDA DLL paths registered.")


def check_and_fix_cuda():
    """Detect GPU tier, register DLLs, and return tier string."""
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")

        if not supported or supported == {"float32"}:
            print("[Setup] No CUDA GPU detected, using CPU.")
            return "cpu"

        if "float16" in supported or "int8_float16" in supported:
            print(f"[Setup] Dedicated GPU detected. Compute types: {supported}")
            _register_cuda_dlls()
            return "dedicated_gpu"

        if "int8_float32" in supported:
            print(f"[Setup] Entry-level GPU detected. Compute types: {supported}")
            _register_cuda_dlls()
            return "entry_gpu"

        print("[Setup] Unknown GPU config, falling back to CPU.")
        return "cpu"

    except Exception as e:
        print(f"[Setup] CUDA check failed ({e}), falling back to CPU.")
        return "cpu"


def check_ffmpeg():
    """Warn if ffmpeg is not on PATH. Does not block setup."""
    if shutil.which("ffmpeg"):
        print("[Setup] ffmpeg found.")
        return True
    print(
        "[Setup] WARNING: ffmpeg not found on PATH.\n"
        "         MP3/MP4/M4A files require ffmpeg to decode.\n"
        "         Install from https://ffmpeg.org and add it to PATH.\n"
        "         WAV files work without it."
    )
    return False


def _load_hf_token() -> str:
    """Read HF_TOKEN from environment or .env file."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            env_file = pathlib.Path(".env")
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
        except Exception:
            pass
    return token


def download_model():
    """
    Download the model to HuggingFace cache during setup so first transcription is instant.
    Skips silently if already cached.
    """
    model_id = "ivrit-ai/whisper-large-v3-turbo-ct2"
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_id, "config.json")
        if cached is not None:
            print(f"[Setup] Model already in cache — skipping download.")
            return True
    except Exception:
        pass

    token = _load_hf_token()
    if not token:
        print("[Setup] No HF_TOKEN found. If the model repo is gated, set HF_TOKEN in .env.")

    print(f"[Setup] Downloading model '{model_id}' (~1.5 GB) — this happens once...")
    t0 = time.time()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            token=token or None,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        elapsed = time.time() - t0
        print(f"[Setup] Model downloaded ({elapsed:.0f}s)")
        return True
    except Exception as e:
        print(
            f"[Setup] Model download failed: {e}\n"
            "         The model will be downloaded on first transcription instead."
        )
        return False


def install_requirements():
    """Install base requirements, then CUDA extras if GPU is available."""
    print("[Setup] Installing base requirements...")
    t0 = time.time()
    _run_with_progress(
        label="Installing base requirements",
        cmd=[sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        estimated_seconds=15
    )
    print(f"[Setup] Base requirements done ({time.time() - t0:.0f}s)")

    print("\n[Setup] Detecting GPU...")
    tier = check_and_fix_cuda()

    if tier in ("dedicated_gpu", "entry_gpu"):
        print("[Setup] Installing CUDA libraries...")
        t1 = time.time()
        _run_with_progress(
            label="Installing CUDA libraries",
            cmd=[
                sys.executable, "-m", "pip", "install",
                "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "-q"
            ],
            estimated_seconds=30
        )
        _register_cuda_dlls()
        print(f"[Setup] CUDA libraries done ({time.time() - t1:.0f}s)")

    print("\n[Setup] All requirements satisfied.\n")
    return tier


def run_benchmark_with_timeout(timeout_seconds: int = 180):
    """Run fast benchmark with a timeout. Falls back to --quick mode if it takes too long."""
    from core.benchmark import run_fast_benchmark, run_benchmark
    result = [None]
    error = [None]

    def _worker():
        try:
            run_fast_benchmark()
            result[0] = "done"
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t0 = time.time()
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        print(
            f"\n[Setup] Benchmark taking longer than expected ({timeout_seconds}s). "
            "Check GPU driver status.",
            flush=True,
        )
        print("[Setup] Falling back to quick mode (no RTF probe)...")
        try:
            run_benchmark(quick=True)
        except Exception as e:
            print(f"[Setup] Quick benchmark also failed: {e}")
    elif error[0]:
        print(f"[Setup] Benchmark error: {error[0]}")
        print("[Setup] Falling back to quick mode...")
        try:
            run_benchmark(quick=True)
        except Exception as e:
            print(f"[Setup] Quick benchmark also failed: {e}")
    else:
        elapsed = time.time() - t0
        print(f"[Setup] Benchmark complete ({elapsed:.0f}s)")


if __name__ == "__main__":
    print("=" * 50)
    print("        STT Engine — Setup")
    print("=" * 50 + "\n")

    # Step 1: Install dependencies
    install_requirements()

    # Step 2: ffmpeg check (non-blocking)
    print("[Setup] Checking ffmpeg...")
    check_ffmpeg()

    # Step 3: Download model into HF cache (skips if already present)
    print("[Setup] Checking model cache...")
    download_model()

    # Step 4: Benchmark if needed
    results_path = pathlib.Path("benchmark_results.json")
    if not results_path.exists():
        print("\n[Setup] No benchmark results found.")
        print("[Setup] Running fast benchmark: rules + primary RTF probe (~60s)...\n")
        run_benchmark_with_timeout(timeout_seconds=180)
    else:
        print("[Setup] Benchmark results found — skipping benchmark.\n")

    print("\n[Setup] Done.\n")
    print("  Offline transcription:   stt-file audio.mp3          (or: python transcribe_file.py audio.mp3)")
    print("  Live/streaming:          stt-live                    (or: python transcribe_live.py)")
    print("  Re-benchmark:            stt-benchmark --force       (or: python run_benchmark.py --force)")
    print("  FastAPI server:          stt-server                  (requires: pip install -e \".[server]\")")
    print()
