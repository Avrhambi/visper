# setup.py
import subprocess
import sys
import os
import pathlib
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


def install_requirements():
    """Install base requirements, then CUDA extras if GPU is available."""

    # Step 1: Base requirements
    _run_with_progress(
        label="📦 Installing base requirements",
        cmd=[sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        estimated_seconds=15
    )

    # Step 2: GPU detection (fast, no bar needed)
    print("\n🔍 Detecting GPU...")
    tier = check_and_fix_cuda()

    # Step 3: CUDA libraries (only if GPU found)
    if tier in ("dedicated_gpu", "entry_gpu"):
        _run_with_progress(
            label="⚡ Installing CUDA libraries ",
            cmd=[
                sys.executable, "-m", "pip", "install",
                "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "-q"
            ],
            estimated_seconds=30
        )
        _register_cuda_dlls()

    print("\n✅ All requirements satisfied.\n")
    return tier


if __name__ == "__main__":
    import pathlib

    print("=" * 50)
    print("        STT Engine — Setup")
    print("=" * 50 + "\n")

    install_requirements()

    # Run benchmark if no results exist yet
    results_path = pathlib.Path("benchmark_results.json")
    if not results_path.exists():
        print("[Setup] No benchmark results found.")
        print("[Setup] Detecting hardware configuration (quick mode — no inference)...\n")
        from core.benchmark import run_benchmark
        run_benchmark(quick=True)
    else:
        print("[Setup] Benchmark results found — skipping benchmark.\n")

    print("[Setup] Done.\n")
    print("  Offline transcription:   python transcribe_file.py audio.mp3")
    print("  Live/streaming:          python transcribe_live.py")
    print("  Re-benchmark:            python run_benchmark.py --force")
    print()
    print("  Or after pip install -e .:")
    print("    from stt_he import transcribe")
    print("    result = transcribe('audio.wav')")
    print()
    print("    from stt_he import stream_transcribe")
    print("    stream_transcribe(lambda text, final: print(text))")