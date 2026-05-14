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

    ai_prompt = (
        "I need to install ffmpeg on Windows and add it to PATH so a speech-to-text app "
        "can open MP3, MP4, and M4A audio files. Please walk me through the full installation "
        "step by step, including how to add it to the system PATH."
    )
    _copy_to_clipboard(ai_prompt)

    print(
        "\n[Setup] WARNING: ffmpeg is not installed.\n"
        "         ffmpeg is required to open MP3, MP4, M4A, AAC, and FLAC files.\n"
        "         WAV files work without it.\n"
        "\n"
        "         To install ffmpeg on Windows:\n"
        "           1. Go to: https://www.gyan.dev/ffmpeg/builds/\n"
        "              (direct link: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)\n"
        "           2. Download the ZIP and extract it (e.g. to C:\\ffmpeg)\n"
        "           3. Open Start → search 'environment variables' → Edit the system environment variables\n"
        "           4. Under System Variables → Path → New → paste: C:\\ffmpeg\\bin\n"
        "           5. Click OK, close the window, then restart this terminal\n"
        "\n"
        "         Need help? A step-by-step guide has been copied to your clipboard.\n"
        "         Paste it into ChatGPT, Claude, or any AI chatbot for guided help.\n"
        "\n"
        "         You can continue using the tool with WAV files right now."
    )
    return False


def _copy_to_clipboard(text: str) -> None:
    """Copy text to clipboard. Tries pyperclip first, then Windows clip command."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return
    except Exception:
        pass
    try:
        subprocess.run("clip", input=text, text=True, check=False)
    except Exception:
        pass


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

    print(
        f"[Setup] Downloading model '{model_id}' (~1.5 GB) — this happens once.\n"
        "         Estimated time: 5–20 minutes depending on your connection.\n"
        "         Do not close this window.\n"
    )
    t0 = time.time()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            token=token or None,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        print(f"[Setup] Model downloaded ({time_str})")
        return True
    except Exception as e:
        ai_prompt = (
            f"I'm trying to download the HuggingFace model 'ivrit-ai/whisper-large-v3-turbo-ct2' "
            f"for a speech-to-text app on Windows but it failed with this error: {e}\n"
            "Can you help me fix this or explain how to download the model files manually?"
        )
        _copy_to_clipboard(ai_prompt)

        print(
            f"\n[Setup] Model download failed: {e}\n"
            "\n"
            "         To download manually:\n"
            "           1. Go to: https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2\n"
            "           2. Click the 'Files and versions' tab\n"
            "           3. Download all files into a folder named 'whisper-large-v3-turbo-ct2'\n"
            "              inside your HuggingFace cache (usually C:\\Users\\<you>\\.cache\\huggingface\\hub)\n"
            "\n"
            "         Or set HF_TOKEN in a .env file if the repo requires authentication.\n"
            "\n"
            "         Need help? A message with the error details has been copied to your clipboard.\n"
            "         Paste it into ChatGPT, Claude, or any AI chatbot for guided help.\n"
            "\n"
            "         The model will be downloaded automatically on first transcription if you skip this now."
        )
        return False


def install_requirements():
    """Install base requirements + server extras, then CUDA extras if GPU is available."""
    print("[Setup] Installing base requirements...")
    t0 = time.time()
    _run_with_progress(
        label="Installing base requirements",
        cmd=[sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        estimated_seconds=15
    )
    print(f"[Setup] Base requirements done ({time.time() - t0:.0f}s)")

    print("[Setup] Installing package + server extras (fastapi, uvicorn)...")
    t1 = time.time()
    _run_with_progress(
        label="Installing server extras",
        cmd=[sys.executable, "-m", "pip", "install", "-e", ".[server]", "-q"],
        estimated_seconds=10
    )
    print(f"[Setup] Server extras done ({time.time() - t1:.0f}s)")

    import platform as _platform
    if sys.platform == "darwin" and _platform.machine() == "arm64":
        print("[Setup] Apple Silicon detected — installing mlx-whisper...")
        t1 = time.time()
        _run_with_progress(
            label="Installing mlx-whisper",
            cmd=[sys.executable, "-m", "pip", "install", "mlx-whisper", "-q"],
            estimated_seconds=20
        )
        print(f"[Setup] mlx-whisper done ({time.time() - t1:.0f}s)")
        print("\n[Setup] All requirements satisfied.\n")
        return "apple_silicon"

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
    from visper.benchmark import run_fast_benchmark, run_benchmark
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

    print("\n[Setup] Setup complete.\n")
    if pathlib.Path("start.bat").exists():
        print("  ─────────────────────────────────────────────────")
        print("   Double-click  start.bat  to launch the web UI.")
        print("  ─────────────────────────────────────────────────")
    print()
    print("  CLI commands:")
    print("    visper-file audio.mp3         — transcribe a file")
    print("    visper-live                   — live microphone transcription")
    print("    visper-server                 — start the web UI server")
    print("    visper-benchmark --force      — re-run hardware benchmark")
    print()
