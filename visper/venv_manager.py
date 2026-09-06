"""
visper/venv_manager.py
--------------------
Manages isolated Python virtual environments for each device backend.

Device venvs live under:  <repo_root>/.venvs/{device}/
  .venvs/cpu/       — faster-whisper CPU backend
  .venvs/cuda/      — faster-whisper CUDA backend + NVIDIA libraries
  .venvs/openvino/  — openvino_genai backend (requires Python 3.12)

Public API
----------
  ensure_venv(device)   — create venv if not already present
  create_venv(device)   — (re)create venv and install packages
  delete_venv(device)   — remove venv directory
  venv_exists(device)   — True if venv has a working Python executable
  venv_path(device)     — absolute Path to the venv root
  python_exe(device)    — absolute Path to python inside the venv
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
VENVS_DIR = ROOT / ".venvs"

# Written only after every package install succeeds. venv_exists() checks for it
# so a venv left half-built by a failed install is treated as absent (and rebuilt)
# rather than "reused" and then failing to import faster_whisper at runtime.
_READY_MARKER = ".visper-ready"

_BASE = ["numpy", "soundfile"]

DEVICE_PACKAGES: dict[str, list[str]] = {
    "cpu": [
        *_BASE,
        "faster-whisper",
    ],
    "cuda": [
        *_BASE,
        "faster-whisper",
        "nvidia-cublas-cu12",
        "nvidia-cudnn-cu12",
    ],
    "openvino": [
        *_BASE,
        "fsspec<=2026.2.0",
        "optimum[openvino,onnx]",
        "optimum-intel[openvino]>=1.25.2",
        "transformers>=4.45.0",
        "librosa",
    ],
}

# The native-wheel stack (ctranslate2, onnxruntime, openvino) lags the newest
# CPython by a release or two, so a device venv built with e.g. 3.14 can fail
# `pip install faster-whisper`. Prefer 3.12 when the host has it; fall back to
# whatever is running this process. OpenVINO *requires* 3.12.
def _preferred_python(min_ok: tuple = (3, 10), max_ok: tuple = (3, 12)) -> list[str]:
    running = sys.version_info[:2]
    if min_ok <= running <= max_ok:
        return [sys.executable]
    for ver in ("3.12", "3.11", "3.10"):
        if shutil.which("py"):
            try:
                subprocess.check_output(["py", f"-{ver}", "-c", "import sys"],
                                        stderr=subprocess.DEVNULL)
                return ["py", f"-{ver}"]
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
    return [sys.executable]


_DEVICE_PYTHON_CMD: dict[str, list[str]] = {
    "cpu":      _preferred_python(),
    "cuda":     _preferred_python(),
    "openvino": ["py", "-3.12"],
}


def venv_path(device: str) -> Path:
    return VENVS_DIR / device


def python_exe(device: str) -> Path:
    base = venv_path(device)
    win = base / "Scripts" / "python.exe"
    return win if win.exists() else base / "bin" / "python"


def venv_exists(device: str) -> bool:
    return python_exe(device).exists() and (venv_path(device) / _READY_MARKER).exists()


def create_venv(device: str) -> None:
    """(Re)create a device venv and install its packages.

    All pip commands run as ``<venv python> -m pip`` — invoking the venv's
    ``pip.exe`` directly to upgrade pip fails on Windows ("To modify pip, please
    run ... -m pip ..."), which used to abort the whole benchmark.
    """
    if device not in DEVICE_PACKAGES:
        raise ValueError(f"Unknown device {device!r}. Valid: {list(DEVICE_PACKAGES)}")

    dest = venv_path(device)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        py_cmd = _DEVICE_PYTHON_CMD[device]
        print(f"[VenvManager] Creating .venvs/{device}/...")
        subprocess.check_call([*py_cmd, "-m", "venv", str(dest)])

        vpy = str(python_exe(device))

        # Non-fatal: a fresh 'python -m venv' already ships a usable pip; the
        # upgrade is only to avoid resolver warnings.
        try:
            subprocess.check_call([vpy, "-m", "pip", "install", "-q", "--upgrade",
                                   "pip", "setuptools", "wheel"])
        except subprocess.CalledProcessError as e:
            print(f"[VenvManager] pip self-upgrade skipped ({e}); continuing.")

        packages = DEVICE_PACKAGES[device]
        print(f"[VenvManager] Installing {device} packages: {', '.join(packages)}")
        subprocess.check_call([vpy, "-m", "pip", "install", "-q", *packages])
    except BaseException:
        # Never leave a half-built venv that venv_exists() would accept.
        shutil.rmtree(dest, ignore_errors=True)
        raise

    (dest / _READY_MARKER).write_text("")
    print(f"[VenvManager] .venvs/{device}/ ready.")


def delete_venv(device: str) -> None:
    dest = venv_path(device)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[VenvManager] Deleted .venvs/{device}/")


def ensure_venv(device: str) -> None:
    """Create the device venv only if it does not already exist."""
    if venv_exists(device):
        print(f"[VenvManager] .venvs/{device}/ already exists — reusing.")
    else:
        create_venv(device)
