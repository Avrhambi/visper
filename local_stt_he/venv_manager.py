"""
local_stt_he/venv_manager.py
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

# OpenVINO requires Python 3.12 for package compatibility
_DEVICE_PYTHON_CMD: dict[str, list[str]] = {
    "cpu":      [sys.executable],
    "cuda":     [sys.executable],
    "openvino": ["py", "-3.12"],
}


def venv_path(device: str) -> Path:
    return VENVS_DIR / device


def python_exe(device: str) -> Path:
    base = venv_path(device)
    win = base / "Scripts" / "python.exe"
    return win if win.exists() else base / "bin" / "python"


def venv_exists(device: str) -> bool:
    return python_exe(device).exists()


def create_venv(device: str) -> None:
    """(Re)create a device venv and install its packages."""
    if device not in DEVICE_PACKAGES:
        raise ValueError(f"Unknown device {device!r}. Valid: {list(DEVICE_PACKAGES)}")

    dest = venv_path(device)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    py_cmd = _DEVICE_PYTHON_CMD[device]
    print(f"[VenvManager] Creating .venvs/{device}/...")
    subprocess.check_call([*py_cmd, "-m", "venv", str(dest)])

    pip = dest / "Scripts" / "pip.exe"
    if not pip.exists():
        pip = dest / "bin" / "pip"

    subprocess.check_call([str(pip), "install", "-q", "--upgrade",
                           "pip", "setuptools", "wheel"])

    packages = DEVICE_PACKAGES[device]
    print(f"[VenvManager] Installing {device} packages: {', '.join(packages)}")
    subprocess.check_call([str(pip), "install", "-q", *packages])
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
