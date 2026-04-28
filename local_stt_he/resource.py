"""
core/resource.py
----------------
Single place that reads resource_profile from config.yaml and enforces
thread caps, GPU restrictions, and process priority.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

PROFILE_DEFAULTS = {
    "foreground": {
        "thread_fraction": 1.0,
        "process_priority": "normal",
        "idle_unload_seconds": 0,
        "allow_gpu": True,
        "force_compute_type": None,
    },
    "background": {
        "thread_fraction": 0.5,
        "process_priority": "low",
        "idle_unload_seconds": 60,
        "allow_gpu": True,
        "force_compute_type": None,
    },
    "minimal": {
        "thread_fraction": 0.25,
        "process_priority": "low",
        "idle_unload_seconds": 30,
        "allow_gpu": False,
        "force_compute_type": "int8",
    },
}


def _load_user_config() -> dict:
    try:
        import yaml
        path = ROOT / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    except Exception:
        pass
    return {}


def _set_process_priority(level: str) -> None:
    if level == "normal":
        return
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(-1, 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:
        pass


def apply_profile(config: dict) -> dict:
    """
    Takes raw hardware config from get_best_config() and applies
    resource profile constraints from config.yaml on top.
    """
    cfg = dict(config)
    user = _load_user_config()

    profile_name = user.get("resource_profile", "foreground")
    # If run_in_background=true and profile is still foreground, upgrade
    if user.get("run_in_background") and profile_name == "foreground":
        profile_name = "background"

    profile = PROFILE_DEFAULTS.get(profile_name, PROFILE_DEFAULTS["foreground"])

    # Apply thread fraction
    import multiprocessing
    logical = multiprocessing.cpu_count()
    base_threads = cfg.get("cpu_threads", 4)
    new_threads = max(1, int(base_threads * profile["thread_fraction"]))

    # Minimal: cap at 2
    if profile_name == "minimal":
        new_threads = min(new_threads, 2)

    # Hard cap from user config
    max_cap = user.get("max_cpu_threads", 0)
    if max_cap > 0:
        new_threads = min(new_threads, max_cap)

    cfg["cpu_threads"] = new_threads

    # GPU restriction
    if not profile["allow_gpu"]:
        cfg["device"] = "cpu"
        cfg["compute_type"] = "int8"

    # Compute type override from profile
    if profile["force_compute_type"]:
        cfg["compute_type"] = profile["force_compute_type"]

    # Set process priority
    _set_process_priority(profile["process_priority"])

    return cfg


def get_idle_unload_seconds() -> int:
    user = _load_user_config()
    explicit = user.get("idle_unload_seconds", 0)
    if explicit > 0:
        return explicit
    profile_name = user.get("resource_profile", "foreground")
    return PROFILE_DEFAULTS.get(profile_name, PROFILE_DEFAULTS["foreground"])["idle_unload_seconds"]


def check_memory_headroom(config: dict) -> dict:
    """
    Check RAM/VRAM limits. Demotes cuda→cpu if VRAM limit exceeded.
    Silently skips if psutil not available.
    """
    cfg = dict(config)
    user = _load_user_config()
    max_ram_mb   = user.get("max_ram_mb", 0)
    max_vram_mb  = user.get("max_vram_mb", 0)

    # RAM check
    if max_ram_mb > 0:
        try:
            import psutil
            used_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if used_mb > max_ram_mb:
                import warnings
                warnings.warn(f"[STT] RAM usage {used_mb:.0f} MB exceeds limit {max_ram_mb} MB")
        except ImportError:
            pass
        except Exception:
            pass

    # VRAM check
    if max_vram_mb > 0 and cfg.get("device") in ("cuda",):
        vram_used = _get_vram_used_mb()
        if vram_used is not None and vram_used > max_vram_mb:
            print(f"[STT] VRAM limit reached ({vram_used:.0f} MB > {max_vram_mb} MB) — falling back to CPU",
                  file=sys.stderr)
            cfg["device"] = "cpu"
            cfg["compute_type"] = "int8"

    return cfg


def check_memory_during_session() -> Optional[str]:
    """
    Periodic check during a live session. Returns 'demote' if VRAM exceeded, else None.
    """
    user = _load_user_config()
    max_vram_mb = user.get("max_vram_mb", 0)
    if max_vram_mb > 0:
        vram_used = _get_vram_used_mb()
        if vram_used is not None and vram_used > max_vram_mb:
            print(f"[STT] Warning — VRAM {vram_used:.0f} MB > {max_vram_mb} MB limit. "
                  "Signalling model demote to CPU.", file=sys.stderr)
            return "demote"
    return None


def _get_vram_used_mb() -> Optional[float]:
    # Try torch first
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved(0) / 1024 / 1024
    except ImportError:
        pass
    # Try nvidia-smi
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        return float(out.split("\n")[0])
    except Exception:
        pass
    return None


def _get_vram_free_mb() -> Optional[float]:
    """Return free VRAM in MB using torch.cuda.mem_get_info (different from reserved)."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info(0)
            return free_bytes / 1024 / 1024
    except ImportError:
        pass
    except Exception:
        pass
    return None


# Estimated VRAM required to load whisper-large-v3-turbo-ct2, by compute type.
_MODEL_VRAM_ESTIMATE_MB = {
    "float16":      1650,
    "bfloat16":     1650,
    "int8_float16": 950,
    "int8_bfloat16": 950,
    "int8_float32": 1200,
    "int8":         900,
}
_VRAM_HEADROOM_MB = 200  # Keep at least this many MB free after load


def check_vram_before_load(config: dict) -> dict:
    """
    Proactively check free VRAM before loading a CUDA model.
    Demotes to CPU+int8 if the estimated model size won't fit with headroom.
    This is a pre-load guard; transcriber._load_direct() also catches OOM at load time.
    """
    cfg = dict(config)
    if cfg.get("device") != "cuda":
        return cfg

    compute_type = cfg.get("compute_type", "int8")
    required_mb = _MODEL_VRAM_ESTIMATE_MB.get(compute_type, 1200) + _VRAM_HEADROOM_MB

    free_mb = _get_vram_free_mb()
    if free_mb is None:
        return cfg  # Can't check — proceed optimistically

    if free_mb < required_mb:
        print(
            f"[STT] VRAM pre-load: {free_mb:.0f} MB free, need ~{required_mb} MB "
            f"({compute_type} + {_VRAM_HEADROOM_MB} MB headroom) — falling back to CPU int8",
            file=sys.stderr,
        )
        cfg["device"] = "cpu"
        cfg["compute_type"] = "int8"

    return cfg
