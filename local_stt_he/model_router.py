"""
local_stt_he/model_router.py
-----------------------------
Single-slot model router. Holds at most one Transcriber in memory.
Swaps models based on requested language. Swap cost = model load time (~5-30s).
Thread-safe — concurrent requests for different languages serialize on the swap lock.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

# Built-in model assignments per language code
_DEFAULT_MODELS: dict[str, str] = {
    "he": "ivrit-ai/whisper-large-v3-turbo-ct2",   # Hebrew fine-tune — best Hebrew quality
    "en": "distil-whisper/distil-large-v3-ct2",    # English-only distil — ~6x faster than turbo
    "ar": "Systran/faster-whisper-large-v3",        # Full v3 — Arabic needs it, turbo degrades here
    "_default": "Systran/faster-whisper-large-v3",         # All other languages
}


def _load_model_map() -> dict[str, str]:
    """Merge config.yaml `models` section over built-in defaults."""
    try:
        import yaml
        path = ROOT / "config.yaml"
        if path.exists():
            cfg = yaml.safe_load(path.read_text()) or {}
            overrides = cfg.get("models") or {}
            return {**_DEFAULT_MODELS, **overrides}
    except Exception:
        pass
    return dict(_DEFAULT_MODELS)


def _load_force_model() -> str:
    try:
        import yaml
        path = ROOT / "config.yaml"
        if path.exists():
            cfg = yaml.safe_load(path.read_text()) or {}
            return cfg.get("force_model", "") or ""
    except Exception:
        pass
    return ""


class ModelRouter:
    """
    Manages a single active Transcriber.
    Call .get(language, hw_config) to get the correct Transcriber for a language.
    Model is swapped (unload → load) only when the target model_id changes.
    """

    def __init__(self, hw_config: dict):
        self._hw = hw_config
        self._lock = threading.Lock()
        self._active_model_id: Optional[str] = None
        self._transcriber = None

    def resolve_model_id(self, language: str) -> str:
        force = _load_force_model()
        if force:
            return force
        model_map = _load_model_map()
        return model_map.get(language, model_map.get("_default", _DEFAULT_MODELS["_default"]))

    def get(self, language: str) -> object:
        from local_stt_he.transcriber import Transcriber
        target = self.resolve_model_id(language)
        with self._lock:
            if self._active_model_id == target and self._transcriber is not None:
                return self._transcriber
            if self._transcriber is not None:
                print(f"[STT] Swapping model: {self._active_model_id} → {target}", file=sys.stderr)
                self._transcriber.unload()
                self._transcriber = None
            else:
                print(f"[STT] Loading model for language '{language}': {target}", file=sys.stderr)
            cfg = {**self._hw, "model_id": target}
            self._transcriber = Transcriber(cfg)
            self._active_model_id = target
            return self._transcriber

    def unload(self) -> None:
        with self._lock:
            if self._transcriber is not None:
                self._transcriber.unload()
                self._transcriber = None
                self._active_model_id = None
