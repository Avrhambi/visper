"""
visper/model_router.py
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

# Built-in model assignments per language code (faster-whisper / CTranslate2)
_DEFAULT_MODELS: dict[str, str] = {
    "he": "ivrit-ai/whisper-large-v3-turbo-ct2",   # Hebrew fine-tune — best Hebrew quality
    "en": "distil-whisper/distil-large-v3-ct2",    # English-only distil — ~6x faster than turbo
    "ar": "Systran/faster-whisper-large-v3",        # Full v3 — Arabic needs it, turbo degrades here
    "_default": "Systran/faster-whisper-large-v3",  # All other languages
}

# MLX model assignments for Apple Silicon (mlx-community namespace)
# ivrit-ai's Hebrew fine-tune has no MLX release — base turbo is used instead.
_DEFAULT_MLX_MODELS: dict[str, str] = {
    "he": "mlx-community/whisper-large-v3-turbo",      # base turbo — strong Hebrew
    "en": "mlx-community/distil-whisper-large-v3-en",  # distil — ~6x faster, English-only
    "ar": "mlx-community/whisper-large-v3",             # full v3 — Arabic needs it
    "_default": "mlx-community/whisper-large-v3",       # Russian, Spanish, French, CJK, etc.
}


def _load_router_config(device: str = "") -> tuple[dict[str, str], str]:
    """Return (model_map, force_model) parsed from config.yaml in one read."""
    try:
        from visper._config import load_config
        cfg = load_config()
        force = cfg.get("force_model", "") or ""
        if device == "mlx":
            model_map = {**_DEFAULT_MLX_MODELS, **(cfg.get("models_mlx") or {})}
        else:
            model_map = {**_DEFAULT_MODELS, **(cfg.get("models") or {})}
        return model_map, force
    except Exception:
        pass
    defaults = _DEFAULT_MLX_MODELS if device == "mlx" else _DEFAULT_MODELS
    return dict(defaults), ""


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
        device = self._hw.get("device", "")
        model_map, force = _load_router_config(device)
        if force:
            return force
        defaults = _DEFAULT_MLX_MODELS if device == "mlx" else _DEFAULT_MODELS
        return model_map.get(language, model_map.get("_default", defaults["_default"]))

    def get(self, language: str) -> object:
        from visper.transcriber import Transcriber
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
