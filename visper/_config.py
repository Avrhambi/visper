"""
visper/_config.py
-----------------
Single source for reading ``config.yaml``.

The file ships *inside* the package (``visper/config.yaml``) so a plain
``pip install visper`` still has it — before this, 8 call sites each did
``yaml.safe_load(REPO_ROOT / "config.yaml")`` and silently fell back to
per-module defaults on a non-editable install.

``load_config()`` is cached; call ``reload_config()`` after editing the file
or running the benchmark within a long-lived process.
"""
from __future__ import annotations

import functools
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    """Parsed ``config.yaml`` as a dict. Returns ``{}`` if missing or unreadable."""
    try:
        import yaml
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def reload_config() -> dict:
    """Drop the cache and re-read the file."""
    load_config.cache_clear()
    return load_config()
