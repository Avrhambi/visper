"""
visper/translate.py
-------------------
Stage 2 of Hebrew → English: a dedicated text MT model.

The ivrit-ai fine-tune (stage 1) is a Hebrew *transcription* specialist; its
own ``task=translate`` output is poor, so Visper transcribes in Hebrew and then
translates the text here with ``Helsinki-NLP/opus-mt-tc-big-he-en`` converted to
CTranslate2 int8. ``ctranslate2`` is already present (transitive via
faster-whisper); the only added dependency is ``sentencepiece``.

The converted model (~210 MB) is not bundled. It is attached to a GitHub
release and downloaded + extracted to ``~/.visper/models/`` on first use, then
loaded fully offline. If it cannot be obtained or loaded,
``get_hebrew_english_translator()`` returns ``None`` and the caller falls back
to Whisper's own translate task — translation is never allowed to hard-fail a
transcription.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Optional

_MODEL_NAME = "opus-mt-tc-big-he-en-ct2"
_MODEL_DIR = Path.home() / ".visper" / "models" / _MODEL_NAME

# Release asset — built with:
#   ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-he-en \
#       --quantization int8
#   (+ source.spm / target.spm / vocab.json copied in from the HF repo)
_ASSET_URL = (
    "https://github.com/Avrhambi/visper/releases/download/"
    "mt-he-en-v1/opus-mt-tc-big-he-en-ct2.tar.gz"
)
_ASSET_SHA256 = "fd378bcb2c52c503b4f6816b6a0a382914621612383e141aacbc6916b1337642"

# Override the download URL for testing / air-gapped mirrors.
_ASSET_URL_ENV = "VISPER_MT_HE_EN_URL"

_EOS = "</s>"
_SPECIALS = frozenset((_EOS, "<pad>", "<unk>"))

_lock = threading.Lock()
_singleton: "Optional[HebrewEnglishTranslator]" = None
_load_failed = False


# ---------------------------------------------------------------------------
# Model acquisition
# ---------------------------------------------------------------------------

def he_en_supported() -> bool:
    """Whether the he->en path *can* run — deps importable and the model either
    already on disk or fetchable on first use. Cheap: no download, no model load.

    ``/health`` uses this to decide whether to advertise Hebrew translation.
    """
    if _load_failed and not _model_present():
        return False
    try:
        import ctranslate2  # noqa: F401
        import sentencepiece  # noqa: F401
    except Exception:
        return False
    return True


def _model_present() -> bool:
    return (_MODEL_DIR / "model.bin").is_file() and \
           (_MODEL_DIR / "source.spm").is_file() and \
           (_MODEL_DIR / "target.spm").is_file()


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract, refusing any member that would escape ``dest`` (path traversal)."""
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if dest != target and dest not in target.parents:
            raise RuntimeError(f"unsafe path in archive: {member.name!r}")
    tar.extractall(dest)


def ensure_model() -> bool:
    """Make the model available locally. Returns True if it is present afterwards.

    No-ops (no network) when the model is already on disk.
    """
    if _model_present():
        return True

    url = os.environ.get(_ASSET_URL_ENV, _ASSET_URL)
    _MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"[translate] fetching he->en model ({url})...", file=sys.stderr, flush=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="visper-mt-", dir=_MODEL_DIR.parent))
    tgz = tmp_dir / "model.tar.gz"
    try:
        with urllib.request.urlopen(url) as resp, open(tgz, "wb") as fh:  # noqa: S310
            digest = hashlib.sha256()
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                digest.update(chunk)

        got = digest.hexdigest()
        # An env override is a deliberate local mirror — don't gate it on the
        # checksum of the canonical release asset.
        if _ASSET_URL_ENV not in os.environ and got != _ASSET_SHA256:
            raise RuntimeError(
                f"checksum mismatch: expected {_ASSET_SHA256}, got {got}")

        with tarfile.open(tgz, "r:gz") as tar:
            _safe_extract(tar, tmp_dir)

        extracted = tmp_dir / _MODEL_NAME
        if not (extracted / "model.bin").is_file():
            raise RuntimeError(f"archive did not contain {_MODEL_NAME}/model.bin")

        if _MODEL_DIR.exists():
            return _model_present()
        os.replace(extracted, _MODEL_DIR)  # atomic on the same filesystem
        return _model_present()
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class HebrewEnglishTranslator:
    """he → en text MT. Load once, reuse; not thread-safe for concurrent calls."""

    def __init__(self, model_dir: Path = _MODEL_DIR, *,
                 device: str = "cpu", compute_type: str = "int8",
                 intra_threads: int = 4, beam_size: int = 4):
        import ctranslate2
        import sentencepiece as spm

        self._beam_size = beam_size
        self._translator = ctranslate2.Translator(
            str(model_dir), device=device, compute_type=compute_type,
            inter_threads=1, intra_threads=intra_threads,
        )
        self._sp_src = spm.SentencePieceProcessor(
            model_file=str(model_dir / "source.spm"))
        self._sp_tgt = spm.SentencePieceProcessor(
            model_file=str(model_dir / "target.spm"))

    def translate(self, texts: list[str]) -> list[str]:
        """Translate each string he → en. Blank strings pass through unchanged.

        One sentence-sized segment per batch entry — Marian degrades on
        multi-sentence input — and the whole list goes in a single
        ``translate_batch`` so the model is exercised once.
        """
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        if not idx:
            return list(texts)

        tokens = [self._sp_src.encode(texts[i].strip(), out_type=str) + [_EOS]
                  for i in idx]
        results = self._translator.translate_batch(
            tokens, beam_size=self._beam_size, max_batch_size=32)

        out = list(texts)
        for i, res in zip(idx, results):
            hyp = [t for t in res.hypotheses[0] if t not in _SPECIALS]
            out[i] = self._sp_tgt.decode(hyp).strip()
        return out


def get_hebrew_english_translator() -> Optional[HebrewEnglishTranslator]:
    """The cached he → en translator, or ``None`` if it can't be made ready.

    ``None`` means: model not downloadable, or ``ctranslate2`` /
    ``sentencepiece`` not importable, or the model failed to load. Callers must
    treat that as "translation unavailable" and degrade, not raise.
    """
    global _singleton, _load_failed
    if _singleton is not None:
        return _singleton
    if _load_failed:
        return None

    with _lock:
        if _singleton is not None:
            return _singleton
        if _load_failed:
            return None
        try:
            if not ensure_model():
                raise RuntimeError("model not present after ensure_model()")
            _singleton = HebrewEnglishTranslator()
            return _singleton
        except Exception as e:  # noqa: BLE001 — any failure => degrade
            _load_failed = True
            print(f"[translate] he->en translation unavailable: {e}",
                  file=sys.stderr, flush=True)
            return None


def reset_cache() -> None:
    """Drop the cached translator (tests, or after a manual model install)."""
    global _singleton, _load_failed
    with _lock:
        _singleton = None
        _load_failed = False
