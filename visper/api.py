"""
visper/api.py
--------------------
Stable public API for cross-project use.

    from visper import transcribe, stream_transcribe, transcribe_chunked
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

# Single router instance — holds at most one model in memory, swaps on language change.
_router = None
_router_lock = threading.Lock()
_config_cache: dict = {}


def _get_config(bucket: str) -> dict:
    if bucket not in _config_cache:
        from visper.benchmark import get_best_config
        _config_cache[bucket] = get_best_config(bucket)
    return _config_cache[bucket]


def _get_router(hw_config: dict):
    global _router
    with _router_lock:
        if _router is None:
            from visper.model_router import ModelRouter
            _router = ModelRouter(hw_config)
    return _router


def transcribe_chunked(
    source: Union[str, Path, np.ndarray],
    on_segment: Callable[[dict], None],
    bucket: str = "auto",
    is_aborted: Optional[Callable[[], bool]] = None,
    language: str = "he",
    initial_prompt: str = None,
) -> str:
    """
    Like transcribe() but calls on_segment({"start", "end", "text"}) for each
    Whisper segment as it is decoded, enabling progress feedback on long files.

    Parameters
    ----------
    source : str | Path | np.ndarray
        File path or float32 numpy array at 16 kHz.
    on_segment : callable(seg: dict)
        Called once per decoded segment with keys 'start', 'end', 'text'.
        Runs on the calling thread (blocking, single-threaded).
    bucket : str
        'auto' = detect duration and pick the correct bucket.
    is_aborted : callable() -> bool, optional
        Checked between segments. If returns True, transcription stops early.

    Returns
    -------
    str : Full transcribed text.
    """
    resolved_bucket = _resolve_bucket(source, bucket)
    config = _get_config(resolved_bucket)
    engine = _get_router(config).get(language)
    result = engine.transcribe(source, bucket=resolved_bucket, on_segment=on_segment,
                               is_aborted=is_aborted, language=language,
                               initial_prompt=initial_prompt)
    return result.text


def transcribe(
    source: Union[str, Path, np.ndarray],
    bucket: str = "auto",
    is_aborted: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Transcribe Hebrew speech from a file or audio array.

    Parameters
    ----------
    source : str | Path | np.ndarray
        File path or float32 numpy array at 16 kHz.
    bucket : str
        'short' (<10s), 'medium' (10-30s), 'long' (30-60s), 'extended' (>60s).
        'auto' = detect duration and pick the correct bucket.
    is_aborted : callable() -> bool, optional
        Checked between segments. If returns True, transcription stops early.

    Returns
    -------
    str : Transcribed Hebrew text.
    """
    resolved_bucket = _resolve_bucket(source, bucket)
    config = _get_config(resolved_bucket)
    language = "he"  # transcribe() is Hebrew-only; use transcribe_chunked() for other languages
    engine = _get_router(config).get(language)
    result = engine.transcribe(source, bucket=resolved_bucket, is_aborted=is_aborted)
    return result.text


def stream_transcribe(
    on_transcript: Callable[[str, bool], None],
    source: Optional[Union[str, Path]] = None,
) -> None:
    """
    Streaming transcription.

    Parameters
    ----------
    on_transcript : callable(text: str, is_final: bool)
        Called for each transcribed chunk.
        is_final=True = silence-gated (complete thought).
        is_final=False = mid-speech forced emit.
    source : str | Path | None
        None = microphone live mode.
        File path = file streaming mode (incremental output).
    """
    from visper.streamer import LiveStreamer

    config = _get_config("streaming")
    streamer = LiveStreamer(on_transcript=on_transcript, config=config, source=source)
    streamer.start()
    try:
        import time
        while streamer.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


def _bucket_for_duration(duration: Optional[float]) -> str:
    if duration is None:
        return "medium"
    if duration < 10:
        return "short"
    if duration < 30:
        return "medium"
    if duration < 60:
        return "long"
    return "extended"


def _resolve_bucket(source, bucket: str) -> str:
    if bucket != "auto":
        return bucket
    return _bucket_for_duration(_get_duration(source))


def _get_duration(source) -> Optional[float]:
    if isinstance(source, np.ndarray):
        return len(source) / 16000.0
    try:
        import soundfile as sf
        info = sf.info(str(source))
        return info.duration
    except Exception:
        pass
    try:
        from mutagen import File as MutagenFile
        f = MutagenFile(str(source))
        if f and f.info:
            return float(f.info.length)
    except Exception:
        pass
    return None
