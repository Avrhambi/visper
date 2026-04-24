#!/usr/bin/env python
"""
transcribe_file.py
------------------
Offline transcription entry point.

Usage:
    python transcribe_file.py audio.mp3
    python transcribe_file.py audio.wav --output srt
    python transcribe_file.py audio.m4a --bucket long
    python transcribe_file.py audio.flac --no-file
    python transcribe_file.py audio.mp3 --clip
    python transcribe_file.py audio.mp3 --progress
    python transcribe_file.py *.wav          # batch mode
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add nvidia wheel DLL directories to PATH before any CUDA library loads (Windows).
if sys.platform == "win32":
    import site as _site
    _nvidia_dirs = []
    for _sp in _site.getsitepackages():
        _nv = Path(_sp) / "nvidia"
        if _nv.is_dir():
            for _pkg in _nv.iterdir():
                _bin = _pkg / "bin"
                if _bin.is_dir():
                    _nvidia_dirs.append(str(_bin))
    if _nvidia_dirs:
        os.environ["PATH"] = ";".join(_nvidia_dirs) + ";" + os.environ.get("PATH", "")

import argparse
import json
import time

ROOT = Path(__file__).parent

# Long-file threshold for auto-enabling progress output
_AUTO_PROGRESS_BUCKETS = {"long", "extended"}


def _resolve_bucket(path: Path, bucket: str) -> str:
    if bucket != "auto":
        return bucket
    try:
        import soundfile as sf
        dur = sf.info(str(path)).duration
    except Exception:
        try:
            from mutagen import File as MutagenFile
            f = MutagenFile(str(path))
            dur = float(f.info.length) if f and f.info else None
        except Exception:
            dur = None

    if dur is None:
        return "medium"
    if dur < 10:
        return "short"
    if dur < 30:
        return "medium"
    if dur < 60:
        return "long"
    return "extended"


def _rtf_speed_label(rtf: float) -> str:
    if rtf < 0.1:
        return "10x+ faster than real-time"
    if rtf < 0.33:
        return "3-10x faster than real-time"
    if rtf < 0.85:
        return "faster than real-time"
    return "slower than real-time"


def _friendly_error(e: Exception, path: Path) -> str:
    msg = str(e)
    low = msg.lower()
    if not path.exists():
        return f"File not found: {path}"
    if isinstance(e, FileNotFoundError):
        return f"File not found: {path}"
    if "ffmpeg" in low or "no such file" in low and path.suffix.lower() in (".mp3", ".mp4", ".m4a", ".aac"):
        return (f"ffmpeg is required to decode {path.suffix} files. "
                "Install from https://ffmpeg.org and add it to PATH.")
    if "no audio" in low or "empty" in low or "invalid data" in low:
        return f"No audio found in {path.name}. Is it a valid audio file?"
    if "out of memory" in low or "oom" in low or "cuda out" in low:
        return "GPU ran out of memory. The engine will retry on CPU automatically."
    if "cublas" in low or "cudnn" in low or "cuda" in low and "dll" in low:
        return "CUDA libraries not found. Re-run setup.py to register them."
    short_msg = msg.splitlines()[0][:120] if msg else "unknown error"
    return f"Transcription failed: {short_msg}"


def _format_txt(result) -> str:
    return result.text


def _format_srt(result) -> str:
    if not result.segments:
        return result.text
    lines = []
    for i, seg in enumerate(result.segments, 1):
        def _ts(s):
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            ms = int((s - int(s)) * 1000)
            return f"{h:02}:{m:02}:{sec:02},{ms:03}"
        lines.append(str(i))
        lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _format_json(result) -> str:
    return json.dumps({
        "text": result.text,
        "segments": result.segments,
        "audio_duration": result.audio_duration,
        "elapsed": result.elapsed,
        "rtf": result.rtf,
        "speed_label": _rtf_speed_label(result.rtf),
        "config": result.config_label,
        "tier": result.tier_used,
    }, ensure_ascii=False, indent=2)


def _load_config() -> dict:
    try:
        import yaml
        path = ROOT / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    except Exception:
        pass
    return {}


def transcribe_one(path: Path, engine, args, cfg: dict) -> tuple[str, float]:
    """Returns (text, rtf). Raises on error."""
    bucket = _resolve_bucket(path, args.bucket)
    fmt = args.output or cfg.get("output_format", "txt")

    print(f"[STT] Transcribing {path.name} (bucket={bucket})...", file=sys.stderr)

    # Auto-enable progress for long/extended files unless explicitly suppressed
    show_progress = getattr(args, "progress", False)
    if not show_progress and bucket in _AUTO_PROGRESS_BUCKETS:
        show_progress = True

    if show_progress:
        def _progress_cb(seg: dict) -> None:
            print(f"  [{seg['start']:.1f}s] {seg['text'].strip()}", file=sys.stderr)
        result = engine.transcribe(source=path, bucket=bucket, on_segment=_progress_cb)
    else:
        result = engine.transcribe(source=path, bucket=bucket)

    print(f"[STT] Done — RTF {result.rtf:.3f} ({_rtf_speed_label(result.rtf)}) "
          f"| {result.audio_duration:.1f}s audio in {result.elapsed:.1f}s",
          file=sys.stderr)

    if fmt == "srt":
        text_out = _format_srt(result)
        ext = ".srt"
    elif fmt == "json":
        text_out = _format_json(result)
        ext = ".json"
    else:
        text_out = _format_txt(result)
        ext = ".txt"

    if cfg.get("print_to_stdout", True):
        print(text_out)

    write_file = cfg.get("write_output_file", True) and not args.no_file
    if write_file:
        out_path = path.with_suffix(ext)
        out_path.write_text(text_out, encoding="utf-8")
        print(f"[STT] Written to {out_path}", file=sys.stderr)

    if args.clip:
        try:
            import pyperclip
            pyperclip.copy(result.text)
            print("[STT] Copied to clipboard.", file=sys.stderr)
        except Exception as e:
            print(f"[STT] Clipboard copy failed: {e}", file=sys.stderr)

    return result.text, result.rtf


def main():
    parser = argparse.ArgumentParser(description="Transcribe Hebrew audio file(s)")
    parser.add_argument("files", nargs="+", help="Audio file(s) to transcribe")
    parser.add_argument("--output", choices=["txt", "srt", "json"],
                        help="Output format (default: from config.yaml)")
    parser.add_argument("--bucket", default="auto",
                        choices=["auto", "short", "medium", "long", "extended"],
                        help="Duration bucket override")
    parser.add_argument("--no-file", action="store_true",
                        help="Print only, don't write output file")
    parser.add_argument("--clip", action="store_true",
                        help="Copy result to clipboard")
    parser.add_argument("--progress", "-p", action="store_true",
                        help="Print each segment to stderr as it is decoded")
    parser.add_argument("--background", action="store_true",
                        help="Run as background process (silent stdout)")
    args = parser.parse_args()

    if args.background:
        if sys.platform != "win32":
            import os
            if os.fork():
                sys.exit(0)
        sys.stdout = open(os.devnull, "w")

    cfg = _load_config()

    from local_stt_he.benchmark import get_best_config
    from local_stt_he.transcriber import Transcriber

    first_path = Path(args.files[0])
    first_bucket = _resolve_bucket(first_path, args.bucket)
    config = get_best_config(first_bucket)

    print("[STT] Loading model...", file=sys.stderr)
    engine = Transcriber(config)

    batch = len(args.files) > 1
    batch_start = time.time()
    succeeded = 0
    failed_files: list[str] = []
    rtf_sum = 0.0

    for f in args.files:
        path = Path(f)
        if not path.exists():
            if batch:
                print(f"[STT] File not found: {f}", file=sys.stderr)
                failed_files.append(f)
                continue
            else:
                print(f"[STT] File not found: {f}", file=sys.stderr)
                sys.exit(1)
        try:
            _, rtf = transcribe_one(path, engine, args, cfg)
            succeeded += 1
            rtf_sum += rtf
        except Exception as e:
            friendly = _friendly_error(e, path)
            print(f"[STT] Error: {friendly}", file=sys.stderr)
            if batch:
                failed_files.append(path.name)
            else:
                sys.exit(1)

    if batch:
        total_wall = time.time() - batch_start
        total = succeeded + len(failed_files)
        avg_rtf = rtf_sum / succeeded if succeeded else 0.0
        m, s = divmod(int(total_wall), 60)
        time_str = f"{m}m{s:02d}s" if m else f"{s}s"
        fail_str = f", {len(failed_files)} failed ({', '.join(failed_files)})" if failed_files else ""
        print(
            f"\n[STT] Batch complete: {succeeded}/{total} files, "
            f"{time_str} total, avg RTF {avg_rtf:.2f}{fail_str}",
            file=sys.stderr,
        )
        if failed_files:
            sys.exit(1)


if __name__ == "__main__":
    main()
