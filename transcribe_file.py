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
    python transcribe_file.py *.wav          # batch mode
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


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


def transcribe_one(path: Path, engine, args, cfg: dict) -> str:
    bucket = _resolve_bucket(path, args.bucket)
    fmt = args.output or cfg.get("output_format", "txt")

    print(f"[STT] Transcribing {path.name} (bucket={bucket})...", file=sys.stderr)
    result = engine.transcribe(source=path, bucket=bucket)
    print(f"[STT] Done — RTF {result.rtf:.3f} ({result.audio_duration:.1f}s audio / {result.elapsed:.1f}s inference)",
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

    return result.text


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
    parser.add_argument("--background", action="store_true",
                        help="Run as background process (silent stdout)")
    args = parser.parse_args()

    if args.background:
        # Daemonize on Unix; subprocess detach on Windows
        if sys.platform != "win32":
            import os
            if os.fork():
                sys.exit(0)
        # silence stdout
        sys.stdout = open(os.devnull, "w")

    cfg = _load_config()

    from core.benchmark import get_best_config
    from core.transcriber import Transcriber

    # Resolve bucket for first file to pick config (batch reuses same engine)
    first_path = Path(args.files[0])
    first_bucket = _resolve_bucket(first_path, args.bucket)
    config = get_best_config(first_bucket)

    print(f"[STT] Loading model...", file=sys.stderr)
    engine = Transcriber(config)

    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"[STT] File not found: {f}", file=sys.stderr)
            sys.exit(1)
        try:
            transcribe_one(path, engine, args, cfg)
        except Exception as e:
            print(f"[STT] Error transcribing {path.name}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
