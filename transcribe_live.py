#!/usr/bin/env python
"""
transcribe_live.py
------------------
Online/streaming transcription entry point.

Usage:
    python transcribe_live.py                        # microphone live mode
    python transcribe_live.py --file audio.mp3       # file streaming mode
    python transcribe_live.py --clip                 # copy final result to clipboard
    python transcribe_live.py --output result.txt    # write accumulated result to file
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Live/streaming Hebrew transcription")
    parser.add_argument("--file", metavar="PATH",
                        help="Stream a file instead of microphone")
    parser.add_argument("--clip", action="store_true",
                        help="Copy final accumulated result to clipboard")
    parser.add_argument("--output", metavar="PATH",
                        help="Write accumulated result to this file")
    args = parser.parse_args()

    from core.benchmark import get_best_config
    from core.streamer import LiveStreamer

    config = get_best_config("streaming")
    source = Path(args.file) if args.file else None

    if source:
        print(f"[STT] Streaming {source.name} ({source.stat().st_size // 1024} KB)...",
              file=sys.stderr)
    else:
        print("[STT] Listening... (Ctrl+C to stop and save)", file=sys.stderr)

    accumulated: list[str] = []

    def on_transcript(text: str, is_final: bool) -> None:
        if is_final:
            print(text)
            accumulated.append(text)
        else:
            print(f"{text}…")

    streamer = LiveStreamer(on_transcript=on_transcript, config=config, source=source)
    streamer.start()

    try:
        while streamer.is_running:
            time.sleep(0.1)
            if source and not streamer.is_running:
                break
    except KeyboardInterrupt:
        print("\n[STT] Stopping...", file=sys.stderr)
    finally:
        streamer.stop()

    stats = streamer.stats
    print(
        f"\n[STT] Session ended — {stats['segments']} segments, "
        f"{stats['total_audio_duration']:.1f}s audio, avg RTF {stats['avg_rtf']:.2f}",
        file=sys.stderr,
    )
    if stats["dropped_chunks"]:
        print(f"[STT] Warning — {stats['dropped_chunks']} chunk(s) dropped during session.",
              file=sys.stderr)

    full_text = "\n".join(accumulated)

    if args.output and full_text:
        Path(args.output).write_text(full_text, encoding="utf-8")
        print(f"[STT] Written to {args.output}", file=sys.stderr)

    if args.clip and full_text:
        try:
            import pyperclip
            pyperclip.copy(full_text)
            print("[STT] Copied to clipboard.", file=sys.stderr)
        except Exception as e:
            print(f"[STT] Clipboard copy failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
