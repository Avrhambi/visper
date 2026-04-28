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
    python transcribe_live.py --progress             # print each segment with timestamp
    python transcribe_live.py --background           # run detached from terminal
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
import time


def main():
    parser = argparse.ArgumentParser(description="Live/streaming Hebrew transcription")
    parser.add_argument("--file", metavar="PATH",
                        help="Stream a file instead of microphone")
    parser.add_argument("--clip", action="store_true",
                        help="Copy final accumulated result to clipboard")
    parser.add_argument("--output", metavar="PATH",
                        help="Write accumulated result to this file")
    parser.add_argument("--progress", "-p", action="store_true",
                        help="Print each segment with a timestamp as it arrives")
    parser.add_argument("--background", action="store_true",
                        help="Run as background process (detached)")
    parser.add_argument("--accuracy",
                        choices=["auto", "fast", "balanced", "accurate"],
                        default=None,
                        help="Override accuracy_mode from config.yaml")
    parser.add_argument("--language", choices=["he", "en"], default="he",
                        help="Language to transcribe (default: he)")
    args = parser.parse_args()

    if args.background:
        if sys.platform != "win32":
            if os.fork():
                sys.exit(0)
        sys.stdout = open(os.devnull, "w")

    if args.accuracy:
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            _cfg_path = _Path(__file__).parent / "config.yaml"
            _cfg = _yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}
            _cfg['accuracy_mode'] = args.accuracy
        except Exception:
            _cfg = {'accuracy_mode': args.accuracy}
        import local_stt_he.params as _p
        _p._load_user_config = lambda: _cfg

    from local_stt_he.benchmark import get_best_config
    from local_stt_he.streamer import LiveStreamer

    config = get_best_config("streaming")
    config["language"] = args.language
    source = Path(args.file) if args.file else None

    if source:
        print(f"[STT] Streaming {source.name} ({source.stat().st_size // 1024} KB)...",
              file=sys.stderr)
    else:
        print("[STT] Listening... (Ctrl+C to stop and save)", file=sys.stderr)

    accumulated: list[str] = []
    _status_active = [False]  # mutable flag so closure can clear it

    def _clear_status():
        if _status_active[0]:
            print("\r" + " " * 20 + "\r", end="", file=sys.stderr, flush=True)
            _status_active[0] = False

    def _show_status(msg: str):
        print(f"\r  {msg}", end="", file=sys.stderr, flush=True)
        _status_active[0] = True

    def on_transcript(text: str, is_final: bool) -> None:
        _clear_status()
        if args.progress:
            ts = time.strftime("%H:%M:%S")
            if is_final:
                print(f"[{ts}] {text}")
                accumulated.append(text)
            else:
                print(f"[{ts}]… {text}")
        else:
            if is_final:
                print(text)
                accumulated.append(text)
            else:
                print(f"{text}…")

    streamer = LiveStreamer(on_transcript=on_transcript, config=config, source=source)
    streamer.start()

    # Status line updater — only in mic mode (file mode runs as fast as possible)
    _last_had_text = [time.time()]

    try:
        while streamer.is_running:
            time.sleep(0.25)
            if source:
                # File mode: no silence indicator needed
                continue
            # Show listening/processing status based on queue depth
            if streamer.buffer_duration > 0.5:
                _show_status("[processing...]")
            else:
                _show_status("[listening...]")
            if source and not streamer.is_running:
                break
    except KeyboardInterrupt:
        _clear_status()
        print("\n[STT] Stopping...", file=sys.stderr)
    finally:
        _clear_status()
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
