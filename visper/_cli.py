"""Entry point wrappers for visper-* CLI commands."""


def file():
    from visper.transcribe_file import main
    main()


def live():
    from visper.transcribe_live import main
    main()


def benchmark():
    from visper.run_benchmark import main
    main()


def help():
    print("""
Local Speech-to-Text — command reference
─────────────────────────────────────────

  visper-file <audio>  [options]        Transcribe an audio file (offline)
  visper-live          [options]        Live microphone or file streaming
  visper-server        [options]        Start the web UI + REST/WebSocket server
  visper-benchmark     [options]        Measure hardware and pick the best backend
  visper-help                           Show this message

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

visper-file <audio> [<audio2> ...]
  Transcribes one or more audio files and writes a text file next to each.
  Pass multiple files or a glob (*.wav) for batch mode.

  --output  txt|srt|vtt|json   Output format (default: txt)
  --language he|en             Language — routes to the right model (default: he)
  --prompt "TEXT"              Seed Whisper with names, terms, or topic context.
                               Use the same language as the audio. Max ~55 words.
  --no-file                    Print to stdout only, don't write an output file
  --clip                       Copy result to clipboard after transcription
  --progress / -p              Print each segment to stderr as it is decoded
  --bucket short|medium|long   Override automatic duration-bucket selection
  --accuracy fast|balanced|accurate   Override accuracy tier from config.yaml
  --profile foreground|background|minimal   Override resource profile
  --background                 Detach from terminal (silences stdout on Windows)

  Examples:
    visper-file meeting.mp3
    visper-file lecture.wav --output srt --language en
    visper-file audio.mp3 --prompt "team standup, participants: Yossi, Rachel"
    visper-file *.wav --output json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

visper-live
  Real-time transcription from the microphone, or streamed from a file.
  Press Ctrl+C to stop. Final accumulated text is printed/saved/clipped.

  --file PATH                  Stream a file instead of the microphone
  --language he|en             Language — routes to the right model (default: he)
  --prompt "TEXT"              Seed Whisper with context for every chunk
  --output PATH                Write the full accumulated transcript to a file on stop
  --clip                       Copy accumulated transcript to clipboard on stop
  --progress / -p              Print each segment with a wall-clock timestamp
  --accuracy fast|balanced|accurate   Override accuracy tier for the session
  --background                 Detach from terminal

  Examples:
    visper-live
    visper-live --language en --prompt "quarterly review, speaker: John"
    visper-live --file interview.mp3 --output transcript.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

visper-server
  Starts the REST/WebSocket API on http://localhost:8000.
  Open web/index.html in your browser to use the web UI.

  --host HOST      Bind address (default: 0.0.0.0)
  --port PORT      Port (default: 8000)
  --reload         Auto-reload on code changes (development mode)

  Endpoints:
    GET  /health                 Device, model, and status info
    POST /transcribe             Upload file → {text, segments, rtf}
    POST /transcribe/stream      Upload file → SSE stream of segment events
    WS   /ws/live                WebSocket live microphone transcription

  curl examples:
    curl -F "file=@audio.mp3" -F "language=he" http://localhost:8000/transcribe
    curl -F "file=@audio.mp3" -F "language=en" \\
         -F "initial_prompt=meeting notes" http://localhost:8000/transcribe/stream
    curl http://localhost:8000/health

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

visper-benchmark
  Measures your hardware once and writes benchmark_results.json.
  Runs automatically on first install. Re-run if you change hardware.

  --force          Re-run even if results already exist
  --fast           Primary device only (~60 s instead of 3–5 min)
  --quick          Heuristic only — no inference timing, instant result
  --full           Exhaustive — tests all compute types × thread counts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tip: edit config.yaml to tune accuracy, language, VAD, and output format
     without passing flags every time.
""")
