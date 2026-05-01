# וִויסְפֶּר — Local Speech-to-Text

Offline speech-to-text on your own hardware. No cloud, no subscriptions, no data leaves your machine. Self-benchmarks and configures itself on first run.

Supports Hebrew, English, Arabic, Russian, and other languages — each routed to the best model automatically.

---

## Getting Started

### Non-developers (Windows)

**One prerequisite:** [Python 3.10+](https://www.python.org/downloads/) — check **"Add Python to PATH"** during installation.

Then:

1. Download this repo ([ZIP](https://github.com/Avrhambi/visper/archive/refs/heads/master.zip)) and extract it
2. Double-click **`start.bat`**

That's it. On first run it installs all dependencies, downloads the model (~1.5 GB, one time), runs the hardware benchmark, and opens the web UI at `http://localhost:8000` in your browser. Every run after that starts in a few seconds.

To get the latest version: double-click **`update.bat`**.

> If anything goes wrong during setup, the error screen will copy a help message to your clipboard automatically — paste it into an ai chatbot like: ChatGPT or Claude for step-by-step guidance.

### Developers

```bash
git clone https://github.com/Avrhambi/visper && cd visper
python install.py           # installs deps, downloads model (~1.5 GB once), runs benchmark
visper-server                  # web UI at http://localhost:8000
visper-file audio.mp3          # or use the CLI directly
```

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org) on PATH (WAV files work without it).

> **HuggingFace token:** If the model repo is gated, copy `.env.example` to `.env` and set `HF_TOKEN=hf_...` before running `install.py`.

---

## What It Does

Transcribes audio files and microphone input using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Auto-detects your hardware (CPU / CUDA / Intel iGPU via OpenVINO) and benchmarks it once to pick the best inference backend and accuracy tier. No manual configuration needed — results are cached in `benchmark_results.json`.

Language routing selects the right model per request: Hebrew fine-tune (`ivrit-ai`) for Hebrew, a distilled fast model for English, and `faster-whisper-large-v3` for Arabic and all other languages. Only the active model stays in memory; a language switch swaps it.

---

## Web UI

```bash
visper-server          # starts on http://localhost:8000
```

Or double-click `start.bat` — it starts the server and opens the browser automatically.

**Features:**
- Upload single or multiple audio files (MP3, WAV, M4A, and more) — or a whole folder
- Batch queue: sequential processing with per-file status, each result saved to library
- Live microphone recording with real-time transcription
- 12-language picker (Hebrew, English, Arabic, Russian, Spanish, French, German, Portuguese, Italian, Chinese, Japanese, Korean) with automatic model routing
- Initial prompt field — seed Whisper with names, terms, or context to improve accuracy
- Transcription library saved locally in the browser; click any timestamp to seek audio
- RTL layout for Hebrew and Arabic; LTR for all other languages

---

## CLI

### Offline file transcription — `visper-file`

```bash
visper-file audio.mp3                                      # transcribe → write audio.txt
visper-file audio.wav --output srt                         # SRT subtitles
visper-file audio.mp3 --output vtt                         # WebVTT subtitles
visper-file audio.mp3 --output json                        # JSON with segments, RTF, config
visper-file audio.mp3 --no-file --clip                     # print + copy to clipboard, no file written
visper-file audio.mp3 --progress                           # print each segment as it is decoded
visper-file audio.mp3 --language en                        # transcribe English
visper-file audio.mp3 --prompt "team meeting, participants: Yossi, Rachel"  # initial prompt
visper-file *.wav                                          # batch mode — all WAV files in current dir
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | from config | `txt` / `srt` / `vtt` / `json` |
| `--bucket` | `auto` | Force duration bucket: `short` / `medium` / `long` / `extended` |
| `--language` | `he` | Language code — routes to the correct model automatically |
| `--prompt` | — | Seed Whisper with context: names, terms, topic. Use the same language as the audio |
| `--progress` / `-p` | off | Print each segment to stderr as it decodes |
| `--no-file` | off | Print to stdout only, don't write an output file |
| `--clip` | off | Copy final transcript to clipboard |
| `--accuracy` | from config | Override accuracy tier: `fast` / `balanced` / `accurate` |
| `--profile` | from config | Override resource profile: `foreground` / `background` / `minimal` |
| `--background` | off | Detach from terminal (Windows: silences stdout) |

### Live / microphone — `visper-live`

```bash
visper-live                                                # microphone transcription
visper-live --file audio.mp3                               # file streaming mode
visper-live --output result.txt                            # save accumulated transcript on stop
visper-live --clip                                         # copy to clipboard on Ctrl+C
visper-live --progress                                     # print each segment with timestamp
visper-live --language en                                  # transcribe English
visper-live --prompt "dev team standup, participants: Yossi, Rachel"  # initial prompt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--file` | — | Stream a file instead of the microphone |
| `--language` | `he` | Language code — routes to the correct model automatically |
| `--prompt` | — | Seed Whisper with context for every chunk in the session |
| `--output` | — | Write accumulated transcript to a file on stop |
| `--clip` | off | Copy accumulated transcript to clipboard on stop |
| `--progress` / `-p` | off | Print each segment with a wall-clock timestamp |
| `--accuracy` | from config | Override accuracy tier for the session |
| `--background` | off | Detach from terminal |

### Web server — `visper-server`

```bash
visper-server                                              # starts on http://localhost:8000
```

The server exposes a full web UI and a REST/WebSocket API.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Device, model, and status info |
| POST | `/transcribe` | Upload a file → `{text, segments, rtf}` |
| POST | `/transcribe/stream` | Upload a file → SSE stream of segment events |
| WS | `/ws/live` | WebSocket live microphone transcription |

```bash
# Quick API examples
curl -F "file=@audio.mp3" -F "language=he" http://localhost:8000/transcribe
curl -F "file=@audio.mp3" -F "language=en" -F "initial_prompt=meeting notes" \
     http://localhost:8000/transcribe/stream
curl http://localhost:8000/health
```

If the CLI entry points aren't on PATH yet (before `pip install -e .`):

```bash
python transcribe_file.py audio.mp3
python transcribe_live.py
python -m uvicorn local_stt_he.server:app --port 8000
```

---

## Python API

```bash
pip install -e .
```

```python
from local_stt_he import transcribe, stream_transcribe, transcribe_chunked

# Offline — returns full transcript text
text = transcribe("audio.mp3")
text = transcribe("audio.wav", bucket="long")

# With per-segment progress callback
def on_segment(seg):
    print(f"[{seg['start']:.1f}s] {seg['text'].strip()}")

text = transcribe_chunked("recording.mp3", on_segment=on_segment)

# Live streaming
def on_transcript(text, is_final):
    print("FINAL:" if is_final else "...", text)

stream_transcribe(on_transcript)                # microphone
stream_transcribe(on_transcript, "audio.mp3")  # file
```

`is_final=True` — emitted on silence (complete utterance).  
`is_final=False` — forced emit when a chunk exceeds `max_chunk_seconds` mid-speech.

---

## Configuration

Edit `config.yaml` to adjust behavior. Key options:

| Key | Default | Options |
|-----|---------|---------|
| `accuracy_mode` | `auto` | `auto` / `fast` / `light` / `balanced` / `accurate` |
| `resource_profile` | `foreground` | `foreground` / `background` / `minimal` |
| `output_format` | `txt` | `txt` / `srt` / `json` |
| `vad_filter` | `true` | Enable Silero VAD |
| `max_chunk_seconds` | `28` | Max live chunk before forced emit (CLI) |
| `audio_denoise` | `false` | Noise reduction before Whisper (opt-in, adds ~1s per file) |
| `audio_highpass` | `false` | 80 Hz high-pass filter — removes HVAC rumble and handling noise |
| `audio_normalize` | `false` | RMS volume normalization — helps whispered or far-mic audio |
| `hotwords` | `""` | Comma-separated terms to bias Whisper toward (names, brands, domain terms) |
| `confidence_retry_enabled` | `null` | Retry at next accuracy tier if confidence is low; `null` = auto |

Full reference with all options is in `config.yaml`.

### Resource profiles

| Profile | Threads | Priority | GPU | Model unload |
|---------|---------|----------|-----|--------------|
| `foreground` | benchmark result | normal | yes | never |
| `background` | 50% | low | yes | after 60s idle |
| `minimal` | 25% (max 2) | low | no (CPU only) | after 30s idle |

### Accuracy tiers (when `accuracy_mode: auto`)

| Tier | beam_size | best_of | temperature fallback ladder | Auto-selected when |
|------|-----------|---------|----------------------------|-------------------|
| `fast` | 1 | 1 | 0.0 → 0.2 → 0.4 | base RTF > 0.47 |
| `light` | 2 | 1 | 0.0 → 0.2 → 0.4 | base RTF 0.28–0.47 |
| `balanced` | 3 | 1 | 0.0 → 0.2 → 0.4 → 0.6 | base RTF 0.19–0.28 |
| `accurate` | 5 | 3 | 0.0 → 0.2 → 0.4 | base RTF < 0.19 |

`best_of` applies only when temperature > 0 (sampling fallback). Beam search at temperature 0 ignores it.

---

## Benchmark & Hardware

The benchmark runs automatically on first `install.py`. To re-run:

```bash
visper-benchmark                  # smart mode — all candidates, accurate RTF (~3–5 min)
visper-benchmark --fast           # primary device only (~60s)
visper-benchmark --quick          # heuristic only, no inference (instant)
visper-benchmark --force          # re-run even if results exist
visper-help                       # full command reference for all visper-* commands
```

**Fallback chain:** If the primary device fails at runtime (OOM, driver crash), the engine automatically tries the next device: `CUDA → OpenVINO HETERO (iGPU+CPU) → OpenVINO iGPU → OpenVINO CPU → CT2 CPU`. Each fallback's RTF is measured and cached on first use.

**Skip the benchmark entirely** — set `skip_benchmark: true` in `config.yaml` together with `force_device` and `force_compute_type`.

### Reference hardware results (Intel i5-1135G7 / MX350 / Iris Xe / 16GB)

| Device | Short (5s) | Medium (20s) | Long (45s) | Extended (90s) |
|--------|------------|--------------|------------|----------------|
| CUDA int8_float32 | RTF 0.55 | RTF 0.18 | RTF 0.16 | RTF 0.13 |
| OpenVINO HETERO iGPU+CPU | RTF 0.78 | RTF 0.24 | RTF 0.23 | RTF 0.23 |
| OpenVINO iGPU only | RTF 0.79 | RTF 0.26 | RTF 0.23 | RTF 0.23 |
| OpenVINO CPU | fail (1.40) | RTF 0.39 | RTF 0.37 | RTF 0.39 |
| CT2 CPU int8 | fail (2.57) | RTF 0.71 | RTF 0.63 | RTF 0.50 |

RTF budget = 0.85 (1.0 = real-time). WER = 0.179, CER = 0.084 on 221 Hebrew files (CUDA accurate tier, VAD on).

---

## Project Structure

```
start.bat                    ← Windows launcher: setup + server + browser (double-click)
update.bat                   ← Windows updater: git pull + pip install (double-click)
install.py                   ← first-run setup: installs deps, downloads model, runs benchmark
web/index.html               ← web UI (served by visper-server)
local_stt_he/benchmark.py    ← hardware detection, candidate selection, fallback chain
local_stt_he/resource.py     ← resource profile enforcement (threads, GPU, VRAM guard, priority)
local_stt_he/params.py       ← Whisper parameter selection per bucket/tier
local_stt_he/model_router.py ← single-slot language-based model manager; swaps on language change
local_stt_he/transcriber.py  ← dispatches to faster-whisper or openvino_genai; audio pre-processing
local_stt_he/api.py          ← public API: transcribe(), stream_transcribe(), transcribe_chunked()
local_stt_he/streamer.py     ← VAD-gated live transcription with sliding window overlap
local_stt_he/postprocess.py  ← text normalization (Hebrew + language-neutral)
local_stt_he/constants.py    ← audio constants (SAMPLE_RATE, CHANNELS, BLOCK_SIZE, DTYPE)
local_stt_he/server.py       ← FastAPI server
transcribe_file.py            ← CLI offline transcription
transcribe_live.py            ← CLI live/streaming transcription
run_benchmark.py              ← benchmark entry point
config.yaml                   ← user-tunable parameters (committed)
benchmark_results.json        ← auto-generated, never hand-edited (gitignored)
records/                      ← Hebrew audio files used as benchmark inputs
tests/                        ← standalone hardware validation scripts
```

---

## Hardware Validation Scripts

Standalone scripts for benchmarking individual backends. Not unit tests — run independently.

```bash
python tests/test_cpu.py          # CPU thread configs
python tests/test_gpu.py          # CUDA compute types
python tests/test_openvino.py     # Intel Iris Xe via OpenVINO
python tests/test_local_config.py # full local config sweep
```

---

## Roadmap

- [ ] **Speaker diarization** — label segments by speaker (Speaker 1, Speaker 2) using pyannote.audio
- [ ] **Live diarization** — post-session speaker labeling for recorded sessions

---

## Contributing

See [ARCHITECTURE.md](ARCHITECTURE.md) for an explanation of component boundaries and key design decisions. Bug reports and PRs welcome.
