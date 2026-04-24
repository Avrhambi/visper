# Hebrew STT Engine

Offline Hebrew speech-to-text on your own hardware. No cloud, no GUI. Self-benchmarks and configures itself on first run.

Model: [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)

---

## Quick Start

```bash
git clone https://github.com/Avrhambi/local-whisper-he && cd local-whisper-he
python install.py           # installs deps, downloads model (~1.5 GB once), runs benchmark
stt-file audio.mp3          # or: python transcribe_file.py audio.mp3
```

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org) on PATH (WAV files work without it).

> **HuggingFace token:** If the model repo is gated, copy `.env.example` to `.env` and set `HF_TOKEN=hf_...` before running `install.py`.

---

## What It Does

Transcribes Hebrew audio files and microphone input using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Auto-detects your hardware (CPU / CUDA / Intel iGPU via OpenVINO) and benchmarks it once to pick the best inference backend and accuracy tier. No manual configuration needed — results are cached in `benchmark_results.json`.

---

## CLI

### Offline file transcription

```bash
stt-file audio.mp3                     # transcribe → write audio.txt
stt-file audio.wav --output srt        # SRT subtitles
stt-file audio.mp3 --output json       # JSON with segments, RTF, confidence scores
stt-file audio.mp3 --no-file --clip    # print + copy to clipboard, no file written
stt-file audio.mp3 --progress          # print each segment as it is decoded
stt-file *.wav                         # batch mode
```

### Live / microphone

```bash
stt-live                               # microphone transcription
stt-live --file audio.mp3             # file streaming mode
stt-live --output result.txt          # save accumulated transcript
stt-live --clip                       # copy to clipboard on Ctrl+C
```

If the CLI entry points aren't on PATH yet (before `pip install -e .`):

```bash
python transcribe_file.py audio.mp3
python transcribe_live.py
```

---

## Python API

Install as an editable package for use from other projects:

```bash
pip install -e .
```

```python
from core.api import transcribe, stream_transcribe, transcribe_chunked

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

## FastAPI Server

```bash
pip install -e ".[server]"
stt-server                    # starts on http://localhost:8000
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Device info and status |
| POST | `/transcribe` | Upload a file, get `{text, segments, rtf}` |
| POST | `/transcribe/stream` | Upload a file, get SSE stream of `{text, is_final}` events |

```bash
curl -F "file=@audio.mp3" http://localhost:8000/transcribe
curl -F "file=@audio.mp3" http://localhost:8000/transcribe/stream
curl http://localhost:8000/health
```

Every non-Python integration — Node, Go, mobile backends — speaks HTTP.

---

## Configuration

Edit `config.yaml` to adjust behavior. Key options:

| Key | Default | Options |
|-----|---------|---------|
| `accuracy_mode` | `auto` | `auto` / `fast` / `light` / `balanced` / `accurate` |
| `resource_profile` | `foreground` | `foreground` / `background` / `minimal` |
| `output_format` | `txt` | `txt` / `srt` / `json` |
| `vad_filter` | `true` | Enable Silero VAD |
| `max_chunk_seconds` | `28` | Max live chunk before forced emit |
| `confidence_retry_enabled` | `false` | Retry at next accuracy tier if confidence is low |

Full reference with all options is in `config.yaml`.

### Resource profiles

| Profile | Threads | Priority | GPU | Model unload |
|---------|---------|----------|-----|--------------|
| `foreground` | benchmark result | normal | yes | never |
| `background` | 50% | low | yes | after 60s idle |
| `minimal` | 25% (max 2) | low | no (CPU only) | after 30s idle |

### Accuracy tiers (when `accuracy_mode: auto`)

| Tier | beam_size | best_of | temperature | Auto-selected when |
|------|-----------|---------|-------------|-------------------|
| `fast` | 1 | 1 | 0.0 | base RTF > 0.47 |
| `light` | 2 | 1 | 0.0 | base RTF 0.28–0.47 |
| `balanced` | 3 | 1 | 0.0 | base RTF 0.19–0.28 |
| `accurate` | 5 | 3 | 0.2 | base RTF < 0.19 |

---

## Benchmark & Hardware

The benchmark runs automatically on first `install.py`. To re-run:

```bash
stt-benchmark                  # smart mode — all candidates, accurate RTF (~3–5 min)
stt-benchmark --fast           # primary device only (~60s)
stt-benchmark --quick          # heuristic only, no inference (instant)
stt-benchmark --force          # re-run even if results exist
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
core/benchmark.py      ← hardware detection, candidate selection, fallback chain
core/resource.py       ← resource profile enforcement (threads, GPU, VRAM guard, priority)
core/params.py         ← Whisper parameter selection per bucket/tier
core/transcriber.py    ← dispatches to faster-whisper or openvino_genai; walks fallback chain
core/api.py            ← public API: transcribe(), stream_transcribe(), transcribe_chunked()
core/streamer.py       ← VAD-gated live transcription with sliding window overlap
core/postprocess.py    ← Hebrew text normalization
core/constants.py      ← audio constants (SAMPLE_RATE, CHANNELS, BLOCK_SIZE, DTYPE)
transcribe_file.py     ← CLI offline transcription
transcribe_live.py     ← CLI live/streaming transcription
server.py              ← FastAPI server (pip install -e ".[server]")
run_benchmark.py       ← benchmark entry point
install.py             ← first-run setup: installs deps, downloads model, runs benchmark
config.yaml            ← user-tunable parameters (committed)
benchmark_results.json ← auto-generated, never hand-edited (gitignored)
records/               ← Hebrew audio files used as benchmark inputs
tests/                 ← standalone hardware validation scripts
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

## Contributing

See [ARCHITECTURE.md](ARCHITECTURE.md) for an explanation of component boundaries and key design decisions. Bug reports and PRs welcome.
