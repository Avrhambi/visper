# Hebrew STT Engine

Local Hebrew speech-to-text engine built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Self-benchmarking, self-configuring, no GUI, no cloud.

Model: [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)

---

## Requirements

- Python 3.10+
- ffmpeg on PATH (required for MP3/MP4/M4A; WAV works without it)
- Windows / Linux / macOS
- GPU optional — runs on CPU, CUDA, or Intel iGPU (OpenVINO). Hardware is auto-detected.

---

## Setup

```bash
# 1. (Optional) Set a Hugging Face token if the model repo is gated
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...

# 2. Install dependencies and auto-configure for your hardware
python setup.py
```

`setup.py` installs dependencies, detects your hardware, and runs a fast benchmark that measures your primary device's real RTF (~60s). No manual configuration needed.

> **Note:** `benchmark_results.json` and `.env` are gitignored — never commit them.

---

## Benchmark

The benchmark profiles your hardware and writes `benchmark_results.json`. It runs automatically on first setup, or manually:

```bash
python run_benchmark.py              # smart mode (default) — all candidates, accurate RTF
python run_benchmark.py --fast       # fast mode — primary device only, ~60s
python run_benchmark.py --quick      # heuristic only — no inference, instant (no RTF)
python run_benchmark.py --full       # exhaustive — all compute types x thread counts
python run_benchmark.py --force      # re-run even if results already exist
```

| Mode | What it measures | Time |
|---|---|---|
| `--fast` | Primary device RTF only. Fallback chain set by rules, RTF probed lazily on first use. | ~60s |
| smart (default) | All promising candidates per backend. | ~3-5 min |
| `--full` | Every compute type x thread count combination. | ~15-20 min |
| `--quick` | No inference — derives config from hardware specs alone. | instant |

**Fallback chain:** If your primary device fails at runtime (e.g. GPU driver crash, OOM), the engine automatically tries the next device in the fallback chain: `CUDA → OpenVINO HETERO (iGPU+CPU) → OpenVINO iGPU → OpenVINO CPU → CT2 CPU`. The first time a fallback device is used, its RTF is measured and cached for accurate tier selection on future calls.

**Skipping the benchmark entirely** — set `skip_benchmark: true` in `config.yaml` together with `force_device` and `force_compute_type`.

---

## Offline Transcription

### CLI

```bash
python transcribe_file.py audio.mp3
python transcribe_file.py audio.wav --output srt
python transcribe_file.py audio.mp3 --output json
python transcribe_file.py audio.mp3 --no-file       # print only, no file written
python transcribe_file.py audio.mp3 --clip          # copy to clipboard
python transcribe_file.py audio.mp3 --progress      # print each segment as decoded
python transcribe_file.py *.wav                     # batch mode
```

### Python API

```python
from stt_he import transcribe

text = transcribe("audio.mp3")               # auto-detects duration bucket
text = transcribe("audio.wav", bucket="long")
```

For progress feedback on long files:

```python
from core.api import transcribe_chunked

def on_segment(seg):
    print(f"[{seg['start']:.1f}s] {seg['text'].strip()}")

text = transcribe_chunked("long_recording.mp3", on_segment=on_segment)
```

Install as editable package to use from other projects:

```bash
pip install -e .
```

Or without installing:

```python
import sys
sys.path.insert(0, "/path/to/repo")
from core.api import transcribe
```

### Input formats

Any format ffmpeg can decode: `.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`, `.wma`, `.webm`, and more.
Also accepts a `numpy.ndarray` (float32, 16 kHz, mono) directly.

### Output formats

| Format | Contents | CLI flag |
|---|---|---|
| `txt` (default) | Plain text | `--output txt` |
| `srt` | Timestamped subtitles | `--output srt` |
| `json` | Text + segments + RTF + timing stats | `--output json` |

Output file is written alongside the input (e.g. `audio.mp3` → `audio.txt`). Use `--no-file` to suppress.

---

## Live / Streaming Transcription

### CLI

```bash
python transcribe_live.py                         # microphone
python transcribe_live.py --file audio.mp3        # file streaming mode
python transcribe_live.py --output result.txt     # save accumulated result
python transcribe_live.py --clip                  # copy to clipboard on stop
```

Press `Ctrl+C` to stop.

### Python API

```python
from stt_he import stream_transcribe

def on_transcript(text, is_final):
    if is_final:
        print("FINAL:", text)   # silence-gated, complete utterance
    else:
        print("...", text)      # mid-speech forced emit

stream_transcribe(on_transcript)                 # microphone
stream_transcribe(on_transcript, "audio.mp3")    # file
```

`is_final=True` — emitted on silence (complete utterance).
`is_final=False` — forced emit when chunk exceeds `max_chunk_seconds` mid-speech.

### Sliding window overlap

Each chunk re-includes the last `overlap_seconds` (default 2s) of the previous chunk. This ensures words at chunk boundaries are always transcribed in context rather than cut off. Duplicate words introduced by the overlap are detected and stripped before the callback is called — the output stream never shows repeated text.

---

## Configuration (`config.yaml`)

### Resource profile

```yaml
resource_profile: "foreground"   # foreground | background | minimal
```

| Profile | Threads | Priority | GPU | Model unload |
|---|---|---|---|---|
| `foreground` | benchmark result | normal | yes | never |
| `background` | 50% | low | yes | after 60s idle |
| `minimal` | 25% (max 2) | low | no (CPU only) | after 30s idle |

### Accuracy mode

```yaml
accuracy_mode: "auto"   # auto | fast | light | balanced | accurate
```

| Tier | beam_size | best_of | temperature | Auto-selected when |
|---|---|---|---|---|
| `fast` | 1 | 1 | 0.0 | base RTF > 0.47 |
| `light` | 2 | 1 | 0.0 | base RTF 0.28–0.47 |
| `balanced` | 3 | 1 | 0.0 | base RTF 0.19–0.28 |
| `accurate` | 5 | 3 | 0.2 | base RTF < 0.19 |

Per-bucket overrides:

```yaml
bucket_accuracy_overrides:
  streaming: "fast"
  long:      "accurate"
  extended:  "accurate"
```

### Streaming

```yaml
max_chunk_seconds: 28       # hard max chunk length before forced emit
overlap_seconds: 2.0        # sliding window overlap for boundary accuracy; 0 = disabled
stream_flush_on_silence: true
```

### VAD

```yaml
vad_filter: true
vad_min_silence_ms: 300              # silence duration that triggers chunk emit
vad_speech_pad_ms: 200               # padding added around speech segments
noise_calibration_seconds: 1.5       # ambient noise calibration at session start; 0 = disabled
```

### Benchmark mode

```yaml
benchmark_mode: "smart"   # smart | full
```

### Manual hardware config (skip benchmark)

```yaml
skip_benchmark: true
force_device: "cpu"           # cpu | cuda | openvino
force_compute_type: "int8"    # int8 | float16 | int8_float16 | int8_float32
force_cpu_threads: 4          # 0 = auto
```

### Other options

```yaml
output_format: "txt"             # default output format: txt | srt | json
idle_unload_seconds: 0           # 0 = never unload model
max_cpu_threads: 0               # 0 = no cap
max_ram_mb: 0                    # 0 = no limit
max_vram_mb: 0                   # 0 = no limit
igpu_preference_margin: 0.05     # prefer Intel iGPU if RTF within 5% of CUDA winner
confidence_retry_enabled: false  # retry at next accuracy tier if confidence is low
```

---

## Benchmark Results (Reference Hardware)

Measured on Intel i5-1135G7 / NVIDIA MX350 (2GB) / Intel Iris Xe / 16GB RAM:

| Device | Short (5s) | Medium (20s) | Long (45s) | Extended (90s) |
|---|---|---|---|---|
| CUDA int8_float32 | RTF 0.55 | RTF 0.18 | RTF 0.16 | RTF 0.13 |
| OpenVINO HETERO iGPU+CPU | RTF 0.78 | RTF 0.24 | RTF 0.23 | RTF 0.23 |
| OpenVINO iGPU only | RTF 0.79 | RTF 0.26 | RTF 0.23 | RTF 0.23 |
| OpenVINO CPU | fail (1.40) | RTF 0.39 | RTF 0.37 | RTF 0.39 |
| CT2 CPU int8 | fail (2.57) | RTF 0.71 | RTF 0.63 | RTF 0.50 |

RTF budget = 0.85 (1.0 = real-time). WER = 0.179, CER = 0.084 on 221 Hebrew files (CUDA accurate tier, VAD on).

---

## Hardware Validation Scripts

Standalone scripts that benchmark individual backends on the reference hardware. Results and insights written to `tests/`.

```bash
python tests/test_cpu.py          # CPU thread configs
python tests/test_gpu.py          # CUDA compute types
python tests/test_openvino.py     # Intel Iris Xe via OpenVINO
python tests/test_local_config.py # full local config sweep
```

---

## Project Structure

```
core/benchmark.py        <- hardware detection, candidate selection, fallback chain
core/resource.py         <- resource profile enforcement (threads, GPU, VRAM guard, priority)
core/params.py           <- Whisper parameter selection per bucket/tier
core/transcriber.py      <- dispatches to faster-whisper or openvino_genai; walks fallback chain
core/api.py              <- public API: transcribe(), stream_transcribe(), transcribe_chunked()
core/streamer.py         <- VAD-gated live transcription with sliding window overlap
core/postprocess.py      <- Hebrew text normalization
core/constants.py        <- audio constants (SAMPLE_RATE, CHANNELS, BLOCK_SIZE, DTYPE)
transcribe_file.py       <- CLI offline transcription
transcribe_live.py       <- CLI live/streaming transcription
run_benchmark.py         <- benchmark entry point
config.yaml              <- user-tunable parameters (committed)
benchmark_results.json   <- auto-generated, never hand-edited (gitignored)
models_ov/               <- OpenVINO model export (gitignored, generated on first OV use)
.venvs/                  <- device-specific venvs (gitignored, managed automatically)
records/                 <- Hebrew audio files used as benchmark inputs
tests/                   <- hardware validation scripts and benchmark insights
```
