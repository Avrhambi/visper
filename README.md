# Hebrew STT Engine

Local Hebrew speech-to-text engine built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Self-benchmarking, self-configuring, no GUI, no cloud.

Model: [`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2)

---

## Requirements

- Python 3.10+
- ffmpeg on PATH (required for MP3/MP4/M4A; WAV works without it)
- Tested hardware: Intel i5-1135G7, NVIDIA MX350 (2GB), Intel Iris Xe, 16GB RAM, Windows

---

## Setup

```bash
# 1. (Optional) Set a Hugging Face token if the model repo is gated
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...

# 2. Install dependencies and run the hardware benchmark
python setup.py
```

Installs dependencies, detects GPU, and runs the hardware benchmark once.

> **Note:** `benchmark_results.json` and `.env` are gitignored — never commit them.
> `.env.example` is the safe template to commit instead.

---

## Benchmark

The benchmark profiles your hardware and writes `benchmark_results.json`. It runs automatically on first use, or manually:

```bash
python run_benchmark.py              # smart mode (default) — ~4-6 candidates, (slow, but accurate)
python run_benchmark.py --force      # re-run even if results already exist 
python run_benchmark.py --quick      # heuristic only, no inference, instant (fast)
python run_benchmark.py --full       # exhaustive — all compute types × thread counts (very slow, most accurate)
```

**Smart mode** tests only the most promising configs for your hardware (1-2 CPU compute types × 2 thread counts), making it 3-5× faster than exhaustive mode. The result is the same winner in virtually all cases.

**Skipping the benchmark entirely** — set `skip_benchmark: true` in `config.yaml` together with `force_device` and `force_compute_type` to bypass benchmarking completely.

---

## Offline Transcription

### CLI

```bash
python transcribe_file.py audio.mp3
python transcribe_file.py audio.wav --output srt
python transcribe_file.py audio.mp3 --output json
python transcribe_file.py audio.mp3 --no-file       # print only, no file written
python transcribe_file.py audio.mp3 --clip          # copy to clipboard
python transcribe_file.py audio.mp3 --progress      # print each segment to stderr as decoded
python transcribe_file.py *.wav                     # batch mode
```

### Python API

```python
from stt_he import transcribe

text = transcribe("audio.mp3")          # auto-detects duration bucket
text = transcribe("audio.wav", bucket="long")
```

For progress feedback on long files, use `transcribe_chunked()`:

```python
from core.api import transcribe_chunked

def on_segment(seg):
    print(f"[{seg['start']:.1f}s] {seg['text'].strip()}")

text = transcribe_chunked("long_recording.mp3", on_segment=on_segment)
```

Install as editable package first to use from other projects:

```bash
cd /path/to/repo
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
Default format can be set in `config.yaml` (`output_format: "srt"`).

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

---

## Configuration (`config.yaml`)

### Resource profile

```yaml
resource_profile: "foreground"   # foreground | background | minimal
```

| Profile | Threads | Priority | GPU | Model unload |
|---|---|---|---|---|
| `foreground` | 100% of benchmark result | normal | yes | never |
| `background` | 50% | low | yes | after 60s idle |
| `minimal` | 25% (max 2) | low | no (CPU only) | after 30s idle |

### Accuracy mode

```yaml
accuracy_mode: "auto"   # auto | fast | balanced | accurate
```

| Tier | beam_size | best_of | temperature | Auto-selected when |
|---|---|---|---|---|
| `fast` | 1 | 1 | 0.0 | base RTF > 0.47 |
| `balanced` | 3 | 1 | 0.0 | base RTF 0.19–0.47 |
| `accurate` | 5 | 3 | 0.2 | base RTF < 0.19 |

Per-bucket overrides:

```yaml
bucket_accuracy_overrides:
  streaming: "fast"
  long:      "accurate"
  extended:  "accurate"
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
vad_filter: true              # Silero VAD
vad_min_silence_ms: 300       # silence duration that triggers a chunk emit
noise_calibration_seconds: 1.5  # record ambient noise at live session start to set VAD threshold; 0 = disabled
max_chunk_seconds: 28         # max chunk before forced emit in streaming
output_format: "txt"          # default output format: txt | srt | json
idle_unload_seconds: 0        # 0 = never unload model
max_cpu_threads: 0            # 0 = no cap
igpu_preference_margin: 0.20  # prefer Intel iGPU if RTF within 20% of best
confidence_retry_enabled: false  # retry at next accuracy tier if avg log-prob below threshold
```

---

## Hardware Validation Scripts

Standalone scripts that benchmark individual backends. Run independently, do not modify project files.

```bash
python tests/test_cpu.py          # CPU thread configs + mic test
python tests/test_gpu.py          # CUDA compute types on MX350
python tests/test_openvino.py     # Intel Iris Xe via OpenVINO
python tests/test_benchmark_full.py  # exhaustive benchmark (all combinations)
```

---

## Project Structure

```
core/benchmark.py        ← hardware detection, candidate selection, config selection
core/resource.py         ← resource profile enforcement (threads, GPU, VRAM guard, priority)
core/params.py           ← Whisper parameter selection per bucket/tier
core/transcriber.py      ← dispatches to faster-whisper or openvino_genai
core/api.py              ← public API: transcribe(), stream_transcribe(), transcribe_chunked()
core/streamer.py         ← VAD-gated live chunked transcription (with noise calibration)
core/postprocess.py      ← Hebrew text normalization (nikud stripping, Gershayim quotes)
core/constants.py        ← audio constants (SAMPLE_RATE, CHANNELS, BLOCK_SIZE, DTYPE)
transcribe_file.py       ← CLI offline transcription
transcribe_live.py       ← CLI live/streaming transcription
run_benchmark.py         ← benchmark entry point
config.yaml              ← user-tunable parameters (committed)
.env.example             ← environment variable template (committed)
.env                     ← secrets / local overrides (gitignored)
benchmark_results.json   ← auto-generated, never hand-edited (gitignored)
.venvs/                  ← device-specific venvs (gitignored, managed automatically)
records/                 ← Hebrew audio files used as benchmark inputs
tests/                   ← standalone hardware validation scripts
```
