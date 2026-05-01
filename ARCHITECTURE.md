# Architecture

## Layered Data Flow

```
records/ (benchmark audio)
        │
        ▼
local_stt_he/benchmark.py ──writes──► benchmark_results.json
        │
        │ get_best_config(bucket)
        ▼
local_stt_he/resource.py  ◄── config.yaml (resource_profile)
(threads, priority, VRAM guard)
        │
        ▼
local_stt_he/model_router.py ◄── config.yaml (models, force_model)
(single-slot model cache; swaps on language change)
        │
        ▼
local_stt_he/transcriber.py  ◄── local_stt_he/params.py ◄── config.yaml (accuracy_mode)
(faster-whisper / openvino_genai; normalize → highpass → denoise pre-processing chain)
        │
        ▼
local_stt_he/postprocess.py
(text normalization)
        │
        ▼
TranscriptResult {text, segments, rtf, tier_used, ...}
        │
        ▼
local_stt_he/api.py: transcribe() / stream_transcribe() / transcribe_chunked()
        │
        ├── transcribe_file.py  (CLI: visper-file)
        ├── transcribe_live.py  (CLI: visper-live)
        └── server.py           (FastAPI: visper-server)
```

---

## Component Roles

**`local_stt_he/benchmark.py`** — The sole hardware-awareness layer. Scans `records/` for representative audio files (one per duration bucket), auto-detects all candidate backends (CPU compute types × thread counts, CUDA, OpenVINO variants), and runs one warm-up + one timed pass per `(candidate × bucket)` pair. Also runs a dedicated streaming trial (5 consecutive calls, median RTF of calls 2–5) to measure model-loaded per-call latency. Writes `benchmark_results.json`. `get_best_config(bucket)` is the only read path — all other modules call it when they need hardware settings.

**`local_stt_he/resource.py`** — Enforces `resource_profile` from `config.yaml`. Translates `foreground` / `background` / `minimal` into thread count fractions, process priority changes, and idle model-unload timers. The only file that reads `resource_profile` or sets process priority.

**`local_stt_he/params.py`** — Owns every Whisper inference parameter. Selects one of four accuracy tiers (`fast` / `light` / `balanced` / `accurate`) based on the RTF headroom measured by the benchmark and the `accuracy_mode` from `config.yaml`. Returns a `WhisperParams` dataclass that `Transcriber` converts to transcribe kwargs. Temperature is a fallback ladder (tuple) — Whisper retries at the next value when quality checks fail. No other module sets `beam_size`, `temperature`, `condition_on_prev_text`, etc.

**`local_stt_he/model_router.py`** — Holds at most one `Transcriber` in memory at a time. Resolves the correct model ID from the `models` map in `config.yaml` based on the requested language, then loads or swaps the `Transcriber` as needed. Thread-safe — concurrent requests for different languages serialize on the swap lock. `force_model` in `config.yaml` bypasses routing entirely.

**`local_stt_he/transcriber.py`** — The only file that imports `faster_whisper` or `openvino_genai`. Accepts a hardware config dict (including `model_id`), loads the model, and exposes `transcribe(source, bucket)`. Walks the fallback chain on load failure or OOM. Runs an optional audio pre-processing chain before inference — volume normalization, 80 Hz high-pass filter, and noise reduction — in that order, skipped for streaming. Resolves Whisper params per-call via `params.get_params()`.

**`local_stt_he/api.py`** — Thin, stable public interface: `transcribe()`, `stream_transcribe()`, `transcribe_chunked()`. Handles duration detection and bucket resolution. Routes to the correct `Transcriber` via `ModelRouter`. These signatures are frozen — external callers depend on them.

**`local_stt_he/streamer.py`** — Implements the two-thread live pipeline. Producer thread reads mic or file chunks, applies RMS energy VAD, and enqueues audio to `queue.Queue(maxsize=4)`. Consumer thread calls `Transcriber.transcribe(chunk, bucket="streaming")` and invokes `on_transcript(text, is_final)`. Includes sliding window overlap and boundary deduplication.

**`local_stt_he/postprocess.py`** — Hebrew-specific text normalization applied after Whisper output: diacritics removal, typographic quote substitution, script boundary spacing, trailing punctuation cleanup, duplicate word collapse.

**`local_stt_he/worker.py`** — Optional subprocess worker. Spawned by `Transcriber` when a device-specific venv is needed. Communicates over stdin/stdout with a JSON line protocol. Lets the engine run OpenVINO or CUDA in a venv with a different Python/package set than the caller.

---

## Key Design Decisions

### No hardcoded hardware config

`DEVICE`, `COMPUTE_TYPE`, `CPU_THREADS`, `NUM_WORKERS` never appear as Python literals in the codebase. They exist only as keys in `benchmark_results.json` and as values in config dicts passed between functions. Any hardware change requires only re-running the benchmark — not a code change.

### Params resolved per-call, not at model load

`Transcriber` calls `params.get_params(bucket, hw_config)` each time `transcribe()` is called. A single loaded model can serve `short` audio at `beam_size=1` and `long` audio at `beam_size=5` without a reload. `accuracy_mode` changes in `config.yaml` take effect immediately on the next call.

### Failed candidates are recorded, not silently dropped

`benchmark_results.json` always stores `"status": "failed"` with an `"error"` string for failing candidates. This makes it possible to diagnose why a backend wasn't chosen and prevents silent regressions after driver or library updates.

### `condition_on_prev_text` is always False for streaming and short buckets

Enforced in `params.get_params()`, not user-overridable. For streaming, previous context is managed by the sliding window overlap — not Whisper's internal conditioning. For short clips there is no previous context to condition on.

### `LiveStreamer` always uses `get_best_config("streaming")`

The streaming trial measures repeated per-call latency with the model already loaded — the correct metric for live use. The `short` bucket config measures one-shot latency from a cold model, which overestimates available headroom.

### Fallback chain: lazy RTF measurement

The benchmark profiles the primary device thoroughly. Fallback devices get their RTF measured the first time they're actually used, then cached. This avoids benchmarking all devices upfront (slow) while still selecting the right accuracy tier when a fallback activates.

---

## LiveStreamer Pipeline

```
Microphone or file
        │
        ▼
Producer thread
  ├── calibrate noise floor for 1.5s at session start (RMS baseline)
  ├── buffer audio until:
  │     silence ≥ vad_min_silence_ms, OR buffer > max_chunk_seconds
  ├── prepend last overlap_seconds of previous chunk (sliding window)
  └── queue.Queue(maxsize=4)
          │ drop oldest + warn after 3 consecutive drops
          ▼
Consumer thread
  ├── Transcriber.transcribe(chunk, bucket="streaming")
  ├── strip overlap duplicates at chunk boundary
  └── on_transcript(text, is_final)
        ├── is_final=True  → silence-gated utterance
        └── is_final=False → forced emit (chunk exceeded max_chunk_seconds)
```

Idle model-unload timer and memory monitor run as daemon threads, both controlled by `config.yaml` (`idle_unload_seconds`, `max_ram_mb`).

---

## Benchmark Algorithm

1. Scan `records/` → categorise files into four buckets: `short` (<10s), `medium` (10–30s), `long` (30–60s), `extended` (>60s). Pick the file closest to the bucket midpoint (5s / 20s / 45s / 90s) as the representative.

2. Build candidate list: CPU (all supported compute types × selected thread counts), CUDA (if GPU found), OpenVINO variants (HETERO:GPU,CPU / GPU / CPU) if Intel iGPU found.

3. For each `(candidate × bucket)` pair: one warm-up pass (discarded) + one timed pass → RTF. Early stop: `elite` if RTF < 0.15; `slow_skip` if warm-up RTF > 2.0.

4. Run streaming trial: load model once, 5 consecutive calls on the same short clip, record median RTF of calls 2–5.

5. Write all results (including failures) to `benchmark_results.json` with timestamp.

6. `get_best_config(bucket)` reads JSON, applies `force_device`/`force_compute_type` overrides from `config.yaml`, falls back to the nearest available bucket if the requested one has no results.
