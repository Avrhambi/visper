# Visper

**Local, offline, Hebrew-first speech-to-text that measures its own hardware and configures itself.**

Visper transcribes Hebrew (and 11 other languages) entirely on the user's
machine — no cloud, no API keys, no audio leaving the box. On first run it
benchmarks every backend it can find (CUDA, Intel OpenVINO, Apple MLX, CPU),
records the real-time factor of each, and writes a config that pins the fastest
path per audio-length bucket. Every inference parameter afterwards is chosen
from that measurement, not guessed.

The defining engineering property: **no hardware constant is a source-code
literal.** Device, compute type, thread count, and accuracy tier live only in
`benchmark_results.json` and the config dicts passed between modules. Moving the
repo to a new machine is a re-benchmark, never a code change.

---

## 1. Problem

Off-the-shelf speech-to-text forces a choice: send audio to a cloud API
(privacy, cost, offline-hostile) or run [OpenAI Whisper][whisper] locally and
hand-tune a dozen knobs per machine. Hebrew makes it worse — stock
`whisper-large-v3` is middling on Hebrew, the good model
([ivrit.ai][ivrit]'s [fine-tune][ivrit-model]) is transcription-only and
translates poorly, and Hebrew RTL text needs script-boundary normalisation that
general tooling doesn't do.

Visper targets one user: someone who wants a **one-command local install** that
produces accurate Hebrew transcripts on whatever hardware they have — a CUDA
laptop, an Intel iGPU, an M-series Mac, or a plain CPU — without reading a
tuning guide.

Ship-readiness criteria ([`docs/design/ship-readiness.md`](docs/design/ship-readiness.md)):

- Every performance/accuracy number in this README comes from a committed
  harness (`visper-eval`, `visper-benchmark`) — [§5](#5-accuracy) and
  [§6](#6-performance) are real runs on the machine in [§6](#6-performance),
  not estimates.
- The default runtime (venv-worker subprocess) and the in-process path return
  identical results — same segments, same confidence-gated retry, same two-stage
  Hebrew translate. Two timing differences: per-segment callbacks fire live
  during decode in-process but replay in one batch after decode on the worker
  path, and a mid-file cancellation takes effect immediately in-process but only
  between requests on the worker path.
- `python -m build` produces a clean sdist and wheel (both carry the web UI);
  `pytest` is green (81 tests, local Python 3.14). CI runs the same suite on
  3.10–3.12 (below).
- `visper-server` binds `127.0.0.1` by default — the GPU is never exposed to the
  LAN or a browser on another origin without an explicit `--host` flag.

---

## 2. System Architecture & Flow

```
                        first run only
 records/*.mp3  ─────▶  visper/benchmark.py  ──writes──▶  benchmark_results.json
 (bucketed audio)       CPU×threads / CUDA / OpenVINO /       { best[bucket] =
                        MLX  ·  warm-up + timed pass            device, compute,
                        per (candidate × bucket)                rtf, tier, venv }
                                                                       │
 ┌─────────────────────────────────────────────────────────────────────┘
 │  get_best_config(bucket)
 ▼
visper/resource.py     ◀── config.yaml : resource_profile
 (thread fractions, process priority, idle-unload timers, RAM guard)
 │
 ▼
visper/model_router.py ◀── config.yaml : models{}   (language → model id)
 (single-slot cache — one Transcriber resident; swaps on language change)
 │
 ▼
visper/transcriber.py  ◀── visper/params.py ◀── config.yaml : accuracy_mode
 │   normalise → high-pass → denoise  (pre-proc chain, skipped for streaming)
 │   in-process  OR  venv-worker subprocess (JSON-line protocol over stdin/stdout)
 │   confidence-gated retry: re-decode at the next tier if mean logprob is low
 │
 ├── task=translate & language=he ──▶ visper/translate.py
 │      stage 1 result (Hebrew) ──▶ opus-mt-he-en (CTranslate2) ──▶ English segments
 │
 ▼
visper/postprocess.py  (Hebrew normalisation: diacritics, quotes, script spacing,
 │                       trailing-punct cleanup, repeat-word collapse)
 ▼
TranscriptResult { text, segments, audio_duration, rtf, tier_used, backend, he_text }
 │
 ├── visper/_cli.py        →  visper-file  /  visper-live
 ├── visper/server.py      →  FastAPI: POST /transcribe, /transcribe/stream (SSE),
 │                             WS /ws/live  — serves visper/web/index.html at /
 └── visper/api.py         →  transcribe() / stream_transcribe() / transcribe_chunked()
```

### End-to-end walk — `POST /transcribe` with a Hebrew MP3, `translate=1`

1. **`server.py:/transcribe`** streams the upload to a temp file (size-capped),
   resolves the duration bucket (`short <10s`, `medium <30s`, `long <60s`,
   `extended ≥60s`).
2. **`api.transcribe_chunked`** calls `_get_config(bucket)` →
   `benchmark.get_best_config(bucket)` reads `benchmark_results.json` and returns
   `{device, compute_type, venv_path, ...}` for the fastest measured path.
3. **`ModelRouter.get("he")`** resolves `he → ivrit-ai/whisper-large-v3-turbo-ct2`
   from `config.yaml:models`, and loads (or reuses) that one `Transcriber`.
4. **`Transcriber.transcribe(task="translate", language="he")`** hits the
   two-stage guard: it calls itself with `task="transcribe"` for the Hebrew
   decode (same bucket, same pinned tier). (On the in-process runtime each
   segment is also translated as it decodes, for a live English preview; the
   default venv-worker runtime replays the Hebrew segments in one batch after
   decode.)
5. The Hebrew decode runs in the **venv-worker subprocess** (the per-device venv
   the benchmark built — e.g. `.venvs/cuda` with the CUDA CTranslate2 build and
   NVIDIA libraries). Params — beam size, temperature ladder, VAD settings —
   come from `params.get_params_for_tier`. Low mean-logprob triggers one retry
   at the next tier.
6. **`translate.py`** SentencePiece-encodes each Hebrew segment, runs one
   `ctranslate2` `translate_batch`, decodes English. Model absent/broken →
   silently falls back to Whisper's own translate task.
7. **`postprocess.normalize_text`** cleans each English segment; `.text` is
   rebuilt from the final segments; `.he_text` keeps the Hebrew.
8. `server.py` streams each English segment as an SSE event, then a `final_text`
   event.

---

## 3. Tech Stack & Engineering Decisions

| Layer | Technology | Rationale & trade-offs |
|---|---|---|
| Hebrew ASR | [`ivrit-ai/whisper-large-v3-turbo-ct2`][ivrit-model] | Purpose-built Hebrew fine-tune of [OpenAI Whisper][whisper] `large-v3-turbo` by [ivrit.ai][ivrit] — the community-standard open model for Hebrew ASR; measured WER in [§5](#5-accuracy). Cost: transcription-only (no usable translate task), ~1.5 GB, no MLX build — Apple Silicon falls back to base [`large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo). |
| Inference runtime | [CTranslate2](https://github.com/OpenNMT/CTranslate2) (via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)) | int8 quantised, deterministic decode, low memory footprint. Cost: a second quantised model format, and its device-specific native builds motivate the per-device venv below. |
| Per-device isolation | venv-worker subprocess (`.venvs/<device>`) | Each accelerator has a heavy, conflicting native stack (CUDA + cuBLAS/cuDNN, or OpenVINO + optimum-intel + onnxruntime). The benchmark builds one venv per device and the decode runs there — the host env stays clean, and OpenVINO's hard Python 3.12 requirement doesn't pin the whole project. Communication is a JSON-line protocol over stdin/stdout. Cost: a process hop and a serialise per call; feature parity had to be re-implemented on the worker path. |
| he→en translation | [`Helsinki-NLP/opus-mt-tc-big-he-en`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-he-en) ([OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT)) → CTranslate2 int8 | Dedicated MT beats asking the ASR fine-tune to translate. Reuses the CT2 runtime already loaded (no torch/transformers at runtime; `sentencepiece` is the only added dependency). Cost: a ~210 MB model fetched from a GitHub release asset on first Hebrew→English use (SHA-256 pinned); `install.py` pre-fetches it. |
| Accelerators | CUDA · [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) · [Apple MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper) · CPU | Cover every consumer machine. The benchmark picks per-bucket; a runtime fallback chain (`CUDA → OpenVINO HETERO → iGPU → CPU → CT2 CPU`) recovers from a device that fails at load. Cost: four code paths behind one `Transcriber`. |
| API server | [FastAPI](https://fastapi.tiangolo.com) (`StreamingResponse` for SSE) | Non-blocking SSE/WebSocket streaming with minimal boilerplate; accepts the async-debugging overhead over Flask. Binds `127.0.0.1` by default (`--host 0.0.0.0` opt-in) so the GPU is never LAN-exposed accidentally. |
| Web UI | Single `visper/web/index.html`, vendored [`lucide`](https://lucide.dev) | Zero build step, served by the API at `/`. Shipped inside the package so a `pip install` serves it too. Icons vendored (not a CDN) to keep the "no cloud dependency" claim literally true. |
| Config | One YAML + one loader (`visper/_config.py`) | A single cached parser; `config.yaml` ships inside the package (`visper/config.yaml`) so a non-editable install behaves identically. Replaced 8 ad-hoc parsers that had drifted. |
| Accuracy eval | [`jiwer`](https://github.com/jitsi/jiwer), local `visper-eval` | WER/CER on real corpora on the user's machine — no cloud eval, no Colab. Both sides run through the shipped normaliser, then a symmetric punctuation strip for scoring. |
| Tests / CI | `pytest`, GitHub Actions | Unit suite on pure logic (params, buckets, config, postprocess, eval scoring, translation wrapper) — no model download in CI. |

### Why a subprocess worker instead of one fat environment

The accelerator backends don't co-exist cleanly in one venv: the CUDA build of
CTranslate2 drags in NVIDIA's cuBLAS/cuDNN wheels, OpenVINO pulls
`optimum-intel` + `onnxruntime` and only supports Python ≤ 3.12, and the
native-wheel stack in general trails new CPython by a release or two. Rather
than pin the whole project to the lowest common denominator, the benchmark
builds a dedicated venv per device (`.venvs/cuda`, `.venvs/openvino`, …) and the
decode runs in whichever one won. The host stays on current Python with a clean
dependency set. The worker is spawned once and kept warm; the cost is one
`np.save` + JSON round-trip per call, negligible against decode time.

---

## 4. Resilience & Error Handling Patterns

- **Device fallback chain** (`transcriber.py`) — on model-load failure or CUDA
  OOM, walk `CUDA → OpenVINO HETERO → OpenVINO iGPU → OpenVINO CPU → CT2 CPU`.
  The fallback device's RTF is measured on first use and cached, so the accuracy
  tier is still chosen correctly after a fallback.
- **Failed benchmark candidates are recorded, not dropped** — every candidate
  writes `status: ok|failed` + an error string to `benchmark_results.json`, so a
  regression after a driver update is diagnosable instead of silent.
- **Confidence-gated retry** (`params.py` + `transcriber.py`) — if a decode's
  duration-weighted mean logprob is below the tier threshold, re-decode once at
  the next tier up. Applies on both the in-process and venv-worker paths.
- **Translation degrades, never fails** (`translate.py`) — a missing/corrupt MT
  model, an un-importable dependency, or a runtime error in the translate pass
  all fall back to returning the Hebrew transcript (or Whisper's own translate
  task). A `task=translate` request never 500s because of stage 2.
- **Model integrity on download** — the he→en asset is fetched over HTTPS with a
  30 s timeout, checked against a pinned SHA-256, and extracted with a
  path-traversal + symlink guard (`tarfile` `data` filter on 3.12+).
- **`/health` reports capability, not just liveness** — it advertises which
  languages can translate, stops advertising Hebrew translation the moment the
  MT path is known broken, and flags (`he_en_pending_download`) when he→en works
  but its model hasn't been fetched yet, so the UI never silently under-delivers.
- **venv creation survives Windows AV locks** (`venv_manager.py`) — Defender
  briefly locks a freshly-copied `python.exe`; venv creation retries with
  backoff and polls for real deletion before recreating.
  ([`docs/lessons.md`](docs/lessons.md).)
- **Streaming back-pressure** (`streamer.py`) — a bounded `queue.Queue(maxsize=4)`
  between the mic producer and the decode consumer; drop-oldest + warn after 3
  consecutive drops rather than unbounded latency growth.

**Known limitation:** the venv-worker subprocess is spawned once and not
respawned — if it dies mid-session, subsequent requests fail until
`visper-server` is restarted. The in-process fallback runtime has no such
single point of failure.

---

## 5. Accuracy

Measured locally with `visper-eval` (`jiwer`), model
`ivrit-ai/whisper-large-v3-turbo-ct2` at the `balanced` tier, over three Hebrew
corpora of different genres — reported **separately**, never pooled. Both
reference and hypothesis pass through the shipped
`visper.postprocess.normalize_text`, then a symmetric lowercase + punctuation
strip for the score (the reference corpora carry no punctuation, so charging the
model for a correctly-placed comma would be an artifact — see
[`docs/lessons.md`](docs/lessons.md)).
Median is given alongside the mean because a few misaligned or truncated pairs
skew the mean on the harder sets. Files scored are a random sample, transcribed
end-to-end with internal VAD chunking. The full run — every reference/hypothesis
pair — is committed at
[`docs/benchmarks/eval-he-balanced.json`](docs/benchmarks/eval-he-balanced.json).

| Corpus | Files scored | Median clip | WER (mean) | WER (median) | CER | WER min / max | What it is |
|---|--:|--:|--:|--:|--:|:--|---|
| Knesset (formal) | 25 / 104 | 1.5 min | **0.141** | 0.123 | 0.082 | 0.04 / 0.30 | Parliamentary speech (`speaker_session_start_end` clips) — clean audio, formal register. The headline case. |
| Longer-form | 25 / 792 | 26 s | 0.245 | 0.217 | 0.124 | 0.00 / 0.76 | Assorted Hebrew speech clips. Mean is dragged by two cases (one early-stopped decode, one foreign-word-contaminated reference); the median is the honest centre. |
| CoSIH (spontaneous) | 15 / 15 | 4.8 min | 0.575 | 0.612 | 0.439 | 0.26 / 0.78 | Spontaneous-conversation *linguistics* corpus — fillers, overlap, phonetic transcription conventions. A limitations data point, not a benchmark; ASR on this genre is hard for any model. |

Reproduce the table in seconds (no audio needed):
`visper-eval --rescore docs/benchmarks/eval-he-balanced.json`. Full re-run:
`visper-eval <corpus-dir> --tier balanced --out eval.md`.

---

## 6. Performance

RTF (real-time factor) — wall-clock ÷ audio duration; **lower is faster**,
`1.0` = real-time. Measured by `visper-benchmark`: one warm-up + one timed pass
per `(backend × bucket)`, plus a 5-call streaming trial for model-loaded
per-call latency. `visper-benchmark --report` prints this table from
`benchmark_results.json`; the reference machine's snapshot is committed at
[`docs/benchmarks/benchmark-i5-1135g7-mx350.json`](docs/benchmarks/benchmark-i5-1135g7-mx350.json).

**Measured on:** 11th Gen Intel Core i5-1135G7 @ 2.40 GHz · NVIDIA GeForce MX350
(2 GB VRAM) · Intel Iris Xe iGPU · 8 logical cores · 16 GB RAM · no Apple MLX
**Benchmark mode:** `--fast` (winner per bucket; `--full` sweeps every backend)
**Model:** `ivrit-ai/whisper-large-v3-turbo-ct2`

| Bucket | Device | Compute | Threads | RTF | Auto tier |
|---|---|---|--:|--:|---|
| short (<10 s) | CUDA | int8_float32 | 4 | 0.558 | `light` |
| medium (10–30 s) | CUDA | int8_float32 | 4 | 0.188 | `accurate` |
| long (30–60 s) | CUDA | int8_float32 | 4 | 0.171 | `accurate` |
| extended (≥60 s) | CUDA | int8_float32 | 4 | 0.158 | `accurate` |
| streaming | CUDA | int8_float32 | 4 | 0.792 | `fast` |

Auto tier is what `params.get_params()` selects for that measured RTF
([Accuracy tiers](#accuracy-tiers), below), recomputed per call — not a stored
constant.

The `short` bucket pays fixed per-call overhead (audio I/O + VAD + a single
decode) that a <10 s clip can't amortise — hence RTF ~3× the longer buckets and
a lower tier. Live streaming (small overlapping chunks) is bounded the same way.

### Accuracy tiers

`params.py` picks the highest tier whose estimated cost keeps RTF under the
`0.85` budget — `estimated = base_rtf × {fast 1.0, light 1.35, balanced 1.8,
accurate 4.5}`:

| Tier | beam_size | Auto-selected when base RTF is |
|---|--:|---|
| `accurate` | 5 | < 0.19 |
| `balanced` | 3 | 0.19 – 0.47 |
| `light` | 2 | 0.47 – 0.63 |
| `fast` | 1 | > 0.63, or any RTF > 0.85 |

`accuracy_mode` in `config.yaml` pins a tier and skips this.

---

## 7. Project Layout

```
install.py               one-time setup: deps, models, first benchmark
benchmark_results.json   the only hardware-config source of truth (git-ignored)

visper/
  config.yaml            the one config file (packaged with the wheel)
  api.py                 stable public surface: transcribe / stream / chunked
  _cli.py                argparse entry points for the console scripts
  server.py              FastAPI: /transcribe, /transcribe/stream, /ws/live, /
  web/index.html         single-file web UI (packaged; served at / by the server)
  model_router.py        single-slot language→model cache
  transcriber.py         the only importer of faster_whisper / openvino_genai;
                         fallback chain, pre-proc, retry, two-stage he→en guard
  worker.py              venv-worker subprocess (JSON-line protocol)
  venv_manager.py        builds the per-device venvs (Windows AV-lock retries)
  params.py              single owner of every Whisper inference parameter
  benchmark.py           hardware detection, candidate sweep, get_best_config
  streamer.py            two-thread VAD-gated live pipeline
  translate.py           he→en stage 2 (opus-mt / CTranslate2), degrades to None
  postprocess.py         Hebrew text normalisation
  eval.py                visper-eval — local WER/CER + re-scorable sidecar
  resource.py            resource_profile enforcement (threads/priority/unload)
  _config.py             the one cached config loader

tests/unit/              pytest — pure logic, no model download
docs/
  design/                per-feature "why" notes
  benchmarks/            committed eval + benchmark runs behind §5/§6
  ship-readiness-audit.md  the defect inventory this rewrite worked from
  review-he-en-translation.md  adversarial review + security review of stage 2
  review-ship-readiness-final.md  the pre-merge adversarial review
  lessons.md             non-obvious bugs and their root causes
```

---

## 8. Local Setup & Quickstart

```bash
git clone https://github.com/Avrhambi/visper && cd visper
python install.py          # deps + ASR model (~1.5 GB) + he→en model (~210 MB) + first benchmark
```

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org) on PATH (WAV works
without it). If the model repo is gated, set `HF_TOKEN` in `.env` first.

```bash
# CLI
visper-file audio.mp3                              # → audio.txt
visper-file audio.mp3 --output srt                 # subtitles
visper-file audio.mp3 --language he --translate    # Hebrew → English
visper-live                                        # microphone, live

# Server + web UI
visper-server                                      # http://127.0.0.1:8000
```

```python
from visper import transcribe, transcribe_chunked
text = transcribe("audio.mp3")                     # Hebrew
en   = transcribe_chunked("audio.mp3", on_segment=print,
                          language="he", task="translate")
```

### Tests

```bash
pip install -e ".[dev]"
pytest                     # unit suite — no model download
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every PR and
every push to `master`:
`pip install -e ".[dev,server]"` on Python 3.10 / 3.11 / 3.12, an sdist build
check, then `pytest`. No model download in CI.

---

## 9. Platform Support

| Platform | CPU | CUDA | OpenVINO (Intel iGPU) | MLX (Apple Silicon) |
|---|:-:|:-:|:-:|:-:|
| Windows | ✅ | ✅ | ✅ | — |
| Linux | ✅ | ✅ | ✅ (needs compute runtime) | — |
| macOS Intel | ✅ | — | — | — |
| macOS Apple Silicon | ✅ | — | — | ✅ |

Apple Silicon uses base `whisper-large-v3-turbo` (the `ivrit-ai` fine-tune has
no MLX build) — strong, but not Hebrew-specialised.

---

## Acknowledgements

Visper is a thin engineering layer over other people's models and runtimes:

- **[ivrit.ai](https://www.ivrit.ai)** — the Hebrew ASR fine-tune
  ([`ivrit-ai/whisper-large-v3-turbo-ct2`](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2))
  that makes Hebrew transcription usable. Visper is a consumer of their work, not
  affiliated with the project.
- **[OpenAI Whisper](https://github.com/openai/whisper)**
  ([paper](https://arxiv.org/abs/2212.04356)) — the base model architecture; the
  non-Hebrew languages use [`whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)
  and [`distil-whisper`](https://huggingface.co/distil-whisper/distil-large-v3-ct2).
- **[OPUS-MT / Helsinki-NLP](https://github.com/Helsinki-NLP/Opus-MT)** —
  [`opus-mt-tc-big-he-en`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-he-en),
  the dedicated Hebrew→English translation model.
- **[SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper)** &
  **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** — the quantised
  inference runtime.
- **[OpenVINO](https://github.com/openvinotoolkit/openvino)**,
  **[Apple MLX](https://github.com/ml-explore/mlx-examples/tree/main/whisper)**,
  **[jiwer](https://github.com/jitsi/jiwer)**,
  **[FastAPI](https://fastapi.tiangolo.com)**,
  **[lucide](https://lucide.dev)**.

---

## Contributing

Component boundaries and design decisions: [ARCHITECTURE.md](ARCHITECTURE.md).

[whisper]: https://github.com/openai/whisper
[ivrit]: https://www.ivrit.ai
[ivrit-model]: https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ct2
