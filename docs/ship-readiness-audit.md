# Visper — Ship-Readiness Audit

Status: validation complete (6 parallel read-only audits, 2026-09-06). Not yet started on fixes.
This file is the source of truth for what's wrong. Grouped by theme, not by the original flat list.

---

## Verdict on the original 63-item list

~48 VALID · ~14 PARTIAL (claim true, a count/number in it is off) · 1 INVALID · 0 blocking-unverifiable.

PARTIAL / INVALID corrections worth knowing:

| # | Correction |
|---|---|
| 5 | `web/index.html` has **0** `window.X =` globals. Real problem: ~30 mutable module-level vars + ~50 global functions, one 1992-line file. Architecture critique stands; the "~40 window globals" figure does not. |
| 9 | `_cli.py` has **4** functions (3 one-line wrappers + `help`), not 5. `help` shadowing the builtin: true. `visper-server` bypassing `_cli`: true. |
| 10 | **2** real VAD chunkers (`LiveStreamer`, server `/ws/live`), not 3. The genuinely unused impl is `api.stream_transcribe` — zero callers anywhere. |
| 14 | Top-level `model_id:` key is dead (nobody reads it). Top-level `language:` key **is** read (`transcriber.py:77`). |
| 16 | `api.py` reads none of the Transcriber privates. `server.py` reads a `ModelRouter` private (`_active_model_id`), not a Transcriber one. `streamer.py` does read `_backend`/`_config` — that part is real. |
| 19 | OV path literals differ (`ov_model` vs `models_ov/...` vs `models_ov_test/...`). README does **not** name an OV path, so "README points at the wrong one" is unsupported. |
| 20 | Streaming "median RTF of calls 2–5" is actually `sorted(measured)[len//2]` = index 2 of 4 ≈ **p75**, biased high. The max-of-4 exists but is honestly labelled `rtf_p95` and is unused. |
| 26 | Duration→bucket logic duplicated in **api + benchmark** (different helper names, same 10/30/60 thresholds). `benchmark_worker` and `web` do **not** bucket. |
| 33 | **INVALID** — ARCHITECTURE and benchmark bucket boundaries match exactly (`<10 / 10–30 / 30–60 / >60`, midpoints 5/20/45/90). |
| 37 | `--quick`, `--profile`, `--clip`-on-live all **do** exist. Real doc-vs-code flag gap: `visper-server --host/--port/--reload` documented, `server.main()` has no argparse. |
| 42 | `@app.on_event("startup")` deprecated: true. There is **no** shutdown handler at all. |
| 46 | `_deduplicate_overlap` is **not** unbounded — capped at 10 words vs the previous chunk only. "No tests": true. |
| 58 | ARCHITECTURE **does** cover `worker.py`. Omitted: `_cli.py`, `venv_manager.py`, `run_benchmark.py`, `constants.py`. |

---

## Theme 1 — The venv-worker path silently drops almost every quality feature  ★ highest impact

Once `run_fast_benchmark` stamps a `venv_path` into `benchmark_results.json` (the **normal** outcome), inference goes through `_transcribe_via_worker` → `worker.py`. That path does **not** forward:

- `initial_prompt` — `worker.py:132-147` fixed key list omits it → `--prompt` flag and web prompt field are no-ops (High)
- `hotwords` — only added on the direct CT2 path (`transcriber.py:320-321`); missing from retry path, worker request, worker, MLX branch, OV branch → `config.yaml: hotwords` inert in the default deployment (Med)
- confidence-retry (`transcriber.py:349-374`) — in-process branch only (Med)
- audio denoise / normalize / highpass (`transcriber.py:303-312`) — in-process branch only; `config.yaml` ships all three `true` and they do nothing (Med)
- `is_aborted` early-stop — worker never checks it → web cancel / client-disconnect can't stop a running transcription (Med)

`worker.py` and `benchmark_worker.py` also duplicate `_load_model` / `_load_audio` / `_infer` / `_TRANSCRIBE_KWARGS` (byte-identical dicts), and the same load logic exists a 4th time in `transcriber.py` and `benchmark.py`.

---

## Theme 2 — Config system: 8 ad-hoc parsers, silent drift, dead keys

8 independent `try/except: pass` YAML readers: `params.py:125`, `resource.py:41`, `streamer.py:83`, `transcriber.py:72`, `model_router.py:35`, `benchmark.py:1046`, `transcribe_file.py:125`, `transcribe_live.py:70`. Each has its own key subset and defaults.

Code-default vs shipped-`config.yaml` drift (only surfaces when config.yaml is absent — i.e. non-editable pip install, see Theme 8):

| key | code default | config.yaml ships | README says |
|---|---|---|---|
| `vad_min_silence_ms` | 300 | 500 | — |
| `audio_denoise` | False | true | false |
| `audio_highpass` | False | true | false |
| `audio_normalize` | False | true | false |
| `igpu_preference_margin` | 0.0 | 0.05 | — |

Dead keys (documented + shipped, read by nobody): `model_id:`, `print_to_stdout:`.
Duplicated source of truth: `models:` / `models_mlx:` maps in `config.yaml` are hardcoded again in `model_router.py:18-32`.

- `api._config_cache` never invalidated → re-running the benchmark has no effect on a running server until restart, contradicting ARCHITECTURE.md:67 (Med)
- `api._get_router` ignores `hw_config` after the first call → whichever bucket transcribes first pins the device/compute-type for **all** buckets; router has no bucket awareness (Med)
- `ModelRouter.get()` re-parses `config.yaml` on every call (Low)
- `get_best_config()` triggers a multi-minute `run_benchmark()` as a side effect, reachable from `GET /health` and from a fresh `import visper; transcribe(...)` — no guard, no timeout (High for UX)
- `bucket_accuracy_overrides: accurate` (all 4 buckets, shipped) is applied **before** the `auto` branch in `params.py:166` → the entire RTF-headroom tier ladder AND the `RTF > 0.85 → force fast` safety demotion are **dead in the default config**. Weak hardware silently runs `beam_size=5, best_of=3` + confidence-retry, RTF can exceed 1.0 with no warning (High)
- Same override forces `tier == "accurate"` → `confidence_retry` defaults **on** for every non-streaming call (Med)
- `--accuracy` CLI flag is therefore a **no-op** under the shipped config (bucket override out-ranks it) (Med)
- `api.transcribe()` routes the model with `language="he"` but passes no language to `engine.transcribe`, which then reads `self._language` from `config.yaml` → set `language: en` and the Hebrew CT2 model gets an `<|en|>` token (Med)

---

## Theme 3 — Streaming: 3 divergent implementations, not duplicates

`server.py /ws/live` is a hand-rolled chunker that never touches `LiveStreamer`:

| | `LiveStreamer` | server `/ws/live` |
|---|---|---|
| max chunk | 28 s (`max_chunk_seconds`) | 8 s (`_MAX_FRAMES`, hardcoded) |
| noise calibration | 1.5 s, `max(0.01, rms*1.5)` | 0.5 s, `max(min(rms)*2.0, 1e-4)` |
| overlap window | yes (`overlap_seconds`) | none (buffer fully reset) |
| boundary dedup | yes | none |
| `max_chunk_seconds` / `overlap_seconds` / `stream_flush_on_silence` / `noise_calibration_seconds` / `vad_speech_pad_ms` | honoured | **zero effect** |

(`vad_filter` / `vad_min_silence_ms` / `accuracy_mode` still reach both via `Transcriber` — only the outer-chunker keys are dead on the server path.)

Streamer-specific bugs found:
- `_load_config` leaves `_overlap_samples` / `_noise_calibration_seconds` unset when `config.yaml` is absent → `AttributeError` on first chunk (Med; High on non-editable install)
- **Idle-unload / VRAM-demote permanently kills the stream** (High): `Transcriber.unload()` nulls `self._backend` in place, but `ModelRouter._transcriber` / `_active_model_id` still advertise it as live. `LiveStreamer` calls `Transcriber.unload()` directly, never `ModelRouter.unload()`. After unload, `_reload_model()` gets the same dead object back → `AttributeError` every chunk, swallowed as "[STT] Transcription error". Fix must move cache invalidation into `ModelRouter`.
- VRAM-demote-to-CPU is a no-op (router keys only on `model_id`, which is unchanged)
- `_reload_model` / `_check_memory` hardcode `"he"` → non-Hebrew session silently switches to the Hebrew model on idle-unload
- `_producer_file` emits the tail chunk twice (`is_final=True` fires on the last real chunk and again on the re-read overlap tail)
- `stop()` → `_flush_buffer()` can race the consumer's exit and drop the final utterance; `_audio_buffer` mutated from 3 places with no lock
- `os.nice(10)` in `resource.py` is relative and re-runs on every `ModelRouter` language swap → stacks (+10, +20, …) on POSIX
- server `/ws/live` VAD gate mixes units (silence counted per-message, buffer measured in 512-sample frames); browser `ScriptProcessor(512)` is routinely clamped up to 1024/2048

---

## Theme 4 — CUDA DLL registration: 8 sites, 3 divergent algorithms

`transcribe_file.py:22-34` + `transcribe_live.py:21-32` (identical, `bin`-dirs only, `win32`-guarded) · `benchmark.py:658` + `benchmark_worker.py:90` + `worker.py:48` (prepend + `rglob`) · `server.py:33` (append + `rglob`, **no platform guard**, `os.environ["PATH"] +=` will `KeyError` if PATH unset) · `install.py:39` (append, its own variant) · `tests/test_gpu.py:93`. Prepend vs append is functionally material (prepend overrides a broken system CUDA on PATH; append does not).

---

## Theme 5 — Duplication (mechanical, safe to dedup once Themes 1–4 land)

- SRT/VTT/JSON formatting: `transcribe_file.py:79-122` and `web/index.html:1943-1968`. **Divergent**: Python truncates ms, web rounds (can emit invalid `00:00:03,1000`); JSON shapes differ; empty-segments handling differs.
- faster-whisper kwargs assembled in 4 places (`transcriber.py:314`, `:358`, `:495`, `worker.py:132`) + `params.as_transcribe_kwargs()` + `_TRANSCRIBE_KWARGS` ×2 in benchmark.
- Audio pre-processing sequencing block duplicated (CT2 branch `transcriber.py:303`, MLX branch `:411`), near-verbatim.
- Duration detection ×4 (`transcriber.py:602` → `0.0`; `api.py:156` → `None`; `benchmark.py:111` → 3-tier → `None`; `web:1481` → `0`).
- Duration→bucket ×2 (`api._bucket_for_duration`, `benchmark._bucket_for`).

---

## Theme 6 — Docs vs reality (README rewrite required)

Numbers that are fabricated or stale — **must not be reproduced in the rewrite without a real source**:

- README RTF table `0.55 / 0.18 / 0.16 / 0.13` (CUDA) contradicts `config.yaml:50` comment `0.58 / 0.20 / 0.18 / 0.15` for the same device. No `benchmark_results.json` committed; no generator for a markdown table.
- README accuracy-tier RTF ranges (`fast >0.47`, `light 0.28–0.47`, `balanced 0.19–0.28`, `accurate <0.19`) reproduce the **old** `params.py` multipliers. Current code: `accurate <0.189`, `balanced 0.189–0.472`, `light 0.472–0.630`, `fast ≥0.630`. 3 of 4 rows wrong.
- `WER 0.179, CER 0.084 on 221 Hebrew files` — **no eval script, no dataset, no reference transcripts anywhere in the repo.**
- README config-defaults table says `audio_denoise/highpass/normalize` default `false`; shipped `config.yaml` ships them `true`.

Other doc-vs-code:
- ARCHITECTURE.md:67 "any hardware change requires only re-running the benchmark — not a code change" — false; `_build_fallback_chain` / `_estimate_config_heuristic` encode the whole hardware decision in literals (`_CUDA_MIN_VRAM_MB = 1800`, `vram >= 4000`, `cpu_threads = ... # always 2`, `_MODEL_VRAM_ESTIMATE_MB` table). `_CUDA_MIN_VRAM_MB` docstring says "3 GB" for value 1800.
- README "Project Structure" lists `transcribe_file.py` / `transcribe_live.py` / `run_benchmark.py` at repo root; they're in `visper/` (commit 596d45b). Same staleness in ARCHITECTURE diagram + 2 error strings.
- Error strings reference nonexistent `setup.py` (×2) and `tests/test_benchmark_full.py`.
- `install.py:327` still branches on removed `start.bat`.
- `requirements.txt` lists `webrtcvad-wheels` — nothing imports it; the comment calls the energy threshold a "fallback" when it's the only VAD path.
- `requirements.txt` vs `pyproject.toml`: only `webrtcvad-wheels` disagrees in the base list; no version pins anywhere; no lock file.
- Project identity: `pyproject` version `1.1.0` vs `server.py` `FastAPI(version="0.2.0")`; `pyproject` description "Hebrew speech-to-text engine" vs README "Local transcription tool / 12 languages"; titles "Hebrew STT" / "Local Speech-to-Text" / "Visper" scattered.
- ARCHITECTURE "median RTF of calls 2–5" — actually p75 (see #20).
- `fallback_order` is only written by `run_fast_benchmark` (`--fast`); `run_benchmark` (smart/full/quick) omits it → after the `visper-benchmark --force` the README tells you to run, a primary-load failure raises "All devices in the fallback chain failed" after **zero** attempts (High).
- `benchmark_worker.py:127` reads the OV model from `ov_model/` while everything else uses `models_ov/...` → OpenVINO can never win the benchmark (High).
- `tests/test_openvino.py` writes the converted model to `models_ov_test/...`; runtime reads `models_ov/...`; the not-found error tells the user to run that exact script (Med).

---

## Theme 7 — Web UI

- Not served by `visper-server` (no route); runs from `file://`; server sets CORS `allow_origins=["*"]` + binds `0.0.0.0`, no auth → **any website the user visits can POST audio to `localhost:8000` and read the transcript; any LAN device can use the GPU** (High).
- XSS: `file.name` / `item.title` interpolated raw into `innerHTML` (`index.html:1348`, `:1779`) and persisted to `localStorage` → re-executes every load. Other render paths correctly use `textContent`.
- Library "saved locally, click any timestamp to seek audio" oversells: `audioUrl` is nulled on save (and `blob:` URLs die on reload anyway) → after reload, no audio element, no seek. Real fix needs IndexedDB, not a localStorage tweak.
- `liveWs` has no `onerror`; a connected-but-silent socket (never sends `{status:"ready"}`) leaves the mic button disabled forever. `JSON.parse` of WS messages not guarded.
- Error surface is `alert()`/`confirm()` only; batch failures show only a red chip, no detail; server `/ws/live` wraps its whole loop in `except Exception: pass`.
- `lucide` loaded from `unpkg.com/lucide@latest` (unpinned, external) and `lucide.createIcons()` is unguarded early in `DOMContentLoaded` → CDN down = the entire Create tab's listeners never wire up. Contradicts the "no cloud dependency" tagline. Google Fonts is a second CDN dependency.
- Object URLs from `createObjectURL` never revoked (upload preview, recording blob, downloads) — leak for the tab lifetime.
- VAD meter (`animateMeter`) never restarts after the first pause/resume.
- Prompt length limit stated 4 ways: CLI "~55 words", web placeholder "55 characters", `maxlength="55"`, JS truncates at 40 words (dead code — maxlength makes it unreachable). Server validates nothing.
- No upload size cap (`server.py:54` streams unbounded to a temp file).
- `GET /health` returns HTTP 200 even on failure.
- Keyboard/a11y: hidden file input reachable only via `onclick` div; non-focusable click targets; no ARIA on menus.
- `@app.on_event` deprecated; `asyncio.get_event_loop()` in a coroutine deprecated; warmup task not retained (can be GC'd).

---

## Theme 8 — Packaging / project hygiene

- Config + `benchmark_results.json` resolve to `Path(__file__).parent.parent` (repo root). `pyproject.toml` `[tool.setuptools.packages.find] include = ["visper"]` — no `package_data`, no `MANIFEST.in`. Non-editable `pip install .` → `config.yaml` absent → every parser silently falls back to code defaults (which differ from shipped — Theme 2), `streamer` raises, `benchmark_results.json` targets an unwritable dir.
- No `LICENSE` file; `pyproject` has no `license` field.
- No CI (`.github/` absent).
- `tests/` are venv-bootstrapping hardware timing scripts — zero `assert`s, no pytest, no correctness check on transcription output.
- No pinned dependencies, no lock file.
- `records/`: `school.mp3` (used) + `sun.wav` (7.5 MB, unused — `_get_records_by_bucket` returns early once `school.mp3` is found). Undocumented (no provenance/licence).
- `postprocess.py` `_REPEAT_CHAR = re.compile(r'(.)\1{2,}')` → `.sub(r'\1', ...)` collapses any run of 3+ identical chars: `"20000" → "20"`, `"1000000" → "10"`, `"10:00:00" → "10:0:0"`. **Silent transcript corruption of numbers** (High).
- `transcribe_file.py:62` and `:69` — operator precedence (`or` / `and`): `"ffmpeg" in low or "no such file" in low and <suffix>` → a `.wav` file with an ffmpeg-mentioning error gets told "ffmpeg is required to decode .wav files"; CUDA branch misattributes errors the same way.
- `_friendly_error` claims "the engine will retry on CPU automatically" — there is no runtime OOM→CPU retry (fallback chain runs at load time only).
- `params._logged_first_call` module global, never reset → stderr-log assertions would be process-order-dependent.
- `--background`: `sys.stdout = open(os.devnull)` never closed; POSIX `fork()` with no `setsid()` → not a real daemon.
- `_spawn_worker` / `unload` call `terminate()` with no `wait()`/`kill()` escalation → zombies on POSIX, orphan `python.exe` holding VRAM on Windows if parent is killed.
