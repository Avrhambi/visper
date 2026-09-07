# Lessons

Running log of non-obvious bugs and their root causes. Read before starting a
debugging session.

---

## 2026-09-06 — device-venv creation aborted the whole benchmark on Windows

**What broke:** `visper-benchmark` / `visper-eval` on a fresh machine died with
`subprocess.CalledProcessError` from
`['.venvs/cpu/Scripts/pip.exe', 'install', '--upgrade', 'pip', ...]`, exit 1,
message: *"ERROR: To modify pip, please run the following command: python.exe -m
pip install ..."*.

**Root cause:** on Windows, pip refuses to upgrade **itself** when invoked as
`pip.exe` because it cannot replace its own running executable. `venv_manager.
create_venv` shelled out to the venv's `pip.exe` directly.

**Fix:** run every pip command as `<venv python> -m pip ...`. Also made the pip
self-upgrade non-fatal (a fresh `python -m venv` already ships a working pip).

**Gotchas carried out of this:**
- A failed `create_venv` used to leave `.venvs/<device>/` with a `python.exe`
  but no packages; `venv_exists()` only checked for `python.exe`, so the next
  run "reused" it and then failed importing `faster_whisper`. Now a
  `.visper-ready` marker file is written only after all installs succeed, and
  `create_venv` `rmtree`s the dir on any failure.
- The native-wheel stack (ctranslate2, onnxruntime, openvino) lags the newest
  CPython. A venv built with 3.14 can't `pip install faster-whisper`. Device
  venvs now prefer `py -3.12` when the host has it.

---

## 2026-09-06 — venv creation intermittently fails on Windows (AV lock)

**What broke:** after the pip.exe fix above, `visper-benchmark --fast` still
failed building `.venvs/cuda`: first run `ensurepip ... returned non-zero exit
status 1`, second run `[Errno 13] Permission denied: '...\\.venvs\\cuda\\Scripts
\\python.exe'`. A bare `py -3.12 -m venv .venvs/_probe` succeeded every time.

**Root cause:** Windows Defender real-time protection scans `python.exe` the
instant `python -m venv` copies it into the new venv, briefly locking the file.
ensurepip (which runs *inside* venv creation) can't use the locked interpreter →
exit 1. Worse: `shutil.rmtree(dir, ignore_errors=True)` reports success while
the locked `python.exe` is still on disk, so the retry starts from a dirty tree
and hits `PermissionError` on that exact file — reproducing the failure
deterministically.

**Fix:** `_create_venv_tree()` retries `python -m venv` up to 3× with backoff;
`_rmtree_confirmed()` polls until the directory is actually gone before any
recreate. The lock clears within ~1–2 s.

**Gotcha:** `rmtree(ignore_errors=True)` is not "delete the tree" — it's "try,
shrug on failure". Anywhere a later step assumes the path is gone, poll for it.

---

## 2026-09-07 — visper-eval WER was inflated ~15% relative by punctuation

**What broke:** spot-checking `visper-eval` output, the Hebrew `short` (Knesset)
WER read ~0.30 where a manual diff of hyp vs ref looked more like ~0.26. Every
run also cost a full multi-hour re-transcribe just to try a scoring tweak.

**Root cause:** `_metrics` scored the shipped `normalize_text` output directly.
`normalize_text` deliberately keeps punctuation (users want it), but every
reference corpus here (`coish`, `short`, `long`) carries **zero** punctuation.
So each correctly-emitted comma/period/maqaf counted as an insertion error.
WER is conventionally punctuation-insensitive; this was a scoring artifact, not
model error.

**Fix (`dfe4337`):** a symmetric `_for_scoring` pass (lowercase + strip
everything non-`\w\s`, Unicode-aware so Hebrew letters/digits survive) applied
to **both** sides inside `_metrics` only. The shipped normalizer is unchanged.
`--out` now also writes a `<out>.json` sidecar with every ref/hyp pair, and
`--rescore SIDECAR.json` recomputes the table in seconds — a scoring change no
longer means re-transcribing.

**Gotchas carried out of this:**
- Reference corpora for ASR eval frequently have no punctuation and no casing.
  Always score with a symmetric normalization pass that is *separate* from the
  product's own text normalization. Don't reuse the shipping normalizer for
  metrics.
- WER **distribution shape** tells you what you're measuring: a smooth unimodal
  spread (e.g. `short`: 0.14–0.39) is genuine difficulty and the mean is real;
  a tight low cluster + a high outlier tail usually means some ref/audio pairs
  are misaligned and the mean is measuring corpus noise. `_summarize` now
  reports min/p25/median/p75/max so the README can tell which it's quoting.
- `coish` (CoSIH) is a spontaneous-conversation **linguistics** corpus, not an
  ASR benchmark — high WER there is the transcription convention (fillers,
  overlap, phonetic spelling), not model failure. Report it as a limitations
  data point, never a headline number.

---

## 2026-09-07 — `/transcribe` returned an empty segment list on the default runtime

**What broke:** on the venv-worker runtime (the default once a benchmark stamps
a `venv_path`), `POST /transcribe` returned `"segments": []` and
`/transcribe/stream` emitted only the final `final_text` event — no per-segment
timestamps, no progress. The web UI builds its transcript view from `d.start` /
`d.end`, so it silently lost all timing. The in-process path was unaffected, so
this only showed up on real hardware, not in tests.

**Root cause:** `Transcriber.transcribe()` forwards `on_segment` only to the
in-process decode loop. `_transcribe_via_worker` never took the parameter — the
worker returns every segment in one batch (its JSON-line protocol has no
incremental message), and nobody replayed that batch to the caller. The server
collects segments purely via the `on_segment` callback (`segments.append`), so
on the worker path the callback never fired and the list stayed empty. The
`TranscriptResult.segments` field *was* populated — but `transcribe_chunked`
returns only `.text`, so the server never saw it.

**Fix (`7a530b3`):** `_transcribe_via_worker` takes `on_segment` and replays the
final (post-retry) segments through it, abort-aware; `transcribe()` forwards it.
Fires once per segment, after decode — not live, but the consumer now sees the
list.

**Gotcha:** a callback-delivered value and a return-value-delivered value are
different data paths. `TranscriptResult.segments` being correct told you nothing
about whether the server's `on_segment` list was. When two runtimes are meant to
be "identical", enumerate every output channel — return value, callback, side
file — and check each one on both.

---

## 2026-09-07 — the benchmark reported a coarser accuracy tier than runs

**What broke:** the `visper-benchmark --report` table (which the README quotes)
showed the `short` bucket at the `fast` tier for a measured RTF of 0.558. But
`params.get_params()` selects `light` for that same RTF at call time — so the
README would have published a tier the engine never actually uses.

**Root cause:** tier selection was implemented **twice**.
`params.py:get_params()` walks `("accurate", "balanced", "light")` and falls to
`fast`; `benchmark.py:_auto_accuracy_tier()` checked only `accurate` and
`balanced` before falling to `fast` — it omitted the `light` branch entirely.
Same policy, two copies, one stale.

**Fix:** `_auto_accuracy_tier()` now imports `RTF_BUDGET` +
`TIER_RTF_MULTIPLIERS` from `params.py` and runs the identical loop. The stored
`auto_accuracy_tier` field is display-only (runtime reads `best[bucket].rtf` and
recomputes), so no re-benchmark was needed — but the two must not drift.

**Gotcha:** when a value appears both in a stored artifact and is recomputed at
runtime, assume they will diverge unless one calls the other. Grep for every
producer of a policy before quoting its output.
