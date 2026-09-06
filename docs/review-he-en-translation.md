# Adversarial review — two-stage Hebrew→English translation

Scope: `bf53d06..HEAD` (4 commits `ccc9917`→`5aa8e5f`). Files: `visper/translate.py` (new),
`visper/transcriber.py`, `visper/server.py`, `pyproject.toml`, tests.

Verification: `python -m pytest tests/unit -q` → all pass (75). Findings below are from
code reading + constructed scenarios; the model is not installed locally so the live
`translate_batch` path and download path were not exercised.

Overall: the core design is sound. Recursion is **not** a risk — stage 1 hardcodes
`task="transcribe"` so the guard cannot re-enter. `_tier_override` and `bucket` are
passed through unchanged and the pinned tier is preserved. The real problems are in the
degradation contract (MT *inference* failure is uncaught) and in `he_en_supported()`
returning a wrong answer after a permanent load failure.

---

## 1. MT inference failure hard-fails the whole transcription — major

`visper/transcriber.py:520` and `:534` — `_translate_hebrew` calls `mt.translate(...)`
with no try/except. The `get_hebrew_english_translator() -> None` contract only covers
translator *construction* failure (missing model, bad import). If `translate_batch`
raises at call time (CT2 runtime error, OOM, a corrupt SentencePiece input, a beam that
returns no hypotheses so `res.hypotheses[0]` throws `IndexError`), the exception
propagates out of `transcribe()` → the caller gets a 500 / crash, **and the already-
completed stage-1 Hebrew ASR is thrown away**.

Design doc line 68 / 147: "translation is never allowed to hard-fail a transcription."
This violates it for the entire runtime-error class.

Direction: wrap the stage-2 block in `_translate_hebrew` in `try/except Exception`; on
failure log once and return the Hebrew `he` result unchanged (English == Hebrew, or
re-dispatch to Whisper's own translate), never raise.

## 2. `he_en_supported()` returns True after a permanent load failure — major

`visper/translate.py:65` —
```python
if _load_failed and not _model_present():
    return False
```
`_model_present()` (`:75`) checks only `model.bin` / `source.spm` / `target.spm`.
`ensure_model()` (`:126`) verifies only `model.bin`. Neither checks `config.json` /
`vocabulary.json`, which `ctranslate2.Translator()` actually requires. So with a model
directory on disk that is present-but-unloadable (truncated asset, missing vocab, CT2
version skew):

- `get_hebrew_english_translator()` sets `_load_failed = True` and returns `None` — good.
- but `he_en_supported()`: `_load_failed and not _model_present()` → `True and not True`
  → `False` → falls through → imports succeed → **returns `True`**.

Result: `/health` advertises Hebrew translation *forever, exactly after the path has
proven itself permanently broken*. Every user click then silently gets Whisper's own
poor Hebrew translate output with no error. Process restart does not help (module global).

Direction: `if _load_failed: return False` (drop the `_model_present()` conjunct), and
clear `_load_failed` when a fresh model dir appears rather than trusting it to stay
consistent. Separately, `_model_present()` / `ensure_model()` should verify the files CT2
actually loads (`config.json`, a vocabulary file), not just `model.bin`.

## 3. `urllib.request.urlopen(url)` has no timeout, inside the singleton lock — major

`visper/translate.py:113` — `with urllib.request.urlopen(url) as resp`. No `timeout=`.
A stalled TCP connection (captive portal, dead mirror, half-open socket) blocks forever.
This runs inside `get_hebrew_english_translator()`'s `with _lock:` block
(`:200`), so one stuck download **wedges every subsequent `transcribe(task="translate",
he)` call in the process on that lock, permanently** — not just the one request.
On the server that is a slow, unrecoverable degradation of the translate feature with no
log after the first line.

Direction: `urlopen(url, timeout=30)`; consider a total-bytes ceiling on the
`while chunk :=` loop; do the download outside the lock (lock only the singleton
assignment) or add a wall-clock cap.

## 4. `/transcribe/stream` emits zero SSE events for the entire ASR pass — major (design/impl mismatch)

`visper/server.py:224` + `visper/transcriber.py:512` — in translate+he mode, stage 1
runs with `on_segment=None`, so `transcribe_chunked`'s per-segment callback never fires
during the Whisper decode; all segment events arrive in a burst after stage 2.

The design doc (line 129) justifies this with "the MT pass is a small fraction of ASR
time" — a non-sequitur. The cost of withholding `on_segment` is the *entire stage-1 ASR
duration* of silence on the SSE stream, not the MT duration. On an `extended`-bucket
file (a 60-minute interview — the stated target user) the client sees nothing for
minutes, then everything at once. The `/transcribe/stream` endpoint's whole purpose is
progressive feedback; in translate mode it stops being a stream.

Direction: either forward stage-1 `on_segment` as Hebrew progress events tagged
`{"stage": "he"}` (client shows "transcribing…" preview), or document honestly that
`translate` mode is non-streaming and have the client show an indeterminate spinner.

## 5. `he_text` (aggregate Hebrew) is unreachable through the public API — minor

`visper/api.py:110` — `transcribe_chunked` returns `result.text` (a `str`); the
`TranscriptResult.he_text` field is dropped at the boundary. `ARCHITECTURE.md:53` marks
these signatures frozen. `visper/server.py:193` `/transcribe` response has no `he_text`
key; `/transcribe/stream` never sends the aggregate; `/ws/live` `_emit`
(`server.py:313`) rebuilds segments with only `start/end/text`, dropping per-segment
`he_text` too.

So the design's "bilingual view is possible" success criterion is met *only* via the
per-segment `he_text` side-channel on the `/transcribe` (non-stream) response, and only
because `segments.append` happens to capture the raw dict. Fragile and undocumented.

Direction: if bilingual output is a real requirement, add `he_text` to the `/transcribe`
and `/transcribe/stream` final payloads explicitly (a new response key is
backward-compatible; the frozen signature is the Python function, and a dict key is
additive).

## 6. `_translate_hebrew` never checks `is_aborted` during/after stage 2 — minor

`visper/transcriber.py:520-536` — `is_aborted` is forwarded to stage 1 only. After a
long ASR pass the user may have disconnected (`/transcribe/stream` sets `abort_event`),
but the MT batch still runs and `on_segment` still fires into a dead queue. Small waste,
not a correctness bug. Direction: early-return the Hebrew result if
`is_aborted and is_aborted()` before the `mt.translate` call.

## 7. Env-override disables checksum even for air-gapped mirrors — minor

`visper/translate.py:120` — `if _ASSET_URL_ENV not in os.environ and got != _ASSET_SHA256`.
An air-gapped mirror (the documented use case) is exactly where you still want integrity
verification — the comment's "deliberate local mirror" reasoning is weak. Direction:
skip the check only when the override URL host differs from `github.com`, or accept an
optional `VISPER_MT_HE_EN_SHA256` companion env var and verify against it.

## 8. First live/translate chunk stalls on cold load (~5 s) or first-use download (210 MB) — minor

`visper/server.py:303` `/ws/live` → guard → `get_hebrew_english_translator()`. It runs
inside `asyncio.to_thread` so the event loop is not blocked, but the first emitted chunk
is delayed by model load (~5 s cold, per design) or, on a fresh install, a 210 MB
download with no progress signal to the client. Per-chunk MT of tiny VAD fragments also
means Marian runs on sub-sentence input repeatedly. Not broken, but "live translate" has
a rough first-chunk and lower quality than file mode.

Direction: warm `get_hebrew_english_translator()` when the ws connects (before the first
chunk) and send a `{"status": "loading-translator"}` frame; or document live+translate
as best-effort.

## 9. `<unk>` silently dropped in decode — nit

`visper/translate.py:48` `_SPECIALS` includes `<unk>`; the design's verified recipe
(doc line 97) filtered only `</s>` / `<pad>`. Dropping `<unk>` rather than rendering it
hides genuine OOV output. Low impact for he→en; note it as a deliberate deviation from
the "verified inference recipe."

## 10. Dead parameter path: `_tier_override` + `task="translate"` together — nit

No caller passes both. `streamer.py` (the only `_tier_override` user) never passes
`task`. `_translate_hebrew`'s `_tier_override` plumbing is correct but exercised only by
tests. Not worth removing (it's the right thing structurally), just noting the guard's
"pinned tier is preserved" claim is currently untested against a real streaming call.

## 11. `he_en_supported()` import cost on every `/health` poll — nit

`visper/server.py:154` imports `visper.translate` (stdlib-only at module level — cheap)
then `he_en_supported()` does `import ctranslate2` / `import sentencepiece` per call.
After first call both are in `sys.modules`; first call may add ~100–300 ms to one health
response. The UI polls `/health` frequently. Direction: memoize the import result in a
module bool.

---

## Simplification notes

- `_translate_hebrew`'s `elif he.text:` branch (`transcriber.py:533`) re-translates the
  full `he.text` when there are no segments. In practice ASR with no segments but
  non-empty text is near-impossible for faster-whisper. Could drop to `text = ""` and
  lose ~4 lines, or keep as defensive.
- The repeated function-local `from visper.postprocess import normalize_text` and the
  double-normalize in the `/transcribe/stream` callback (`server.py:226` re-normalizes
  text `_translate_hebrew` already normalized) are pre-existing style — out of scope for
  this change, flagged only for a later pass.
- `he_en_supported()` and `_model_present()` and `ensure_model()` each encode a slightly
  different notion of "the model is usable." Collapsing to one `_model_files_ok()` helper
  that lists every file CT2 needs would remove the class of bug in finding #2.

---

## Resolution (commit after review)

| # | Sev | Action |
|---|-----|--------|
| 1 | major | **Fixed.** `_translate_hebrew` wraps the stage-2 MT calls in `try/except`; on any failure it logs once and returns the Hebrew `TranscriptResult` unchanged. `HebrewEnglishTranslator.translate` also tolerates an empty beam (`getattr(res, "hypotheses", None) or [[]]`). Test: `test_mt_failure_returns_the_hebrew_transcript`. |
| 2 | major | **Fixed.** `he_en_supported()` is now `if _load_failed: return False` then a memoised dep check. `_MODEL_FILES` lists every file CT2 needs; `_model_present()` and the post-download check both use it. |
| 3 | major | **Fixed.** `urlopen(url, timeout=30)`. |
| 4 | major | **Fixed.** Stage 1 now runs with a preview callback: each Hebrew segment is translated as it decodes and forwarded as an English `on_segment` event. The returned result is still rebuilt from the final (post-retry) Hebrew segments — when the preview count matches (no retry) that work is reused, else a batch pass runs. Streaming clients get live English; `text`/`segments` stay authoritative. |
| 5 | minor | **Deferred.** Bilingual output via a new `he_text` response key is additive and out of scope for the lean pass; the per-segment `he_text` side-channel on `/transcribe` is documented here as the current mechanism. |
| 6 | minor | **Fixed.** `_translate_hebrew` early-returns the Hebrew result if `is_aborted()` before the authoritative MT pass. |
| 7 | minor | **Fixed.** Mirror URLs verify against `VISPER_MT_HE_EN_SHA256` when set; the check is skipped only for a mirror with no companion hash. |
| 8 | minor | **Deferred.** Live+translate cold-load stall — acceptable for v1; noted as best-effort. |
| 9 | nit | **Fixed.** `_SPECIALS` is now `{</s>, <pad>}`, matching the verified recipe. |
| 10 | nit | No action (structurally correct). |
| 11 | nit | **Fixed.** Dep-import result memoised in `_deps_ok`. |

Simplification: `_model_files_ok` consolidation done (`_MODEL_FILES`). The `elif en_full` branch kept as defensive (≈3 lines).

## Unresolved / could not verify

- **`sentencepiece` wheel availability on Python 3.13/3.14.** *Resolved:* PyPI ships
  `sentencepiece 0.2.2` cp310–cp314 wheels for win_amd64, macOS (x86_64 + arm64) and
  manylinux_2_28 (x86_64 + aarch64). No packaging regression.
- **`HebrewEnglishTranslator` thread safety.** The docstring says "not thread-safe for
  concurrent calls." `translate()` reads `self._translator/_sp_src/_sp_tgt` and keeps all
  mutable state in locals, so it is re-entrant at the Python level; whether
  `ctranslate2.Translator.translate_batch` with `inter_threads=1` serializes or errors
  under concurrent calls from multiple `asyncio.to_thread` workers (two simultaneous
  `/transcribe` translate requests) was not tested. If it is genuinely unsafe, add a
  per-translator lock around `translate_batch`.
- **Live `translate_batch` correctness / quality** — model not installed; the roundtrip
  test is `skipif(not _model_present())` and was skipped.
- **`os.replace(extracted, _MODEL_DIR)`** is same-filesystem by construction (`tmp_dir`
  is `mkdtemp(dir=_MODEL_DIR.parent)`), so the cross-device concern does not apply.
  Confirmed by reading, not by running.
