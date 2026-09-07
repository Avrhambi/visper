# Final adversarial review — `ship-readiness` → `master`

Scope: `git diff master...ship-readiness` (38 commits). Validation run locally:
`pytest` = **80 passed**; `python -m build --sdist` = **clean** (`visper-1.1.0.tar.gz`).
Host Python 3.14.3; `ctranslate2 4.7.1` and `sentencepiece` both import on the host
(so the in-process he→en MT path is viable on the reference machine).

Ranked findings, most severe first.

---

## 1. should-fix (fix before the repo is shown to anyone) — every §5/§6 number is untraceable, and the one committed eval artifact contradicts the README

**Where:** `README.md:178-199` (§5 Accuracy), `README.md:203-229` (§6 Performance);
`.gitignore:14` (`benchmark_results.json`), `.gitignore` "Results / plans" block
(`eval_results.md`, `*.md.json`).

**What's wrong:**
- `benchmark_results.json` and `eval_results.md` are both `git check-ignore`-positive —
  neither is committed. The harnesses (`visper/benchmark.py`, `visper/eval.py`) are
  committed, but the *outputs* the README quotes are not, and they are machine-specific
  measurements a reviewer cannot reproduce without a multi-hour benchmark + corpora they
  don't have.
- The **one** accuracy artifact that exists in the working tree, `eval_results.md`,
  disagrees with README §5:
  - `eval_results.md`: `coish` 15 files → WER **0.698**, CER 0.463.
    README §5 "CoSIH (spontaneous)" 15 files → WER **0.575**, CER 0.439.
  - `eval_results.md`: `short` 25 files → WER **0.264**, CER 0.110.
    README §5 has no matching row at that number (closest: "Longer-form" 0.245 / "Knesset" 0.141).
- `README.md:43` ship-criterion #1 states the numbers "come from a committed harness … real
  runs on the machine in §6, not estimates." The runs are real; the evidence is not in the repo.

**Failure scenario:** an interviewer clones the repo, runs
`visper-eval <corpus> --tier balanced`, gets 0.698 WER on spontaneous speech where the
README claims 0.575 — the single most damaging outcome for a portfolio repo. They cannot
tell which number is current.

**Suggested fix:** commit a sanitized evidence bundle the README links to — e.g.
`docs/benchmarks/results.json` (strip the machine-local `venv_path` absolute paths) and the
eval sidecar JSON `visper-eval --rescore` consumes — and delete the stale root
`eval_results.md`, or regenerate it from the same run that produced §5. Point §5/§6 at those
committed files by path.

---

## 2. should-fix — `format_report_table()` prints a *stored* tier that is stale and contradicts the README

**Where:** `visper/benchmark.py:149` (`tier = cfg.get("auto_accuracy_tier", "?")`),
`README.md:216-225` (§6 table + "recomputed per call — not a stored constant").

**What's wrong:** commit `3c3e118` fixed `_auto_accuracy_tier()` to add the `light` rung and
mirror `params.get_params()`. But `benchmark_results.json` was written *before* that fix, so
its stored `auto_accuracy_tier` for `short` (RTF 0.558) is `"fast"`. The policy now says
`light` (0.558 × 1.35 = 0.753 < 0.85) — and the README table correctly shows `light`.
`format_report_table()` reads the stored field verbatim, so `visper-benchmark --report`
today prints `fast` for `short`, contradicting both the README and the live
`params.get_params()` decision. The README's own claim that the tier is "recomputed per
call, not a stored constant" is false for the report.

**Failure scenario:** `visper-benchmark --report` output ≠ README §6 table on the exact
machine both describe.

**Suggested fix:** in `format_report_table()` compute the tier from the RTF:
`tier = _auto_accuracy_tier(rtf) if isinstance(rtf, (int, float)) else "?"`. Then regenerate
the committed results bundle from finding #1 so the stored field agrees too.

---

## 3. should-fix — the web UI is absent from the sdist/wheel; `pip install visper` then `visper-server` 404s `/`

**Where:** `pyproject.toml` `[tool.setuptools.package-data]` (lists `config.yaml` only, no
`web/`); no `MANIFEST.in`; `visper/server.py:36`
`_WEB_DIR = pathlib.Path(__file__).parent.parent / "web"`.

**What's wrong:** `tar tzf visper-1.1.0.tar.gz` contains `visper/config.yaml` and `LICENSE`
but **no `web/`**. On any non-editable install `_WEB_DIR` resolves to
`site-packages/web`, which doesn't exist, so `GET /` raises `HTTPException(404,
"web/index.html not found")` and `/vendor` is never mounted. README §3 ("opens from the
server", "zero build step"), §8 (`visper-server → http://127.0.0.1:8000`) and the ship
criterion's "packaged so a non-editable install behaves identically" claim (`README.md:132`)
do not hold for the UI.

**Failure scenario:** `pip install dist/visper-1.1.0.tar.gz && visper-server` → API works,
`http://127.0.0.1:8000/` is a 404. CI only exercises `pip install -e` (editable, `web/` sits
at `../web`), so CI is green while the shipped artifact is broken.

**Suggested fix:** move `web/` under the package (`visper/web/`) or add it to
`package-data` + a `MANIFEST.in`, and make `_WEB_DIR` package-relative
(`pathlib.Path(__file__).parent / "web"`). If the UI is intentionally source-checkout-only,
say so explicitly in §3/§8 and downgrade the "behaves identically" claim.

---

## 4. should-fix — `/health` advertises Hebrew translation before the (currently 404) model download is even attempted

**Where:** `visper/translate.py:80-88` (`he_en_supported()`), `visper/server.py:~157`
(`no_translate = [] if (device == "mlx" or he_en_supported()) else ["he"]`),
`README.md:165-168` (§4 "stops advertising … the moment the MT path is known broken").

**What's wrong:** `he_en_supported()` is `_deps_importable() and not _load_failed`. On a
fresh install `_load_failed` is `False` and `ctranslate2`/`sentencepiece` import fine, so
`/health` returns `no_translate: []` and the UI shows the Hebrew→English toggle. The release
asset `_ASSET_URL` (`translate.py:38-41`) is a **guaranteed 404 today**, so the *first*
click: `ensure_model()` raises `HTTPError` → `_load_failed = True` → the request silently
falls through to Whisper's own `task=translate` — the output README §1/§3 explicitly call
"poor" and the entire reason stage 2 exists. §4's promise is true only *after* the first
failure, i.e. never on first use.

**Failure scenario:** every user's first Hebrew→English request produces the bad
Whisper-native translation with no signal that stage 2 didn't run; `/health` claimed it was
available.

**Suggested fix:** do **not** just add `_model_present()` to `he_en_supported()` — that
deadlocks (the model only downloads on first use; first use only happens if the toggle is
shown). Either warm `ensure_model()` in `install.py` and in the server lifespan task, or make
`/health` tri-state (`available` / `available-after-download` / `unsupported`) and let the UI
show a "first use downloads ~210 MB" affordance. At minimum, land the release asset + correct
`_ASSET_SHA256` before merge and document the first-run download.

---

## 5. should-fix (pre-publish) — build recipe comment and `_MODEL_FILES` disagree on the vocab filename

**Where:** `visper/translate.py:36-37` (comment: `+ source.spm / target.spm / vocab.json`)
vs `visper/translate.py:52-53` (`_MODEL_FILES = (… "shared_vocabulary.json")`).

**What's wrong:** `_model_present()` and the post-extract check in `ensure_model()` both gate
on `shared_vocabulary.json`. `ct2-transformers-converter` (CT2 4.x) emits
`shared_vocabulary.json` for a shared-vocab Marian model, so `_MODEL_FILES` is almost
certainly right and the comment is stale — but the asset does not exist yet, so nobody has
verified. If whoever builds the asset follows the comment and ships `vocab.json`,
`ensure_model()` raises `archive is missing ['shared_vocabulary.json']` on every attempt and
stage 2 is dead on arrival with only a stderr line.

**Failure scenario:** asset built to the comment spec → he→en MT never loads, permanently,
silently.

**Suggested fix:** fix the comment to name `shared_vocabulary.json`; when building the
asset, `tar tzf` it and confirm the five `_MODEL_FILES` names match exactly before
publishing; consider a tiny test that asserts `_MODEL_FILES` equals the converter's known
output set.

---

## 6. nit / should-fix — in-process confidence-retry drops `hotwords`; the worker retry keeps them (parity break)

**Where:** `visper/transcriber.py:387-397` (in-process `kwargs2` — sets language, task,
initial_prompt, vad only) vs `visper/transcriber.py:647-662` + `:685`
(`_build_kwargs` always re-adds `self._hotwords`).

**What's wrong:** README §1/§4 claim "same confidence-gated retry" on both runtimes. On the
in-process path the retry decode silently loses hotword biasing; on the worker path it
keeps it. The retry is the *higher-accuracy* pass, so this is backwards.

**Failure scenario:** a config with `hotwords` set, low-confidence clip → in-process retry
produces different (unbiased) text than the worker retry would for the same audio.

**Suggested fix:** build the in-process retry kwargs through the same helper the first pass
uses, or add `if self._hotwords: kwargs2["hotwords"] = self._hotwords`.

---

## 7. nit — abort is a no-op against the worker decode; "identical results on both runtimes" is overstated for cancellation

**Where:** `visper/transcriber.py:585-696` (`_transcribe_via_worker` — `is_aborted` checked
only at entry and during the post-decode replay loop), `README.md:37-42`,
`visper/server.py` `/transcribe/stream` `_disconnect_watcher`.

**What's wrong:** the JSON-line protocol has no cancel message, so a client disconnect
mid-file lets the worker decode the *entire* file before the replay loop notices the abort;
the in-process path breaks out of the segment generator promptly and returns partial text.
For `bucket="streaming"` chunks are tiny so it doesn't matter, but for a long `/transcribe/
stream` upload the default runtime burns the full decode after the client is gone. §1
already hedges callback *timing*; it doesn't mention that abort semantics differ.

**Suggested fix:** acceptable to ship (inherent to a batch worker) — just add one sentence
to §1/§4 noting cancellation only takes effect between requests on the worker runtime. A
real fix would need a worker-side interrupt.

---

## 8. nit — `/transcribe/stream` leaks the temp upload if the client never consumes the response

**Where:** `visper/server.py` `transcribe_stream` — `tmp_path.unlink(missing_ok=True)` lives
only in the `_generate()` `finally`; the `_run()` worker thread never unlinks and never
aborts on its own.

**What's wrong:** if the SSE response body is never iterated (client drops the connection
between POST and first read), `_generate()`'s `finally` may not run, the temp file stays,
and the detached `_run` thread transcribes the whole file for nothing.

**Suggested fix:** unlink in `_run()`'s `finally` too (idempotent with `missing_ok=True`),
and have `_run` check `abort_event` between chunks via the `is_aborted` it already passes.

---

## 9. nit — `sse-starlette` is a declared dependency and a README claim, but nothing imports it

**Where:** `pyproject.toml` `[server]` (`sse-starlette>=1.6`), `requirements.txt`,
`README.md:130` ("FastAPI + `sse-starlette`"). `grep -rn "sse_starlette" visper/` → no hits;
both SSE endpoints use `fastapi.responses.StreamingResponse`.

**Suggested fix:** drop the dep and the README row, or actually use `EventSourceResponse`.
Harmless but it's a "claim the code contradicts."

---

## 10. nit — model-size figure inconsistent

`README.md:128` "~220 MB" vs `translate.py:13` / `:14` "~210 MB" vs finding #4 wording.
Pick one.

---

## 11. nit — in-process he→en preview reuse can serve stale pre-retry translations

**Where:** `visper/transcriber.py:544-547` (`len(preview_en) == len(segments) and None not
in preview_en` → reuse).

**What's wrong:** on the in-process path `_preview_cb` fires during the *first* decode; if a
confidence-retry then runs, `he.segments` are post-retry. Usually the counts differ and the
batch re-translate path is taken — correct. But if pre- and post-retry segment counts
coincide, the guard reuses English translated from the *discarded* pre-retry Hebrew. The
worker path is not affected (it replays post-retry segments). Low probability, low impact.

**Suggested fix:** gate the reuse on identity of the Hebrew text too, or skip reuse whenever
`he.tier_used != params.tier_used` (retry happened).

---

## Section verdicts

- **`visper/transcriber.py`** — the 7a530b3 replay is correct: no double-firing (single
  runtime path executes; `_translate_hebrew` invokes the outer `on_segment` only via
  `_preview_cb`), replay uses the post-retry `response["segments"]`, and on the worker path
  translation happens exactly once per final segment (previews built during replay, then
  reused via the `None not in preview_en` guard whose length check now matches because the
  replay is post-retry). `.text` from `_translate_hebrew` is single-language English
  (` " ".join` of `en` segment texts). Abort paths are safe (early empty result, early
  `return he`). Issues: #6 (hotwords), #7 (abort vs worker), #11 (reuse edge).
- **`visper/translate.py`** — SHA-256 pinning, `_safe_extract` (symlink/hardlink reject +
  pre-check + `filter="data"`), `_model_present()`/`he_en_supported()` gating, and
  "degrade never fail" are sound: a 404 raises `HTTPError` → caught in
  `get_hebrew_english_translator` → `_load_failed=True`, returns `None`, caller falls back to
  Whisper translate; `task=translate` never 500s from stage 2. Issues: #4 (health
  advertising), #5 (vocab filename), #10 (size).
- **`visper/server.py`** — `--host` default `127.0.0.1` (env `VISPER_HOST` override), 0.0.0.0
  warns; CORS locked to explicit localhost origins; upload size-capped before full write
  with `BaseException` cleanup; `/health` → 503 JSONResponse on error; `/transcribe` unlinks
  in `finally`. Clean apart from #3 (web packaging), #8 (stream temp leak), #9 (sse dep).
- **`visper/worker.py` + `visper/params.py` vs `benchmark.py:_auto_accuracy_tier`** — the two
  tier policies now agree: both iterate `("accurate","balanced","light")`, both use
  `RTF_BUDGET=0.85` and `TIER_RTF_MULTIPLIERS`, `benchmark` imports the constants from
  `params`. The `rtf > 0.85 → fast` branch in `get_params` is subsumed by the loop. Worker
  JSON-line protocol is coherent; retry is parent-side; temp `.npy` ownership is correct
  (worker unlinks on success, parent `_worker_roundtrip._cleanup` on every error, retry
  writes a fresh file). Only reporting bug is #2 (stale stored field). Not a new defect but
  worth noting: a worker that dies mid-session has **no respawn** — subsequent requests
  raise `RuntimeError` forever until the process restarts; the default runtime therefore
  has a weaker resilience story than README §4 implies.
- **`README.md`** — tier table (beam 5/3/2/1, RTF bands `<0.19` / `0.19–0.47` /
  `0.47–0.63` / `>0.63`) matches `params.py` exactly. §6 RTF values match the local
  `benchmark_results.json` (0.558 / 0.188 / 0.171 / 0.158 / 0.792); the `short` auto-tier
  cell (`light`) matches current policy but not the stale stored field (#2). Fallback-chain
  text matches `benchmark.py:_build_fallback_chain` (`CUDA → OV HETERO → OV iGPU → OV CPU →
  CT2 CPU`, MLX prepended as primary on Apple). Issues: #1 (traceability), #2, #3, #7, #9,
  #10.

## Resolution (2026-09-07, same session)

| # | Finding | Resolution | Commit |
|--:|---|---|---|
| 1 | §5/§6 numbers untraceable; stale `eval_results.md` | committed `docs/benchmarks/` evidence bundle; `--rescore` reproduces §5; deleted the stale file | `c493033` |
| 2 | `--report` prints stored (stale) tier | `format_report_table()` recomputes from RTF | `65d96bd` |
| 3 | web UI absent from sdist/wheel | `git mv web visper/web`; package-data; verified in both artifacts | `cefb463` |
| 4 | `/health` over-advertises he→en | added `he_en_pending_download`; UI tooltip; `install.py` pre-fetch | `3f0ab64` |
| 5 | vocab filename comment vs `_MODEL_FILES` | comment corrected; tarball verified to contain all five `_MODEL_FILES` | `b945597` |
| 6 | in-process retry drops `hotwords` | added to `kwargs2` | `afef98e` |
| 7 | abort is a no-op vs worker decode | documented in README §1 (between-request cancellation) | `ec21480` |
| 8 | `/transcribe/stream` temp-file leak | worker thread is sole owner, unlinks in its `finally` | `bd00e0c` |
| 9 | `sse-starlette` declared, unused | dropped from deps + README | `c8b00dd` |
| 10 | 210/220 MB inconsistency | standardised on ~210 MB | `b945597` / `ec21480` |
| 11 | preview reuse can serve pre-retry translations | reuse gated on preview↔final Hebrew text match | `b945597` |
| — | venv-worker not respawned | documented as a known limitation in README §4 | `ec21480` |

Still blocking merge: the `mt-he-en-v1` GitHub release asset must be published
(the tarball is verified — sha256 matches `_ASSET_SHA256`, contents match
`_MODEL_FILES` — so it is correct by construction once uploaded).

## Merge recommendation

**Do not merge as-is, but the blockers are documentation/packaging, not engine
correctness.** The core change under review — the 7a530b3 worker-segment replay and its
interaction with `_translate_hebrew`, the confidence-retry, and abort — is correct, and the
params/benchmark policy duplication is genuinely reconciled. `pytest` (80) and the sdist
build are green. What isn't ship-ready: the README's headline accuracy/performance numbers
are not backed by anything in the repo and the one committed eval file reports materially
worse numbers (#1); the shipped sdist can't serve its own web UI (#3); `visper-benchmark
--report` contradicts the README (#2); and the flagship he→en feature will silently
under-deliver on every first run because the release asset is missing and `/health`
over-promises (#4, #5). Land #1–#5 (commit an evidence bundle, package `web/`, recompute the
report tier, publish the MT asset + fix the hash/comment, make `/health` honest about the
download), fix or explicitly document #6–#9, then merge.
