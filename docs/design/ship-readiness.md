# Ship-Readiness — Why

## Problem
Visper works but is not portfolio-worthy: the venv-worker (default) runtime silently
drops `initial_prompt` / `hotwords` / confidence-retry / audio pre-processing /
cancellation; the config system is 8 ad-hoc YAML parsers that drift from the shipped
`config.yaml`; streaming has 3 divergent chunkers; the README carries fabricated
numbers (`WER 0.179 / CER 0.084 on 221 files` — no eval harness exists; an RTF table
that contradicts the `config.yaml` comment for the same device); the web UI is not
served by the server yet the server binds `0.0.0.0` with `CORS *` and no auth; there
is no LICENSE, no CI, no pytest with assertions, no pinned deps, and `config.yaml` is
not packaged so a non-editable install silently runs on code defaults that differ
from what ships.

Full defect inventory: `docs/ship-readiness-audit.md` (6 parallel audits, 2026-09-06).

## Who it's for
A reviewer reading this repo as an engineering portfolio piece, and any user doing a
plain `pip install`.

## Success criteria
- Every number in `README.md` is reproducible by a committed harness the user has run.
- Default runtime path and in-process path produce identical features.
- One config loader, one streaming chunker, one CUDA-DLL helper, one output formatter.
- `pip install .` into a clean venv works (config packaged) and `pytest` passes in CI.
- `visper-server` serves the UI and does not expose the GPU to the LAN / arbitrary sites by default.
- No behavior regressions: existing CLI flags and Python API calls keep working.

## Hard constraints
- No invented numbers. README performance/accuracy numbers stay `TBD` until the user
  runs `visper-eval` / `visper-benchmark --report` and pastes output.
- Smallest safe change; no drive-by refactors outside a phase's scope.
- No new runtime dependencies without asking.
- Atomic commits: one per numbered plan step.

## Phased plan
0. Safety net & hygiene — LICENSE, CI, pytest w/ assertions, pin+reconcile deps,
   package `config.yaml`, fix `_REPEAT_CHAR` number-corruption bug, version alignment.
1. Config unification — one `visper/config.py`; kill drift + dead keys; remove
   `bucket_accuracy_overrides` from shipped config; guard benchmark side-effect.
2. Inference-path convergence — venv-worker forwards prompt/hotwords/retry/pre-proc/abort;
   collapse the 4 duplicated load/infer blocks.
3. Streaming convergence — `/ws/live` delegates to `LiveStreamer`; cache invalidation
   moves into `ModelRouter`; fix idle-unload stream death.
4. CUDA-DLL dedup — one `visper/_cuda.py`, 8 call sites.
5. Mechanical dedup — shared SRT/VTT/JSON formatter, duration + bucket helpers.
6. Web UI + server hardening — serve UI at `/`; localhost bind + CORS by default;
   fix XSS; upload cap; lifespan handler; vendor `lucide`.
7. Eval + benchmark harnesses — `visper/eval.py` + `visper-eval` (local WER/CER over
   `datasets/`, single config, imports the shipped normalizer, dataset-labelled
   markdown table); `visper-benchmark --report` for the RTF table. No Colab notebook —
   the 18-way sweep is dropped; the raw `speed_test*.ipynb` stay gitignored as history.
8. Docs rewrite (LAST) — README as engineering blueprint with real numbers only;
   ARCHITECTURE.md corrected.

## Behavior changes (user-authorized 2026-09-06, proceed unless objected)
1. Ship `config.yaml` without `bucket_accuracy_overrides` — restores auto tier ladder.
2. `/ws/live` → `LiveStreamer` — server streaming honors streaming config keys.
3. `visper-server` serves UI at `/`, defaults to `127.0.0.1` + localhost CORS
   (`--host 0.0.0.0` opt-in).
4. `transcribe()` gains optional `language=` param (current behavior when omitted).
