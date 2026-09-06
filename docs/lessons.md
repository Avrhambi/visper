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
