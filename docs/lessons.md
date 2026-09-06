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
