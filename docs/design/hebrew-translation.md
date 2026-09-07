# Hebrew → English translation

## Why

**Problem.** Visper's Hebrew route uses `ivrit-ai/whisper-large-v3-turbo-ct2`,
a fine-tune specialised for Hebrew *transcription*. Its `task=translate` output
is poor, so the app currently disables translation for Hebrew entirely
(`/health` returns `no_translate: ["he"]`, the UI hides the toggle). A
Hebrew-first transcription tool that cannot produce an English version of a
Hebrew recording is missing an obvious, expected feature — journalists,
researchers and legal users routinely need both.

**Hard constraint.** The Hebrew transcript must keep coming from the ivrit-ai
fine-tune — that is the whole point of the project and the source of its
accuracy edge. Translation must not degrade the Hebrew output.

**Who it's for.** Anyone transcribing Hebrew audio who also needs an English
rendering: bilingual meeting notes, subtitling, sharing an interview with a
non-Hebrew-speaking colleague.

**Success criteria.**
- `task=translate, language=he` returns fluent English, clearly better than the
  fine-tune's own translate output.
- The Hebrew transcript is byte-identical to a plain `task=transcribe` run —
  translation is strictly additive.
- English segments carry the Hebrew segment timestamps, so SRT/VTT/JSON export
  works and a bilingual view is possible.
- Added latency is a small fraction of the ASR pass (text MT, not a second
  audio pass).
- Runs fully offline inside the existing venv-worker.

**Hard constraints.**
- No second Whisper pass (that path — routing to `large-v3` — was rejected: 2×
  ASR time, model swap, and it still doesn't give an ivrit-quality Hebrew
  transcript alongside).
- Minimal new dependencies. Reuse the CTranslate2 runtime that faster-whisper
  already pulls in.

## Design

Two-stage pipeline. Stage 1 is unchanged ivrit-ai transcription. Stage 2 takes
the Hebrew **segments** and translates each one's text to English with a
dedicated he→en model under CTranslate2.

```
audio ──▶ ivrit-ai (Whisper/CT2) ──▶ Hebrew segments ──┬──▶ return {he text, segments}
                                                        │
                              task=translate only ──────▶ he→en MT (OPUS-MT/CT2)
                                                        │      batch, one segment per entry
                                                        └──▶ return {en text, segments(en, he timestamps),
                                                                     he_text}   (bilingual)
```

**Where stage 2 runs — the main process, not the worker.** The venv-worker
exists to isolate the *faster-whisper* runtime (no 3.13/3.14 wheels). But
`ctranslate2` is already a declared transitive dependency of `faster-whisper`
in `pyproject.toml`, and it imports and runs on the host Python (verified:
`ctranslate2 4.7.1` on 3.14, int8 compute types available). So stage 2 is a
thin wrapper around `Transcriber.transcribe()`:

- `task == "translate"` and Hebrew and MT model present → recurse into
  `transcribe(..., task="transcribe", language="he")` (same `bucket` and
  `_tier_override`, so the pinned tier is preserved), then translate the
  Hebrew segments and return an English `TranscriptResult` with `he_text` set.
- This covers the in-process **and** venv-worker runtimes with one code path —
  no new worker action, no protocol change, no double implementation.
- If the MT model is absent / fails to import: fall through to the existing
  behaviour (Whisper's own `task=translate`). Never hard-fail.

### Model  *(finalised + verified)*

`Helsinki-NLP/opus-mt-tc-big-he-en` (Marian, 0.2 B params, BLEU 53.8 Tatoeba /
44.1 FLORES), converted to CTranslate2 int8. SentencePiece tokenisation loaded
directly from the model's `source.spm` / `target.spm` — no `transformers` /
`torch` at runtime.

- Converted with `ct2-transformers-converter --model
  Helsinki-NLP/opus-mt-tc-big-he-en --quantization int8`; `source.spm`,
  `target.spm`, `vocab.json` copied in from the HF repo (the converter does not
  copy them). Result: `model.bin` ~241 MB + vocab + spm.
- Conversion needs `transformers` + `torch`, **build-time only**. The converted
  dir is packed as `opus-mt-tc-big-he-en-ct2.tar.gz` and attached to a GitHub
  release on the Visper repo (`gh` is authenticated as `Avrhambi`). On first
  `translate` it is downloaded + extracted to
  `~/.visper/models/opus-mt-tc-big-he-en-ct2/`, then loaded offline forever
  after. HF Hub was considered and declined — it would put a low-value format
  conversion under a personal ML identity; a release asset keeps the artifact
  with the project.
- Runtime dependency added: **`sentencepiece` only** (ctranslate2 is already
  transitive via faster-whisper).

**Verified inference recipe** (quality confirmed on Hebrew test sentences):

```python
src = sp_src.encode(text, out_type=str) + ["</s>"]          # the </s> is essential —
res = translator.translate_batch([src], beam_size=4)        # without it the model loops
out = [t for t in res[0].hypotheses[0] if t not in ("</s>", "<pad>")]
english = sp_tgt.decode(out)
```

No `>>eng<<` prefix (bilingual model; adding it *degraded* output in testing).
One Whisper segment per batch entry — segments are already sentence-sized, and
Marian degrades on multi-sentence input. All segments go in a single
`translate_batch` call so the model load cost (~5 s cold) is paid once.

### `visper/translate.py`

- `HebrewEnglishTranslator` — loads `ctranslate2.Translator` + two
  `SentencePieceProcessor`s from `~/.visper/models/opus-mt-tc-big-he-en-ct2/`.
  `.translate(list[str]) -> list[str]`, batched, `</s>`-terminated.
- `get_hebrew_english_translator() -> HebrewEnglishTranslator | None` —
  module-level cached singleton. Returns `None` (and logs once) if the model
  dir is missing, download fails, or `ctranslate2` / `sentencepiece` won't
  import. Callers treat `None` as "translation unavailable".
- `ensure_model() ` — download + extract the release asset if the dir is
  absent. Skipped entirely when the dir exists (offline).

### Transcriber / API

- `Transcriber.transcribe()` gains a guard at the top: if
  `task == "translate"` and the effective language is `he` and
  `get_hebrew_english_translator()` is not `None`, recurse with
  `task="transcribe", language="he"` (passing `bucket` and `_tier_override`
  through unchanged), then translate the resulting segments, rebuild `.text`
  from the English segments, set `.he_text`, and fire `on_segment` **once per
  English segment** after stage 2. Translate mode is therefore final-result
  shaped: the segment callbacks carry English, matching `.text`. During the
  stage-1 Whisper decode no segments stream (documented trade-off — the MT
  pass is a small fraction of ASR time).
- `TranscriptResult` gains `he_text: str = ""` (last field; the only legal
  position — every existing field is non-default).
- `api.transcribe_chunked(..., task="translate", language="he")` — entry point
  unchanged. `_norm_lang` already switches to `"en"` for the post-normaliser.

### Routing / health

- `resolve_model_id` unchanged — stage 1 is still ivrit-ai.
- `server.py`: drop `he` from `no_translate` **iff** the CT2 he→en model is
  available; keep the guard so a missing model degrades cleanly.
- UI: the translate toggle re-appears for Hebrew via the existing
  `_refreshTranslateAvailability()` path — no UI code change.

### Fallback

If the he→en CT2 model is absent or fails to load: log a warning, fall back to
the current behaviour (`task=translate` passes through to the fine-tune, or the
UI keeps the toggle hidden). Never hard-fail a transcription because translation
is unavailable.

### Non-goals

- English → Hebrew (Whisper is EN-source only for `translate`; reverse MT is a
  separate model — out of scope).
- Languages other than he→en in stage 2 (fr/es/de/ru/ar already translate well
  via `large-v3` in one pass — unchanged).
- Sentence-context translation across segment boundaries (per-segment is enough
  for v1; revisit if quality needs it).
