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
dedicated he→en model running under CTranslate2 in the worker process.

```
audio ──▶ ivrit-ai (Whisper/CT2) ──▶ Hebrew segments ──┬──▶ return {he text, segments}
                                                        │
                              task=translate only ──────▶ he→en MT (OPUS-MT/CT2)
                                                        │      per-segment
                                                        └──▶ return {en text, segments(en, he timestamps),
                                                                     he_text}   (bilingual)
```

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
  dir is published as a GitHub release asset and downloaded on first use
  (`gh` is authenticated); `~/.visper/models/opus-mt-tc-big-he-en-ct2/`.
- Runtime dependency added: **`sentencepiece` only**.

**Verified inference recipe** (quality confirmed on Hebrew test sentences):

```python
src = sp_src.encode(text, out_type=str) + ["</s>"]          # the </s> is essential —
res = translator.translate_batch([src], beam_size=4)        # without it the model loops
out = [t for t in res[0].hypotheses[0] if t not in ("</s>", "<pad>")]
english = sp_tgt.decode(out)
```

No `>>eng<<` prefix (bilingual model; adding it *degraded* output in testing).
Translate one Whisper segment per batch entry — segments are already
sentence-sized, and Marian degrades on multi-sentence input.

### Worker protocol

New action in `visper/worker.py`:

```json
{"action": "translate_text", "segments": [{"start", "end", "text"}], "src": "he", "tgt": "en"}
→ {"status": "ok", "segments": [{"start", "end", "text"}]}   // text now English
```

The MT model is lazy-loaded on first `translate_text` and kept resident
(~small). It never displaces the ASR model — separate slot.

### Transcriber / API

- `Transcriber._transcribe_via_worker`: when `task == "translate"`, after the
  normal round-trip, send a second `translate_text` round-trip with the Hebrew
  segments, swap the segment texts, rebuild `TranscriptResult.text` from the
  English segments, and attach `he_text` (the original) for a bilingual view.
- In-process path (`_transcribe_direct`): same, calling a stage-2 helper that
  runs the CT2 translator in-process (needs the CT2 model present; degrade to
  the old behaviour with a warning if missing).
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
