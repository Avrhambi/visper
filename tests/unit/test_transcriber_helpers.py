"""Pure helpers on Transcriber that don't need a model loaded."""
from visper.transcriber import Transcriber, TranscriptResult


class TestConfidenceOkDicts:
    """Worker path gets segments as dicts ({'start','end','confidence'});
    the retry gate must weigh them the same way as the in-process Segment path."""

    def test_empty_is_ok(self):
        assert Transcriber._confidence_ok_dicts([], -1.0) is True

    def test_high_confidence_passes(self):
        segs = [{"start": 0.0, "end": 2.0, "confidence": -0.3},
                {"start": 2.0, "end": 4.0, "confidence": -0.4}]
        assert Transcriber._confidence_ok_dicts(segs, -0.8) is True

    def test_low_confidence_fails(self):
        segs = [{"start": 0.0, "end": 2.0, "confidence": -1.2},
                {"start": 2.0, "end": 4.0, "confidence": -1.5}]
        assert Transcriber._confidence_ok_dicts(segs, -0.8) is False

    def test_duration_weighted(self):
        # one long bad segment outweighs a short good one
        segs = [{"start": 0.0, "end": 0.2, "confidence": 0.0},
                {"start": 0.2, "end": 20.0, "confidence": -1.0}]
        assert Transcriber._confidence_ok_dicts(segs, -0.5) is False


class _FakeParams:
    tier_used = "balanced"
    confidence_retry_enabled = False
    log_prob_threshold = -1.0

    def as_transcribe_kwargs(self):
        return {"beam_size": 3}


class TestWorkerSegmentReplay:
    """The worker returns every segment in one batch, so _transcribe_via_worker
    replays them through the caller's on_segment — otherwise /transcribe returns
    an empty segment list and /transcribe/stream emits no per-segment events on
    the default (venv-worker) runtime."""

    def _make(self):
        tr = Transcriber.__new__(Transcriber)
        tr._language = "he"
        tr._denoise = tr._normalize_volume = tr._highpass = False
        tr._hotwords = None   # type: ignore[assignment]
        tr._config_label = "c"
        tr._backend_type = "cpu"
        tr._get_duration = lambda s: 2.0   # type: ignore[method-assign]
        return tr

    _RESP = {
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " hello", "confidence": -0.2},
            {"start": 1.0, "end": 2.0, "text": " world", "confidence": -0.3},
        ],
        "audio_duration": 2.0,
    }

    def test_replays_each_segment(self):
        tr = self._make()
        tr._worker_roundtrip = lambda *a, **k: self._RESP   # type: ignore[method-assign]
        seen = []
        res = tr._transcribe_via_worker(
            "a.wav", "medium", _FakeParams(), False, 500, 400,
            language="he", on_segment=seen.append,
        )
        assert [s["text"] for s in seen] == [" hello", " world"]
        assert res.segments == self._RESP["segments"]

    def test_no_callback_is_fine(self):
        tr = self._make()
        tr._worker_roundtrip = lambda *a, **k: self._RESP   # type: ignore[method-assign]
        res = tr._transcribe_via_worker(
            "a.wav", "medium", _FakeParams(), False, 500, 400, language="he",
        )
        assert res.text == "hello world"

    def test_abort_before_replay_emits_nothing(self):
        tr = self._make()
        tr._worker_roundtrip = lambda *a, **k: self._RESP   # type: ignore[method-assign]
        seen = []
        tr._transcribe_via_worker(
            "a.wav", "medium", _FakeParams(), False, 500, 400, language="he",
            on_segment=seen.append, is_aborted=lambda: True,
        )
        assert seen == []


class _FakeMT:
    """Maps each non-blank Hebrew segment to a distinct English string."""
    _EN = {"שלום עולם": "hello world", "מה שלומך": "how are you"}

    def translate(self, texts):
        return [self._EN.get(t.strip(), "x") if t and t.strip() else t for t in texts]


def _fake_he_result():
    return TranscriptResult(
        text="שלום עולם מה שלומך",
        segments=[
            {"start": 0.0, "end": 1.0, "text": " שלום עולם", "confidence": -0.2},
            {"start": 1.0, "end": 2.0, "text": " מה שלומך", "confidence": -0.3},
        ],
        audio_duration=2.0, elapsed=1.0, rtf=0.5, config_label="c",
        backend="venv-worker/cpu", tier_used="balanced", whisper_params={"a": 1},
    )


class TestTranslateHebrew:
    """Stage 2: Hebrew transcript in, English TranscriptResult out."""

    def _run(self, he_result=None, **kw):
        tr = Transcriber.__new__(Transcriber)
        result = he_result if he_result is not None else _fake_he_result()

        def fake_transcribe(source, *, on_segment=None, **k):  # noqa: ARG001
            assert k.get("task") == "transcribe"  # guard must not re-enter
            if on_segment is not None:
                for s in result.segments:
                    on_segment(dict(s))
            return result

        tr.transcribe = fake_transcribe   # type: ignore[method-assign]
        return tr._translate_hebrew("audio.wav", "medium", _FakeMT(), **kw)

    def test_segments_become_english_and_text_matches(self):
        res = self._run()
        assert [s["text"] for s in res.segments] == ["hello world", "how are you"]
        assert res.text == "hello world how are you"

    def test_original_hebrew_is_preserved(self):
        res = self._run()
        assert res.he_text == "שלום עולם מה שלומך"
        assert res.segments[0]["he_text"] == " שלום עולם"

    def test_on_segment_fires_once_per_english_segment(self):
        seen = []
        self._run(on_segment=seen.append)
        assert [s["text"] for s in seen] == ["hello world", "how are you"]

    def test_backend_records_the_mt_stage(self):
        assert self._run().backend == "venv-worker/cpu+opus-mt-he-en"

    def test_empty_transcript_does_not_crash(self):
        empty = TranscriptResult(text="", segments=[], audio_duration=0.0, elapsed=0.0,
                                 rtf=0.0, config_label="c", backend="b",
                                 tier_used="fast", whisper_params={})
        res = self._run(he_result=empty)
        assert res.text == "" and res.segments == [] and res.he_text == ""

    def test_retry_changed_segments_are_retranslated_not_reused(self):
        # Preview streams the pre-retry Hebrew; a confidence-retry then replaces
        # he.segments. Counts happen to match, but the text differs — the reuse
        # guard must fall through to a batch re-translate of the final segments.
        final = TranscriptResult(
            text="שלום עולם מה שלומך",
            segments=[
                {"start": 0.0, "end": 1.0, "text": " שלום עולם", "confidence": -0.1},
                {"start": 1.0, "end": 2.0, "text": " מה שלומך", "confidence": -0.1},
            ],
            audio_duration=2.0, elapsed=1.0, rtf=0.5, config_label="c",
            backend="venv-worker/cpu", tier_used="accurate", whisper_params={},
        )
        pre_retry = [
            {"start": 0.0, "end": 1.0, "text": " גיבריש אחד", "confidence": -2.0},
            {"start": 1.0, "end": 2.0, "text": " גיבריש שתיים", "confidence": -2.0},
        ]
        tr = Transcriber.__new__(Transcriber)

        def fake_transcribe(source, *, on_segment=None, **k):  # noqa: ARG001
            if on_segment is not None:
                for s in pre_retry:            # preview sees the discarded decode
                    on_segment(dict(s))
            return final                        # result carries the retry decode

        tr.transcribe = fake_transcribe          # type: ignore[method-assign]
        res = tr._translate_hebrew("a.wav", "medium", _FakeMT(), on_segment=lambda s: None)
        assert [s["text"] for s in res.segments] == ["hello world", "how are you"]

    def test_mt_failure_returns_the_hebrew_transcript(self):
        class BoomMT:
            def translate(self, texts):
                raise RuntimeError("ct2 exploded")

        tr = Transcriber.__new__(Transcriber)
        tr.transcribe = lambda *a, **k: _fake_he_result()   # type: ignore[method-assign]
        res = tr._translate_hebrew("a.wav", "medium", BoomMT())
        # translation failed -> Hebrew transcript comes back untouched, no raise
        assert res.text == "שלום עולם מה שלומך"
        assert res.he_text == ""  # this IS the plain transcript, not a translation
