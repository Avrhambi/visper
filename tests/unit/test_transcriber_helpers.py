"""Pure helpers on Transcriber that don't need a model loaded."""
from visper.transcriber import Transcriber


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
