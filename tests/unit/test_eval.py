"""visper.eval helpers — pairing, reference cleaning, scoring (no model needed)."""
from visper.eval import (
    _clean_reference,
    _for_scoring,
    _metrics,
    _summarize,
    find_pairs,
    format_markdown,
    rescore,
)


def test_find_pairs_nested_layout(tmp_path):
    (tmp_path / "audios").mkdir()
    (tmp_path / "refs").mkdir()
    (tmp_path / "audios" / "a.wav").write_bytes(b"x")
    (tmp_path / "audios" / "b.mp3").write_bytes(b"x")
    (tmp_path / "audios" / "orphan.wav").write_bytes(b"x")
    (tmp_path / "refs" / "a.txt").write_text("ref a", encoding="utf-8")
    (tmp_path / "refs" / "bND.txt").write_text("ref b", encoding="utf-8")

    pairs = find_pairs(tmp_path)
    stems = sorted(p[0].stem for p in pairs)
    assert stems == ["a", "b"]  # orphan skipped


def test_find_pairs_flat_layout(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"x")
    (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 1 and pairs[0][0].name == "x.wav"


def test_clean_reference_strips_annotations_and_normalizes():
    raw = "שלום [לא ברור] <צחוק> עולם\nמה שלומך"
    out = _clean_reference(raw, "he")
    assert "[" not in out and "<" not in out
    assert "שלום" in out and "עולם" in out
    assert "\n" not in out


def test_format_markdown_shape():
    report = {"language": "he", "model": "m", "tier": "balanced",
              "datasets": {"coish": {"n": 12, "wer": 0.409, "cer": 0.274,
                                     "wer_min": 0.20, "wer_median": 0.40,
                                     "wer_max": 0.66, "rows": []}}}
    md = format_markdown(report)
    assert "| coish | 12 | 0.409 | 0.274 | 0.20 / 0.40 / 0.66 |" in md
    assert "balanced tier" in md


def test_for_scoring_is_symmetric_and_punctuation_insensitive():
    # A correctly transcribed sentence that only differs by punctuation/case
    # the reference never had must score WER 0.
    ref = "שלום עולם מה שלומך"
    hyp = "שלום עולם, מה שלומך?"
    assert _for_scoring(ref) == _for_scoring(hyp)
    w, c = _metrics(ref, hyp)
    assert w == 0.0 and c == 0.0


def test_for_scoring_keeps_hebrew_and_digits():
    assert _for_scoring("תיקון מס 13") == "תיקון מס 13"


def test_rescore_recomputes_without_transcription():
    sidecar = {
        "language": "he", "tier": "balanced", "model": "m",
        "datasets": {
            "d": [
                {"file": "a.wav", "wer": 9.9, "cer": 9.9, "sec": 1.0,
                 "ref": "אחת שתיים שלוש", "hyp": "אחת שתיים שלוש"},
                {"file": "b.wav", "wer": 9.9, "cer": 9.9, "sec": 1.0,
                 "ref": "אחת שתיים שלוש", "hyp": "אחת שתיים ארבע"},
            ]
        },
    }
    report = rescore(sidecar)
    d = report["datasets"]["d"]
    assert d["n"] == 2
    assert d["rows"][0]["wer"] == 0.0
    assert abs(d["rows"][1]["wer"] - 1 / 3) < 1e-9
    assert abs(d["wer"] - 1 / 6) < 1e-9


def test_summarize_spread():
    rows = [{"wer": w, "cer": 0.1} for w in (0.1, 0.2, 0.3, 0.4, 0.9)]
    s = _summarize(rows)
    assert s["wer_min"] == 0.1 and s["wer_max"] == 0.9
    assert s["wer_median"] == 0.3
