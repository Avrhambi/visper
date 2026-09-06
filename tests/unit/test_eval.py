"""visper.eval helpers — pairing and reference cleaning (no model needed)."""
from visper.eval import _clean_reference, find_pairs, format_markdown


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
              "datasets": {"coish": {"n": 12, "wer": 0.409, "cer": 0.274, "rows": []}}}
    md = format_markdown(report)
    assert "| coish | 12 | 0.409 | 0.274 |" in md
    assert "balanced tier" in md
