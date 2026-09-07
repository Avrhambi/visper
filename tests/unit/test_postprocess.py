"""Unit tests for visper.postprocess — pure regex, no models, hermetic."""
from visper.postprocess import normalize_hebrew, normalize_text


class TestRepeatCharCollapse:
    r"""The _REPEAT_CHAR rule must collapse hallucinated letter runs without
    corrupting numbers — regression for the old r'(.)\1{2,}' pattern that
    turned '20000' into '20' and '10:00:00' into '10:0:0'."""

    def test_digit_runs_survive(self):
        assert normalize_hebrew("המחיר הוא 20000 שקל") == "המחיר הוא 20000 שקל"
        assert normalize_text("1000000 downloads", language="en") == "1000000 downloads"

    def test_time_string_survives(self):
        assert "10:00:00" in normalize_hebrew("הפגישה ב 10:00:00")

    def test_hallucinated_letter_run_collapses(self):
        assert normalize_hebrew("אאאאאא") == "א"
        assert normalize_text("hmmmmmm yes", language="en") == "hm yes"

    def test_run_below_threshold_is_kept(self):
        # 3 identical letters (מממ) is below the 4+ threshold — left intact
        assert normalize_hebrew("מממן") == "מממן"


class TestHebrewNormalization:
    def test_strips_nikud(self):
        assert normalize_hebrew("שָׁלוֹם") == "שלום"

    def test_trailing_comma_stripped(self):
        assert normalize_hebrew("שלום עולם,") == "שלום עולם"

    def test_script_boundary_spacing(self):
        assert normalize_hebrew("שלוםhello") == "שלום hello"

    def test_letter_digit_spacing(self):
        assert normalize_hebrew("גרסה3") == "גרסה 3"

    def test_repeated_word_collapse(self):
        assert normalize_hebrew("כן כן כן כן") == "כן"

    def test_youtube_hallucination_phrase_removed(self):
        assert normalize_hebrew("תודה שצפיתם").strip() == ""


class TestNonHebrewNormalization:
    def test_strips_injected_hebrew_chars(self):
        assert normalize_text("hello שלום world", language="en") == "hello  world"

    def test_english_untouched_otherwise(self):
        assert normalize_text("The quick brown fox.", language="en") == "The quick brown fox."
