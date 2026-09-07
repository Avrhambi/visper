"""Unit tests for visper.params — tier selection math, hermetic via monkeypatch."""
import pytest

from visper import params
from visper.params import (
    RTF_BUDGET,
    TIER_RTF_MULTIPLIERS,
    estimate_rtf_cost,
    get_params,
    get_params_for_tier,
    next_tier,
)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Neutralise on-disk config + benchmark results and the first-call log latch."""
    monkeypatch.setattr(params, "_load_user_config", lambda: {})
    monkeypatch.setattr(params, "_rtf_headroom", lambda hw, bucket: None)
    monkeypatch.setattr(params, "_logged_first_call", False, raising=False)
    yield


def _set_headroom(monkeypatch, rtf):
    monkeypatch.setattr(params, "_rtf_headroom", lambda hw, bucket: rtf)


class TestRtfCost:
    def test_multipliers_are_stable(self):
        assert TIER_RTF_MULTIPLIERS == {
            "fast": 1.0, "light": 1.35, "balanced": 1.8, "accurate": 4.5,
        }

    def test_estimate(self):
        assert estimate_rtf_cost(0.1, "accurate") == pytest.approx(0.45)


class TestAutoTierSelection:
    """Locks the real RTF→tier thresholds (budget 0.85, the multipliers above)."""

    def test_no_benchmark_falls_back_to_fast(self, monkeypatch):
        _set_headroom(monkeypatch, None)
        assert get_params("medium", {}).tier_used == "fast"

    def test_near_realtime_forces_fast(self, monkeypatch):
        _set_headroom(monkeypatch, RTF_BUDGET + 0.01)
        assert get_params("medium", {}).tier_used == "fast"

    @pytest.mark.parametrize("base_rtf, expected", [
        (0.15, "accurate"),   # 0.15 * 4.5 = 0.675 < 0.85
        (0.30, "balanced"),   # 0.30 * 1.8 = 0.54  < 0.85 ; * 4.5 = 1.35 over
        (0.50, "light"),      # 0.50 * 1.35 = 0.675 < 0.85 ; * 1.8 = 0.90 over
        (0.70, "fast"),       # 0.70 * 1.35 = 0.945 over → fast
    ])
    def test_threshold_ladder(self, monkeypatch, base_rtf, expected):
        _set_headroom(monkeypatch, base_rtf)
        assert get_params("medium", {}).tier_used == expected


class TestExplicitMode:
    def test_explicit_tier_passthrough(self, monkeypatch):
        monkeypatch.setattr(params, "_load_user_config", lambda: {"accuracy_mode": "balanced"})
        p = get_params("medium", {})
        assert p.tier_used == "balanced"
        assert p.auto_selected is False
        assert p.beam_size == 3

    def test_bucket_override_outranks_mode(self, monkeypatch):
        monkeypatch.setattr(params, "_load_user_config", lambda: {
            "accuracy_mode": "fast",
            "bucket_accuracy_overrides": {"long": "accurate"},
        })
        assert get_params("long", {}).tier_used == "accurate"
        assert get_params("short", {}).tier_used == "fast"


class TestInvariants:
    def test_streaming_never_conditions_on_prev_text(self, monkeypatch):
        monkeypatch.setattr(params, "_load_user_config", lambda: {"accuracy_mode": "accurate"})
        assert get_params("streaming", {}).condition_on_prev_text is False
        assert get_params("short", {}).condition_on_prev_text is False

    def test_long_bucket_does_condition_on_prev_text(self, monkeypatch):
        monkeypatch.setattr(params, "_load_user_config", lambda: {"accuracy_mode": "accurate"})
        assert get_params("long", {}).condition_on_prev_text is True

    def test_manual_params_override(self, monkeypatch):
        monkeypatch.setattr(params, "_load_user_config", lambda: {
            "accuracy_mode": "fast",
            "manual_params": {"beam_size": 4, "temperature": 0.0},
        })
        p = get_params("medium", {})
        assert p.beam_size == 4
        assert p.temperature == (0.0,)


class TestTierLadder:
    def test_next_tier(self):
        assert next_tier("fast") == "light"
        assert next_tier("light") == "balanced"
        assert next_tier("balanced") == "accurate"
        assert next_tier("accurate") is None

    def test_get_params_for_tier_respects_streaming_invariant(self):
        assert get_params_for_tier("accurate", "streaming", {}).condition_on_prev_text is False
        assert get_params_for_tier("accurate", "long", {}).condition_on_prev_text is True
