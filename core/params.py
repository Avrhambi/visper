"""
core/params.py
--------------
Single owner of all Whisper inference parameter decisions.
No other module sets beam_size, temperature, condition_on_prev_text, etc.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

RTF_BUDGET = 0.85  # never push estimated RTF above this

TIERS = {
    "fast": {
        "beam_size": 1,
        "best_of": 1,
        "temperature": 0.0,
        "patience": 1.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "per_bucket": {
            "streaming": {"condition_on_prev_text": False, "without_timestamps": True},
            "short":     {"condition_on_prev_text": False, "without_timestamps": True},
            "medium":    {"condition_on_prev_text": False, "without_timestamps": True},
            "long":      {"condition_on_prev_text": True,  "without_timestamps": True},
            "extended":  {"condition_on_prev_text": True,  "without_timestamps": True},
        },
    },
    "light": {
        "beam_size": 2,
        "best_of": 1,
        "temperature": 0.0,
        "patience": 1.0,
        "compression_ratio_threshold": 2.3,
        "log_prob_threshold": -0.9,
        "no_speech_threshold": 0.55,
        "per_bucket": {
            "streaming": {"condition_on_prev_text": False, "without_timestamps": True},
            "short":     {"condition_on_prev_text": False, "without_timestamps": True},
            "medium":    {"condition_on_prev_text": False, "without_timestamps": False},
            "long":      {"condition_on_prev_text": True,  "without_timestamps": False},
            "extended":  {"condition_on_prev_text": True,  "without_timestamps": False},
        },
    },
    "balanced": {
        "beam_size": 3,
        "best_of": 1,
        "temperature": 0.0,
        "patience": 1.0,
        "compression_ratio_threshold": 2.2,
        "log_prob_threshold": -0.8,
        "no_speech_threshold": 0.5,
        "per_bucket": {
            "streaming": {"condition_on_prev_text": False, "without_timestamps": True},
            "short":     {"condition_on_prev_text": False, "without_timestamps": False},
            "medium":    {"condition_on_prev_text": False, "without_timestamps": False},
            "long":      {"condition_on_prev_text": True,  "without_timestamps": False},
            "extended":  {"condition_on_prev_text": True,  "without_timestamps": False},
        },
    },
    "accurate": {
        "beam_size": 5,
        "best_of": 3,
        "temperature": 0.2,
        "patience": 1.5,
        "compression_ratio_threshold": 1.8,
        "log_prob_threshold": -0.5,
        "no_speech_threshold": 0.4,
        "per_bucket": {
            "streaming": {"condition_on_prev_text": False, "without_timestamps": True},
            "short":     {"condition_on_prev_text": False, "without_timestamps": False},
            "medium":    {"condition_on_prev_text": True,  "without_timestamps": False},
            "long":      {"condition_on_prev_text": True,  "without_timestamps": False},
            "extended":  {"condition_on_prev_text": True,  "without_timestamps": False},
        },
    },
}

TIER_RTF_MULTIPLIERS = {
    "fast":     1.0,
    "light":    1.35,
    "balanced": 1.8,
    "accurate": 4.5,
}

_logged_first_call = False


@dataclass
class WhisperParams:
    beam_size: int
    best_of: int
    temperature: float
    patience: float
    condition_on_prev_text: bool
    without_timestamps: bool
    compression_ratio_threshold: float
    log_prob_threshold: float
    no_speech_threshold: float
    tier_used: str
    auto_selected: bool
    confidence_retry_enabled: bool = False

    def as_transcribe_kwargs(self) -> dict:
        return {
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "patience": self.patience,
            "condition_on_previous_text": self.condition_on_prev_text,
            "without_timestamps": self.without_timestamps,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
        }


def _load_user_config() -> dict:
    try:
        import yaml
        path = ROOT / "config.yaml"
        if path.exists():
            return yaml.safe_load(path.read_text()) or {}
    except Exception:
        pass
    return {}


def _rtf_headroom(hw_config: dict, bucket: str) -> Optional[float]:
    results_path = ROOT / "benchmark_results.json"
    if not results_path.exists():
        return None
    try:
        data = json.loads(results_path.read_text())
        best = data.get("best", {}).get(bucket)
        if best and best.get("rtf") is not None:
            return best["rtf"]
    except Exception:
        pass
    return None


def estimate_rtf_cost(base_rtf: float, tier: str) -> float:
    return base_rtf * TIER_RTF_MULTIPLIERS.get(tier, 1.0)


def get_params(bucket: str, hw_config: dict) -> WhisperParams:
    """
    Main entry point. Called by Transcriber before every transcription.
    Never raises — falls back to fast tier on any error.
    """
    global _logged_first_call

    try:
        user = _load_user_config()
        accuracy_mode = user.get("accuracy_mode", "auto")

        # Per-bucket override
        overrides = user.get("bucket_accuracy_overrides") or {}
        if bucket in overrides and overrides[bucket]:
            accuracy_mode = overrides[bucket]

        # Resolve tier
        auto_selected = False
        if accuracy_mode == "auto":
            auto_selected = True
            base_rtf = _rtf_headroom(hw_config, bucket)
            if base_rtf is None:
                tier = "fast"
            elif base_rtf > RTF_BUDGET:
                tier = "fast"
                if not _logged_first_call:
                    print(f"[STT] Hardware is near real-time limit (RTF {base_rtf:.2f}). Using fast tier.",
                          file=sys.stderr)
            else:
                # Pick highest tier that fits budget
                tier = "fast"
                for t in ("accurate", "balanced", "light"):
                    if estimate_rtf_cost(base_rtf, t) < RTF_BUDGET:
                        tier = t
                        break
        else:
            tier = accuracy_mode if accuracy_mode in TIERS else "fast"

        tier_def = TIERS[tier]
        per_bucket = tier_def["per_bucket"].get(bucket, tier_def["per_bucket"].get("medium", {}))

        # confidence_retry: use config value; default True only when accurate tier
        cfg_retry = user.get("confidence_retry_enabled")
        if cfg_retry is None:
            confidence_retry = (tier == "accurate")
        else:
            confidence_retry = bool(cfg_retry)

        params = WhisperParams(
            beam_size=tier_def["beam_size"],
            best_of=tier_def["best_of"],
            temperature=tier_def["temperature"],
            patience=tier_def["patience"],
            condition_on_prev_text=per_bucket.get("condition_on_prev_text", False),
            without_timestamps=per_bucket.get("without_timestamps", True),
            compression_ratio_threshold=tier_def["compression_ratio_threshold"],
            log_prob_threshold=tier_def["log_prob_threshold"],
            no_speech_threshold=tier_def["no_speech_threshold"],
            tier_used=tier,
            auto_selected=auto_selected,
            confidence_retry_enabled=confidence_retry,
        )

        # Hard rule: streaming/short must never have condition_on_prev_text=True
        if bucket in ("streaming", "short") and params.condition_on_prev_text:
            print(f"[STT] Warning — condition_on_prev_text forced to False for '{bucket}' bucket "
                  "(hallucination prevention).", file=sys.stderr)
            params.condition_on_prev_text = False

        # Apply manual_params overrides
        manual = user.get("manual_params") or {}
        if manual.get("beam_size", 0) > 0:
            params.beam_size = manual["beam_size"]
        if manual.get("temperature", -1) >= 0:
            params.temperature = manual["temperature"]
        if manual.get("best_of", 0) > 0:
            params.best_of = manual["best_of"]
        if manual.get("patience", -1) >= 0:
            params.patience = manual["patience"]
        if manual.get("condition_on_prev_text") is not None and bucket not in ("streaming", "short"):
            params.condition_on_prev_text = manual["condition_on_prev_text"]
        if manual.get("without_timestamps") is not None:
            params.without_timestamps = manual["without_timestamps"]
        if manual.get("compression_ratio_threshold", -1) >= 0:
            params.compression_ratio_threshold = manual["compression_ratio_threshold"]
        if manual.get("log_prob_threshold", -1) >= -0.999:
            params.log_prob_threshold = manual["log_prob_threshold"]
        if manual.get("no_speech_threshold", -1) >= 0:
            params.no_speech_threshold = manual["no_speech_threshold"]

        # Log on first call
        if not _logged_first_call:
            base_rtf = _rtf_headroom(hw_config, bucket)
            headroom_str = f", RTF {base_rtf:.2f}" if base_rtf else ""
            print(f"[STT] Accuracy: {accuracy_mode} → {tier} tier "
                  f"(beam={params.beam_size}, temp={params.temperature})"
                  f" [{bucket} bucket{headroom_str}]", file=sys.stderr)
            _logged_first_call = True

        return params

    except Exception as e:
        print(f"[STT] params.get_params error ({e}), falling back to fast tier.", file=sys.stderr)
        return WhisperParams(
            beam_size=1, best_of=1, temperature=0.0, patience=1.0,
            condition_on_prev_text=False, without_timestamps=True,
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
            no_speech_threshold=0.6, tier_used="fast", auto_selected=False,
            confidence_retry_enabled=False,
        )


def describe_params(params: WhisperParams) -> str:
    return (f"{params.tier_used} tier — beam={params.beam_size} "
            f"best_of={params.best_of} temp={params.temperature} "
            f"prev_text={params.condition_on_prev_text}")


# ---------------------------------------------------------------------------
# Confidence-gated retry helpers
# ---------------------------------------------------------------------------

_TIER_UPGRADE = {"fast": "light", "light": "balanced", "balanced": "accurate", "accurate": None}


def next_tier(tier: str) -> Optional[str]:
    """Return the next higher accuracy tier, or None if already at maximum."""
    return _TIER_UPGRADE.get(tier)


def get_params_for_tier(tier: str, bucket: str, hw_config: dict) -> WhisperParams:
    """
    Return WhisperParams for a specific tier without going through auto-selection.
    Used by the confidence-gated retry path in Transcriber.
    Respects the condition_on_prev_text=False invariant for streaming/short buckets.
    """
    tier = tier if tier in TIERS else "fast"
    tier_def = TIERS[tier]
    per_bucket = tier_def["per_bucket"].get(bucket, tier_def["per_bucket"].get("medium", {}))

    params = WhisperParams(
        beam_size=tier_def["beam_size"],
        best_of=tier_def["best_of"],
        temperature=tier_def["temperature"],
        patience=tier_def["patience"],
        condition_on_prev_text=per_bucket.get("condition_on_prev_text", False),
        without_timestamps=per_bucket.get("without_timestamps", True),
        compression_ratio_threshold=tier_def["compression_ratio_threshold"],
        log_prob_threshold=tier_def["log_prob_threshold"],
        no_speech_threshold=tier_def["no_speech_threshold"],
        tier_used=tier,
        auto_selected=False,
        confidence_retry_enabled=False,  # retry path never re-retries
    )

    # Hard rule: streaming/short must never have condition_on_prev_text=True
    if bucket in ("streaming", "short"):
        params.condition_on_prev_text = False

    return params
