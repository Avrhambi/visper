"""
core/postprocess.py
-------------------
Rules-based Hebrew text normalization applied after Whisper transcription.
Zero runtime cost — pure regex, no models.

normalize_hebrew() is the only public function.
"""
from __future__ import annotations

import re

# Hebrew nikud (vowel diacritics) and cantillation marks — U+05B0..U+05C7
_NIKUD = re.compile(r'[ְ-ׇ]')

# Straight double-quote between/after Hebrew letters → Gershayim (״)
_GERSHAYIM = re.compile(r'(?<=[א-ת])"(?=[א-ת\s]|$)')

# Straight single-quote after a Hebrew letter → Geresh (׳)
_GERESH = re.compile(r"(?<=[א-ת])'")

# Hebrew letter immediately followed by Latin (no space) — insert space
_HE_THEN_LATIN = re.compile(r'([֐-׿])([A-Za-z])')
# Latin letter immediately followed by Hebrew (no space) — insert space
_LATIN_THEN_HE = re.compile(r'([A-Za-z])([֐-׿])')

# Trailing comma or semicolon at end of a segment (Whisper hallucination)
_TRAILING_COMMA = re.compile(r'[,;]\s*$')

# Three or more consecutive identical words (hallucination collapse)
_REPEAT_WORD = re.compile(r'\b(\S+)(?:\s+\1){2,}\b')

# Letter (Latin or Hebrew) immediately adjacent to a digit — insert space
_LETTER_THEN_DIGIT = re.compile(r'([A-Za-zא-ת])(\d)')
_DIGIT_THEN_LETTER = re.compile(r'(\d)([A-Za-zא-ת])')

# Hebrew script block (U+0590–U+05FF) and Unicode bidi control chars injected by Hebrew-tuned model
_HEBREW_CHARS = re.compile(r'[֐-׿]+')
_BIDI_CONTROLS = re.compile(r'[‎‏‪-‮⁦-⁩]')


def normalize_hebrew(text: str) -> str:
    """
    Normalize raw Whisper Hebrew output:
    - Strip nikud / cantillation diacritics (Whisper rarely produces them correctly)
    - Convert ASCII straight quotes adjacent to Hebrew letters to typographic marks:
        "  →  ״  (Gershayim, U+05F4)  — used for abbreviations like ארה"ב
        '  →  ׳  (Geresh,    U+05F3)  — used for units, proper names
    - Insert a space at Hebrew/Latin script boundaries (Whisper drops spaces at code-switch)
    - Strip trailing comma or semicolon (Whisper often ends segments with ,)
    - Collapse 3+ consecutive identical words to 1 (hallucination pattern)
    """
    text = _NIKUD.sub('', text)
    text = _GERSHAYIM.sub('״', text)
    text = _GERESH.sub('׳', text)
    text = _HE_THEN_LATIN.sub(r'\1 \2', text)
    text = _LATIN_THEN_HE.sub(r'\1 \2', text)
    text = _LETTER_THEN_DIGIT.sub(r'\1 \2', text)
    text = _DIGIT_THEN_LETTER.sub(r'\1 \2', text)
    text = _TRAILING_COMMA.sub('', text)
    text = _REPEAT_WORD.sub(r'\1', text)
    return text.strip()


def normalize_text(text: str, language: str = "he") -> str:
    """Language-aware wrapper. Hebrew: full pipeline. Other languages: neutral rules only."""
    if language == "he":
        return normalize_hebrew(text)
    # Strip Hebrew characters and bidi control marks injected by the Hebrew-tuned model
    text = _HEBREW_CHARS.sub('', text)
    text = _BIDI_CONTROLS.sub('', text)
    text = _LETTER_THEN_DIGIT.sub(r'\1 \2', text)
    text = _DIGIT_THEN_LETTER.sub(r'\1 \2', text)
    text = _TRAILING_COMMA.sub('', text)
    text = _REPEAT_WORD.sub(r'\1', text)
    return text.strip()
