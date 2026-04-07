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
_NIKUD = re.compile(r'[\u05B0-\u05C7]')

# Straight double-quote between/after Hebrew letters → Gershayim (״)
# Matches: Hebrew letter, then ", then Hebrew letter or whitespace or end of string
_GERSHAYIM = re.compile(r'(?<=[א-ת])"(?=[א-ת\s]|$)')

# Straight single-quote after a Hebrew letter → Geresh (׳)
_GERESH = re.compile(r"(?<=[א-ת])'")


def normalize_hebrew(text: str) -> str:
    """
    Normalize raw Whisper Hebrew output:
    - Strip nikud / cantillation diacritics (Whisper rarely produces them correctly)
    - Convert ASCII straight quotes adjacent to Hebrew letters to proper typographic marks:
        "  →  ״  (Gershayim, U+05F4)  — used for abbreviations like ארה"ב
        '  →  ׳  (Geresh,    U+05F3)  — used for units, proper names
    """
    text = _NIKUD.sub('', text)
    text = _GERSHAYIM.sub('\u05F4', text)
    text = _GERESH.sub('\u05F3', text)
    return text.strip()
