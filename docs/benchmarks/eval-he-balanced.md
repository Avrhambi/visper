### Accuracy — `ivrit-ai/whisper-large-v3-turbo-ct2` (he, balanced tier)

| Dataset | Files | WER | CER | WER min / median / max |
|---|--:|--:|--:|:--|
| coish | 15 | 0.575 | 0.439 | 0.26 / 0.61 / 0.78 |
| short | 25 | 0.141 | 0.082 | 0.04 / 0.12 / 0.30 |
| long | 25 | 0.245 | 0.124 | 0.00 / 0.22 / 0.76 |

_WER/CER: `visper.postprocess.normalize_text` on both sides, then a symmetric lowercase + punctuation strip for the score (references carry no punctuation). Mean over files; min/median/max show the spread._
