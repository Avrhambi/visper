"""
visper/eval.py
--------------
Local accuracy evaluation — WER / CER over a directory of audio files and their
reference transcripts. This is the source of the accuracy numbers in the README:
they are produced here, on real data, not estimated.

Both the hypothesis and the reference are run through the **shipped** normalizer
(``visper.postprocess.normalize_text``), so the reported WER is the WER of what
Visper actually outputs — not of some eval-only cleanup.

Dataset layout (either form works)::

    <dataset>/audios/<stem>.wav      <dataset>/refs/<stem>.txt   (or <stem>ND.txt)
    <dataset>/<stem>.wav             <dataset>/<stem>.txt

Usage::

    visper-eval datasets/coish datasets/short --limit 25 --out eval_results.md
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Annotation markers that appear in reference transcripts but are never speech.
_ANNOTATION = re.compile(r"<[^>]*>|>[^<]*<|\[[^\]]*\]|\(\([^)]*\)\)|\{[^}]*\}", re.DOTALL)
_AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus")


def _clean_reference(text: str, language: str) -> str:
    from visper.postprocess import normalize_text
    text = " ".join(text.splitlines())
    text = _ANNOTATION.sub(" ", text)
    return normalize_text(text, language)


def find_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each audio file with its reference transcript by filename stem."""
    audio_dir = dataset_dir / "audios" if (dataset_dir / "audios").is_dir() else dataset_dir
    ref_dir = dataset_dir / "refs" if (dataset_dir / "refs").is_dir() else dataset_dir

    pairs: list[tuple[Path, Path]] = []
    for audio in sorted(audio_dir.iterdir()):
        if audio.suffix.lower() not in _AUDIO_EXT:
            continue
        for cand in (ref_dir / f"{audio.stem}.txt",
                     ref_dir / f"{audio.stem}ND.txt",
                     ref_dir / f"{audio.stem}.ND.txt"):
            if cand.exists():
                pairs.append((audio, cand))
                break
    return pairs


def _metrics(reference: str, hypothesis: str) -> tuple[float, float]:
    import jiwer
    if not reference.strip():
        return (0.0, 0.0)
    return (jiwer.wer(reference, hypothesis), jiwer.cer(reference, hypothesis))


def evaluate(
    dataset_dirs: list[Path],
    language: str = "he",
    limit: Optional[int] = None,
    seed: int = 0,
    per_file: bool = False,
) -> dict:
    """Transcribe every pair and return aggregate + per-dataset WER/CER."""
    from visper.api import _get_config, _get_router
    from visper.postprocess import normalize_text

    engine = _get_router(_get_config("medium")).get(language)

    datasets: dict[str, dict] = {}
    rng = random.Random(seed)

    for dataset_dir in dataset_dirs:
        label = dataset_dir.name
        pairs = find_pairs(dataset_dir)
        if not pairs:
            print(f"[eval] {label}: no audio/ref pairs found — skipped", file=sys.stderr)
            continue
        if limit and len(pairs) > limit:
            pairs = rng.sample(pairs, limit)
            pairs.sort()

        wers, cers, rows = [], [], []
        print(f"[eval] {label}: {len(pairs)} file(s)", file=sys.stderr)
        for audio, ref_path in pairs:
            reference = _clean_reference(ref_path.read_text(encoding="utf-8"), language)
            t0 = time.time()
            result = engine.transcribe(str(audio), bucket="auto", language=language)
            hypothesis = normalize_text(result.text, language)
            w, c = _metrics(reference, hypothesis)
            wers.append(w)
            cers.append(c)
            rows.append({"file": audio.name, "wer": w, "cer": c,
                         "sec": round(time.time() - t0, 1)})
            print(f"[eval]   {audio.name}: WER={w:.3f} CER={c:.3f}", file=sys.stderr)

        datasets[label] = {
            "n": len(wers),
            "wer": sum(wers) / len(wers) if wers else float("nan"),
            "cer": sum(cers) / len(cers) if cers else float("nan"),
            "rows": rows if per_file else [],
        }

    return {
        "language": language,
        "datasets": datasets,
        "model": getattr(engine, "_model_id", "?"),
    }


def format_markdown(report: dict) -> str:
    lines = [
        f"### Accuracy — `{report['model']}` ({report['language']})",
        "",
        "| Dataset | Files | WER | CER |",
        "|---|--:|--:|--:|",
    ]
    for label, d in report["datasets"].items():
        lines.append(f"| {label} | {d['n']} | {d['wer']:.3f} | {d['cer']:.3f} |")
    lines.append("")
    lines.append("_WER/CER computed with `visper.postprocess.normalize_text` applied to "
                 "both hypothesis and reference._")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local WER/CER evaluation for Visper")
    parser.add_argument("datasets", nargs="+", type=Path,
                        help="Dataset directories (each with audios/ + refs/ or flat)")
    parser.add_argument("--language", default="he")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sample at most N files per dataset (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--per-file", action="store_true", help="Show every file's WER/CER")
    parser.add_argument("--out", type=Path, help="Write the markdown table to this file")
    args = parser.parse_args()

    report = evaluate(args.datasets, language=args.language, limit=args.limit,
                      seed=args.seed, per_file=args.per_file)

    if args.per_file:
        for label, d in report["datasets"].items():
            print(f"\n{label}:")
            for r in d["rows"]:
                print(f"  {r['file']:<45} WER={r['wer']:.3f} CER={r['cer']:.3f} ({r['sec']}s)")

    md = format_markdown(report)
    print("\n" + md)
    if args.out:
        args.out.write_text(md + "\n", encoding="utf-8")
        print(f"\n[eval] written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
