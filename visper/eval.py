"""
visper/eval.py
--------------
Local accuracy evaluation — WER / CER over a directory of audio files and their
reference transcripts. This is the source of the accuracy numbers in the README:
they are produced here, on real data, not estimated.

Both the hypothesis and the reference are run through the **shipped** normalizer
(``visper.postprocess.normalize_text``), so the reported WER is the WER of what
Visper actually outputs — not of some eval-only cleanup. For scoring only, a
final symmetric pass lowercases and strips punctuation from *both* sides
(``_for_scoring``): WER is conventionally punctuation-insensitive and these
corpora's references carry none, so charging the model for every correctly
placed comma would be a scoring artifact, not a real error.

Every run also drops a JSON sidecar (``<out>.json``) holding each file's
reference and hypothesis, so the table can be re-scored (``--rescore``) without
re-transcribing.

Dataset layout (either form works)::

    <dataset>/audios/<stem>.wav      <dataset>/refs/<stem>.txt   (or <stem>ND.txt)
    <dataset>/<stem>.wav             <dataset>/<stem>.txt

Usage::

    visper-eval datasets/coish datasets/short --limit 25 --out eval_results.md
"""
from __future__ import annotations

import argparse
import json
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


# Anything that is not a word char or whitespace — punctuation, the Hebrew
# geresh/gershayim in acronyms, quote marks. Unicode-aware: ``\w`` keeps Hebrew
# letters and digits.
_SCORE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _for_scoring(text: str) -> str:
    """Symmetric scoring normalization: lowercase, drop punctuation, collapse space.

    Applied to reference *and* hypothesis alike, only for the WER/CER math. The
    shipped ``normalize_text`` deliberately keeps punctuation (users want it);
    the references here have none, so without this every correctly emitted
    comma or period counts as an error.
    """
    return " ".join(_SCORE_PUNCT.sub(" ", text.lower()).split())


def _metrics(reference: str, hypothesis: str) -> tuple[float, float]:
    import jiwer
    reference = _for_scoring(reference)
    hypothesis = _for_scoring(hypothesis)
    if not reference.strip():
        return (0.0, 0.0)
    return (jiwer.wer(reference, hypothesis), jiwer.cer(reference, hypothesis))


def _eval_engine(language: str):
    """The engine used for accuracy runs.

    Pinned to CPU int8 through the shipped venv-worker runtime (``.venvs/cpu``)
    when that venv exists, else in-process CPU. Never CUDA: the speed device
    varies per machine and only matters for tier auto-selection, which we
    override explicitly here — the decoding math itself is device-independent.
    """
    from visper.api import _get_router
    from visper import venv_manager

    cfg: dict = {"device": "cpu", "compute_type": "int8",
                 "cpu_threads": 4, "num_workers": 1, "omp_threads": 4}
    if venv_manager.venv_exists("cpu"):
        cfg["venv_path"] = str(venv_manager.venv_path("cpu"))
    return _get_router(cfg).get(language)


def _summarize(rows: list[dict]) -> dict:
    """Aggregate + spread for one dataset's per-file rows.

    The spread matters: a smooth unimodal WER spread is genuine difficulty and
    the mean stands; a low cluster plus a high tail usually means some
    reference/audio pairs are misaligned and the mean is measuring dataset
    noise. The README needs to know which it is looking at.
    """
    wers = sorted(r["wer"] for r in rows)
    cers = [r["cer"] for r in rows]
    n = len(rows)

    def _pct(p: float) -> float:
        if not wers:
            return float("nan")
        return wers[min(n - 1, int(p * n))]

    return {
        "n": n,
        "wer": sum(wers) / n if n else float("nan"),
        "cer": sum(cers) / n if n else float("nan"),
        "wer_min": wers[0] if wers else float("nan"),
        "wer_p25": _pct(0.25),
        "wer_median": _pct(0.5),
        "wer_p75": _pct(0.75),
        "wer_max": wers[-1] if wers else float("nan"),
        "rows": rows,
    }


def evaluate(
    dataset_dirs: list[Path],
    language: str = "he",
    tier: str = "balanced",
    limit: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Transcribe every pair and return aggregate + per-dataset WER/CER.

    ``tier`` fixes the decoding tier (beam size, temperature schedule) so the
    reported WER is reproducible. Left to ``bucket="auto"`` it would swing with
    the host's RTF headroom — a GPU machine lands on ``accurate``, a CPU one on
    ``fast`` — and an unlabelled WER is not a reportable number.

    The returned report always carries every file's reference and hypothesis
    (``datasets[label]["rows"]``) so the run can be re-scored offline.
    """
    from visper.benchmark import _audio_duration, _bucket_for
    from visper.params import get_params_for_tier
    from visper.postprocess import normalize_text

    engine = _eval_engine(language)

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

        rows = []
        print(f"[eval] {label}: {len(pairs)} file(s)", file=sys.stderr)
        for audio, ref_path in pairs:
            reference = _clean_reference(ref_path.read_text(encoding="utf-8"), language)
            bucket = _bucket_for(_audio_duration(audio) or 30.0)
            tp = get_params_for_tier(tier, bucket, {})
            t0 = time.time()
            result = engine.transcribe(str(audio), bucket=bucket, language=language,
                                       _tier_override=tp)
            hypothesis = normalize_text(result.text, language)
            w, c = _metrics(reference, hypothesis)
            rows.append({"file": audio.name, "wer": w, "cer": c,
                         "sec": round(time.time() - t0, 1),
                         "ref": reference, "hyp": hypothesis})
            print(f"[eval]   {audio.name}: WER={w:.3f} CER={c:.3f}", file=sys.stderr)

        datasets[label] = _summarize(rows)

    return {
        "language": language,
        "tier": tier,
        "datasets": datasets,
        "model": getattr(engine, "_model_id", "?"),
    }


def rescore(sidecar: dict) -> dict:
    """Rebuild a report from a saved sidecar — recompute WER/CER, no transcription.

    Lets a scoring-normalization change be validated against a past run in
    seconds instead of a multi-hour re-transcribe.
    """
    datasets: dict[str, dict] = {}
    for label, rows in sidecar["datasets"].items():
        rescored = []
        for r in rows:
            w, c = _metrics(r["ref"], r["hyp"])
            rescored.append({**r, "wer": w, "cer": c})
        datasets[label] = _summarize(rescored)
    return {
        "language": sidecar.get("language", "?"),
        "tier": sidecar.get("tier", "?"),
        "datasets": datasets,
        "model": sidecar.get("model", "?"),
    }


def format_markdown(report: dict) -> str:
    lines = [
        f"### Accuracy — `{report['model']}` ({report['language']}, {report.get('tier', '?')} tier)",
        "",
        "| Dataset | Files | WER | CER | WER min / median / max |",
        "|---|--:|--:|--:|:--|",
    ]
    for label, d in report["datasets"].items():
        spread = (f"{d.get('wer_min', float('nan')):.2f} / "
                  f"{d.get('wer_median', float('nan')):.2f} / "
                  f"{d.get('wer_max', float('nan')):.2f}")
        lines.append(f"| {label} | {d['n']} | {d['wer']:.3f} | {d['cer']:.3f} | {spread} |")
    lines.append("")
    lines.append("_WER/CER: `visper.postprocess.normalize_text` on both sides, then a "
                 "symmetric lowercase + punctuation strip for the score (references carry "
                 "no punctuation). Mean over files; min/median/max show the spread._")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local WER/CER evaluation for Visper")
    parser.add_argument("datasets", nargs="*", type=Path,
                        help="Dataset directories (each with audios/ + refs/ or flat)")
    parser.add_argument("--language", default="he")
    parser.add_argument("--tier", default="balanced",
                        choices=["fast", "light", "balanced", "accurate"],
                        help="Decoding tier to pin (default: balanced)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sample at most N files per dataset (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--per-file", action="store_true", help="Show every file's WER/CER")
    parser.add_argument("--out", type=Path, help="Write the markdown table to this file")
    parser.add_argument("--rescore", type=Path, metavar="SIDECAR.json",
                        help="Recompute the table from a saved sidecar (no transcription)")
    args = parser.parse_args()

    if args.rescore:
        sidecar = json.loads(args.rescore.read_text(encoding="utf-8"))
        report = rescore(sidecar)
    elif args.datasets:
        report = evaluate(args.datasets, language=args.language, tier=args.tier,
                          limit=args.limit, seed=args.seed)
    else:
        parser.error("give at least one dataset directory, or --rescore a sidecar")

    if args.per_file:
        for label, d in report["datasets"].items():
            print(f"\n{label}:")
            for r in d["rows"]:
                print(f"  {r['file']:<45} WER={r['wer']:.3f} CER={r['cer']:.3f} "
                      f"({r.get('sec', '?')}s)")

    md = format_markdown(report)
    print("\n" + md)

    if args.out:
        args.out.write_text(md + "\n", encoding="utf-8")
        print(f"\n[eval] written to {args.out}", file=sys.stderr)
        sidecar_path = args.out.with_suffix(args.out.suffix + ".json")
        payload = {
            "language": report["language"], "tier": report["tier"],
            "model": report["model"],
            "datasets": {label: d["rows"] for label, d in report["datasets"].items()},
        }
        sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"[eval] hypotheses saved to {sidecar_path} (re-score with --rescore)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
