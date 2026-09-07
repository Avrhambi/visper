# Benchmark evidence

The numbers in the main `README.md` §5 (Accuracy) and §6 (Performance) come from
these committed run artifacts. Both are machine-specific — reproduce on your own
hardware with the commands below.

Reference machine: 11th Gen Intel Core i5-1135G7 @ 2.40 GHz · NVIDIA GeForce
MX350 (2 GB VRAM) · Intel Iris Xe iGPU · 8 logical cores · 16 GB RAM · no MLX.

## `eval-he-balanced.json` / `.md`

`visper-eval` output for `ivrit-ai/whisper-large-v3-turbo-ct2` at the `balanced`
tier over three Hebrew corpora (Knesset / longer-form / CoSIH). The `.md` is the
table; the `.json` sidecar holds every reference/hypothesis pair.

Re-score the table in seconds, no audio needed:

```bash
visper-eval --rescore docs/benchmarks/eval-he-balanced.json
```

Full re-run (needs the corpora): `visper-eval <corpus-dir> --tier balanced --out eval.md`.

## `benchmark-i5-1135g7-mx350.json`

A `benchmark_results.json` snapshot from the reference machine, with the
machine-local `venv_path` stripped. `visper-benchmark --report` formats §6's
table from a file of this shape (the accuracy-tier column is recomputed from the
measured RTF, so a snapshot taken before a tier-policy change still reports the
current tier).

Re-run on your hardware: `visper-benchmark` (writes `benchmark_results.json`),
then `visper-benchmark --report`.
