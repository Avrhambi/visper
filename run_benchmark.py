#!/usr/bin/env python
"""Entry point: python run_benchmark.py [--force] [--quick] [--full]"""
import argparse
import json
from core.benchmark import run_benchmark, RESULTS_PATH


def main():
    parser = argparse.ArgumentParser(description="Benchmark hardware for STT transcription")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if benchmark_results.json already exists")
    parser.add_argument("--quick", action="store_true",
                        help="Heuristic only — derive config from hardware detection, "
                             "no inference timing. Instant but no RTF measurements.")
    parser.add_argument("--full", action="store_true",
                        help="Exhaustive mode — test all compute types × thread counts "
                             "[2,4,6,8]. Slow but covers every combination.")
    args = parser.parse_args()

    if args.quick and args.full:
        parser.error("--quick and --full are mutually exclusive.")

    run_benchmark(force=args.force, quick=args.quick, full=args.full)

    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
        mode = results.get("mode", "smart")
        print(f"\nBest configs by bucket (mode: {mode}):")
        for bucket, cfg in results["best"].items():
            if cfg:
                rtf = cfg.get("rtf")
                rtf_str = f"RTF {rtf:.3f}" if rtf else "(estimated)"
                if cfg["device"] == "openvino":
                    hw_str = f"openvino/{cfg.get('openvino_device', '')}"
                else:
                    hw_str = f"{cfg['device']} {cfg['compute_type']} {cfg['cpu_threads']}t"
                print(f"  {bucket:<12} → {hw_str:<28} {rtf_str}"
                      f"  [{cfg.get('auto_accuracy_tier', '?')} tier]")
            else:
                print(f"  {bucket:<12} → no audio file in records/ for this bucket")


if __name__ == "__main__":
    main()
