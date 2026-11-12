"""Run the original experiment sequence on CIFAR-10."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "baseline_cifar10.py",
    "optimized_quantized.py",
    "optimized_pruned.py",
    "optimized_combined.py",
    "federated_simulation.py",
    "analyze_results.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Reuse the previously trained CIFAR-10 baseline.")
    args = parser.parse_args()
    if args.skip_baseline:
        for name in ["models/baseline_model.pt", "results/results_baseline.json"]:
            if not (ROOT / name).is_file():
                parser.error(f"Missing baseline artifact: {ROOT / name}")
    for script in SCRIPTS[1:] if args.skip_baseline else SCRIPTS:
        print(f"\nRunning {script}", flush=True)
        environment = dict(os.environ, PYTHONIOENCODING="utf-8")
        subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT,
                       env=environment, check=True)
    print("\nCIFAR-10 experiments completed.", flush=True)


if __name__ == "__main__":
    main()
