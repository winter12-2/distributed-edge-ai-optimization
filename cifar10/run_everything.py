"""Run the CIFAR-10 experiments from the saved tuned baseline."""

import argparse
import os
import subprocess
import sys

if __package__:
    from .tuned_experiment import ROOT, TUNED_CHECKPOINT, TUNED_METRICS
else:
    from tuned_experiment import ROOT, TUNED_CHECKPOINT, TUNED_METRICS

EXPERIMENTS = [
    ["optimized_quantized.py"],
    ["optimized_pruned.py"],
    ["optimized_combined.py"],
    ["prune_tuned.py", "--epochs", "2", "--unstructured", "0.30", "0.60", "--run-tag", "tuned_suite"],
    ["federated_simulation.py"],
    ["federated_noniid.py", "--preset", "all", "--data-fraction", "0.1", "--rounds", "10"],
    ["federated_update3.py"],
    ["analyze_results.py"],
    ["plot_update2.py"],
    ["plot_update2.py", "--update3"],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--skip-baseline", action="store_true",
                          help="Reuse the previously trained tuned baseline (default).")
    baseline.add_argument("--train-baseline", action="store_true",
                          help="Retrain the tuned baseline before running the experiments.")
    args = parser.parse_args()
    if not args.train_baseline:
        for artifact in (TUNED_CHECKPOINT, TUNED_METRICS):
            if not artifact.is_file():
                parser.error(f"Missing tuned baseline artifact: {artifact}. Use --train-baseline to create it.")
    commands = EXPERIMENTS
    if args.train_baseline:
        commands = [["baseline_cifar10.py", "--tuned", "--seed", "42"], *commands]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    for script, *options in commands:
        print(f"\nRunning {' '.join([script, *options])}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / script), *options], cwd=ROOT,
                       env=environment, check=True)
    print("\nCIFAR-10 experiments completed.", flush=True)


if __name__ == "__main__":
    main()
