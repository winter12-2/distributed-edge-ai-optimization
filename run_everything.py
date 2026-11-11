import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent / "mnist"

SCRIPTS = [
    "baseline_mnist.py",
    "optimized_quantized.py",
    "optimized_pruned.py",
    "optimized_combined.py",
    "federated_simulation.py",
]


def main():
    python = sys.executable

    for script in SCRIPTS:
        print("\n" + "=" * 80)
        print(f"Running {script}")
        print("=" * 80)
        subprocess.run([python, script], cwd=PROJECT_ROOT, check=True)

    print("\nAll scripts completed successfully.")


if __name__ == "__main__":
    main()
