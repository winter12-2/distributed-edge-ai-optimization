"""Generate the copied MNIST analysis figures from CIFAR-10 result files."""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def display(value):
    # Notebook tables may be pandas Styler objects rather than DataFrames.
    if hasattr(value, "data") and hasattr(value.data, "to_string"):
        value = value.data
    print(value.to_string(index=False) if hasattr(value, "to_string") else value)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)
    path = ROOT / "notebooks" / "results_analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"display": display, "__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, f"{path}:cell_{index}", "exec"), namespace)
            plt.close("all")
    print(f"Saved analysis figures in {ROOT / 'results'}")


if __name__ == "__main__":
    main()
