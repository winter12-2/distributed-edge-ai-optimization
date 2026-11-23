import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd  

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "plots_analysis"

UPDATE3 = "--update3" in sys.argv
SUMMARY_FILE = "federated_update3_summary.csv" if UPDATE3 else "federated_update2_summary.csv"
PREFIX = "update3_" if UPDATE3 else ""
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a63", "#e1e0d9"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(f"saved cifar10/plots_analysis/{name}")


def hbar(ax, labels, values, title, fmt):
    y = np.arange(len(labels))
    ax.barh(y, values, color=SERIES[0], height=0.6, edgecolor="white", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, vmax * 1.18)


def plot_training_history():
    path = RESULTS / "results_baseline.json"
    if not path.exists():
        return
    import json
    losses = json.loads(path.read_text()).get("epoch_losses", [])
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(losses) + 1), losses, color=SERIES[0], lw=2, marker="o", ms=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training loss")
    ax.set_title("CIFAR-10 baseline CNN training loss", loc="left")
    save(fig, "cifar10_training_loss.png")


def plot_federated():
    summary_path = RESULTS / SUMMARY_FILE
    if not summary_path.exists():
        print(f"skip: results/{SUMMARY_FILE} not found")
        return
    summary = pd.read_csv(summary_path)
    tags = summary["experiment"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, tag in enumerate(tags):
        f = RESULTS / f"federated_{tag}_rounds.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f)
        color = SERIES[i % len(SERIES)]
        ax.plot(r["round"], r["test_accuracy"] * 100, color=color, lw=2, marker="o", ms=4, label=tag)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Global test accuracy (%)")
    title = ("CIFAR-10 federated (improved CNN): FedAvg vs FedProx vs client selection" if UPDATE3
             else "FedAvg on CIFAR-10: IID vs non-IID, ideal vs heterogeneous clients")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, PREFIX + "federated_accuracy_per_round.png")
    panels = [
        ("final_accuracy", "Final accuracy (%)", "{:.1f}", 100),
        ("mean_round_time_s", "Mean round time (simulated s)", "{:.1f}", 1),
        ("mean_completion_rate", "Client completion rate (%)", "{:.0f}", 100),
        ("total_comm_mb", "Total communication (MB)", "{:.1f}", 1),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, (col, title, fmt, scale) in zip(axes, panels):
        hbar(ax, tags, (summary[col] * scale).tolist(), title, fmt)
    save(fig, PREFIX + "federated_summary_metrics.png")

    # Label distribution heatmaps (Update 2 only)
    if UPDATE3:
        return
    for tag in tags:
        f = RESULTS / f"federated_{tag}_label_counts.csv"
        if not f.exists() or "hetero" in tag:
            continue
        counts = pd.read_csv(f).drop(columns=["client"]).values
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(counts, cmap="Blues", aspect="auto")
        ax.set_xlabel("Class")
        ax.set_ylabel("Client")
        ax.set_xticks(range(10))
        ax.set_yticks(range(counts.shape[0]))
        ax.grid(False)
        ax.set_title(f"Images per class on each client: {tag}", loc="left")
        fig.colorbar(im, ax=ax, label="Images")
        save(fig, f"federated_label_distribution_{tag}.png")


if __name__ == "__main__":
    if not UPDATE3:
        plot_training_history()
    plot_federated()