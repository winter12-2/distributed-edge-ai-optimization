from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
 
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "plots_analysis"
 
# categorical colours, always used in this fixed order
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
 
 
def hbar(ax, labels, values, title, fmt, highlight=None):
    y = np.arange(len(labels))
    colors = [SERIES[0] if (highlight is None or i == highlight) else "#9fc1ec" for i in range(len(labels))]
    ax.barh(y, values, color=colors, height=0.6, edgecolor="white", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, vmax * 1.18)
 
 
def plot_cifar_variants():
    path = RESULTS / "cifar10_model_comparison.csv"
    if not path.exists():
        print("skip: results/cifar10_model_comparison.csv not found")
        return
    df = pd.read_csv(path)
    labels = df["model"].tolist()
    metrics = [
        ("test_accuracy", "Test accuracy (%)", "{:.1f}", 100),
        ("size_mb", "Model size on disk (KB)", "{:.1f}", 1024),
        ("compressed_size_mb", "Compressed size, zlib (KB)", "{:.1f}", 1024),
        ("latency_ms_batch1", "Latency, batch 1 (ms)", "{:.3f}", 1),
        ("peak_ram_increase_mb", "Peak RAM increase, 256 images (MB)", "{:.2f}", 1),
        ("macs_per_image", "Compute per image (million MACs)", "{:.2f}", 1e-6),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    for ax, (col, title, fmt, scale) in zip(axes.flat, metrics):
        hbar(ax, labels, (df[col] * scale).tolist(), title, fmt)
    fig.suptitle("CIFAR-10: optimization variants compared (CPU)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    save(fig, "cifar10_variant_metrics.png")
 
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(df["size_mb"] * 1024, df["test_accuracy"] * 100, s=60, color=SERIES[0],
               edgecolor="white", linewidth=2, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r["size_mb"] * 1024, r["test_accuracy"] * 100), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Model size on disk (KB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. model size (up and left is better)", loc="left")
    save(fig, "cifar10_accuracy_vs_size.png")
 
 
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
    summary_path = RESULTS / "federated_update2_summary.csv"
    if not summary_path.exists():
        print("skip: results/federated_update2_summary.csv not found")
        return
    summary = pd.read_csv(summary_path)
    tags = summary["experiment"].tolist()
 
    # Accuracy per round, one line per experiment
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
    ax.set_title("FedAvg on CIFAR-10: IID vs non-IID, ideal vs heterogeneous clients", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "federated_accuracy_per_round.png")
 
    # one measure per panel (no dual axes)
    panels = [
        ("final_accuracy", "Final accuracy (%)", "{:.1f}", 100),
        ("mean_round_time_s", "Mean round time (simulated s)", "{:.1f}", 1),
        ("mean_completion_rate", "Client completion rate (%)", "{:.0f}", 100),
        ("total_comm_mb", "Total communication (MB)", "{:.0f}", 1),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for ax, (col, title, fmt, scale) in zip(axes, panels):
        hbar(ax, tags, (summary[col] * scale).tolist(), title, fmt)
    save(fig, "federated_summary_metrics.png")
 
    # label distribution heatmaps
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
    plot_cifar_variants()
    plot_training_history()
    plot_federated()