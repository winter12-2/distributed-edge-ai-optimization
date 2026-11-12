# Distributed Edge AI Optimization

CNN compression and federated communication experiments, separated by dataset.

## Layout

```text
mnist/
    baseline_mnist.py
    optimized_quantized.py
    optimized_pruned.py
    optimized_combined.py
    federated_simulation.py
    notebooks/
    plots_analysis/
    data/
    models/
    results/
cifar10/
    baseline_cifar10.py
    optimized_quantized.py
    optimized_pruned.py
    optimized_combined.py
    federated_simulation.py
    federated_noniid.py
    run_everything.py
    analyze_results.py
    plot_update2.py
    notebooks/
    plots_analysis/
    data/
    models/
    results/
run_everything.py
requirements.txt
```

Existing MNIST artifacts are preserved in `mnist/`. The root runner runs the
five MNIST scripts in their original order. Generated data, models, and results
directories are ignored by Git unless files were already tracked.

## Run

From the repository root using the existing Windows environment:

```powershell
# CIFAR-10 baseline
.\venv\Scripts\python.exe cifar10\baseline_cifar10.py

# CIFAR-10 quantization, four pruning variants, combined compression,
# three federated modes, and result graphs using the existing baseline
.\venv\Scripts\python.exe cifar10\run_everything.py --skip-baseline

# Full CIFAR-10 pipeline, including baseline retraining
.\venv\Scripts\python.exe cifar10\run_everything.py

# Regenerate CIFAR-10 analysis graphs from saved results
.\venv\Scripts\python.exe cifar10\analyze_results.py

# CIFAR-10 non-IID and heterogeneous-client federated experiments (Update 2)
.\venv\Scripts\python.exe cifar10\federated_noniid.py --preset all --data-fraction 0.1 --rounds 10

# Quick check that the federated experiments run (a few minutes)
.\venv\Scripts\python.exe cifar10\federated_noniid.py --preset all --quick

# Graphs for the non-IID and heterogeneous experiments
.\venv\Scripts\python.exe cifar10\plot_update2.py

# Full MNIST pipeline, including baseline retraining
.\venv\Scripts\python.exe run_everything.py

# A single MNIST experiment
.\venv\Scripts\python.exe mnist\optimized_quantized.py
```

Open MNIST notebooks with `mnist/notebooks/` as the notebook working directory.

`federated_noniid.py` additionally requires `psutil` and `matplotlib`
(`pip install psutil matplotlib`).

## CIFAR-10 Baseline

The CIFAR-10 experiment preserves the MNIST baseline procedure: convolution
layers with 8 and 16 output channels, ReLU, two max pooling layers, a linear
classifier, cross-entropy, Adam at 0.001, batch size 64, and five epochs.
No augmentation, architecture search, or accuracy tuning is applied.

Only input-dependent dimensions change: three RGB channels instead of one,
and `16 * 8 * 8` classifier inputs instead of `16 * 7 * 7` because CIFAR-10
images are 32x32. The original normalization mean 0.1307 and standard deviation
0.3081 are repeated across RGB channels.

Outputs are `cifar10/models/baseline_model.pt` and
`cifar10/results/results_baseline.json`, including epoch losses and training
settings. Training is stochastic, as in the original MNIST code.

## CIFAR-10 Algorithm Transfer

The four experiment scripts and three notebooks are copied from `mnist/`.
Changes are limited to dataset loading, RGB normalization matching the CIFAR-10
baseline, three input channels, imports, labels, and paths. Compression and
federated settings are preserved: dynamic int8 quantization of Linear layers;
30% and 60% structured/unstructured pruning with two fine-tuning epochs at
0.0005; combined 30% unstructured pruning plus int8; five federated clients,
ten rounds, one local epoch, batch size 64, and Adam at 0.001.

The runner reuses the saved baseline with `--skip-baseline`. Federated modes
start from new random models, as in MNIST, and train separately. Accuracy
differences are single-run observations, not controlled multi-seed comparisons.
Pruned models receive two extra fine-tuning epochs, so accuracy gains over the
baseline do not isolate pruning from the effect of additional training.
The inherited communication costs are estimates rather than measured network
traffic; combined costs assume sparse encoding and omit encoding overhead.

`cifar10/notebooks/results_analysis.ipynb` only analyzes saved JSON/CSV results.
The other two notebooks retain the older separate dense-model experiments;
they are not executed by the runner. The TensorFlow notebook requires TensorFlow
and uses 3072 flattened RGB inputs; TensorFlow is not in the current environment.
Copied notebooks have their old MNIST outputs cleared to avoid mixing results.
All CIFAR-10 checkpoints, CSV/JSON results, and figures stay under `cifar10/`.

## CIFAR-10 Non-IID Data and Heterogeneous Clients

`cifar10/federated_noniid.py` extends the federated simulation toward realistic
edge conditions. It imports the `CNN` class, data folder, and normalization from
`baseline_cifar10.py`, so the model is identical to the baseline. Training uses
FedAvg with SGD (learning rate 0.03, momentum 0.9), one local epoch, and batch
size 32, starting from a new random model.

**Data distribution.** Client data is split either IID (random) or non-IID with
a Dirichlet label split: alpha 0.5 is moderately skewed and alpha 0.1 is highly
skewed, so most clients see only a few classes.

**Heterogeneous clients.** With `--hetero`, each client belongs to one of four
device tiers with an explicit RAM budget, CPU thread limit, speed factor,
network bandwidth, and probability of going offline in a round:

| Tier | RAM budget | CPU threads | Speed factor | Bandwidth | Offline chance |
|---|---|---|---|---|---|
| high | 64 MB | 4 | 1.00 | 50 Mbps | 2% |
| mid | 8 MB | 2 | 0.60 | 10 Mbps | 5% |
| low | 4 MB | 1 | 0.35 | 1 Mbps | 10% |
| tiny | 1 MB | 1 | 0.20 | 0.25 Mbps | 15% |

The budgets are scaled to microcontroller and IoT-class devices because the
baseline CNN needs only about 4-5 MB to train with batch size 32. A client whose
estimated training memory exceeds its budget halves its batch size until it fits
(`--ram-policy adapt`) or is dropped (`--ram-policy drop`). Clients slower than
four times the fastest client in a round (`--deadline-factor`) are dropped as
stragglers. With `--compress int8`, model updates are quantized to INT8 before
upload; downloads stay float32.

**Experiments** (`--preset all`): IID, non-IID alpha 0.5, and non-IID alpha 0.1
with ideal clients; IID and non-IID alpha 0.5 with heterogeneous clients; and
non-IID alpha 0.5 with heterogeneous clients and INT8 updates. The Update 2
results used 10 clients, 10 rounds, and a 5,000-image training subset
(`--data-fraction 0.1`).

**Metrics:** global test accuracy per round, round time, client completion rate,
reason for each dropped client (RAM, offline, deadline), estimated training
memory per client, and upload/download traffic.

**Outputs** in `cifar10/results/`:

- `federated_update2_summary.csv` - one row per experiment (tracked in Git)
- `federated_update2_settings.json` - settings used (tracked in Git)
- `federated_<experiment>_rounds.csv` - per-round accuracy, time, and traffic
- `federated_<experiment>_clients.csv` - per-client status, batch size, and memory
- `federated_<experiment>_label_counts.csv` - class distribution per client

The per-experiment files are ignored by Git; regenerate them by running the
script. `cifar10/plot_update2.py` saves accuracy-per-round, summary, and
label-distribution graphs to `cifar10/plots_analysis/`.

**Limitations.** Client compute and network times are simulated: real local
training time on the test machine is divided by the tier's speed factor, and
communication time is computed from update size and bandwidth. Training memory
is an analytical estimate, not a measurement on real devices. Round times
include background load on the test machine. Models were still improving at
round 10, and results are single-run observations, not multi-seed comparisons.
