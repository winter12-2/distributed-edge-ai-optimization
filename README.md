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

# Full MNIST pipeline, including baseline retraining
.\venv\Scripts\python.exe run_everything.py

# A single MNIST experiment
.\venv\Scripts\python.exe mnist\optimized_quantized.py
```

Open MNIST notebooks with `mnist/notebooks/` as the notebook working directory.

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
