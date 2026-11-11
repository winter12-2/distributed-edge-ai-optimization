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
