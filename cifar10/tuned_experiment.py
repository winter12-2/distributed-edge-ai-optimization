"""Shared defaults for experiments using the tuned CIFAR-10 baseline."""

import torch

if __package__:
    from .baseline_cifar10 import CIFAR10CNN, ROOT
else:
    from baseline_cifar10 import CIFAR10CNN, ROOT


TUNED_CHECKPOINT = ROOT / "models" / "baseline_tuned_model.pt"
TUNED_METRICS = ROOT / "results" / "results_baseline_tuned.json"


def load_tuned_model(checkpoint=TUNED_CHECKPOINT, *, device="cpu"):
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing CIFAR-10 checkpoint: {checkpoint}. Train it with baseline_cifar10.py --tuned."
        )
    model = CIFAR10CNN().to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model
