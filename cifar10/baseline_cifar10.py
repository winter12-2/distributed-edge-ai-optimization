"""Train the original or improved CIFAR-10 baseline."""

import argparse
from copy import copy
import json
import os
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
VALIDATION_SIZE = 5_000
VALIDATION_SPLIT_SEED = 42
ERASING_PROBABILITY = 0.25
ERASING_SCALE = (0.02, 0.20)
CIFAR10_WIDTHS = (32, 64, 96, 96)


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # CIFAR-10 starts at 32x32, leaving 8x8 after two pooling layers.
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc1(x.reshape(x.shape[0], -1))


class CIFAR10CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, *, widths=CIFAR10_WIDTHS):
        super().__init__()
        first, second, third, fourth = widths
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, first, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(first),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(second, third, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.ReLU(),
            nn.Conv2d(third, fourth, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fourth),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(fourth * 4 * 4, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def cifar10_transform(*, train=False, random_erasing=False):
    operations = []
    if train:
        operations.extend([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    if train and random_erasing:
        operations.append(transforms.RandomErasing(p=ERASING_PROBABILITY, scale=ERASING_SCALE))
    return transforms.Compose(operations)


def check_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def tuned_split_indices(dataset_size):
    generator = torch.Generator().manual_seed(VALIDATION_SPLIT_SEED)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices[VALIDATION_SIZE:], indices[:VALIDATION_SIZE]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    recipe = parser.add_mutually_exclusive_group()
    recipe.add_argument("--improved", action="store_true", help="Train the CIFAR-10-specific CNN with Adam.")
    recipe.add_argument("--tuned", action="store_true", help="Train the same CNN with AdamW and OneCycleLR.")
    parser.add_argument("--epochs", type=int, help="Training epochs (default: 40 with --tuned, otherwise 5).")
    parser.add_argument("--seed", type=int, help="Seed PyTorch model initialization, shuffling, and augmentation.")
    parser.add_argument("--random-erasing", action="store_true", help="Add RandomErasing to the --tuned run.")
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (40 if args.tuned else 5)
    if epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.random_erasing and not args.tuned:
        parser.error("--random-erasing requires --tuned")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    if args.improved or args.tuned:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        train_transform = cifar10_transform(train=True, random_erasing=args.random_erasing)
        eval_transform = cifar10_transform()
        model = CIFAR10CNN().to(device=device, memory_format=torch.channels_last)
        artifact_name = "baseline_tuned" if args.tuned else "baseline_improved"
        if args.random_erasing:
            artifact_name += "_erasing"
    else:
        # The original experiment repeated MNIST normalization across RGB channels.
        mean, std = (0.1307,) * 3, (0.3081,) * 3
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_transform = eval_transform
        model = CNN().to(device)
        artifact_name = "baseline"

    train_dataset = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=eval_transform)
    train_eval_dataset = copy(train_dataset)
    train_eval_dataset.transform = eval_transform
    if args.tuned:
        train_indices, validation_indices = tuned_split_indices(len(train_dataset))
        validation_dataset = Subset(train_eval_dataset, validation_indices)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        train_dataset = Subset(train_dataset, train_indices)
        train_eval_dataset = Subset(train_eval_dataset, train_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    scheduler = None
    if args.tuned:
        max_lr, weight_decay, label_smoothing = 3e-3, 5e-2, 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader),
        )
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    initial_lr = optimizer.param_groups[0]["lr"]
    print(f"Device: {device}\n{model}", flush=True)
    losses = []
    validation_accuracies = []
    epoch_end_learning_rates = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", mininterval=10):
            scores = model(images.to(device))
            loss = criterion(scores, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))
        message = f"Epoch {epoch + 1} Average Loss: {losses[-1]:.4f}"
        if args.tuned:
            validation_accuracies.append(check_accuracy(validation_loader, model, device))
            epoch_end_learning_rates.append(optimizer.param_groups[0]["lr"])
            message += f" | Validation Accuracy: {validation_accuracies[-1]:.2f}%"
        print(message, flush=True)
    train_acc = check_accuracy(train_eval_loader, model, device)
    test_acc = check_accuracy(test_loader, model, device)
    model_path = models_dir / f"{artifact_name}_model.pt"
    torch.save(model.state_dict(), model_path)
    results = {
        "dataset": "CIFAR-10",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": initial_lr,
        "optimizer": type(optimizer).__name__,
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "epoch_losses": losses,
        "device": str(device),
    }
    if args.improved or args.tuned:
        results["architecture"] = type(model).__name__
        results["train_augmentation"] = ["reflect-padded random crop (4 px)", "random horizontal flip"]
    if args.tuned:
        results.update({
            "max_learning_rate": max_lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "scheduler": type(scheduler).__name__,
            "epoch_end_learning_rates": epoch_end_learning_rates,
            "train_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "validation_split_seed": VALIDATION_SPLIT_SEED,
            "epoch_validation_accuracies": validation_accuracies,
        })
    if args.random_erasing:
        results["train_augmentation"].append("random erasing")
        results["random_erasing"] = {"probability": ERASING_PROBABILITY, "scale": ERASING_SCALE}
    if args.seed is not None:
        results["seed"] = args.seed
    with open(results_dir / f"results_{artifact_name}.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(json.dumps(results, indent=4), flush=True)


if __name__ == "__main__":
    main()
