"""Run the MNIST baseline procedure on CIFAR-10 without accuracy tuning."""

import json
import os
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent


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


def main():
    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 5
    # Repeat the existing MNIST normalization across RGB channels, without tuning.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,) * 3, (0.3081,) * 3),
    ])
    train_dataset = datasets.CIFAR10(ROOT / "data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(ROOT / "data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Device: {device}\n{model}", flush=True)
    losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", mininterval=10):
            scores = model(images.to(device))
            loss = criterion(scores, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))
        print(f"Epoch {epoch + 1} Average Loss: {losses[-1]:.4f}", flush=True)
    train_acc = check_accuracy(train_loader, model, device)
    test_acc = check_accuracy(test_loader, model, device)
    model_path = models_dir / "baseline_model.pt"
    torch.save(model.state_dict(), model_path)
    results = {
        "dataset": "CIFAR-10",
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "model_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "normalization_mean": [0.1307] * 3,
        "normalization_std": [0.3081] * 3,
        "epoch_losses": losses,
        "device": str(device),
    }
    with open(results_dir / "results_baseline.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)
    print(json.dumps(results, indent=4), flush=True)


if __name__ == "__main__":
    main()
