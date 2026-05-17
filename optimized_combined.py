import time
import json
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

from baseline_mnist import CNN

DEVICE = torch.device("cpu")

PRUNING_TYPE     = "unstructured"
PRUNING_AMOUNT   = 0.3
FINE_TUNE_LR     = 0.0005
FINE_TUNE_EPOCHS = 2
BATCH_SIZE       = 64

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

criterion = nn.CrossEntropyLoss()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    start = time.perf_counter()
    with torch.no_grad():
        for data, targets in loader:
            data    = data.to(device)
            targets = targets.to(device)
            scores  = model(data)
            loss    = criterion(scores, targets)
            total_loss += loss.item()
            _, predictions = scores.max(1)
            total   += targets.size(0)
            correct += (predictions == targets).sum().item()
    elapsed = time.perf_counter() - start

    avg_loss      = total_loss / len(loader)
    accuracy      = 100 * correct / total
    ms_per_sample = (elapsed / total) * 1000
    return avg_loss, accuracy, ms_per_sample


def count_zero_weights(model):
    total  = 0
    zeroed = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total  += param.numel()
            zeroed += (param == 0).sum().item()
    return zeroed, total


def apply_unstructured_pruning(model, amount):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def apply_structured_pruning(model, amount):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
    return model


def make_pruning_permanent(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


def fine_tune(model, loader, criterion, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, targets in loader:
            data    = data.to(device)
            targets = targets.to(device)
            scores  = model(data)
            loss    = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"  Fine-tune Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    return model


if __name__ == "__main__":

    if not os.path.exists("models/baseline_model.pt"):
        print("Error: baseline_model.pt not found. Run baseline_mnist.py first.")
        exit()

    baseline_model = CNN(in_channels=1, num_classes=10)
    baseline_model.load_state_dict(torch.load("models/baseline_model.pt", map_location=DEVICE))
    baseline_model.to(DEVICE)

    total_params = sum(p.numel() for p in baseline_model.parameters())

    bl_loss, bl_acc, bl_ms = evaluate(baseline_model, test_loader, criterion, DEVICE)
    print(f"Baseline accuracy: {bl_acc:.2f}%")

    pruned_model = CNN(in_channels=1, num_classes=10)
    pruned_model.load_state_dict(torch.load("models/baseline_model.pt", map_location=DEVICE))
    pruned_model.to(DEVICE)

    if PRUNING_TYPE == "unstructured":
        pruned_model = apply_unstructured_pruning(pruned_model, PRUNING_AMOUNT)
    elif PRUNING_TYPE == "structured":
        pruned_model = apply_structured_pruning(pruned_model, PRUNING_AMOUNT)

    _, acc_before_ft, _ = evaluate(pruned_model, test_loader, criterion, DEVICE)
    print(f"Accuracy before fine-tune: {acc_before_ft:.2f}%")

    print(f"Fine-tuning for {FINE_TUNE_EPOCHS} epoch(s)...")
    pruned_model = fine_tune(pruned_model, train_loader, criterion, DEVICE, epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR)

    pruned_model = make_pruning_permanent(pruned_model)

    zeroed, total_w = count_zero_weights(pruned_model)
    sparsity = 100 * zeroed / total_w if total_w > 0 else 0.0
    print(f"Sparsity: {sparsity:.1f}% ({zeroed}/{total_w} weights zeroed)")

    pr_loss, pr_acc, pr_ms = evaluate(pruned_model, test_loader, criterion, DEVICE)
    print(f"Pruned accuracy: {pr_acc:.2f}%")

    pruned_path = "models/combined_pruned_only.pt"
    torch.save(pruned_model.state_dict(), pruned_path)
    pruned_size_mb = os.path.getsize(pruned_path) / (1024 * 1024)

    combined_model = torch.quantization.quantize_dynamic(pruned_model, {nn.Linear}, dtype=torch.qint8)

    cb_loss, cb_acc, cb_ms = evaluate(combined_model, test_loader, criterion, DEVICE)
    print(f"Combined accuracy: {cb_acc:.2f}%")

    combined_path = "models/combined_model.pt"
    torch.save(combined_model.state_dict(), combined_path)
    combined_size_mb = os.path.getsize(combined_path) / (1024 * 1024)

    print("\n===== COMBINED MODEL SUMMARY =====")
    print(f"{'Model':<22} {'Accuracy':>10} {'Size (MB)':>11} {'ms/sample':>11}")
    print("-" * 58)
    print(f"{'Baseline':<22} {bl_acc:>9.2f}%  {'N/A':>10}  {bl_ms:>10.3f}")
    print(f"{'Pruned only':<22} {pr_acc:>9.2f}%  {pruned_size_mb:>10.4f}  {pr_ms:>10.3f}")
    print(f"{'Pruned + Quantized':<22} {cb_acc:>9.2f}%  {combined_size_mb:>10.4f}  {cb_ms:>10.3f}")

    results = {
        "model_name": "Combined Pruned + Quantized CNN",
        "pruning_type": PRUNING_TYPE,
        "pruning_amount": PRUNING_AMOUNT,
        "sparsity_percent": round(sparsity, 2),
        "quantization_type": "PyTorch dynamic int8 (Linear layers)",
        "parameters": total_params,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "baseline": {
            "test_accuracy": round(bl_acc, 2),
            "test_loss":     round(bl_loss, 4),
            "ms_per_sample": round(bl_ms, 4),
        },
        "pruned_only": {
            "test_accuracy":           round(pr_acc, 2),
            "test_accuracy_before_ft": round(acc_before_ft, 2),
            "test_loss":               round(pr_loss, 4),
            "model_size_mb":           round(pruned_size_mb, 4),
            "ms_per_sample":           round(pr_ms, 4),
        },
        "combined": {
            "test_accuracy": round(cb_acc, 2),
            "test_loss":     round(cb_loss, 4),
            "model_size_mb": round(combined_size_mb, 4),
            "ms_per_sample": round(cb_ms, 4),
        },
    }

    results_path = "results/results_combined.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nModel saved to {pruned_path}")
    print(f"Model saved to {combined_path}")
    print(f"Results saved to {results_path}")