import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import json
import os
from baseline_mnist import CNN

DEVICE = torch.device("cpu")

os.makedirs("models", exist_ok=True)
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            loss = criterion(scores, targets)
            total_loss += loss.item()
            _, predictions = scores.max(1)
            total += targets.size(0)
            correct += (predictions == targets).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def count_zero_weights(model):
    total = 0
    zeroed = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            zeroed += (param == 0).sum().item()
    return zeroed, total


def apply_unstructured_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def apply_structured_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
    return model


def make_pruning_permanent(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


def fine_tune(model, train_loader, criterion, device, epochs=2, lr=0.0005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"  Fine-tune Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    return model

criterion = nn.CrossEntropyLoss()

if not os.path.exists("models/baseline_model.pt"):
    print("Error: baseline_model.pt not found. Run baseline_mnist.py first.")
    exit()

def run_pruning_experiment(pruning_type, amount, fine_tune_epochs=2):
    print(f"\n{'='*50}")
    print(f"Pruning type : {pruning_type}  |  amount : {amount}")
    print(f"{'='*50}")

    model = CNN(in_channels=1, num_classes=10)
    model.load_state_dict(
        torch.load("models/baseline_model.pt", map_location=DEVICE)
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())


    if pruning_type == "unstructured":
        model = apply_unstructured_pruning(model, amount=amount)
    elif pruning_type == "structured":
        model = apply_structured_pruning(model, amount=amount)

    loss_before, acc_before = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Accuracy before fine-tune : {acc_before:.2f}%")

    if fine_tune_epochs > 0:
        print(f"Fine-tuning for {fine_tune_epochs} epoch(s)...")
        model = fine_tune(model, train_loader, criterion, DEVICE, epochs=fine_tune_epochs)

    model = make_pruning_permanent(model)

    zeroed, total_w = count_zero_weights(model)
    sparsity = 100 * zeroed / total_w if total_w > 0 else 0
    print(f"Sparsity after pruning : {sparsity:.1f}%  ({zeroed}/{total_w} weights zeroed)")
    loss_after, acc_after = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Accuracy after  fine-tune : {acc_after:.2f}%")

    tag = f"{pruning_type}_p{int(amount*100)}"
    model_path = f"models/pruned_{tag}.pt"
    torch.save(model.state_dict(), model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    results = {
        "model_name": f"Pruned CNN ({pruning_type}, {int(amount*100)}%)",
        "pruning_type": pruning_type,
        "pruning_amount": amount,
        "sparsity_percent": round(sparsity, 2),
        "test_accuracy_before_finetune": round(acc_before, 2),
        "test_accuracy_after_finetune": round(acc_after, 2),
        "test_loss": round(loss_after, 4),
        "model_size_mb": round(model_size_mb, 4),
        "parameters": total_params,
        "fine_tune_epochs": fine_tune_epochs,
    }

    results_path = f"results/results_pruned_{tag}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Results saved to {results_path}")

    return results

if __name__ == "__main__":

    experiments = [
        ("unstructured",   0.3),
        ("unstructured",   0.6),
        ("structured",     0.3),
        ("structured",     0.6),
    ]

    all_results = []
    for pruning_type, amount in experiments:
        result = run_pruning_experiment(pruning_type, amount, fine_tune_epochs=2)
        all_results.append(result)

    print("\n\n===== PRUNING SUMMARY =====")
    print(f"{'Model':<40} {'Sparsity':>10} {'Accuracy':>10} {'Size (MB)':>10}")
    print("-" * 75)
    for r in all_results:
        print(
            f"{r['model_name']:<40} "
            f"{r['sparsity_percent']:>9.1f}% "
            f"{r['test_accuracy_after_finetune']:>9.2f}% "
            f"{r['model_size_mb']:>10.4f}"
        )

    with open("results/results_pruning_summary.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nCombined summary saved to results/results_pruning_summary.json")