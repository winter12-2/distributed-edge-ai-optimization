import os
import copy
import json
import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import pandas as pd
import matplotlib.pyplot as plt

from baseline_mnist import CNN

os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLIENTS   = 5
ROUNDS        = 10
LOCAL_EPOCHS  = 1
BATCH_SIZE    = 64
LEARNING_RATE = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

client_size     = len(train_dataset) // NUM_CLIENTS
lengths         = [client_size] * NUM_CLIENTS
client_datasets = random_split(train_dataset, lengths)

client_loaders = [
    DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    for ds in client_datasets
]


def train_local(model, loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(LOCAL_EPOCHS):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def average_weights(local_weights):
    avg = copy.deepcopy(local_weights[0])
    for key in avg.keys():
        for i in range(1, len(local_weights)):
            avg[key] += local_weights[i][key]
        avg[key] = avg[key] / len(local_weights)
    return avg


def quantize_weights(state_dict):
    quantized = {}
    for key, tensor in state_dict.items():
        t = tensor.float()
        scale = t.abs().max() / 127.0
        scale = scale if scale > 0 else torch.tensor(1.0)
        q = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int8)
        quantized[key] = (q, scale)
    return quantized


def dequantize_weights(quantized):
    state_dict = {}
    for key, (q, scale) in quantized.items():
        state_dict[key] = q.float() * scale
    return state_dict


def average_quantized_weights(quantized_list):
    float_weights = [dequantize_weights(q) for q in quantized_list]
    return average_weights(float_weights)


def evaluate(model):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def model_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def communication_cost_mb(parameter_count, clients, bytes_per_param):
    return (2 * parameter_count * clients * bytes_per_param) / (1024 * 1024)


def rounds_to_target(accuracy_log, target_pct):
    for rnd, acc in enumerate(accuracy_log, start=1):
        if acc >= target_pct:
            return rnd
    return None


def run_fedavg(global_model, quantized=False):
    model = copy.deepcopy(global_model)
    accuracy_log = []
    results      = []

    parameter_count     = model_parameter_count(model)
    bytes_per_param     = 1 if quantized else 4
    cost_per_round      = communication_cost_mb(parameter_count, NUM_CLIENTS, bytes_per_param)
    cumulative_cost     = 0.0

    label = "int8 (quantized)" if quantized else "float32"

    for round_num in range(1, ROUNDS + 1):
        print(f"\n  --- Round {round_num} ({label}) ---")

        local_weights = []
        for client_id in range(NUM_CLIENTS):
            local_model = copy.deepcopy(model)
            weights     = train_local(local_model, client_loaders[client_id])

            if quantized:
                local_weights.append(quantize_weights(weights))
            else:
                local_weights.append(weights)

        if quantized:
            averaged = average_quantized_weights(local_weights)
        else:
            averaged = average_weights(local_weights)

        model.load_state_dict(averaged)

        accuracy         = evaluate(model)
        cumulative_cost += cost_per_round
        accuracy_log.append(accuracy)

        print(f"  Accuracy:         {accuracy:.2f}%")
        print(f"  Cumulative cost:  {cumulative_cost:.4f} MB")

        results.append({
            "round":           round_num,
            "accuracy":        round(accuracy, 2),
            "cost_per_round":  round(cost_per_round, 4),
            "cumulative_cost": round(cumulative_cost, 4),
        })

    return accuracy_log, results, cost_per_round, cumulative_cost


if __name__ == "__main__":
    baseline_acc = None
    if os.path.exists("results/results_baseline.json"):
        with open("results/results_baseline.json") as f:
            baseline_acc = json.load(f)["test_accuracy"]

    parameter_count = model_parameter_count(CNN(in_channels=1, num_classes=10))
    print("\n" + "=" * 55)
    print("FedAvg — float32 communication")
    print("=" * 55)

    global_model_f32 = CNN(in_channels=1, num_classes=10).to(device)
    acc_log_f32, results_f32, cost_round_f32, total_cost_f32 = run_fedavg(
        global_model_f32, quantized=False
    )
    print("\n" + "=" * 55)
    print("FedAvg — int8 quantized communication")
    print("=" * 55)

    global_model_i8 = CNN(in_channels=1, num_classes=10).to(device)
    acc_log_i8, results_i8, cost_round_i8, total_cost_i8 = run_fedavg(
        global_model_i8, quantized=True
    )
    rounds_95_f32_self     = rounds_to_target(acc_log_f32, 0.95 * acc_log_f32[-1])
    rounds_95_i8_self      = rounds_to_target(acc_log_i8,  0.95 * acc_log_i8[-1])
    rounds_95_f32_baseline = rounds_to_target(acc_log_f32, 0.95 * baseline_acc) if baseline_acc else None
    rounds_95_i8_baseline  = rounds_to_target(acc_log_i8,  0.95 * baseline_acc) if baseline_acc else None

    print(f"\nRounds to 95% of own final accuracy  — f32: {rounds_95_f32_self}  |  int8: {rounds_95_i8_self}")
    print(f"Rounds to 95% of baseline accuracy   — f32: {rounds_95_f32_baseline}  |  int8: {rounds_95_i8_baseline}")
    df_f32 = pd.DataFrame(results_f32)
    df_i8  = pd.DataFrame(results_i8)

    df_f32.to_csv("results/federated_results_f32.csv", index=False)
    df_i8.to_csv("results/federated_results_i8.csv",   index=False)

    df_combined = pd.DataFrame({
        "round":       df_f32["round"],
        "accuracy_f32": df_f32["accuracy"],
        "accuracy_i8":  df_i8["accuracy"],
        "cumulative_cost_f32": df_f32["cumulative_cost"],
        "cumulative_cost_i8":  df_i8["cumulative_cost"],
    })
    df_combined.to_csv("results/federated_results.csv", index=False)
    summary = {
        "num_clients":  NUM_CLIENTS,
        "rounds":       ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "parameters":   parameter_count,
        "float32": {
            "final_accuracy":         round(acc_log_f32[-1], 2),
            "cost_per_round_mb":      round(cost_round_f32,  4),
            "total_cost_mb":          round(total_cost_f32,  4),
            "rounds_to_95pct_self":   rounds_95_f32_self,
            "rounds_to_95pct_baseline": rounds_95_f32_baseline,
            "per_round":              results_f32,
        },
        "int8": {
            "final_accuracy":         round(acc_log_i8[-1], 2),
            "cost_per_round_mb":      round(cost_round_i8,  4),
            "total_cost_mb":          round(total_cost_i8,  4),
            "rounds_to_95pct_self":   rounds_95_i8_self,
            "rounds_to_95pct_baseline": rounds_95_i8_baseline,
            "per_round":              results_i8,
        },
    }

    with open("results/federated_results.json", "w") as f:
        json.dump(summary, f, indent=4)

    plt.figure()
    plt.plot(df_combined["round"], df_combined["accuracy_f32"], marker="o", label="float32")
    plt.plot(df_combined["round"], df_combined["accuracy_i8"],  marker="s", label="int8", linestyle="--")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning Accuracy: float32 vs int8")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/federated_accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(df_combined["round"], df_combined["cumulative_cost_f32"], marker="o", label="float32")
    plt.plot(df_combined["round"], df_combined["cumulative_cost_i8"],  marker="s", label="int8", linestyle="--")
    plt.xlabel("Communication Round")
    plt.ylabel("Cumulative Cost (MB)")
    plt.title("Communication Cost: float32 vs int8")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/federated_communication_cost.png")
    plt.close()

    print("\nTraining complete.")
    print(df_combined.to_string(index=False))
    print(f"\nSaved results to results/federated_results.csv")
    print(f"Saved summary to results/federated_results.json")
    print(f"Saved plots to results/federated_accuracy.png and results/federated_communication_cost.png")