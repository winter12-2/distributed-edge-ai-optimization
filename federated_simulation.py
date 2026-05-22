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
PRUNING_AMOUNT = 0.3          # fraction of weights zeroed in combined mode

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True,  download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
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
    """Convert float32 state dict → int8 (q, scale) pairs."""
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


def prune_state_dict(state_dict, amount=PRUNING_AMOUNT):
    """
    Magnitude-based unstructured pruning on weight tensors only.
    Zeros out the bottom `amount` fraction of weights by absolute value.
    Bias tensors are left untouched.
    """
    pruned = {}
    for key, tensor in state_dict.items():
        t = tensor.float()
        if "weight" in key and t.numel() > 1:
            threshold = torch.quantile(t.abs().flatten(), amount)
            mask = (t.abs() >= threshold).float()
            pruned[key] = t * mask
        else:
            pruned[key] = t
    return pruned


def prune_and_quantize_weights(state_dict, amount=PRUNING_AMOUNT):
    """Prune first, then quantize — combined compression for transmission."""
    pruned = prune_state_dict(state_dict, amount=amount)
    return quantize_weights(pruned)


def average_quantized_weights(quantized_list):
    float_weights = [dequantize_weights(q) for q in quantized_list]
    return average_weights(float_weights)


def sparsity_of_state_dict(state_dict):
    """Returns fraction of weight values that are zero."""
    total  = 0
    zeroed = 0
    for key, tensor in state_dict.items():
        if "weight" in key:
            total  += tensor.numel()
            zeroed += (tensor == 0).sum().item()
    return zeroed / total if total > 0 else 0.0

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
    """Upload + download per round in MB."""
    return (2 * parameter_count * clients * bytes_per_param) / (1024 * 1024)


def effective_bytes_per_param(sparsity, bytes_per_param):
    """
    Effective bytes accounting for sparsity (assumes sparse encoding
    saves storage for zero weights — conservative 1-bit mask overhead ignored).
    """
    return bytes_per_param * (1.0 - sparsity)


def rounds_to_target(accuracy_log, target_pct):
    for rnd, acc in enumerate(accuracy_log, start=1):
        if acc >= target_pct:
            return rnd
    return None


def run_fedavg(global_model, mode="float32"):
    """
    mode options
    ------------
    float32  – transmit raw float32 weights (4 bytes/param)
    int8     – quantize weights before transmission (1 byte/param)
    combined – prune (PRUNING_AMOUNT) then quantize before transmission
               effective bytes per param ≈ 1 × (1 - sparsity)
    """
    assert mode in ("float32", "int8", "combined"), f"Unknown mode: {mode}"

    model        = copy.deepcopy(global_model)
    accuracy_log = []
    results      = []

    parameter_count = model_parameter_count(model)

    base_bpp = 4 if mode == "float32" else 1

    cumulative_cost = 0.0

    label_map = {
        "float32":  "float32",
        "int8":     "int8 (quantized)",
        "combined": "int8 + pruned (combined)",
    }
    label = label_map[mode]

    round_sparsities = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n  --- Round {round_num} ({label}) ---")

        local_weights = []
        for client_id in range(NUM_CLIENTS):
            local_model = copy.deepcopy(model)
            weights     = train_local(local_model, client_loaders[client_id])

            if mode == "float32":
                local_weights.append(weights)
            elif mode == "int8":
                local_weights.append(quantize_weights(weights))
            elif mode == "combined":
                local_weights.append(prune_and_quantize_weights(weights))

        if mode == "float32":
            averaged = average_weights(local_weights)
        else:
            averaged = average_quantized_weights(local_weights)

        model.load_state_dict(averaged)

        round_sparsity = sparsity_of_state_dict(averaged)
        round_sparsities.append(round_sparsity)

        if mode == "combined":
            eff_bpp = effective_bytes_per_param(round_sparsity, base_bpp)
        else:
            eff_bpp = base_bpp

        cost_this_round  = communication_cost_mb(parameter_count, NUM_CLIENTS, eff_bpp)
        cumulative_cost += cost_this_round

        accuracy = evaluate(model)
        accuracy_log.append(accuracy)

        print(f"  Accuracy:           {accuracy:.2f}%")
        print(f"  Sparsity:           {round_sparsity*100:.1f}%")
        print(f"  Cost this round:    {cost_this_round:.4f} MB")
        print(f"  Cumulative cost:    {cumulative_cost:.4f} MB")

        results.append({
            "round":            round_num,
            "accuracy":         round(accuracy, 2),
            "sparsity_pct":     round(round_sparsity * 100, 2),
            "cost_per_round":   round(cost_this_round, 4),
            "cumulative_cost":  round(cumulative_cost, 4),
        })

    avg_sparsity     = sum(round_sparsities) / len(round_sparsities)
    final_cost_round = communication_cost_mb(
        parameter_count, NUM_CLIENTS,
        effective_bytes_per_param(avg_sparsity, base_bpp) if mode == "combined" else base_bpp
    )
    return accuracy_log, results, final_cost_round, cumulative_cost, avg_sparsity


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
    acc_log_f32, results_f32, cost_round_f32, total_cost_f32, sparsity_f32 = run_fedavg(
        global_model_f32, mode="float32"
    )
    print("\n" + "=" * 55)
    print("FedAvg — int8 quantized communication")
    print("=" * 55)

    global_model_i8 = CNN(in_channels=1, num_classes=10).to(device)
    acc_log_i8, results_i8, cost_round_i8, total_cost_i8, sparsity_i8 = run_fedavg(
        global_model_i8, mode="int8"
    )

    print("\n" + "=" * 55)
    print(f"FedAvg — combined (pruned {int(PRUNING_AMOUNT*100)}% + int8 quantized)")
    print("=" * 55)
    global_model_cb = CNN(in_channels=1, num_classes=10).to(device)
    acc_log_cb, results_cb, cost_round_cb, total_cost_cb, sparsity_cb = run_fedavg(
        global_model_cb, mode="combined"
    )

    rounds_95_f32_self = rounds_to_target(acc_log_f32, 0.95 * acc_log_f32[-1])
    rounds_95_i8_self  = rounds_to_target(acc_log_i8,  0.95 * acc_log_i8[-1])
    rounds_95_cb_self  = rounds_to_target(acc_log_cb,  0.95 * acc_log_cb[-1])

    rounds_95_f32_bl = rounds_to_target(acc_log_f32, 0.95 * baseline_acc) if baseline_acc else None
    rounds_95_i8_bl  = rounds_to_target(acc_log_i8,  0.95 * baseline_acc) if baseline_acc else None
    rounds_95_cb_bl  = rounds_to_target(acc_log_cb,  0.95 * baseline_acc) if baseline_acc else None

    print(f"\nRounds to 95% of own final accuracy  — f32: {rounds_95_f32_self}  |  int8: {rounds_95_i8_self}  |  combined: {rounds_95_cb_self}")
    print(f"Rounds to 95% of baseline accuracy   — f32: {rounds_95_f32_bl}  |  int8: {rounds_95_i8_bl}  |  combined: {rounds_95_cb_bl}")

    df_f32 = pd.DataFrame(results_f32)
    df_i8  = pd.DataFrame(results_i8)
    df_cb  = pd.DataFrame(results_cb)

    df_f32.to_csv("results/federated_results_f32.csv",      index=False)
    df_i8.to_csv("results/federated_results_i8.csv",        index=False)
    df_cb.to_csv("results/federated_results_combined.csv",  index=False)

    df_combined = pd.DataFrame({
        "round":                  df_f32["round"],
        "accuracy_f32":           df_f32["accuracy"],
        "accuracy_i8":            df_i8["accuracy"],
        "accuracy_combined":      df_cb["accuracy"],
        "cumulative_cost_f32":    df_f32["cumulative_cost"],
        "cumulative_cost_i8":     df_i8["cumulative_cost"],
        "cumulative_cost_combined": df_cb["cumulative_cost"],
        "sparsity_combined_pct":  df_cb["sparsity_pct"],
    })
    df_combined.to_csv("results/federated_results.csv", index=False)

    summary = {
        "num_clients":    NUM_CLIENTS,
        "rounds":         ROUNDS,
        "local_epochs":   LOCAL_EPOCHS,
        "pruning_amount": PRUNING_AMOUNT,
        "parameters":     parameter_count,
        "float32": {
            "final_accuracy":           round(acc_log_f32[-1], 2),
            "cost_per_round_mb":        round(cost_round_f32, 4),
            "total_cost_mb":            round(total_cost_f32, 4),
            "avg_sparsity_pct":         round(sparsity_f32 * 100, 2),
            "rounds_to_95pct_self":     rounds_95_f32_self,
            "rounds_to_95pct_baseline": rounds_95_f32_bl,
            "per_round":                results_f32,
        },
        "int8": {
            "final_accuracy":           round(acc_log_i8[-1], 2),
            "cost_per_round_mb":        round(cost_round_i8, 4),
            "total_cost_mb":            round(total_cost_i8, 4),
            "avg_sparsity_pct":         round(sparsity_i8 * 100, 2),
            "rounds_to_95pct_self":     rounds_95_i8_self,
            "rounds_to_95pct_baseline": rounds_95_i8_bl,
            "per_round":                results_i8,
        },
        "combined": {
            "final_accuracy":           round(acc_log_cb[-1], 2),
            "cost_per_round_mb":        round(cost_round_cb, 4),
            "total_cost_mb":            round(total_cost_cb, 4),
            "avg_sparsity_pct":         round(sparsity_cb * 100, 2),
            "rounds_to_95pct_self":     rounds_95_cb_self,
            "rounds_to_95pct_baseline": rounds_95_cb_bl,
            "per_round":                results_cb,
        },
    }

    with open("results/federated_results.json", "w") as f:
        json.dump(summary, f, indent=4)

    rounds_axis = df_combined["round"]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds_axis, df_combined["accuracy_f32"],      marker="o", label="float32")
    plt.plot(rounds_axis, df_combined["accuracy_i8"],       marker="s", label="int8",     linestyle="--")
    plt.plot(rounds_axis, df_combined["accuracy_combined"], marker="^", label="combined", linestyle=":")
    if baseline_acc:
        plt.axhline(y=baseline_acc, color="grey", linestyle="-.", linewidth=1, label="Centralised baseline")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Federated Learning Accuracy: float32 vs int8 vs Combined")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/federated_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(rounds_axis, df_combined["cumulative_cost_f32"],      marker="o", label="float32")
    plt.plot(rounds_axis, df_combined["cumulative_cost_i8"],       marker="s", label="int8",     linestyle="--")
    plt.plot(rounds_axis, df_combined["cumulative_cost_combined"], marker="^", label="combined", linestyle=":")
    plt.xlabel("Communication Round")
    plt.ylabel("Cumulative Cost (MB)")
    plt.title("Communication Cost: float32 vs int8 vs Combined")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/federated_communication_cost.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_combined["cumulative_cost_f32"],      df_combined["accuracy_f32"],      marker="o", label="float32")
    plt.plot(df_combined["cumulative_cost_i8"],       df_combined["accuracy_i8"],       marker="s", label="int8",     linestyle="--")
    plt.plot(df_combined["cumulative_cost_combined"], df_combined["accuracy_combined"], marker="^", label="combined", linestyle=":")
    plt.xlabel("Cumulative Communication Cost (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Communication Cost (efficiency frontier)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/federated_efficiency_frontier.png", dpi=150)
    plt.close()

    print("\n\n===== FEDERATED LEARNING SUMMARY =====")
    print(f"{'Mode':<22} {'Final Acc':>10} {'Total Cost (MB)':>16} {'Rounds→95% self':>17}")
    print("-" * 70)
    for tag, acc_log, total_cost, r95 in [
        ("float32",  acc_log_f32, total_cost_f32, rounds_95_f32_self),
        ("int8",     acc_log_i8,  total_cost_i8,  rounds_95_i8_self),
        ("combined", acc_log_cb,  total_cost_cb,  rounds_95_cb_self),
    ]:
        print(f"{tag:<22} {acc_log[-1]:>9.2f}%  {total_cost:>15.4f}  {str(r95):>17}")

    print("\nTraining complete.")
    print(df_combined.to_string(index=False))
    print("\nSaved results to results/federated_results.csv")
    print("Saved summary to results/federated_results.json")
    print("Saved plots   to results/federated_accuracy.png,")
    print("                 results/federated_communication_cost.png,")
    print("                 results/federated_efficiency_frontier.png")