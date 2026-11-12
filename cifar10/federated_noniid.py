import argparse
import copy
import json
import os
import time
 
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
 
from torchvision import datasets, transforms

from baseline_cifar10 import CNN, ROOT
 
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
NORM_MEAN, NORM_STD = (0.1307,) * 3, (0.3081,) * 3  # same values as baseline_cifar10.py
 
 
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
 
def get_datasets():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
    train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    return train, test
 
 
@torch.no_grad()
def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen
CLIENT_TIERS = {
    "high": {"ram_mb": 64.0, "threads": 4, "cpu_speed": 1.00, "bandwidth_mbps": 50.0, "dropout": 0.02},
    "mid":  {"ram_mb": 8.0,  "threads": 2, "cpu_speed": 0.60, "bandwidth_mbps": 10.0, "dropout": 0.05},
    "low":  {"ram_mb": 4.0,  "threads": 1, "cpu_speed": 0.35, "bandwidth_mbps": 1.0,  "dropout": 0.10},
    "tiny": {"ram_mb": 1.0,  "threads": 1, "cpu_speed": 0.20, "bandwidth_mbps": 0.25, "dropout": 0.15},
}
TIER_MIX = ["high", "high", "mid", "mid", "mid", "low", "low", "low", "tiny", "tiny"]
 
PRESETS = [
    {"tag": "iid_ideal",            "partition": "iid",       "alpha": None, "hetero": False, "compress": "none"},
    {"tag": "noniid_a0.5_ideal",    "partition": "dirichlet", "alpha": 0.5,  "hetero": False, "compress": "none"},
    {"tag": "noniid_a0.1_ideal",    "partition": "dirichlet", "alpha": 0.1,  "hetero": False, "compress": "none"},
    {"tag": "iid_hetero",           "partition": "iid",       "alpha": None, "hetero": True,  "compress": "none"},
    {"tag": "noniid_a0.5_hetero",   "partition": "dirichlet", "alpha": 0.5,  "hetero": True,  "compress": "none"},
    {"tag": "noniid_a0.5_hetero_int8", "partition": "dirichlet", "alpha": 0.5, "hetero": True, "compress": "int8"},
]
 

def partition_iid(labels, n_clients, rng):
    idx = rng.permutation(len(labels))
    return [part.tolist() for part in np.array_split(idx, n_clients)]
 
 
def partition_dirichlet(labels, n_clients, alpha, rng, min_size=20):
    labels = np.asarray(labels)
    n_classes = int(labels.max()) + 1
    for attempt in range(1000):
        if attempt == 500:  # very skewed split on a small dataset: relax the minimum
            min_size = max(1, min_size // 4)
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx_c, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [rng.permutation(p).tolist() for p in parts]
    raise RuntimeError("Could not create a Dirichlet split; try a larger alpha or more data")
 
 
def label_counts(labels, parts):
    labels = np.asarray(labels)
    return np.stack([np.bincount(labels[p], minlength=10) for p in parts])
 
 
# Memory / communication estimates
def activation_bytes_per_image(model):
    """Bytes of layer outputs for one image (forward pass)."""
    total = []
    hooks = [m.register_forward_hook(lambda mod, i, o: total.append(o.numel() * 4))
             for m in model.modules() if len(list(m.children())) == 0]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32))
    for h in hooks:
        h.remove()
    return sum(total)
 
 
def training_memory_mb(param_bytes, act_bytes, batch_size):
    """Estimated peak memory for local SGD training:
    weights + gradients + momentum (3x params) + activations kept for backward and their
    gradients (2x activations) + the input batch."""
    input_bytes = batch_size * 3 * 32 * 32 * 4
    return (3 * param_bytes + 2 * act_bytes * batch_size + input_bytes) / (1024 * 1024)
 
 
def state_bytes(state, compress):
    """Bytes needed to send a model state / update."""
    total = 0
    for v in state.values():
        if v.is_floating_point():
            total += v.numel() * (1 if compress == "int8" else 4) + (4 if compress == "int8" else 0)
        else:
            total += v.numel() * v.element_size()
    return total
 
 
def int8_roundtrip(t):
    """Symmetric per-tensor INT8 quantize -> dequantize (what the server receives)."""
    scale = t.abs().max() / 127.0
    if scale == 0:
        return t.clone()
    return torch.clamp(torch.round(t / scale), -127, 127) * scale
# Client training and aggregation
def local_train(global_model, dataset, indices, batch_size, epochs, lr, threads):
    torch.set_num_threads(threads)
    model = copy.deepcopy(global_model)
    model.train()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()
    return model.state_dict(), time.perf_counter() - start
 
 
def fedavg(global_state, client_states, weights, compress):
    new_state = copy.deepcopy(global_state)
    total = float(sum(weights))
    for key, g in global_state.items():
        if not g.is_floating_point():  
            new_state[key] = client_states[0][key].clone()
            continue
        avg_delta = torch.zeros_like(g)
        for s, w in zip(client_states, weights):
            delta = s[key] - g
            if compress == "int8":
                delta = int8_roundtrip(delta)
            avg_delta += delta * (w / total)
        new_state[key] = g + avg_delta
    return new_state
 

def run_experiment(cfg, args, train_set, test_loader, labels):
    tag = cfg["tag"]
    print(f"\n=== Experiment: {tag} ===")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
 
    if cfg["partition"] == "iid":
        parts = partition_iid(labels, args.clients, rng)
    else:
        parts = partition_dirichlet(labels, args.clients, cfg["alpha"], rng)
    counts = label_counts(labels, parts)
    pd.DataFrame(counts, columns=[f"class_{c}" for c in range(10)]).assign(
        client=range(args.clients)).to_csv(RESULTS_DIR / f"federated_{tag}_label_counts.csv", index=False)
 
    max_threads = max(1, os.cpu_count() or 1)
    if cfg["hetero"]:
        tiers = [TIER_MIX[i % len(TIER_MIX)] for i in range(args.clients)]
    else:
        tiers = ["ideal"] * args.clients
    profiles = []
    for t in tiers:
        if t == "ideal":
            profiles.append({"ram_mb": float("inf"), "threads": max_threads, "cpu_speed": 1.0,
                             "bandwidth_mbps": float("inf"), "dropout": 0.0})
        else:
            p = dict(CLIENT_TIERS[t])
            p["threads"] = min(p["threads"], max_threads)
            profiles.append(p)
 
    global_model = CNN()
    param_bytes = sum(p.numel() * 4 for p in global_model.parameters())
    act_bytes = activation_bytes_per_image(global_model)
    download_bytes = state_bytes(global_model.state_dict(), "none")
 
    round_rows, client_rows = [], []
    cumulative_mb = 0.0
    for rnd in range(1, args.rounds + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        results = []  # (client, state, n_samples, sim_time)
        for c in range(args.clients):
            prof = profiles[c]
            row = {"round": rnd, "client": c, "tier": tiers[c], "samples": len(parts[c])}
 
            # RAM check
            bs = args.batch_size
            mem = training_memory_mb(param_bytes, act_bytes, bs)
            if args.ram_policy == "adapt":
                while mem > prof["ram_mb"] and bs > 1:
                    bs //= 2
                    mem = training_memory_mb(param_bytes, act_bytes, bs)
            row.update({"batch_size": bs, "est_train_mem_mb": round(mem, 2), "ram_mb": prof["ram_mb"]})
 
            if mem > prof["ram_mb"]:
                row.update({"status": "dropped_ram", "sim_time_s": None})
                client_rows.append(row)
                continue
            if rng.random() < prof["dropout"]:
                row.update({"status": "dropped_offline", "sim_time_s": None})
                client_rows.append(row)
                continue
 
            state, wall = local_train(global_model, train_set, parts[c], bs,
                                      args.local_epochs, args.lr, prof["threads"])
            upload = state_bytes(state, cfg["compress"])
            comm_s = 0.0
            if prof["bandwidth_mbps"] != float("inf"):
                comm_s = (upload + download_bytes) * 8 / (prof["bandwidth_mbps"] * 1e6)
            sim_time = wall / prof["cpu_speed"] + comm_s
            row.update({"status": "ok", "sim_time_s": round(sim_time, 3),
                        "compute_s": round(wall / prof["cpu_speed"], 3), "comm_s": round(comm_s, 3),
                        "upload_mb": upload / (1024 * 1024)})
            client_rows.append(row)
            results.append((c, state, len(parts[c]), sim_time, upload))
 
    
        if results and args.deadline_factor > 0 and cfg["hetero"]:
            deadline = args.deadline_factor * min(r[3] for r in results)
            kept = [r for r in results if r[3] <= deadline]
            kept_ids = {r[0] for r in kept}
            for r in results:
                if r[0] not in kept_ids:
                    for cr in client_rows:
                        if cr["round"] == rnd and cr["client"] == r[0]:
                            cr["status"] = "dropped_deadline"
            wasted = sum(r[4] for r in results if r[0] not in kept_ids)
            results_used = kept
        else:
            deadline, wasted, results_used = None, 0, results
 
        if results_used:
            new_state = fedavg(global_state, [r[1] for r in results_used],
                               [r[2] for r in results_used], cfg["compress"])
            global_model.load_state_dict(new_state)
 
        torch.set_num_threads(max_threads)
        test_loss, test_acc = evaluate(global_model, test_loader)
        upload_mb = (sum(r[4] for r in results_used) + wasted) / (1024 * 1024)
        down_mb = download_bytes * args.clients / (1024 * 1024)
        cumulative_mb += upload_mb + down_mb
        # synchronous FedAvg: the round lasts as long as the slowest client the server waits for
        if deadline is not None:
            round_time = deadline if len(results_used) < len(results) else max(r[3] for r in results_used)
        else:
            round_time = max((r[3] for r in results_used), default=0.0)
        round_rows.append({
            "experiment": tag, "round": rnd, "test_accuracy": round(test_acc, 4),
            "test_loss": round(test_loss, 4), "round_time_s": round(round_time, 2),
            "clients_completed": len(results_used), "clients_total": args.clients,
            "completion_rate": round(len(results_used) / args.clients, 3),
            "upload_mb": round(upload_mb, 4), "download_mb": round(down_mb, 4),
            "cumulative_comm_mb": round(cumulative_mb, 4),
        })
        print(f"  round {rnd:>2}/{args.rounds}  acc={test_acc:.4f}  completed={len(results_used)}/"
              f"{args.clients}  round_time={round_time:.1f}s  comm={cumulative_mb:.1f}MB")
 
    rounds_df = pd.DataFrame(round_rows)
    clients_df = pd.DataFrame(client_rows)
    rounds_df.to_csv(RESULTS_DIR / f"federated_{tag}_rounds.csv", index=False)
    clients_df.to_csv(RESULTS_DIR / f"federated_{tag}_clients.csv", index=False)
 
    reach = rounds_df[rounds_df["test_accuracy"] >= args.target_acc]["round"]
    return {
        "experiment": tag,
        "partition": cfg["partition"] if cfg["alpha"] is None else f"dirichlet(alpha={cfg['alpha']})",
        "heterogeneous": cfg["hetero"],
        "compression": cfg["compress"],
        "final_accuracy": rounds_df["test_accuracy"].iloc[-1],
        "best_accuracy": rounds_df["test_accuracy"].max(),
        f"rounds_to_{int(args.target_acc * 100)}pct": int(reach.iloc[0]) if len(reach) else None,
        "mean_round_time_s": round(rounds_df["round_time_s"].mean(), 2),
        "total_time_s": round(rounds_df["round_time_s"].sum(), 1),
        "mean_completion_rate": round(rounds_df["completion_rate"].mean(), 3),
        "total_comm_mb": round(cumulative_mb, 2),
        "dropped_ram": int((clients_df["status"] == "dropped_ram").sum()),
        "dropped_offline": int((clients_df["status"] == "dropped_offline").sum()),
        "dropped_deadline": int((clients_df["status"] == "dropped_deadline").sum()),
        "max_est_train_mem_mb": round(clients_df["est_train_mem_mb"].max(), 2),
    }
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Federated CIFAR-10: IID vs non-IID, heterogeneous clients")
    p.add_argument("--preset", choices=["all", "none"], default="none",
                   help="'all' runs the full experiment grid (PRESETS)")
    p.add_argument("--tag", default="custom")
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="dirichlet")
    p.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (smaller = more non-IID)")
    p.add_argument("--hetero", action="store_true", help="heterogeneous clients with RAM/CPU limits")
    p.add_argument("--compress", choices=["none", "int8"], default="none")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--ram-policy", choices=["adapt", "drop"], default="adapt")
    p.add_argument("--deadline-factor", type=float, default=4.0,
                   help="hetero only: drop clients slower than factor x fastest (0 = no deadline)")
    p.add_argument("--target-acc", type=float, default=0.45)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="use part of the training set to make runs faster (e.g. 0.3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="tiny run to check the code works")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.quick:
        args.rounds, args.data_fraction = 2, 0.04
    train_set, test_set = get_datasets()
    labels = np.array(train_set.targets)
    if args.data_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(labels), int(len(labels) * args.data_fraction), replace=False))
        train_set = Subset(train_set, keep.tolist())
        labels = labels[keep]
        test_set = Subset(test_set, range(1000 if args.quick else 10000))
    test_loader = DataLoader(test_set, batch_size=500, shuffle=False)
 
    if args.preset == "all":
        configs = PRESETS
    else:
        configs = [{"tag": args.tag, "partition": args.partition,
                    "alpha": args.alpha if args.partition == "dirichlet" else None,
                    "hetero": args.hetero, "compress": args.compress}]
 
    summaries = []
    for cfg in configs:
        summaries.append(run_experiment(cfg, args, train_set, test_loader, labels))
 
    summary_df = pd.DataFrame(summaries)
    out = RESULTS_DIR / "federated_update2_summary.csv"
    if args.preset != "all" and out.exists():  # append single runs to the existing summary
        old = pd.read_csv(out)
        summary_df = pd.concat([old[old["experiment"] != args.tag], summary_df], ignore_index=True)
    summary_df.to_csv(out, index=False)
    with open(RESULTS_DIR / "federated_update2_settings.json", "w") as f:
        json.dump({"args": vars(args), "tiers": CLIENT_TIERS, "tier_mix": TIER_MIX}, f, indent=2)
    print("\n=== Federated summary ===")
    print(summary_df.to_string(index=False))
 
 
if __name__ == "__main__":
    main()