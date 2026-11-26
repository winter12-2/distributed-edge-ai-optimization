import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import baseline_cifar10 as base
from federated_noniid import (
    activation_bytes_per_image,
    evaluate,
    fedavg,
    get_datasets,
    label_counts,
    partition_dirichlet,
    partition_iid,
    set_seed,
    state_bytes,
    training_memory_mb,
)
from tuned_experiment import TUNED_CHECKPOINT, load_tuned_model

ROOT = base.ROOT
RESULTS_DIR = ROOT / "results"


CLIENT_TIERS = {
    "high": {"ram_mb": 256.0, "threads": 4, "cpu_speed": 1.00, "bandwidth_mbps": 100.0, "dropout": 0.02},
    "mid":  {"ram_mb": 64.0,  "threads": 2, "cpu_speed": 0.60, "bandwidth_mbps": 50.0,  "dropout": 0.05},
    "low":  {"ram_mb": 16.0,  "threads": 1, "cpu_speed": 0.35, "bandwidth_mbps": 20.0,  "dropout": 0.10},
    "tiny": {"ram_mb": 8.0,   "threads": 1, "cpu_speed": 0.20, "bandwidth_mbps": 5.0,   "dropout": 0.15},
}
TIER_MIX = ["high", "high", "mid", "mid", "mid", "low", "low", "low", "tiny", "tiny"]

PRESETS = [
    {"tag": "u3_iid_ideal",               "partition": "iid",       "hetero": False, "mu": 0.0,  "selection": False},
    {"tag": "u3_noniid_ideal",            "partition": "dirichlet", "hetero": False, "mu": 0.0,  "selection": False},
    {"tag": "u3_noniid_hetero_fedavg",    "partition": "dirichlet", "hetero": True,  "mu": 0.0,  "selection": False},
    {"tag": "u3_noniid_hetero_fedprox",   "partition": "dirichlet", "hetero": True,  "mu": None, "selection": False},
    {"tag": "u3_noniid_hetero_selection", "partition": "dirichlet", "hetero": True,  "mu": 0.0,  "selection": True},
    {"tag": "u3_noniid_hetero_both",      "partition": "dirichlet", "hetero": True,  "mu": None, "selection": True},
]


def build_model(name, checkpoint=None):
    if name == "improved":
        return load_tuned_model(Path(checkpoint) if checkpoint else TUNED_CHECKPOINT)
    return base.CNN()


def local_train(global_model, dataset, indices, batch_size, epochs, lr, threads, mu):
    torch.set_num_threads(threads)
    model = copy.deepcopy(global_model)
    model.train()
    global_params = [p.detach().clone() for p in global_model.parameters()]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    for _ in range(epochs):
        for x, y in loader:
            if x.size(0) < 2 and any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
                continue  # BatchNorm cannot train on a batch of 1
            opt.zero_grad()
            loss = criterion(model(x), y)
            if mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (mu / 2) * prox
            loss.backward()
            opt.step()
    return model.state_dict(), time.perf_counter() - start


def run_experiment(cfg, args, train_set, test_loader, labels):
    tag = cfg["tag"]
    mu = args.mu if cfg["mu"] is None else cfg["mu"]
    print(f"\n=== Experiment: {tag}  (model={args.model}, FedProx mu={mu}, selection={cfg['selection']}) ===")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if cfg["partition"] == "iid":
        parts = partition_iid(labels, args.clients, rng)
    else:
        parts = partition_dirichlet(labels, args.clients, args.alpha, rng)
    counts = label_counts(labels, parts)
    pd.DataFrame(counts, columns=[f"class_{c}" for c in range(10)]).assign(
        client=range(args.clients)).to_csv(RESULTS_DIR / f"federated_{tag}_label_counts.csv", index=False)

    max_threads = max(1, os.cpu_count() or 1)
    tiers = [TIER_MIX[i % len(TIER_MIX)] for i in range(args.clients)] if cfg["hetero"] else ["ideal"] * args.clients
    profiles = []
    for t in tiers:
        if t == "ideal":
            profiles.append({"ram_mb": float("inf"), "threads": max_threads, "cpu_speed": 1.0,
                             "bandwidth_mbps": float("inf"), "dropout": 0.0})
        else:
            p = dict(CLIENT_TIERS[t])
            p["threads"] = min(p["threads"], max_threads)
            profiles.append(p)

    global_model = build_model(args.model, args.checkpoint)
    _, initial_accuracy = evaluate(global_model, test_loader)
    print(f"  Initial model accuracy: {initial_accuracy:.4f}")
    param_bytes = sum(p.numel() * 4 for p in global_model.parameters())
    act_bytes = activation_bytes_per_image(global_model)
    download_bytes = state_bytes(global_model.state_dict(), "none")
    upload_bytes = state_bytes(global_model.state_dict(), args.compress)

    def comm_seconds(c):
        bw = profiles[c]["bandwidth_mbps"]
        return 0.0 if bw == float("inf") else (upload_bytes + download_bytes) * 8 / (bw * 1e6)

    sec_per_sample = {}  # learned compute time per sample for each client
    round_rows, client_rows = [], []
    cumulative_mb = 0.0
    for rnd in range(1, args.rounds + 1):
        global_state = copy.deepcopy(global_model.state_dict())
        work = {c: 1.0 for c in range(args.clients)}
        if cfg["selection"] and cfg["hetero"] and sec_per_sample:
            pred_full = {c: sec_per_sample[c] * len(parts[c]) + comm_seconds(c) for c in sec_per_sample}
            planned_deadline = args.deadline_factor * min(pred_full.values()) * args.safety
            for c in sec_per_sample:
                budget = planned_deadline - comm_seconds(c)
                work[c] = float(np.clip(budget / (sec_per_sample[c] * len(parts[c])), 0.0, 1.0))

        results = []
        for c in range(args.clients):
            prof = profiles[c]
            row = {"round": rnd, "client": c, "tier": tiers[c], "samples": len(parts[c]),
                   "work_fraction": round(work[c], 3)}

            bs = args.batch_size
            mem = training_memory_mb(param_bytes, act_bytes, bs)
            while mem > prof["ram_mb"] and bs > 2:
                bs //= 2
                mem = training_memory_mb(param_bytes, act_bytes, bs)
            row.update({"batch_size": bs, "est_train_mem_mb": round(mem, 2), "ram_mb": prof["ram_mb"]})

            if mem > prof["ram_mb"]:
                row["status"] = "dropped_ram"
                client_rows.append(row)
                continue
            if work[c] < args.min_work:
                row["status"] = "skipped_selection"  
                client_rows.append(row)
                continue
            if rng.random() < prof["dropout"]:
                row["status"] = "dropped_offline"
                client_rows.append(row)
                continue

            n_use = max(2, int(round(work[c] * len(parts[c]))))
            idx = parts[c] if n_use >= len(parts[c]) else rng.choice(parts[c], n_use, replace=False).tolist()
            state, wall = local_train(global_model, train_set, idx, bs, args.local_epochs, args.lr,
                                      prof["threads"], mu)
            compute_s = wall / prof["cpu_speed"]
            sec_per_sample[c] = compute_s / len(idx)
            sim_time = compute_s + comm_seconds(c)
            row.update({"status": "ok", "samples_used": len(idx), "sim_time_s": round(sim_time, 3),
                        "compute_s": round(compute_s, 3), "comm_s": round(comm_seconds(c), 3)})
            client_rows.append(row)
            results.append((c, state, len(idx), sim_time))
        deadline = None
        used = results
        if results and cfg["hetero"] and args.deadline_factor > 0:
            deadline = args.deadline_factor * min(r[3] for r in results)
            used = [r for r in results if r[3] <= deadline]
            kept = {r[0] for r in used}
            for cr in client_rows:
                if cr["round"] == rnd and cr["status"] == "ok" and cr["client"] not in kept:
                    cr["status"] = "dropped_deadline"

        if used:
            global_model.load_state_dict(fedavg(global_state, [r[1] for r in used],
                                                [r[2] for r in used], args.compress))

        torch.set_num_threads(max_threads)
        test_loss, test_acc = evaluate(global_model, test_loader)
        sent_model = sum(1 for cr in client_rows if cr["round"] == rnd
                         and cr["status"] not in ("skipped_selection", "dropped_ram"))
        up_mb = len(results) * upload_bytes / 1024 ** 2
        down_mb = sent_model * download_bytes / 1024 ** 2
        cumulative_mb += up_mb + down_mb
        if deadline is not None and len(used) < len(results):
            round_time = deadline
        else:
            round_time = max((r[3] for r in used), default=0.0)
        round_rows.append({
            "experiment": tag, "round": rnd, "test_accuracy": round(test_acc, 4),
            "test_loss": round(test_loss, 4), "round_time_s": round(round_time, 2),
            "clients_completed": len(used), "clients_total": args.clients,
            "completion_rate": round(len(used) / args.clients, 3),
            "upload_mb": round(up_mb, 4), "download_mb": round(down_mb, 4),
            "cumulative_comm_mb": round(cumulative_mb, 4),
        })
        print(f"  round {rnd:>2}/{args.rounds}  acc={test_acc:.4f}  completed={len(used)}/{args.clients}"
              f"  round_time={round_time:.1f}s  comm={cumulative_mb:.1f}MB")

    rounds_df = pd.DataFrame(round_rows)
    clients_df = pd.DataFrame(client_rows)
    rounds_df.to_csv(RESULTS_DIR / f"federated_{tag}_rounds.csv", index=False)
    clients_df.to_csv(RESULTS_DIR / f"federated_{tag}_clients.csv", index=False)
    ok = clients_df[clients_df["status"] == "ok"]
    status = clients_df["status"]
    return {
        "experiment": tag,
        "model": args.model,
        "partition": "iid" if cfg["partition"] == "iid" else f"dirichlet(alpha={args.alpha})",
        "heterogeneous": cfg["hetero"],
        "fedprox_mu": mu,
        "client_selection": cfg["selection"],
        "initial_test_accuracy": initial_accuracy,
        "final_accuracy": rounds_df["test_accuracy"].iloc[-1],
        "best_accuracy": rounds_df["test_accuracy"].max(),
        "mean_last3_accuracy": round(rounds_df["test_accuracy"].tail(3).mean(), 4),
        "mean_round_time_s": round(rounds_df["round_time_s"].mean(), 2),
        "mean_completion_rate": round(rounds_df["completion_rate"].mean(), 3),
        "total_comm_mb": round(cumulative_mb, 2),
        "mean_work_fraction": round(ok["work_fraction"].mean(), 3) if len(ok) else None,
        "dropped_ram": int((status == "dropped_ram").sum()),
        "dropped_offline": int((status == "dropped_offline").sum()),
        "dropped_deadline": int((status == "dropped_deadline").sum()),
        "skipped_selection": int((status == "skipped_selection").sum()),
        "max_est_train_mem_mb": round(clients_df["est_train_mem_mb"].max(), 2),
    }
def parse_args():
    p = argparse.ArgumentParser(description="Update 3: FedProx and hardware-aware client selection")
    p.add_argument("--model", choices=["improved", "original"], default="improved")
    p.add_argument("--checkpoint", help="Override the tuned checkpoint for the improved model.")
    p.add_argument("--output-dir", help="Directory for this run's metrics.")
    p.add_argument("--only", nargs="+", help="run only these experiment tags")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--mu", type=float, default=0.01, help="FedProx strength used by the FedProx runs")
    p.add_argument("--compress", choices=["none", "int8"], default="none")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.003, help="small because we fine-tune the tuned 88% model")
    p.add_argument("--deadline-factor", type=float, default=4.0)
    p.add_argument("--safety", type=float, default=0.8, help="selection plans for 80%% of the deadline")
    p.add_argument("--min-work", type=float, default=0.1, help="skip clients that can do less than this")
    p.add_argument("--data-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="tiny run to check the code works")
    args = p.parse_args()
    if args.model == "original" and args.checkpoint:
        p.error("--checkpoint is for the improved model")
    return args


def main():
    global RESULTS_DIR
    args = parse_args()
    if args.model == "improved" and args.checkpoint is None:
        args.checkpoint = str(TUNED_CHECKPOINT)
    RESULTS_DIR = Path(args.output_dir).resolve() if args.output_dir else ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_size = 10000
    if args.quick:
        args.rounds, args.data_fraction, test_size = 2, 0.04, 1000

    train_set, test_set = get_datasets(args.model)
    labels = np.array(train_set.targets)
    if args.data_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(labels), int(len(labels) * args.data_fraction), replace=False))
        train_set = Subset(train_set, keep.tolist())
        labels = labels[keep]
    test_loader = DataLoader(Subset(test_set, range(test_size)), batch_size=500, shuffle=False)

    configs = [c for c in PRESETS if not args.only or c["tag"] in args.only]
    summaries = [run_experiment(cfg, args, train_set, test_loader, labels) for cfg in configs]

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS_DIR / "federated_update3_summary.csv", index=False)
    with open(RESULTS_DIR / "federated_update3_settings.json", "w") as f:
        json.dump({"args": vars(args), "tiers": CLIENT_TIERS, "tier_mix": TIER_MIX}, f, indent=2)
    print("\n=== Update 3 summary ===")
    cols = ["experiment", "final_accuracy", "mean_last3_accuracy", "mean_round_time_s",
            "mean_completion_rate", "total_comm_mb", "skipped_selection", "dropped_deadline"]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
