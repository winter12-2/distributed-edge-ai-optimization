"""Compare compact channel-pruned models with the tuned CIFAR-10 CNN."""

import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
import hashlib
from io import BytesIO
from itertools import product
import json
from math import ceil
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

if __package__:
    from .baseline_cifar10 import (
        CIFAR10_WIDTHS, ROOT, VALIDATION_SPLIT_SEED, CIFAR10CNN, check_accuracy,
        cifar10_transform, tuned_split_indices,
    )
    from .tuned_experiment import TUNED_CHECKPOINT, load_tuned_model
else:
    from baseline_cifar10 import (
        CIFAR10_WIDTHS, ROOT, VALIDATION_SPLIT_SEED, CIFAR10CNN, check_accuracy,
        cifar10_transform, tuned_split_indices,
    )
    from tuned_experiment import TUNED_CHECKPOINT, load_tuned_model


# Indices of Conv2d modules within CIFAR10CNN.features. BN and ReLU follow each.
CONVOLUTIONS = (0, 4, 8, 11)
FEATURE_SIDES = (32, 16, 8, 8)
MIN_CHANNELS = 16
SENSITIVITY_RATES = (0.10, 0.20, 0.30, 0.50)
RECOVERY_EPOCHS = (2, 10, 20)
BATCH_SIZE = 64
INITIAL_LR = 5e-4
MIN_LR = 5e-5
WEIGHT_DECAY = 0.05


def model_cost(widths):
    inputs = (3, *widths[:-1])
    kernels = [9 * n_in * n_out for n_in, n_out in zip(inputs, widths)]
    head_weights = widths[-1] * 16 * 10
    parameters = sum(kernels) + 2 * sum(widths) + head_weights + 10
    macs = sum(k * side**2 for k, side in zip(kernels, FEATURE_SIDES)) + head_weights
    return {"parameters": parameters, "conv_linear_macs": macs}


def channel_ranking(model, method):
    ranking = []
    for index in CONVOLUTIONS:
        if method == "l1":
            score = model.features[index].weight.detach().abs().flatten(1).sum(dim=1)
        else:
            score = model.features[index + 1].weight.detach().abs()
        ranking.append(score.argsort(descending=True, stable=True).cpu())
    return ranking


def retained_channels(ranking, widths):
    return [order[:width].sort().values for order, width in zip(ranking, widths)]


def compact_uniform(model, amount):
    widths = tuple(max(MIN_CHANNELS, width - round(width * amount)) for width in CIFAR10_WIDTHS)
    retained = retained_channels(channel_ranking(model, "gamma"), widths)
    return compact_model(model, retained), widths


@torch.no_grad()
def compact_model(model, retained):
    """Copy the selected channels into an ordinary, smaller dense network."""
    device = model.classifier.weight.device
    compact = CIFAR10CNN(widths=tuple(len(keep) for keep in retained))
    compact.to(device=device, memory_format=torch.channels_last)
    previous = torch.arange(3, device=device)
    for index, keep in zip(CONVOLUTIONS, retained):
        keep = keep.to(device)
        source_conv = model.features[index]
        compact.features[index].weight.copy_(
            source_conv.weight.index_select(0, keep).index_select(1, previous)
        )
        source_bn = model.features[index + 1]
        target_bn = compact.features[index + 1]
        for field in ("weight", "bias", "running_mean", "running_var"):
            getattr(target_bn, field).copy_(getattr(source_bn, field).index_select(0, keep))
        target_bn.num_batches_tracked.copy_(source_bn.num_batches_tracked)
        previous = keep
    columns = (previous[:, None] * 16 + torch.arange(16, device=device)).flatten()
    compact.classifier.weight.copy_(model.classifier.weight.index_select(1, columns))
    compact.classifier.bias.copy_(model.classifier.bias)
    return compact.eval()


@contextmanager
def activation_masks(model, retained):
    """Mask after BatchNorm and ReLU so removed channels contribute zero."""
    hooks = []
    try:
        for index, keep in zip(CONVOLUTIONS, retained):
            conv = model.features[index]
            if keep.numel() == conv.out_channels:
                continue
            mask = conv.weight.new_zeros(conv.out_channels)
            mask[keep] = 1
            mask = mask.view(1, -1, 1, 1)
            hooks.append(model.features[index + 2].register_forward_hook(
                lambda _module, _inputs, output, mask=mask: output * mask
            ))
        yield
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def verify_compaction(model, compact, retained, examples):
    model.eval()
    with activation_masks(model, retained):
        expected = model(examples)
    torch.testing.assert_close(compact(examples), expected, atol=1e-4, rtol=1e-4)
    expected_count = model_cost(tuple(map(len, retained)))["parameters"]
    assert sum(parameter.numel() for parameter in compact.parameters()) == expected_count


def artifact_bytes(model, widths):
    state = model.state_dict()
    for key, tensor in state.items():
        state[key] = tensor.cpu()
    with BytesIO() as buffer:
        torch.save({"widths": list(widths), "format": "fp32", "state_dict": state}, buffer)
        return buffer.tell()


def choose_widths(model, validation_loader, device, targets, baseline_accuracy):
    rankings = {method: channel_ranking(model, method) for method in ("l1", "gamma")}
    sweep = []
    estimates = {}
    for method, ranking in rankings.items():
        layer_estimates = []
        for layer, full_width in enumerate(CIFAR10_WIDTHS):
            drops = {full_width: 0.0}
            for rate in SENSITIVITY_RATES:
                width = max(MIN_CHANNELS, full_width - round(full_width * rate))
                widths = list(CIFAR10_WIDTHS)
                widths[layer] = width
                trial = compact_model(model, retained_channels(ranking, widths))
                accuracy = check_accuracy(validation_loader, trial, device)
                drops[width] = max(0.0, baseline_accuracy - accuracy)
                sweep.append({
                    "score": method, "layer": layer + 1, "requested_channel_fraction": rate,
                    "retained_width": width, "validation_accuracy": accuracy,
                    **model_cost(widths),
                })
            layer_estimates.append(drops)
            print(f"Sensitivity {method}, conv {layer + 1}: {drops}", flush=True)
        estimates[method] = layer_estimates

    dense_parameters = model_cost(CIFAR10_WIDTHS)["parameters"]
    dense_bytes = artifact_bytes(model, CIFAR10_WIDTHS)
    selections = []
    for target in targets:
        shortlist = []
        for method, ranking in rankings.items():
            drops = estimates[method]
            candidates = []
            for widths in product(*drops):
                parameters = model_cost(widths)["parameters"]
                if parameters <= dense_parameters * (1 - target):
                    estimated_drop = sum(layer[width] for layer, width in zip(drops, widths))
                    candidates.append((estimated_drop, -parameters, widths))
            accepted = 0
            for _, _, widths in sorted(candidates):
                retained = retained_channels(ranking, widths)
                trial = compact_model(model, retained)
                trial_bytes = artifact_bytes(trial, widths)
                if trial_bytes > dense_bytes * (1 - target):
                    continue
                accuracy = check_accuracy(validation_loader, trial, device)
                shortlist.append({
                    "score": method, "widths": widths, "validation_accuracy": accuracy,
                    "estimated_artifact_bytes": trial_bytes, **model_cost(widths),
                })
                accepted += 1
                if accepted == 3:
                    break
        if not shortlist:
            raise ValueError(f"No candidate meets the parameter and artifact reduction target {target:.0%}.")
        winner = max(shortlist, key=lambda trial: (
            trial["validation_accuracy"], -trial["parameters"],
        ))
        selections.append({"target": target, "selected": winner, "shortlist": shortlist})
        print(f"Target {target:.0%}: {winner}", flush=True)
    return sweep, selections, rankings


def train_epoch(model, loader, optimizer, device, teacher=None):
    model.train()
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        if teacher is not None:
            temperature = 4.0
            with torch.no_grad():
                targets = (teacher(images) / temperature).softmax(dim=1)
            soft_loss = F.kl_div(
                (logits / temperature).log_softmax(dim=1), targets, reduction="batchmean",
            )
            loss = 0.5 * loss + 0.5 * temperature**2 * soft_loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().item()
    return loss_sum / len(loader)


def recover(model, train_loader, validation_loader, device, seed, epochs, name,
            *, masks_for_epoch=None, teacher=None):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=MIN_LR)
    snapshots = {}
    history = []
    for epoch in range(1, epochs + 1):
        retained = masks_for_epoch(epoch) if masks_for_epoch is not None else []
        with activation_masks(model, retained):
            # Reset the augmentation stream per epoch so all arms see the same batches.
            torch.manual_seed(seed + epoch)
            loss = train_epoch(model, train_loader, optimizer, device, teacher)
            accuracy = check_accuracy(validation_loader, model, device)
        history.append({
            "epoch": epoch, "train_loss": loss, "validation_accuracy": accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "widths": list(map(len, retained)) if retained else list(CIFAR10_WIDTHS),
        })
        if epoch in RECOVERY_EPOCHS or epoch == epochs:
            snapshots[epoch] = accuracy
        scheduler.step()
        print(f"{name}: epoch {epoch}/{epochs}, val {accuracy:.2f}%, loss {loss:.4f}", flush=True)
    return {
        "validation_accuracy": history[-1]["validation_accuracy"],
        "history": history, "recovery_accuracy": snapshots, "updates": epochs * len(train_loader),
    }


def staged_widths(widths, epoch, epochs):
    largest_fraction = max((full - width) / full for full, width in zip(CIFAR10_WIDTHS, widths))
    stages = min(epochs, max(2, ceil(largest_fraction / 0.10)))
    interval = 2 if 2 * stages - 1 <= epochs else 1
    stage = min(stages, 1 + (epoch - 1) // interval)
    return tuple(
        full - (full - final) * stage // stages
        for full, final in zip(CIFAR10_WIDTHS, widths)
    )


@torch.inference_mode()
def cpu_latency(model, example):
    model = model.cpu().eval()
    example = example[:1].cpu()
    for _ in range(20):
        model(example)
    samples = []
    for _ in range(100):
        start = time.perf_counter_ns()
        model(example)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[94],
        "threads": torch.get_num_threads(), "batch_size": 1,
    }


def load_artifact(path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = CIFAR10CNN(widths=payload["widths"]).to(memory_format=torch.channels_last).eval()
    if payload["format"] == "dynamic_int8":
        if torch.device(device).type != "cpu":
            raise ValueError("Dynamic INT8 checkpoints require CPU inference.")
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    state = payload["state_dict"]
    if payload["format"] == "csr":
        shapes = model.state_dict()
        for key, value in state.items():
            if value.layout == torch.sparse_csr:
                state[key] = value.to_dense().reshape(shapes[key].shape)
    model.load_state_dict(state)
    return model.to(device).eval()


def export(model, widths, path, example, *, encoding="fp32"):
    model = model.cpu().eval()
    state = model.state_dict()
    if encoding == "csr":
        # Encode only matrices where values and 32-bit indices beat dense storage.
        for key, tensor in list(state.items()):
            if tensor.ndim != 2 and tensor.ndim != 4:
                continue
            matrix = tensor.reshape(tensor.size(0), -1)
            entries = matrix.count_nonzero().item()
            if entries * 8 + (matrix.size(0) + 1) * 4 >= matrix.numel() * 4:
                continue
            sparse = matrix.to_sparse_csr()
            state[key] = torch.sparse_csr_tensor(
                sparse.crow_indices().int(), sparse.col_indices().int(), sparse.values(),
                size=sparse.shape, check_invariants=True,
            )
        # Convolution tensors are flattened in CSR and restored on load.
    with path.open("wb") as handle:
        torch.save({"widths": list(widths), "format": encoding, "state_dict": state}, handle)
    restored = load_artifact(path)
    with torch.inference_mode():
        torch.testing.assert_close(restored(example.cpu()), model(example.cpu()))
    return restored, {"file": path.name, "bytes": path.stat().st_size}


def make_loaders():
    train = CIFAR10(ROOT / "data", train=True, download=True, transform=cifar10_transform(train=True))
    validation = copy(train)
    validation.transform = cifar10_transform()
    train_ids, validation_ids = tuned_split_indices(len(train))
    test = CIFAR10(ROOT / "data", train=False, download=True, transform=cifar10_transform())
    return (
        DataLoader(Subset(train, train_ids), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(Subset(validation, validation_ids), batch_size=BATCH_SIZE),
        DataLoader(test, batch_size=BATCH_SIZE),
    )


def evaluate_int8(model, widths, path, examples, test_loader, fp32_accuracy):
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    quantized, artifact = export(quantized, widths, path, examples, encoding="dynamic_int8")
    accuracy = check_accuracy(test_loader, quantized, "cpu")
    return {
        "artifact": artifact, "test_accuracy": accuracy,
        "test_delta_vs_fp32": accuracy - fp32_accuracy, "latency": cpu_latency(quantized, examples),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def compare_seeds(paths):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(reports) < 2 or len({report["seed"] for report in reports}) != len(reports):
        raise ValueError("Use at least two reports with distinct seeds.")
    first = reports[0]
    settings = ("epochs", "validation_split_seed", "batch_size", "initial_lr", "min_lr", "weight_decay")
    if any(any(report[key] != first[key] for key in settings) for report in reports[1:]):
        raise ValueError("The reports must use a common split and training budget.")
    grouped = {}
    for report in reports:
        grouped_for_seed = {}
        for row in report["structured"]:
            plan = row["selected"]
            key = (row["target"], tuple(plan["widths"]), plan["score"], row["winner"])
            winner = row["variants"][row["winner"]]
            grouped_for_seed[key] = {
                "dense_test_accuracy": report["dense_control"]["test_accuracy"],
                "compact_test_accuracy": row["test_accuracy"],
                "test_delta_vs_dense": row["test_delta_vs_dense"],
                "compact_bytes": winner["artifact"]["bytes"],
                "compact_parameters": plan["parameters"],
                "median_cpu_ms": row["latency"]["median_ms"],
                "p95_cpu_ms": row["latency"]["p95_ms"],
                "int8_test_accuracy": row["dynamic_int8"]["test_accuracy"],
            }
        if not grouped_for_seed:
            raise ValueError("Each report must contain a completed structured experiment.")
        for key, row in grouped_for_seed.items():
            grouped.setdefault(key, []).append(row)
    summary = []
    for (target, widths, score, variant), measurements in sorted(grouped.items()):
        if len(measurements) != len(reports):
            continue
        metrics = {}
        for metric in measurements[0]:
            values = [row[metric] for row in measurements]
            metrics[metric] = {"mean": statistics.mean(values), "stdev": statistics.stdev(values)}
        summary.append({
            "target": target, "widths": widths, "score": score, "variant": variant,
            "metrics": metrics,
        })
    if not summary:
        raise ValueError("The reports need a common target, compact widths, score, and winning recovery method.")
    return {
        "seeds": [report["seed"] for report in reports],
        "distinct_source_checkpoints": len({report["source_checkpoint_sha256"] for report in reports}),
        "results": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=TUNED_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20, help="Extra training epochs per experiment.")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--sensitivity-only", action="store_true")
    parser.add_argument("--selection", type=Path, help="Reuse selected widths and scoring from sensitivity.json.")
    parser.add_argument("--only", choices=["one_shot", "gradual"], help="Train a fixed recovery method.")
    parser.add_argument("--run-tag", help="Append a label to the output directory.")
    parser.add_argument("--summarize", type=Path, nargs="+", metavar="REPORT",
                        help="Print across-seed means and sample standard deviations for matching finalists.")
    parser.add_argument("--unstructured", type=float, nargs="+", default=[],
                        help="Optional per-layer weight sparsity targets such as 0.30 0.60.")
    parser.add_argument("--unstructured-only", action="store_true",
                        help="Run the dense control and the requested unstructured arms.")
    parser.add_argument("--distill", action="store_true", help="Add one-shot distillation at the first target.")
    args = parser.parse_args()
    if args.summarize:
        try:
            print(json.dumps(compare_seeds(args.summarize), indent=2))
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        return
    if args.epochs < 2:
        parser.error("--epochs must be at least 2 to compare recovery schedules")
    if any(not 0 < target < 1 for target in args.targets + args.unstructured):
        parser.error("Targets must be fractions strictly between 0 and 1")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.only and args.distill:
        parser.error("--distill adds an arm and cannot be used with --only")
    if args.unstructured_only and (not args.unstructured or args.sensitivity_only or args.selection
                                   or args.only or args.distill):
        parser.error("--unstructured-only needs --unstructured and excludes structured experiment options")
    if args.run_tag and not args.run_tag.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-tag can use letters, numbers, underscores, and hyphens")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.checkpoint.stem}_s{args.seed}_e{args.epochs}"
    if args.run_tag:
        run_name += f"_{args.run_tag}"
    output = ROOT / "results" / "pruning" / run_name
    artifacts = ROOT / "models" / "pruning" / run_name
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    model = load_tuned_model(args.checkpoint, device=device)
    train_loader, validation_loader, test_loader = make_loaders()
    examples = next(iter(validation_loader))[0][:8]
    baseline_accuracy = check_accuracy(validation_loader, model, device)
    print(f"Dense checkpoint validation: {baseline_accuracy:.2f}%", flush=True)
    if args.unstructured_only:
        sweep, selections, rankings = [], [], {}
    elif args.selection:
        prior = json.loads(args.selection.read_text(encoding="utf-8"))
        available = {row["target"]: row for row in prior["targets"]}
        missing = set(args.targets) - available.keys()
        if missing:
            parser.error(f"The selection file needs target(s): {sorted(missing)}")
        sweep = []
        selections = []
        rankings = {method: channel_ranking(model, method) for method in ("l1", "gamma")}
        for target in args.targets:
            plan = available[target]["selected"].copy()
            widths = plan["widths"]
            if (plan["score"] not in rankings or len(widths) != len(CIFAR10_WIDTHS)
                    or any(not MIN_CHANNELS <= w <= full for w, full in zip(widths, CIFAR10_WIDTHS))):
                parser.error(f"Invalid widths or score in the selection for {target:.0%}")
            compact = compact_model(model, retained_channels(rankings[plan["score"]], widths))
            plan.update(model_cost(widths))
            plan["estimated_artifact_bytes"] = artifact_bytes(compact, widths)
            if (plan["estimated_artifact_bytes"] > artifact_bytes(model, CIFAR10_WIDTHS) * (1 - target)
                    or plan["parameters"] > model_cost(CIFAR10_WIDTHS)["parameters"] * (1 - target)):
                parser.error(f"The selected widths miss the {target:.0%} parameter or artifact reduction budget.")
            plan["validation_accuracy"] = check_accuracy(validation_loader, compact, device)
            selections.append({"target": target, "selected": plan, "shortlist": []})
    else:
        sweep, selections, rankings = choose_widths(
            model, validation_loader, device, args.targets, baseline_accuracy,
        )
    write_json(output / "sensitivity.json", {
        "baseline_validation_accuracy": baseline_accuracy, "measurements": sweep, "targets": selections,
    })
    if args.sensitivity_only:
        return

    source, source_file = export(deepcopy(model), CIFAR10_WIDTHS, artifacts / "source.pt", examples)
    report = {
        "source_checkpoint": args.checkpoint.name, "source_checkpoint_bytes": args.checkpoint.stat().st_size,
        "source_checkpoint_sha256": checkpoint_hash,
        "seed": args.seed, "validation_split_seed": VALIDATION_SPLIT_SEED,
        "epochs": args.epochs, "training_device": str(device), "batch_size": BATCH_SIZE,
        "optimizer": "AdamW", "initial_lr": INITIAL_LR, "min_lr": MIN_LR,
        "weight_decay": WEIGHT_DECAY, "scheduler": "cosine per epoch", "label_smoothing": 0.1,
        "baseline": {
            "validation_accuracy": baseline_accuracy, "test_accuracy": check_accuracy(test_loader, source, "cpu"),
            "artifact": source_file, "latency": cpu_latency(source, examples), **model_cost(CIFAR10_WIDTHS),
        },
        "structured": [], "unstructured": [],
    }
    report["quantization_only"] = evaluate_int8(
        source, CIFAR10_WIDTHS, artifacts / "source_int8.pt", examples, test_loader,
        report["baseline"]["test_accuracy"],
    )
    dense = deepcopy(model)
    dense_result = recover(dense, train_loader, validation_loader, device, args.seed, args.epochs, "dense")
    dense, dense_file = export(dense, CIFAR10_WIDTHS, artifacts / "dense.pt", examples)
    report["dense_control"] = {
        **dense_result, "artifact": dense_file, "latency": cpu_latency(dense, examples),
        "test_accuracy": check_accuracy(test_loader, dense, "cpu"), **model_cost(CIFAR10_WIDTHS),
    }
    write_json(output / "report.json", report)

    for selection_index, selection in enumerate(selections):
        target = selection["target"]
        selected = selection["selected"]
        widths = selected["widths"]
        ranking = rankings[selected["score"]]
        final_channels = retained_channels(ranking, widths)
        verified = compact_model(model, final_channels)
        verify_compaction(model, verified, final_channels, examples.to(device))
        variants = [(name, None) for name in ("one_shot", "gradual") if args.only in (None, name)]
        if args.distill and selection_index == 0:
            variants.append(("distilled", model.eval()))
        results = {}
        tag = f"target{target:.4g}"
        for name, teacher in variants:
            def masks_for_epoch(epoch):
                stage_widths = staged_widths(widths, epoch, args.epochs) if name == "gradual" else widths
                return retained_channels(ranking, stage_widths)

            working = deepcopy(model)
            with activation_masks(working, masks_for_epoch(1)):
                immediate = check_accuracy(validation_loader, working, device)
            recovery = recover(
                working, train_loader, validation_loader, device, args.seed, args.epochs,
                f"{tag} {name}", masks_for_epoch=masks_for_epoch, teacher=teacher,
            )
            compact = compact_model(working, final_channels)
            verify_compaction(working, compact, final_channels, examples.to(device))
            _, file = export(compact, widths, artifacts / f"{tag}_{name}.pt", examples)
            results[name] = {
                **recovery, "immediate_validation_accuracy": immediate,
                "artifact": file, "artifact_reduction": 1 - file["bytes"] / dense_file["bytes"],
                "validation_delta_vs_dense": recovery["validation_accuracy"] - dense_result["validation_accuracy"],
            }
        winner = max(results, key=lambda name: results[name]["validation_accuracy"])
        best = load_artifact(artifacts / results[winner]["artifact"]["file"])
        best_test = check_accuracy(test_loader, best, "cpu")
        report["structured"].append({
            **selection, "variants": results, "winner": winner,
            "test_accuracy": best_test, "test_delta_vs_dense": best_test - report["dense_control"]["test_accuracy"],
            "latency": cpu_latency(best, examples),
            "dynamic_int8": evaluate_int8(
                best, widths, artifacts / f"{tag}_int8.pt", examples, test_loader, best_test,
            ),
        })
        write_json(output / "report.json", report)

    for amount in args.unstructured:
        working = deepcopy(model)
        weights = [module for module in working.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]
        for module in weights:
            scores = module.weight.detach().contiguous()
            prune.l1_unstructured(module, name="weight", amount=amount, importance_scores=scores)
        immediate = check_accuracy(validation_loader, working, device)
        recovery = recover(
            working, train_loader, validation_loader, device, args.seed, args.epochs, f"unstructured {amount:.0%}",
        )
        for module in weights:
            prune.remove(module, "weight")
        zero_weights = sum(module.weight.numel() - module.weight.count_nonzero().item() for module in weights)
        weight_count = sum(module.weight.numel() for module in weights)
        tag = f"unstructured{amount:.4g}"
        restored, dense_export = export(working, CIFAR10_WIDTHS, artifacts / f"{tag}.pt", examples)
        _, csr_export = export(working, CIFAR10_WIDTHS, artifacts / f"{tag}_csr.pt", examples, encoding="csr")
        test_accuracy = check_accuracy(test_loader, restored, "cpu")
        report["unstructured"].append({
            **recovery, "amount": amount, "weight_sparsity": zero_weights / weight_count,
            "immediate_validation_accuracy": immediate, "dense_artifact": dense_export,
            "storage_only_sparse_artifact": csr_export, **model_cost(CIFAR10_WIDTHS),
            "test_accuracy": test_accuracy,
            "latency_after_dense_reload": cpu_latency(restored, examples),
            "dynamic_int8": evaluate_int8(
                restored, CIFAR10_WIDTHS, artifacts / f"{tag}_int8.pt", examples, test_loader, test_accuracy,
            ),
        })
        write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
