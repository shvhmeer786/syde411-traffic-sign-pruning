"""Build the validation comparison figure and table from recorded checkpoints and logs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiment import model, weights


def summarize(folder, sparsity):
    names = ["dense", f"magnitude{sparsity}", f"admm{sparsity}"]
    labels = ["Dense", "Magnitude", "ADMM"]
    colors = ["#42546b", "#bf541b", "#156d74"]
    records, fractions, training = [], [], []
    for name, label in zip(names, labels):
        with (folder / f"{name}.csv").open() as f:
            series = list(csv.DictReader(f))
        training.append(series)
        net = model()
        net.load_state_dict(torch.load(folder / f"{name}.pt", weights_only=True))
        retained = [p.count_nonzero().item() for p in weights(net)]
        fractions.append([100 * n / p.numel() for n, p in zip(retained, weights(net))])
        last = series[-1]
        records.append(
            {
                "method": label,
                "validation_accuracy": float(last["accuracy"]),
                "macro_recall": float(last["macro_recall"]),
                "cross_entropy": float(last["ce"]),
                "updates": int(last["updates"]),
                "nonzeros": sum(retained),
                "training_seconds": float(last["training_seconds"]),
            }
        )
    configs = [json.loads((folder / f"{n}.json").read_text()) for n in names]
    assert len({c["manifest_sha256"] for c in configs}) == 1
    assert (
        configs[1]["dense_checkpoint_sha256"] == configs[2]["dense_checkpoint_sha256"]
    )
    assert records[1]["updates"] == records[2]["updates"]
    assert all(r["nonzeros"] <= 98528 * (100 - sparsity) // 100 for r in records[1:])
    with (folder / f"comparison{sparsity}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    fig, (ax, layers) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for i in (1, 2):
        series = training[i]
        ax.plot(
            [int(r["updates"]) for r in series],
            [float(r["ce"]) for r in series],
            marker="o",
            label=labels[i],
            color=colors[i],
        )
        offsets = [j + (i - 1.5) * 0.36 for j in range(4)]
        layers.bar(offsets, fractions[i], width=0.34, color=colors[i], label=labels[i])
    ax.axhline(
        records[0]["cross_entropy"],
        linestyle=":",
        color=colors[0],
        label="Dense starting model",
    )
    ax.set(
        xlabel="Gradient updates after dense training",
        ylabel="Validation cross-entropy",
        title="Loss through pruning and retraining",
    )
    ax.legend(frameon=False, fontsize=8)
    layers.set(
        xticks=range(4),
        xticklabels=["Conv 1", "Conv 2", "Conv 3", "Linear"],
        ylabel="Layer weights retained (%)",
        title="Where the final models kept weights",
        ylim=(0, 105),
    )
    layers.legend(frameon=False, fontsize=8)
    for a in (ax, layers):
        a.spines[["right", "top"]].set_visible(False)
    fig.suptitle(
        f"GTSRB validation | {sparsity}% sparsity | {folder.name}", fontsize=13
    )
    fig.savefig(folder / f"comparison{sparsity}.png", dpi=180)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--sparsity", type=int, default=80)
    args = p.parse_args()
    summarize(args.folder, args.sparsity)
