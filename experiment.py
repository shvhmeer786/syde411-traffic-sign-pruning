"""Dense training and matched-budget pruning on the frozen GTSRB training/validation split."""

import argparse
import csv
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from project_paths import WORKSPACE

BATCH = 128
LAMBDA = 1e-5
ORDER_SEED = 2901


def model():
    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 43),
    )
    assert sum(p.numel() for p in weights(net)) == 98528
    assert sum(p.numel() for p in net.parameters() if p.ndim == 1) == 267
    return net


def weights(net):
    return [p for p in net.parameters() if p.ndim > 1]


def vector(params):
    return torch.cat([p.reshape(-1) for p in params])


def support(v, k):
    # Stable CPU sort breaks a tie by original flattened coordinate.
    arr = v.detach().cpu().numpy()
    indices = np.argsort(-np.abs(arr), kind="stable")[:k]
    mask = torch.zeros_like(v, dtype=torch.bool)
    mask[torch.from_numpy(indices.copy()).to(v.device)] = True
    return mask


def mask_weights(params, mask):
    for p, part in zip(params, mask.split([p.numel() for p in params])):
        p.mul_(part.reshape_as(p))


@torch.no_grad()
def metrics(net, x, y):
    net.eval()
    preds, loss = [], 0.0
    for xb, yb in zip(x.split(256), y.split(256)):
        logits = net(xb)
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        preds.append(logits.argmax(1).cpu().numpy())
    actual = y.cpu().numpy()
    correct = np.concatenate(preds) == actual
    recall = np.bincount(actual[correct], minlength=43) / np.bincount(
        actual, minlength=43
    )
    return {
        "ce": loss / len(y),
        "accuracy": float(correct.mean()),
        "macro_recall": float(recall.mean()),
    }


def train(args):
    target = WORKSPACE / "runs" / args.tag / f"seed{args.seed}"
    target.mkdir(parents=True, exist_ok=True)
    name = "dense" if args.method == "dense" else f"{args.method}{args.sparsity}"
    output = target / (name + ".pt")
    if output.exists():
        raise FileExistsError(f"{output} already exists. Use a fresh run tag.")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    net = model().to(device, memory_format=torch.channels_last)
    params = weights(net)
    with np.load(WORKSPACE / "data/prepared.npz") as data:
        x = (
            torch.from_numpy(data["images"])
            .permute(0, 3, 1, 2)
            .to(device, torch.float32)
            / 127.5
            - 1
        )
        y = torch.from_numpy(data["labels"]).to(device)
        train_ids, val_ids = [
            torch.from_numpy(np.flatnonzero(mask)).to(device)
            for mask in (~data["validation"], data["validation"])
        ]
    xv, yv = x[val_ids], y[val_ids]
    dense_file = target / "dense.pt"
    reference = None
    if args.method != "dense":
        net.load_state_dict(
            torch.load(dense_file, map_location=device, weights_only=True)
        )
        reference = hashlib.sha256(dense_file.read_bytes()).hexdigest()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    valid_hash = hashlib.sha256(
        (WORKSPACE / "data/manifest.csv").read_bytes()
    ).hexdigest()
    config = dict(
        **vars(args),
        batch=BATCH,
        weight_decay_lambda=LAMBDA,
        data_order_seed=ORDER_SEED,
        torch_version=str(torch.__version__),
        device=str(device),
        manifest_sha256=valid_hash,
        dense_checkpoint_sha256=reference,
    )
    (target / (name + ".json")).write_text(json.dumps(config, indent=2))
    k = 98528 * (100 - args.sparsity) // 100
    if device.type == "mps":
        torch.mps.synchronize()
    setup_start = time.perf_counter()
    v = vector(params).detach()
    mask = None
    z, u = v.clone(), torch.zeros_like(v)
    if args.method == "magnitude":
        mask = support(v, k)
        with torch.no_grad():
            mask_weights(params, mask)
    generator = torch.Generator().manual_seed(ORDER_SEED)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - setup_start
    total_steps = 0
    records = []
    previous_primal = previous_dual = None
    for epoch in range(args.epochs + 1):
        phase = "dense" if args.method == "dense" else "masked"
        average_train = None
        if epoch:
            admm = args.method == "admm" and epoch <= args.epochs - 1
            phase = "admm" if admm else phase
            net.train()
            running_loss = torch.zeros((), device=device)
            if device.type == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            ids = train_ids[
                torch.randperm(len(train_ids), generator=generator).to(device)
            ]
            for batch_ids in ids.split(BATCH):
                optimizer.zero_grad(set_to_none=True)
                logits = net(x[batch_ids].contiguous(memory_format=torch.channels_last))
                v = vector(params)
                objective = (
                    F.cross_entropy(logits, y[batch_ids])
                    + LAMBDA / 2 * v.square().sum()
                )
                loss = objective
                if admm:
                    loss = loss + args.rho / 2 * (v - z + u).square().sum()
                loss.backward()
                optimizer.step()
                if mask is not None:
                    with torch.no_grad():
                        mask_weights(params, mask)
                running_loss += objective.detach() * len(batch_ids)
                total_steps += 1
            if admm:
                v = vector(params).detach()
                new_mask = support(v + u, k)
                new_z = (v + u) * new_mask
                previous_primal = torch.linalg.vector_norm(v - new_z).item()
                previous_dual = args.rho * torch.linalg.vector_norm(new_z - z).item()
                u = u + v - new_z
                z = new_z
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed += time.perf_counter() - start
            average_train = running_loss.item() / len(train_ids)
        validation = metrics(net, xv, yv)
        row = dict(
            epoch=epoch,
            phase=phase if epoch else "initial",
            updates=total_steps,
            training_seconds=elapsed,
            online_train_F=average_train,
            **validation,
            nonzeros=int(vector(params).detach().count_nonzero().item()),
            primal=previous_primal if epoch and phase == "admm" else None,
            dual=previous_dual if epoch and phase == "admm" else None,
        )
        records.append(row)
        print(
            name,
            epoch,
            total_steps,
            round(validation["accuracy"], 4),
            round(elapsed, 1),
            flush=True,
        )
        if args.method == "admm" and epoch == args.epochs - 1:
            # Project the learned weights to the final ADMM support before fixed-mask retraining.
            start = time.perf_counter()
            mask = new_mask
            with torch.no_grad():
                mask_weights(params, mask)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed += time.perf_counter() - start
    with (target / (name + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    tmp = output.with_suffix(".tmp")
    torch.save(net.cpu().state_dict(), tmp)
    check = model()
    check.load_state_dict(torch.load(tmp, weights_only=True))
    retained = [int(p.count_nonzero()) for p in weights(check)]
    assert args.method == "dense" or sum(retained) <= k
    tmp.replace(output)
    print("Verified reloaded checkpoint", retained, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["dense", "magnitude", "admm"])
    p.add_argument("--seed", type=int, default=411)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--sparsity", type=int, choices=[50, 80, 90], default=80)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--rho", type=float, default=0.001)
    p.add_argument("--tag", default="pilot")
    args = p.parse_args()
    if args.epochs < (2 if args.method == "admm" else 1):
        p.error("ADMM needs at least one split epoch and one masked epoch.")
    train(args)
