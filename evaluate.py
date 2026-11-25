"""Evaluate every frozen checkpoint on the official GTSRB test set and verify paired runs."""

import csv
import hashlib
import io
import json
import subprocess
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

from experiment import ROOT, metrics, model, weights
from grid import GRID, SEEDS
from prepare import URL

OUT = ROOT / "results"


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def archive(name, checksum):
    dest = ROOT / "data" / name
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "3",
                URL.rsplit("/", 1)[0] + "/" + name,
                "-o",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(dest)
    with dest.open("rb") as f:
        assert hashlib.file_digest(f, "md5").hexdigest() == checksum
    return ZipFile(dest)


def test_data(device):
    with archive("GTSRB_Final_Test_GT.zip", "fe31e9c9270bbcd7b84b7f21a9d9d9e5") as gt:
        path = next(p for p in gt.namelist() if p.endswith("GT-final_test.csv"))
        with gt.open(path) as f:
            labels = list(csv.DictReader(io.TextIOWrapper(f), delimiter=";"))
    imgs = []
    with archive(
        "GTSRB_Final_Test_Images.zip", "c7e4e6327067d32654124b0fe9e82185"
    ) as z:
        for row in labels:
            with Image.open(
                io.BytesIO(z.read("GTSRB/Final_Test/Images/" + row["Filename"]))
            ) as im:
                box = tuple(
                    int(row[key]) + int(key.endswith("2"))
                    for key in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
                )
                imgs.append(
                    np.asarray(
                        im.convert("RGB")
                        .crop(box)
                        .resize((32, 32), Image.Resampling.BILINEAR)
                    )
                )
    y = torch.tensor([int(r["ClassId"]) for r in labels], device=device)
    assert len(imgs) == 12630 and len(set(y.cpu().tolist())) == 43
    x = (
        torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device, torch.float32)
        / 127.5
        - 1
    )
    return x, y


def evaluate():
    for seed in SEEDS:
        for s in (50, 80, 90):
            assert (GRID / f"seed{seed}" / f"comparison{s}.csv").exists()
    sensitivity = []
    for name in ("comparison", "rho_half", "rho_double", "rate_half", "rate_double"):
        folder = ROOT / "runs" / name / "seed411"
        cfg = json.loads((folder / "admm80.json").read_text())
        hist = read_csv(folder / "admm80.csv")
        assert (folder / "admm80.pt").exists()
        sensitivity.append(
            {
                "setting": name,
                "rho": cfg["rho"],
                "learning_rate": cfg["lr"],
                "validation_ce": hist[-1]["ce"],
                "validation_accuracy": hist[-1]["accuracy"],
                "nonzeros": hist[-1]["nonzeros"],
                "final_split_primal": hist[-2]["primal"],
                "final_split_dual": hist[-2]["dual"],
            }
        )
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "validation_sensitivity.csv", sensitivity)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x, y = test_data(device)
    rows, layer_rows = [], []
    for seed in SEEDS:
        folder = GRID / f"seed{seed}"
        dense_sha = hashlib.sha256((folder / "dense.pt").read_bytes()).hexdigest()
        for s, method in [(0, "dense")] + [
            (s, m) for s in (50, 80, 90) for m in ("magnitude", "admm")
        ]:
            stem = "dense" if not s else f"{method}{s}"
            cfg = json.loads((folder / f"{stem}.json").read_text())
            if s:
                assert cfg["dense_checkpoint_sha256"] == dense_sha
            net = model().to(device, memory_format=torch.channels_last)
            net.load_state_dict(
                torch.load(
                    folder / f"{stem}.pt", weights_only=True, map_location=device
                )
            )
            result = metrics(net, x, y)
            counts = [int(p.count_nonzero().item()) for p in weights(net)]
            actual = sum(counts)
            budget = 98528 * (100 - s) // 100
            assert actual <= budget
            hist = read_csv(folder / f"{stem}.csv")[-1]
            rows.append(
                dict(
                    seed=seed,
                    method=method,
                    sparsity=s,
                    **result,
                    nonzeros=actual,
                    cap=budget,
                    updates=int(hist["updates"]),
                    training_seconds=float(hist["training_seconds"]),
                )
            )
            for layer, p, n in zip(
                ("Conv 1", "Conv 2", "Conv 3", "Linear"), weights(net), counts
            ):
                layer_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "sparsity": s,
                        "layer": layer,
                        "retained": n,
                        "total": p.numel(),
                        "pct": 100 * n / p.numel(),
                    }
                )
    write_csv(OUT / "test_results.csv", rows)
    write_csv(OUT / "layer_retention.csv", layer_rows)
    pairs = []
    for seed in SEEDS:
        dense = next(r for r in rows if r["seed"] == seed and r["method"] == "dense")
        for s in (50, 80, 90):
            m, a = (
                next(
                    r
                    for r in rows
                    if r["seed"] == seed
                    and r["sparsity"] == s
                    and r["method"] == method
                )
                for method in ("magnitude", "admm")
            )
            assert a["updates"] == m["updates"]
            pairs.append(
                {
                    "seed": seed,
                    "sparsity": s,
                    "admm_minus_magnitude_pp": 100 * (a["accuracy"] - m["accuracy"]),
                    "admm_minus_magnitude_ce": a["ce"] - m["ce"],
                    "admm_minus_magnitude_macro_pp": 100
                    * (a["macro_recall"] - m["macro_recall"]),
                    "magnitude_accuracy_drop_pp": 100
                    * (dense["accuracy"] - m["accuracy"]),
                    "admm_accuracy_drop_pp": 100 * (dense["accuracy"] - a["accuracy"]),
                }
            )
    write_csv(OUT / "paired_differences.csv", pairs)
    print("Verified 21 checkpoints, 9 matched pairs and 5 sensitivity settings.")


if __name__ == "__main__":
    evaluate()
