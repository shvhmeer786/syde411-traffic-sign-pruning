"""Run the predeclared validation grid, resuming only completed checkpoints."""

import shutil
import subprocess
import sys

from project_paths import SOURCE, WORKSPACE

GRID = WORKSPACE / "runs" / "comparison"
SEEDS = (411, 412, 413)


def run(method, seed, sparsity=80):
    stem = method if method == "dense" else f"{method}{sparsity}"
    dest = GRID / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / f"{stem}.pt").exists():
        return
    if method == "dense" and seed == 411:
        src = WORKSPACE / "runs/pilot/seed411"
        if (src / "dense.pt").exists():
            for ext in ("csv", "json", "pt"):
                shutil.copy2(src / f"dense.{ext}", dest / f"dense.{ext}")
            return
    args = [
        sys.executable,
        str(SOURCE / "experiment.py"),
        method,
        "--seed",
        str(seed),
        "--sparsity",
        str(sparsity),
        "--epochs",
        "12" if method == "dense" else "4",
        "--lr",
        ".003" if method == "dense" else ".001",
        "--tag",
        "comparison",
    ]
    subprocess.run(args, check=True)


if __name__ == "__main__":
    for seed in SEEDS:
        run("dense", seed)
        for sparsity in (50, 80, 90):
            methods = ["magnitude", "admm"]
            if (SEEDS.index(seed) + (50, 80, 90).index(sparsity)) % 2:
                methods.reverse()
            for method in methods:
                run(method, seed, sparsity)
            subprocess.run(
                [
                    sys.executable,
                    str(SOURCE / "summarize.py"),
                    str(GRID / f"seed{seed}"),
                    "--sparsity",
                    str(sparsity),
                ],
                check=True,
            )
