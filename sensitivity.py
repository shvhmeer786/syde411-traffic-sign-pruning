"""One-factor sensitivity analysis on validation data, starting from the same dense model."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sweep():
    reference = ROOT / "runs/comparison/seed411/dense.pt"
    for name, rho, rate in [
        ("rho_half", 0.0005, 0.001),
        ("rho_double", 0.002, 0.001),
        ("rate_half", 0.001, 0.0005),
        ("rate_double", 0.001, 0.002),
    ]:
        folder = ROOT / "runs" / name / "seed411"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "admm80.pt").exists():
            continue
        shutil.copy2(reference, folder / "dense.pt")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiment.py"),
                "admm",
                "--epochs",
                "4",
                "--rho",
                str(rho),
                "--lr",
                str(rate),
                "--tag",
                name,
            ],
            check=True,
        )


if __name__ == "__main__":
    sweep()
