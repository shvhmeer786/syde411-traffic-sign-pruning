import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
dest = root / "deliverables/Group28_Python_and_Results.zip"
paths = [
    *root.glob("*.py"),
    root / "requirements.txt",
    root / "requirements-notebook.txt",
    root / "notebooks/results.ipynb",
    root / "README.md",
    root / "data/manifest.csv",
    *[p for p in (root / "results").glob("*") if p.is_file()],
]
for p in (root / "runs").rglob("*"):
    if p.is_file() and p.suffix in {".csv", ".json", ".pt"}:
        paths.append(p)
with ZipFile(dest, "w", compression=ZIP_DEFLATED) as archive:
    for p in sorted(paths):
        archive.write(p, p.relative_to(root))
with ZipFile(dest) as z:
    assert z.testzip() is None
    assert {
        "prepare.py",
        "experiment.py",
        "evaluate.py",
        "data/manifest.csv",
        "results/test_results.csv",
    } <= set(z.namelist())
    assert (
        len(
            list(
                csv.DictReader(z.read("results/test_results.csv").decode().splitlines())
            )
        )
        == 21
    )
    for seed in (411, 412, 413):
        prefix = f"runs/comparison/seed{seed}/"
        dense = hashlib.sha256(z.read(prefix + "dense.pt")).hexdigest()
        for s in (50, 80, 90):
            for m in ("admm", "magnitude"):
                assert (
                    json.loads(z.read(prefix + f"{m}{s}.json"))[
                        "dense_checkpoint_sha256"
                    ]
                    == dense
                )
print(len(paths), "files verified in archive", dest.stat().st_size)
