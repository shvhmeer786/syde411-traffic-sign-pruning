"""Prepare the official GTSRB training archive and a reproducible track-level split."""

import csv
import hashlib
import io
import json
import random
import subprocess
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent / "data"
URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
SHA256 = "d32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c"
SEED = 411


def prepare():
    DATA.mkdir(exist_ok=True)
    path = DATA / "GTSRB_Final_Training_Images.zip"
    if not path.exists():
        partial = path.with_suffix(".part")
        subprocess.run(
            ["curl", "-fL", "--retry", "3", URL, "-o", str(partial)], check=True
        )
        partial.replace(path)
    with path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == SHA256, (
            "archive checksum mismatch"
        )
    rng = random.Random(SEED)
    images, manifest = [], []
    with ZipFile(path) as archive:
        for label in range(43):
            folder = f"GTSRB/Final_Training/Images/{label:05d}/"
            with archive.open(folder + f"GT-{label:05d}.csv") as f:
                rows = sorted(
                    csv.DictReader(io.TextIOWrapper(f), delimiter=";"),
                    key=lambda r: r["Filename"],
                )
            tracks = sorted({r["Filename"].split("_")[0] for r in rows})
            validation = set(rng.sample(tracks, max(1, round(len(tracks) * 0.20))))
            for row in rows:
                assert int(row["ClassId"]) == label
                name = row["Filename"]
                track = name.split("_")[0]
                with Image.open(io.BytesIO(archive.read(folder + name))) as source:
                    box = tuple(
                        int(row[key]) + int(key.endswith("2"))
                        for key in ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2")
                    )
                    img = (
                        source.convert("RGB")
                        .crop(box)
                        .resize((32, 32), Image.Resampling.BILINEAR)
                    )
                    images.append(np.asarray(img))
                manifest.append(
                    (
                        folder + name,
                        label,
                        f"{label:05d}/{track}",
                        "validation" if track in validation else "train",
                    )
                )
    labels = np.array([row[1] for row in manifest])
    valid = np.array([row[3] == "validation" for row in manifest])
    sets = [
        {r[2] for r in manifest if r[3] == split} for split in ("train", "validation")
    ]
    assert len(manifest) == 39209 and not sets[0].intersection(sets[1])
    for mask in (valid, ~valid):
        assert len(np.unique(labels[mask])) == 43
    with (DATA / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["archive_path", "class", "track", "partition"])
        writer.writerows(manifest)
    tmp = DATA / "prepared.tmp.npz"
    np.savez(tmp, images=np.stack(images), labels=labels, validation=valid)
    tmp.replace(DATA / "prepared.npz")
    print(
        json.dumps(
            {
                "train": int((~valid).sum()),
                "validation": int(valid.sum()),
                "train_tracks": len(sets[0]),
                "validation_tracks": len(sets[1]),
                "common_tracks": 0,
                "classes": 43,
                "seed": SEED,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
