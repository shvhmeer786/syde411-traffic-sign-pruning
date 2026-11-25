"""Measure the training objective at saved checkpoints and test one feasible descent direction."""

import numpy as np
import torch

from evaluate import GRID, OUT, read_csv, write_csv
from experiment import LAMBDA, model, weights
from project_paths import WORKSPACE


@torch.inference_mode()
def measure(net, images, labels):
    # A last-layer bias shift adds the same class-specific offsets to each logit vector.
    logits = torch.cat([net(batch) for batch in images.split(256)]).double()
    probabilities = logits.softmax(dim=1)
    frequencies = torch.bincount(labels, minlength=43).double() / len(labels)
    bias_gradient = probabilities.mean(dim=0) - frequencies
    penalty = (LAMBDA / 2) * sum(w.double().square().sum().item() for w in weights(net))
    ce = torch.nn.functional.cross_entropy(logits, labels).item()
    perturbed_ce = torch.nn.functional.cross_entropy(
        logits - bias_gradient, labels
    ).item()
    return {
        "training_ce": ce,
        "regularization": penalty,
        "training_F": ce + penalty,
        "output_bias_gradient_norm": bias_gradient.norm().item(),
        "feasible_decrease": ce - perturbed_ce,
        "nonzeros": sum(int(w.count_nonzero()) for w in weights(net)),
    }


def main():
    torch.set_num_threads(4)
    with np.load(WORKSPACE / "data/prepared.npz") as data:
        train = ~data["validation"]
        images = (
            torch.from_numpy(data["images"][train]).permute(0, 3, 1, 2).float() / 127.5
            - 1
        )
        labels = torch.from_numpy(data["labels"][train])
    results = []
    for row in read_csv(OUT / "test_results.csv"):
        method = row["method"]
        stem = method if method == "dense" else method + row["sparsity"]
        net = model().eval()
        net.load_state_dict(
            torch.load(
                GRID / ("seed" + row["seed"]) / (stem + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        values = measure(net, images, labels)
        assert values["nonzeros"] == int(row["nonzeros"]) <= int(row["cap"])
        results.append(
            dict(seed=row["seed"], method=method, sparsity=row["sparsity"], **values)
        )
    write_csv(OUT / "training_objective_audit.csv", results)


if __name__ == "__main__":
    main()
