"""Draw the report figures from the original experiment logs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from evaluate import GRID, OUT, read_csv

INK = "#282725"
CLARET = "#782D4B"
BLUE = "#285575"
GREY = "#77736D"
METHODS = {
    "magnitude": {
        "label": "Magnitude pruning",
        "color": BLUE,
        "marker": "o",
        "linestyle": "-",
    },
    "admm": {
        "label": "ADMM",
        "color": CLARET,
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
}


def axis(ax, title):
    ax.set_title(title, loc="left", fontsize=10.7, fontweight="bold", pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1DFDB", linewidth=0.55)
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#AAA69F")
    ax.tick_params(width=0.65, length=3, color="#847F79", pad=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def legend(fig, dense=False):
    handles = [Line2D([], [], **style) for style in METHODS.values()]
    if dense:
        handles.append(
            Line2D([], [], label="Dense reference", color=GREY, linestyle=":")
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 1.005),
        ncol=len(handles),
        frameon=False,
        handlelength=2.1,
        columnspacing=1.9,
    )


def make_figures():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "font.size": 9.3,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.4,
            "legend.fontsize": 9.1,
        }
    )
    rows = read_csv(OUT / "test_results.csv")
    levels = (50, 80, 90)
    fig, panels = plt.subplots(1, 2, figsize=(7.1, 2.65))
    fig.subplots_adjust(left=0.071, right=0.983, bottom=0.19, top=0.73, wspace=0.28)
    for ax, key, scale, title in zip(
        panels,
        ("accuracy", "ce"),
        (100, 1),
        ("Test accuracy (%)", "Test cross-entropy"),
    ):
        axis(ax, title)
        ref = np.mean([float(r[key]) * scale for r in rows if r["method"] == "dense"])
        ax.axhline(ref, color=GREY, linestyle=":", lw=1.1)
        for method, style in METHODS.items():
            samples = np.array(
                [
                    [
                        float(r[key]) * scale
                        for r in rows
                        if r["method"] == method and int(r["sparsity"]) == s
                    ]
                    for s in levels
                ]
            )
            ax.errorbar(
                levels,
                samples.mean(axis=1),
                yerr=samples.std(axis=1, ddof=1),
                capsize=3.0,
                elinewidth=1.0,
                **style,
            )
        ax.set(xlabel="Weights pruned (%)", xticks=levels, xlim=(47, 93))
    legend(fig, dense=True)
    fig.savefig(OUT / "test_comparison.png", dpi=320)
    plt.close(fig)

    folder = GRID / "seed411"
    fig, panels = plt.subplots(2, 2, figsize=(7.1, 4.44))
    fig.subplots_adjust(
        left=0.09, right=0.986, bottom=0.105, top=0.85, hspace=0.80, wspace=0.30
    )
    for ax, title in zip(
        panels.flat,
        (
            "Training objective",
            "Validation cross-entropy",
            "ADMM primal residual",
            "ADMM dual residual",
        ),
    ):
        axis(ax, title)
    for method, style in METHODS.items():
        history = read_csv(folder / f"{method}80.csv")
        for ax, xkey, ykey in (
            (panels[0, 0], "updates", "online_train_F"),
            (panels[0, 1], "training_seconds", "ce"),
        ):
            points = [r for r in history if r[ykey]]
            ax.plot(
                [float(r[xkey]) for r in points],
                [float(r[ykey]) for r in points],
                **style,
            )
    panels[0, 0].set(xlabel="Gradient updates", xticks=[246, 492, 738, 984])
    panels[0, 1].set(xlabel="Training time (s)", xticks=[0, 3, 6, 9, 12])
    history = [r for r in read_csv(folder / "admm80.csv") if r["primal"]]
    for ax, key in zip(panels[1], ("primal", "dual")):
        ax.plot(
            [int(r["epoch"]) for r in history],
            [float(r[key]) for r in history],
            **METHODS["admm"],
        )
        ax.set(xticks=(1, 2, 3), xlabel="Outer iteration")
    panels[1, 1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    legend(fig)
    fig.savefig(OUT / "solver_behavior.png", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    make_figures()
