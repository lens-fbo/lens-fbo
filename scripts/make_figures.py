"""Build the manuscript's figures from the pulled run summaries.

    python scripts/make_figures.py convergence --methods ours-spectral --reported l0mo \\
        --out paper/figures/convergence.pdf
    python scripts/make_figures.py decay --out paper/figures/decay.pdf

Convergence panels show, per benchmark, the median running best over seeds and its 95%
order-statistic CI band, at the configuration each method is reported at, and a dashed line at the
median final value Sartor et al. (2026) report for each method given to --reported. Both figures
pool the same runs as the tables (fbo.results.pooled).
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fbo import benchmarks, results, selection  # noqa: E402

# categorical slots in fixed order (validated default palette, light mode); grey is the reference
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REFERENCE = "#8a8984"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

LABELS = {
    "sinc1d": "Sinc-1D", "sinc2d": "Sinc-2D", "sinc3d": "Sinc-3D", "sinc4d": "Sinc-4D", "sinc5d": "Sinc-5D",
    "gramacylee": "GramacyLee", "ackley": "Ackley", "hartmann": "Hartmann", "rosenbrock": "Rosenbrock",
    "michalewicz": "Michalewicz", "brachistochrone": "Brachistochrone", "mnist": "MNIST",
    "pendulum": "Pendulum", "pinwheel": "Pinwheel", "hopper": "Hopper",
}
METHOD_NAMES = {"l0mo": "L0MO", "random": "Random", "vien": "Vien", "shilton": "Shilton", "kundu": "Kundu",
                "vellanky": "Vellanki", "ours-l2": "LeNS-L2", "ours-decay": "LeNS-decay",
                "ours-spectral": "LeNS"}
LINEAR = ("mnist", "pendulum", "hopper")  # MNIST spans less than a decade


def style() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 7, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "text.color": INK, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def band(ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = np.minimum.accumulate(ys, axis=1)
    n = len(best)
    half = 1.96 * np.sqrt(0.25 / n)
    return (np.median(best, 0), np.quantile(best, max(0.5 - half, 0), axis=0),
            np.quantile(best, min(0.5 + half, 1), axis=0))


def convergence(df, methods: list[str], reported: list[str], targets: list[str], out: str,
                colors: dict[str, str]) -> None:
    style()
    source = results.source_table("results_final_ci")
    ncol = 5
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.9, 1.45 * nrow + 0.35), squeeze=False)
    for ax, t in zip(axes.flat, targets):
        for m in methods:
            g = df[(df.target == t) & (df.method == m)]
            if g.empty:
                continue
            n_eval = min(len(y) for y in g.ys)
            med, lo, hi = band(np.stack([y[:n_eval] for y in g.ys]))
            x = np.arange(1, n_eval + 1)
            ax.fill_between(x, lo, hi, color=colors[m], alpha=0.18, linewidth=0)
            ax.plot(x, med, color=colors[m], linewidth=1.1, label=METHOD_NAMES[m])
        for m in reported:
            r = source[(source.target == t) & (source.method == m)]
            if len(r):
                ax.axhline(r["median"].iloc[0], color=colors[m], linewidth=0.9, linestyle=(0, (3, 2)),
                           label=f"{METHOD_NAMES[m]}, reported final value")
        if t not in LINEAR and (df[df.target == t].final > 0).all():
            ax.set_yscale("log")
        ax.set_title(LABELS[t], color=INK, pad=2)
        ax.grid(True, color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(1, 110)
        ax.set_xticks([1, 50, 110])
    for ax in axes.flat[len(targets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Evaluations")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (1.45 * nrow + 0.35)), h_pad=0.6, w_pad=0.4)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convergence")
    c.add_argument("--campaigns", nargs="+", default=None, help="default: every campaign")
    c.add_argument("--methods", nargs="+", required=True)
    c.add_argument("--reported", nargs="*", default=["l0mo"], help="baselines drawn at their reported final value")
    c.add_argument("--targets", nargs="+", default=list(benchmarks.BENCHMARKS))
    c.add_argument("--root", default=results.default_runs())
    c.add_argument("--out", required=True)
    d = sub.add_parser("decay")
    d.add_argument("--campaigns", nargs="+", default=None, help="default: every campaign")
    d.add_argument("--methods", nargs="+", default=["ours-spectral"])
    d.add_argument("--profile", default="matern52", help="one GP profile for every benchmark")
    d.add_argument("--root", default=results.default_runs())
    d.add_argument("--out", required=True)
    args = ap.parse_args()

    df = results.pooled(results.load_runs(args.root, args.campaigns))
    if args.cmd == "decay":
        # one profile for every benchmark, at the source's lengthscale: no selection enters the figure
        df = df[df.apply(lambda r: r.lengthscale == benchmarks.SELECTED[r.target][0], axis=1)]
        decay(df[df.profile == args.profile], args.methods, args.out)
        return
    # every method at the configuration it is reported at (fbo.selection.reported_runs)
    families = [m for m in args.methods if m.startswith("ours-")]
    df = selection.reported_runs(df, families)
    df = df.assign(method=df.row)
    targets = [t for t in args.targets if t in set(df.target)]
    colors, slot = {}, 0
    for m in args.methods + args.reported:  # fixed order: the first method listed takes slot 1, and so on
        if m == "random":
            colors[m] = REFERENCE
        else:
            colors[m] = SLOTS[slot]
            slot += 1
    convergence(df, args.methods, args.reported, targets, args.out, colors)



def decay(df, methods: list[str], out: str) -> None:
    """The decay b of the last fit of every run, per benchmark: the geometry the data choose."""
    style()
    targets = [t for t in benchmarks.BENCHMARKS if t in set(df.target)]
    fig, ax = plt.subplots(figsize=(3.3, 0.18 * len(targets) + 0.7))
    offsets = np.linspace(-0.15, 0.15, len(methods)) if len(methods) > 1 else [0.0]
    for color, off, m in zip(SLOTS, offsets, methods):
        for i, t in enumerate(targets):
            g = df[(df.target == t) & (df.method == m)]
            bs = [p[-1][lay["m"]] for p, lay in zip(g.surrogate_params, g.surrogate_layout) if p and lay]
            if not bs:
                continue
            y = len(targets) - 1 - i + off
            ax.scatter(bs, np.full(len(bs), y), s=9, color=color, alpha=0.55, linewidths=0,
                       label=METHOD_NAMES[m] if i == 0 else None)
    ax.axvline(0.0, color=INK_MUTED, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([LABELS[t] for t in reversed(targets)])
    ax.set_xlabel("Fitted decay $b$ at the end of the run")
    ax.grid(True, axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    if len(methods) > 1:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(methods),
                  handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
