"""Load run summaries and compute the statistics the source reports.

A run directory holds config.json and metrics.json (scripts/run.py). Statistics follow Sartor et
al. (2026): the median over seeds, with a 95% CI for the median from the order statistics at the
quantiles 0.5 +- 1.96 sqrt(0.25 / n) (normal approximation to the binomial).
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

SOURCE_TABLES = "sources/arXiv-2610.07417v1/tables"
# the source's numbers extracted from its LaTeX (scripts/extract_source_tables.py), for trees without sources/
SOURCE_JSON = "data/sartor2026_tables.json"

# column headers of the source's tables, to our benchmark names
SOURCE_NAMES = {
    "Sinc-1D": "sinc1d", "Sinc-2D": "sinc2d", "Sinc-3D": "sinc3d", "Sinc-4D": "sinc4d", "Sinc-5D": "sinc5d",
    "GramacyLee": "gramacylee", "Ackley": "ackley", "Hartmann": "hartmann", "Rosenbrock": "rosenbrock",
    "Michalewicz": "michalewicz", "Brachistochrone": "brachistochrone", "MNIST": "mnist",
    "Pendulum": "pendulum", "Pinwheel": "pinwheel", "Hopper": "hopper",
}
SOURCE_METHODS = {"Random": "random", r"\kundu": "kundu", r"\vien": "vien", r"\shilton": "shilton",
                  r"\vellanky": "vellanky", r"\wycoff": "l0mo"}


def default_runs() -> str:
    """Where the run summaries are: $FBO_RUNS, else the shipped results/runs, else out/runs."""
    return os.environ.get("FBO_RUNS") or ("results/runs" if os.path.isdir("results/runs") else "out/runs")


def default_diag() -> str:
    """Where the surrogate diagnostics are: $FBO_DIAG, else results/diag, else out/diag/final."""
    return os.environ.get("FBO_DIAG") or ("results/diag" if os.path.isdir("results/diag") else "out/diag/final")


def load_runs(root: str = "out/runs", campaigns: list[str] | None = None) -> pd.DataFrame:
    """Every run under root/<campaign>/<run>/; campaigns match by name or by the part before a '-'."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "metrics.json"))):  # one order on every filesystem
        run_dir = os.path.dirname(path)
        campaign = os.path.basename(os.path.dirname(run_dir))
        if campaigns is not None and campaign not in campaigns and campaign.split("-")[0] not in campaigns:
            continue
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg, met = json.load(open(cfg_path)), json.load(open(path))
        rows.append(dict(
            campaign=campaign, run=os.path.basename(run_dir), method=cfg["method"], target=cfg["target"],
            tau=float(cfg.get("tau", 1.0)), max_degree=cfg.get("max_degree"),
            profile=cfg["profile"], lengthscale=float(cfg["lengthscale"]), seed=int(cfg["seed"]),
            final=met["final"], average=met["average"], ys=np.asarray(met["ys"]),
            fit_time=met["surrogate_fit_time"], acq_time=met["acquisition_time"],
            eval_time=met["target_evaluation_time"], wall_time=met["wall_time"],
            max_rss_mb=met.get("max_rss_mb", np.nan),
            surrogate_params=met.get("surrogate_params"), surrogate_layout=met.get("surrogate_layout"),
        ))
    return pd.DataFrame(rows)


def pooled(df: pd.DataFrame) -> pd.DataFrame:
    """The runs the tables and figures pool: not the pilot (c0), the timing study (c8) or the
    sensitivity study (c9), which repeat seeds of the main campaigns, and MNIST only from c11 and
    c12, on the padded objective (earlier MNIST runs used the unpadded one)."""
    prefix = df.campaign.map(lambda c: c.split("-")[0])
    return df[~prefix.isin(["c0", "c8", "c9"]) & ((df.target != "mnist") | prefix.isin(["c11", "c12"]))]


def median_ci(values) -> tuple[float, float, float]:
    """Median and the 95% order-statistic CI of the median, as in the source's plots.py."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    half = 1.96 * np.sqrt(0.25 / n)
    return float(np.median(v)), float(np.quantile(v, max(0.5 - half, 0.0))), float(np.quantile(v, min(0.5 + half, 1.0)))


def summarize(df: pd.DataFrame, keys=("target", "method", "profile", "lengthscale")) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(list(keys)):
        row = dict(zip(keys, key))
        row["n"] = len(g)
        for metric in ("final", "average"):
            m, lo, hi = median_ci(g[metric])
            row.update({metric: m, f"{metric}_lo": lo, f"{metric}_hi": hi})
        for t in ("fit_time", "acq_time", "eval_time"):
            row[t] = float(np.median(g[t]))
        rows.append(row)
    return pd.DataFrame(rows)


def _number(s: str) -> float:
    return float(s.replace(r"\ ", "").strip())


def source_table(name: str, root: str = SOURCE_TABLES) -> pd.DataFrame:
    """The source's results table (results_final_ci or results_avg_ci) as numbers.

    Read from the extracted JSON when the LaTeX sources are absent.

    Cells read `$m\\ [lo, hi]$`, possibly bold, with a power of ten per column in the header, e.g.
    `[$10^{-2}$]`; Hopper's header carries `[$-10^{0}$]`, a sign as well.
    """
    if not os.path.isdir(root) and os.path.exists(SOURCE_JSON):
        return pd.DataFrame(json.load(open(SOURCE_JSON))[name])
    text = open(os.path.join(root, f"{name}.tex")).read()
    rows, scales = [], {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(r"\textbf{Method}"):
            cols = [c.strip() for c in line.rstrip("\\").split("&")[1:]]
            header = []
            for c in cols:
                bench = re.search(r"\\textbf\{([^}]*)\}", c).group(1)
                sign, exponent = re.search(r"\[\$(-?)10\^\{(-?\d+)\}\$\]", c).groups()
                header.append((SOURCE_NAMES[bench], (-1.0 if sign else 1.0) * 10.0 ** int(exponent)))
            continue
        method = next((v for k, v in SOURCE_METHODS.items() if line.startswith(k + " ") or line.startswith(k + "&")
                       or line.split("&")[0].strip() == k), None)
        if method is None or "&" not in line:
            continue
        cells = [c.strip() for c in line.rstrip("\\").split("&")[1:]]
        for (bench, scale), cell in zip(header, cells):
            m = re.search(r"\$\\?(?:mathbf\{)?\s*([-\d.]+)\\ \[([-\d.]+),\s*([-\d.]+)\]", cell)
            if m is None:
                continue  # the method does not apply (x mark)
            med, lo, hi = (float(x) * scale for x in m.groups())
            if scale < 0:
                lo, hi = hi, lo
            rows.append(dict(target=bench, method=method, median=med, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def source_lengthscale_table(root: str = SOURCE_TABLES) -> pd.DataFrame:
    """The source's medians at every RKHS lengthscale, each method at its best GP profile there
    (Table tab:lengthscale_full): one row per (benchmark, method, lengthscale) with `final` and
    `average`.

    Read from the extracted JSON when the LaTeX sources are absent. Rows read `Name [$10^{e}$] & avg &
    final & ...`, one pair of cells per method, under a `lengthscale $\\ell = x$` heading; the power
    of ten is per row and Hopper's carries a sign.
    """
    if not os.path.isdir(root) and os.path.exists(SOURCE_JSON):
        return pd.DataFrame(json.load(open(SOURCE_JSON))["lengthscale_full"])
    text = open(os.path.join(root, "lengthscale_full.tex")).read()
    methods = ["random", "kundu", "vien", "shilton", "vellanky", "l0mo"]
    rows, ell = [], None
    for line in text.splitlines():
        heading = re.search(r"lengthscale \$\\ell = ([\d.]+)\$", line)
        if heading:
            ell = float(heading.group(1))
            continue
        head = re.match(r"\s*([A-Za-z0-9-]+) \[\$(-?)10\^\{(-?\d+)\}\$\]\s*&", line)
        if head is None or ell is None or head.group(1) not in SOURCE_NAMES:
            continue
        scale = (-1.0 if head.group(2) else 1.0) * 10.0 ** int(head.group(3))
        cells = [c.strip() for c in line.rstrip().rstrip("\\").split("&")[1:]]
        for i, method in enumerate(methods):
            values = [re.search(r"\$\\?(?:mathbf\{)?\s*([-\d.]+)", c) for c in cells[2 * i:2 * i + 2]]
            if None in values:
                continue  # the method does not apply (x mark)
            avg, final = (float(v.group(1)) * scale for v in values)
            rows.append(dict(target=SOURCE_NAMES[head.group(1)], method=method, lengthscale=ell,
                             average=avg, final=final))
    return pd.DataFrame(rows)


SMOOTHNESS = {r"\infty": "rbf", "2": "matern52", "1": "matern32", "0": "matern12"}


def source_selection(root: str = SOURCE_TABLES) -> dict[tuple[str, str], tuple[float, str]]:
    """(benchmark, method) -> (RKHS lengthscale, GP profile) the source selected (Table tab:hyperparams).

    The table lists Kundu, Vien, Shilton and L0MO; Vellanki and Random are not in it.
    """
    if not os.path.isdir(root) and os.path.exists(SOURCE_JSON):
        return {(t, m): (ell, p) for t, m, ell, p in json.load(open(SOURCE_JSON))["selection"]}
    text = open(os.path.join(root, "best_lengthscales.tex")).read()
    methods = ["kundu", "vien", "shilton", "l0mo"]
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        if len(cells) != 6 or cells[0] not in SOURCE_NAMES:
            continue
        ell = float(cells[1].strip("$"))
        for method, cell in zip(methods, cells[2:]):
            nu = re.search(r"\\mathcal\{C\}\((\\infty|\d)\)", cell).group(1)
            out[(SOURCE_NAMES[cells[0]], method)] = (ell, SMOOTHNESS[nu])
    return out
