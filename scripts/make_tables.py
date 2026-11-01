"""Build the manuscript's tables from the pulled run summaries.

    python scripts/make_tables.py --out paper/tables

The baselines' numbers are those Sartor et al. (2026) report (fbo.results.source_table and
source_lengthscale_table); everything about LeNS comes from the runs' metrics.json files. Cells are
the median over seeds and the 95% order-statistic CI of the median; bold marks the lowest median and
every method whose CI contains it. Units follow the source: one power of ten per benchmark.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fbo import benchmarks, results, selection  # noqa: E402

# three benchmarks per block: more columns of median and CI overflow the journal's text width
BLOCKS = [
    ("sinc1d", "sinc2d", "sinc3d"),
    ("sinc4d", "sinc5d", "gramacylee"),
    ("ackley", "hartmann", "rosenbrock"),
    ("michalewicz", "brachistochrone", "mnist"),
    ("pendulum", "pinwheel", "hopper"),
]
LABELS = {
    "sinc1d": "Sinc-1D", "sinc2d": "Sinc-2D", "sinc3d": "Sinc-3D", "sinc4d": "Sinc-4D", "sinc5d": "Sinc-5D",
    "gramacylee": "GramacyLee", "ackley": "Ackley", "hartmann": "Hartmann", "rosenbrock": "Rosenbrock",
    "michalewicz": "Michalewicz", "brachistochrone": "Brachistochrone", "mnist": "MNIST", "pendulum": "Pendulum",
    "pinwheel": "Pinwheel", "hopper": "Hopper",
}
METHOD_LABELS = {
    "ours-spectral@m52": r"\ourss{} (Mat\'ern-5/2)", "ours-decay@m52": r"\oursd{} (Mat\'ern-5/2)",
    "ours-l2@m52": r"\oursl{} (Mat\'ern-5/2)",
    "random": "Random", "kundu": "Kundu", "vien": "Vien", "shilton": "Shilton", "vellanky": "Vellanki",
    "l0mo": "L0MO", "ours-l2": r"\oursl{}", "ours-decay": r"\oursd{}", "ours-spectral": r"\ourss{}",
}


def source_scales() -> dict[str, float]:
    """The power of ten (with sign) the source uses for each benchmark's column."""
    if not os.path.isdir(results.SOURCE_TABLES):
        import json

        return json.load(open(results.SOURCE_JSON))["scales"]
    text = open(os.path.join(results.SOURCE_TABLES, "results_final_ci.tex")).read()
    scales = {}
    for bench, sign, exponent in re.findall(r"\\textbf\{([^}]*)\} \[\$(-?)10\^\{(-?\d+)\}\$\]", text):
        scales[results.SOURCE_NAMES[bench]] = (-1.0 if sign else 1.0) * 10.0 ** int(exponent)
    return scales


def fmt(x: float) -> str:
    """Three significant digits, as in the source's tables."""
    if x == 0:
        return "0"
    digits = max(0, 2 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{digits}f}"


def fmt_reported(x: float) -> str:
    """A reported number as the source prints it: two decimals below 10, one below 100."""
    return f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}" if abs(x) < 100 else f"{x:.0f}"


def results_table(summary: pd.DataFrame, methods: list[str], metric: str, caption: str, label: str,
                  marked: tuple[str, ...] = ()) -> str:
    scales = source_scales()
    lines = [
        r"\begin{table}[!tp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
        r"\begin{tabular}{l" + "c" * max(len(b) for b in BLOCKS) + "}", r"\toprule",
    ]
    for b, block in enumerate(BLOCKS):
        if b:
            lines += [r"\midrule\midrule"]
        head = []
        for t in block:
            exp = int(round(np.log10(abs(scales[t]))))
            sign = "-" if scales[t] < 0 else ""
            dagger = r"$^\dagger$" if t in marked else ""
            head.append(rf"\textbf{{{LABELS[t]}}}{dagger} [${sign}10^{{{exp}}}$]")
        lines += [r"\textbf{Method} & " + " & ".join(head) + r" \\", r"\midrule"]
        cells = {}
        for t in block:
            rows = summary[summary.target == t].set_index("method")
            present = [m for m in methods if m in rows.index]
            if not present:
                continue
            best = min(rows.loc[m, metric] for m in present)
            for m in present:
                r = rows.loc[m]
                s = scales[t]
                med, lo, hi = r[metric] / s, r[f"{metric}_lo"] / s, r[f"{metric}_hi"] / s
                lo, hi = min(lo, hi), max(lo, hi)
                f = fmt if m.startswith("ours-") else fmt_reported  # the baselines as published
                text = f"{f(med)}\\ [{f(lo)}, {f(hi)}]"
                tied = r[f"{metric}_lo"] <= best <= r[f"{metric}_hi"] or r[metric] == best
                cells[(t, m)] = rf"$\mathbf{{{text}}}$" if tied else f"${text}$"
        for m in methods:
            row = [cells.get((t, m), r"\text{\ding{55}}") for t in block]
            lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def reported_summary() -> pd.DataFrame:
    """The baselines as Sartor et al. (2026) report them, in the columns of results.summarize."""
    final, avg = results.source_table("results_final_ci"), results.source_table("results_avg_ci")
    r = final.merge(avg, on=["target", "method"], suffixes=("_f", "_a"))
    return pd.DataFrame(dict(target=r.target, method=r.method, final=r.median_f, final_lo=r.lo_f, final_hi=r.hi_f,
                             average=r.median_a, average_lo=r.lo_a, average_hi=r.hi_a))


def diagnostic_table(diag_dir: str, n_train: int, caption: str, label: str) -> str:
    """Held-out Spearman / NLPD of GPs that differ only in their metric, and the fitted decay b.

    The oracle, which exists for the ridge functionals only, is quoted in the text instead of a
    column that would be empty on the other benchmarks.
    """
    import glob
    import json

    models = [("rkhs", "RKHS"), ("l2", r"$L^2(\mathcal{D})$"), ("fm-decay", "decay"),
              ("fm-spectral", "spectral")]
    data = {}
    for path in glob.glob(os.path.join(diag_dir, "*_all.json")):
        rows = json.load(open(path))
        data[rows[0]["target"]] = [r for r in rows if r["n_train"] == n_train]
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "c" * len(models) + "r}",
        r"\toprule",
        "Benchmark & " + " & ".join(name for _, name in models) + r" & $\hat b$ \\", r"\midrule",
    ]
    for t in benchmarks.BENCHMARKS:
        rows = data.get(t)
        if not rows:
            continue
        cells = []
        spear = {m: np.mean([r["spearman"] for r in rows if r["model"] == m]) for m, _ in models}
        best = max(spear.values())
        for m, _ in models:
            sel = [r for r in rows if r["model"] == m]
            sp, nl = np.mean([r["spearman"] for r in sel]), np.mean([r["nlpd"] for r in sel])
            sign = r"\textminus{}" if round(nl, 2) < 0 else "+"  # a minus sign, not a hyphen
            text = f"{sp:.2f} / {sign}{abs(nl):.2f}"
            cells.append(rf"\textbf{{{text}}}" if round(sp, 2) == round(best, 2) else text)
        b = [r["b"] for r in rows if r["model"] == "fm-decay" and "b" in r and np.isfinite(r["b"])]
        cells.append(f"${np.mean(b):+.2f}$")
        lines.append(f"{LABELS[t]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def diagnostic_numbers(diag_dir: str, n_train: int = 50) -> dict[str, str]:
    """Numbers the text quotes from the diagnostics, as LaTeX macro bodies."""
    import glob
    import json

    data = {}
    for path in glob.glob(os.path.join(diag_dir, "*_all.json")):
        rows = json.load(open(path))
        data[rows[0]["target"]] = rows
    better, lam = 0, []
    for t, rows in data.items():
        at = [r for r in rows if r["n_train"] == n_train]
        sp = lambda m: np.mean([r["spearman"] for r in at if r["model"] == m])
        if max(sp("l2"), sp("fm-decay"), sp("fm-spectral")) > sp("rkhs"):
            better += 1
        lam += [r["lam_over_ell"] for r in rows if r["model"] == "lam"]
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen"]
    per_target = {t: np.median([r["lam_over_ell"] for r in rows if r["model"] == "lam"]) for t, rows in data.items()}
    above = sum(v > 1.2 for v in per_target.values())
    oracle, closed = [], []
    for t, rows in data.items():
        at = [r for r in rows if r["n_train"] == n_train]
        sp = lambda m: np.mean([r["spearman"] for r in at if r["model"] == m])
        if any(r["model"] == "oracle" for r in at):  # the ridge functionals
            oracle.append(sp("oracle"))
            closed.append((max(sp("fm-decay"), sp("fm-spectral")) - sp("rkhs")) / (sp("oracle") - sp("rkhs")))
    return {
        "DiagOracleMin": f"{min(oracle):.2f}",
        "DiagOracleMax": f"{max(oracle):.2f}",
        "DiagGapMin": f"{100 * min(closed):.0f}",
        "DiagGapMax": f"{100 * min(max(closed), 1.0):.0f}",
        "DiagNBetter": words[better],
        "DiagNBenchmarks": words[len(data)],
        "DiagLamAbove": words[above],
        "DiagLamMin": f"{min(per_target.values()):.2f}",
        "DiagLamMax": f"{max(per_target.values()):.2f}",
    }


def write_numbers(path: str, numbers: dict[str, str]) -> None:
    with open(path, "w") as fh:
        fh.write("% generated by scripts/make_tables.py from the run summaries: numbers quoted in the text\n")
        for k, v in sorted(numbers.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def robustness_table(df: pd.DataFrame, family: str, metric: str, caption: str, label: str) -> tuple[str, dict]:
    """Per RKHS lengthscale: ours at Matern-5/2 against the median the source reports for L0MO at its
    best GP profile there (Table tab:lengthscale_full).

    Cells give both medians in the benchmark's unit; the lower is bold when L0MO's median lies outside
    our CI (the source reports no CI at every lengthscale), and both are plain otherwise.
    """
    scales = source_scales()
    # the ten main benchmarks of the source's lengthscale study, without MNIST (cost; see the text)
    main10 = ["gramacylee", "ackley", "hartmann", "rosenbrock", "michalewicz", "brachistochrone",
              "pendulum", "pinwheel", "hopper"]
    ells = [0.4, 0.2, 0.1, 0.05]
    d = df[df.seed < 16]
    reported = results.source_lengthscale_table()
    reported = reported[reported.method == "l0mo"]
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{l" + "c" * len(ells) + "}", r"\toprule",
        "Benchmark & " + " & ".join(rf"$\ell = {e:g}$" for e in ells) + r" \\", r"\midrule",
    ]
    wins = {e: [0, 0, 0] for e in ells}  # ours better, tie, L0MO better
    for t in main10:
        s = scales[t]
        exp = int(round(np.log10(abs(s))))
        cells = []
        for e in ells:
            lz = reported[(reported.target == t) & np.isclose(reported.lengthscale, e)]
            us = d[(d.target == t) & (d.method == family) & np.isclose(d.lengthscale, e) & (d.profile == "matern52")]
            if lz.empty or us.empty:
                cells.append("--")
                continue
            a = results.median_ci(us[metric])
            b = float(lz[metric].iloc[0])
            # runs that all stop on one plateau collapse the CI: a difference that vanishes at the
            # printed precision is a tie, whatever the interval says
            visible = fmt(a[0] / s) != fmt(b / s)
            ours_wins = b > a[2] and visible
            l0_wins = b < a[1] and visible
            wins[e][0 if ours_wins else 2 if l0_wins else 1] += 1
            fa, fb = fmt(a[0] / s), fmt_reported(b / s)
            fa, fb = (rf"\textbf{{{fa}}}" if ours_wins else fa), (rf"\textbf{{{fb}}}" if l0_wins else fb)
            cells.append(f"{fa} / {fb}")
        unit = rf"[${'-' if s < 0 else ''}10^{{{exp}}}$]"
        lines.append(f"{LABELS[t]} {unit} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", "wins / ties / losses & " + " & ".join(f"{w[0]} / {w[1]} / {w[2]}" for w in wins.values())
              + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines), wins


def profile_table(df: pd.DataFrame, family: str, caption: str, label: str) -> tuple[str, dict]:
    """Median final value of ours at each GP profile (the source's lengthscale, seeds 0-15), beside the
    value the source reports for L0MO at the profile it selected; and, per profile, on how many
    benchmarks ours is ahead of that value."""
    scales = source_scales()
    reported = results.source_table("results_final_ci")
    profiles = [("rbf", "RBF"), ("matern52", "M-5/2"), ("matern32", "M-3/2"), ("matern12", "M-1/2")]
    ahead = {p: 0 for p, _ in profiles}
    d = df[(df.seed < 16) & (df.method == family)
           & df.apply(lambda r: r.lengthscale == benchmarks.SELECTED[r.target][0], axis=1)]
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\footnotesize", r"\begin{tabular}{l" + "c" * len(profiles) + "c}", r"\toprule",
        r" & \multicolumn{" + str(len(profiles)) + r"}{c}{\method{}} & \lzero{} \\",
        rf"\cmidrule(lr){{2-{1 + len(profiles)}}} \cmidrule(lr){{{2 + len(profiles)}-{2 + len(profiles)}}}",
        "Benchmark & " + " & ".join(name for _, name in profiles) + r" & reported \\", r"\midrule",
    ]
    for t in benchmarks.BENCHMARKS:
        g = d[d.target == t]
        if g.empty:
            continue
        s = scales[t]
        exp = int(round(np.log10(abs(s))))
        cells = []
        r = reported[(reported.target == t) & (reported.method == "l0mo")]
        for p, _ in profiles:
            v = g[g.profile == p].final
            cells.append(fmt(np.median(v) / s) if len(v) else "--")
            ahead[p] += bool(len(v) and len(r) and np.median(v) < r["median"].iloc[0])
        cells.append(fmt_reported(r["median"].iloc[0] / s) if len(r) else "--")
        unit = rf"[${'-' if s < 0 else ''}10^{{{exp}}}$]"
        lines.append(f"{LABELS[t]} {unit} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    names = {"rbf": "Rbf", "matern52": "Mfivehalves", "matern32": "Mthreehalves", "matern12": "Monehalf"}
    return "\n".join(lines), {f"ProfAhead{names[p]}": WORDS[n] for p, n in ahead.items()}


def cost_table(rep: pd.DataFrame, family: str, caption: str, label: str) -> tuple[str, dict]:
    """Median surrogate-fit and acquisition time per run, from the timing campaign (same nodes)."""
    d = rep[rep.campaign.notna() & rep.row.isin(["l0mo", family])]
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\footnotesize", r"\begin{tabular}{lrrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{Surrogate fit (s)} & \multicolumn{2}{c}{Acquisition (s)} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Benchmark & \lzero{} & \method{} & \lzero{} & \method{} \\", r"\midrule",
    ]
    ratios = []
    for t in benchmarks.BENCHMARKS:
        g = d[d.target == t]
        a, b = g[g.row == "l0mo"], g[g.row == family]
        if a.empty or b.empty:
            continue
        fa, fb, qa, qb = (float(np.median(x)) for x in (a.fit_time, b.fit_time, a.acq_time, b.acq_time))
        ratios.append((fb + qb) / (fa + qa))
        lines.append(f"{LABELS[t]} & {fa:.0f} & {fb:.0f} & {qa:.0f} & {qb:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    numbers = {"CostRatioMedian": f"{np.median(ratios):.1f}" if ratios else "--",
               "CostRatioMax": f"{np.max(ratios):.1f}" if ratios else "--"}
    return "\n".join(lines), numbers


def sensitivity_table(default: pd.DataFrame, variants: pd.DataFrame, family: str, caption: str, label: str) -> str:
    """Median final value of ours at Matern-5/2 at the default (tau, J) and at each variant, seeds 0-15."""
    from fbo.surrogate import Grid

    scales = source_scales()
    targets = [t for t in benchmarks.BENCHMARKS if t in set(variants.target)]
    lines = [
        r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
        r"Benchmark & default & $\tau = 0.3$ & $\tau = 3$ & $J_0 - 1$ & $J_0 + 1$ \\", r"\midrule",
    ]
    for t in targets:
        s = scales[t]
        exp = int(round(np.log10(abs(s))))
        d = 2 if t == "sinc2d" else 4 if t == "rosenbrock" else 1
        j0 = Grid.build(d).max_degree
        base = default[(default.target == t) & (default.method == family) & (default.profile == "matern52")
                       & (default.seed < 16) & np.isclose(default.lengthscale, benchmarks.SELECTED[t][0])]
        cells = [fmt(base.final.median() / s) if len(base) else "--"]
        v = variants[variants.target == t]
        for sel in (np.isclose(v.tau, 0.3), np.isclose(v.tau, 3.0), v.max_degree == j0 - 1, v.max_degree == j0 + 1):
            g = v[sel]
            cells.append(fmt(g.final.median() / s) if len(g) else "--")
        unit = rf"[${'-' if s < 0 else ''}10^{{{exp}}}$]"
        lines.append(rf"{LABELS[t]} {unit} ($J_0 = {j0}$) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


CAMEL = {"sinc1d": "SincOneD", "sinc2d": "SincTwoD", "sinc3d": "SincThreeD", "sinc4d": "SincFourD",
         "sinc5d": "SincFiveD", "gramacylee": "GramacyLee", "ackley": "Ackley", "hartmann": "Hartmann",
         "rosenbrock": "Rosenbrock", "michalewicz": "Michalewicz", "brachistochrone": "Brachistochrone",
         "mnist": "Mnist", "pendulum": "Pendulum", "pinwheel": "Pinwheel", "hopper": "Hopper"}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen"]


def comparison_numbers(summary: pd.DataFrame, ours: str, prefix: str) -> dict[str, str]:
    """Counts and ratios against L0MO that the text quotes."""
    out, lower, strict = {}, {"final": 0, "average": 0}, {"final": 0, "average": 0}
    n = 0
    for t in benchmarks.BENCHMARKS:
        g = summary[summary.target == t].set_index("row")
        if "l0mo" not in g.index or ours not in g.index:
            continue
        n += 1
        for metric in ("final", "average"):
            a, b = g.loc[ours], g.loc["l0mo"]
            if a[metric] < b[metric]:
                lower[metric] += 1
                if a[f"{metric}_hi"] < b[f"{metric}_lo"]:  # the two CIs do not overlap
                    strict[metric] += 1
            ratio = a[metric] / b[metric]
            out[f"{prefix}Ratio{metric.capitalize()}{CAMEL[t]}"] = f"{ratio:.2f}"
    for metric in ("final", "average"):
        out[f"{prefix}NLower{metric.capitalize()}"] = WORDS[lower[metric]]
        out[f"{prefix}NStrict{metric.capitalize()}"] = WORDS[strict[metric]]
    out[f"{prefix}NBenchmarks"] = WORDS[n]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=results.default_runs(), help="default: $FBO_RUNS, results/runs, out/runs")
    ap.add_argument("--out", required=True, help="directory of the tables; numbers.tex goes in its parent")
    ap.add_argument("--campaigns", nargs="+", default=None)
    ap.add_argument("--diag", default=results.default_diag(), help="surrogate diagnostics (scripts/diag_surrogate.py)")
    ap.add_argument("--family", default="ours-spectral", help="the main family")
    args = ap.parse_args()

    everything = results.load_runs(args.root, args.campaigns)
    df, timing = results.pooled(everything), everything[everything.campaign.map(lambda c: c.split("-")[0] == "c8")]
    dup = df.duplicated(["method", "target", "profile", "lengthscale", "seed"])
    if dup.any():
        raise SystemExit(f"{dup.sum()} duplicate runs across campaigns: {df[dup].run.head().tolist()}")
    os.makedirs(args.out, exist_ok=True)
    numbers = {}
    families = ["ours-l2", "ours-decay", "ours-spectral"]

    # the baselines as published; LeNS from its runs, at the configuration it is reported at
    reported = reported_summary()
    rep = selection.reported_runs(df, families)
    main_rows = ["random", "kundu", "vien", "shilton", "vellanky", "l0mo", args.family, f"{args.family}@m52"]
    ours = results.summarize(rep[rep.row.isin(main_rows[-2:])], keys=("target", "row")).rename(columns={"row": "method"})
    main_summary = pd.concat([reported, ours], ignore_index=True)
    for metric, what in (("final", "Best value at the end of the run"), ("average", "Average regret (mean of the running best)")):
        with open(os.path.join(args.out, f"main_{metric}.tex"), "w") as fh:
            fh.write(results_table(
                main_summary, main_rows, metric,
                f"{what}: median and 95\\% CI of the median over 32 seeds. The baselines as reported by "
                "\\citet{sartor2026sparse}, each at the configuration they report; \\method{} at the GP profile "
                "selected by the same rule on its own runs, and at Mat\\'ern-5/2 on every benchmark. Bold: "
                "lowest median, and every method whose CI contains it. \\ding{55}: not applicable (Vellanki "
                "needs a one-dimensional domain). $^\\dagger$: the released code does not attain the value "
                "reported for \\lzero{} there (Section~\\ref{sec:setup}).",
                f"tab:main-{metric}", marked=("mnist", "pinwheel")))
    numbers.update(comparison_numbers(main_summary.rename(columns={"method": "row"}), args.family, "Main"))
    numbers.update(comparison_numbers(main_summary.rename(columns={"method": "row"}), f"{args.family}@m52", "Fixed"))

    # the evaluation at which the median running best of ours first falls below the final value
    # reported for L0MO (Fig. convergence)
    for t in benchmarks.BENCHMARKS:
        g = rep[(rep.target == t) & (rep.row == args.family)]
        r = reported[(reported.target == t) & (reported.method == "l0mo")]
        if len(g) and len(r):
            best = np.median(np.stack([np.minimum.accumulate(y[:110]) for y in g.ys]), axis=0)
            below = np.nonzero(best < r.final.iloc[0])[0]
            numbers[f"Pass{CAMEL[t]}"] = str(below[0] + 1) if len(below) else "--"

    # Pinwheel and MNIST: run with the released code, L0MO ends far from the values reported there
    # (the setup says so, with these numbers; MNIST as a test error in percent)
    runs_l0mo = selection.reported_runs(df, [])
    for t, name, unit in (("pinwheel", "Pinwheel", 1.0), ("mnist", "Mnist", 100.0)):
        g = runs_l0mo[(runs_l0mo.target == t) & (runs_l0mo.row == "l0mo")]
        r = reported[(reported.target == t) & (reported.method == "l0mo")]
        if len(g) and len(r):
            numbers[f"Released{name}"] = f"{unit * results.median_ci(g.final)[0]:.2f}"
            numbers[f"Reported{name}"] = f"{unit * r.final.iloc[0]:.2f}"

    tex, cost_numbers = cost_table(
        timing.assign(row=timing.method), args.family,
        "Median time per run spent fitting the surrogate and optimizing the acquisition over the 110 "
        "evaluations, one CPU core per run, four seeds per method and benchmark, both methods at "
        "Mat\\'ern-5/2 and run interleaved on the same nodes.",
        "tab:cost")
    with open(os.path.join(args.out, "cost.tex"), "w") as fh:
        fh.write(tex)
    numbers.update(cost_numbers)

    # ablation: the three families, each at its selected profile, seeds 0-15; L0MO as reported
    abl = selection.reported_runs(df[df.seed < 16], families)
    abl_rows = ["l0mo", "ours-l2", "ours-decay", "ours-spectral"]
    abl_ours = results.summarize(abl[abl.row.isin(abl_rows[1:])], keys=("target", "row")).rename(columns={"row": "method"})
    abl_summary = pd.concat([reported[reported.method == "l0mo"], abl_ours], ignore_index=True)
    with open(os.path.join(args.out, "ablation.tex"), "w") as fh:
        fh.write(results_table(
            abl_summary, abl_rows, "final",
            "Ablation of the metric family, best value at the end of the run (median and 95\\% CI of the "
            "median over 16 seeds), each family at the GP profile selected on its own runs; \\lzero{} as "
            "reported by \\citet{sartor2026sparse} (32 seeds). \\oursl{}: isotropic $L^2(\\mathcal{D})$ "
            "metric; \\oursd{}: spectral decay without per-mode deviations; \\ourss{}: the full family. "
            "$^\\dagger$: as in Table~\\ref{tab:main-final}.",
            "tab:ablation", marked=("mnist", "pinwheel")))

    # sensitivity to the method's own hyperparameters (campaign c9)
    sens = everything[everything.campaign.map(lambda c: c.split("-")[0] == "c9")]
    if len(sens):
        with open(os.path.join(args.out, "sensitivity.tex"), "w") as fh:
            fh.write(sensitivity_table(
                df, sens, args.family,
                "Sensitivity of \\method{} to its own hyperparameters: median best value at the end of the run "
                "over 16 seeds, Mat\\'ern-5/2, at the default prior scale $\\tau = 1$ and basis degree $J_0$ and "
                "with one of them changed.",
                "tab:sensitivity"))

    # sensitivity to the GP profile
    tex, profile_numbers = profile_table(
        df, args.family,
        "Median best value at the end of the run of \\method{} over 16 seeds at each GP profile (M: "
        "Mat\\'ern), at the RKHS lengthscale selected in the source, and the median reported for \\lzero{} "
        "by \\citet{sartor2026sparse} at the profile they selected for it.",
        "tab:profiles")
    with open(os.path.join(args.out, "profiles.tex"), "w") as fh:
        fh.write(tex)
    numbers.update(profile_numbers)

    # robustness to the RKHS lengthscale
    tex, wins = robustness_table(
        df, args.family, "final",
        "Robustness to the RKHS lengthscale $\\ell$: median best value at the end of the run, \\method{} at "
        "Mat\\'ern-5/2 over 16 seeds / \\lzero{} at the best of the four GP profiles for that $\\ell$, as "
        "reported by \\citet{sartor2026sparse}. Bold: the lower median, when the median of \\lzero{} lies "
        "outside the CI of \\method{} and the two differ at the printed precision. MNIST is left out (see "
        "the text).",
        "tab:robustness")
    with open(os.path.join(args.out, "robustness.tex"), "w") as fh:
        fh.write(tex)
    for e, (w, t_, l) in wins.items():
        key = {0.4: "A", 0.2: "B", 0.1: "C", 0.05: "D"}[e]
        numbers[f"RobWins{key}"], numbers[f"RobTies{key}"], numbers[f"RobLosses{key}"] = str(w), str(t_), str(l)
    numbers["RobWinsTotal"] = str(sum(w[0] for w in wins.values()))
    numbers["RobLossesTotal"] = str(sum(w[2] for w in wins.values()))
    numbers["RobCells"] = str(sum(sum(w) for w in wins.values()))

    if os.path.isdir(args.diag):
        with open(os.path.join(args.out, "diagnostic.tex"), "w") as fh:
            fh.write(diagnostic_table(
                args.diag, 50,
                "Held-out predictive quality of GPs that differ only in their metric, on candidates drawn as "
                "\\lzero{} screens them: mean Spearman correlation / negative log predictive density over four "
                "random splits, 50 training and 150 test candidates (100 for MNIST, Pendulum, Pinwheel and "
                "Hopper), Matérn-5/2 profile. $\\hat b$: fitted decay of the decay family. Best Spearman "
                "correlation in bold.",
                "tab:diagnostic"))
        numbers.update(diagnostic_numbers(args.diag))
    write_numbers(os.path.join(os.path.dirname(args.out.rstrip("/")), "numbers.tex"), numbers)
    print(f"wrote the tables in {args.out} and {len(numbers)} numbers")


if __name__ == "__main__":
    main()
