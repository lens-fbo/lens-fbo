"""Profile selection and ranks, following the rules of Sartor et al. (2026).

Selection: for each (benchmark, method), the GP profile with the lowest median final value.
Ties (Table tab:lengthscale_ranks of the source): a method ties with any better one whose median
lies within its own 95% CI; its rank is that of the best method it ties with.
"""

import numpy as np
import pandas as pd

from . import benchmarks, results


def per_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Median and CI of final and average per (target, method, profile, lengthscale)."""
    return results.summarize(df, keys=("target", "method", "profile", "lengthscale"))


def select_profiles(summary: pd.DataFrame, metric: str = "final") -> pd.DataFrame:
    """The row of each (target, method, lengthscale) at its best profile."""
    idx = summary.groupby(["target", "method", "lengthscale"])[metric].idxmin()
    return summary.loc[idx].reset_index(drop=True)


def ranks(table: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    """Rank per target with the source's tie rule; rows are targets, columns methods."""
    out = {}
    for target, g in table.groupby("target"):
        g = g.set_index("method")
        present = [m for m in methods if m in g.index]
        order = sorted(present, key=lambda m: g.loc[m, metric])
        rank = {}
        for i, m in enumerate(order):
            rank[m] = i + 1
            lo, hi = g.loc[m, f"{metric}_lo"], g.loc[m, f"{metric}_hi"]
            for j, better in enumerate(order[:i]):
                if lo <= g.loc[better, metric] <= hi:
                    rank[m] = rank[better]
                    break
        out[target] = {m: rank.get(m, np.nan) for m in methods}
    return pd.DataFrame(out).T


def reported_runs(df: pd.DataFrame, families: list[str], select_seeds: int = 16) -> pd.DataFrame:
    """The runs each method is reported at, labeled in a `row` column.

    The source's methods use its per-method selection (Table tab:hyperparams); Vellanki, absent from
    it, and ours use the same rule (lowest median final value over the four profiles) applied to
    their own runs on seeds below `select_seeds`. Ours also appears at Matern-5/2 (`@m52`).
    """
    sel = results.source_selection()
    at_ell = df[df.apply(lambda r: r.lengthscale == benchmarks.SELECTED[r.target][0], axis=1)]
    parts = []
    for (t, m), g in at_ell.groupby(["target", "method"]):
        if (t, m) in sel:
            ell, prof = sel[(t, m)]
            parts.append(g[g.profile == prof].assign(row=m))
        elif m == "random":
            parts.append(g.assign(row=m))
        elif m == "vellanky":
            prof = g[g.seed < select_seeds].groupby("profile").final.median().idxmin()
            parts.append(g[g.profile == prof].assign(row=m))
        elif m in families:
            pick = g[g.seed < select_seeds]
            if len(pick):  # no selection before any selection seed has finished (partial campaigns)
                prof = pick.groupby("profile").final.median().idxmin()
                parts.append(g[g.profile == prof].assign(row=m))
            parts.append(g[g.profile == "matern52"].assign(row=f"{m}@m52"))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]
