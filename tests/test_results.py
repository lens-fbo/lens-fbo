import os

import numpy as np
import pandas as pd
import pytest

from fbo import results, selection

HAVE_SOURCE = os.path.isdir(results.SOURCE_TABLES)


def source_median_and_ci(data):
    """The statistic of the source's plots.py (median_and_ci), for comparison."""
    median = np.median(data)
    qu = np.clip(0.5 + 1.96 * np.sqrt(0.25 / len(data)), 0, 1)
    ql = np.clip(0.5 - 1.96 * np.sqrt(0.25 / len(data)), 0, 1)
    return median, np.quantile(data, ql), np.quantile(data, qu)


@pytest.mark.parametrize("n", [4, 16, 32])
def test_median_ci_is_the_sources_statistic(n):
    x = np.random.default_rng(n).lognormal(size=n)
    np.testing.assert_allclose(results.median_ci(x), source_median_and_ci(x))


def test_ranks_follow_the_tie_rule():
    table = pd.DataFrame([
        dict(target="t", method="a", final=1.0, final_lo=0.9, final_hi=1.1),
        dict(target="t", method="b", final=1.05, final_lo=0.95, final_hi=1.2),  # a's median in b's CI
        dict(target="t", method="c", final=2.0, final_lo=1.8, final_hi=2.2),
    ])
    r = selection.ranks(table, "final", ["a", "b", "c"])
    assert r.loc["t"].tolist() == [1, 1, 3]


@pytest.mark.skipif(not HAVE_SOURCE, reason="the source's LaTeX tables are not in this tree")
def test_source_tables_parse():
    final = results.source_table("results_final_ci")
    assert len(final) == 81 and set(final.method) == {"random", "kundu", "vien", "shilton", "vellanky", "l0mo"}
    hop = final[(final.target == "hopper") & (final.method == "l0mo")].iloc[0]
    assert hop["median"] < 0 and hop["lo"] <= hop["median"] <= hop["hi"]
    assert len(results.source_selection()) == 60
    ells = results.source_lengthscale_table()
    assert len(ells) == 220 and sorted(ells.lengthscale.unique()) == [0.05, 0.1, 0.2, 0.4]
    row = ells[(ells.target == "pinwheel") & (ells.method == "l0mo") & (ells.lengthscale == 0.2)].iloc[0]
    assert row["final"] == final[(final.target == "pinwheel") & (final.method == "l0mo")]["median"].iloc[0]


def test_extracted_json_agrees_with_the_latex(tmp_path):
    if not (HAVE_SOURCE and os.path.exists(results.SOURCE_JSON)):
        pytest.skip("needs both the LaTeX sources and the extracted JSON")
    from_latex = results.source_table("results_final_ci")
    from_json = results.source_table("results_final_ci", root=str(tmp_path / "absent"))
    pd.testing.assert_frame_equal(from_latex.reset_index(drop=True), from_json.reset_index(drop=True))
    assert results.source_selection(root=str(tmp_path / "absent"))[("hopper", "l0mo")] == (0.1, "matern32")
    pd.testing.assert_frame_equal(results.source_lengthscale_table().reset_index(drop=True),
                                  results.source_lengthscale_table(root=str(tmp_path / "absent")).reset_index(drop=True))
