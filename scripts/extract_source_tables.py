"""Extract the numbers this project uses from the source paper's LaTeX tables into one JSON file.

    python scripts/extract_source_tables.py

The JSON (data/sartor2026_tables.json) carries the reported medians and CIs of Tables
tab:regret_final and tab:regret_average, the medians at every RKHS lengthscale of Table
tab:lengthscale_full, the selected hyperparameters of Table tab:hyperparams, and the power of ten of
each benchmark's column, all from Sartor et al. (2026), arXiv:2610.07417.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fbo import results  # noqa: E402


def main() -> None:
    out = {name: results.source_table(name).to_dict(orient="records")
           for name in ("results_final_ci", "results_avg_ci")}
    out["lengthscale_full"] = results.source_lengthscale_table().to_dict(orient="records")
    out["selection"] = [[t, m, ell, p] for (t, m), (ell, p) in sorted(results.source_selection().items())]
    text = open(os.path.join(results.SOURCE_TABLES, "results_final_ci.tex")).read()
    out["scales"] = {results.SOURCE_NAMES[b]: (-1.0 if sign else 1.0) * 10.0 ** int(e)
                     for b, sign, e in re.findall(r"\\textbf\{([^}]*)\} \[\$(-?)10\^\{(-?\d+)\}\$\]", text)}
    os.makedirs(os.path.dirname(results.SOURCE_JSON), exist_ok=True)
    with open(results.SOURCE_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {results.SOURCE_JSON}")


if __name__ == "__main__":
    main()
