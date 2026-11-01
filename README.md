# LeNS: a learned domain metric for Bayesian optimization over function spaces

Functional Bayesian optimization searches a space of functions u: D → R, such as a control
signal, a curve or an activation function, for one that minimizes an expensive objective f(u).
The methods of the RKHS line, from Vien et al. (2018) to L0MO (Sartor et al., 2026), represent
candidates as kernel expansions, and their Gaussian-process surrogate compares two candidates by
their RKHS distance.

LeNS, the learned-norm surrogate, replaces that distance with a metric learned on the domain.
Candidates are compared through their values on a quadrature grid of D, by a weighted L2(D)
distance in a cosine basis. The weights follow a spectral decay, free in sign, with a deviation
per basis function, and they are fitted with the GP by maximum a posteriori estimation, the
deviations shrunk toward the decay. Everything else (the candidate space, the acquisition
function, its optimizer and the schedule) is that of L0MO, so the comparison isolates the
surrogate.

This repository holds the implementation, the scripts that run the experiments, the outputs of
every run, and the scripts that rebuild the tables and figures of the paper from those outputs.

## Contents

```
src/fbo/          the package: the surrogate (surrogate.py) and its features (features.py), runs
                  with checkpoints (runner.py), the benchmarks and their configurations
                  (benchmarks.py), the statistics (results.py, selection.py)
scripts/          run.py (one run), campaign.py (a list of runs), make_tables.py, make_figures.py,
                  diag_surrogate.py (held-out predictive quality of the metrics),
                  extract_source_tables.py, get_bofus.sh
campaigns/        the list of runs of every campaign
results/runs/     config.json, metrics.json and env.json of every run
results/diag/     the outputs of the surrogate diagnostics
paper/            the tables (LaTeX), the figures (PDF) and numbers.tex, built from results/
data/             the numbers reported by Sartor et al. (2026), read from their tables
tests/
```

## Installation

Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/) and git. The experiments ran on Linux
x86-64, on the CPU only.

```bash
bash scripts/get_bofus.sh    # bofus, the code of Sartor et al. (2026), at the commit used here
uv sync --extra dev
uv run pytest
```

bofus provides the baselines, the benchmarks and the acquisition machinery; it is fetched from
its own repository into `third_party/bofus`. The numerical packages (JAX, NumPy, SciPy and the
others) are pinned to the versions of bofus's own lockfile, since the benchmark functions are
defined with them. torch is installed only to read the MNIST files.

## A first run

```bash
uv run python scripts/run.py --method ours-spectral --target sinc1d --seed 0 \
    --maximum-k 2 --acquisitions-each-k 5 --run-dir out/example
```

runs LeNS on the Sinc-1D benchmark for 20 evaluations (10 initial candidates, then 5 with one
atom and 5 with two) in about half a minute, and prints

```
final 0.0189608  average 0.0912278  wall 24s
```

the best value found, the average regret over the run and the wall time; the wall time depends
on the machine, and the last digits of the values can differ on another CPU. `out/example/` then
holds `config.json`, `env.json`, `metrics.json` and `observations.npz`. Without the two schedule
options, the run follows the schedule of the paper: 110 evaluations, with candidates of 1 to 10
atoms.

The methods are `ours-spectral` (LeNS), its ablations `ours-decay` and `ours-l2`, and the
baselines `l0mo`, `vien`, `shilton`, `kundu`, `vellanky` and `random`. `--target` takes any of the
15 benchmarks; `src/fbo/benchmarks.py` lists them with the RKHS lengthscale and GP profile
selected for each by Sartor et al. (2026), which are the defaults.

## Rebuilding the tables and figures

From the shipped runs:

```bash
uv run python scripts/make_tables.py --out paper/tables
uv run python scripts/make_figures.py convergence --methods ours-spectral --reported l0mo \
    --out paper/figures/convergence.pdf
uv run python scripts/make_figures.py decay --out paper/figures/decay.pdf
```

`make_tables.py` writes every table and `paper/numbers.tex`, the numbers quoted in the text:

| file | content |
|---|---|
| `main_final.tex`, `main_average.tex` | final value and average regret of LeNS and of the baselines on the 15 benchmarks |
| `robustness.tex` | LeNS at one GP profile against L0MO at its best profile, at each RKHS lengthscale |
| `ablation.tex` | the three families of metrics |
| `profiles.tex` | LeNS at each GP profile |
| `cost.tex` | surrogate-fit and acquisition time per run |
| `sensitivity.tex` | LeNS with other prior scales and basis degrees |
| `diagnostic.tex` | held-out predictive quality of GPs that differ only in their metric |

`convergence.pdf` shows the running best value of LeNS on each benchmark, and `decay.pdf` the
spectral decay LeNS has fitted at the end of each run. The baselines' numbers are those Sartor et
al. (2026) report, read from `data/sartor2026_tables.json`; everything else comes from the runs.
The scripts read `results/runs/` and `results/diag/`, or the directories in `$FBO_RUNS` and
`$FBO_DIAG`. The tables and figures pool every campaign but the timing study (c8) and the
sensitivity study (c9), which have tables of their own.

## Rerunning the experiments

Each file in `campaigns/` lists one run per line: its name, then the arguments of
`scripts/run.py`. `campaign.py work` runs the lines whose `metrics.json` is missing, each on one
core and single-threaded, and writes them to `out/runs/<campaign>/<run>/`. An interrupted run
resumes from its checkpoint, so a campaign can be restarted as it is. On one machine:

```bash
uv run python scripts/campaign.py work campaigns/c1.txt --workers 8
```

On a Slurm cluster, as an array of chunks:

```bash
sbatch --array=0-7 --cpus-per-task=32 --mem=200G --wrap \
  ".venv/bin/python scripts/campaign.py work campaigns/c1.txt --n-chunks 8 --workers 32"
```

Then build the tables from the new runs, into another directory:
`FBO_RUNS=out/runs uv run python scripts/make_tables.py --out out/tables`. MNIST needs its data
first: `uv run python -m fbo.benchmarks --fetch-mnist` downloads it to `out/data` (or
`$FBO_DATA`).

| campaign | runs | content |
|---|---|---|
| c1 | 528 | the three LeNS families on the 11 inexpensive benchmarks, at the lengthscale and profile selected for L0MO, seeds 0–15 |
| c2 | 1584 | the same at the other three GP profiles |
| c3b | 32 | L0MO on Pinwheel at its reported configuration, seeds 0–31 |
| c5h, c5p | 384, 192 | the three LeNS families at every profile on Pendulum and Hopper, and on Pinwheel, seeds 0–15 |
| c6a, c6b | 464, 144 | LeNS at Matérn-5/2 at every lengthscale |
| c7a, c10 | 128, 80 | LeNS, seeds 16–31, at the profile selected for it, and at Matérn-5/2 on Pendulum, Pinwheel and Hopper |
| c8 | 88 | timing: L0MO and LeNS side by side on the same machines, seeds 0–3 |
| c9 | 256 | LeNS with other prior scales τ and basis degrees J, on four benchmarks |
| c11 | 288 | MNIST: the three LeNS families at every profile (LeNS with seeds 0–31, the others 0–15), and L0MO at its reported configuration, seeds 0–31 |

Each run uses one CPU core and at most 6 GB of memory. The 4,168 runs took about 2,600
core-hours, 2,200 of them on MNIST, where every evaluation trains five networks. On the AMD EPYC
7763 processors most of them ran on, a run takes a median of 5 to 11 minutes, depending on the
benchmark, and at most 16 minutes, while an MNIST run takes 3.5 to 10 hours.

## Data

The benchmarks are those of Sartor et al. (2026), defined in bofus. MNIST, which one benchmark
trains a network on, is downloaded by torchvision; its copyright holders, Yann LeCun and Corinna
Cortes, make it available under the Creative Commons Attribution-Share Alike 3.0 license, and it
is not included here. Pendulum and Hopper are Gymnasium environments. `data/sartor2026_tables.json`
holds the numbers reported in the tables of Sartor et al. (2026, arXiv:2610.07417), from which the
tables take the baselines; `scripts/extract_source_tables.py` extracts it from the LaTeX source of
their paper, placed in `sources/arXiv-2610.07417v1/`.

## License

MIT (see `LICENSE`). bofus, which this code imports and adapts in one place (the MNIST benchmark
in `src/fbo/benchmarks.py`), is MIT-licensed and fetched from its own repository.
