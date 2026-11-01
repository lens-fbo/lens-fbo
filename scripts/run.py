"""Run one (method, benchmark, configuration, seed) and write its run directory.

    python scripts/run.py --method ours-spectral --target hartmann --seed 0 --run-dir out/runs/NAME

The run directory gets config.json, env.json and, once the run finishes, metrics.json and
observations.npz. A run whose metrics.json exists is skipped, and an interrupted run resumes from
its checkpoint (state.pkl), so a campaign can be resubmitted as it is.
"""

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from importlib import metadata

import numpy as np


def git_commit() -> dict:
    """The commit the code came from: .git_commit in a tree copied without its .git, else git."""
    if os.path.exists(".git_commit"):
        return dict(line.strip().split("=", 1) for line in open(".git_commit") if "=" in line)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return dict(commit=commit, dirty=str(dirty).lower())
    except (OSError, subprocess.CalledProcessError):
        return {}


def environment() -> dict:
    versions = {}
    for pkg in ["jax", "jaxlib", "numpy", "scipy", "equinox", "bofus", "jaxvlse", "gymnasium", "torch"]:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    cpu = ""
    if os.path.exists("/proc/cpuinfo"):
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    env_keys = ["SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_CPUS_PER_TASK", "XLA_FLAGS",
                "JAX_COMPILATION_CACHE_DIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    return dict(
        python=sys.version.split()[0],
        versions=versions,
        host=socket.gethostname(),
        platform=platform.platform(),
        cpu=cpu,
        env={k: os.environ[k] for k in env_keys if k in os.environ},
        git=git_commit(),
        argv=sys.argv,
    )


def main() -> None:
    from fbo import benchmarks, runner

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", required=True, choices=runner.METHODS)
    ap.add_argument("--target", required=True, choices=benchmarks.BENCHMARKS)
    ap.add_argument("--lengthscale", type=float, help="RKHS lengthscale; default: the source's per benchmark")
    ap.add_argument("--profile", choices=list(benchmarks.PROFILES),
                    help="GP kernel profile; default: the one the source selected for L0MO")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tau", type=float, default=1.0, help="prior scale of the spectral deviations")
    ap.add_argument("--max-degree", type=int, default=None, help="cosine basis degree J; default: per dimension")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--initial-acquisitions", type=int, default=10)
    ap.add_argument("--minimum-k", type=int, default=1)
    ap.add_argument("--maximum-k", type=int, default=10)
    ap.add_argument("--acquisitions-each-k", type=int, default=10)
    ap.add_argument("--acquisition-raw-samples", type=int, default=1024)
    ap.add_argument("--acquisition-max-restarts", type=int, default=16)
    args = ap.parse_args()

    selected_ell, selected_profile = benchmarks.SELECTED[args.target]
    config = dict(vars(args))
    config["lengthscale"] = args.lengthscale or selected_ell
    config["profile"] = args.profile or selected_profile
    config.pop("run_dir")

    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"{metrics_path} exists, nothing to do")
        return
    with open(os.path.join(args.run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    with open(os.path.join(args.run_dir, "env.json"), "w") as fh:
        json.dump(environment(), fh, indent=1)

    started = time.time()
    out = runner.run(
        method=args.method,
        target_name=args.target,
        lengthscale=config["lengthscale"],
        profile=config["profile"],
        seed=args.seed,
        state_path=os.path.join(args.run_dir, "state.pkl"),
        tau=args.tau,
        max_degree=args.max_degree,
        initial_acquisitions=args.initial_acquisitions,
        minimum_k=args.minimum_k,
        maximum_k=args.maximum_k,
        acquisitions_each_k=args.acquisitions_each_k,
        acquisition_raw_samples=args.acquisition_raw_samples,
        acquisition_max_restarts=args.acquisition_max_restarts,
    )

    ys = np.asarray(out["observation_values"], dtype=float)
    best = np.minimum.accumulate(ys)
    metrics = dict(
        ys=ys.tolist(),
        final=float(best[-1]),
        average=float(best.mean()),
        n_evaluations=len(ys),
        surrogate_fit_time=float(out.get("surrogate_fit_time", 0.0)),
        acquisition_time=float(out.get("acquisition_time", 0.0)),
        target_evaluation_time=float(out.get("target_evaluation_time", 0.0)),
        wall_time=float(out["wall_time"]),
        resumed=bool(out["resumed"]),
        started=started,
        finished=time.time(),
        max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # Linux reports KiB
    )
    if out.get("surrogate_params") is not None:
        metrics["surrogate_params"] = out["surrogate_params"]
        metrics["surrogate_layout"] = out.get("surrogate_layout")
    locations = out["observation_locations"]
    if hasattr(locations[0], "x"):
        kmax = max(len(f.x) for f in locations)
        d, m = locations[0].x.shape[1], np.asarray(locations[0].a).reshape(len(locations[0].x), -1).shape[1]
        X, A = np.zeros((len(ys), kmax, d)), np.zeros((len(ys), kmax, m))
        for i, f in enumerate(locations):
            k = len(f.x)
            X[i, :k], A[i, :k] = np.asarray(f.x), np.asarray(f.a).reshape(k, m)
        np.savez_compressed(os.path.join(args.run_dir, "observations.npz"), x=X, a=A, y=ys)
    else:
        C = np.stack([np.asarray(f.as_degree(args.maximum_k).c) for f in locations])
        np.savez_compressed(os.path.join(args.run_dir, "observations.npz"), c=C, y=ys)
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=1)
    state = os.path.join(args.run_dir, "state.pkl")
    if os.path.exists(state):
        os.remove(state)
    print(f"final {metrics['final']:.6g}  average {metrics['average']:.6g}  wall {metrics['wall_time']:.0f}s")


if __name__ == "__main__":
    main()
