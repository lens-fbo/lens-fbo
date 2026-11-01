"""Write a campaign list, or run one chunk of it.

    python scripts/campaign.py make c1 --methods l0mo ours-spectral --targets hartmann --seeds 0-15
    python scripts/campaign.py work campaigns/c1.txt --chunk 0 --n-chunks 8 --workers 32

`make` writes campaigns/<name>.txt. Each line is a run name and the arguments of scripts/run.py;
runs land in out/runs/<name>/<run>/. `work` runs every line of its chunk whose metrics.json is
missing, `workers` at a time, each pinned to one core and single-threaded, all sharing one
persistent JAX compilation cache. On a Slurm cluster, one array task per chunk:

    sbatch --array=0-7 --cpus-per-task=32 --mem=200G --wrap \\
      "python scripts/campaign.py work campaigns/c1.txt --n-chunks 8 --workers 32"

(the chunk defaults to $SLURM_ARRAY_TASK_ID and the workers to $SLURM_CPUS_PER_TASK; allow about
5.5 GB per worker, 8 GB on MNIST). Without a scheduler, run `work` with --chunk 0 --n-chunks 1.
"""

import argparse
import itertools
import os
import platform
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def make(args) -> None:
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from fbo import benchmarks

    lines = []
    for method, target, seed in itertools.product(args.methods, args.targets, seeds(args.seeds)):
        ell_sel, prof_sel = benchmarks.SELECTED[target]
        for profile in args.profiles or [prof_sel]:
            for ell in args.lengthscales or [ell_sel]:
                name = f"{method}__{target}__{profile}__l{ell:g}__s{seed}"
                extra = " ".join(args.extra)
                lines.append(f"{name}\t--method {method} --target {target} --profile {profile} "
                             f"--lengthscale {ell:g} --seed {seed} {extra}".rstrip())
    os.makedirs(os.path.join(ROOT, "campaigns"), exist_ok=True)
    path = os.path.join(ROOT, "campaigns", f"{args.name}.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{path}: {len(lines)} runs")


def single_thread_env() -> dict:
    env = dict(os.environ)
    # compiled executables bake in host cpu features, so key the cache on the instruction set
    isa = "AVX512" if "avx512f" in open("/proc/cpuinfo").read() else "AVX2"
    # FBO_JAX_CACHE moves the cache off a slow shared home (a network home made runs 40x slower)
    base = os.environ.get("FBO_JAX_CACHE") or os.path.join(ROOT, "out", ".jaxcache")
    if base == "local":  # the job's node-local temporary directory ($TMPDIR)
        base = os.environ.get("TMPDIR", "/tmp")
    cache = os.path.join(base, f"{platform.machine()}_{isa}")
    os.makedirs(cache, exist_ok=True)
    env.update(
        XLA_FLAGS=f"--xla_cpu_max_isa={isa} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        JAX_COMPILATION_CACHE_DIR=cache,
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0.2",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NPY_NUM_THREADS="1",
    )
    return env


def work(args) -> None:
    campaign = os.path.splitext(os.path.basename(args.list))[0]
    lines = [l for l in open(args.list).read().splitlines() if l.strip()]
    chunk = args.chunk if args.chunk is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    mine = lines[chunk :: args.n_chunks]
    todo = []
    for line in mine:
        name, cli = line.split("\t", 1)
        run_dir = os.path.join(ROOT, "out", "runs", campaign, name)
        if not os.path.exists(os.path.join(run_dir, "metrics.json")):
            todo.append((name, cli, run_dir))
    print(f"chunk {chunk}/{args.n_chunks}: {len(mine)} runs, {len(todo)} to do, {args.workers} at a time", flush=True)
    env = single_thread_env()
    # each run is pinned to one allocated core: an unpinned XLA process sizes its thread pools to
    # the whole node, and dozens of them then contend for the job's cores
    cores = queue.Queue()
    for c in sorted(os.sched_getaffinity(0)):
        cores.put(c)

    def one(item):
        name, cli, run_dir = item
        os.makedirs(run_dir, exist_ok=True)
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run.py"), *cli.split(), "--run-dir", run_dir]
        core = cores.get()
        try:
            with open(os.path.join(run_dir, "log.txt"), "a") as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                       preexec_fn=lambda: os.sched_setaffinity(0, {core}))
        finally:
            cores.put(core)
        print(f"{'ok' if code == 0 else f'FAILED ({code})'}  {name}", flush=True)
        return code

    with ThreadPoolExecutor(args.workers) as pool:
        codes = list(pool.map(one, todo))
    failed = sum(c != 0 for c in codes)
    print(f"chunk {chunk}: {len(codes) - failed} ok, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("name")
    m.add_argument("--methods", nargs="+", required=True)
    m.add_argument("--targets", nargs="+", required=True)
    m.add_argument("--profiles", nargs="+", default=None, help="default: the source's selection for L0MO")
    m.add_argument("--lengthscales", nargs="+", type=float, default=None, help="default: the source's")
    m.add_argument("--seeds", default="0-15")
    m.add_argument("--extra", nargs="*", default=[], help="further arguments of scripts/run.py")
    w = sub.add_parser("work")
    w.add_argument("list")
    w.add_argument("--chunk", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    w.add_argument("--n-chunks", type=int, default=1)
    w.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = ap.parse_args()
    make(args) if args.cmd == "make" else work(args)


if __name__ == "__main__":
    main()
