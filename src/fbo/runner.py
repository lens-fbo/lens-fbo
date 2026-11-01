"""One FBO run: bofus's runners, with a checkpoint after every evaluation.

The methods of Sartor et al. (2026) are bofus's own Runner subclasses; ours is bofus.run.Ours (the
L0MO search: sparse candidates, LHS screening, multi-start L-BFGS-B, growing k) with the surrogate
replaced by FunctionalMetricGP. Everything else, the schedule included, is shared.
"""

import os
import pickle
import time

import jax.numpy as jnp
import numpy as np
import scipy as sp
from bofus import gp as bgp
from bofus import rkhs
from bofus import run as brun

from . import benchmarks
from .surrogate import METRICS, FunctionalMetricGP, Grid

SOURCE_METHODS = ("l0mo", "random", "vien", "kundu", "shilton", "vellanky")
OUR_METHODS = tuple(f"ours-{m}" for m in METRICS)
METHODS = SOURCE_METHODS + OUR_METHODS


def make_runner_class(method: str):
    """bofus's runner class for the method, extended with checkpointing."""
    base = {
        "l0mo": brun.Ours,
        "random": brun.Random,
        "vien": brun.Vien,
        "kundu": brun.Kundu,
        "shilton": brun.Shilton,
        "vellanky": brun.Vellanky,
    }.get(method, brun.Ours)
    return type(f"Checkpointed{base.__name__}", (Checkpointed, base), {})


def make_surrogate(method: str, profile_name: str, tau: float = 1.0, d: int = 1, max_degree: int | None = None):
    profile = benchmarks.PROFILES[profile_name]()
    if method == "random":
        return None
    if method == "vellanky":
        return bgp.GaussianProcess(profile=profile)
    if method in OUR_METHODS:
        grid = Grid.build(d, max_degree=max_degree) if max_degree is not None else None
        return FunctionalMetricGP(profile=profile, family=method.removeprefix("ours-"), tau=tau, grid=grid)
    return bgp.FunctionalGaussianProcess(profile=profile)


def method_kwargs(method: str) -> dict:
    # the source's main results run L0MO with the Euclidean gradient (App. F); so do we, for ours
    if method == "l0mo" or method in OUR_METHODS:
        return dict(use_natural_gradient=False)
    return {}


class Checkpointed:
    """Mixin over a bofus Runner: same schedule, state saved after every evaluation."""

    state_path: str | None = None

    def fit(self) -> None:
        super().fit()
        # the fitted metric of our surrogate after every fit: what geometry the data choose
        params = getattr(self.surrogate_model, "params", None)
        if params is not None:
            if not hasattr(self, "surrogate_params"):
                self.surrogate_params = []
            self.surrogate_params.append(np.asarray(params).tolist())

    def _save(self, stage: tuple[int, int]) -> None:
        if self.state_path is None:
            return
        # scipy's LHS engines draw from children spawned off self.rng, so the generator is pickled
        # whole (its spawn counter included) together with the engines and their own streams
        samplers = {k: v for k, v in vars(self).items() if isinstance(v, sp.stats.qmc.QMCEngine)}
        state = dict(
            fs=[_to_numpy(f) for f in self.fs],
            ys=np.asarray(self.ys),
            rng=self.rng,
            samplers=samplers,
            timings=dict(self.timings),
            stage=stage,
            surrogate_params=getattr(self, "surrogate_params", None),
        )
        tmp = self.state_path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh)
        os.replace(tmp, self.state_path)

    def _load(self) -> tuple[int, int] | None:
        if self.state_path is None or not os.path.exists(self.state_path):
            return None
        with open(self.state_path, "rb") as fh:
            state = pickle.load(fh)
        self.fs = [_from_numpy(f, self.kernel) for f in state["fs"]]
        self.ys = jnp.asarray(state["ys"])
        self.rng = state["rng"]
        for name, sampler in state["samplers"].items():
            setattr(self, name, sampler)
        self.timings.update(state["timings"])
        if state.get("surrogate_params") is not None:
            self.surrogate_params = state["surrogate_params"]
        self._fit_surrogate()
        return tuple(state["stage"])

    def run(self) -> dict:
        # stages: (0, 0) is the initial design; (k, i) the i-th batch at complexity k
        done = self._load()
        self.resumed = done is not None
        if done is None:
            fs = self._acquire_initial()
            self._record(fs, self._evaluate(fs))
            done = (0, 0)
            self._save(done)
        for k in range(self.minimum_k, self.maximum_k + 1):
            steps = self.steps_per_k(k)
            if (k, steps - 1) <= done:
                continue
            if done[0] != k:  # resuming inside level k keeps the restored sampler
                self.on_new_k(k)
            for i in range(steps):
                if (k, i) <= done:
                    continue
                fs = self._acquire(k)
                self._record(fs, self._evaluate(fs))
                done = (k, i)
                self._save(done)
        return dict(
            observation_locations=self.fs,
            observation_values=self.ys,
            **{f"{stage}_time": t for stage, t in self.timings.items()},
        )


def _to_numpy(f):
    if isinstance(f, rkhs.Function):
        return ("function", np.asarray(f.x), np.asarray(f.a))
    return ("bernstein", np.asarray(f.c))


def _from_numpy(item, kernel):
    if item[0] == "function":
        return rkhs.Function(kernel=kernel, x=jnp.asarray(item[1]), a=jnp.asarray(item[2]))
    return rkhs.BernsteinPolynomial(c=jnp.asarray(item[1]))


def run(
    method: str,
    target_name: str,
    lengthscale: float,
    profile: str,
    seed: int,
    state_path: str | None = None,
    tau: float = 1.0,
    max_degree: int | None = None,
    initial_acquisitions: int = 10,
    minimum_k: int = 1,
    maximum_k: int = 10,
    acquisitions_each_k: int = 10,
    acquisition_raw_samples: int = 1024,
    acquisition_max_restarts: int = 16,
) -> dict:
    target = benchmarks.make_target(target_name)
    kernel = benchmarks.make_kernel(target, lengthscale)
    cls = make_runner_class(method)
    runner = cls(
        seed=seed,
        target_fn=target,
        kernel=kernel,
        surrogate_model=make_surrogate(method, profile, tau, target.d, max_degree),
        initial_acquisitions=initial_acquisitions,
        minimum_k=minimum_k,
        maximum_k=maximum_k,
        acquisitions_each_k=acquisitions_each_k,
        acquisition_raw_samples=acquisition_raw_samples,
        acquisition_max_restarts=acquisition_max_restarts,
        **method_kwargs(method),
    )
    runner.state_path = state_path
    t0 = time.perf_counter()
    out = runner.run()
    out["wall_time"] = time.perf_counter() - t0
    out["resumed"] = runner.resumed
    out["surrogate_params"] = getattr(runner, "surrogate_params", None)
    model = runner.surrogate_model
    if isinstance(model, FunctionalMetricGP) and model.grid is not None:
        # params = [a (m), b (decay, spectral), delta (L m) and delta_perp (m) (spectral), log g]
        out["surrogate_layout"] = dict(family=model.family, m=kernel.m, L=len(model.grid.degree),
                                       max_degree=model.grid.max_degree)
    return out
