import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bofus import rkhs
from bofus import run as brun

from fbo import benchmarks, runner
from fbo.surrogate import Features, FunctionalMetricGP, Grid, Metric, sq_distances

jax.config.update("jax_enable_x64", True)


def random_functions(kernel, n, kmax=5, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(1, kmax + 1)
        p = rng.random((k, kernel.d + kernel.m))
        out.append(rkhs.Function.from_array(kernel, jnp.asarray(p)))
    return out


@pytest.mark.parametrize("d", [1, 2, 3])
def test_basis_is_orthonormal_on_the_grid(d):
    grid = Grid.build(d)
    gram = grid.phi.T @ grid.phi / len(grid.z)
    np.testing.assert_allclose(gram, np.eye(len(grid.degree)), atol=1e-10)


def test_unit_weights_give_the_discrete_l2_distance():
    target = benchmarks.make_target("hopper")  # two outputs
    kernel = benchmarks.make_kernel(target, 0.1)
    grid = Grid.build(1)
    fs = random_functions(kernel, 6)
    U = jnp.stack([grid.values(f) for f in fs])
    feats = Features.of(grid, U)
    L, m = len(grid.degree), kernel.m
    d2 = sq_distances(Metric(jnp.zeros((L, m)), jnp.zeros(m)), feats, feats)
    ref = jnp.mean(jnp.sum((U[:, None] - U[None, :]) ** 2, axis=-1), axis=-1)
    np.testing.assert_allclose(d2, ref, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("family", ["l2", "decay", "spectral"])
def test_fit_and_predict_a_smooth_functional(family):
    kernel = benchmarks.make_kernel(benchmarks.make_target("sinc1d"), 0.2)
    grid = Grid.build(1)
    fs = random_functions(kernel, 40)
    w = jnp.cos(np.pi * grid.z[:, 0])
    ys = jnp.array([jnp.mean(w * grid.values(f)[:, 0]) ** 2 for f in fs])
    model = FunctionalMetricGP(profile=benchmarks.PROFILES["matern52"](), family=family).fit(fs[:30], ys[:30])
    mean, cov = model.predict(fs[30:])
    assert np.all(np.isfinite(mean)) and np.all(np.diag(cov) > 0)
    assert np.corrcoef(mean, ys[30:])[0, 1] > 0.8


def test_bofus_acquisition_runs_on_our_posterior():
    kernel = benchmarks.make_kernel(benchmarks.make_target("ackley"), 0.4)
    fs = random_functions(kernel, 12)
    ys = jnp.array([float(jnp.sum(f.a**2)) for f in fs])
    post = FunctionalMetricGP(profile=benchmarks.PROFILES["matern52"]()).fit(fs, ys).posterior
    p = jnp.asarray(np.random.default_rng(1).random(3 * (kernel.d + kernel.m)))
    val, grad = brun.rkhs_loss(p, post, kernel, False)
    assert np.isfinite(val) and np.all(np.isfinite(grad)) and np.any(grad != 0)


def small_run(method, state_path, maximum_k):
    return runner.run(method, "sinc1d", 0.2, "matern52", seed=3, state_path=state_path,
                      initial_acquisitions=3, maximum_k=maximum_k, acquisitions_each_k=2,
                      acquisition_raw_samples=32, acquisition_max_restarts=2)


class Interrupt(Exception):
    pass


@pytest.mark.parametrize("method", ["l0mo", "ours-spectral"])
def test_a_run_interrupted_inside_a_level_resumes(method, tmp_path, monkeypatch):
    state = str(tmp_path / "state.pkl")
    calls = {"n": 0}
    evaluate = brun.Runner._evaluate

    def failing(self, fs):
        calls["n"] += 1
        if calls["n"] == 5:  # initial design (one call), two at k=1, then the second at k=2
            raise Interrupt
        return evaluate(self, fs)

    monkeypatch.setattr(brun.Runner, "_evaluate", failing)
    with pytest.raises(Interrupt):
        small_run(method, state, maximum_k=2)
    monkeypatch.setattr(brun.Runner, "_evaluate", evaluate)
    resumed = small_run(method, state, maximum_k=2)
    assert resumed["resumed"] and len(resumed["observation_values"]) == 3 + 4
    if method == "l0mo":  # no warm start in its surrogate, so resuming changes nothing
        straight = small_run(method, None, maximum_k=2)
        np.testing.assert_allclose(resumed["observation_values"], straight["observation_values"])
