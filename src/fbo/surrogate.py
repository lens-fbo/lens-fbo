"""A Gaussian process on functions whose metric is learned on the domain D.

Prior FBO surrogates compare two candidates by their RKHS distance, a norm that weighs frequencies
exponentially and counts the part of each atom that lies outside D. Here candidates are compared
through what the objective can see of them: their values on a quadrature grid of D. The squared
distance is a weighted L2(D) distance in a cosine basis,

    d2(u, u') = sum_{l, j} theta_lj (c_lj - c'_lj)^2 + sum_j theta_perp_j |(I - P)(u_j - u'_j)|^2,

with c_lj the coefficient of output j on basis function l (total degree deg(l) <= J) and P the
projection on the basis, so the second term is the L2(D) distance beyond degree J. The weights
follow a spectral decay with per-coefficient deviations,

    log theta_lj = a_j + b deg(l) + delta_lj,    log theta_perp_j = a_j + b (J + 1) + delta_perp_j,

and the deviations are shrunk toward zero by a N(0, tau^2) prior (MAP). With b = 0 and delta = 0
the metric is the isotropic L2(D) distance.

FunctionalMetricGP.fit returns a model whose .posterior is the drop-in for
bofus.gp.FunctionalPosterior: bofus's acquisition code calls posterior.predict([f]) and reads
posterior.y_best, so L0MO's search runs unchanged on top of this surrogate.
"""

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from bofus import gp as bgp
from bofus import kernels, rkhs
from jaxtyping import Array, Float

from . import features

jax.config.update("jax_enable_x64", True)
EPS = float(np.sqrt(np.finfo(float).eps))
NUGGET_RANGE = (EPS, 1e2)

# the families compared in the paper, from the isotropic L2(D) metric to the full spectral one
METRICS = ("l2", "decay", "spectral")


class Grid(eqx.Module):
    """Quadrature grid of D with a cosine basis, orthonormal for the midpoint rule."""

    z: Float[Array, "n d"]
    phi: Float[Array, "n L"]
    degree: Float[Array, "L"]
    max_degree: int = eqx.field(static=True)

    @classmethod
    def build(cls, d: int, total_points: int = 1024, max_degree: int | None = None) -> Self:
        max_degree = {1: 12, 2: 6, 3: 4, 4: 3, 5: 2}.get(d, 2) if max_degree is None else max_degree
        n_axis = max(features.grid_size(d, total_points), max_degree + 1)
        z = features.midpoint_grid(d, n_axis)
        phi, degrees = features.cosine_basis(z, max_degree)
        return cls(jnp.asarray(z), jnp.asarray(phi), jnp.asarray(degrees.sum(-1), float), max_degree)

    def values(self, f: rkhs.Function) -> Float[Array, "n m"]:
        return (f.kernel(self.z, f.x) @ f.a).reshape(len(self.z), -1)

    def coefficients(self, U: Float[Array, "... n m"]) -> Float[Array, "... L m"]:
        return jnp.einsum("...nm,nl->...lm", U, self.phi) / len(self.z)


class Metric(eqx.Module):
    """Log weights of the domain metric: per-coefficient log theta (L, m) and beyond-basis (m,)."""

    log_theta: Float[Array, "L m"]
    log_theta_perp: Float[Array, "m"]


def unpack(params: Float[Array, "p"], family: str, grid: Grid, m: int) -> tuple[Metric, Float[Array, ""]]:
    """Map the flat parameter vector of a family to the metric and the log nugget."""
    L = len(grid.degree)
    a = params[:m]
    if family == "l2":
        b, delta, delta_perp = 0.0, jnp.zeros((L, m)), jnp.zeros(m)
    elif family == "decay":
        b, delta, delta_perp = params[m], jnp.zeros((L, m)), jnp.zeros(m)
    elif family == "spectral":
        b = params[m]
        delta = params[m + 1 : m + 1 + L * m].reshape(L, m)
        delta_perp = params[m + 1 + L * m : m + 1 + L * m + m]
    else:
        raise ValueError(f"unknown metric family {family}")
    log_theta = a[None, :] + b * grid.degree[:, None] + delta
    log_theta_perp = a + b * (grid.max_degree + 1) + delta_perp
    return Metric(log_theta, log_theta_perp), params[-1]


def n_params(family: str, L: int, m: int) -> int:
    return {"l2": m, "decay": m + 1, "spectral": m + 1 + L * m + m}[family] + 1


def log_prior(params: Float[Array, "p"], family: str, L: int, m: int, tau: float) -> Float[Array, ""]:
    """Gaussian shrinkage of the deviations from the spectral decay, and a weak N(0, 1) on b."""
    if family == "l2":
        return jnp.zeros(())
    lp = -0.5 * params[m] ** 2
    if family == "spectral":
        delta = params[m + 1 : m + 1 + L * m + m]
        lp = lp - 0.5 * jnp.sum(delta**2) / tau**2
    return lp


class Features(eqx.Module):
    """What the metric needs of a batch of functions: coefficients, grid values and energies."""

    C: Float[Array, "N L m"]
    U: Float[Array, "N n m"]
    e: Float[Array, "N m"]

    @classmethod
    def of(cls, grid: Grid, U: Float[Array, "N n m"]) -> Self:
        return cls(C=grid.coefficients(U), U=U, e=jnp.mean(U**2, axis=1))


def sq_distances(metric: Metric, f1: Features, f2: Features) -> Float[Array, "N1 N2"]:
    """Weighted L2(D) squared distances between two batches of functions."""
    theta, theta_perp = jnp.exp(metric.log_theta), jnp.exp(metric.log_theta_perp)
    n = f1.U.shape[1]
    # total L2(D) distance per output, then split into basis part and the rest
    total = f1.e[:, None, :] + f2.e[None, :, :] - 2 * jnp.einsum("anm,bnm->abm", f1.U, f2.U) / n
    dc2 = (f1.C[:, None] - f2.C[None, :]) ** 2  # (N1, N2, L, m)
    basis = jnp.sum(dc2, axis=2)
    weighted = jnp.einsum("ablm,lm->ab", dc2, theta)
    beyond = jnp.einsum("abm,m->ab", jnp.maximum(total - basis, 0.0), theta_perp)
    return jnp.maximum(weighted + beyond, 0.0)


class MetricPosterior(eqx.Module):
    """Everything predict needs, padded to a power-of-two number of observations."""

    profile: kernels.Profile
    grid: Grid
    metric: Metric
    scale: Float[Array, ""]  # features are divided by this so the median L2 distance is one
    nu: Float[Array, ""]
    b: Float[Array, ""]
    y_best: Float[Array, ""]
    obs: Features
    mask: Float[Array, "N"]
    ys: Float[Array, "N"]
    Koo_lu: tuple
    Koo_inv_sum: Float[Array, ""]

    def features(self, fs: list[rkhs.Function]) -> Features:
        U = jnp.stack([self.grid.values(f) for f in fs]) / self.scale
        return Features.of(self.grid, U)

    @eqx.filter_jit
    def predict(self, fs: list[rkhs.Function]) -> bgp.Gaussian:
        x = self.features(fs)
        Kxx = self.nu * self.profile(sq_distances(self.metric, x, x))
        Kox = self.nu * self.profile(sq_distances(self.metric, self.obs, x)) * self.mask[:, None]
        return bgp.gp_posterior(
            Kxx, Kox, None, self.ys, self.b, Koo_lu=self.Koo_lu, Koo_inv_sum=self.Koo_inv_sum
        )


def masked_covariance(K: Float[Array, "N N"], g, mask: Float[Array, "N"]) -> Float[Array, "N N"]:
    """Padded rows and columns become an inert identity block."""
    return K * mask[:, None] * mask[None, :] + jnp.diag(g * mask + (1 - mask))


class FunctionalMetricGP(eqx.Module):
    """GP surrogate on functions with a learned domain metric (see the module docstring)."""

    profile: kernels.Profile = eqx.field(static=True)
    family: str = eqx.field(static=True, default="spectral")
    tau: float = eqx.field(static=True, default=1.0)
    grid: Grid | None = None
    params: Float[Array, "p"] | None = None
    posterior: MetricPosterior | None = None

    def fit(self, fs: list[rkhs.Function], ys: Float[Array, "n"]) -> Self:
        grid = self.grid if self.grid is not None else Grid.build(fs[0].kernel.d)
        m, L = fs[0].kernel.m, len(grid.degree)
        n, N = len(ys), bgp.padded_to(len(ys))
        mask = jnp.zeros(N).at[:n].set(1.0)
        padded_ys = jnp.zeros(N).at[:n].set(ys)

        U = jnp.stack([grid.values(f) for f in fs])
        # one unit of distance is the median L2(D) distance between observations
        total = Features.of(grid, U)
        d2 = sq_distances(Metric(jnp.zeros((L, m)), jnp.zeros(m)), total, total)
        pairs = d2[jnp.triu_indices(n, 1)] if n > 1 else jnp.ones(1)
        med = jnp.median(pairs[pairs > 0]) if jnp.any(pairs > 0) else 1.0
        scale = jnp.sqrt(med)
        U = jnp.zeros((N, *U.shape[1:])).at[:n].set(U / scale)
        obs = Features.of(grid, U)

        # precompute the pieces of the distance that do not depend on the weights
        nz = U.shape[1]
        total = obs.e[:, None, :] + obs.e[None, :, :] - 2 * jnp.einsum("anm,bnm->abm", obs.U, obs.U) / nz
        dc2 = (obs.C[:, None] - obs.C[None, :]) ** 2
        beyond = jnp.maximum(total - jnp.sum(dc2, axis=2), 0.0)

        # bounds on the overall log weight a: hetGP-style range of the L2(D) lengthscale
        auto = bgp.auto_lengthscale_range(self.profile, jnp.sum(total, -1)[:n, :n])
        lo, hi = (np.log(1.0 / auto[1]), np.log(1.0 / auto[0])) if auto is not None else (-6.0, 6.0)
        family, tau, profile = self.family, self.tau, self.profile

        bounds = [(lo, hi)] * m
        if family in ("decay", "spectral"):
            bounds += [(-3.0, 3.0)]
        if family == "spectral":
            bounds += [(-6.0, 6.0)] * (L * m + m)
        bounds += [tuple(np.log(NUGGET_RANGE))]

        x0 = np.zeros(len(bounds))
        x0[:m] = 0.5 * (lo + hi)
        x0[-1] = np.log(0.1)
        starts = [x0]
        if self.params is not None and len(self.params) == len(x0):
            starts.append(np.clip(np.asarray(self.params), [b[0] for b in bounds], [b[1] for b in bounds]))

        result = fit_metric(dc2, beyond, padded_ys, mask, profile, family, tau, grid, m, starts, bounds)
        metric, log_g = unpack(jnp.asarray(result), family, grid, m)
        K = profile(weighted_sq(metric, dc2, beyond))
        Koo = masked_covariance(K, jnp.exp(log_g), mask)
        _, b, nu = bgp.loglikelihood(Koo, padded_ys, mask)
        scaled = nu * Koo * mask[:, None] * mask[None, :] + jnp.diag(1 - mask)
        posterior = MetricPosterior(
            profile=profile,
            grid=grid,
            metric=metric,
            scale=scale,
            nu=nu,
            b=b,
            y_best=ys.min(),
            obs=obs,
            mask=mask,
            ys=padded_ys,
            Koo_lu=jsp.linalg.lu_factor(scaled),
            Koo_inv_sum=(jnp.linalg.inv(scaled) * mask[:, None] * mask[None, :]).sum(),
        )
        return eqx.tree_at(
            lambda s: (s.grid, s.params, s.posterior),
            self,
            (grid, jnp.asarray(result), posterior),
            is_leaf=lambda x: x is None,
        )

    def predict(self, fs: list[rkhs.Function]) -> bgp.Gaussian:
        return self.posterior.predict(fs)


def weighted_sq(metric: Metric, dc2: Float[Array, "N N L m"], beyond: Float[Array, "N N m"]) -> Float[Array, "N N"]:
    theta, theta_perp = jnp.exp(metric.log_theta), jnp.exp(metric.log_theta_perp)
    return jnp.einsum("ablm,lm->ab", dc2, theta) + jnp.einsum("abm,m->ab", beyond, theta_perp)


@eqx.filter_jit
def map_loss(params, dc2, beyond, ys, mask, profile, family, tau, grid, m):
    metric, log_g = unpack(params, family, grid, m)
    Koo = masked_covariance(profile(weighted_sq(metric, dc2, beyond)), jnp.exp(log_g), mask)
    loglik = bgp.loglikelihood(Koo, ys, mask)[0]
    L = len(grid.degree)
    return -(loglik + log_prior(params, family, L, m, tau))


map_value_and_grad = eqx.filter_jit(eqx.filter_value_and_grad(map_loss))


def fit_metric(dc2, beyond, ys, mask, profile, family, tau, grid, m, starts, bounds) -> np.ndarray:
    """MAP estimate of the metric and nugget: L-BFGS-B from each start, best kept."""

    def fun(p):
        val, grad = map_value_and_grad(jnp.asarray(p), dc2, beyond, ys, mask, profile, family, tau, grid, m)
        val, grad = float(val), np.asarray(grad)
        if not np.isfinite(val) or not np.all(np.isfinite(grad)):
            return 1e10, np.zeros_like(p)
        return val, grad

    best = None
    for x0 in starts:
        res = sp.optimize.minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                                   options=dict(maxiter=200, ftol=EPS, gtol=0.0))
        if best is None or res.fun < best.fun:
            best = res
    return best.x
