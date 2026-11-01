"""Held-out predictive quality of functional GP surrogates that differ only in their metric.

Candidates are drawn the way L0MO screens them (k = 1..kmax atoms, LHS over the bounded
parameterisation), the benchmark is evaluated once per candidate (cached under --cache-dir),
then GPs with each metric are fitted on n_train points and scored on a fixed held-out set.

    python scripts/diag_surrogate.py --target hartmann --out out/diag
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from bofus import gp as bgp
from bofus import rkhs

from fbo import benchmarks, features
from fbo.surrogate import FunctionalMetricGP

jax.config.update("jax_enable_x64", True)
EPS = float(np.sqrt(np.finfo(float).eps))


def sample_candidates(kernel, n_per_k: int, kmax: int, rng: np.random.Generator):
    p_dim = kernel.d + kernel.m
    n = kmax * n_per_k
    X, A = np.zeros((n, kmax, kernel.d)), np.zeros((n, kmax, kernel.m))
    fs, i = [], 0
    for k in range(1, kmax + 1):
        for p in sp.stats.qmc.LatinHypercube(d=k * p_dim, rng=rng).random(n_per_k):
            f = rkhs.Function.from_array(kernel, jnp.asarray(p.reshape(k, p_dim)))
            fs.append(f)
            X[i, :k], A[i, :k] = f.x, f.a.reshape(k, kernel.m)
            i += 1
    return fs, X, A


def load_or_sample(target_name: str, lengthscale: float, n_per_k: int, kmax: int, seed: int, cache: str):
    path = f"{cache}/{target_name}_l{lengthscale}_n{n_per_k}x{kmax}_s{seed}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["A"], z["y"], float(z["t_eval"])
    target = benchmarks.make_target(target_name)
    kernel = benchmarks.make_kernel(target, lengthscale)
    fs, X, A = sample_candidates(kernel, n_per_k, kmax, np.random.default_rng(seed))
    t0 = time.perf_counter()
    y = np.array([float(target(f)) for f in fs])
    t_eval = time.perf_counter() - t0
    os.makedirs(cache, exist_ok=True)
    np.savez(path, X=X, A=A, y=y, t_eval=t_eval)
    return X, A, y, t_eval


# --- squared distances between batches of candidates ---------------------------------------


def rkhs_sq(kernel, X1, A1, X2, A2):
    b1 = bgp.Basis(kernel=kernel, x=jnp.asarray(X1), a=jnp.asarray(A1))
    b2 = bgp.Basis(kernel=kernel, x=jnp.asarray(X2), a=jnp.asarray(A2))
    return bgp.rkhs_sq_distances(b1, b2)


@jax.jit
def gauss_inner(X1, A1, X2, A2, lam):
    """sum_pq a_p a'_q exp(-|x_p - x'_q|^2 / (2 lam^2)) for every pair of candidates."""
    d2 = jnp.sum((X1[:, None, :, None, :] - X2[None, :, None, :, :]) ** 2, axis=-1)
    w = jnp.einsum("ipm,jqm->ijpq", A1, A2)
    return jnp.sum(w * jnp.exp(-0.5 * d2 / lam**2), axis=(-2, -1))


def lam_sq(X1, A1, X2, A2, lam, d):
    """Distance under a Gaussian spectral weight: atoms compared at pair lengthscale lam.

    lam = l is the RKHS norm, lam = sqrt(2) l the L2 norm on R^d, larger lam a smoothing norm.
    The (1 / lam)^d factor is the Gaussian integral's; constant factors go into the GP lengthscale.
    """
    n1 = jnp.diagonal(gauss_inner(X1, A1, X1, A1, lam))
    n2 = jnp.diagonal(gauss_inner(X2, A2, X2, A2, lam))
    cross = gauss_inner(X1, A1, X2, A2, lam)
    return jnp.maximum(n1[:, None] + n2[None, :] - 2 * cross, 0.0) / lam**d


def gram_sq(U1, U2):
    """Mean squared difference over the grid (discrete L2), summed over outputs."""
    e1, e2 = jnp.sum(U1**2, -1), jnp.sum(U2**2, -1)
    return jnp.maximum(e1[:, None] + e2[None, :] - 2 * U1 @ U2.T, 0.0)


# --- models: each maps a parameter vector to squared distances, the last entry is log g --


class Iso:
    """One lengthscale on a fixed precomputed distance (the RKHS or the L2 norm)."""

    def __init__(self, name, D):  # D: dict split -> matrix
        self.name, self.D = name, D

    def starts(self, rng, lo, hi):
        mid = 0.5 * (lo + hi)
        return [np.array([mid, np.log(0.1)]), np.array([lo + 0.25 * (hi - lo), np.log(1e-3)])]

    def bounds(self, lo, hi):
        return [(lo, hi), (np.log(EPS), np.log(100.0))]

    def sqdist(self, theta, split):
        return self.D[split] / jnp.exp(theta[0])


class Lam:
    """Learned pair lengthscale lam of the Gaussian spectral family, plus the GP lengthscale."""

    name = "lam"

    def __init__(self, F, ell, d):
        self.F, self.ell, self.d = F, ell, d

    def starts(self, rng, lo, hi):
        mid = 0.5 * (lo + hi)
        return [
            np.array([mid, np.log(self.ell), np.log(0.1)]),
            np.array([mid, np.log(np.sqrt(2) * self.ell), np.log(0.1)]),
            np.array([mid, np.log(4 * self.ell), np.log(0.1)]),
        ]

    def bounds(self, lo, hi):
        # rho's bounds are those of the L2 distance; lam moves the scale, so widen them
        return [(lo - 6, hi + 6), (np.log(0.7 * self.ell), np.log(4.0)), (np.log(EPS), np.log(100.0))]

    def sqdist(self, theta, split):
        (X1, A1), (X2, A2) = self.F[split]
        return lam_sq(X1, A1, X2, A2, jnp.exp(theta[1]), self.d) / jnp.exp(theta[0])


class ARD:
    """One weight per cosine coefficient, plus one for the residual energy beyond the basis."""

    name = "ard"

    def __init__(self, C, R):  # C: split -> (C1, C2); R: split -> residual distance
        self.C, self.R = C, R
        self.L = C["tr"][0].shape[1]

    def starts(self, rng, lo, hi):
        base = np.concatenate([np.zeros(self.L), [0.0], [np.log(0.1)]])
        return [base] + [base + np.r_[rng.normal(0, 1, self.L + 1), 0.0] for _ in range(2)]

    def bounds(self, lo, hi):
        return [(np.log(1e-4), np.log(1e3))] * (self.L + 1) + [(np.log(EPS), np.log(100.0))]

    def sqdist(self, theta, split):
        C1, C2 = self.C[split]
        w = jnp.exp(theta[: self.L])
        diff2 = (C1[:, None, :] - C2[None, :, :]) ** 2
        return jnp.einsum("ijl,l->ij", diff2, w) + jnp.exp(theta[self.L]) * self.R[split]


class LowRank:
    """|W^T (c - c')|^2 + gamma |u - u'|^2: a rank-r projection plus an isotropic residual."""

    def __init__(self, C, D2, r):
        self.C, self.D2, self.r = C, D2, r
        self.L = C["tr"][0].shape[1]
        self.name = f"lowrank{r}"

    def starts(self, rng, lo, hi):
        C = np.asarray(self.C["tr"][0])
        _, _, vt = np.linalg.svd(C - C.mean(0), full_matrices=False)
        pca = vt[: self.r].T * 2.0
        out = []
        for W in [pca, rng.normal(0, 1.0, (self.L, self.r)), rng.normal(0, 1.0, (self.L, self.r))]:
            out.append(np.concatenate([W.ravel(), [np.log(0.1), np.log(0.1)]]))
        return out

    def bounds(self, lo, hi):
        return [(-50.0, 50.0)] * (self.L * self.r) + [(np.log(1e-4), np.log(1e3)), (np.log(EPS), np.log(100.0))]

    def sqdist(self, theta, split):
        C1, C2 = self.C[split]
        W = theta[: self.L * self.r].reshape(self.L, self.r)
        Z1, Z2 = C1 @ W, C2 @ W
        return gram_sq(Z1, Z2) + jnp.exp(theta[-2]) * self.D2[split]


class OracleRidge:
    """ARD GP on the ridge's true linear features b + L[u]: the best any linear metric can do."""

    name = "oracle"

    def __init__(self, Z):
        self.Z = Z
        self.q = Z["tr"][0].shape[1]

    def starts(self, rng, lo, hi):
        return [np.r_[np.zeros(self.q), np.log(0.1)], np.r_[np.full(self.q, 2.0), np.log(1e-3)]]

    def bounds(self, lo, hi):
        return [(np.log(1e-3), np.log(1e4))] * self.q + [(np.log(EPS), np.log(100.0))]

    def sqdist(self, theta, split):
        Z1, Z2 = self.Z[split]
        w = jnp.exp(theta[: self.q])
        return gram_sq(Z1 * jnp.sqrt(w), Z2 * jnp.sqrt(w))


# --- likelihood, fit and prediction --------------------------------------------------------


def concentrated_nll(K, y):
    n = len(y)
    L = jnp.linalg.cholesky(K)
    one = jnp.ones(n)
    Ki1, Kiy = jsp.linalg.cho_solve((L, True), jnp.stack([one, y], 1)).T
    b = jnp.sum(Kiy) / jnp.sum(Ki1)
    r = y - b
    nu = r @ jsp.linalg.cho_solve((L, True), r) / n
    return 0.5 * (n * jnp.log(nu) + 2 * jnp.sum(jnp.log(jnp.diag(L)))), (L, b, nu)


def fit(model, profile, y, rng, lo, hi):
    n = len(y)

    def loss(theta):
        K = profile(model.sqdist(theta, "tr")) + jnp.exp(theta[-1]) * jnp.eye(n)
        val = concentrated_nll(K, y)[0]
        return jnp.where(jnp.isfinite(val), val, 1e10)

    vg = jax.jit(jax.value_and_grad(loss))

    def fun(theta):
        v, g = vg(jnp.asarray(theta))
        g = np.nan_to_num(np.asarray(g), nan=0.0, posinf=0.0, neginf=0.0)
        return float(v), g

    best = None
    for x0 in model.starts(rng, lo, hi):
        bnds = model.bounds(lo, hi)
        x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])
        res = sp.optimize.minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bnds, options=dict(maxiter=300))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


def predict(model, profile, theta, y):
    n = len(y)
    g = jnp.exp(theta[-1])
    K = profile(model.sqdist(theta, "tr")) + g * jnp.eye(n)
    _, (L, b, nu) = concentrated_nll(K, y)
    Kx = profile(model.sqdist(theta, "te"))
    mean = b + Kx @ jsp.linalg.cho_solve((L, True), y - b)
    v = jsp.linalg.solve_triangular(L, Kx.T, lower=True)
    var_f = nu * jnp.maximum(1.0 - jnp.sum(v**2, axis=0), 1e-12)
    return mean, var_f + nu * g


def scores(mean, var, y):
    nlpd = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var
    rmse = np.sqrt(np.mean((y - mean) ** 2))
    rho = sp.stats.spearmanr(mean, y).statistic
    return dict(nlpd=float(np.mean(nlpd)), rmse=float(rmse), spearman=float(rho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--lengthscale", type=float, default=None, help="RKHS lengthscale; default: the source's")
    ap.add_argument("--profile", default="matern52")
    ap.add_argument("--n-per-k", type=int, default=40)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--n-train", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=["rkhs", "l2", "lam", "ard", "lowrank1", "lowrank2", "oracle"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="out/diag/cache")
    ap.add_argument("--out", default="out/diag")
    ap.add_argument("--tag", default=None, help="suffix of the results file")
    args = ap.parse_args()

    ell = args.lengthscale or benchmarks.SELECTED[args.target][0]
    target = benchmarks.make_target(args.target)
    kernel = benchmarks.make_kernel(target, ell)
    d, m = kernel.d, kernel.m
    X, A, y_all, t_eval = load_or_sample(args.target, ell, args.n_per_k, args.kmax, args.seed, args.cache_dir)
    print(f"{args.target}: {len(y_all)} candidates, evaluation {t_eval:.1f}s, l={ell}", flush=True)

    # grid features: values, cosine coefficients, residual energy beyond the basis
    n_axis = features.grid_size(d)
    Z = features.midpoint_grid(d, n_axis)
    max_degree = {1: 12, 2: 6, 3: 4, 4: 3, 5: 2}[d]
    phi, _ = features.cosine_basis(Z, max_degree)
    U = np.asarray(features.batch_atom_values(jnp.asarray(X), jnp.asarray(A), jnp.asarray(Z), kernel.rho))
    nz = len(Z)
    C = np.einsum("nzm,zl->nml", U, phi).reshape(len(U), -1) / nz
    Uflat = U.reshape(len(U), -1) / np.sqrt(nz)
    profile = benchmarks.PROFILES[args.profile]()

    # the ridge's true linear features, for the oracle
    ridge_z = None
    if hasattr(target, "w") and hasattr(target, "b"):
        Wq = np.asarray(target.weights_at(jnp.asarray(Z)))  # (d_ridge, nz)
        ridge_z = np.asarray(target.b)[None] + U[..., 0] @ Wq.T / nz

    rng = np.random.default_rng(args.seed + 1)
    split_rng = np.random.default_rng(args.seed + 2)  # splits do not depend on the models fitted
    results = []
    for rep in range(args.reps):
        perm = split_rng.permutation(len(y_all))
        te = perm[: args.n_test]
        for n_train in args.n_train:
            tr = perm[args.n_test : args.n_test + n_train]
            mu, sd = y_all[tr].mean(), y_all[tr].std() + 1e-12
            ytr, yte = jnp.asarray((y_all[tr] - mu) / sd), (y_all[te] - mu) / sd

            # scale features so the median train L2 distance is one
            D2_tr = np.asarray(gram_sq(Uflat[tr], Uflat[tr]))
            s = np.median(D2_tr[np.triu_indices(n_train, 1)])
            Us, Cs = Uflat / np.sqrt(s), C / np.sqrt(s)
            pairs = {"tr": (tr, tr), "te": (te, tr)}
            D2 = {k: gram_sq(jnp.asarray(Us[i]), jnp.asarray(Us[j])) for k, (i, j) in pairs.items()}
            Dc = {k: gram_sq(jnp.asarray(Cs[i]), jnp.asarray(Cs[j])) for k, (i, j) in pairs.items()}
            R = {k: jnp.maximum(D2[k] - Dc[k], 0.0) for k in pairs}
            Dh = {k: rkhs_sq(kernel, X[i], A[i], X[j], A[j]) for k, (i, j) in pairs.items()}
            Csplit = {k: (jnp.asarray(Cs[i]), jnp.asarray(Cs[j])) for k, (i, j) in pairs.items()}
            Fsplit = {k: ((jnp.asarray(X[i]), jnp.asarray(A[i])), (jnp.asarray(X[j]), jnp.asarray(A[j]))) for k, (i, j) in pairs.items()}

            for name in args.models:
                if name.startswith("fm-"):  # the surrogate the BO runs use, end to end
                    t0 = time.perf_counter()
                    fs = [rkhs.Function(kernel=kernel, x=jnp.asarray(X[i]), a=jnp.asarray(A[i])) for i in range(len(X))]
                    model = FunctionalMetricGP(profile=profile, family=name[3:]).fit([fs[i] for i in tr], ytr)
                    mean, cov = model.predict([fs[i] for i in te])
                    var = np.diag(np.asarray(cov)) + float(model.posterior.nu * np.exp(model.params[-1]))
                    row = dict(target=args.target, rep=rep, n_train=n_train, model=name, nll_train=float("nan"),
                               fit_s=time.perf_counter() - t0, **scores(np.asarray(mean), var, yte))
                    if name != "fm-l2":
                        row["b"] = float(model.params[m])
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    continue
                if name == "rkhs":
                    model, Dref = Iso("rkhs", Dh), Dh["tr"]
                elif name == "l2":
                    model, Dref = Iso("l2", D2), D2["tr"]
                elif name == "lam":
                    model, Dref = Lam(Fsplit, ell, d), lam_sq(*Fsplit["tr"][0], *Fsplit["tr"][1], np.sqrt(2) * ell, d)
                elif name == "ard":
                    model, Dref = ARD(Csplit, R), None
                elif name.startswith("lowrank"):
                    model, Dref = LowRank(Csplit, D2, int(name[7:])), None
                elif name == "oracle":
                    if ridge_z is None:
                        continue
                    zs = ridge_z / ridge_z[tr].std(0)
                    model = OracleRidge({k: (jnp.asarray(zs[i]), jnp.asarray(zs[j])) for k, (i, j) in pairs.items()})
                    Dref = None
                else:
                    raise ValueError(name)
                if Dref is not None:
                    rng_ = bgp.auto_lengthscale_range(profile, jnp.asarray(Dref))
                    lo, hi = np.log(rng_[0]), np.log(rng_[1])
                else:
                    lo, hi = 0.0, 0.0
                t0 = time.perf_counter()
                theta, nll = fit(model, profile, ytr, rng, lo, hi)
                mean, var = predict(model, profile, jnp.asarray(theta), ytr)
                row = dict(target=args.target, rep=rep, n_train=n_train, model=name, nll_train=float(nll) / n_train,
                           fit_s=time.perf_counter() - t0, **scores(np.asarray(mean), np.asarray(var), yte))
                if name == "lam":
                    row["lam_over_ell"] = float(np.exp(theta[1]) / ell)
                results.append(row)
                print(json.dumps(row), flush=True)

    os.makedirs(args.out, exist_ok=True)
    tag = "_".join(sorted(args.models)) if args.tag is None else args.tag
    with open(f"{args.out}/{args.target}_{args.profile}_{tag}.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
