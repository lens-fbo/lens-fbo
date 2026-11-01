"""The fifteen benchmarks of Sartor et al. (2026), built exactly as bofus.run builds them."""

import os

import equinox as eqx
import jax.numpy as jnp
import vlse
from bofus import kernels, rkhs, targets

PROFILES = {
    "rbf": lambda: kernels.SquaredExponential(),
    "matern52": lambda: kernels.Matern(nu=5 / 2),
    "matern32": lambda: kernels.Matern(nu=3 / 2),
    "matern12": lambda: kernels.Matern(nu=1 / 2),
}

# RKHS lengthscale per benchmark and the GP profile selected for L0MO (Sartor et al. 2026,
# Table tab:hyperparams; C(inf) = RBF, C(2) = Matern-5/2, C(1) = Matern-3/2, C(0) = Matern-1/2)
SELECTED = {
    "sinc1d": (0.05, "matern32"),
    "sinc2d": (0.1, "rbf"),
    "sinc3d": (0.2, "rbf"),
    "sinc4d": (0.4, "rbf"),
    "sinc5d": (0.4, "rbf"),
    "gramacylee": (0.4, "matern12"),
    "ackley": (0.4, "matern52"),
    "hartmann": (0.4, "matern52"),
    "rosenbrock": (0.4, "matern32"),
    "michalewicz": (0.4, "rbf"),
    "brachistochrone": (0.2, "matern52"),
    "mnist": (0.4, "matern12"),
    "pendulum": (0.2, "matern32"),
    "pinwheel": (0.2, "matern52"),
    "hopper": (0.1, "matern32"),
}

BENCHMARKS = tuple(SELECTED)

# MNIST is read from $FBO_DATA (default out/data); fetch it once, on a machine with internet
# access, with `python -m fbo.benchmarks --fetch-mnist`
DATA_ROOT = os.environ.get("FBO_DATA", "out/data")
MNIST_ATOMS = 10  # the largest k of the schedule: every MNIST candidate is padded to this shape


class MNIST(targets.MNIST):
    """bofus's MNIST target, reading the dataset from DATA_ROOT instead of downloading into ./data.

    The constructor is adapted from bofus (github.com/davidesartor/bofus, MIT license, copyright 2026
    D. Sartor, M. E. Huber, D. Kim, N. Wycoff); only the data location and the padding differ.
    """

    def __init__(self, seed=0, n_runs=5, batch_size=100, width_size=128, depth=2, lr=1e-3, epochs=5):
        import optax
        import torchvision

        assert 10000 % batch_size == 0, "Batch size must divide 10000"
        self.seed, self.batch_size, self.width_size, self.depth = seed, batch_size, width_size, depth
        self.optimizer, self.epochs, self.n_runs = optax.adam(lr), epochs, n_runs
        train = torchvision.datasets.MNIST(root=DATA_ROOT, train=True, download=False)
        test = torchvision.datasets.MNIST(root=DATA_ROOT, train=False, download=False)
        self.train_data = jnp.array(train.data.numpy()) / 255.0
        self.train_labels = jnp.array(train.targets.numpy())
        self.test_data = jnp.array(test.data.numpy()) / 255.0
        self.test_labels = jnp.array(test.targets.numpy())
        self.jit_initialize = eqx.filter_jit(self.initialize)
        self.jit_fit = eqx.filter_jit(self.fit)
        self.jit_test = eqx.filter_jit(self.test)

    def __call__(self, f) -> jnp.ndarray:
        # The training loop is compiled once per shape of the activation, and every compilation
        # embeds the dataset as a constant: a run whose candidates take ten shapes grows past 12 GB.
        # A fixed shape (zero-weight atoms, or a degree-elevated polynomial) is the same activation
        # and compiles once. Training amplifies rounding, so values differ from the unpadded call by
        # about as much as two training seeds do; every method gets the same padded objective.
        if isinstance(f, rkhs.Function) and len(f.x) < MNIST_ATOMS:
            k, pad = len(f.x), MNIST_ATOMS - len(f.x)
            f = rkhs.Function(kernel=f.kernel, x=jnp.concatenate([f.x, jnp.zeros((pad, f.x.shape[1]))]),
                              a=jnp.concatenate([f.a.reshape(k, -1), jnp.zeros((pad, f.a.reshape(k, -1).shape[1]))]))
        elif isinstance(f, rkhs.BernsteinPolynomial) and f.degree < MNIST_ATOMS:
            f = f.as_degree(MNIST_ATOMS)
        return super().__call__(f)


def fetch_mnist() -> None:
    import torchvision

    for train in (True, False):
        torchvision.datasets.MNIST(root=DATA_ROOT, train=train, download=True)
    print(f"MNIST in {DATA_ROOT}")


def make_target(name: str) -> targets.TestFunction:
    return {
        "sinc1d": lambda: targets.SincProjection(d=1),
        "sinc2d": lambda: targets.SincProjection(d=2),
        "sinc3d": lambda: targets.SincProjection(d=3),
        "sinc4d": lambda: targets.SincProjection(d=4),
        "sinc5d": lambda: targets.SincProjection(d=5),
        "gramacylee": lambda: targets.Ridge(vlse.GramacyLee(normalized=True)),
        "ackley": lambda: targets.Ridge(vlse.Ackley(d=2, normalized=True)),
        "hartmann": lambda: targets.Ridge(vlse.Hartmann3(normalized=True)),
        "rosenbrock": lambda: targets.Ridge(vlse.Rosenbrock(d=4, normalized=True)),
        "michalewicz": lambda: targets.Ridge(vlse.Michalewicz(d=5, normalized=True)),
        "pendulum": targets.Pendulum,
        "pinwheel": targets.PinWheel,
        "brachistochrone": targets.Brachistochrone,
        "mnist": MNIST,
        "hopper": targets.HoppingRobot,
    }[name]()


def make_kernel(target: targets.TestFunction, lengthscale: float) -> rkhs.RKHS:
    """The RBF RKHS the candidates live in, with one lengthscale on every input axis."""
    return rkhs.RKHS(
        metric=kernels.Euclidean(),
        profile=kernels.SquaredExponential(),
        rho=jnp.array([lengthscale] * target.d),
        m=getattr(target, "m", 1),
    )


if __name__ == "__main__":
    import sys

    if "--fetch-mnist" in sys.argv:
        fetch_mnist()
