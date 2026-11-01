"""Features of candidate functions: values on a quadrature grid, and coefficients on a cosine basis.

Candidates are sums of Gaussian atoms, u(t) = sum_i a_i exp(-|t - x_i|^2 / (2 l^2)), stored as
padded arrays (x, a) of shape (k, d) and (k, m); a padded atom has a = 0 and contributes nothing.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def midpoint_grid(d: int, n: int) -> Float[np.ndarray, "n**d d"]:
    """Tensor midpoint grid with n points per axis on [0, 1]^d."""
    axis = (np.arange(n) + 0.5) / n
    return np.stack(np.meshgrid(*[axis] * d, indexing="ij"), axis=-1).reshape(-1, d)


def grid_size(d: int, total: int = 2048) -> int:
    """Points per axis so the grid holds about `total` points, never fewer than 4."""
    return max(round(total ** (1 / d)), 4)


def cosine_basis(
    z: Float[np.ndarray, "n d"], max_degree: int
) -> tuple[Float[np.ndarray, "n L"], np.ndarray]:
    """Tensor cosine basis prod_j c_j cos(pi l_j t_j) with |l|_1 <= max_degree, evaluated on z.

    On a midpoint grid with more points per axis than max_degree, the columns are orthonormal for
    the midpoint rule (the DCT-II), so coefficients are plain averages. Columns are sorted by total
    degree, the constant first.
    """
    d = z.shape[1]
    degrees = [l for l in itertools.product(range(max_degree + 1), repeat=d) if sum(l) <= max_degree]
    degrees = np.array(sorted(degrees, key=lambda l: (sum(l), l)))
    scale = np.where(degrees == 0, 1.0, np.sqrt(2.0))
    phi = np.prod(scale[None] * np.cos(np.pi * degrees[None] * z[:, None, :]), axis=-1)
    return phi, degrees


def atom_values(
    x: Float[Array, "k d"], a: Float[Array, "k m"], z: Float[Array, "n d"], lengthscale: Float[Array, "d"]
) -> Float[Array, "n m"]:
    """Values of the Gaussian-atom function (x, a) at the points z."""
    d2 = jnp.sum(((z[:, None, :] - x[None, :, :]) / lengthscale) ** 2, axis=-1)
    return jnp.exp(-0.5 * d2) @ a


batch_atom_values = jax.jit(jax.vmap(atom_values, in_axes=(0, 0, None, None)))
