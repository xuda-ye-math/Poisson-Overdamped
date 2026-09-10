"""Exact construction of the coarse noise from a fine Brownian grid.

On the fine grid of step dt each subinterval carries the pair

    dB_i = B_{t_{i+1}} - B_{t_i}                        = sqrt(dt) xi_i,
    dA_i = int_{t_i}^{t_{i+1}} ( B_s - B_{t_i} ) ds     = dt^{3/2} ( xi_i/2 + eta_i/(2 sqrt 3) ),

with xi_i, eta_i independent standard Gaussians.  This is the exact joint law
of the increment and its time integral, so no discretization of the Brownian
path is involved anywhere below: the coarse quantities are exact functionals
of the same path the reference solution uses.

Every routine is jitted.  The prefix sums that the aggregation needs are the
largest arrays in the study, and keeping them inside a compiled region lets
XLA free them at the end of the call instead of holding them live in Python.
"""

from functools import partial

import jax
import jax.numpy as jnp

SQRT3 = jnp.sqrt(3.0)


@partial(jax.jit, static_argnums=(1, 2))
def fine_noise(key, n_fine, n_paths, dt):
    """Return (dB, dA), each of shape (n_fine, n_paths, 2)."""
    k1, k2 = jax.random.split(key)
    xi = jax.random.normal(k1, (n_fine, n_paths, 2))
    eta = jax.random.normal(k2, (n_fine, n_paths, 2))
    dB = jnp.sqrt(dt) * xi
    dA = dt**1.5 * (0.5 * xi + eta / (2.0 * SQRT3))
    return dB, dA


@partial(jax.jit, static_argnums=(2,))
def coarsen(dB, dA, n_steps, dt):
    """Aggregate the fine pair onto `n_steps` coarse steps, exactly.

    Over a coarse step made of the fine subintervals i = 0..m-1,

        DB = sum_i dB_i,
        DA = sum_i [ (B_{t_i} - B_{kh}) dt + dA_i ],

    the inner difference being the exclusive prefix sum of dB in the block.
    """
    n_fine, n_paths, dim = dB.shape
    m = n_fine // n_steps
    assert m * n_steps == n_fine, (n_fine, n_steps)
    b = dB.reshape(n_steps, m, n_paths, dim)
    a = dA.reshape(n_steps, m, n_paths, dim)
    incl = jnp.cumsum(b, axis=1)
    return incl[:, -1], jnp.sum((incl - b) * dt + a, axis=1)


@partial(jax.jit, static_argnums=(2,))
def midpoint_noise(key, dB, n_steps, dt, h):
    """Brownian increment up to a uniformly random point inside each step.

    The random time is drawn uniformly from the fine grid points of the step,
    kh + j dt with j = 1..m, so zeta = j dt / h; this is the discrete
    counterpart of the uniform variable of the scheme, exact for the path.
    """
    n_fine, n_paths, dim = dB.shape
    m = n_fine // n_steps
    incl = jnp.cumsum(dB.reshape(n_steps, m, n_paths, dim), axis=1)
    j = jax.random.randint(key, (n_steps, n_paths), 0, m)
    idx = jnp.broadcast_to(j[:, None, :, None], (n_steps, 1, n_paths, dim))
    DBmid = jnp.take_along_axis(incl, idx, axis=1)[:, 0]
    zeta = ((j + 1).astype(dB.dtype) * dt / h)[..., None]
    return DBmid, zeta
