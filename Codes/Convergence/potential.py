"""Two dimensional test potential for the strong error study.

    U(x) = a/2 |x|^2 - sum_i c_i exp( -|x - p_i|^2 / (2 s_i^2) )

A quadratic bowl with three Gaussian bumps carved into it.  The bumps leave
two wells, at (-1.2103, -0.4831) and (1.0766, 0.3814), separated by a saddle at
(-0.1151, 0.0988); the third bump is too weak against the bowl to carve a well
of its own.  They make U nonconvex on a bounded region and give the dynamics
two metastable states: nabla^2 U is positive definite at every bump centre, its
most negative eigenvalue is -2.9123 at (-0.617, 0.583), on the barrier, and no
eigenvalue is negative beyond |x| = 3.671.  Outside that region the Gaussians
and all their derivatives are exponentially small, so nabla^2 U -> a I.  Every
derivative of U is bounded, so the potential satisfies the assumption of the
manuscript: convex outside a bounded region, with bounded derivatives of every
order.
"""

import jax.numpy as jnp

A_QUAD = 1.0

# (amplitude c_i, centre p_i, width s_i)
BUMPS = (
    (3.0, (-1.5, -0.6), 0.8),
    (2.5, (1.6, 0.4), 0.9),
    (2.0, (0.1, 1.7), 0.7),
)

X0 = jnp.array([-1.5, -0.6])   # deterministic initial state, the centre of the deepest bump
T_FINAL = 1.0


def potential(x):
    """U(x) for x of shape (..., 2)."""
    out = 0.5 * A_QUAD * jnp.sum(x**2, axis=-1)
    for c, p, s in BUMPS:
        d2 = jnp.sum((x - jnp.asarray(p)) ** 2, axis=-1)
        out = out - c * jnp.exp(-d2 / (2.0 * s**2))
    return out


def grad_potential(x):
    """nabla U(x) for x of shape (..., 2), returned with the same shape."""
    out = A_QUAD * x
    for c, p, s in BUMPS:
        pv = jnp.asarray(p)
        d = x - pv
        d2 = jnp.sum(d**2, axis=-1, keepdims=True)
        out = out + c * jnp.exp(-d2 / (2.0 * s**2)) * d / s**2
    return out


def hessian_potential(x):
    """nabla^2 U(x) for x of shape (..., 2), returned with shape (..., 2, 2)."""
    eye = jnp.eye(2)
    out = A_QUAD * jnp.broadcast_to(eye, x.shape + (2,))
    for c, p, s in BUMPS:
        pv = jnp.asarray(p)
        d = x - pv
        d2 = jnp.sum(d**2, axis=-1)[..., None, None]
        e = jnp.exp(-d2 / (2.0 * s**2))
        outer = d[..., :, None] * d[..., None, :]
        out = out + c * e * (eye / s**2 - outer / s**4)
    return out
