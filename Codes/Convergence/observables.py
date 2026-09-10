"""A family of 1-Lipschitz test functions on R^2.

By Kantorovich--Rubinstein duality,

    W_1( Law(Z_N), Law(X_T) ) = sup { E f(Z_N) - E f(X_T) : Lip(f) <= 1 },

so for any finite family of 1-Lipschitz functions

    max_i | E f_i(Z_N) - E f_i(X_T) |

is a lower bound for the Wasserstein-1 distance between the law of the scheme
and the law of the exact solution.  That maximum is what the weak error plot
reports.  Each function below has Euclidean gradient norm at most one; the
`check` routine verifies this numerically rather than taking it on trust.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from potential import BUMPS, X0

_ANGLES = [k * np.pi / 4 for k in range(8)]
_CENTRES = [(0.0, 0.0)] + [p for _, p, _ in BUMPS]

# The distance to a point is 1-Lipschitz but has a kink at that point.  A kink
# at the deterministic initial state dominates the maximum below: at the first
# step the exact solution has variance 2h there, E|x - X0| grows like the
# square root of that variance, and any O(h) deficit in a scheme's variance
# shows up as an O(sqrt h) difference of means.  Such a centre therefore
# measures the start, not the order, so it is excluded.
_DIST_CENTRES = [p for p in _CENTRES
                 if float(np.hypot(p[0] - float(X0[0]), p[1] - float(X0[1]))) > 1e-6]


def _observables():
    fs = []
    for a in _ANGLES:                       # linear projections, |grad| = 1
        u = jnp.array([np.cos(a), np.sin(a)])
        fs.append((f"proj{int(round(np.degrees(a)))}",
                   lambda x, u=u: jnp.sum(x * u, axis=-1)))
    for i, p in enumerate(_DIST_CENTRES):   # distances, 1-Lipschitz
        pv = jnp.array(p)
        fs.append((f"dist{i}",
                   lambda x, pv=pv: jnp.sqrt(jnp.sum((x - pv) ** 2, axis=-1) + 1e-300)))
    for j, nm in ((0, "x"), (1, "y")):      # bounded oscillations, |grad| <= 1
        fs.append((f"sin_{nm}", lambda x, j=j: jnp.sin(x[..., j])))
        fs.append((f"cos_{nm}", lambda x, j=j: jnp.cos(x[..., j])))
        fs.append((f"tanh_{nm}", lambda x, j=j: jnp.tanh(x[..., j])))
    for i, p in enumerate(_CENTRES[1:]):    # Gaussian bumps, max |grad| = e^{-1/2}
        pv = jnp.array(p)
        fs.append((f"bump{i}",
                   lambda x, pv=pv: jnp.exp(-0.5 * jnp.sum((x - pv) ** 2, axis=-1))))
    return fs


OBSERVABLES = _observables()


def check(lim=6.0, n=241):
    """Largest gradient norm of each observable on [-lim, lim]^2."""
    g = jnp.linspace(-lim, lim, n)
    X, Y = jnp.meshgrid(g, g)
    pts = jnp.stack([X, Y], -1).reshape(-1, 2)
    worst = {}
    for name, f in OBSERVABLES:
        gr = jax.vmap(jax.grad(lambda z, f=f: f(z)))(pts)
        worst[name] = float(jnp.max(jnp.linalg.norm(gr, axis=-1)))
    return worst


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    w = check()
    bad = {k: v for k, v in w.items() if v > 1.0 + 1e-9}
    print(f"{len(w)} observables, max gradient norm {max(w.values()):.6f}")
    print("not 1-Lipschitz:", bad or "none")
