"""Averages of the test functions under the invariant measure, by quadrature.

The dynamics has invariant density

    pi(x)  proportional to  exp( -U(x) )                    on R^2,

so for each test function of observables.py

    pi(f) = int f(x) exp(-U(x)) dx / int exp(-U(x)) dx ,

which is what a long-time average of f along a trajectory converges to.  Both
integrals are taken by the midpoint rule on one uniform grid, so the cell area
cancels and pi(f) is a weighted mean of f over the grid.

Outside the wells U(x) grows like |x|^2/2, so the density is Gaussian-tailed
and a box of half-width L = 10 leaves a mass of order exp(-L^2/2) = 2e-22
outside.  The grid is N x N with N = 10000, a spacing of 0.002, and the whole
sweep is a hundred blocks of a hundred rows so that no array of 10^8 points is
ever held at once.

Run once; the result is saved and read by the plot.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # potential.py, observables.py

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from potential import potential
from observables import OBSERVABLES

L = 10.0            # half-width of the box
N_GRID = 10000      # points per axis
N_BLOCK = 100       # rows per block
OUT = os.path.join(HERE, "artifacts", "pi_observables.npz")


@jax.jit
def block_sums(x1, x2):
    """Unnormalized weight and weighted f sums over one block of rows."""
    pts = jnp.stack(jnp.meshgrid(x1, x2, indexing="ij"), axis=-1)   # (b, N, 2)
    w = jnp.exp(-potential(pts))
    return w.sum(), jnp.stack([(w * f(pts)).sum() for _, f in OBSERVABLES])


def compute(n_grid=N_GRID, half_width=L, n_block=N_BLOCK):
    """pi(f) for every observable, on an n_grid x n_grid midpoint grid."""
    edges = jnp.linspace(-half_width, half_width, n_grid + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    wsum = 0.0
    fsum = np.zeros(len(OBSERVABLES))
    for i in range(0, n_grid, n_block):
        w, f = block_sums(centres[i:i + n_block], centres)
        wsum += float(w)
        fsum += np.asarray(f)
    return fsum / wsum, wsum


def main():
    t0 = time.time()
    names = [n for n, _ in OBSERVABLES]
    values, wsum = compute()
    seconds = time.time() - t0
    print(f"{N_GRID} x {N_GRID} grid on [-{L:g}, {L:g}]^2, "
          f"spacing {2*L/N_GRID:.4g}, {seconds:.1f}s on {jax.devices()[0].platform}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, names=np.array(names), values=values,
                        n_grid=N_GRID, half_width=L, seconds=seconds)
    print(f"wrote {OUT}\n")
    for n, v in zip(names, values):
        print(f"  pi({n:8s}) = {v:+.8f}")


if __name__ == "__main__":
    main()
