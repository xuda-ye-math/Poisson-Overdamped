"""Strong and weak error of the integrators of Table 1 at equal cost.

Cost is the number of grad U evaluations used to reach the horizon T, taken
from C = 3 * 4^j, which is divisible by 1, 2, 3 and 4, so a scheme costing e
evaluations per step runs exactly N = C / e steps and rounding favours nobody.

For every (method, cost, seed, path) this script stores

    msd      = E | Z_k - X_{kh} |^2             at every k (the strong error)
    mabs     = E | Z_k - X_{kh} |                at every k
    m4       = E | Z_k - X_{kh} |^4              at every k
    supk     = E max_{j<=k} | Z_j - X_{jh} |^2   at every k
    zsq      = E | Z_k |^2                       at every k
    sup_err  = max_{0<=k<=N} | Z_k - X_{kh} |    (the pathwise sup over [0,T])
    term_err = | Z_N - X_T |
    zN       = Z_N                               (terminal state of the scheme)

and, once per (seed, path), xT = X_T from the reference.  Both plots are made
from this file alone: the strong plot from msd, the weak plot from zN and xT,
so the list of test functions can be changed without recomputing.  The strong
error reported is max_k ( E |Z_k - X_{kh}|^2 )^{1/2}, the quantity the strong
convergence theorem bounds, read off msd.  The other per-step arrays are kept
so that the error can be followed along the interval rather than only summarized
by one number: supk gives the larger ( E max_{j<=k} |...|^2 )^{1/2} at every k,
so the two orderings of maximum and expectation can be compared step by step,
m4 gives the fourth moment the local estimates are stated in, and zsq the
moment bound the weights rely on.

The reference is SRK-LD (strong order 3/2, three stages), which is not one of
the compared methods, run on a fine grid that every coarse grid divides.

Memory.  A chunk of paths carries the fine pair (dB, dA) of shape
(N_REF, CHUNK, 2), and every quantity derived from it is larger still.  The
whole per-method measurement is therefore one jitted call: XLA assigns and
frees the intermediates inside the executable, and only the reduced outputs,
which are smaller by a factor N_REF, ever reach Python.  Each of those is
copied to host memory and the device buffer dropped as soon as it is made, so
a worker's footprint is flat across the 32 chunks of a seed and the 16
workers fit in memory at once.
"""

import os
import sys
import time
from functools import lru_cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # the common modules, one level up

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from potential import X0, T_FINAL
from noise import fine_noise, coarsen, midpoint_noise
from integrators import METHODS, srk_ld, srk_ld_blocks
from observables import OBSERVABLES

J_LIST = [1, 2, 3, 4, 5, 6]
COSTS = [3 * 4**j for j in J_LIST]
N_COARSE = max(COSTS)          # every N in the study divides this
N_REF = 49152                  # fine steps of the reference
# The reference only needs to be far more accurate than the smallest error
# being measured.  The strong error here is the sup over [0,T], whose smallest
# value in this study is of order 1e-5, while the reference self-gap at this
# resolution is of order 5e-7, contributing about 0.2 percent in quadrature.
# For the weak error what matters is the reference's own weak bias, smaller
# still.  N_COARSE divides N_REF, as the coarsening requires.
N_SEEDS = 16                   # one worker process per seed
N_PATHS = 4096                 # paths per seed
CHUNK = 128                    # paths held in memory at once
BASE_SEED = 20260829


@lru_cache(maxsize=None)
def measure(name, n_steps):
    """The whole measurement for one (method, step count), compiled once.

    Returns the per-path strong error, the terminal error and state, and the
    observable sums over this chunk's paths at every step of the scheme's own
    grid.  Every intermediate of size (n_steps+1, CHUNK, 2) lives and dies
    inside the executable.
    """
    fn = METHODS[name][0]
    stride = N_COARSE // n_steps

    @jax.jit
    def go(z0, h, DB, DA, extra, ref):
        Z = fn(z0, h, DB, DA, extra, traj=True)            # (n_steps+1, paths, 2)
        Xk = ref[::stride]                                 # same grid
        d = jnp.sqrt(jnp.sum((Z - Xk) ** 2, axis=-1))      # (n_steps+1, paths)
        # sums over this chunk's paths, so the seed's means can be
        # accumulated across chunks before any difference is taken
        fz = jnp.stack([f(Z).sum(axis=1) for _, f in OBSERVABLES], axis=1)
        fx = jnp.stack([f(Xk).sum(axis=1) for _, f in OBSERVABLES], axis=1)
        # per-step sums over this chunk's paths, so every seed mean below
        # accumulates across chunks before any maximum or root is taken
        d2 = d ** 2
        run = jax.lax.cummax(d2, axis=0)          # max_{j<=k} |Z_j - X_jh|^2
        return (d.max(axis=0), d[-1], Z[-1], fz, fx,
                d.sum(axis=1), d2.sum(axis=1), (d2 ** 2).sum(axis=1),
                run.sum(axis=1), (Z ** 2).sum(axis=(1, 2)))

    return go


def host(*xs):
    """Copy to host memory and release the device buffers."""
    out = tuple(np.asarray(x) for x in xs)
    for x in xs:
        x.delete()
    return out


def run_chunk(key, n_paths, want_gap, methods):
    """One block of paths: reference, then every (method, cost) on it."""
    dt = T_FINAL / N_REF
    k1, key = jax.random.split(key)
    dB, dA = fine_noise(k1, N_REF, n_paths, dt)
    z0 = jnp.broadcast_to(X0, (n_paths, 2))

    ref = srk_ld_blocks(z0, dt, dB, dA, N_COARSE)     # (N_COARSE+1, paths, 2)
    xT, = host(ref[-1])

    gap = None
    if want_gap:
        DBc, DAc = coarsen(dB, dA, N_REF // 4, dt)
        rc = srk_ld(z0, T_FINAL / (N_REF // 4), DBc, DAc)
        gap = float(jnp.sqrt(jnp.mean(jnp.sum((ref[-1] - rc) ** 2, -1))))
        del DBc, DAc, rc

    out = {}
    for name in methods:
        evals = METHODS[name][1]
        for C in COSTS:
            n_steps = C // evals
            h = T_FINAL / n_steps
            DB, DA = coarsen(dB, dA, n_steps, dt)
            extra = None
            if name == "leimkuhler_matthews":
                key, k = jax.random.split(key)
                extra = jnp.sqrt(h) * jax.random.normal(k, (n_paths, 2))
            elif name == "randomized_midpoint":
                key, k = jax.random.split(key)
                extra = midpoint_noise(k, dB, n_steps, dt, h)
            elif name.startswith("random_splitting"):
                key, k = jax.random.split(key)
                extra = jax.random.bernoulli(k, 0.5, (n_steps, n_paths, 1))
            out[(name, C)] = host(*measure(name, n_steps)(z0, h, DB, DA, extra, ref))
            del DB, DA, extra
    del dB, dA, ref
    return out, xT, gap


def run_seed(seed_index, shards, methods=None):
    """One seed, written to its own shard so the seeds can run in parallel.

    `methods` restricts the run to those schemes; the driving noise and the
    reference are replayed from the seed's key exactly as in a full run, so a
    method added later is measured on the same Brownian paths.
    """
    methods = methods or list(METHODS)
    t0 = time.time()
    sup, term, zN, fsum_z, fsum_x = {}, {}, {}, {}, {}
    # per-step means, each an array over k: the mean absolute error, the mean
    # square error, its fourth moment, the running maximum of the mean square
    # over steps, and the chain's own second moment
    step = {t: {} for t in ("mabs", "msd", "m4", "supk", "zsq")}
    xT = np.zeros((N_PATHS, 2))
    gaps = []
    key = jax.random.PRNGKey(BASE_SEED + seed_index)
    for c0 in range(0, N_PATHS, CHUNK):
        key, k = jax.random.split(key)
        sl = slice(c0, c0 + CHUNK)
        res, x, gap = run_chunk(k, CHUNK, want_gap=(c0 == 0), methods=methods)
        if gap is not None:
            gaps.append(gap)
        xT[sl] = x
        for kk, (a, b, z, fz, fx, s1, s2, s4, sr, sz) in res.items():
            sup.setdefault(kk, np.zeros(N_PATHS))[sl] = a
            term.setdefault(kk, np.zeros(N_PATHS))[sl] = b
            zN.setdefault(kk, np.zeros((N_PATHS, 2)))[sl] = z
            fsum_z[kk] = fsum_z.get(kk, 0.0) + fz
            fsum_x[kk] = fsum_x.get(kk, 0.0) + fx
            for tag, v in (("mabs", s1), ("msd", s2), ("m4", s4),
                           ("supk", sr), ("zsq", sz)):
                step[tag][kk] = step[tag].get(kk, 0.0) + v
        del res
        print(f"    seed {seed_index}: {c0 + CHUNK}/{N_PATHS} paths "
              f"({time.time()-t0:.1f}s, {rss_gb():.2f} GB)", flush=True)

    payload = {"xT": xT, "reference_self_gap": float(np.mean(gaps)),
               "seed": BASE_SEED + seed_index,
               "obs_names": np.array([n for n, _ in OBSERVABLES])}
    for (n, C) in sup:
        payload[f"sup|{n}|{C}"] = sup[(n, C)]
        payload[f"term|{n}|{C}"] = term[(n, C)]
        payload[f"zN|{n}|{C}"] = zN[(n, C)]
        # E f_i(Z_k) - E f_i(X_kh) at every step of this scheme's own grid
        payload[f"wdiff|{n}|{C}"] = (fsum_z[(n, C)] - fsum_x[(n, C)]) / N_PATHS
        # every per-step mean, as a function of the step index k
        for tag in step:
            payload[f"{tag}|{n}|{C}"] = step[tag][(n, C)] / N_PATHS
    out = os.path.join(shards, f"seed_{seed_index:02d}.npz")
    np.savez_compressed(out, **payload)
    print(f"  seed {seed_index} done in {time.time()-t0:.1f}s, "
          f"peak {peak_gb():.2f} GB -> {out}", flush=True)


def rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def peak_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


if __name__ == "__main__":
    run_seed(int(sys.argv[1]), sys.argv[2], sys.argv[3:] or None)
