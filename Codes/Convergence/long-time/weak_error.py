"""The weak error of the time average, computed once over the full data.

For each of the ten runs this sweeps every stored sample and records

    weak error (t) = max_i | 1/(k+1) sum_{j<=k} f_i(Z_j) - pi(f_i) |,   t = k,

over the 1-Lipschitz family of ../observables.py, at a fixed grid of
checkpoints, and writes the curves to artifacts/weak_error.npz.  Plotting
reads that file and nothing else, so the 239 GB of trajectories is touched
only here, and only once.

Memory.  The trajectory is a run of segment files, each memory mapped and read
in blocks of BLOCK samples.  Only the running sum of the test functions, one
number per observable, is carried between blocks; a block is released before
the next is read.  What the process holds is therefore one block, a few tens
of megabytes, whatever the horizon.

The checkpoint spacing is BLOCK, which divides the segment length, so a block
never straddles a checkpoint and every checkpoint is an exact running average
over all samples up to it.  Nothing is subsampled: every stored sample enters
the sums.  Segment 0 carries the initial state as one extra leading sample,
which is counted before the blocks begin.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # observables.py, potential.py
sys.path.insert(0, HERE)                      # simulate.py, for the grid it defines

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from observables import OBSERVABLES
from simulate import COSTS, METHOD_NAMES, SEGMENT, segment_path, segments_on_disk

BLOCK = 400_000                     # samples per block, and the checkpoint spacing
PI = os.path.join(HERE, "artifacts", "pi_observables.npz")
OUT = os.path.join(HERE, "artifacts", "weak_error.npz")


@jax.jit
def block_sums(blk):
    """Sum of every test function over one block of samples."""
    return jnp.stack([f(blk).sum() for _, f in OBSERVABLES])


def rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def sweep(stem, n_seg, pi, label):
    """Running deviation from pi(f) at every checkpoint, over all samples."""
    assert SEGMENT % BLOCK == 0, (SEGMENT, BLOCK)
    per_seg = SEGMENT // BLOCK
    dev = np.empty((n_seg * per_seg, len(pi)))
    acc = np.zeros(len(pi))
    count = 0
    j = 0
    t0 = time.time()
    for k in range(n_seg):
        Z = np.load(segment_path(stem, k), mmap_mode="r")
        off = 0
        if k == 0:                                  # the initial state
            acc += np.asarray(block_sums(np.asarray(Z[0:1])))
            count += 1
            off = 1
        for b in range(per_seg):
            s = off + b * BLOCK
            blk = np.asarray(Z[s:s + BLOCK])
            acc += np.asarray(block_sums(blk))
            count += BLOCK
            dev[j] = np.abs(acc / count - pi)
            j += 1
            del blk
        del Z
        print(f"    {label}: segment {k + 1}/{n_seg} "
              f"({time.time() - t0:.0f}s, {rss_gb():.2f} GB)", flush=True)
    assert j == len(dev) and count == n_seg * SEGMENT + 1, (j, count)
    return dev


SCALARS = ("h", "evals", "cost", "stride", "sample_dt", "seconds", "n_segments")


def main():
    """Sweep every method, or only the ones named on the command line:

        python weak_error.py                    all methods
        python weak_error.py stochastic_heun    this method's runs only

    A partial sweep merges its fresh curves with the ones already in the
    output file, so a method added later never repeats the finished sweeps.
    """
    methods = sys.argv[1:] or list(METHOD_NAMES)
    assert all(m in METHOD_NAMES for m in methods), methods

    p = np.load(PI, allow_pickle=False)
    names = [str(x) for x in p["names"]]
    assert names == [n for n, _ in OBSERVABLES], "pi(f) is stale; rerun invariant.py"
    pi = p["values"]

    saved = {}
    if os.path.exists(OUT):
        with np.load(OUT, allow_pickle=False) as d:
            for i in range(len(d["methods"])):
                saved[(str(d["methods"][i]), int(d["costs"][i]))] = (
                    d["dev"][i], {k: float(d[k][i]) for k in SCALARS})

    t0 = time.time()
    runs, dev, err, meta = [], [], [], {}
    for m in METHOD_NAMES:
        for cost in COSTS:
            label = f"{m} c{cost}"
            if m not in methods:
                if (m, cost) not in saved:
                    continue
                v, scalars = saved[(m, cost)]
                print(f"{label}: kept from the saved sweep", flush=True)
            else:
                stem = os.path.join(HERE, "artifacts", f"trajectory_{m}_c{cost}")
                n_seg = segments_on_disk(stem)
                assert n_seg, f"no finished segments for {m} c{cost}"
                d = np.load(stem + "_state.npz", allow_pickle=False)
                print(f"{label}: {n_seg} segments, "
                      f"{n_seg * SEGMENT:,} samples", flush=True)
                v = sweep(stem, n_seg, pi, label)
                scalars = {k: float(d[k]) for k in SCALARS if k != "n_segments"}
                scalars["n_segments"] = n_seg
                print(f"{label}: weak error at the horizon {v.max(axis=1)[-1]:.4e}, "
                      f"{time.time() - t0:.0f}s so far\n", flush=True)
            runs.append((m, cost))
            dev.append(v)
            err.append(v.max(axis=1))
            for k in SCALARS:
                meta.setdefault(k, []).append(scalars[k])

    n = min(len(e) for e in err)
    out = {
        "t": (np.arange(1, n + 1) * BLOCK).astype(np.float64),
        "err": np.stack([e[:n] for e in err]),
        "dev": np.stack([v[:n] for v in dev]),
        "methods": np.array([m for m, _ in runs]),
        "costs": np.array([c for _, c in runs]),
        "obs_names": np.array(names),
        "pi": pi,
        "block": BLOCK,
        **{k: np.array(v) for k, v in meta.items()},
    }
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB), "
          f"{out['err'].shape[1]} checkpoints per curve, "
          f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
