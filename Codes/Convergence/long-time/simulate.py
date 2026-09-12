"""One integrator at one cost, integrated to T and streamed to disk.

The schemes are compared at equal cost, the number of gradient evaluations
spent per unit of time.  Integrator I and stochastic Heun take 2 evaluations
per step, integrator II and the RK3 random splitting take 3, so a cost C fixes
h = 2/C and h = 3/C respectively, and the schemes cover the horizon in
different numbers of steps for the same budget.

Sampling.  The chain is stepped at h but recorded once per unit of time, so
every run stores the same T samples whatever its step size, and the file size
is set by the horizon rather than by the cost.  The stride C / evals is a whole
number for every cost in the grid, so a sample always lands exactly on a step.
A subsampled chain has the same invariant measure, so an average over the
stored samples converges to the same pi(f) that an average over every step
would.

Segments and resuming.  The horizon is cut into segments of SEGMENT samples,
each its own .npy.  A segment is filled block by block, and when it is complete
the run writes a checkpoint holding the chain state, the random key and the
number of finished segments.  Restarting picks that up and carries on, so an
interrupted run loses at most the segment it was in the middle of, never more.
Segment 0 also carries the initial state, so it is one sample longer than the
rest.

    python simulate.py <method> <cost>            carry on where it left off
    python simulate.py <method> <cost> --fresh    begin again at t = 0
    python simulate.py <method> <cost> --from T   carry on from the data at T

The key is deterministic in the seed and the number of blocks consumed, so
`--from` rebuilds it by replay and needs nothing but the trajectory itself.

Memory.  A block's noise and states are released as soon as the block is
flushed, and the segment's map is dropped when the segment closes, so what a
worker holds is set by the block and not by the horizon: the longest run here
is 76.8 billion steps and 25.6 GB on disk, and still integrates in a few
hundred MB of live memory.

Blocking is exact.  The schemes are Markov in the state, so continuing from
the last state of a block with fresh noise is the same chain as one unbroken
sweep, and the same is true across a segment boundary or a restart.
"""

import os
import sys
import time
from functools import lru_cache

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # potential.py, noise.py, integrators.py

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from noise import fine_noise
from integrators import METHODS

SEGMENT = 100_000_000                 # samples per segment, and per save
N_SEGMENTS = 16                       # total horizon T = N_SEGMENTS * SEGMENT
T_FINAL = N_SEGMENTS * SEGMENT
Z0 = jnp.zeros(2)                     # the chain starts at the origin
COSTS = [6 * 2**j for j in range(5)]  # 6, 12, 24, 48, 96 evaluations per unit time
METHOD_NAMES = ["srk1", "srk2", "stochastic_heun", "random_splitting_rk3",
                "euler_maruyama", "leimkuhler_matthews_z"]
WINDOWS = 100_000                     # unit-time windows per block; divides SEGMENT
FLUSH_EVERY = 25                      # blocks between flushes of the segment map
BASE_SEED = 20260831


@lru_cache(maxsize=None)
def block_run(name, windows, stride):
    """One block from a given state: `windows` unit-time windows of `stride`
    steps each, returning the final state and the state at the end of every
    window, shape (windows, 1, 2).

    The scan over windows carries the scheme's own scan over the steps of one
    window, so what XLA compiles is a window and a loop: the compiled size is
    set by the stride, not by the block, and compiling takes about a second
    at any block length.  (Compiling the block as one scan took 571 s for
    300,000 steps under jax 0.11.1, and grows faster than linearly.)  The
    steps run in the same order on the same noise as one unbroken scan, so
    the samples are the same to the last bit.
    """
    fn = METHODS[name][0]

    @jax.jit
    def go(z, h, dB, dA, extra):
        shape = (windows, stride) + dB.shape[1:]
        if extra is not None:                    # the random splitting flags
            extra = extra.reshape((windows, stride) + extra.shape[1:])

        def window(z, d):
            zn = fn(z, h, d[0], d[1], d[2])
            # the original Leimkuhler--Matthews form carries (Z, previous
            # increment); the sample recorded is Z, the first two columns
            return zn, zn[:, :2]

        return jax.lax.scan(window, z,
                            (dB.reshape(shape), dA.reshape(shape), extra))

    return go


def rss_gb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def advance_key(key, n):
    """The key after `n` blocks, replaying the splits the loop would make."""
    for _ in range(n):
        key, _ = jax.random.split(key)
    return key


def segment_path(stem, k):
    return f"{stem}_seg{k:02d}.npy"


def segments_on_disk(stem):
    """How many segments are present and finished, counting from 0.

    A segment is created at full length and filled from the front, so its
    shape says nothing about how far it got: a run stopped part way through
    leaves a file of the right size with an unwritten tail of zeros.  The last
    row is written last, and a real state is never exactly the origin, so that
    row is what distinguishes a finished segment from one in progress.
    """
    k = 0
    while os.path.exists(segment_path(stem, k)):
        rows = SEGMENT + 1 if k == 0 else SEGMENT
        Z = np.load(segment_path(stem, k), mmap_mode="r")
        ok = Z.shape == (rows, 2) and bool(np.any(np.asarray(Z[-1]) != 0.0))
        del Z
        if not ok:
            break
        k += 1
    return k


def state_from_segments(stem, seed, blocks_per_segment, done):
    """Rebuild the chain state and the key after `done` finished segments.

    The key is deterministic in the seed and the number of blocks consumed,
    one split per block, so it can be replayed rather than stored.
    """
    Z = np.load(segment_path(stem, done - 1), mmap_mode="r")
    z = jnp.asarray(np.array(Z[-1]))[None]
    del Z
    key = advance_key(jax.random.PRNGKey(seed), done * blocks_per_segment)
    return z, key


def load_state(stem, seed, blocks_per_segment, start=None, fresh=False):
    """Where to carry on from.

    fresh     begin at t = 0, ignoring whatever is on disk
    start=T   resume from the saved trajectory at time T, a multiple of SEGMENT
    neither   the checkpoint if there is one, else the segments on disk

    The checkpoint is the authority when it exists.  Without one the segments
    on disk are counted, and a segment left half written by an interrupted run
    is not counted, since its last row is still the zeros it was created with.
    Pass `start` to override both and say how far the data is good for.
    """
    if fresh:
        return 0, jnp.broadcast_to(Z0, (1, 2)), jax.random.PRNGKey(seed), 0.0

    if start is not None:
        assert start % SEGMENT == 0, f"start {start} is not a multiple of {SEGMENT}"
        done = start // SEGMENT
        assert 0 <= done <= N_SEGMENTS, (start, done)
        if done == 0:
            return 0, jnp.broadcast_to(Z0, (1, 2)), jax.random.PRNGKey(seed), 0.0
        have = segments_on_disk(stem)
        assert have >= done, (f"asked to resume at {start} but only {have} "
                              f"finished segments on disk")
        z, key = state_from_segments(stem, seed, blocks_per_segment, done)
        return done, z, key, 0.0

    path = stem + "_state.npz"
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as s:
            return (int(s["segments_done"]), jnp.asarray(s["z"]),
                    jnp.asarray(s["key"]), float(s["seconds"]))

    done = segments_on_disk(stem)
    if done:
        seconds = 0.0
        meta = stem + "_meta.npz"
        if os.path.exists(meta):
            with np.load(meta, allow_pickle=False) as d:
                seconds = float(d["seconds"])
        z, key = state_from_segments(stem, seed, blocks_per_segment, done)
        return done, z, key, seconds

    return 0, jnp.broadcast_to(Z0, (1, 2)), jax.random.PRNGKey(seed), 0.0


def save_state(stem, segments_done, z, key, seconds, **meta):
    """Checkpoint, written whole to a temporary file and moved into place."""
    tmp = stem + "_state.tmp.npz"
    np.savez(tmp, segments_done=segments_done, z=np.asarray(z),
             key=np.asarray(key), seconds=seconds, **meta)
    os.replace(tmp, stem + "_state.npz")


def run(name, cost, start=None, fresh=False):
    t0 = time.time()
    evals = METHODS[name][1]
    stride = cost // evals                 # steps per unit of time
    assert stride * evals == cost, (cost, evals)
    assert SEGMENT % WINDOWS == 0, (SEGMENT, WINDOWS)
    h = 1.0 / stride
    blocks = SEGMENT // WINDOWS
    block_steps = WINDOWS * stride
    seed = BASE_SEED + 100 * METHOD_NAMES.index(name) + COSTS.index(cost)

    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    stem = os.path.join(HERE, "artifacts", f"trajectory_{name}_c{cost}")
    done, z, key, seconds = load_state(stem, seed, blocks, start, fresh)
    if name == "leimkuhler_matthews_z" and z.shape[-1] == 2:
        # the increment of the step before t = 0, from a key of its own so the
        # block key sequence, and its replay on resume, is the same for every
        # scheme; a resumed run takes the full state from the checkpoint
        z = jnp.concatenate([z, jnp.sqrt(h) * jax.random.normal(
            jax.random.PRNGKey(seed + 1), (1, 2))], axis=-1)

    print(f"{name} cost {cost}: h = {h:.6g}, stride {stride}, "
          f"{blocks} blocks of {block_steps:,} steps per segment, "
          f"resuming at segment {done}/{N_SEGMENTS}, "
          f"{seconds:.0f}s already spent", flush=True)
    if done >= N_SEGMENTS:
        print(f"{name} cost {cost}: already complete", flush=True)
        return

    for seg in range(done, N_SEGMENTS):
        ts = time.time()
        rows = SEGMENT + 1 if seg == 0 else SEGMENT
        Z = np.lib.format.open_memmap(segment_path(stem, seg), mode="w+",
                                      dtype=np.float64, shape=(rows, 2))
        off = 0
        if seg == 0:
            Z[0] = np.asarray(Z0)
            off = 1

        for b in range(blocks):
            tb = time.time()
            key, k = jax.random.split(key)
            extra = None
            if name == "random_splitting_rk3":
                # the order flags come from a sub-split of the block's key, so
                # the key sequence the resume replays is the same for every
                # scheme: one split per block
                k, kf = jax.random.split(k)
                extra = jax.random.bernoulli(kf, 0.5, (block_steps, 1, 1))
            dB, dA = fine_noise(k, block_steps, 1, h)
            z, samples = block_run(name, WINDOWS, stride)(z, h, dB, dA, extra)
            samples.block_until_ready()
            # one sample per unit of time: the state at the end of each window
            Z[off + b * WINDOWS:off + (b + 1) * WINDOWS] = np.asarray(samples[:, 0])
            del dB, dA, extra, samples   # release the block before the next one
            if seg == done and b == 0:   # the first block carries the compilation
                print(f"    {name} c{cost}: first block done in "
                      f"{time.time() - tb:.1f}s, compilation included "
                      f"({rss_gb():.2f} GB)", flush=True)
            if (b + 1) % FLUSH_EVERY == 0 or b + 1 == blocks:
                Z.flush()                # and push it out of memory to disk
                print(f"    {name} c{cost}: segment {seg} block {b + 1}/{blocks} "
                      f"({time.time() - t0:.0f}s, {rss_gb():.2f} GB)", flush=True)

        Z.flush()
        del Z                            # drop the segment's map before the next
        seconds += time.time() - ts
        save_state(stem, seg + 1, z, key, seconds, h=h, evals=evals, cost=cost,
                   stride=stride, sample_dt=1.0, segment=SEGMENT,
                   n_segments=N_SEGMENTS, T=T_FINAL, seed=seed, name=name)
        finite = bool(np.isfinite(np.asarray(z)).all())
        print(f"{name} cost {cost}: segment {seg} done and saved, "
              f"{(seg + 1) * SEGMENT:,} samples, finite {finite}, "
              f"{time.time() - ts:.0f}s for this segment, "
              f"{seconds:.0f}s total", flush=True)

    print(f"{name} cost {cost}: ALL {N_SEGMENTS} SEGMENTS DONE, "
          f"T = {T_FINAL:,}, {seconds:.0f}s of integration", flush=True)


def parse(argv):
    """<name> <cost> [--fresh] [--from T]"""
    name, cost = argv[0], int(argv[1])
    fresh = "--fresh" in argv
    start = None
    if "--from" in argv:
        start = int(float(argv[argv.index("--from") + 1]))
    assert not (fresh and start is not None), "--fresh and --from are exclusive"
    return name, cost, start, fresh


if __name__ == "__main__":
    nm, c, st, fr = parse(sys.argv[1:])
    run(nm, c, start=st, fresh=fr)
