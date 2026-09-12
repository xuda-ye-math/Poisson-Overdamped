"""Driver: run the 16 seeds in parallel, then merge them into
artifacts/trajectories.npz.  One invocation produces everything.

Each worker is confined to one core and to one BLAS/XLA thread, so the 16
processes do not oversubscribe the machine and each seed is an independent
replicate of the whole study.  Worker memory is flat in the number of paths
(see run_experiment.py), so all 16 run at once.

The per-seed shards exist only to carry results between processes.  They are
written to a temporary directory outside the project and removed once the
merge is written, so the only data the folder keeps is the merged file.  The
merge fills preallocated arrays one shard at a time and never holds more than
a single shard beyond the result.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))     # potential.py, noise.py, integrators.py
PY = sys.executable
N_SEEDS = 16
COSTS = [3 * 4**j for j in (1, 2, 3, 4, 5, 6)]
OUT = os.path.join(HERE, "artifacts", "trajectories.npz")
LOGS = os.path.join(HERE, "artifacts", "logs")


def worker_env():
    env = dict(os.environ)
    env.update({
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false "
                     "--xla_force_host_platform_device_count=1",
    })
    return env


def main(cores=None, methods=None):
    """`cores` selects the CPUs to pin the workers to, one seed each, cycling
    if there are fewer cores than seeds.  Useful when part of the machine is
    already busy.  `methods` restricts the run to those schemes; their fresh
    results are merged into the existing output file, the rest kept as saved.
    """
    cores = cores or list(range(N_SEEDS))
    t0 = time.time()
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shards = tempfile.mkdtemp(prefix="poisson_shards_")

    procs = []
    for i in range(N_SEEDS):
        log = open(os.path.join(LOGS, f"seed_{i:02d}.log"), "w")
        core = cores[i % len(cores)]
        cmd = ["taskset", "-c", str(core), PY, "-u",
               os.path.join(HERE, "run_experiment.py"), str(i), shards] \
              + (methods or [])
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env=worker_env()), log))
        print(f"launched seed {i:2d} on core {core}", flush=True)

    failed = []
    for i, p, log in procs:
        rc = p.wait()
        log.close()
        print(f"seed {i:2d} exited {rc}  ({time.time()-t0:.1f}s)", flush=True)
        if rc != 0:
            failed.append(i)
    if failed:
        print(f"FAILED seeds: {failed}; see artifacts/logs/, shards kept in {shards}")
        sys.exit(1)

    merge(shards, subset=bool(methods))
    shutil.rmtree(shards)
    print(f"\ndone in {time.time()-t0:.1f}s")


def merge(shards, subset=False):
    """Fill the merged arrays one shard at a time.

    A subset run's shards carry only its own methods' keys; everything already
    in the output file is kept and the fresh keys are added over it, so a
    method measured later never recomputes the others.
    """
    from integrators import METHODS
    names = list(METHODS)
    path = lambda i: os.path.join(shards, f"seed_{i:02d}.npz")

    out = {}
    if subset:
        with np.load(OUT, allow_pickle=False) as d:
            out = {k: d[k] for k in d.files}

    with np.load(path(0)) as s0:
        keys = [k for k in s0.files if "|" in k] + ["xT"]
        out.update({k: np.empty((N_SEEDS,) + s0[k].shape, s0[k].dtype)
                    for k in keys})
        out.update({
            "costs": np.array(COSTS),
            "evals": np.array([METHODS[n][1] for n in names]),
            "labels": np.array([METHODS[n][2] for n in names]),
            "methods": np.array(names),
            "T": 1.0, "n_ref": 49152, "n_seeds": N_SEEDS,
            "n_paths": s0["xT"].shape[0],
            "obs_names": s0["obs_names"],
        })

    gaps = []
    for i in range(N_SEEDS):
        with np.load(path(i)) as s:
            for k in keys:
                out[k][i] = s[k]
            gaps.append(float(s["reference_self_gap"]))
        print(f"merged seed {i:2d}", flush=True)
    out["reference_self_gap"] = float(np.mean(gaps))

    for n in names:                      # every listed method must carry data,
        if subset and f"msd|{n}|{COSTS[0]}" not in out:
            continue                     # unless it was never run in this study
        assert f"msd|{n}|{COSTS[0]}" in out, f"no data for {n}"

    np.savez_compressed(OUT, **out)
    print(f"merged {N_SEEDS} shards -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    # optional: --cores 0,1,2,5,6 and --methods random_splitting_rk3,...
    argv = sys.argv[1:]
    sel = mets = None
    if "--cores" in argv:
        sel = [int(c) for c in argv[argv.index("--cores") + 1].split(",")]
    if "--methods" in argv:
        mets = argv[argv.index("--methods") + 1].split(",")
    main(sel, mets)
