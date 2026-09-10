"""Driver: the ten trajectories in parallel, one pinned core each.

Five costs for each of the two integrators, every one an independent chain to
T, each on its own core and its own seed.  There is nothing to merge; each
worker writes its own trajectory and its own metadata.

Every worker streams its trajectory to disk block by block, so the memory the
whole set holds is set by the block size and not by the horizon, and the ten
run together on one machine.

Restartable.  Each worker resumes from its own checkpoint, so running this
again after an interruption carries on where the segments left off rather than
starting over.  The logs are appended to, not truncated, so the record of
earlier attempts survives.  Launch it detached if it is to outlive the shell:

    setsid nohup python long-time/run_long.py >> long-time/artifacts/run.log 2>&1 &
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from simulate import COSTS, METHOD_NAMES

PY = sys.executable
LOGS = os.path.join(HERE, "artifacts", "logs")
RUNS = [(name, cost) for name in METHOD_NAMES for cost in COSTS]


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


def main():
    """Any arguments are passed straight to every worker, so

        python long-time/run_long.py --fresh        start the whole grid again
        python long-time/run_long.py --from 100000000   carry on from that time
    """
    t0 = time.time()
    os.makedirs(LOGS, exist_ok=True)
    procs = []
    for core, (name, cost) in enumerate(RUNS):
        log = open(os.path.join(LOGS, f"{name}_c{cost}.log"), "a")
        cmd = ["taskset", "-c", str(core), PY, "-u",
               os.path.join(HERE, "simulate.py"), name, str(cost)] + sys.argv[1:]
        procs.append(((name, cost),
                      subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                       env=worker_env()), log))
        print(f"launched {name:14s} cost {cost:3d} on core {core}", flush=True)

    failed = []
    for (name, cost), p, log in procs:
        rc = p.wait()
        log.close()
        print(f"{name:14s} cost {cost:3d} exited {rc}  ({time.time()-t0:.1f}s)",
              flush=True)
        if rc != 0:
            failed.append((name, cost))
    if failed:
        print(f"FAILED: {failed}; see artifacts/logs/")
        sys.exit(1)
    print(f"\ndone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
