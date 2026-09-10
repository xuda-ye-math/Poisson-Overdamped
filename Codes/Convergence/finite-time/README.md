# Strong and weak error over a finite horizon

Produces the two panels of Figure 1 in `../../../Paper/main.tex`: strong error
$\max_k (\mathbb E |Z_k - X_{kh}|^2)^{1/2}$ and weak error
$\max_i \max_k |\mathbb E f_i(Z_k) - \mathbb E f_i(X_{kh})|$, both against the
number of evaluations of $\nabla U$, at $T = 1$.

The potential, the noise, the integrators and the figure style are shared and
live one level up; see `../README.md`. Run everything from `Convergence/`.

## Files

| file | role |
| --- | --- |
| `observables.py` | the family of 1-Lipschitz test functions; `check()` verifies the constant |
| `run_experiment.py` | one seed: simulates every scheme at every cost, writes a shard |
| `run_all.py` | the 16 seeds in parallel, then the merge; one invocation produces everything |
| `plot_strong.py` | left panel, from `artifacts/trajectories.npz` |
| `plot_weak.py` | right panel, from `artifacts/trajectories.npz` |

## Reproducing

```sh
python finite-time/run_all.py       # -> artifacts/trajectories.npz, artifacts/logs/
python finite-time/plot_strong.py   # -> results/strong_error.png, artifacts/strong_error_summary.csv
python finite-time/plot_weak.py     # -> results/weak_error.png,   artifacts/weak_error_summary.csv
```

`run_all.py` takes about 13 minutes: 16 seeds of 4096 trajectories, every
scheme at every cost `3*4^j` for `j = 1..6`, one worker per core. It writes
`artifacts/trajectories.npz` outright. The plots read that file and simulate
nothing, so either panel rebuilds in seconds.

The two files in `results/` are copied to `../../../Paper/figures/` for the
manuscript.

## Memory

The fine grid carries `(N_REF, CHUNK, 2)` arrays for the Brownian increment
and its time integral, and every quantity derived from them is larger still.
The whole per-method measurement is therefore one jitted call: XLA assigns and
frees the intermediates inside the executable, and only the reduced outputs,
smaller by a factor `N_REF`, reach Python, where each is copied to host memory
and its device buffer dropped at once. A worker peaks at `1.6` GB, flat
across the 32 chunks of a seed, so all 16 fit in memory together.

The per-seed shards exist only to carry results between processes. `run_all.py`
writes them to a temporary directory outside the project, fills the merged
arrays one shard at a time, and removes the directory once
`artifacts/trajectories.npz` is written.

## Notes

`artifacts/trajectories.npz` holds, for every (method, cost, seed), the mean
square error at every step and the other per-step means (`mabs`, `m4`, `supk`,
`zsq`), and for every path the terminal error and state and the pathwise sup
over `[0, T]`, together with the observable means at every step of the
scheme's own grid. The strong error reported is `max_k` of the root mean
square, the quantity Theorem 2 bounds; `supk` carries the larger
`(E max_k |.|^2)^{1/2}` at every step so the two orderings of maximum and
expectation can be compared. The plots take their
statistics from that file, so the list of test functions in `observables.py`
can be changed and the weak panel redrawn without simulating again.

No distance in `observables.py` is centred at the deterministic initial state.
The kink of `|x - X0|` there dominates the maximum at the first step: the
exact solution has variance `2h`, `E|x - X0|` grows like its square root, and
any `O(h)` deficit in a scheme's variance appears as an `O(sqrt h)` difference
of means. Such a centre measures the first step rather than the order.
