# Long-time weak error of the time average

Six schemes run to `T = 1.6 * 10^9` from the origin, at five costs each, and
the weak error of the running time average is plotted against the horizon:
the two SRK integrators of the manuscript, stochastic Heun, the random
splitting LMC with the RK3 drift flow, Euler–Maruyama, and Leimkuhler–Matthews
in its original form `Z_k` (see `../README.md` for the two forms). Thirty
independent chains in all: five costs times six schemes, one trajectory and
one seed each.

The definition is the one `../finite-time/` uses, with the exact solution
replaced by the measure the average converges to:

    weak error (t) = max_i | 1/(k+1) sum_{j<=k} f_i(Z_j) - pi(f_i) |,   t = k h,

over the same 1-Lipschitz family in `../observables.py`. By Kantorovich--
Rubinstein duality it is a lower bound for the Wasserstein-1 distance between
the empirical measure of the trajectory and `pi`.

## The equal-cost grid

Cost is the number of gradient evaluations spent per unit of time, taken along
`C = 6, 12, 24, 48, 96`. A scheme at `e` evaluations per step runs at
`h = e/C`, so the schemes cover the horizon in different numbers of steps for
the same budget: `e = 2` for integrator I and stochastic Heun, `e = 3` for
integrator II and the RK3 random splitting, `e = 1` for Euler–Maruyama and
Leimkuhler–Matthews. At the baseline `C = 6` this is `h = 1/3`, `1/2` and
`1/6`.

| `C` | `e = 1`: `h`, steps | `e = 2`: `h`, steps | `e = 3`: `h`, steps |
| --- | --- | --- | --- |
| 6 | 1/6, 9,600,000,000 | 1/3, 4,800,000,000 | 1/2, 3,200,000,000 |
| 12 | 1/12, 19,200,000,000 | 1/6, 9,600,000,000 | 1/4, 6,400,000,000 |
| 24 | 1/24, 38,400,000,000 | 1/12, 19,200,000,000 | 1/8, 12,800,000,000 |
| 48 | 1/48, 76,800,000,000 | 1/24, 38,400,000,000 | 1/16, 25,600,000,000 |
| 96 | 1/96, 153,600,000,000 | 1/48, 76,800,000,000 | 1/32, 51,200,000,000 |

Cost per unit time is the same for every scheme at a given `C`, so time and
cost are the same axis and any two curves are comparable at every point.
About 1.09 trillion steps in all.

## Sampling

The chain is stepped at `h` but recorded once per unit of time, so every run
stores the same `1.6 * 10^9` samples whatever its step size: 25.6 GB each and
768 GB for the thirty, where keeping every step would have been many petabytes.
The stride `C / evals` is a whole number for every cost in the grid, so a
sample always lands exactly on a step.

A subsampled chain has the same invariant measure, so an average over the
stored samples converges to the same `pi(f)` an average over every step would.
It is the average over those samples that the figures report.

## Files

| file | role |
| --- | --- |
| `invariant.py` | `pi(f_i)` for every test function, by quadrature against `exp(-U)` |
| `simulate.py` | one integrator at one cost: the trajectory to `T`, in segments |
| `run_long.py` | the thirty runs in parallel, one pinned core each |
| `weak_error.py` | sweeps every stored sample once and saves the curves |
| `plot_weak.py` | one figure per scheme, from the saved curves alone |

## Reproducing

Run from `Convergence/`:

```sh
python long-time/invariant.py    # -> artifacts/pi_observables.npz   (once)
python long-time/run_long.py     # -> artifacts/trajectory_*_segNN.npy, logs/
python long-time/weak_error.py   # -> artifacts/weak_error.npz       (once)
python long-time/plot_weak.py    # -> results/long_time_weak_error_{I,II,heun,rk3,em,lmz}.png
```

The measurement and the drawing are separate, because the trajectories are
768 GB and the curves drawn from them are a few megabytes. `weak_error.py`
reads every stored sample once and writes the running weak error at a grid of
checkpoints to `artifacts/weak_error.npz`; `plot_weak.py` opens that file and
nothing else, so a figure can be redrawn in a second without going near the
trajectories again.

`run_long.py` takes about six hours: thirty chains, one pinned core each,
about 1.09 trillion steps in total, and the longest of them, the
one-evaluation schemes at `C = 96` with 153.6 billion steps, sets the wall
clock. Both scripts accept a list of scheme names to run or sweep a subset. To have
it outlive the shell that starts it:

```sh
setsid nohup python long-time/run_long.py >> long-time/artifacts/run.log 2>&1 &
```

## Segments, saving and resuming

The horizon is cut into segments of `10^8` samples, each its own `.npy`. A
segment is filled block by block, and when it is complete the run saves it and
writes a checkpoint holding the chain state, the random key and the number of
finished segments. An interruption therefore costs at most the segment in
progress, never more, and never anything already saved.

```sh
python long-time/run_long.py                  # carry on where it left off
python long-time/run_long.py --fresh          # start the whole grid again at t = 0
python long-time/run_long.py --from 100000000 # carry on from the data at that time
```

Arguments are passed straight through to every worker, and `simulate.py` takes
the same three forms for a single run. `--from T` needs `T` to be a multiple of
the segment size and to be covered by the segments on disk; it rebuilds the key
by replay, since the key is determined by the seed and the number of blocks
consumed, so it needs nothing but the trajectory itself.

Without a checkpoint the segments on disk are counted instead. A segment is
created at full length and filled from the front, so its size says nothing
about how far it got: a half written one is recognised by its last row, which
is still the zeros the file was created with, and is not counted.

Nothing downstream reads the logs or the driver's output. Everything the
figures report, the recorded integration time included, is a field of the
saved files, so they can be redrawn at any time without simulating and
without the run that produced the data being present.

`invariant.py` depends only on the potential and the test functions, not on
any trajectory, so it is run once and reused. It checks nothing about the
runs; `weak_error.py` asserts that the saved names still match
`../observables.py` and stops if the family has changed.

Each run writes a numbered segment `trajectory_<method>_c<C>_segNN.npy`, one
sample per unit of time, `10^8` samples each, segment `00` one longer because
it carries the initial state. Alongside them
`trajectory_<method>_c<C>_state.npz` is the checkpoint: the chain state, the
random key, the number of finished segments, the step size, the evaluations
per step, the cost, the stride, the sample spacing, the horizon, the seed, and
`seconds`, the accumulated wall clock of the integration itself, recorded for
reference.

## Memory

Neither the simulation nor the plot holds a trajectory whole. `simulate.py`
integrates in blocks of `100000` unit-time windows, writes each block's
samples into the memory-mapped `.npy`, releases the block before starting the
next and drops the segment's map when the segment closes, so a worker's
footprint is set by the block and not by the horizon: the 76.8-billion-step
run holds a few hundred MB of live memory, and the thirty together sit near
30 GB resident, most of it reclaimable page cache.

`weak_error.py` sweeps the same way. Each segment is memory mapped and read in
blocks of `400000` samples, and only the running sum of the 20 test functions
crosses from one block to the next, so it holds one block whatever the
horizon. The block size is also the checkpoint spacing and divides the segment
length, so a block never straddles a checkpoint and every checkpoint is an
exact running average over all samples up to it. Nothing is subsampled in the
measurement; `STRIDE` in `plot_weak.py` thins only what is drawn. `plot_weak.py` maps each trajectory read-only and
sweeps it in chunks, carrying only the running sums of the 20 test functions,
so the longest run costs no more to plot than the shortest.

## pi(f) by quadrature

The dynamics has invariant density proportional to `exp(-U)` on `R^2`, so

    pi(f) = int f exp(-U) / int exp(-U) ,

and both integrals are midpoint sums on one uniform grid, whose cell area
cancels. Outside the wells `U` grows like `|x|^2/2`, so a box of half-width
`L = 10` leaves a mass of order `exp(-L^2/2) = 2e-22` outside it. The grid is
`10000 x 10000`, a spacing of `0.002`, swept in blocks of a hundred rows so
that no array of `10^8` points is held at once. On the GPU it takes under a
second.

The values are converged far below anything the trajectories resolve: halving
or doubling the resolution, or taking the box to `L = 8` or `L = 14`, moves
every `pi(f_i)` by at most `4e-10`. Antipodal projections come out exactly
opposite, `pi(proj180) = -pi(proj0)` to `7e-17`, which the grid does not
enforce.

## What the figure shows

One figure per scheme, five non-negative curves each: simulation time on a
uniform axis from `0` to the horizon reached, the error on a log axis in
powers of two, one colour per step size. Each figure has its own axis
limits, except that the two SRK figures share one pair, so they can be read
side by side. Only finished
segments are read, so the figures can be drawn while the simulation is still
running and will show whatever horizon has been reached.

The two coarse curves flatten early and stay flat for the rest of the horizon.
That plateau is the step-size bias: the gap between `pi` and the measure the
chain actually samples at that `h`, which no amount of further time removes.
At `C = 6` integrator II sits above integrator I, as the step sizes predict,
since equal cost buys it `h = 1/2` against `h = 1/3`.

The finer curves are still descending at `T`, roughly like `1/sqrt(t)`, and by
`C = 48` they have run into the sampling error of a single ergodic average and
are no longer ordered by cost: at the horizon integrator I reads `1.1e-3` at
`C = 48` but `3.4e-3` at `C = 96`, and integrator II the other way round.
Those two costs are not resolved by this measurement. Each curve is one
trajectory, so the level a curve reaches mixes bias with sampling error, and
only where the plateau is flat and well above the others is the bias what is
being read. Separating the two at the finer costs would need several
independent paths per point.

Euler–Maruyama gives five flat plateaus that halve from one cost to the
next: first order. Leimkuhler–Matthews is second order in its plateaus,
`2.5e-3`, `7.6e-4`, `2.0e-4` at `C = 6, 12, 24`, and reaches the sampling
floor of a single trajectory two costs earlier than the SRK schemes; at equal
cost it lies below both SRK integrators at every `C`, by a factor 15 to 19
where its bias is resolved.

## Notes

The noise is drawn directly at the step size. `fine_noise` returns the
increment and its exact time integral over each step, so the pair the schemes
consume is exact for the Brownian path and nothing is discretized.

No reference trajectory is involved. A long-time average is judged by whether
it settles as the horizon grows, and by schemes of the same order agreeing at
equal cost, not by comparison against an exact solution over a fixed
interval; that is what `../finite-time/` measures.
