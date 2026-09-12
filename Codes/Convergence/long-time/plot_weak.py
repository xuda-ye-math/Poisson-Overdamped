"""The weak error of the time average against the horizon, at five costs.

The definition is the one the finite-time study uses, with the exact solution
replaced by the invariant measure the average converges to:

    weak error (t) = max_i | 1/(k+1) sum_{j<=k} f_i(Z_j) - pi(f_i) |,   t = k,

over the 1-Lipschitz family of ../observables.py.  By Kantorovich--Rubinstein
duality it is a lower bound for the Wasserstein-1 distance between the
empirical measure of the trajectory and pi.

This script draws only.  The curves were computed once by weak_error.py, which
swept every one of the 1.6e9 stored samples of each run, and were written to
artifacts/weak_error.npz; here nothing larger than that file is opened, so the
239 GB of trajectories is never touched and memory is a few megabytes.

One figure per integrator, five cost curves each: the horizon on a uniform
axis, the error on a log axis in powers of two.  Both share one pair of axis
limits, so the two can be read side by side, and one colour per step size in
the default cycle, the k-th coarsest step of either integrator sharing a
colour.  Each curve is named where it runs rather than in a legend.  STRIDE
thins the plotted points; the saved curves are always the full computation.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # plot_style.py

import numpy as np

from plot_style import plt, LABELS, DPI

RESULTS = os.path.join(HERE, "results")
CURVES = os.path.join(HERE, "artifacts", "weak_error.npz")
TAG = {"srk1": "I", "srk2": "II", "stochastic_heun": "heun",
       "random_splitting_rk3": "rk3", "euler_maruyama": "em",
       "leimkuhler_matthews_z": "lmz"}
# smaller than the finite-time panels, so that at the same width on the page
# the fonts come out correspondingly larger
FIGSIZE = (4.5, 3.5)
SHARED = ("srk1", "srk2")   # the two SRK figures share their y-range
# (method, cost index) whose step-size label is written below its curve
LABEL_BELOW = {("leimkuhler_matthews_z", 3)}
STRIDE = 1           # plot every STRIDE-th checkpoint; the file keeps them all
# the default colour cycle, one colour per step size.  The two integrators run
# different step sizes at the same cost, so the k-th coarsest of each carries
# the same colour and the legend names the step size rather than the cost.
COLOURS = [f"C{k}" for k in range(10)]


def step_label(h):
    """The step size as 1/h in the form base x 2^k, base 3 for I and 2 for II."""
    inv = int(round(1.0 / h))
    base = 3 if inv % 3 == 0 else 2
    k = int(round(np.log2(inv / base)))
    assert base * 2**k == inv, (h, inv)
    return rf"$h^{{-1}} = {base} \times 2^{{{k}}}$"


def main():
    """Draw every method in TAG, or only the ones named on the command line:

        python plot_weak.py                                       all methods
        python plot_weak.py euler_maruyama leimkuhler_matthews_y  these only

    The shared axis limits are taken over the methods drawn, so a partial call
    leaves the figures of the other methods exactly as they are.
    """
    d = np.load(CURVES, allow_pickle=False)
    t = d["t"][::STRIDE]
    err = d["err"][:, ::STRIDE]
    methods = [str(m) for m in d["methods"]]
    costs = [int(c) for c in d["costs"]]
    order = sorted(set(costs))
    T = float(d["t"][-1])

    drawn = sys.argv[1:] or list(TAG)
    assert all(m in TAG for m in drawn), drawn
    os.makedirs(RESULTS, exist_ok=True)
    for m in drawn:
        rows = [i for i, mm in enumerate(methods) if mm == m]
        # every figure has its own y-range, except that the two SRK figures
        # share one, so that they can be read side by side
        share = SHARED if m in SHARED else (m,)
        lim = [i for i, mm in enumerate(methods) if mm in share]
        lo, hi = err[lim].min(), err[lim].max()
        if not rows:                 # not swept yet: no figure for it
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for i in sorted(rows, key=lambda i: costs[i]):
            c = order.index(costs[i])
            ax.plot(t, err[i], color=COLOURS[c], linewidth=1.1)
            # the step size written on the curve itself, a third of an octave
            # above it, rather than in a legend; below it where the two finest
            # curves of a panel overlap and the labels would collide
            j = len(t) // 2
            below = (m, c) in LABEL_BELOW
            ax.text(t[j], err[i][j] * 2**(-0.33 if below else 0.33),
                    step_label(float(d["h"][i])), color=COLOURS[c],
                    fontsize=9, ha="center", va="top" if below else "bottom")
        ax.set_yscale("log", base=2)
        ax.set_xlim(0.0, T)
        ax.set_xticks(np.linspace(0.0, T, 5))
        # the same limits on both figures, with headroom for the top label
        ax.set_ylim(lo / 1.6, hi * 2.5)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\max_i\, |\, \overline{f_i} - \pi(f_i) \,|$")
        ax.set_title(LABELS[m], fontsize=10)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        out = os.path.join(RESULTS, f"long_time_weak_error_{TAG[m]}.png")
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        print(f"wrote {out}")

    names = [str(x) for x in d["obs_names"]]
    print(f"\n{len(names)} 1-Lipschitz observables, T = {T:g}, "
          f"{d['err'].shape[1]} checkpoints per curve "
          f"(every {int(d['block']):,} samples)\n")
    print(f"{'integrator':28s}{'cost':>6s}{'h':>10s}{'seconds':>10s}"
          f"{'weak error':>14s}   argmax f")
    for i in sorted(range(len(methods)), key=lambda i: (methods[i], costs[i])):
        last = d["dev"][i, -1]
        print(f"{LABELS[methods[i]]:28s}{costs[i]:>6d}{d['h'][i]:>10.5g}"
              f"{d['seconds'][i]:>10.0f}{d['err'][i, -1]:>14.4e}"
              f"   {names[int(last.argmax())]}")


if __name__ == "__main__":
    main()
