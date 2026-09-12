"""Log-log plot of the weak error against cost.

Reads artifacts/trajectories.npz only.  The weak error reported is

    max_i max_k | E f_i(Z_k) - E f_i(X_kh) |

over the 1-Lipschitz family of observables.py, which by Kantorovich--
Rubinstein duality is a lower bound for W_1( Law(Z_N), Law(X_T) ).  Both
expectations are taken over the same Brownian paths, so the estimator is a
mean of coupled differences and its noise is set by the strong error rather
than by the spread of f_i itself.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # plot_style.py, one level up

import numpy as np

from plot_style import plt, STYLE, LABELS, FIGSIZE, DPI, cost_axis, slope_guides, label_guides

NPZ = os.path.join(HERE, "artifacts", "trajectories.npz")
FIG = os.path.join(HERE, "results", "weak_error.png")
CSV = os.path.join(HERE, "artifacts", "weak_error_summary.csv")

# (order, value at the first cost, position along, y offset, text): the slope -1
# guide runs just above the first order bundle, the slope -2 guide just below
# the two SRK curves, so neither crosses a curve and both sit next to the
# curves they refer to
GUIDES = ((1.0, 0.10, 0.80, 2.64, r"slope $=-1$"),
          (2.0, 0.008, 0.85, 0.45, r"slope $=-2$"))
# the methods drawn, in legend order: the two SRK schemes last.  The RK4
# random splitting is saved but not drawn, since it gives the same error as
# the RK3 one at equal step.
DRAWN = ["euler_maruyama", "leimkuhler_matthews_y", "stochastic_heun", "randomized_midpoint",
         "random_splitting_rk3", "srk1", "srk2"]


def main():
    d = np.load(NPZ, allow_pickle=False)
    costs = d["costs"]
    labels = {str(m): LABELS.get(str(m), str(l)) for m, l in zip(d["methods"], d["labels"])}
    evals = {str(m): int(e) for m, e in zip(d["methods"], d["evals"])}
    methods = [m for m in DRAWN if m in labels]
    obs_names = [str(o) for o in d["obs_names"]]

    mean, std, argmax = {}, {}, {}
    for m in methods:
        per_seed = []
        for C in costs:
            w = np.abs(d[f"wdiff|{m}|{C}"])          # (seeds, n_steps+1, n_obs)
            per_seed.append(w.max(axis=(1, 2)))      # sup over [0,T], max over observables
        per_seed = np.stack(per_seed, axis=1)        # (seeds, costs)
        mean[m], std[m] = per_seed.mean(0), per_seed.std(0)
        w = np.abs(d[f"wdiff|{m}|{costs[-1]}"]).max(axis=1).mean(axis=0)
        argmax[m] = obs_names[int(w.argmax())]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    placed = slope_guides(ax, costs, GUIDES)
    for m in methods:
        colour, marker = STYLE[m]
        ax.errorbar(costs, mean[m], yerr=std[m], color=colour, marker=marker,
                    linestyle="-", markersize=6.5, capsize=3, linewidth=1.5,
                    label=labels[m])
    cost_axis(ax, costs)
    ax.set_ylabel("weak error")
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.canvas.draw()
    label_guides(ax, costs, placed)
    fig.savefig(FIG, dpi=DPI)
    print(f"wrote {FIG}")

    with open(CSV, "w") as f:
        f.write("method,label,evals_per_step,cost,n_steps,h,weak_error,std_over_seeds\n")
        for m in methods:
            for i, C in enumerate(costs):
                n = C // evals[m]
                f.write(f"{m},{labels[m]},{evals[m]},{C},{n},{float(d['T'])/n:.12e},"
                        f"{mean[m][i]:.12e},{std[m][i]:.12e}\n")
    print(f"wrote {CSV}")

    print(f"\nlog2 weak error   ({len(obs_names)} 1-Lipschitz observables, "
          f"{int(d['n_seeds'])} seeds x {int(d['n_paths'])} paths)")
    hdr = "method".ljust(28) + "e/step" + "".join(f"{np.log2(c):>9.2f}" for c in costs) + "    slope   argmax f"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        s = np.polyfit(np.log2(costs), np.log2(mean[m]), 1)[0]
        print(labels[m].ljust(28) + f"{evals[m]:<6d}"
              + "".join(f"{v:>9.2f}" for v in np.log2(mean[m]))
              + f"   {s:+.3f}   {argmax[m]}")


if __name__ == "__main__":
    main()
