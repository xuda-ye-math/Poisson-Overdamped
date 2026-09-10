"""Log-log plot of the strong error against cost.

Reads artifacts/trajectories.npz only; it runs no simulation.  The strong
error is the one the strong convergence theorem bounds,

    max_{0<=k<=N} ( E | Z_k - X_{kh} |^2 )^{1/2},

the maximum over steps taken after the expectation, not before.  It is
computed per seed and then averaged, the bars being the spread over seeds.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # plot_style.py, one level up

import numpy as np
from plot_style import plt, STYLE, LABELS, FIGSIZE, DPI, cost_axis, slope_guides, label_guides

NPZ = os.path.join(HERE, "artifacts", "trajectories.npz")
FIG = os.path.join(HERE, "results", "strong_error.png")
CSV = os.path.join(HERE, "artifacts", "strong_error_summary.csv")

# (order, value at the first cost, position along, y offset, text); the labels
# sit on the right of the figure, clear of every curve by 2.3, 1.4 and 1.1
# octaves
GUIDES = ((0.5, 0.90, 0.60, 2.22, r"slope $=-0.5$"),
          (1.0, 0.11, 0.80, 0.45, r"slope $=-1$"),
          (1.5, 0.080, 0.85, 0.38, r"slope $=-1.5$"))
# the methods drawn, in legend order: the two SRK schemes last.  The RK4
# random splitting is saved but not drawn, since it gives the same error as
# the RK3 one at equal step.
DRAWN = ["euler_maruyama", "leimkuhler_matthews", "stochastic_heun", "randomized_midpoint",
         "random_splitting_rk3", "srk1", "srk2"]


def main():
    d = np.load(NPZ, allow_pickle=False)
    costs = d["costs"]
    labels = {str(m): LABELS.get(str(m), str(l)) for m, l in zip(d["methods"], d["labels"])}
    evals = {str(m): int(e) for m, e in zip(d["methods"], d["evals"])}
    methods = [m for m in DRAWN if m in labels]

    mean, std = {}, {}
    for m in methods:
        # msd[seed, k] is E |Z_k - X_kh|^2 for that seed; the maximum over k
        # comes after the expectation
        per_seed = np.stack([np.sqrt(d[f"msd|{m}|{C}"].max(axis=1))
                             for C in costs], axis=1)      # (seeds, costs)
        mean[m], std[m] = per_seed.mean(0), per_seed.std(0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    placed = slope_guides(ax, costs, GUIDES)
    for m in methods:
        colour, marker = STYLE[m]
        ax.errorbar(costs, mean[m], yerr=std[m], color=colour, marker=marker,
                    linestyle="-", markersize=6.5, capsize=3, linewidth=1.5,
                    label=labels[m])
    cost_axis(ax, costs)
    ax.set_ylabel("strong error")
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.canvas.draw()
    label_guides(ax, costs, placed)
    fig.savefig(FIG, dpi=DPI)
    print(f"wrote {FIG}")

    with open(CSV, "w") as f:
        f.write("method,label,evals_per_step,cost,n_steps,h,strong_error,std_over_seeds\n")
        for m in methods:
            for i, C in enumerate(costs):
                n = C // evals[m]
                f.write(f"{m},{labels[m]},{evals[m]},{C},{n},{float(d['T'])/n:.12e},"
                        f"{mean[m][i]:.12e},{std[m][i]:.12e}\n")
    print(f"wrote {CSV}")

    print("\nlog2 strong error   (max over [0,T] of the root mean square, "
          f"{int(d['n_seeds'])} seeds x {int(d['n_paths'])} paths)")
    hdr = "method".ljust(28) + "e/step" + "".join(f"{np.log2(c):>9.2f}" for c in costs) + "    slope"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        s = np.polyfit(np.log2(costs), np.log2(mean[m]), 1)[0]
        print(labels[m].ljust(28) + f"{evals[m]:<6d}"
              + "".join(f"{v:>9.2f}" for v in np.log2(mean[m])) + f"   {s:+.3f}")
    print(f"\nreference self-gap: {float(d['reference_self_gap']):.3e}")


if __name__ == "__main__":
    main()
