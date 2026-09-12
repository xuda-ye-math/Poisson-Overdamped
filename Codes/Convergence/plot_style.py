"""Shared figure style, matching the KLXX manuscript figures."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "cm", "font.family": "serif",
})

FIGSIZE = (5.4, 4.6)
DPI = 400

# display names for the manuscript: the two schemes it analyzes are I and II
LABELS = {
    "srk1": "stochastic Runge\u2013Kutta I",
    "srk2": "stochastic Runge\u2013Kutta II",
    "stochastic_heun": "stochastic Heun",
    "random_splitting_rk3": "random splitting LMC (RK3)",
    "euler_maruyama": "Euler\u2013Maruyama",
    "leimkuhler_matthews_y": "Leimkuhler\u2013Matthews",
    "leimkuhler_matthews_z": "Leimkuhler\u2013Matthews",
}

STYLE = {
    "euler_maruyama":      ("#4C72B0", "o"),
    "leimkuhler_matthews": ("#937860", "v"),
    "leimkuhler_matthews_y": ("#937860", "v"),
    "leimkuhler_matthews_z": ("#937860", "v"),
    "stochastic_heun":     ("#55A868", "s"),
    "randomized_midpoint": ("#8172B3", "^"),
    "srk1":                ("#DD8452", "D"),
    "srk2":                ("#C44E52", "*"),
    "random_splitting_rk3":("#64B5CD", "X"),
    "random_splitting":    ("#64B5CD", "P"),
}


def cost_axis(ax, costs):
    ax.set_xscale("log", base=4)
    ax.set_yscale("log", base=2)
    ax.set_xticks(costs)
    ax.set_xticklabels([f"$3\\cdot4^{{{int(round(np.log(c / 3) / np.log(4)))}}}$"
                        for c in costs])
    ax.set_xlabel(r"evaluations of $\nabla U$")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)


def slope_guides(ax, costs, guides):
    """guides: (order, value at the first cost, position along, y offset, text)."""
    c0, c1 = costs[0], costs[-1]
    out = []
    for order, y0, frac, off, txt in guides:
        y = y0 * (np.array([c0, c1]) / c0) ** (-order)
        ax.plot([c0, c1], y, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
        out.append((order, y0, c0 * (c1 / c0) ** frac, off, txt))
    return out


def label_guides(ax, costs, placed):
    """Call after the axes are drawn, so the rotations match the screen."""
    c0, c1 = costs[0], costs[-1]
    for order, y0, xa, off, txt in placed:
        p0 = ax.transData.transform((c0, y0))
        p1 = ax.transData.transform((c1, y0 * (c1 / c0) ** (-order)))
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(xa, y0 * (xa / c0) ** (-order) * off, txt, color="0.35",
                fontsize=9, rotation=angle, rotation_mode="anchor",
                ha="center", va="center", zorder=5)
