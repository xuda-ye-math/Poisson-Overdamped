"""One table of images: the reference, the four integrators, their differences.

    python plot_samples.py [N] [--paths 1,3,5]
                                    reads artifacts/samples_N{N}.npz (N = 60 by default)
                                    -> results/samples_N{N}.png
                                    --paths draws only the listed paths (1-based, in the
                                    order given)

Nine rows: the reference (the SRK-LD chain on the fine grid), then the four
integrators (stochastic Heun, random splitting, SRK-I, SRK-II), then the
four differences |x - x_ref| averaged over the colour channels, on one colour
scale, each row labelled with its largest L2 error to the reference over the
drawn paths; a small gap with a dashed grey line separates the three blocks.  Columns are the images;
one column is one Brownian path and one starting noise, shared by all rows.
The columns carry no title.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# the serif style of the other figures of the manuscript (../Convergence/plot_style.py)
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from integrators import METHODS

DPI = 400
GAP = 0.10                                             # height of a block gap, in image heights (twice the row gap)
LABEL_WIDTH = 2.0                                      # inches reserved for the row labels
FONTSIZE = 14                                          # row labels
# rows of the tables, top to bottom, with their labels
ROWS = [("stochastic_heun", "stochastic Heun"), ("random_splitting_rk3", "random splitting\nLMC (RK3)"),
        ("srk1", "SRK-I"), ("srk2", "SRK-II")]
REF_ROW = ("ref", "reference")


def table(d, out, rows, vmax, gaps):
    """rows: (key, label, cmap) per row; gaps: row indices preceded by a block gap."""
    n = d["img_ref"].shape[0]
    width = 1.25 * n + LABEL_WIDTH + 0.2
    heights, grid_row = [], []                          # spacer rows of the grid carry the gaps
    for r in range(len(rows)):
        if r in gaps:
            heights.append(GAP)
        grid_row.append(len(heights))
        heights.append(1.0)
    fig = plt.figure(figsize=(width, 1.25 * sum(heights) + 0.2))
    gs = fig.add_gridspec(len(heights), n, height_ratios=heights, left=LABEL_WIDTH / width,
                          right=1 - 0.2 / width, top=0.98, bottom=0.02, wspace=0.05, hspace=0.05)
    for r, (key, label, cmap) in enumerate(rows):
        for c in range(n):
            ax = fig.add_subplot(gs[grid_row[r], c])
            if cmap is None:
                ax.imshow(d[key][c], interpolation="nearest")
            else:
                ax.imshow(d[key][c], cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:                                  # method names centred in the label column
                pos = ax.get_position()
                fig.text(0.5 * LABEL_WIDTH / width, 0.5 * (pos.y0 + pos.y1), label, fontsize=FONTSIZE,
                         ha="center", va="center", multialignment="center")
                if r in gaps:                           # dashed grey line across the gap
                    y = 0.5 * (pos.y1 + fig.axes[-1 - n].get_position().y0)
                    fig.add_artist(Line2D([0.01, 0.99], [y, y], color="0.6", linestyle="--",
                                          linewidth=1.0, transform=fig.transFigure))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main(N, paths=None):
    d = dict(np.load(os.path.join(HERE, "artifacts", f"samples_N{N}.npz"), allow_pickle=False))
    names = [str(m) for m in d["methods"]]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_all = d["img_ref"].shape[0]
    stored = [int(x) for x in d["paths"]] if "paths" in d else list(range(1, n_all + 1))
    sel = [stored.index(p) for p in paths] if paths else list(range(n_all))
    labels = [stored[i] for i in sel]
    for k in list(d):                                  # restrict every per-image array to the selection
        if k in ("img_ref", "raw_ref") or "|" in k:
            d[k] = d[k][sel]

    res = os.path.join(HERE, "results")
    # the denoised images, (n, 3, 32, 32) in [-1, 1] -> (n, 32, 32, 3) in [0, 1] for display
    for m in names:
        d[f"show|{m}"] = np.clip(0.5 * (d[f"img|{m}"] + 1.0), 0.0, 1.0).transpose(0, 2, 3, 1)
        d[f"diff|{m}"] = np.abs(d[f"img|{m}"] - d["img_ref"]).mean(axis=1)
    d["ref|ref"] = np.clip(0.5 * (d["img_ref"] + 1.0), 0.0, 1.0).transpose(0, 2, 3, 1)
    vmax = max(d[f"diff|{m}"].max() for m in names)
    # the difference rows carry the L2 error to the reference, the largest over the drawn paths
    err = {m: float(d[f"rms_ref|{m}"].max()) for m in names}
    rows = ([("ref|ref", REF_ROW[1], None)] + [(f"show|{m}", label, None) for m, label in ROWS]
            + [(f"diff|{m}", f"{label}\n($L^2$ error $=$ $\\mathbf{{{err[m]:.3f}}}$)", "magma") for m, label in ROWS])
    table(d, os.path.join(res, f"samples_N{N}.png"), rows, vmax, gaps={1, 1 + len(ROWS)})
    print(f"difference colour scale 0 to {vmax:.3f}")

    print(f"\nN = {int(d['N'])} steps, h = {float(d['h']):.4f}, reference {int(d['N_ref'])} SRK-LD steps "
          f"on the same path; checkpoint epoch {int(d['epoch'])}; paths {labels}")
    for key, what in (("rms_ref", "denoised"),):
        print(f"{'integrator':28s} {'evals/step':>10s}   rms to reference per image ({what})    mean")
        for m in names:
            v = d[f"{key}|{m}"]
            print(f"{METHODS[m][2]:28s} {METHODS[m][1]:>10d}   " + " ".join(f"{x:.4f}" for x in v)
                  + f"   {v.mean():.4f}")
        print()
    print("\nrms between integrators, mean over images:")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {METHODS[a][2]:28s} vs {METHODS[b][2]:28s} {d[f'rms|{a}|{b}'].mean():.4f}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    sel = [int(x) for x in argv[argv.index("--paths") + 1].split(",")] if "--paths" in argv else None
    main(int(argv[0]) if argv and argv[0] != "--paths" else 60, sel)
