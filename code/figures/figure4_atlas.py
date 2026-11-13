"""Figure 4. Memory capacity of single-node pulses across the eight proliferative attractors.
(a) capacity (bits, L = 8) of every pulse that stores more than 0.05 bit from one attractor or more;
(b) number of pulses storing 0.5 bit or more against the number of free molecular nodes;
(c) pulse duration at which the capacity reaches 0.5 bit (exactly solved combinations)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style as S
from paths import FIGURES, TAB

S.apply()
A = pd.read_csv(TAB + "/I5_atlas_capacity.csv")
LAND = pd.read_csv(TAB + "/I1_landscape_entropy.csv").set_index("context")
COLS = ["C0101_A1", "C1101_A1", "C0100_A1", "C1100_A1", "C1001_A1", "C0001_A1", "C0001_A3", "C0000_A2"]
KIND = A.drop_duplicates("attractor").set_index("attractor").attractor_type.map({"fixed point": "FP", "complex recurrent class": "CRC"})


def panel_heat(ax):
    M = A.pivot(index="pulse", columns="attractor", values="capacity_bits")[COLS]
    M = M[M.max(axis=1) > 0.05]
    M = M.loc[M.assign(n=(M >= 0.5).sum(axis=1), s=M.sum(axis=1)).sort_values(["n", "s"], ascending=False).index]
    im = ax.imshow(M.values, cmap=S.BITS, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.55 else S.INK)
    ax.set_yticks(range(len(M)))
    ax.set_yticklabels([S.pulse_label(p) for p in M.index], fontsize=7.5)
    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([f"{c[:5]}\n{KIND[c]}\n{LAND.loc[c[:5], 'free_molecular_nodes']}" for c in COLS], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.text(-0.9, -0.62, "context\nattractor\nfree nodes", ha="right", va="bottom", fontsize=7, style="italic")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.set_xticks(np.arange(-0.5, len(COLS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(M)), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    return im


def panel_scatter(ax):
    n = A[A.capacity_bits >= 0.5].groupby("attractor").size().reindex(COLS).fillna(0)
    x = np.array([LAND.loc[c[:5], "free_molecular_nodes"] for c in COLS])
    fp = np.array([KIND[c] == "FP" for c in COLS])
    ax.scatter(x[fp], n.values[fp], s=22, color=S.BLUE, zorder=3, label="fixed point")
    ax.scatter(x[~fp], n.values[~fp], s=22, facecolor="white", edgecolor=S.BLUE, lw=1.0, zorder=3, label="complex class")
    ax.set_xlabel("Free molecular nodes")
    ax.set_ylabel("Pulses storing $\\geq 0.5$ bit")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 12.5)
    ax.set_yticks([0, 4, 8, 12])
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.2, labelspacing=0.25)


def panel_write(ax):
    W = pd.read_csv(TAB + "/I5b_write_time.csv")
    W["label"] = W.pulse.map(S.pulse_label)
    order = W.groupby("label").L_half_bit.min().sort_values().index.tolist()
    for i, lab in enumerate(order):
        g = W[W.label == lab]
        fp = g.attractor.map(KIND) == "FP"
        ax.plot([g.L_half_bit.min(), g.L_half_bit.max()], [i, i], color=S.GREY, lw=0.7, zorder=1)
        ax.scatter(g.L_half_bit[fp], [i] * fp.sum(), s=20, color=S.BLUE, zorder=3)
        ax.scatter(g.L_half_bit[~fp], [i] * (~fp).sum(), s=20, facecolor="white", edgecolor=S.BLUE, lw=1.0, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 12.5)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_xlabel("Duration for 0.5 bit, $L_{0.5}$")


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_axes([0.13, 0.03, 0.44, 0.83])
    im = panel_heat(ax)
    S.panel(ax, "a", dx=-0.22, dy=1.10)
    cax = fig.add_axes([0.585, 0.03, 0.012, 0.25])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("bits", fontsize=7.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7, length=2)
    ax_b = fig.add_axes([0.745, 0.60, 0.245, 0.30])
    panel_scatter(ax_b)
    S.panel(ax_b, "b", dx=-0.22, dy=1.04)
    ax_c = fig.add_axes([0.80, 0.10, 0.19, 0.33])
    panel_write(ax_c)
    S.panel(ax_c, "c", dx=-0.5, dy=1.04)
    S.save(fig, FIGURES, "Figure4")
