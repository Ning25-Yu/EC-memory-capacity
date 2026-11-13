"""Figure 5. What holds the memory and what erases it.
(a) edges of the seven-node switch whose deletion abolishes the memory (thick) and the four mutually redundant
inhibitory inputs to p27 (dashed); (b) capacity of a p27 pulse (L = 8) under permanent driver lesions;
(c) TCGA tumours in contexts with a proliferative attractor, by molecular subtype and PI3K lesion."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as S
import netdraw as N
from paths import FIGURES, TAB

S.apply()
COLS = ["C0101_A1", "C1101_A1", "C0100_A1", "C1100_A1", "C1001_A1", "C0001_A1", "C0001_A3", "C0000_A2"]
ROWS = [
    ("none", "p27_ON_bits", "intact network"),
    ("TP53 lesion (p21 OFF)", "p27_ON_bits", "p21 fixed OFF"),
    ("WNT lesion (MYC ON)", "p27_ON_bits", "MYC fixed ON"),
    ("PI3K lesion (AKT2 ON)", "p27_ON_bits", "AKT2 fixed ON"),
    ("PI3K lesion (AKT2 ON)", "p27_ON_with_AKT2_OFF_bits", "AKT2 fixed ON,\nAKT2 OFF pulsed"),
    ("AKT blockade (AKT2 OFF)", "p27_ON_bits", "AKT2 held OFF"),
]


ATLAS = pd.read_csv(TAB + "/I5_atlas_capacity.csv").query("pulse == 'CDKN1B=1'").set_index("attractor").capacity_bits


def panel_lesions(ax):
    F = pd.read_csv(TAB + "/I8_lesion_capacity.csv")
    M = np.full((len(ROWS), len(COLS)), np.nan)
    for i, (lesion, col, _) in enumerate(ROWS):
        d = F[F.lesion == lesion].set_index("attractor")
        for j, a in enumerate(COLS):
            if d.loc[a, "proliferative_mass"] > 0.5:
                M[i, j] = d.loc[a, col]
            if lesion == "none":  # intact network: the atlas value (larger simulations where simulated)
                M[i, j] = ATLAS.loc[a]
    im = ax.imshow(np.nan_to_num(M, nan=0.0), cmap=S.BITS, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n.p.", ha="center", va="center", fontsize=6.3, color=S.GREY)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3, color="white" if v > 0.55 else S.INK)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=7.5)
    ax.set_xticks(range(len(COLS)))
    kind = {"C1001_A1": "CRC", "C0001_A1": "CRC"}
    ax.set_xticklabels([f"{c[:5]}\n{kind.get(c, 'FP')}" for c in COLS], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(COLS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def panel_tumours(ax):
    T = pd.read_csv(TAB + "/I9_tumour_counts.csv").set_index("group")
    rows = [("all", "proliferative contexts, evaluable"), ("CN-high", "proliferative contexts, CN_HIGH"), ("CN-low", "proliferative contexts, CN_LOW"),
            ("MSI", "proliferative contexts, MSI"), ("POLE", "proliferative contexts, POLE")]
    for i, (lab, key) in enumerate(rows):
        les, intact = T.loc[key, "PI3K_lesion"], T.loc[key, "PI3K_intact"]
        ax.barh(i, intact, color=S.BLUE, height=0.62)
        ax.barh(i, les, left=intact, color=S.LIGHT, height=0.62)
        ax.text(intact + les + 2, i, f"{int(intact)} / {int(intact + les)}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, 215)
    ax.set_xticks([0, 60, 120, 180])
    ax.set_xlabel("Tumours in proliferative contexts")
    ax.barh([], [], color=S.BLUE, label="PTEN and PIK3CA wild type")
    ax.barh([], [], color=S.LIGHT, label="PTEN or PIK3CA mutated")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=S.BLUE, label="PTEN and PIK3CA wild type"), Patch(color=S.LIGHT, label="PTEN or PIK3CA mutated")],
              loc="lower right", fontsize=7.5, handlelength=1.0, handletextpad=0.4, labelspacing=0.25)


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.2, 4.6))
    ax_a = N.net_axes(fig, 0.12, 1.15, 3.55, 2.35)
    essential = {("CDKN1B", "MYC"), ("CDKN1B", "CCND2")}
    redundant = {("MYC", "CDKN1B"), ("CCND2", "CDKN1B"), ("CCND3", "CDKN1B"), ("AKT2", "CDKN1B")}
    N.draw_switch(ax_a, 3.55, 2.35, highlight=essential, dashed=redundant)
    S.panel(ax_a, "a", dx=0.02, dy=1.04)
    ax_a.legend(
        handles=[
            Line2D([], [], color=S.RUST, lw=2.0, label="deletion abolishes the memory"),
            Line2D([], [], color=S.RUST, lw=0.8, ls=(0, (2.2, 1.6)), label="any three of four may be deleted"),
            Line2D([], [], color=S.BLUE, lw=0.8, label="activation"),
            Line2D([], [], color=S.RUST, lw=0.8, label="inhibition"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7.5, handlelength=2.2, columnspacing=1.2, labelspacing=0.3,
    )
    ax_b = fig.add_axes([0.635, 0.50, 0.355, 0.36])
    panel_lesions(ax_b)
    S.panel(ax_b, "b", dx=-0.38, dy=1.17)
    ax_c = fig.add_axes([0.60, 0.09, 0.39, 0.27])
    panel_tumours(ax_c)
    S.panel(ax_c, "c", dx=-0.19, dy=1.04)
    S.save(fig, FIGURES, "Figure5")
