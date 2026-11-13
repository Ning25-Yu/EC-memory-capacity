"""Figure 1. The network, its input-constrained attractor landscape and the seven-node switch.
(a) signed interaction matrix of the 37 nodes; (b) the 16 input contexts: nodes fixed by the inputs and history
entropy; (c) the seven free nodes of C0101 (orthogonal edges) and the states of its two fixed points."""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import ecbn
import netdraw as N
import style as S
from paths import FIGURES, TAB

S.apply()
ORDER = [n for _, g in S.MODULES for n in g]


def signed_edges():
    out = []
    for target, rule in ecbn.RULES.items():
        if target in ecbn.INPUTS:
            continue
        seen = {}
        for neg, src in re.findall(r"(!?)([A-Za-z_][A-Za-z0-9_]*)", rule):
            seen[src] = -1 if neg else 1
        out += [(src, target, sign) for src, sign in seen.items()]
    return out


def panel_matrix(ax):
    ix = {n: i for i, n in enumerate(ORDER)}
    edges = signed_edges()
    assert len(edges) == 168
    n = len(ORDER)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    start = 0
    for k, (name, group) in enumerate(S.MODULES):
        if k % 2 == 0:
            ax.add_patch(Rectangle((start - 0.5, -0.5), len(group), n, color=S.PALE, lw=0, zorder=0))
            ax.add_patch(Rectangle((-0.5, start - 0.5), n, len(group), color=S.PALE, lw=0, zorder=0))
        start += len(group)
    for src, tgt, sign in edges:
        ax.add_patch(
            Rectangle((ix[tgt] - 0.4, ix[src] - 0.4), 0.8, 0.8, color=S.BLUE if sign > 0 else S.RUST, lw=0, zorder=2)
        )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([S.label(x) for x in ORDER], rotation=90, fontsize=5.6)
    ax.set_yticklabels([S.label(x) for x in ORDER], fontsize=5.6)
    ax.tick_params(length=0, pad=1.5)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.set_ylabel("Regulator")
    ax.set_xlabel("Target")
    ax.xaxis.set_label_position("top")
    ax.scatter([], [], marker="s", s=22, color=S.BLUE, label="activation")
    ax.scatter([], [], marker="s", s=22, color=S.RUST, label="inhibition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, handletextpad=0.2, columnspacing=1.2)


def panel_contexts(ax_code, ax_fix, ax_h):
    D = pd.read_csv(TAB + "/I1_landscape_entropy.csv")
    D = D.sort_values(["n_proliferative_attractors", "nodes_fixed_by_inputs"], ascending=[False, False]).reset_index(drop=True)
    D["prolif"] = D.n_proliferative_attractors > 0
    D = pd.concat([D[D.prolif].sort_values("nodes_fixed_by_inputs", ascending=False), D[~D.prolif].sort_values("nodes_fixed_by_inputs", ascending=False)]).reset_index(drop=True)
    y = np.arange(len(D))
    for i, ctx in enumerate(D.context):
        for j, b in enumerate(ctx[1:]):
            ax_code.add_patch(Rectangle((j - 0.4, i - 0.36), 0.8, 0.72, fc=S.INK if b == "1" else "white", ec=S.INK, lw=0.4))
    ax_code.set_xlim(-0.6, 3.6)
    ax_code.set_xticks(range(4))
    ax_code.set_xticklabels(ecbn.INPUTS, rotation=90, fontsize=6.5)
    ax_code.xaxis.tick_top()
    ax_code.tick_params(length=0, pad=1.5)
    ax_code.set_yticks(y)
    ax_code.set_yticklabels(D.context, fontsize=6.5)
    for s in ax_code.spines.values():
        s.set_visible(False)
    col = [S.BLUE if p else S.LIGHT for p in D.prolif]
    ax_fix.barh(y, D.nodes_fixed_by_inputs, color=col, height=0.72)
    ax_fix.set_xlim(0, 33)
    ax_fix.set_xticks([0, 11, 22, 33])
    ax_fix.set_xlabel("Nodes fixed by\nthe inputs (of 33)", fontsize=8)
    ax_h.barh(y, D.history_entropy_bits, color=col, height=0.72)
    ax_h.set_xlim(0, 2)
    ax_h.set_xticks([0, 1, 2])
    ax_h.set_xlabel("History entropy\n$H(A\\,|\\,u)$ (bits)", fontsize=8)
    for a in (ax_fix, ax_h):
        a.set_yticks([])
        a.spines["left"].set_visible(False)
    for a in (ax_code, ax_fix, ax_h):
        a.set_ylim(len(D) - 0.5, -0.5)
    k = int(D.prolif.sum())
    for a in (ax_fix, ax_h):
        a.axhline(k - 0.5, color=S.GREY, lw=0.5, ls=(0, (3, 2)))
    from matplotlib.patches import Patch

    ax_fix.legend(handles=[Patch(color=S.BLUE, label="context with a proliferative attractor"), Patch(color=S.LIGHT, label="context without")],
                  loc="upper left", bbox_to_anchor=(-0.85, -0.2), ncol=1, fontsize=7, handlelength=1.0, handletextpad=0.4, labelspacing=0.2)


GROW = {"AKT2": 1, "CCND2": 1, "CCND3": 1, "CDKN1B": 0, "IRF1": 0, "MYC": 1, "Proliferation": 1}
ARREST = {k: 1 - v for k, v in GROW.items()}

def state_table(ax):
    A3 = pd.read_csv(TAB + "/A3_context_attractor_probabilities.csv").set_index("attractor").MaBoSS_probability_t1000
    nodes = ["MYC", "AKT2", "CCND2", "CCND3", "CDKN1B", "IRF1", "Proliferation"]
    rows = [("proliferative", GROW, A3["C0101_A1"]), ("arrested", ARREST, A3["C0101_A2"])]
    for r, (name, st, w) in enumerate(rows):
        for c, n in enumerate(nodes):
            ax.add_patch(Rectangle((c - 0.4, r - 0.36), 0.8, 0.72, fc=S.INK if st[n] else "white", ec=S.INK, lw=0.5))
        ax.text(-0.75, r, name, ha="right", va="center", fontsize=8)
        ax.text(len(nodes) + 0.75, r, f"{w:.2f}", ha="center", va="center", fontsize=8)
    for c, n in enumerate(nodes):
        ax.text(c, -0.6, S.label(n), ha="left", va="bottom", rotation=50, fontsize=7.5)
    ax.text(len(nodes) + 0.75, -0.6, "basin\nweight", ha="center", va="bottom", fontsize=7.5)
    ax.text(-0.75, -0.75, "fixed point", ha="right", va="bottom", fontsize=7.5, style="italic")
    ax.set_xlim(-3.6, len(nodes) + 2.0)
    ax.set_ylim(1.7, -2.3)
    ax.set_aspect("equal")
    ax.axis("off")


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.2, 6.7))
    ax_m = fig.add_axes([0.07, 0.385, 0.50, 0.54])
    panel_matrix(ax_m)
    S.panel(ax_m, "a", dx=-0.09, dy=1.10)
    ax_c = fig.add_axes([0.68, 0.50, 0.06, 0.40])
    ax_f = fig.add_axes([0.755, 0.50, 0.105, 0.40])
    ax_h = fig.add_axes([0.885, 0.50, 0.105, 0.40])
    panel_contexts(ax_c, ax_f, ax_h)
    S.panel(ax_c, "b", dx=-0.75, dy=1.13)
    ax1 = N.net_axes(fig, 0.30, 0.20, 3.55, 1.95)
    N.draw_switch(ax1, 3.55, 1.95)
    S.panel(ax1, "c", dx=0.0, dy=1.0)
    ax2 = fig.add_axes([0.55, 0.07, 0.44, 0.22])
    state_table(ax2)
    S.save(fig, FIGURES, "Figure1")
