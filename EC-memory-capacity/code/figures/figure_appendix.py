"""Appendix figures.
Figure A1  the 34 attractors of the 16 input contexts, shown for 18 of the 37 nodes.
Figure A2  the network in three parts, matching the three rule tables: the regulators of (a) the interferon and
           inflammatory nodes, (b) the growth and cell-cycle nodes, (c) the effectors and phenotypes.
Figure A3  the network in context C0101: nodes fixed ON or OFF by the inputs and the seven free nodes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import itertools
import re

import netdraw as N
import style as S
from paths import FIGURES, TAB

S.apply()
SHOW = ["Proliferation", "Apoptosis", "EMT", "Invasion", "Migration", "MYC", "AKT2", "CCND2", "CCND3", "CDKN1B", "CDKN2A", "CDKN1A", "IRF1", "IFNG", "IL6", "STAT3", "NFKB1", "KLF4"]


def figure_a1():
    A = pd.read_csv(TAB + "/A2_attractor_repertoire_16_contexts.csv")
    W = pd.read_csv(TAB + "/A3_context_attractor_probabilities.csv").set_index("attractor").MaBoSS_probability_t1000
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    y = 0
    ticks, labels = [], []
    for ctx, g in A.groupby("context", sort=True):
        for _, r in g.iterrows():
            for j, n in enumerate(SHOW):
                v = str(r[n]).upper()
                if v == "X":
                    ax.add_patch(Rectangle((j - 0.42, y - 0.38), 0.84, 0.76, fc="white", ec=S.RUST, lw=0.6, hatch="//////"))
                else:
                    ax.add_patch(Rectangle((j - 0.42, y - 0.38), 0.84, 0.76, fc=S.INK if v == "1" else "white", ec=S.GREY, lw=0.3))
            ax.text(len(SHOW) - 0.3, y, f"{W[r['attractor']]:.2f}", va="center", ha="left", fontsize=6.5)
            ticks.append(y)
            labels.append(r["attractor"].replace("_A", " · ") + ("  CRC" if r["type"] != "fixed point" else ""))
            y += 1
        y += 0.5
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xticks(range(len(SHOW)))
    ax.set_xticklabels([S.label(n) for n in SHOW], rotation=60, ha="left", fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.text(len(SHOW) + 0.25, -1.4, "basin\nweight", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlim(-0.6, len(SHOW) + 1.2)
    ax.set_ylim(y - 0.5, -0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    S.save(fig, FIGURES, "FigureA1")


PARTS = [
    ("a", "Interferon and inflammatory nodes", S.MODULES[1][1] + S.MODULES[2][1]),
    ("b", "Growth and cell-cycle nodes", S.MODULES[3][1]),
    ("c", "Effectors and phenotypes", S.MODULES[4][1] + S.MODULES[5][1]),
]


def figure_a2(W=7.2):
    """Each panel draws the interactions that enter the nodes of one rule table. Nodes keep the same position."""
    E = N.signed_edges()
    boxes, bands, H = N.full_layout(W, py=0.19, head=0.36, foot=0.06)
    fig = plt.figure(figsize=(W, 3 * H + 0.22))
    for k, (letter, title, targets) in enumerate(PARTS):
        ax = N.net_axes(fig, 0, 0.22 + (2 - k) * H, W, H)
        N.draw_bands(ax, bands, labels=True, fs=6.2)
        sub = [e for e in E if e[1] in targets]
        used = {e[0] for e in sub} | set(targets)
        N.draw_edges(ax, boxes, sub, lw=0.42, head=0.026, lane_sep=0.016, track_sep=0.016)

        def style(n):
            if n in targets:
                return ("white", S.INK, S.INK, 1.0, "-")
            if n in used:
                return (S.INK, S.INK, "white", 0.5, "-") if n in N.INPUTS else ("white", S.GREY, S.INK, 0.5, "-")
            return ("none", S.LIGHT, S.LIGHT, 0.4, "-")

        N.draw_nodes(ax, boxes, style)
        ax.text(0.0, H - 0.02, f"({letter})", fontsize=10, fontweight="bold", va="top", ha="left")
        ax.text(0.30, H - 0.035, f"{title}: {len(sub)} interactions", fontsize=7.6, va="top", ha="left")
    from matplotlib.lines import Line2D

    fig.legend(handles=[Line2D([], [], color=N.ACT, lw=0.9, label="activation"), Line2D([], [], color=N.INH, lw=0.9, label="inhibition")],
               loc="lower center", ncol=2, fontsize=7.5, handlelength=2.2, columnspacing=1.6, borderaxespad=0.1)
    S.save(fig, FIGURES, "FigureA2")


def fixed_by_inputs(context):
    """Values of the nodes whose rule is constant once the inputs are fixed (logical propagation)."""
    rules = {}
    for line in open(N.DATA + "/EC_network.bnet"):
        if "," in line:
            n, r = [x.strip() for x in line.split(",", 1)]
            rules[n] = r
    val = dict(zip(N.INPUTS, [int(c) for c in context[1:]]))
    code = {n: compile(re.sub(r"!", " not ", r).replace("&", " and ").replace("|", " or ").strip(), n, "eval") for n, r in rules.items()}
    regs = {n: sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", r))) for n, r in rules.items()}
    changed = True
    while changed:
        changed = False
        for n in rules:
            if n in val:
                continue
            unknown = [r for r in regs[n] if r not in val]
            out = set()
            for combo in itertools.product([0, 1], repeat=len(unknown)):
                env = {r: val[r] for r in regs[n] if r in val}
                env.update(zip(unknown, combo))
                out.add(int(bool(eval(code[n], {}, env))))
                if len(out) == 2:
                    break
            if len(out) == 1:
                val[n] = out.pop()
                changed = True
    return val


def figure_a3(W=7.2, context="C0101"):
    """Fixed ON nodes filled, fixed OFF nodes pale, free nodes outlined in blue. Interactions leaving a fixed OFF node
    are omitted, because such a node neither activates nor inhibits in this context."""
    val = fixed_by_inputs(context)
    free = [n for n in N.signed_nodes() if n not in val]
    assert len(val) - 4 == 26 and len(free) == 7, (len(val), free)
    boxes, bands, H = N.full_layout(W, py=0.205, head=0.30, foot=0.30)
    fig = plt.figure(figsize=(W, H))
    ax = N.net_axes(fig, 0, 0, W, H)
    N.draw_bands(ax, bands)
    E = [e for e in N.signed_edges() if val.get(e[0], 1) == 1]
    N.draw_edges(ax, boxes, E, lw=0.40, head=0.024, lane_sep=0.016, track_sep=0.016)

    def style(n):
        if n in free:
            return ("white", S.BLUE, S.INK, 1.3, "-")
        if val[n]:
            return (S.INK, S.INK, "white", 0.5, "-")
        return ("white", S.LIGHT, S.GREY, 0.5, "-")

    N.draw_nodes(ax, boxes, style)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(fc=S.INK, ec=S.INK, label="fixed ON"), Patch(fc="white", ec=S.LIGHT, label="fixed OFF"),
                 Patch(fc="white", ec=S.BLUE, lw=1.3, label="free"),
                 Line2D([], [], color=N.ACT, lw=0.9, label="activation"), Line2D([], [], color=N.INH, lw=0.9, label="inhibition")],
        loc="lower center", ncol=5, fontsize=7.5, handlelength=1.8, columnspacing=1.6, borderaxespad=0.2)
    S.save(fig, FIGURES, "FigureA3")


if __name__ == "__main__":
    figure_a1()
    figure_a2()
    figure_a3()
