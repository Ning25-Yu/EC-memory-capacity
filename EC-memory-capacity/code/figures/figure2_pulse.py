"""Figure 2. A finite pulse as a write operation.
(a) pulse protocol; (b) arrest probability a(L) from the closed forms (lines) and the numerical matrix exponential
(circles); (c) memory capacity C(a(L)); (d) information in the phenotype at the end of an L = 8 pulse and stored
after withdrawal, for the 13 pulses that change Proliferation during exposure."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

import style as S
from infotheory import arrest_for_capacity
from paths import FIGURES, TAB

S.apply()
I2 = pd.read_csv(TAB + "/I2_duration_capacity.csv")
D3 = pd.read_csv(TAB + "/D3_exact_pulse_duration_response.csv")
CURVES = [
    ("CDKN1B ON", "p27 ON", S.BLUE, "-"),
    ("MYC OFF + CCND2 OFF", "MYC + CCND2 OFF", S.BLUE, (0, (5, 2))),
    ("CDKN2A ON", "p16 ON", S.GREY, "-"),
    ("MYC OFF", "six other pulses", S.RUST, "-"),
]


def panel_protocol(ax):
    ax.set_xlim(-1.2, 10.6)
    ax.set_ylim(-2.3, 2.6)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0.55), 4, 0.9, fc=S.LIGHT, ec=S.INK, lw=0.6))
    ax.text(2, 1.0, "clamp $x_i = c$", ha="center", va="center", fontsize=8.5)
    ax.annotate("", xy=(10.3, 0), xytext=(-1.0, 0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=S.INK))
    for x, t in [(0, "0"), (4, "$L$")]:
        ax.plot([x, x], [-0.12, 0.12], color=S.INK, lw=0.7)
        ax.text(x, -0.25, t, ha="center", va="top", fontsize=8.5)
    ax.text(10.3, -0.25, "time", ha="right", va="top", fontsize=8.5)
    ax.text(-0.5, 1.9, "$p_0=\\delta_{s_0}$", ha="center", fontsize=8.5)
    ax.text(2, 1.9, "$\\dot p = p\\,Q_c$", ha="center", fontsize=8.5)
    ax.text(6.4, 1.9, "$\\dot p = p\\,Q$", ha="center", fontsize=8.5)
    ax.plot([4, 4], [0.12, 2.3], color=S.GREY, lw=0.5, ls=(0, (3, 2)))
    ax.plot([0, 0], [0.12, 2.3], color=S.GREY, lw=0.5, ls=(0, (3, 2)))
    for y, lab, col in [(1.0, "arrested, probability $a(L)$", S.BLUE), (0.35, "proliferative, $1-a(L)$", S.RUST)]:
        ax.add_patch(FancyArrowPatch((4.15, 0.68), (6.3, y), arrowstyle="-|>", mutation_scale=6, lw=0.8, color=col, connectionstyle="arc3,rad=0.0"))
        ax.text(6.45, y, lab, ha="left", va="center", fontsize=8, color=col)
    ax.text(4.6, -1.25, "$a(L)=\\delta_{s_0^{c}}\\,e^{Q_cL}\\,h$,     $Qh=0$", ha="center", va="center", fontsize=9.5)
    ax.text(4.6, -2.0, "$C(a)=\\log_2\\!\\left[1+a\\,(1-a)^{(1-a)/a}\\right]$", ha="center", va="center", fontsize=9.5)


def panel_curves(ax, column, ylabel):
    for key, lab, col, ls in CURVES:
        d = I2[I2.pulse == key]
        ax.plot(d.L, d[column], color=col, ls=ls, label=lab, lw=1.3)
    if column == "arrest":
        pts = D3[D3.clamp.isin(["CDKN1B ON", "MYC OFF + CCND2 OFF"])]
        ax.plot(pts.duration, pts.eventual_arrest, ls="none", marker="o", ms=3, mfc="white", mec=S.INK, mew=0.6, zorder=5)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("Pulse duration $L$")
    ax.set_ylabel(ylabel)


def panel_screen(ax):
    K = pd.read_csv(TAB + "/I3_screen_information.csv")
    K = K.sort_values(["stored_bits_after_withdrawal", "phenotype_bits_end_of_pulse"]).reset_index(drop=True)
    y = np.arange(len(K))
    for i, r in K.iterrows():
        stored = r.stored_bits_after_withdrawal
        col = S.BLUE if stored > 0.5 else (S.GREY if stored > 0 else S.RUST)
        ax.plot([stored, r.phenotype_bits_end_of_pulse], [i, i], color=col, lw=0.8, zorder=1)
        ax.plot(r.phenotype_bits_end_of_pulse, i, marker="o", ms=4.5, mfc="white", mec=col, mew=0.9, zorder=2)
        ax.plot(stored, i, marker="o", ms=4.5, mfc=col, mec=col, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([S.pulse_label(p) for p in K.pulse], fontsize=7.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim(-0.7, len(K) - 0.3)
    ax.set_xlabel("Information (bits)")
    ax.plot([], [], ls="none", marker="o", ms=4.5, mfc="white", mec=S.INK, mew=0.9, label="phenotype, end of pulse")
    ax.plot([], [], ls="none", marker="o", ms=4.5, mfc=S.INK, mec=S.INK, label="stored after withdrawal")
    ax.legend(loc="lower center", bbox_to_anchor=(0.3, 1.0), ncol=1, handletextpad=0.2, fontsize=7.5, labelspacing=0.2)


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.2, 5.9))
    ax_a = fig.add_axes([0.03, 0.60, 0.60, 0.37])
    panel_protocol(ax_a)
    S.panel(ax_a, "a", dx=0.035, dy=0.93)
    ax_d = fig.add_axes([0.765, 0.08, 0.22, 0.80])
    panel_screen(ax_d)
    S.panel(ax_d, "d", dx=-0.85, dy=1.09)
    ax_b = fig.add_axes([0.075, 0.08, 0.215, 0.40])
    panel_curves(ax_b, "arrest", "Arrest probability $a(L)$")
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.03, 0.04), ncol=1, fontsize=7, handlelength=1.8, labelspacing=0.25)
    S.panel(ax_b, "b", dx=-0.22, dy=1.03)
    ax_c = fig.add_axes([0.375, 0.08, 0.215, 0.40])
    panel_curves(ax_c, "capacity_bits", "Memory capacity $C$ (bits)")
    ax_c.axhline(0.5, color=S.GREY, lw=0.5, ls=(0, (3, 2)))
    ax_c.text(11.9, 0.52, "0.5 bit", ha="right", va="bottom", fontsize=7, color=S.GREY)
    S.panel(ax_c, "c", dx=-0.22, dy=1.03)
    S.save(fig, FIGURES, "Figure2")
