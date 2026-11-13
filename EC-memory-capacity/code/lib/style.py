"""Shared figure style: restrained palette, serif mathematical type, thin open axes."""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#1a1a1a"
GREY = "#7d828a"
LIGHT = "#d5d8dc"
PALE = "#f0f1f3"
BLUE = "#2b5c8a"  # stored information, activation
RUST = "#b0522d"  # transient information, inhibition
SAND = "#c9a66b"  # third series, used sparingly

BITS = LinearSegmentedColormap.from_list("bits", ["#ffffff", "#c9d6e3", "#6f93b5", BLUE, "#16324d"])

NAME = {"CDKN1A": "p21", "CDKN1B": "p27", "CDKN2A": "p16", "Proliferation": "Prolif."}
MODULES = [
    ("Inputs", ["KLRB1", "IRF5", "STAT6", "TSC22D3"]),
    ("Interferon", ["IFNG", "IRF1", "CXCL10", "LTA", "SOCS1", "SOCS3"]),
    ("Inflammatory", ["NFKB1", "MAPK14", "IL6", "STAT3", "CCL2", "CCL4", "CCL5", "CXCL8", "TNFRSF11B"]),
    ("Growth and cell cycle", ["IGF1", "AKT2", "MYC", "CCND2", "CCND3", "CDKN1A", "CDKN1B", "CDKN2A", "KLF4", "SPDEF"]),
    ("Effectors", ["TFF3", "MMP2"]),
    ("Phenotypes", ["Proliferation", "Apoptosis", "EMT", "Invasion", "Migration", "Metastasis"]),
]


def label(node):
    return NAME.get(node, node)


def pulse_label(p):
    """'CDKN1B=1' -> 'p27 ON'; 'MYC=0 + CCND2=0' -> 'MYC + CCND2 OFF' style labels."""
    parts = [x.strip() for x in p.split("+")]
    names = [label(x.split("=")[0]) for x in parts]
    vals = [x.split("=")[1] for x in parts]
    if len(set(vals)) == 1:
        return " + ".join(names) + (" ON" if vals[0] == "1" else " OFF")
    return " + ".join(f"{n} {'ON' if v == '1' else 'OFF'}" for n, v in zip(names, vals))


def apply():
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def panel(ax, letter, dx=-0.02, dy=1.04):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, name + ".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(folder, name + ".png"), bbox_inches="tight", pad_inches=0.03, dpi=600)
    plt.close(fig)
