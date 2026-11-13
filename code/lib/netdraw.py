"""Network diagrams with orthogonal (horizontal and vertical) edges, built on orthonet.route.

All coordinates are inches. net_axes() returns an axes whose data units are inches, so that line widths, node sizes
and font sizes are the same in every figure. The 37 nodes keep one position in every drawing of the full network
(module columns: inputs | interferon | inflammatory | growth and cell cycle | effectors | phenotypes).
"""

import re

import matplotlib.patches as mp

import style as S
from orthonet import Box, draw_edge, route
from paths import DATA

INPUTS = ["KLRB1", "IRF5", "STAT6", "TSC22D3"]
PHENO = ["Proliferation", "Apoptosis", "EMT", "Invasion", "Migration", "Metastasis"]
ACT, INH = S.BLUE, S.RUST

COLS = [
    ["KLRB1", "IRF5", "STAT6", "TSC22D3"],
    ["IFNG", "IRF1", "CXCL10", "LTA", "SOCS1", "SOCS3"],
    ["NFKB1", "MAPK14", "CCL2", "CCL5", "TNFRSF11B"],
    ["IL6", "STAT3", "CCL4", "CXCL8"],
    ["IGF1", "MYC", "CDKN1A", "CDKN2A", "CCND3", "SPDEF"],
    ["AKT2", "CDKN1B", "CCND2", "KLF4"],
    ["TFF3", "MMP2"],
    ["Metastasis", "Migration", "Invasion", "EMT", "Proliferation", "Apoptosis"],
]
ROWS = [
    [1, 3, 5, 7],
    [0, 2, 4, 6, 8, 10],
    [0, 2, 4, 6, 8],
    [1, 3, 5, 7],
    [0, 2, 4, 6, 8, 10],
    [1, 5, 7, 9],
    [2, 4],
    [0, 2, 4, 6, 8, 10],
]
BANDS = [
    ([0], "Inputs"),
    ([1], "Interferon"),
    ([2, 3], "Inflammatory"),
    ([4, 5], "Growth and cell cycle"),
    ([6], "Effectors"),
    ([7], "Phenotypes"),
]
GAPS = [0.52, 0.42, 0.17, 0.42, 0.17, 0.40, 0.45]


def signed_edges():
    """The 168 signed interactions (regulator, target, inhibitory) read from the logical rules."""
    out = []
    for line in open(DATA + "/EC_network.bnet"):
        if "," not in line:
            continue
        target, rule = [x.strip() for x in line.split(",", 1)]
        if rule == target:
            continue
        seen = {}
        for neg, src in re.findall(r"(!?)([A-Za-z_][A-Za-z0-9_]*)", rule):
            seen[src] = bool(neg)
        out += [(src, target, neg) for src, neg in seen.items()]
    return sorted(out)


def signed_nodes():
    return [n for c in COLS for n in c]


def net_axes(fig, x, y, w, h):
    """Axes at (x, y) with size (w, h), all in inches; data coordinates are inches."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


def text_width(lab, fs, pad=0.07):
    return fs / 72 * 0.56 * len(lab) + pad


def node(ax, b, text, fc="white", ec=S.INK, tc=S.INK, fs=5.8, lw=0.55, ls="-", zorder=3):
    ax.add_patch(
        mp.FancyBboxPatch((b.l, b.y - b.h / 2), b.w, b.h, boxstyle="round,pad=0,rounding_size=0.02",
                          fc=fc, ec=ec, lw=lw, ls=ls, zorder=zorder)
    )
    ax.text(b.x, b.y - 0.003, text, ha="center", va="center", fontsize=fs, color=tc, zorder=zorder + 1)


def full_layout(W, py=0.205, fs=5.8, h=0.125, head=0.30, foot=0.10):
    """Node boxes of the full network for a panel of width W. Returns (boxes, bands, H)."""
    lab = S.label
    wcol = [max(text_width(lab(n), fs) for n in c) for c in COLS]
    wcol[-1] = max(wcol[-1], 0.62)
    xs = [0.10 + wcol[0] / 2]
    for i in range(1, len(COLS)):
        xs.append(xs[-1] + wcol[i - 1] / 2 + GAPS[i - 1] + wcol[i] / 2)
    scale = (W - 0.10) / (xs[-1] + wcol[-1] / 2)
    xs = [x * scale for x in xs]
    H = head + 10 * py + h + foot
    y0 = H - head - h / 2
    boxes = {}
    for c, (names, rr) in enumerate(zip(COLS, ROWS)):
        for n, r in zip(names, rr):
            boxes[n] = Box(xs[c], y0 - r * py, wcol[c], h)
    bands = []
    for cc, name in BANDS:
        lo = min(xs[c] - wcol[c] / 2 for c in cc) - 0.06
        hi = max(xs[c] + wcol[c] / 2 for c in cc) + 0.06
        bands.append((lo, hi, y0 - 10 * py - h / 2 - 0.05, y0 + h / 2 + 0.05, name))
    return boxes, bands, H


def draw_bands(ax, bands, fs=6.6, labels=True):
    for lo, hi, ylo, yhi, name in bands:
        ax.add_patch(mp.FancyBboxPatch((lo, ylo), hi - lo, yhi - ylo, boxstyle="round,pad=0,rounding_size=0.04",
                                       fc=S.PALE, ec="none", zorder=0))
        if labels:
            ax.text((lo + hi) / 2, yhi + 0.035, name, ha="center", va="bottom", fontsize=fs, style="italic")


def draw_edges(ax, boxes, edges, color=None, lw=0.32, head=0.022, alpha=0.95, lane_sep=0.012, track_sep=0.012,
               margin=0.12, port_frac=0.70, styles=None):
    """Route and draw edges. color: None (blue activation, rust inhibition) or one colour for all.
    styles: optional {(source, target): dict(lw=..., ls=..., color=...)} overrides."""
    edges = [e for e in edges if e[0] in boxes and e[1] in boxes and e[0] != e[1]]
    R = route(boxes, edges, margin=margin, lane_sep=lane_sep, track_sep=track_sep, port_frac=port_frac)
    for e in sorted(edges, key=lambda e: e[2]):
        st = dict(lw=lw, ls="-", color=color or (INH if e[2] else ACT))
        st.update((styles or {}).get((e[0], e[1]), {}))
        draw_edge(ax, R[e], e[2], st["color"], lw=st["lw"], ls=st["ls"], head=head * (1.25 if st["lw"] > 2 * lw else 1),
                  alpha=alpha, zorder=2.5 if st["lw"] > 2 * lw else 2)
    return R


def default_node_style(n):
    """(face, edge, text colour, line width, line style)."""
    if n in INPUTS:
        return S.INK, S.INK, "white", 0.55, "-"
    if n in PHENO:
        return "white", S.INK, S.INK, 1.0, "-"
    return "white", S.INK, S.INK, 0.55, "-"


def draw_nodes(ax, boxes, style=default_node_style, fs=5.8):
    for n, b in boxes.items():
        fc, ec, tc, lw, ls = style(n)
        node(ax, b, S.label(n), fc, ec, tc, fs, lw, ls)


# ------------------------------------------------------------------ the seven-node switch of C0101
SWITCH_EDGES = [
    ("CDKN1B", "MYC", True), ("CDKN1B", "CCND2", True), ("CDKN1B", "CCND3", True), ("CDKN1B", "Proliferation", True),
    ("MYC", "CDKN1B", True), ("CCND2", "CDKN1B", True), ("CCND3", "CDKN1B", True), ("AKT2", "CDKN1B", True),
    ("MYC", "AKT2", False), ("AKT2", "MYC", False), ("MYC", "CCND2", False), ("MYC", "CCND3", False),
    ("AKT2", "CCND2", False), ("IRF1", "CDKN1B", False), ("IRF1", "MYC", False), ("MYC", "IRF1", True),
    ("IRF1", "CCND3", True), ("CCND2", "Proliferation", False), ("CCND3", "Proliferation", False),
    ("MYC", "Proliferation", False),
]
SWITCH_GRID = {"IRF1": (0, 0), "MYC": (0, 1), "AKT2": (0, 2), "CDKN1B": (1, 1), "CCND3": (2, 0), "CCND2": (2, 2),
               "Proliferation": (3, 1)}


def switch_layout(W, H, bw=0.50, bh=0.19, mx=0.14):
    px = (W - 2 * mx - bw) / 3
    py = (H - bh - 0.24) / 2
    return {n: Box(mx + bw / 2 + c * px, H - 0.12 - bh / 2 - r * py, bw, bh) for n, (c, r) in SWITCH_GRID.items()}


def draw_switch(ax, W, H, state=None, highlight=(), dashed=(), fs=7.2):
    """The switch with orthogonal edges. state: {node: 0/1} fills ON nodes; highlight: edges drawn thick;
    dashed: edges drawn dashed."""
    boxes = switch_layout(W, H)
    styles = {e: dict(lw=1.9) for e in highlight}
    styles.update({e: dict(ls=(0, (2.2, 1.6))) for e in dashed})
    draw_edges(ax, boxes, SWITCH_EDGES, lw=0.7, head=0.04, lane_sep=0.05, track_sep=0.045, margin=0.12,
               port_frac=0.72, styles=styles)
    for n, b in boxes.items():
        on = (state or {}).get(n, 0)
        node(ax, b, S.label(n), S.INK if on else "white", S.INK, "white" if on else S.INK, fs, 0.7)
    return boxes
