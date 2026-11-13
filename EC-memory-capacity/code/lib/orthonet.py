"""Orthogonal (horizontal/vertical) drawing of small signed networks, in the style of the network overview (Figure 2A).

Nodes sit on a grid of columns and rows (coordinates in inches). Every edge leaves its source from the side that faces
the target, runs along a vertical lane in the gap next to the source column, along a horizontal channel between two rows
when it has to pass other columns, along a vertical lane in the gap next to the target column, and enters the target
from the side that faces the source. Lanes and channel tracks are assigned so that parallel segments never overlap,
and each node side spreads its ports over the node height.

route(nodes, edges) -> {edge: [(x, y), ...]} polylines (first point on the source border, last on the target border).
draw_edge(ax, pts, inhibitory, **style) draws one polyline with an arrowhead (activation) or a bar (inhibition).
"""

from collections import defaultdict

import numpy as np


class Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def l(self):
        return self.x - self.w / 2

    @property
    def r(self):
        return self.x + self.w / 2


def _cols(nodes):
    xs = sorted({round(b.x, 4) for b in nodes.values()})
    col = {n: xs.index(round(b.x, 4)) for n, b in nodes.items()}
    left = [min(b.l for n, b in nodes.items() if col[n] == c) for c in range(len(xs))]
    right = [max(b.r for n, b in nodes.items() if col[n] == c) for c in range(len(xs))]
    return xs, col, left, right


def _intervals(items, key_lo, key_hi, sep):
    """Greedy interval colouring: items (id, lo, hi) -> {id: track} so that overlapping intervals get different tracks."""
    tracks, out = [], {}
    for i, lo, hi in sorted(items, key=lambda t: (t[1], t[2])):
        for k, end in enumerate(tracks):
            if lo > end + sep:
                tracks[k] = hi
                out[i] = k
                break
        else:
            tracks.append(hi)
            out[i] = len(tracks) - 1
    return out, max(len(tracks), 1)


def route(nodes, edges, margin=0.10, lane_sep=0.016, track_sep=0.016, port_frac=0.62):
    """nodes: name -> Box; edges: list of (s, t, inhibitory)."""
    xs, col, left, right = _cols(nodes)
    nc = len(xs)
    # gap g lies between column g-1 and column g (g = 0 left of everything, g = nc right of everything)
    gap_lo = [left[0] - margin] + [right[c - 1] for c in range(1, nc)] + [right[-1]]
    gap_hi = [left[0]] + [left[c] for c in range(1, nc)] + [right[-1] + margin]
    rows = sorted({round(b.y, 4) for b in nodes.values()})
    hmax = max(b.h for b in nodes.values())
    # horizontal channels between rows (and one above / below)
    ch_y = (
        [rows[0] - (rows[1] - rows[0] if len(rows) > 1 else 0.2) / 2]
        + [(rows[i] + rows[i + 1]) / 2 for i in range(len(rows) - 1)]
        + [rows[-1] + (rows[-1] - rows[-2] if len(rows) > 1 else 0.2) / 2]
    )
    ch_w = [min((rows[1] - rows[0]) if len(rows) > 1 else 0.2, 1) - hmax] * len(ch_y)
    plan, direct = {}, {}

    def clear_h(s, t):
        a, b = nodes[s], nodes[t]
        if abs(a.y - b.y) > 1e-6:
            return False
        lo, hi = min(a.x, b.x), max(a.x, b.x)
        return not any(
            lo < o.x < hi and abs(o.y - a.y) < (o.h + a.h) / 2 for n, o in nodes.items() if n not in (s, t)
        )

    def clear_v(s, t):
        a, b = nodes[s], nodes[t]
        if abs(a.x - b.x) > 1e-6:
            return False
        lo, hi = min(a.y, b.y), max(a.y, b.y)
        return not any(
            lo < o.y < hi and abs(o.x - a.x) < (o.w + a.w) / 2 for n, o in nodes.items() if n not in (s, t)
        )

    for e in edges:
        s, t, _ = e
        if clear_v(s, t):
            direct[e] = "V"
    for e in edges:
        s, t, _ = e
        if e in direct:
            continue
        if clear_h(s, t):
            direct[e] = "H"
    for e in edges:
        s, t, _ = e
        if direct.get(e) == "V":
            continue
        cs, ct = col[s], col[t]
        if ct > cs:
            so, ti = "R", "L"
        elif ct < cs:
            so, ti = "L", "R"
        else:
            so = ti = "R" if cs < nc - 1 or nc == 1 else "L"
        g1 = cs + 1 if so == "R" else cs  # gap next to the source on its exit side
        g2 = ct if ti == "L" else ct + 1  # gap next to the target on its entry side
        plan[e] = dict(so=so, ti=ti, g1=g1, g2=g2)
    # ports: spread the edges of each node side over the node height, ordered by the y of the other end
    sides = defaultdict(list)
    for e, p in plan.items():
        s, t, _ = e
        if direct.get(e) == "H":
            p["g1"] = p["g2"] = None
        sides[(s, p["so"])].append((e, "out", nodes[t].y))
        sides[(t, p["ti"])].append((e, "in", nodes[s].y))
    for (n, side), lst in sides.items():
        lst.sort(key=lambda z: -z[2])
        k = len(lst)
        b = nodes[n]
        span = b.h * port_frac
        for i, (e, d, _) in enumerate(lst):
            yy = b.y + (span / 2 - i * span / (k - 1) if k > 1 else 0.0)
            plan[e]["ys" if d == "out" else "yt"] = yy
    # channels for edges whose two gaps differ
    hseg = defaultdict(list)
    for e, p in plan.items():
        if p["g1"] == p["g2"]:
            continue
        ys, yt = p["ys"], p["yt"]
        lo, hi = min(ys, yt), max(ys, yt)
        cands = [i for i, y in enumerate(ch_y) if lo - 1e-9 <= y <= hi + 1e-9] or [
            min(range(len(ch_y)), key=lambda i: abs(ch_y[i] - (ys + yt) / 2))
        ]
        # least used channel, then nearest the source
        p["ch"] = min(cands, key=lambda i: (len(hseg[i]), abs(ch_y[i] - ys)))
        x1 = (gap_lo[p["g1"]] + gap_hi[p["g1"]]) / 2
        x2 = (gap_lo[p["g2"]] + gap_hi[p["g2"]]) / 2
        hseg[p["ch"]].append((e, min(x1, x2), max(x1, x2)))
    for ch, items in hseg.items():
        tr, n = _intervals(items, 1, 2, 0.02)
        sep = min(track_sep, ch_w[ch] * 0.9 / max(n - 1, 1)) if n > 1 else 0
        for e, k in tr.items():
            plan[e]["yc"] = ch_y[ch] + (k - (n - 1) / 2) * sep
    # vertical lanes
    vseg = defaultdict(list)
    for e, p in plan.items():
        if p["g1"] is None:
            continue
        if p["g1"] == p["g2"]:
            vseg[p["g1"]].append(((e, 0), min(p["ys"], p["yt"]), max(p["ys"], p["yt"])))
        else:
            vseg[p["g1"]].append(((e, 1), min(p["ys"], p["yc"]), max(p["ys"], p["yc"])))
            vseg[p["g2"]].append(((e, 2), min(p["yt"], p["yc"]), max(p["yt"], p["yc"])))
    lane_x = {}
    for g, items in vseg.items():
        tr, n = _intervals(items, 1, 2, 0.012)
        w = gap_hi[g] - gap_lo[g]
        c = (gap_lo[g] + gap_hi[g]) / 2
        sep = min(lane_sep, w * 0.7 / max(n - 1, 1)) if n > 1 else 0
        # order lanes so that tracks keep a stable left-to-right order
        for key, k in tr.items():
            lane_x[key] = c + (k - (n - 1) / 2) * sep
    out = {}
    # straight vertical edges between neighbours of one column: leave the bottom / top, offset reciprocal pairs
    pairs = defaultdict(list)
    for e, m in direct.items():
        if m == "V":
            pairs[frozenset(e[:2])].append(e)
    for pr, lst in pairs.items():
        for i, e in enumerate(sorted(lst)):
            s, t, _ = e
            bs, bt = nodes[s], nodes[t]
            off = (i - (len(lst) - 1) / 2) * 0.36 * min(bs.w, bt.w)
            down = bt.y < bs.y
            y0 = bs.y - bs.h / 2 if down else bs.y + bs.h / 2
            y1 = bt.y + bt.h / 2 if down else bt.y - bt.h / 2
            out[e] = [(bs.x + off, y0), (bs.x + off, y1)]
    for e, p in plan.items():
        s, t, _ = e
        bs, bt = nodes[s], nodes[t]
        if p["g1"] is None:
            xs0 = bs.r if p["so"] == "R" else bs.l
            xt0 = bt.r if p["ti"] == "R" else bt.l
            out[e] = [(xs0, p["ys"]), (xt0, p["ys"])]
            continue
        xs0 = bs.r if p["so"] == "R" else bs.l
        xt0 = bt.r if p["ti"] == "R" else bt.l
        if p["g1"] == p["g2"]:
            xl = lane_x[(e, 0)]
            pts = [(xs0, p["ys"]), (xl, p["ys"]), (xl, p["yt"]), (xt0, p["yt"])]
        else:
            x1, x2 = lane_x[(e, 1)], lane_x[(e, 2)]
            pts = [(xs0, p["ys"]), (x1, p["ys"]), (x1, p["yc"]), (x2, p["yc"]), (x2, p["yt"]), (xt0, p["yt"])]
        # drop zero-length segments
        clean = [pts[0]]
        for q in pts[1:]:
            if abs(q[0] - clean[-1][0]) > 1e-6 or abs(q[1] - clean[-1][1]) > 1e-6:
                clean.append(q)
        out[e] = clean
    return out


def draw_edge(ax, pts, inhibitory, color, lw=0.45, ls="-", head=0.028, zorder=2, alpha=1.0):
    xs, ys = zip(*pts)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = np.array([x1 - x0, y1 - y0])
    d = d / (np.hypot(*d) or 1)
    nrm = np.array([-d[1], d[0]])
    if inhibitory:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=zorder, alpha=alpha)
        a = np.array([x1, y1]) - d * lw / 144
        b1, b2 = a + nrm * head * 0.9, a - nrm * head * 0.9
        ax.plot(
            [b1[0], b2[0]],
            [b1[1], b2[1]],
            color=color,
            lw=lw * 1.6,
            solid_capstyle="butt",
            zorder=zorder,
            alpha=alpha,
        )
    else:
        tip = np.array([x1, y1])
        base = tip - d * head * 1.5
        xs2 = list(xs[:-1]) + [base[0]]
        ys2 = list(ys[:-1]) + [base[1]]
        ax.plot(xs2, ys2, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=zorder, alpha=alpha)
        tri = np.array([tip, base + nrm * head * 0.65, base - nrm * head * 0.65])
        ax.fill(tri[:, 0], tri[:, 1], color=color, lw=0, zorder=zorder, alpha=alpha)
