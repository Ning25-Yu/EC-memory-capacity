"""E2. Mechanistic dissection of proliferative memory by exact edge ablation, and the network's feedback circuits.
Manuscript: Figure 5A, Figure S11 and Table S6 (edge ablation and feedback circuits).

For every regulatory edge src -> tgt (src occurring in tgt's rule; input self-loops excluded) the edge is ablated by
replacing src with constant OFF in tgt's rule only (the edge no longer transmits; every other use of src is intact).
For each ablated network the following are recomputed EXACTLY (exact_engine):
  * C0101: start from the original proliferative fixed point, let the ablated network relax (no pulse) to its terminal
    classes; record the probability that it stays proliferative; from the proliferative terminal classes reached,
    apply a p27 (CDKN1B ON) and a p16 (CDKN2A ON) pulse (L = 8) and record durable arrest;
  * C1001: same, starting from the within-class stationary distribution of the original complex recurrent class,
    with a p21 (CDKN1A ON) and a p27 pulse.
An edge is called essential for a read-out if ablating it lowers that read-out by >= 0.5 (or abolishes the
proliferative state); the column <read-out>_status separates the two cases ('arrest abolished' versus
'proliferative state removed', where the read-out is undefined). Signed simple cycles (length <= 8) are enumerated on the full regulatory graph and on the
non-constant core of C0101 / C1001.
Outputs: results/tables/E2_edge_ablation.csv, results/tables/E2_feedback_circuits.csv"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import re

import networkx as nx
import numpy as np
import pandas as pd
import sympy

import ecbn
import exact_engine as E
from paths import IN_TAB, TAB

A = pd.read_csv(IN_TAB + "/A2_attractor_repertoire_16_contexts.csv")
TOK = re.compile(r"(!?)([A-Za-z_][A-Za-z0-9_]*)")
EDGES = []
for tgt in ecbn.NODES:
    for sign, src in TOK.findall(ecbn.RULES[tgt]):
        if src in ecbn.NODES and not (src == tgt and tgt in ecbn.INPUTS):
            EDGES.append((src, tgt, "-" if sign else "+"))
EDGES = sorted(set(EDGES))
PAIRS = sorted({(s, t) for s, t, _ in EDGES})
ORIG = {k: dict(getattr(E, k)) for k in ("PYR", "REG", "_EXPR")}


def ablate(src, tgt):
    rule = re.sub(r"\b%s\b" % src, "ZZZ", ecbn.RULES[tgt])
    E.PYR[tgt] = compile(
        rule.replace("!", " not ").replace("&", " and ").replace("|", " or ").replace("ZZZ", "False").strip(),
        tgt,
        "eval",
    )
    E.REG[tgt] = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rule)) & set(ecbn.NODES)
    E._EXPR[tgt] = ORIG["_EXPR"][tgt].subs(E._SYM[src], sympy.false)


def restore():
    for k, v in ORIG.items():
        getattr(E, k).clear()
        getattr(E, k).update(v)


def relax_and_pulse(ctx, start_full, pulses, L=8.0):
    """start_full: dict(full-state dict as tuple of items) -> prob. Returns proliferative mass and arrest per pulse."""
    out = {}
    for node in pulses:
        sp = E.Space(ctx, (node,))
        if "Proliferation" not in sp.ix:  # ablation fixes Proliferation by percolation in this context
            on = sp.const["Proliferation"]
            out["proliferative_mass"] = float(on)
            out[f"{node}_arrest"] = np.nan if not on else 0.0
            continue
        st = {}
        for fs, w in start_full.items():
            d = dict(fs)
            d.update({k: v for k, v in sp.const.items()})
            key = tuple(int(d[n]) for n in sp.nodes)
            st[key] = st.get(key, 0.0) + w
        o, pos, Q, Aa = sp.explore(list(st), set())
        p0 = np.zeros(len(o))
        for s, w in st.items():
            p0[pos[s]] += w
        cls = E.terminal_classes(Aa)
        ab = E.absorption(Q, cls, p0)
        prol = sp.ix.get("Proliferation")
        pro_start = {}
        for c, m in enumerate(cls):
            pi = E.stationary(Q, m)
            vals = {o[i][prol] for i in m}
            if 1 in vals and ab[c] > 1e-12:
                for i, p in zip(m, pi):
                    pro_start[o[i]] = pro_start.get(o[i], 0.0) + ab[c] * p
        mass = sum(pro_start.values())
        out["proliferative_mass"] = mass
        if mass < 1e-9:
            out[f"{node}_arrest"] = np.nan
            continue
        pro_start = {k: v / mass for k, v in pro_start.items()}
        r = E.exact_pulse(sp, pro_start, {node: 1}, L, cap=300_000)
        out[f"{node}_arrest"] = r["P_arrest_FP"] if r else np.nan
    return out


def full_states_fixed(row):
    return {tuple((n, int(row[n])) for n in ecbn.NODES): 1.0}


def full_states_class(ctx, row):
    sp = E.Space(ctx, ())
    mem, pi, _ = E.complex_class(sp, row)
    out = {}
    for s, p in zip(mem, pi):
        d = {n: int(row[n]) if str(row[n]).upper() != "X" else 0 for n in ecbn.NODES}
        d.update(sp.const)
        d.update(dict(zip(sp.nodes, s)))
        out[tuple((n, d[n]) for n in ecbn.NODES)] = p
    return out


if __name__ == "__main__":
    fp = A[(A.context == "C0101") & (A.Proliferation.astype(str) == "1")].iloc[0]
    cx = A[(A.context == "C1001") & (A.type.astype(str).str.startswith("complex"))].iloc[0]
    S_fp = full_states_fixed(fp)
    S_cx = full_states_class("C1001", cx)
    base_fp = relax_and_pulse("C0101", S_fp, ["CDKN1B", "CDKN2A"])
    base_cx = relax_and_pulse("C1001", S_cx, ["CDKN1A", "CDKN1B"])
    print("baseline C0101", base_fp, "| C1001", base_cx, flush=True)
    rows = []
    for k, (s, t) in enumerate(PAIRS):
        sign = "/".join(sorted({g for a, b, g in EDGES if (a, b) == (s, t)}))
        ablate(s, t)
        try:
            a = relax_and_pulse("C0101", S_fp, ["CDKN1B", "CDKN2A"])
            b = relax_and_pulse("C1001", S_cx, ["CDKN1A", "CDKN1B"])
        finally:
            restore()
        rows.append(
            {
                "source": s,
                "target": t,
                "sign": sign,
                "C0101_proliferative_mass": a["proliferative_mass"],
                "C0101_p27_arrest": a["CDKN1B_arrest"],
                "C0101_p16_arrest": a["CDKN2A_arrest"],
                "C1001_proliferative_mass": b["proliferative_mass"],
                "C1001_p21_arrest": b["CDKN1A_arrest"],
                "C1001_p27_arrest": b["CDKN1B_arrest"],
            }
        )
        if k % 20 == 0:
            print(
                k, s, t, {kk: round(v, 3) for kk, v in rows[-1].items() if isinstance(v, float)}, flush=True
            )
    D = pd.DataFrame(rows)
    for col, ref in [
        ("C0101_p27_arrest", base_fp["CDKN1B_arrest"]),
        ("C0101_p16_arrest", base_fp["CDKN2A_arrest"]),
        ("C1001_p21_arrest", base_cx["CDKN1A_arrest"]),
        ("C1001_p27_arrest", base_cx["CDKN1B_arrest"]),
    ]:
        D[col.replace("_arrest", "_essential")] = D[col].fillna(0) <= ref - 0.5
        D[col.replace("_arrest", "_status")] = np.where(
            D[col].isna(),
            "proliferative state removed",
            np.where(D[col] <= ref - 0.5, "arrest abolished", "not essential"),
        )
    D.to_csv(TAB + "/E2_edge_ablation.csv", index=False)
    # signed feedback circuits
    G = nx.DiGraph()
    for s, t, g in EDGES:
        G.add_edge(s, t, sign=g)
    circ = []
    core01 = set(E.Space("C0101", ("CDKN1B", "CDKN2A")).nodes)
    core1001 = set(E.Space("C1001", ("CDKN1A",)).nodes)
    for cyc in nx.simple_cycles(G, length_bound=8):
        signs = [G[cyc[i]][cyc[(i + 1) % len(cyc)]]["sign"] for i in range(len(cyc))]
        circ.append(
            {
                "length": len(cyc),
                "nodes": " -> ".join(cyc + [cyc[0]]),
                "negative_edges": signs.count("-"),
                "type": "positive" if signs.count("-") % 2 == 0 else "negative",
                "in_C0101_core": set(cyc) <= core01,
                "in_C1001_core": set(cyc) <= core1001,
            }
        )
    C = pd.DataFrame(circ).sort_values(["length", "type"])
    C.to_csv(TAB + "/E2_feedback_circuits.csv", index=False)
    print(
        "edges ablated:",
        len(D),
        "| essential for C0101 p27 memory:",
        D.C0101_p27_essential.sum(),
        "| p16:",
        D.C0101_p16_essential.sum(),
        "| C1001 p21:",
        D.C1001_p21_essential.sum(),
    )
    print("circuits:", len(C), C.type.value_counts().to_dict(), "| in C0101 core:", C.in_C0101_core.sum())
