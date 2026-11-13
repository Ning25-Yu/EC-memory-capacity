"""A1/A2. Attractor repertoire of the 37-node network in the 16 input contexts.
Manuscript: Figure 2B (34 attractors of the 16 input contexts), Figure S5.

For each context (KLRB1, IRF5, STAT6, TSC22D3 clamped), attractors of the asynchronous dynamics are obtained in three ways:
  1. minimal trap spaces (most-permissive semantics, mpbn); every asynchronous attractor lies in a trap space and
     every minimal trap space contains at least one attractor;
  2. exact asynchronous state-transition graph inside each minimal trap space (all 2^k states of its k free nodes):
     terminal strongly connected components = attractors; a single state is a fixed point, otherwise a complex
     recurrent class (CRC; its free nodes are written 'X');
  3. independent checks: the fixed points returned by mpbn.fixedpoints(), and the long-run MaBoSS joint states
     (t = 1000) of every context must all be accounted for by the attractors found.
Attractor labels inside a context (C0101_A1, ...) keep the numbering of the published repertoire. The sort key that
reproduces it is: CCND3 descending, CCL2 descending, AKT2 ascending, then all other nodes alphabetically descending,
with X > 1 > 0. Labels are names only; no result depends on them.

Outputs: results/tables/A2_attractor_repertoire_16_contexts.csv (one row per attractor, node columns 0 / 1 / X),
         results/tables/A1_attractor_counts_16_contexts.csv (counts per context and the result of the independent checks).
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import mpbn
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import ecbn
from paths import TAB

NODES = ecbn.NODES
PYR = {
    n: compile(
        ecbn.RULES[n].replace("!", " not ").replace("&", " and ").replace("|", " or ").strip(), n, "eval"
    )
    for n in NODES
}
CTX = [f"C{i:04b}" for i in range(16)]


def context_network(ctx):
    bits = ecbn.ctx_bits(ctx)
    return {n: (bool(bits[n]) if n in ecbn.INPUTS else ecbn.RULES[n]) for n in NODES}


def async_attractors_in(ts):
    """Terminal SCCs of the asynchronous STG restricted to the trap space ts (dict node -> 0/1/'*')."""
    free = [n for n in NODES if ts[n] == "*"]
    fixed = {n: int(ts[n]) for n in NODES if ts[n] != "*"}
    if not free:
        return [[dict(fixed)]]
    states = list(itertools.product([0, 1], repeat=len(free)))
    pos = {s: i for i, s in enumerate(states)}
    I, J = [], []
    for s in states:
        env = dict(fixed)
        env.update(zip(free, s))
        for k, n in enumerate(free):
            t = int(eval(PYR[n], {}, env))
            if t != s[k]:
                u = list(s)
                u[k] = t
                I.append(pos[s])
                J.append(pos[tuple(u)])
    A = coo_matrix((np.ones(len(I)), (I, J)), shape=(len(states),) * 2).tocsr()
    nc, lab = connected_components(A, directed=True, connection="strong")
    out = []
    for c in range(nc):
        m = np.where(lab == c)[0]
        if np.all(np.isin(A[m].nonzero()[1], m)):
            out.append([{**fixed, **dict(zip(free, states[i]))} for i in m])
    return out


VAL = {"0": 0, "1": 1, "X": 2}


def label_key(a):
    rest = [n for n in NODES if n not in ("CCND3", "CCL2", "AKT2")]
    return (-VAL[a["CCND3"]], -VAL[a["CCL2"]], VAL[a["AKT2"]]) + tuple(-VAL[a[n]] for n in rest)


def summarise(members):
    row = {}
    for n in NODES:
        v = {st[n] for st in members}
        row[n] = str(v.pop()) if len(v) == 1 else "X"
    return row


def main():
    rows, counts = [], []
    for ctx in CTX:
        net = context_network(ctx)
        f = mpbn.MPBooleanNetwork(net)
        tss = [dict(t) for t in f.attractors()]
        atts = []
        for ts in tss:
            for members in async_attractors_in(ts):
                r = summarise(members)
                r["type"] = "fixed point" if len(members) == 1 else "complex recurrent class"
                r["n_states"] = len(members)
                atts.append(r)
        # independent check 1: fixed points
        fps = {tuple(int(v) for v in (fp[n] for n in NODES)) for fp in f.fixedpoints()}
        found_fp = {tuple(int(a[n]) for n in NODES) for a in atts if a["type"] == "fixed point"}
        assert fps == found_fp, f"{ctx}: fixed-point sets differ"
        atts.sort(key=label_key)
        for k, a in enumerate(atts, 1):
            a.update(context=ctx, attractor=f"{ctx}_A{k}")
            rows.append(a)
        counts.append(
            {
                "context": ctx,
                "n_attractors": len(atts),
                "n_fixed_points": sum(a["type"] == "fixed point" for a in atts),
                "n_complex_recurrent_classes": sum(a["type"] != "fixed point" for a in atts),
                "n_minimal_trap_spaces": len(tss),
                "n_fixed_points_mpbn": len(fps),
            }
        )
    A = pd.DataFrame(rows)[["context", "attractor", "type", "n_states"] + NODES]
    C = pd.DataFrame(counts)
    # independent check 2: every long-run MaBoSS joint state (t = 1000, all nodes observed) lies in an attractor
    ok = []
    for ctx in CTX:
        init = {k: float(v) for k, v in ecbn.ctx_bits(ctx).items()}
        dist, _ = ecbn.simulate(init=init, outputs=NODES, max_time=1000, time_tick=10, sample_count=2000)
        sub = A[A.context == ctx]
        bad = 0.0
        for st, p in dist.items():
            on = set(st.split(" -- ")) if st != "<nil>" else set()
            full = {n: int(n in on) for n in NODES}
            hit = any(all(r[n] == "X" or int(r[n]) == full[n] for n in NODES) for _, r in sub.iterrows())
            if not hit:
                bad += p
        ok.append(bad)
    C["MaBoSS_mass_outside_attractors_t1000"] = ok
    A.to_csv(os.path.join(TAB, "A2_attractor_repertoire_16_contexts.csv"), index=False)
    C.to_csv(os.path.join(TAB, "A1_attractor_counts_16_contexts.csv"), index=False)
    print(C.to_string(index=False))
    print(
        "total attractors",
        len(A),
        "| fixed points",
        (A.type == "fixed point").sum(),
        "| complex recurrent classes",
        (A.type != "fixed point").sum(),
    )


if __name__ == "__main__":
    main()
