"""E4. Exact time course of a finite pulse: during the clamp (0 <= t <= L) and after withdrawal (L < t <= L + 80).
Manuscript: Figure 5D-E and Figure S13 (time courses during and after the pulse).

For selected (context, attractor, pulse) cases the probability distribution over the tracked state space is
propagated exactly with the sparse matrix exponential (phase 1 under the clamp, phase 2 on the released chain).

Three quantities are recorded at every time point, because reaching the arrested fixed point and having the arrest
fate decided are different events:
  * P_prolif_on        - probability that Proliferation is ON;
  * P_at_core_arrest_FP - probability of being in a Proliferation-OFF fixed point of the tracked Proliferation core
                         (core fixed-point arrival; the seven acyclic downstream nodes may still be relaxing);
  * P_at_full_network_arrest_FP - the same for the full network (all non-constant nodes tracked);
  * P_fate_sealed      - probability of being in a state from which the RELEASED network reaches a Proliferation-OFF
                         fixed point with probability 1 (h(s) >= 1 - 1e-9), i.e. arrest is already certain if the
                         pulse were withdrawn now;
  * P_sealed_non_arrest / P_fate_undecided - mass with h(s) = 0 / with 0 < h(s) < 1;
  * expected_arrest    - sum_s p(s, t) h(s): the eventual arrest if the pulse were withdrawn at time t (during the
                         clamp this is the exact duration response; after withdrawal it is constant).
h(s) is the exact absorption probability of the released chain into Proliferation-OFF fixed points, computed on the
union of all states visited under the clamp and their released descendants.
Summary per case: eventual arrest; at withdrawal the core and full-network fixed-point mass, the arrest-sealed,
non-arrest-sealed and undecided masses, and the arrest later contributed by undecided states (eventual - sealed);
core and full-network fixed-point half-times after withdrawal (first time point at which half the eventual arrest is
reached, read on the 0.5-unit output grid; not a continuous-time root).
Outputs: results/tables/E4_pulse_kinetics_timecourse.csv, results/tables/E4_pulse_kinetics_summary.csv"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
from scipy.sparse import identity
from scipy.sparse.linalg import expm_multiply, spsolve

import exact_engine as E
from paths import IN_TAB, TAB

A = pd.read_csv(IN_TAB + "/A2_attractor_repertoire_16_contexts.csv")
L = 8.0
T2 = 80.0
CASES = [
    ("C0101_A1", "CDKN1B", 1),
    ("C0101_A1", "CDKN2A", 1),
    ("C0101_A1", "CDKN1A", 1),
    ("C0101_A1", "MYC", 0),
    ("C1001_A1", "CDKN1A", 1),
    ("C1001_A1", "CDKN1B", 1),
    ("C1001_A1", "NFKB1", 0),
    ("C1001_A1", "CCND2", 0),
    ("C1100_A1", "MYC", 0),
    ("C1100_A1", "IL6", 0),
    ("C0100_A1", "IL6", 0),
    ("C1101_A1", "CDKN1B", 1),
    ("C0101_A1", "MYC+CCND2", (0, 0)),
    ("C0101_A1", "MYC+CDKN1A", (0, 1)),
]  # the last two are emergent pairs (E3)


def arrest_hitting(sp, o, pos, Q, A_):
    """Exact probability h(s) that the released chain from s is absorbed in a Proliferation-OFF fixed point."""
    n = len(o)
    prol = sp.ix["Proliferation"]
    cls = E.terminal_classes(A_)
    target = np.zeros(n, bool)
    terminal = np.zeros(n, bool)
    for m in cls:
        terminal[m] = True
        if len(m) == 1 and o[m[0]][prol] == 0:
            target[m[0]] = True
    T = np.where(~terminal)[0]
    h = target.astype(float)
    if len(T) == 0:
        return h
    QT = Q[T][:, T].tocsc()
    b = -(Q[T][:, np.where(target)[0]] @ np.ones(target.sum()))
    if len(T) <= 20000:
        h[T] = spsolve(QT, b)
    else:  # monotone iteration of the uniformised chain; stops when successive iterates differ
        # by < 1e-13 (a stopping tolerance, not an absolute error bound)
        lam = float(-Q.diagonal().min())
        P = identity(n, format="csr") + Q / lam
        x = target.astype(float)
        for _ in range(200000):
            y = P @ x
            y[target] = 1.0
            y[terminal & ~target] = 0.0
            if np.abs(y - x).max() < 1e-13:
                x = y
                break
            x = y
        h = x
    return np.clip(h, 0, 1)


def full_network_arrival(row, nodes, vals, cap=3_000_000):
    """P(in a Proliferation-OFF fixed point of the FULL network) on the clamp and release grids (all non-constant nodes
    tracked, including the seven acyclic downstream nodes). Returns (clamp array, release array) or (None, None).
    """
    sp = E.Space(row.context, tuple(nodes), scope="all")
    start, _, _, _ = E.attractor_start(sp, row)
    clamp = {n: v for n, v in zip(nodes, vals) if n in sp.ix}
    p1 = E._onset(sp, start, clamp)
    r1 = sp.explore(list(p1), set(clamp), cap)
    if r1 is None:
        return None, None
    o1, pos1, Q1, _ = r1
    v0 = np.zeros(len(o1))
    for st, w in p1.items():
        v0[pos1[st]] = w
    rR = sp.explore(list(o1), set(), cap)
    if rR is None:
        return None, None
    oR, posR, QR, _ = rR
    prol = sp.ix["Proliferation"]
    fp = np.array([(len(sp.successors(st, set())) == 0) and st[prol] == 0 for st in oR])
    m1 = np.array([posR[st] for st in o1])
    V1 = expm_multiply(Q1.T, v0, start=0, stop=L, num=33, endpoint=True)
    pR = np.zeros(len(oR))
    pR[m1] = np.clip(V1[-1], 0, None)
    V2 = expm_multiply(QR.T, pR, start=0, stop=T2, num=161, endpoint=True)
    return V1 @ fp[m1], V2 @ fp


def course(row, node, val):
    nodes = node.split("+")
    vals = val if isinstance(val, tuple) else (val,)
    ctx = row.context
    sp = E.Space(ctx, tuple(nodes))
    start, kind, _, _ = E.attractor_start(sp, row)
    clamp = {n: v for n, v in zip(nodes, vals) if n in sp.ix}
    p1 = E._onset(sp, start, clamp)
    o1, pos1, Q1, _ = sp.explore(list(p1), set(clamp))
    v0 = np.zeros(len(o1))
    for s, w in p1.items():
        v0[pos1[s]] = w
    # released chain on every state reachable from any state visited under the clamp
    oR, posR, QR, AR = sp.explore(list(o1), set())
    h = arrest_hitting(sp, oR, posR, QR, AR)
    sealed = h >= 1 - 1e-9
    prol = sp.ix["Proliferation"]
    fpoff = np.array([(len(sp.successors(s, set())) == 0) and s[prol] == 0 for s in oR])
    m1 = np.array([posR[s] for s in o1])
    pr1 = np.array([s[prol] for s in o1])
    t1 = np.linspace(0, L, 33)
    V1 = expm_multiply(Q1.T, v0, start=0, stop=L, num=33, endpoint=True)
    vL = np.clip(V1[-1], 0, None)
    pR = np.zeros(len(oR))
    pR[m1] = vL
    t2 = np.linspace(0, T2, 161)
    V2 = expm_multiply(QR.T, pR, start=0, stop=T2, num=161, endpoint=True)
    prR = np.array([s[prol] for s in oR])
    sealed0 = h <= 1e-9
    undecided = ~sealed & ~sealed0
    F1, F2 = full_network_arrival(row, nodes, vals)
    nanv = lambda F, k: float(F[k]) if F is not None else np.nan
    rows = [
        {
            "phase": "clamp",
            "t": t,
            "P_prolif_on": float(V1[k] @ pr1),
            "P_at_core_arrest_FP": float(V1[k] @ fpoff[m1]),
            "P_at_full_network_arrest_FP": nanv(F1, k),
            "P_fate_sealed": float(V1[k] @ sealed[m1]),
            "P_sealed_non_arrest": float(V1[k] @ sealed0[m1]),
            "P_fate_undecided": float(V1[k] @ undecided[m1]),
            "expected_arrest": float(V1[k] @ h[m1]),
        }
        for k, t in enumerate(t1)
    ]
    rows += [
        {
            "phase": "released",
            "t": L + t,
            "P_prolif_on": float(V2[k] @ prR),
            "P_at_core_arrest_FP": float(V2[k] @ fpoff),
            "P_at_full_network_arrest_FP": nanv(F2, k),
            "P_fate_sealed": float(V2[k] @ sealed),
            "P_sealed_non_arrest": float(V2[k] @ sealed0),
            "P_fate_undecided": float(V2[k] @ undecided),
            "expected_arrest": float(V2[k] @ h),
        }
        for k, t in enumerate(t2)
    ]
    lab = " + ".join(f"{n}={v}" for n, v in zip(nodes, vals))
    D = pd.DataFrame(rows)
    D.insert(0, "case", f"{row.attractor} | {lab}")
    fin = E.exact_pulse(sp, start, clamp, L)
    assert abs(fin["P_arrest_FP"] - float(vL @ h[m1])) < 1e-8, (lab, fin["P_arrest_FP"], float(vL @ h[m1]))
    rel = D[D.phase == "released"]
    on_rel = rel.P_prolif_on.iloc[0]
    t50 = (
        float(rel.t[rel.P_at_core_arrest_FP >= 0.5 * fin["P_arrest_FP"]].iloc[0] - L)
        if fin["P_arrest_FP"] > 1e-6
        else np.nan
    )
    if F2 is not None and fin["P_arrest_FP"] > 1e-6:
        assert abs(F2[-1] - fin["P_arrest_FP"]) < 1e-3, (lab, F2[-1], fin["P_arrest_FP"])
        t50f = float(rel.t[rel.P_at_full_network_arrest_FP >= 0.5 * fin["P_arrest_FP"]].iloc[0] - L)
    else:
        t50f = np.nan
    S = {
        "case": f"{row.attractor} | {lab}",
        "context": ctx,
        "attractor_type": kind,
        "pulse": lab,
        "P_prolif_on_before": float(v0 @ pr1),
        "P_prolif_off_at_withdrawal": 1 - on_rel,
        "at_core_arrest_FP_at_withdrawal": float(rel.P_at_core_arrest_FP.iloc[0]),
        "at_full_network_arrest_FP_at_withdrawal": float(rel.P_at_full_network_arrest_FP.iloc[0]),
        "fate_sealed_at_withdrawal": float(rel.P_fate_sealed.iloc[0]),
        "sealed_non_arrest_at_withdrawal": float(rel.P_sealed_non_arrest.iloc[0]),
        "undecided_mass_at_withdrawal": float(rel.P_fate_undecided.iloc[0]),
        "arrest_contributed_by_undecided_states": float(fin["P_arrest_FP"] - rel.P_fate_sealed.iloc[0]),
        "eventual_arrest": fin["P_arrest_FP"],
        "core_FP_t50_after_withdrawal": t50,
        "full_network_FP_t50_after_withdrawal": t50f,
        "P_prolif_on_t80_after": float(rel.P_prolif_on.iloc[-1]),
        "relapse_after_withdrawal": float(rel.P_prolif_on.max() - on_rel),
        "states_clamp/released": f"{len(o1)}/{len(oR)}",
    }
    return D, S


if __name__ == "__main__":
    out, summ = [], []
    for att, node, val in CASES:
        row = A[A.attractor == att].iloc[0]
        D, S = course(row, node, val)
        out.append(D)
        summ.append(S)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in S.items()}, flush=True)
    pd.concat(out).to_csv(TAB + "/E4_pulse_kinetics_timecourse.csv", index=False)
    pd.DataFrame(summ).to_csv(TAB + "/E4_pulse_kinetics_summary.csv", index=False)
