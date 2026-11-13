"""Exact continuous-time Markov chain (CTMC) analysis of the native C0101 subspace.
Manuscript: Figure 3B (D3), Figure 3D (D4), Figure S8 (D5) and the model values compared in Figure 6.

With the four inputs fixed to C0101, logical propagation fixes 26 of the 33 non-input nodes; the remaining seven
variables (AKT2, CCND2, CCND3, CDKN1B, IRF1, MYC, Proliferation) span 2^7 = 128 states. The generator uses the ORIGINAL
37-node rules evaluated with the propagated constants (no rule is rewritten). Asynchronous update, unit rates
(or user-specified rates). Outputs: reduced-rule certificate, committor h(x) = P(eventual arrest | x), and exact
pulse-duration responses for clamps of the free molecular nodes, plus the equal-budget order experiment."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import itertools
import json

import numpy as np
import pandas as pd
from scipy.linalg import expm

import ecbn
from paths import TAB

OUT = TAB
CTX = "C0101"
FREE = ["AKT2", "CCND2", "CCND3", "CDKN1B", "IRF1", "MYC", "Proliferation"]


def ev(rule, s):
    return int(eval(rule.replace("!", " not ").replace("&", " and ").replace("|", " or "), {}, s))


def propagate(ctx):
    """Logical propagation of constants from the clamped inputs (fixpoint of rules whose value is determined)."""
    import sympy

    const = dict(ecbn.ctx_bits(ctx))
    changed = True
    syms = {n: sympy.Symbol(n) for n in ecbn.NODES}
    expr = {n: sympy.sympify(ecbn.RULES[n].replace("!", "~"), locals=syms) for n in ecbn.NODES}
    while changed:
        changed = False
        for n in ecbn.NODES:
            if n in const:
                continue
            e = expr[n].subs({syms[k]: bool(v) for k, v in const.items()})
            if e in (sympy.true, sympy.false):
                const[n] = int(bool(e))
                changed = True
    return const


CONST = propagate(CTX)
assert sorted(set(ecbn.NODES) - set(CONST)) == sorted(FREE), sorted(set(ecbn.NODES) - set(CONST))
STATES = list(itertools.product([0, 1], repeat=len(FREE)))
IDX = {s: i for i, s in enumerate(STATES)}


def full(s):
    d = dict(CONST)
    d.update(dict(zip(FREE, s)))
    return d


def generator(clamp=None, rates=None):
    clamp = clamp or {}
    rates = rates or {}
    n = len(STATES)
    Q = np.zeros((n, n))
    for s in STATES:
        x = full(s)
        i = IDX[s]
        for k, node in enumerate(FREE):
            if node in clamp:
                continue
            tgt = ev(ecbn.RULES[node], x)
            if tgt != s[k]:
                t = list(s)
                t[k] = tgt
                Q[i, IDX[tuple(t)]] += rates.get(node, 1.0)
        Q[i, i] = -Q[i].sum()
    return Q


def fixed_points(Q):
    return [i for i in range(len(STATES)) if Q[i, i] == 0]


def committor(rates=None):
    Q = generator(rates=rates)
    fps = fixed_points(Q)
    go = [i for i in fps if STATES[i][FREE.index("Proliferation")] == 0]
    grow = [i for i in fps if STATES[i][FREE.index("Proliferation")] == 1]
    assert len(go) == 1 and len(grow) == 1
    n = len(STATES)
    A = Q.copy()
    b = np.zeros(n)
    for i in go + grow:
        A[i] = 0
        A[i, i] = 1
        b[i] = 1.0 if i in go else 0.0
    h = np.linalg.solve(A, b)
    return h, go[0], grow[0]


def project(p, clamp):
    """Instantaneous setting of clamped nodes at pulse onset (conserves probability, keeps node correlations)."""
    q = np.zeros_like(p)
    for i, s in enumerate(STATES):
        if p[i] == 0:
            continue
        t = list(s)
        for node, v in clamp.items():
            t[FREE.index(node)] = v
        q[IDX[tuple(t)]] += p[i]
    return q


def run_schedule(phases, rates=None, start="grow"):
    """phases: list of (clamp dict, duration). Returns eventual arrest probability after the last phase is withdrawn."""
    h, go, grow = committor(rates)
    p = np.zeros(len(STATES))
    p[grow if start == "grow" else go] = 1.0
    for clamp, dur in phases:
        if clamp:
            p = project(p, clamp)
        if dur > 0:
            p = p @ expm(generator(clamp, rates) * dur)
    return float(p @ h)


if __name__ == "__main__":
    # certificate: derived reduced rules evaluated on all 128 states vs original rules (trivially identical by construction)
    Q0 = generator()
    h, go, grow = committor()
    cert = {
        "context": CTX,
        "n_constants_including_inputs": len(CONST),
        "free_variables": FREE,
        "n_states": len(STATES),
        "fixed_points": [dict(zip(FREE, STATES[i])) for i in fixed_points(Q0)],
    }
    json.dump(cert, open(f"{OUT}/D1_C0101_native_subspace_certificate.json", "w"), indent=1)
    print(cert["fixed_points"])
    # reduced truth table (for the schematic)
    tt = []
    for s in STATES:
        x = full(s)
        tt.append({**dict(zip(FREE, s)), **{f"f_{n}": ev(ecbn.RULES[n], x) for n in FREE}})
    pd.DataFrame(tt).to_csv(f"{OUT}/D2_C0101_native_truth_table.csv", index=False)
    # pulse-duration responses (exact) for free-node clamps
    clamps = {
        "CDKN1B ON": {"CDKN1B": 1},
        "MYC OFF": {"MYC": 0},
        "AKT2 OFF": {"AKT2": 0},
        "CCND2 OFF": {"CCND2": 0},
        "CCND3 OFF": {"CCND3": 0},
        "CCND2+CCND3 OFF": {"CCND2": 0, "CCND3": 0},
        "IRF1 ON": {"IRF1": 1},
        "MYC OFF + CCND2 OFF": {"MYC": 0, "CCND2": 0},
    }
    durs = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 12, 16]
    rows = []
    for lab, c in clamps.items():
        for t in durs:
            rows.append({"clamp": lab, "duration": t, "eventual_arrest": run_schedule([(c, t)])})
    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/D3_exact_pulse_duration_response.csv", index=False)
    print(D.pivot(index="duration", columns="clamp", values="eventual_arrest").round(4).to_string())
    # t95 for p27
    from scipy.optimize import brentq

    t95 = brentq(lambda t: run_schedule([({"CDKN1B": 1}, t)]) - 0.95, 0.1, 16)
    print("p27 t95", t95)
    # equal-budget order experiment (phase length L, both arms total node-time 2L, total elapsed 2L)
    order = []
    for L in [0.5, 1, 2, 4, 8, 12]:
        order.append(
            {
                "phase_length": L,
                "MYC_then_CCND2": run_schedule([({"MYC": 0}, L), ({"CCND2": 0}, L)]),
                "CCND2_then_MYC": run_schedule([({"CCND2": 0}, L), ({"MYC": 0}, L)]),
                "concurrent_then_rest": run_schedule([({"MYC": 0, "CCND2": 0}, L), ({}, L)]),
            }
        )
    O = pd.DataFrame(order)
    O.to_csv(f"{OUT}/D4_equal_budget_order_exact.csv", index=False)
    print(O.round(4).to_string())
    # rate heterogeneity: 200 log-uniform draws in [0.3, 3]
    rng = np.random.default_rng(20261027)
    rr = []
    for k in range(200):
        r = {n: float(np.exp(rng.uniform(np.log(0.3), np.log(3)))) for n in FREE}
        rr.append(
            {
                "draw": k,
                **{f"rate_{n}": v for n, v in r.items()},
                "p27_pulse_L4": run_schedule([({"CDKN1B": 1}, 4)], r),
                "p27_pulse_L8": run_schedule([({"CDKN1B": 1}, 8)], r),
                "MYC_then_CCND2_L8": run_schedule([({"MYC": 0}, 8), ({"CCND2": 0}, 8)], r),
                "CCND2_then_MYC_L8": run_schedule([({"CCND2": 0}, 8), ({"MYC": 0}, 8)], r),
                "concurrent_L8": run_schedule([({"MYC": 0, "CCND2": 0}, 8), ({}, 8)], r),
            }
        )
    R = pd.DataFrame(rr)
    R.to_csv(f"{OUT}/D5_rate_heterogeneity_200draws.csv", index=False)
    print(
        "forward>reverse",
        (R.MYC_then_CCND2_L8 > R.CCND2_then_MYC_L8).mean(),
        "concurrent>forward",
        (R.concurrent_L8 > R.MYC_then_CCND2_L8).mean(),
        "p27 L8 median",
        R.p27_pulse_L8.median(),
        "min",
        R.p27_pulse_L8.min(),
    )
