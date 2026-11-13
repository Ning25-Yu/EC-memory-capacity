"""Closed-form pulse responses of the seven-node switch of context C0101 (exact rational arithmetic).
Manuscript: Equations (6)-(8), Figure 2B, Supplementary Note S2.

With the inputs fixed to C0101, seven nodes remain free (AKT2, CCND2, CCND3, CDKN1B, IRF1, MYC, Proliferation). The
asynchronous dynamics with unit rates are a continuous-time Markov chain on 2^7 = 128 states with generator Q and
two absorbing states, the proliferative and the arrested fixed point.

  committor   h(s) = P(absorbed in the arrested fixed point | start in s), from Q h = 0 with h = 1 / 0 on the two
              fixed points; solved over the rationals.
  pulse       the clamp sets a node at onset and removes its transitions (generator Q_c); after a duration L the
              clamp is released. The arrest probability is a(L) = e_0 exp(Q_c L) h, with e_0 the state at onset.
  closed form the Laplace transform  e_0 (s I - Q_c)^(-1) h  is a rational function of s; its partial fractions
              give a(L) as a finite sum of terms L^j exp(-k L) with rational coefficients.

Outputs: results/tables/H1_committor_C0101.csv, results/tables/H2_closed_forms.json
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import pandas as pd
import sympy as sp

import ecbn
import exact_engine as E
from paths import TAB

CTX = "C0101"
CONST = E.percolate(CTX)
FREE = sorted(set(ecbn.NODES) - set(CONST))
STATES = list(itertools.product([0, 1], repeat=len(FREE)))
IDX = {s: i for i, s in enumerate(STATES)}
N = len(STATES)
s_, L = sp.Symbol("s"), sp.Symbol("L", positive=True)


def target(node, state):
    env = dict(CONST)
    env.update(zip(FREE, state))
    return int(eval(E.PYR[node], {}, env))


def generator(clamp=()):
    Q = sp.zeros(N, N)
    for st in STATES:
        i = IDX[st]
        for k, node in enumerate(FREE):
            if node in clamp:
                continue
            t = target(node, st)
            if t != st[k]:
                u = list(st)
                u[k] = t
                Q[i, IDX[tuple(u)]] += 1
        Q[i, i] = -sum(Q[i, :])
    return Q


Q0 = generator()
FIXED = [i for i in range(N) if Q0[i, i] == 0]
PROL = FREE.index("Proliferation")
ARREST = [i for i in FIXED if STATES[i][PROL] == 0][0]
GROW = [i for i in FIXED if STATES[i][PROL] == 1][0]


def committor():
    A = Q0.copy()
    b = sp.zeros(N, 1)
    for i in FIXED:
        for j in range(N):
            A[i, j] = 0
        A[i, i] = 1
    b[ARREST] = 1
    return A.LUsolve(b)


H = committor()


def onset(state, clamp):
    u = list(state)
    for node, value in clamp.items():
        u[FREE.index(node)] = value
    return tuple(u)


def reachable(Q, start):
    order = [start]
    k = 0
    while k < len(order):
        i = order[k]
        for j in range(N):
            if j != i and Q[i, j] != 0 and j not in order:
                order.append(j)
        k += 1
    return order


def closed_form(clamp):
    """a(L) for a single pulse from the proliferative fixed point, as a sympy expression in L."""
    Qc = generator(tuple(clamp))
    reach = reachable(Qc, IDX[onset(STATES[GROW], clamp)])
    Qr = Qc.extract(reach, reach)
    hr = sp.Matrix([H[i] for i in reach])
    y = (s_ * sp.eye(len(reach)) - Qr).LUsolve(hr)
    F = sp.apart(sp.cancel(y[0]), s_)
    a = sp.inverse_laplace_transform(F, s_, L)
    a = sp.expand(a.subs(sp.Heaviside(L), 1))
    return a, len(reach)


def limit_distribution(p, clamp):
    """Distribution reached when the clamp is held for ever (exact). p: dict state index -> probability."""
    Qc = generator(tuple(clamp))
    out = {}
    for i0, w in p.items():
        start = IDX[onset(STATES[i0], clamp)]
        reach = reachable(Qc, start)
        Qr = Qc.extract(reach, reach)
        absorbing = [k for k in range(len(reach)) if Qr[k, k] == 0]
        transient = [k for k in range(len(reach)) if Qr[k, k] != 0]
        assert absorbing, "clamped chain has no absorbing state"
        if 0 in absorbing:
            out[reach[0]] = out.get(reach[0], 0) + w
            continue
        T = Qr.extract(transient, transient)
        R = Qr.extract(transient, absorbing)
        B = (-T).LUsolve(R)  # absorption probabilities
        row = transient.index(0)
        for c, k in enumerate(absorbing):
            if B[row, c] != 0:
                out[reach[k]] = out.get(reach[k], 0) + w * B[row, c]
    return out


def schedule_limit(phases):
    p = {GROW: sp.Integer(1)}
    for clamp in phases:
        p = limit_distribution(p, clamp)
    return sp.nsimplify(sum(w * H[i] for i, w in p.items()))


if __name__ == "__main__":
    rows = [dict(zip(FREE, st), committor=str(H[IDX[st]]), committor_float=float(H[IDX[st]])) for st in STATES]
    pd.DataFrame(rows).to_csv(TAB + "/H1_committor_C0101.csv", index=False)
    values = sorted(set(H))
    print("free nodes:", FREE)
    print("proliferative fixed point:", dict(zip(FREE, STATES[GROW])))
    print("arrested fixed point:     ", dict(zip(FREE, STATES[ARREST])))
    print(len(values), "distinct committor values; common denominator", sp.ilcm(*[sp.fraction(v)[1] for v in values]))

    out = {"free_nodes": FREE, "n_distinct_committor_values": len(values)}
    pulses = {
        "CDKN1B ON": {"CDKN1B": 1},
        "MYC OFF + CCND2 OFF": {"MYC": 0, "CCND2": 0},
        "MYC OFF": {"MYC": 0},
        "CCND2 OFF": {"CCND2": 0},
        "CCND3 OFF": {"CCND3": 0},
        "CCND2 OFF + CCND3 OFF": {"CCND2": 0, "CCND3": 0},
        "AKT2 OFF": {"AKT2": 0},
        "IRF1 ON": {"IRF1": 1},
    }
    for label, clamp in pulses.items():
        a, n = closed_form(clamp)
        a0 = sp.nsimplify(sp.limit(a, L, 0))
        out[label] = {
            "a_of_L": str(a),
            "latex": sp.latex(a),
            "reachable_states_under_clamp": n,
            "a_at_onset": str(a0),
            "a_at_L8": float(a.subs(L, 8)),
        }
        print(f"{label:24s} states {n:3d}  a(0+) = {a0}  a(8) = {float(a.subs(L, 8)):.10f}\n    a(L) = {a}")

    limits = {
        "MYC OFF then CCND2 OFF": schedule_limit([{"MYC": 0}, {"CCND2": 0}]),
        "CCND2 OFF then MYC OFF": schedule_limit([{"CCND2": 0}, {"MYC": 0}]),
        "MYC OFF + CCND2 OFF (concurrent)": schedule_limit([{"MYC": 0, "CCND2": 0}]),
        "CDKN1B ON": schedule_limit([{"CDKN1B": 1}]),
    }
    out["long_pulse_limits"] = {k: str(v) for k, v in limits.items()}
    print("long-pulse limits:", out["long_pulse_limits"])
    json.dump(out, open(TAB + "/H2_closed_forms.json", "w"), indent=1)
