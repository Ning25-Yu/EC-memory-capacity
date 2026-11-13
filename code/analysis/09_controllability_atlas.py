"""E1. Controllability atlas: durable arrest after EVERY single-node finite pulse (27 non-input molecular nodes x
Manuscript: Figure 4, Figure S10, Table 1 and Table S5 (controllability atlas).

{OFF, ON}, L = 8) from EVERY proliferative attractor in the 16 input contexts (6 proliferative fixed points + 2
proliferative complex recurrent classes). Exact (exact_engine) whenever the reachable set is <= CAP states;
otherwise replicated simulation of the same Markov chain (3 x 20,000 trajectories). For every pulse that writes
arrest >= 0.5 from some attractor, the exact duration response (L = 0.5 ... 16) and the minimal duration giving 95%
of the L = 16 plateau (t95) are computed.
Simulated cells keep every replicate (E1r) with absorption at t = 60 and t = 120 after release and the unresolved mass.
Duration responses are computed only where every L is exactly solvable (the exactly solvable subset of the writer cells).
Outputs: results/tables/E1_controllability_atlas.csv, results/tables/E1r_simulation_replicates.csv, results/tables/E1_duration_response.csv
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import time

import numpy as np
import pandas as pd

import ecbn
import exact_engine as E
from paths import IN_TAB, TAB

CAP = 60_000
L = 8.0
A = pd.read_csv(IN_TAB + "/A2_attractor_repertoire_16_contexts.csv")
PRO = A[A.Proliferation.astype(str).str.upper().isin(["1", "X"])]
PULSES = [(f"{n}={v}", n, v) for n in ecbn.INTERNAL_MOL for v in (0, 1)]
REPS = []  # per-replicate simulation records (seed, arrest, absorption at t = 60 / 120, unresolved mass)


def classes(sp, ctx):
    sets, offs = [], []
    for _, r in A[(A.context == ctx) & (A.type.astype(str).str.startswith("complex"))].iterrows():
        mem, pi, off = E.complex_class(sp, r)
        sets.append(set(mem))
        offs.append(off)
    return sets, offs


def cell(ctx, row, node, val, L=L, sim_ok=True):
    sp = E.Space(ctx, (node,))
    start, kind, cs, off = E.attractor_start(sp, row)
    if node not in sp.ix:  # not upstream of Proliferation in this context: cannot change the outcome
        return dict(P_arrest_FP=0.0, method="no path to Proliferation", SE=0.0)
    o = E.exact_pulse(sp, start, {node: val}, L, cap=CAP)
    if o is not None:
        return dict(P_arrest_FP=o["P_arrest_FP"], method="exact", SE=0.0)
    if not sim_ok:
        return None
    cls, offs = classes(sp, ctx)
    R = [E.simulate_pulse(sp, start, {node: val}, cls, offs, L=L, n=20000, seed=s) for s in (1, 2, 3)]
    for r in R:
        REPS.append({"context": ctx, "attractor": row.attractor, "pulse": f"{node}={val}", "L": L, **r})
    reps = [r["P_arrest_FP"] for r in R]
    return dict(
        P_arrest_FP=float(np.mean(reps)),
        method="simulation (3 x 20,000)",
        SE=float(np.std(reps, ddof=1) / np.sqrt(3)),
        max_unresolved_t120=float(max(r["unresolved_at_t120"] for r in R)),
        max_abs_change_t60_t120=float(max(abs(r["arrest_FP_by_t120"] - r["arrest_FP_by_t60"]) for r in R)),
    )


if __name__ == "__main__":
    rows = []
    t0 = time.time()
    for _, r in PRO.iterrows():
        for lab, node, val in PULSES:
            c = cell(r.context, r, node, val)
            rows.append(
                {
                    "context": r.context,
                    "attractor": r.attractor,
                    "attractor_type": (
                        "fixed point" if str(r.type).startswith("fixed") else "complex recurrent class"
                    ),
                    "pulse": lab,
                    "node": node,
                    "value": val,
                    **c,
                }
            )
        print(r.attractor, f"{time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(TAB + "/E1_controllability_atlas.csv", index=False)
    pd.DataFrame(REPS).to_csv(TAB + "/E1r_simulation_replicates.csv", index=False)
    writers = D[D.P_arrest_FP >= 0.5][["pulse", "node", "value"]].drop_duplicates()
    print("writers:", writers.pulse.tolist())
    dur = []
    for _, w in writers.iterrows():
        for _, r in PRO.iterrows():
            for Lx in [0.5, 1, 2, 3, 4, 6, 8, 12, 16]:
                c = cell(r.context, r, w.node, w.value, L=float(Lx), sim_ok=False)
                if c is None:
                    break
                dur.append(
                    {
                        "pulse": w.pulse,
                        "context": r.context,
                        "attractor": r.attractor,
                        "L": Lx,
                        "P_arrest_FP": c["P_arrest_FP"],
                    }
                )
    R = pd.DataFrame(dur)
    R.to_csv(TAB + "/E1_duration_response.csv", index=False)
    print("done", f"{time.time() - t0:.0f}s")
