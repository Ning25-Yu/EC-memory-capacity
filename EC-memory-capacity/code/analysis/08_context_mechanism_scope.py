"""Scope of single-node finite pulses (p27/CDKN1B, p16/CDKN2A, p21/CDKN1A; L = 8) across all 16 input contexts,
Manuscript: Figure S9 (single-node pulse control across contexts and starting attractors).

from EVERY proliferative attractor (fixed point or complex recurrent class). No value is hard-coded.

Method per (attractor, pulse) cell:
  * exact (exact_engine.exact_pulse) whenever the reachable state set after release is <= exact_engine.EXACT_CAP;
  * otherwise (weakly canalised contexts, ~10^7 reachable states) the SAME CTMC is sampled with the seeded
    vectorised simulator: 10 independent two-phase replicates of 50,000 trajectories; mean, empirical SE across
    replicates, and absorption status at t = 60 and t = 120 after release are recorded.
The simulator is validated against every exact cell (3 replicates each; results/tables/D7v_simulator_validation.csv).

Outputs
  results/tables/D7_context_mechanism_scope.csv      one row per (context, proliferative attractor)
  results/tables/D7r_replicate_records.csv           every simulation replicate (incl. t = 60 / t = 120 records)
  results/tables/D7v_simulator_validation.csv        simulator vs exact on all exact cells
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd

import exact_engine as E
from paths import TAB

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = TAB
PULSES = {"p27": "CDKN1B", "p16": "CDKN2A", "p21": "CDKN1A"}
L = 8.0
N_REP = 10
N_TRAJ = 50_000
N_VAL = 3


def known_classes(sp, sub):
    """All complex recurrent classes of this context on the tracked space (from the A2 repertoire)."""
    sets, offs = [], []
    for _, r in sub.iterrows():
        if str(r["type"]).startswith("complex"):
            mem, pi, off = E.complex_class(sp, r)
            sets.append(set(mem))
            offs.append(off)
    return sets, offs


def replicates(sp, start, clamp, cls, offs, seeds):
    return [E.simulate_pulse(sp, start, clamp, cls, offs, L=L, n=N_TRAJ, seed=s) for s in seeds]


if __name__ == "__main__":
    A = pd.read_csv(os.path.join(TAB, "A2_attractor_repertoire_16_contexts.csv"))
    rows, recs, val = [], [], []
    for ctx in sorted(A.context.unique()):
        sub = A[A.context == ctx]
        pro = sub[sub.Proliferation.astype(str).str.upper().isin(["1", "X"])]
        if pro.empty:
            rows.append({"context": ctx, "attractor": "", "attractor_type": "non-proliferative"})
            continue
        for _, r in pro.iterrows():
            rec = {"context": ctx, "attractor": r["attractor"]}
            for lab, node in PULSES.items():
                sp = E.Space(ctx, (node,))
                start, kind, cs, off = E.attractor_start(sp, r)
                cls, offs = known_classes(sp, sub)
                rec["attractor_type"] = kind
                if off is not None:
                    rec["class_states"] = len(cs)
                    rec["class_prolif_off_occupancy"] = round(off, 6)
                    rec["untreated_P_arrest_FP"] = 0.0
                    rec["untreated_P_prolif0_late"] = round(off, 6)
                else:
                    rec["untreated_P_arrest_FP"] = 0.0
                    rec["untreated_P_prolif0_late"] = 0.0
                ex = E.exact_pulse(sp, start, {node: 1}, L)
                if ex is not None:
                    rec[f"{lab}_P_arrest_FP"] = round(ex["P_arrest_FP"], 10)
                    rec[f"{lab}_P_prolif0_late"] = round(ex["P_prolif0_late"], 10)
                    rec[f"{lab}_SE"] = 0.0
                    rec[f"{lab}_method"] = "exact"
                    rec[f"{lab}_states"] = f"{ex['n_phase1']}/{ex['n_phase2']}"
                    reps = replicates(sp, start, {node: 1}, cls, offs, range(101, 101 + N_VAL))
                    m = np.array([x["P_arrest_FP"] for x in reps])
                    se = (
                        m.std(ddof=1) / np.sqrt(len(m))
                        if m.std() > 0
                        else np.sqrt(max(m.mean() * (1 - m.mean()), 1e-12) / (N_TRAJ * len(m)))
                    )
                    val.append(
                        {
                            "context": ctx,
                            "attractor": r["attractor"],
                            "pulse": lab,
                            "exact": ex["P_arrest_FP"],
                            "sim_mean": m.mean(),
                            "sim_SE": se,
                            "z": (m.mean() - ex["P_arrest_FP"]) / se if se > 0 else 0.0,
                            "replicates": len(m),
                            "trajectories_per_replicate": N_TRAJ,
                        }
                    )
                else:
                    reps = replicates(sp, start, {node: 1}, cls, offs, range(1, 1 + N_REP))
                    m = np.array([x["P_arrest_FP"] for x in reps])
                    l0 = np.array([x["P_prolif0_late"] for x in reps])
                    rec[f"{lab}_P_arrest_FP"] = round(m.mean(), 6)
                    rec[f"{lab}_P_prolif0_late"] = round(l0.mean(), 6)
                    rec[f"{lab}_SE"] = round(m.std(ddof=1) / np.sqrt(len(m)), 6)
                    rec[f"{lab}_method"] = f"simulation ({N_REP} x {N_TRAJ:,})"
                    rec[f"{lab}_states"] = f"> {E.EXACT_CAP:,}"
                    for x in reps:
                        recs.append({"context": ctx, "attractor": r["attractor"], "pulse": lab, **x})
                print(
                    ctx,
                    r["attractor"],
                    kind[:7],
                    lab,
                    rec[f"{lab}_P_arrest_FP"],
                    rec[f"{lab}_method"],
                    flush=True,
                )
            rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(TAB, "D7_context_mechanism_scope.csv"), index=False)
    pd.DataFrame(recs).to_csv(os.path.join(TAB, "D7r_replicate_records.csv"), index=False)
    V = pd.DataFrame(val)
    V.to_csv(os.path.join(TAB, "D7v_simulator_validation.csv"), index=False)
    print(
        "wrote D7 (%d rows), D7r (%d replicates), D7v (%d cells; max |z| = %.2f)"
        % (len(D), len(recs), len(V), V.z.abs().max())
    )
