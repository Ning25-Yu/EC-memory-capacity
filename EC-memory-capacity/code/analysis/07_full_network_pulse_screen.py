"""Finite-pulse screen from the C0101 proliferative-invasive fixed point, computed EXACTLY on the full network.
Manuscript: Figure 3C (arrest at the end of the pulse and after withdrawal, full network).

All 27 non-input molecular nodes x {0, 1} (54 single pulses) plus pre-specified multi-node pulses, pulse length L.
Uses exact_engine with scope='all' (every non-constant node tracked; constants from percolation with the pulse
nodes left symbolic), so Proliferation, Invasion and Apoptosis read-outs are exact: no Monte Carlo, no truncation.
on_*  : expected value at the end of the clamp (t = L);
off_* : expected value at the absorbing attractor after release;
eventual_arrest = P_arrest_FP (probability of absorption into a Proliferation-OFF fixed point).
Output: results/tables/D6_full_network_pulse_screen_L8.csv
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import pandas as pd

import ecbn
import exact_engine as E
from paths import TAB

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = TAB
L = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
A = pd.read_csv(os.path.join(TAB, "A2_attractor_repertoire_16_contexts.csv"))
FP = A[(A.context == "C0101") & (A.Proliferation.astype(str) == "1")].iloc[0]
PULSES = (
    [("Untreated", {})]
    + [(f"{n}={v}", {n: v}) for n in ecbn.INTERNAL_MOL for v in (0, 1)]
    + [
        ("IL6=0 + CDKN1A=1", {"IL6": 0, "CDKN1A": 1}),
        ("IL6=0 + CCND2=0 + CCND3=0", {"IL6": 0, "CCND2": 0, "CCND3": 0}),
        ("CCND2=0 + CCND3=0", {"CCND2": 0, "CCND3": 0}),
        ("MYC=0 + IL6=0", {"MYC": 0, "IL6": 0}),
        ("AKT2=0 + CDKN1A=1", {"AKT2": 0, "CDKN1A": 1}),
        ("STAT3=0 + CCND2=0 + CCND3=0", {"STAT3": 0, "CCND2": 0, "CCND3": 0}),
    ]
)

if __name__ == "__main__":
    rows = []
    for lab, clamp in PULSES:
        sp = E.Space("C0101", tuple(clamp), scope="all")
        start, _, _, _ = E.attractor_start(sp, FP)
        o = E.exact_pulse(sp, start, clamp, L)
        assert o is not None, f"{lab}: reachable set exceeds EXACT_CAP"
        rows.append(
            {
                "pulse": lab,
                "length": L,
                **{f"on_{k}": round(o["on"].get(k, float("nan")), 10) for k in E.OUTPUTS},
                **{f"off_{k}": round(o["off"][k], 10) for k in E.OUTPUTS},
                "eventual_arrest": round(o["P_arrest_FP"], 10),
                "P_prolif0_late": round(o["P_prolif0_late"], 10),
                "absorbed_mass": round(o["absorbed_mass"], 12),
                "tracked_nodes": len(sp.nodes),
                "states_phase1/phase2": f"{o['n_phase1']}/{o['n_phase2']}",
                "method": "exact",
            }
        )
        print(
            f"{lab:32s} arrest={o['P_arrest_FP']:.6f} offInv={o['off']['Invasion']:.3f} states={o['n_phase1']}/{o['n_phase2']}",
            flush=True,
        )
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(TAB, f"D6_full_network_pulse_screen_L{int(L)}.csv"), index=False)
    print("wrote D6:", len(D), "pulses; max |1 - absorbed_mass| =", float((1 - D.absorbed_mass).abs().max()))
