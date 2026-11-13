"""E2b. Which repressive return edges into p27 (CDKN1B) are needed for proliferative memory?
Manuscript: Figure 5B (repressive return edges into p27).

p27 is repressed by four module nodes (MYC, CCND2, CCND3, AKT2). Every non-empty subset of these four return edges
(15 subsets) is ablated simultaneously (source replaced by constant OFF in the CDKN1B rule only), and the C0101 and
C1001 read-outs of E2 are recomputed exactly. The two output edges of p27 that E2 identified (p27 -| MYC,
p27 -| CCND2) are included as reference rows.
Outputs: results/tables/E2b_return_edge_combinations.csv"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import importlib.util
import itertools
import re

import pandas as pd
import sympy

import ecbn
import exact_engine as E
from paths import TAB

spec = importlib.util.spec_from_file_location(
    "e2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_edge_ablation.py")
)
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)


def ablate_many(edges):
    by_t = {}
    for s, t in edges:
        by_t.setdefault(t, []).append(s)
    for t, srcs in by_t.items():
        rule = ecbn.RULES[t]
        for s in srcs:
            rule = re.sub(r"\b%s\b" % s, "ZZZ", rule)
        E.PYR[t] = compile(
            rule.replace("!", " not ")
            .replace("&", " and ")
            .replace("|", " or ")
            .replace("ZZZ", "False")
            .strip(),
            t,
            "eval",
        )
        E.REG[t] = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rule)) & set(ecbn.NODES)
        E._EXPR[t] = e2.ORIG["_EXPR"][t].subs({E._SYM[s]: sympy.false for s in srcs})


if __name__ == "__main__":
    A = e2.A
    fp = A[(A.context == "C0101") & (A.Proliferation.astype(str) == "1")].iloc[0]
    cx = A[(A.context == "C1001") & (A.type.astype(str).str.startswith("complex"))].iloc[0]
    S_fp = e2.full_states_fixed(fp)
    S_cx = e2.full_states_class("C1001", cx)
    RET = ["MYC", "CCND2", "CCND3", "AKT2"]
    sets = [("none (intact)", [])] + [
        (" + ".join(f"{s} -| p27" for s in c), [(s, "CDKN1B") for s in c])
        for k in range(1, 5)
        for c in itertools.combinations(RET, k)
    ]
    sets += [("p27 -| MYC (output)", [("CDKN1B", "MYC")]), ("p27 -| CCND2 (output)", [("CDKN1B", "CCND2")])]
    rows = []
    for lab, edges in sets:
        ablate_many(edges)
        try:
            a = e2.relax_and_pulse("C0101", S_fp, ["CDKN1B", "CDKN2A"])
            b = e2.relax_and_pulse("C1001", S_cx, ["CDKN1A", "CDKN1B"])
        finally:
            e2.restore()
        rows.append(
            {
                "ablated_edges": lab,
                "n_edges": len(edges),
                "C0101_proliferative_mass": a["proliferative_mass"],
                "C0101_p27_arrest": a["CDKN1B_arrest"],
                "C0101_p16_arrest": a["CDKN2A_arrest"],
                "C1001_proliferative_mass": b["proliferative_mass"],
                "C1001_p21_arrest": b["CDKN1A_arrest"],
                "C1001_p27_arrest": b["CDKN1B_arrest"],
            }
        )
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(TAB + "/E2b_return_edge_combinations.csv", index=False)
