"""A3/QC. Long-run MaBoSS phenotype probabilities of the 16 input contexts.
Manuscript: Figure 2C and Figure S5A (attractor and phenotype probabilities of each context).

Inputs are clamped by their initial state (self-loop rules); every other node starts at P(ON) = 0.5; all rates 1;
10,000 trajectories, seed 12345. Read-outs are the six phenotype nodes.
  t = 50   : results/tables/QC_reproduced_16_context_probabilities.csv (re-run of the published 96 context-output values)
  t = 1000 : results/tables/QC_16_context_probabilities_t1000.csv (long-run values used in the figures; at t = 50 the contexts
             C0000 and C1000 still carry a little transient mass, all other contexts are identical at t = 50 and 1000)
Also writes results/tables/A3_context_attractor_probabilities.csv: probability of each attractor of A2 at t = 1000
(joint states over all nodes mapped to the attractors)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import pandas as pd

import ecbn
from paths import TAB

CTX = [f"C{i:04b}" for i in range(16)]


def run(ctx, t, outputs=ecbn.PHENO, n=10000):
    init = {k: float(v) for k, v in ecbn.ctx_bits(ctx).items()}
    return ecbn.simulate(init=init, max_time=t, time_tick=min(1.0, t / 50), sample_count=n, outputs=outputs)


def main():
    A = pd.read_csv(os.path.join(TAB, "A2_attractor_repertoire_16_contexts.csv"), dtype=str)
    for t, name in [
        (50, "QC_reproduced_16_context_probabilities.csv"),
        (1000, "QC_16_context_probabilities_t1000.csv"),
    ]:
        rows = []
        for c in CTX:
            _, m = run(c, t)
            rows.append({"context": c, **{p: float(m[p]) for p in ecbn.PHENO}})
        pd.DataFrame(rows).to_csv(os.path.join(TAB, name), index=False)
        print(name)
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    rows = []
    for c in CTX:
        dist, _ = run(c, 1000, outputs=ecbn.NODES)
        sub = A[A.context == c]
        acc = {a: 0.0 for a in sub.attractor}
        for st, p in dist.items():
            on = set(st.split(" -- ")) if st != "<nil>" else set()
            for _, r in sub.iterrows():
                if all(r[n] == "X" or int(r[n]) == int(n in on) for n in ecbn.NODES):
                    acc[r.attractor] += p
                    break
        for a, p in acc.items():
            rows.append({"context": c, "attractor": a, "MaBoSS_probability_t1000": p})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "A3_context_attractor_probabilities.csv"), index=False)


if __name__ == "__main__":
    main()
