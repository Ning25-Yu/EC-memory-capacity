"""A6, A8, A9. Input canalisation, the C0101 commitment time course and a trajectory of the C1001 recurrent class.
Manuscript: Figure S5B (trajectory of the C1001 complex recurrent class), Figure S6A (A6) and Figure S6B (A8).

A6  logical propagation of the four inputs (exact_engine.percolate): number of non-input nodes fixed in each context
    and the free nodes (results/tables/A6_input_canalisation_16_contexts.csv).
A8  MaBoSS time course of C0101 (10,000 trajectories, inputs clamped, other nodes P(ON) = 0.5, t = 0-50, step 0.5);
    joint states of Proliferation and Invasion, so that grow-and-go = Proliferation & Invasion and go = Invasion
    without Proliferation (results/tables/A8_C0101_state_probability_timecourse.csv).
A9  one asynchronous (Gillespie, unit rates) trajectory inside the C1001 complex recurrent class, started from a state of
    the class drawn from its stationary distribution (seed 7); 200 jumps (results/tables/A9_C1001_cyclic_attractor_trajectory.csv).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import tempfile

import cmaboss
import numpy as np
import pandas as pd

import ecbn
import exact_engine as E
from paths import TAB

CTX = [f"C{i:04b}" for i in range(16)]


def a6():
    rows = []
    for c in CTX:
        k = E.percolate(c)
        non = [n for n in ecbn.NODES if n not in ecbn.INPUTS]
        rows.append(
            {
                "context": c,
                "n_fixed_by_inputs": sum(n in k for n in non),
                "n_free": sum(n not in k for n in non),
                "free_nodes": "; ".join(n for n in non if n not in k),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "A6_input_canalisation_16_contexts.csv"), index=False)


def a8():
    init = {k: float(v) for k, v in ecbn.ctx_bits("C0101").items()}
    cfg = ecbn.cfg_string(
        init=init, outputs=("Proliferation", "Invasion"), max_time=50, time_tick=0.5, sample_count=10000
    )
    with tempfile.TemporaryDirectory() as d:
        open(d + "/m.bnd", "w").write(ecbn.bnd_string())
        open(d + "/m.cfg", "w").write(cfg)
        P, t, states, _ = cmaboss.MaBoSSSim(network=d + "/m.bnd", config=d + "/m.cfg").run().get_probtraj()
    df = pd.DataFrame(np.asarray(P), index=pd.Index(t, name="time"), columns=states)
    df.to_csv(os.path.join(TAB, "A8_C0101_state_probability_timecourse.csv"))


def a9(seed=7, jumps=200):
    A = pd.read_csv(os.path.join(TAB, "A2_attractor_repertoire_16_contexts.csv"), dtype=str)
    row = A[(A.context == "C1001") & (A.type == "complex recurrent class")].iloc[0]
    sp = E.Space("C1001", scope="all")
    mem, pi, _ = E.complex_class(sp, row)
    rng = np.random.default_rng(seed)
    s = mem[rng.choice(len(mem), p=pi / pi.sum())]
    t = 0.0
    out = []
    for _ in range(jumps):
        full = {n: sp.value(s, n) for n in ecbn.NODES}
        out.append({"time": t, **full})
        succ = sp.successors(s, set())
        t += rng.exponential(1.0 / len(succ))
        s = succ[rng.integers(len(succ))]
    pd.DataFrame(out).to_csv(os.path.join(TAB, "A9_C1001_cyclic_attractor_trajectory.csv"), index=False)


if __name__ == "__main__":
    a6()
    a8()
    a9()
    print(pd.read_csv(os.path.join(TAB, "A6_input_canalisation_16_contexts.csv")).to_string(index=False))
