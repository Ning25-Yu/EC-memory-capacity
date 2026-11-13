"""Robustness of the controllability results to modelling choices.
Manuscript: Results 2.5 and 2.6, Table S5D (robustness of writer calls and of the p27 memory module).

Four checks, each written as one block of results/tables/R1_robustness_checks.csv:
  A  writer threshold and pulse length: the atlas (script 09) is recomputed at L = 4 and L = 16, and guaranteed
     writers of every proliferative context are called at thresholds 0.4, 0.5 and 0.6;
  B  transition rates in the full network: every exactly solved atlas cell with an arrest probability strictly between
     0 and 1 (cells at exactly 0 or 1 do not depend on positive rates) is recomputed for 200 sets of node rates drawn
     log-uniformly between 0.3 and 3;
  C  network structure: k = 2, 4 or 8 randomly chosen edges (about 1%, 2.5% and 5% of the 168 edges) are deleted
     together, 200 draws each, and the p27 and p16 pulses are applied from the proliferative state of C0101;
  D  update scheme: the 54 single-node pulses are repeated from the six proliferative fixed points under synchronous
     (deterministic) updating and compared with the asynchronous writer calls.
The full per-draw records are written to results/tables/R1b_robustness_draws.csv."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import importlib.util
import re

import numpy as np
import pandas as pd
import sympy
from scipy.sparse import coo_matrix, csr_matrix

import ecbn
import exact_engine as E
from paths import TAB

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


atlas = load("atlas", "09_controllability_atlas.py")
abl = load("abl", "10_edge_ablation.py")
A2 = atlas.A
PRO = atlas.PRO
SUMMARY = []
DRAWS = []


def add(block, setting, quantity, value):
    SUMMARY.append({"check": block, "setting": setting, "quantity": quantity, "value": value})


# ---------------------------------------------------------------- A: threshold and pulse length
def guaranteed(D, threshold):
    """Guaranteed writers per context: arrest >= threshold from every proliferative attractor of the context."""
    out = {}
    for ctx, d in D.groupby("context"):
        ok = d.groupby("pulse").P_arrest_FP.min()
        out[ctx] = tuple(sorted(ok[ok >= threshold].index))
    return out


def check_threshold_and_length():
    tables = {8.0: pd.read_csv(TAB + "/E1_controllability_atlas.csv")}
    for L in (4.0, 16.0):
        rows = []
        for _, r in PRO.iterrows():
            for lab, node, val in atlas.PULSES:
                c = atlas.cell(r.context, r, node, val, L=L)
                rows.append({"context": r.context, "attractor": r.attractor, "pulse": lab, **c})
        tables[L] = pd.DataFrame(rows)
        print(f"A: atlas recomputed at L = {L:g}", flush=True)
    reference = guaranteed(tables[8.0], 0.5)
    for L, D in tables.items():
        for threshold in (0.4, 0.5, 0.6):
            g = guaranteed(D, threshold)
            setting = f"L = {L:g}, threshold = {threshold}"
            for ctx in sorted(g):
                add("A threshold and pulse length", setting, f"guaranteed writers of {ctx}", "; ".join(g[ctx]))
            changed = [ctx for ctx in g if set(g[ctx]) != set(reference[ctx])]
            add("A threshold and pulse length", setting, "contexts whose guaranteed writers differ from L = 8, 0.5",
                "; ".join(sorted(changed)) or "none")
            both = all({"CDKN1B=1", "CDKN2A=1"} <= set(v) for v in g.values())
            add("A threshold and pulse length", setting, "p27 ON and p16 ON guaranteed in every context", both)


# ---------------------------------------------------------------- B: transition rates in the full network
class RateSpace(E.Space):
    """Tracked state space in which each node flips at its own rate (the same rate for switching ON and OFF)."""

    rates = {}

    def explore(self, seeds, clamp, cap=None):
        order = list(dict.fromkeys(seeds))
        pos = {s: i for i, s in enumerate(order)}
        I, J, W = [], [], []
        k = 0
        while k < len(order):
            if cap and len(order) > cap:
                return None
            s = order[k]
            env = dict(self.const)
            env.update(zip(self.nodes, s))
            for n in self.nodes:
                if n in clamp:
                    continue
                target = int(eval(E.PYR[n], {}, env))
                if target != s[self.ix[n]]:
                    u = list(s)
                    u[self.ix[n]] = target
                    u = tuple(u)
                    if u not in pos:
                        pos[u] = len(order)
                        order.append(u)
                    I.append(k)
                    J.append(pos[u])
                    W.append(self.rates.get(n, 1.0))
            k += 1
        n = len(order)
        R = coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
        Q = R - csr_matrix((np.asarray(R.sum(1)).ravel(), (range(n), range(n))), shape=(n, n))
        adjacency = coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr()
        return order, pos, Q.tocsr(), adjacency


def check_rates(n_draws=200, seed=20261111):
    D = pd.read_csv(TAB + "/E1_controllability_atlas.csv")
    cells = D[(D.method == "exact") & (D.P_arrest_FP > 1e-9) & (D.P_arrest_FP < 1 - 1e-9)]
    rng = np.random.default_rng(seed)
    nodes = [n for n in ecbn.NODES if n not in ecbn.INPUTS]
    draws = [dict(zip(nodes, 10 ** rng.uniform(np.log10(0.3), np.log10(3.0), len(nodes)))) for _ in range(n_draws)]
    flips = 0
    for _, c in cells.iterrows():
        row = A2[A2.attractor == c.attractor].iloc[0]
        values = []
        for k, rates in enumerate(draws):
            RateSpace.rates = rates
            sp = RateSpace(c.context, (c.node,))
            start = E.attractor_start(sp, row)[0]
            o = E.exact_pulse(sp, start, {c.node: int(c.value)}, 8.0, cap=atlas.CAP)
            values.append(o["P_arrest_FP"])
            DRAWS.append({"check": "B", "attractor": c.attractor, "pulse": c.pulse, "draw": k, "P_arrest_FP": values[-1]})
        values = np.array(values)
        changed = int(((values >= 0.5) != (c.P_arrest_FP >= 0.5)).sum())
        flips += changed
        label = f"{c.attractor}, {c.pulse} (unit rates {c.P_arrest_FP:.3f})"
        add("B transition rates", label, "median (minimum to maximum) over 200 rate sets",
            f"{np.median(values):.3f} ({values.min():.3f} to {values.max():.3f})")
        add("B transition rates", label, "rate sets in which the writer call changes", changed)
        print("B:", label, changed, flush=True)
    add("B transition rates", f"{len(cells)} exact cells with 0 < P < 1", "writer calls changed (cell x rate set)",
        f"{flips} of {len(cells) * n_draws}")


# ---------------------------------------------------------------- C: network structure
def ablate_many(edges):
    by_target = {}
    for s, t in edges:
        by_target.setdefault(t, []).append(s)
    for t, sources in by_target.items():
        rule = ecbn.RULES[t]
        expr = abl.ORIG["_EXPR"][t]
        for k, s in enumerate(sources):
            rule = re.sub(r"\b%s\b" % s, f"ZZZ{k}", rule)
            expr = expr.subs(E._SYM[s], sympy.false)
        py = rule.replace("!", " not ").replace("&", " and ").replace("|", " or ")
        py = re.sub(r"ZZZ\d+", "False", py)
        E.PYR[t] = compile(py.strip(), t, "eval")
        E.REG[t] = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", re.sub(r"ZZZ\d+", "", rule))) & set(ecbn.NODES)
        E._EXPR[t] = expr


def check_structure(n_draws=200, seed=20261112):
    single = pd.read_csv(TAB + "/E2_edge_ablation.csv")
    critical = {(r.source, r.target) for r in single.itertuples()
                if r.C0101_p27_status != "not essential" or r.C0101_p16_status != "not essential"}
    kept = int(((single.C0101_p27_status == "not essential") & (single.C0101_p16_status == "not essential")).sum())
    add("C network structure", "single-edge deletion (all 168 edges)", "p27 and p16 arrest both retained",
        f"{kept} of {len(single)}")
    fp = A2[(A2.context == "C0101") & (A2.Proliferation.astype(str) == "1")].iloc[0]
    start = abl.full_states_fixed(fp)
    rng = np.random.default_rng(seed)
    for k in (2, 4, 8):
        retained = with_critical = retained_without = 0
        for d in range(n_draws):
            edges = [abl.PAIRS[i] for i in rng.choice(len(abl.PAIRS), size=k, replace=False)]
            ablate_many(edges)
            try:
                r = abl.relax_and_pulse("C0101", start, ["CDKN1B", "CDKN2A"])
            finally:
                abl.restore()
            p27 = r.get("CDKN1B_arrest", np.nan)
            p16 = r.get("CDKN2A_arrest", np.nan)
            ok = bool(r["proliferative_mass"] > 0.5 and p27 >= 0.5 and p16 >= 0.5)
            hit = bool(critical & set(edges))
            retained += ok
            with_critical += hit
            retained_without += ok and not hit
            DRAWS.append({"check": "C", "k_edges": k, "draw": d, "edges": "; ".join(f"{s}>{t}" for s, t in edges),
                          "proliferative_mass": r["proliferative_mass"], "p27_arrest": p27, "p16_arrest": p16,
                          "contains_critical_edge": hit, "retained": ok})
        setting = f"{k} random edges deleted, {n_draws} draws"
        add("C network structure", setting, "proliferative state, p27 and p16 arrest all retained",
            f"{retained} of {n_draws}")
        add("C network structure", setting, "retained among draws without a single-edge critical edge",
            f"{retained_without} of {n_draws - with_critical}")
        print("C:", setting, retained, retained_without, n_draws - with_critical, flush=True)
    add("C network structure", "single-edge critical edges (C0101)", "edges",
        "; ".join(f"{s}>{t}" for s, t in sorted(critical)))


# ---------------------------------------------------------------- D: synchronous updating
def synchronous_step(state, clamp):
    new = {n: (state[n] if n in clamp or n in ecbn.INPUTS else int(eval(E.PYR[n], {}, state))) for n in ecbn.NODES}
    new.update(clamp)
    return new


def check_synchronous(L=8):
    D = pd.read_csv(TAB + "/E1_controllability_atlas.csv")
    agree = total = 0
    disagreements = []
    for _, row in PRO[PRO.type.astype(str).str.startswith("fixed")].iterrows():
        for lab, node, val in atlas.PULSES:
            state = {n: int(row[n]) for n in ecbn.NODES}
            state[node] = val
            for _ in range(L):
                state = synchronous_step(state, {node: val})
            seen = []
            while state not in seen:
                seen.append(state)
                state = synchronous_step(state, {})
            cycle = seen[seen.index(state):]
            arrested = all(s["Proliferation"] == 0 for s in cycle)
            asynchronous = float(D[(D.attractor == row.attractor) & (D.pulse == lab)].P_arrest_FP.iloc[0]) >= 0.5
            total += 1
            agree += arrested == asynchronous
            if arrested != asynchronous:
                disagreements.append(f"{row.attractor} {lab} (synchronous {'arrest' if arrested else 'relapse'})")
    add("D synchronous updating", "54 pulses x 6 proliferative fixed points, L = 8 steps",
        "agreement with the asynchronous writer call", f"{agree} of {total}")
    add("D synchronous updating", "54 pulses x 6 proliferative fixed points, L = 8 steps", "disagreements",
        "; ".join(disagreements) or "none")
    print("D:", agree, total, disagreements, flush=True)


if __name__ == "__main__":
    check_synchronous()
    check_structure()
    check_rates()
    check_threshold_and_length()
    pd.DataFrame(SUMMARY).to_csv(TAB + "/R1_robustness_checks.csv", index=False)
    pd.DataFrame(DRAWS).to_csv(TAB + "/R1b_robustness_draws.csv", index=False)
    print("R1: rows", len(SUMMARY))
