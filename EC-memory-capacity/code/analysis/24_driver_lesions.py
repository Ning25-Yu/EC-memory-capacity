"""Driver lesions of endometrial cancer imposed on the network.
Manuscript: Results 2.5 and 2.8, Figure 7C, Table S10E (controllability under driver lesions).

The recurrent driver events of endometrial cancer converge on nodes of the network. Each event is imposed as a
permanent lesion, that is, the rule of the node is replaced by a constant that holds before, during and after a pulse:
  PI3K lesion  (PTEN loss or PIK3CA mutation)  AKT2 fixed ON
  TP53 lesion  (TP53 mutation)                 CDKN1A (p21) fixed OFF
  WNT lesion   (CTNNB1 mutation)               MYC fixed ON
  AKT blockade (sustained AKT inhibition)      AKT2 fixed OFF, used to ask whether the PI3K lesion can be reversed
For every proliferative attractor and lesion the script first relaxes the attractor in the lesioned network, records
the probability that a proliferative state is kept, and then applies each of the 54 single-node pulses (L = 8) and the
combined pulse p27 ON with AKT2 OFF from the proliferative states.
Combinations above the exact state limit are simulated for the p27 and p16 pulses (3 x 2,000 Gillespie trajectories
of the full network; arrest is the fraction in a fixed point with Proliferation OFF at t = 120 after withdrawal).
The second part counts the lesions in the TCGA cohort (data/TCGA_UCEC_519_driver_mutations.csv: non-synonymous
PTEN, PIK3CA, TP53 and CTNNB1 mutations per tumour; the main analysis uses calls with FILTER = PASS, the sensitivity
analysis calls with any filter value). A tumour carries the PI3K lesion when PTEN or PIK3CA is mutated.
Outputs: results/tables/F1_driver_lesion_atlas.csv (one row per attractor, lesion and pulse),
         results/tables/F1_driver_lesion_summary.csv (writers per attractor and lesion),
         results/tables/F2_driver_lesions_in_patients.csv (lesion frequencies by controllability class and subtype),
         results/tables/F2_lesion_readouts.csv (expression read-outs of the lesioned nodes).
Run with --patients-only to repeat the second part without recomputing the lesion atlas."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import importlib.util

import numpy as np
import pandas as pd
import sympy

import ecbn
import exact_engine as E
from paths import DATA, TAB

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 300_000

LESIONS = {
    "none": {},
    "PI3K lesion (AKT2 ON)": {"AKT2": 1},
    "TP53 lesion (p21 OFF)": {"CDKN1A": 0},
    "WNT lesion (MYC ON)": {"MYC": 1},
    "AKT blockade (AKT2 OFF)": {"AKT2": 0},
}


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


atlas = load("atlas", "09_controllability_atlas.py")
abl = load("abl", "10_edge_ablation.py")


def impose(lesion):
    for node, value in lesion.items():
        E.PYR[node] = compile(str(bool(value)), node, "eval")
        E.REG[node] = set()
        E._EXPR[node] = sympy.true if value else sympy.false


def attractor_states(row):
    """Full states of a proliferative attractor of the unlesioned network, with their probabilities."""
    if str(row.type).startswith("fixed"):
        return abl.full_states_fixed(row)
    return abl.full_states_class(row.context, row)


def start_states(states, lesion):
    """The attractor states with the lesioned nodes set to their fixed values."""
    out = {}
    for full, weight in states.items():
        d = dict(full)
        d.update(lesion)
        key = tuple((n, d[n]) for n in ecbn.NODES)
        out[key] = out.get(key, 0.0) + weight
    return out


def pulse_under_lesion(ctx, start_full, clamp, L=8.0):
    """Relax the start in the current (lesioned) network, then pulse from the proliferative terminal states."""
    sp = E.Space(ctx, tuple(clamp))
    if "Proliferation" not in sp.ix:
        on = float(sp.const["Proliferation"])
        return on, (0.0 if on else np.nan), "constant"
    st = {}
    for full, weight in start_full.items():
        d = dict(full)
        d.update(sp.const)
        key = tuple(int(d[n]) for n in sp.nodes)
        st[key] = st.get(key, 0.0) + weight
    explored = sp.explore(list(st), set(), CAP)
    if explored is None:
        return np.nan, np.nan, "above state limit"
    order, pos, Q, adjacency = explored
    p0 = np.zeros(len(order))
    for s, weight in st.items():
        p0[pos[s]] += weight
    classes = E.terminal_classes(adjacency)
    absorbed = E.absorption(Q, classes, p0)
    prol = sp.ix["Proliferation"]
    proliferative = {}
    for c, members in enumerate(classes):
        if absorbed[c] > 1e-12 and any(order[i][prol] == 1 for i in members):
            for i, p in zip(members, E.stationary(Q, members)):
                proliferative[order[i]] = proliferative.get(order[i], 0.0) + absorbed[c] * p
    mass = sum(proliferative.values())
    if mass < 1e-9:
        return mass, np.nan, "no proliferative state"
    proliferative = {k: v / mass for k, v in proliferative.items()}
    result = E.exact_pulse(sp, proliferative, clamp, L, cap=CAP)
    if result is None:
        return mass, np.nan, "above state limit"
    return mass, result["P_arrest_FP"], "exact"


KEY_PULSES = ("CDKN1B=1", "CDKN2A=1", "CDKN1B=1 + AKT2=0")


def simulate(ctx, start_full, clamp, L=8.0, n=2000, seed=0, t_free=120.0):
    """Gillespie simulation of the full network with unit rates. Returns the fraction of trajectories that are in a
    fixed point with Proliferation OFF at t_free after withdrawal, and the fraction not yet in a fixed point."""
    rng = np.random.default_rng(seed)
    inputs = dict(zip(ecbn.INPUTS, (int(b) for b in ctx[1:])))
    free = [m for m in ecbn.NODES if m not in ecbn.INPUTS]
    states = [dict(full) for full in start_full]
    weights = np.array(list(start_full.values()))
    weights = weights / weights.sum()
    arrested = unresolved = 0

    def unstable(x, frozen):
        return [m for m in free if m not in frozen and int(eval(E.PYR[m], {}, x)) != x[m]]

    def run(x, t_end, frozen):
        t = 0.0
        while True:
            moves = unstable(x, frozen)
            if not moves:
                return
            t += rng.exponential(1.0 / len(moves))
            if t > t_end:
                return
            m = moves[rng.integers(len(moves))]
            x[m] = 1 - x[m]

    for _ in range(n):
        x = dict(states[rng.choice(len(states), p=weights)])
        x.update(inputs)
        x.update(clamp)
        run(x, L, set(clamp))
        run(x, t_free, set())
        if unstable(x, set()):
            unresolved += 1
        elif x["Proliferation"] == 0:
            arrested += 1
    return arrested / n, unresolved / n


def simulate_unresolved(rows):
    """Replace the p27 and p16 results that exceeded the exact state limit by simulation."""
    by_attractor = {r.attractor: r for _, r in atlas.PRO.iterrows()}
    for r in rows:
        if r["method"] != "above state limit" or r["pulse"] not in KEY_PULSES:
            continue
        lesion = LESIONS[r["lesion"]]
        clamp = {a.split("=")[0]: int(a.split("=")[1]) for a in r["pulse"].split(" + ")}
        states = attractor_states(by_attractor[r["attractor"]])
        impose(lesion)
        try:
            reps = [simulate(r["context"], start_states(states, lesion), clamp, seed=k) for k in (1, 2, 3)]
        finally:
            abl.restore()
        r["P_arrest_FP"] = float(np.mean([a for a, _ in reps]))
        r["SE"] = float(np.std([a for a, _ in reps], ddof=1) / np.sqrt(3))
        r["unresolved_at_t120"] = float(max(u for _, u in reps))
        r["method"] = "simulation (3 x 2,000)"
        print("simulated", r["attractor"], r["lesion"], r["pulse"], round(r["P_arrest_FP"], 4), flush=True)


def input_class(context):
    """Controllability class from the input bits (same rule as script 19)."""
    if context[3] == "1" or context == "C1000":
        return "I"
    return {("1", "1"): "II", ("1", "0"): "III", ("0", "1"): "IV", ("0", "0"): "V"}[(context[2], context[4])]


def patients():
    from scipy.stats import mannwhitneyu

    M = pd.read_csv(DATA + "/TCGA_UCEC_519_driver_mutations.csv")
    M["class"] = M.Context.map(input_class)
    rows = []
    for calls in ("PASS", "any_filter"):
        d = M.dropna(subset=[f"PTEN_mutated_{calls}"]).copy()
        d["PI3K"] = ((d[f"PTEN_mutated_{calls}"] == 1) | (d[f"PIK3CA_mutated_{calls}"] == 1)).astype(int)
        groups = [(f"class {k}", d[d["class"] == k]) for k in ["I", "II", "III", "IV", "V"]]
        pro = d[d["class"] != "I"]
        groups.append(("classes II-V", pro))
        groups += [(f"classes II-V, {s}", pro[pro.subtype == s]) for s in ["POLE", "MSI", "CN_LOW", "CN_HIGH"]]
        for name, g in groups:
            rows.append({"calls": calls, "group": name, "n_evaluable": len(g),
                         "PTEN": int(g[f"PTEN_mutated_{calls}"].sum()),
                         "PIK3CA": int(g[f"PIK3CA_mutated_{calls}"].sum()),
                         "PI3K_lesion": int(g.PI3K.sum()),
                         "TP53": int(g[f"TP53_mutated_{calls}"].sum()),
                         "CTNNB1": int(g[f"CTNNB1_mutated_{calls}"].sum()),
                         "CTNNB1_without_PI3K_lesion": int(((g[f"CTNNB1_mutated_{calls}"] == 1) & (g.PI3K == 0)).sum()),
                         "PI3K_intact": int((g.PI3K == 0).sum()),
                         "PI3K_intact_fraction": float((g.PI3K == 0).mean()) if len(g) else np.nan})
    pd.DataFrame(rows).to_csv(TAB + "/F2_driver_lesions_in_patients.csv", index=False)

    # expression read-outs of the lesioned nodes (the mutation is expected to move the node in the lesion direction)
    G = pd.read_csv(DATA + "/TCGA_UCEC_502_network_gene_log2RSEM.csv")
    d = M.dropna(subset=["PTEN_mutated_PASS"]).merge(G, on="sample")
    d["PI3K"] = ((d.PTEN_mutated_PASS == 1) | (d.PIK3CA_mutated_PASS == 1)).astype(int)
    out = []
    # AKT is activated by phosphorylation, so the PI3K lesion has no expression read-out in these data
    for flag, gene in [("TP53_mutated_PASS", "CDKN1A"), ("CTNNB1_mutated_PASS", "MYC")]:
        a, b = d[d[flag] == 1][gene], d[d[flag] == 0][gene]
        out.append({"lesion": flag, "node_expression": gene, "n_mutated": len(a), "n_wild_type": len(b),
                    "median_mutated": float(a.median()), "median_wild_type": float(b.median()),
                    "Mann_Whitney_P": float(mannwhitneyu(a, b).pvalue)})
    pd.DataFrame(out).to_csv(TAB + "/F2_lesion_readouts.csv", index=False)
    print("F2: rows", len(rows))


if __name__ == "__main__":
    if "--patients-only" in sys.argv:
        patients()
        sys.exit()
    pulses = [(label, {node: value}) for label, node, value in atlas.PULSES]
    pulses.append(("CDKN1B=1 + AKT2=0", {"CDKN1B": 1, "AKT2": 0}))
    rows = []
    partial = TAB + "/F1_driver_lesion_atlas.csv"
    if "--resume" in sys.argv and os.path.exists(partial):
        rows = pd.read_csv(partial).to_dict("records")
    finished = {r["attractor"] for r in rows}
    for _, row in atlas.PRO.iterrows():
        if row.attractor in finished:
            continue
        states = attractor_states(row)
        for name, lesion in LESIONS.items():
            impose(lesion)
            try:
                start = start_states(states, lesion)
                for label, clamp in pulses:
                    if any(n in lesion and name != "PI3K lesion (AKT2 ON)" for n in clamp):
                        continue  # a pulse on the lesioned node is only meaningful as inhibition of the PI3K lesion
                    mass, arrest, method = pulse_under_lesion(row.context, start, clamp)
                    rows.append({"context": row.context, "attractor": row.attractor, "attractor_type": row.type,
                                 "lesion": name, "pulse": label, "proliferative_mass": mass,
                                 "P_arrest_FP": arrest, "method": method})
            finally:
                abl.restore()
            print(row.attractor, name, flush=True)
        pd.DataFrame(rows).to_csv(TAB + "/F1_driver_lesion_atlas.csv", index=False)
    simulate_unresolved(rows)
    D = pd.DataFrame(rows)
    D.to_csv(TAB + "/F1_driver_lesion_atlas.csv", index=False)

    summary = []
    for (attractor, lesion), d in D.groupby(["attractor", "lesion"], sort=False):
        single = d[~d.pulse.str.contains(r"\+")]
        writers = single[single.P_arrest_FP >= 0.5].pulse.tolist()

        def value(pulse):
            v = d[d.pulse == pulse].P_arrest_FP
            return float(v.iloc[0]) if len(v) else np.nan

        summary.append({"context": d.context.iloc[0], "attractor": attractor, "lesion": lesion,
                        "proliferative_mass": d.proliferative_mass.max(),
                        "p27_ON": value("CDKN1B=1"), "p16_ON": value("CDKN2A=1"),
                        "p27_ON_with_AKT2_OFF": value("CDKN1B=1 + AKT2=0"),
                        "n_writers": len(writers), "writers": "; ".join(writers),
                        "n_unresolved": int((single.method == "above state limit").sum())})
    pd.DataFrame(summary).to_csv(TAB + "/F1_driver_lesion_summary.csv", index=False)
    print("F1: rows", len(D))
    patients()
