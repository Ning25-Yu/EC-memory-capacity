"""Information measures of the attractor landscape and of finite pulses (all in bits).
Manuscript: Figures 1B, 2C-D, 3, 4, 5; Table 1; Supplementary Tables S3-S4.

Reads the exact results written by scripts 01-24 and 30 and converts them into the quantities reported in the
manuscript. Nothing is re-simulated here.

  I1  landscape      per input context: nodes fixed by the inputs, free nodes, attractors, history entropy
                     H(A | u) of the attractor reached from a uniformly random initial state, tumours in TCGA
  I1b landscape_mi   decomposition H(Y) = I(U;Y) + H(Y|U) of the long-run phenotype Y over the 16 contexts
  I2  duration       arrest probability a(L) and memory capacity C(a(L)) of the eight pulses of the seven-node
                     switch, on a fine grid from the closed forms of script 30 (zero pulses included)
  I3  screen         C0101 full-network screen: information in the phenotype at the end of the pulse and after
                     withdrawal for every pulse that changes Proliferation during exposure
  I4  timecourse     information visible in Proliferation and information already committed, during and after
                     the pulse (from the exact time courses of script 14)
  I5  atlas          capacity of every single-node pulse from every proliferative attractor (L = 8)
  I5b write_time     pulse duration at which the capacity reaches 0.5 bit (exact duration responses only)
  I6  schedules      capacity of sequential and concurrent MYC / CCND2 suppression at a matched budget
  I7  rates          capacity under 200 random rate sets (0.3 to 3 times the reference rates)
  I8  lesions        capacity of p27 and p16 pulses under the driver lesions
  I9  tumours        TCGA tumours by context type and PI3K lesion
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import brentq

import ecbn
import exact_engine as E
from infotheory import arrest_for_capacity, entropy, z_capacity
from paths import DATA, TAB

PHENO = set(ecbn.PHENO)
CONTEXTS = [ecbn.ctx_name(b) for b in __import__("itertools").product([0, 1], repeat=4)]
A2 = pd.read_csv(TAB + "/A2_attractor_repertoire_16_contexts.csv")
A3 = pd.read_csv(TAB + "/A3_context_attractor_probabilities.csv")
ATLAS = pd.read_csv(TAB + "/E1_controllability_atlas.csv")
MUT = pd.read_csv(DATA + "/TCGA_UCEC_519_driver_mutations.csv")
A_HALF_BIT = arrest_for_capacity(0.5)


def fate(row):
    """Long-run phenotype label of an attractor."""
    if str(row["Proliferation"]).upper() in ("1", "X"):
        return "proliferative"
    return "apoptotic" if str(row["Apoptosis"]) == "1" else "quiescent"


def landscape():
    rows = []
    n_tumours = MUT.Context.value_counts()
    for ctx in CONTEXTS:
        const = E.percolate(ctx)
        free = sorted(set(ecbn.NODES) - set(const))
        att = A2[A2.context == ctx].merge(A3, on=["context", "attractor"])
        att["fate"] = att.apply(fate, axis=1)
        prolif = att[att.fate == "proliferative"]
        writers = ATLAS[(ATLAS.context == ctx) & (ATLAS.P_arrest_FP >= A_HALF_BIT)]
        guaranteed = (
            sorted(set.intersection(*[set(g.pulse) for _, g in writers.groupby("attractor")]))
            if len(prolif) and writers.attractor.nunique() == len(prolif)
            else []
        )
        rows.append(
            dict(
                context=ctx,
                nodes_fixed_by_inputs=len(const) - 4,
                free_nodes=len(free),
                free_molecular_nodes=len([n for n in free if n not in PHENO]),
                n_attractors=len(att),
                n_fixed_points=int((att.type == "fixed point").sum()),
                n_complex_classes=int((att.type != "fixed point").sum()),
                history_entropy_bits=entropy(att.MaBoSS_probability_t1000),
                n_proliferative_attractors=len(prolif),
                P_proliferative_attractor=float(prolif.MaBoSS_probability_t1000.sum()),
                n_pulses_at_least_half_bit_from_every_attractor=len(guaranteed),
                pulses_at_least_half_bit_from_every_attractor="; ".join(guaranteed),
                TCGA_tumours=int(n_tumours.get(ctx, 0)),
            )
        )
    D = pd.DataFrame(rows)
    D.to_csv(TAB + "/I1_landscape_entropy.csv", index=False)

    # H(Y) = I(U;Y) + H(Y|U) for the long-run phenotype Y, with U uniform or weighted by the TCGA frequencies
    att = A2.merge(A3, on=["context", "attractor"])
    att["fate"] = att.apply(fate, axis=1)
    cond = att.pivot_table(index="context", columns="fate", values="MaBoSS_probability_t1000", aggfunc="sum").fillna(0)
    out = []
    for label, w in [
        ("uniform over the 16 contexts", pd.Series(1 / 16, index=cond.index)),
        ("TCGA frequencies (519 tumours)", (n_tumours / n_tumours.sum()).reindex(cond.index).fillna(0)),
    ]:
        marginal = (cond.mul(w, axis=0)).sum()
        h_y = entropy(marginal)
        h_y_u = float(sum(w[c] * entropy(cond.loc[c]) for c in cond.index))
        out.append(dict(input_distribution=label, H_Y=h_y, H_Y_given_U=h_y_u, I_U_Y=h_y - h_y_u, H_U=entropy(w)))
    pd.DataFrame(out).to_csv(TAB + "/I1b_landscape_mutual_information.csv", index=False)
    return D, pd.DataFrame(out)


def duration():
    H2 = json.load(open(TAB + "/H2_closed_forms.json"))
    L = sp.Symbol("L", positive=True)
    grid = np.round(np.concatenate([[0.0], np.arange(0.05, 16.0001, 0.05)]), 2)
    rows, summary = [], []
    for label, v in H2.items():
        if not isinstance(v, dict) or "a_of_L" not in v:
            continue
        f = sp.lambdify(L, sp.sympify(v["a_of_L"], locals={"L": L}), "numpy")
        a = np.clip(np.broadcast_to(f(grid), grid.shape).astype(float), 0, 1)
        for x, y in zip(grid, a):
            rows.append(dict(pulse=label, L=x, arrest=y, capacity_bits=z_capacity(y)))
        s = dict(pulse=label, arrest_L8=float(a[grid == 8.0][0]), capacity_L8=z_capacity(float(a[grid == 8.0][0])))
        if a.max() > A_HALF_BIT:
            s["L_half_bit"] = brentq(lambda t: float(f(t)) - A_HALF_BIT, 1e-6, 16)
            s["L_arrest_0.95"] = brentq(lambda t: float(f(t)) - 0.95, 1e-6, 16)
            s["L_0.9_bit"] = brentq(lambda t: float(f(t)) - arrest_for_capacity(0.9), 1e-6, 16)
        summary.append(s)
    # p16 pulse: 13 tracked nodes, no closed form derived; exact numerical solution on a grid
    sp16 = E.Space("C0101", ("CDKN2A",), scope="core")
    start = E.attractor_start(sp16, A2[A2.attractor == "C0101_A1"].iloc[0])[0]
    g16 = np.round(np.concatenate([[0.0], np.arange(0.25, 16.0001, 0.25)]), 2)
    a16 = np.array([E.exact_pulse(sp16, start, {"CDKN2A": 1}, L=float(x))["P_arrest_FP"] if x > 0 else np.nan for x in g16])
    a16[0] = E.exact_pulse(sp16, start, {"CDKN2A": 1}, L=1e-9)["P_arrest_FP"]
    for x, y in zip(g16, a16):
        rows.append(dict(pulse="CDKN2A ON", L=x, arrest=y, capacity_bits=z_capacity(y)))
    f16 = lambda t: E.exact_pulse(sp16, start, {"CDKN2A": 1}, L=float(t))["P_arrest_FP"]
    summary.append(
        dict(
            pulse="CDKN2A ON",
            arrest_L8=f16(8.0),
            capacity_L8=z_capacity(f16(8.0)),
            L_half_bit=brentq(lambda t: f16(t) - A_HALF_BIT, 0.5, 8, xtol=1e-6),
            **{"L_arrest_0.95": brentq(lambda t: f16(t) - 0.95, 0.5, 12, xtol=1e-6), "L_0.9_bit": brentq(lambda t: f16(t) - arrest_for_capacity(0.9), 0.5, 12, xtol=1e-6)},
        )
    )
    pd.DataFrame(rows).to_csv(TAB + "/I2_duration_capacity.csv", index=False)
    S = pd.DataFrame(summary)
    S.to_csv(TAB + "/I2b_duration_summary.csv", index=False)
    # check the closed forms against the numerical matrix exponential of script 05
    D3 = pd.read_csv(TAB + "/D3_exact_pulse_duration_response.csv").replace({"clamp": {"CCND2+CCND3 OFF": "CCND2 OFF + CCND3 OFF"}})
    R = pd.DataFrame(rows)
    chk = D3.merge(R, left_on=["clamp", "duration"], right_on=["pulse", "L"])
    assert len(chk) > 60 and np.abs(chk.eventual_arrest - chk.arrest).max() < 1e-10
    return S


def screen():
    D6 = pd.read_csv(TAB + "/D6_full_network_pulse_screen_L8.csv")
    D6 = D6[D6.pulse != "Untreated"].copy()
    D6["phenotype_bits_end_of_pulse"] = z_capacity(1 - D6.on_Proliferation)
    D6["stored_bits_after_withdrawal"] = z_capacity(D6.eventual_arrest)
    D6["arrest_at_end_of_pulse"] = 1 - D6.on_Proliferation
    keep = D6[D6.arrest_at_end_of_pulse > 1e-9][
        ["pulse", "arrest_at_end_of_pulse", "eventual_arrest", "phenotype_bits_end_of_pulse", "stored_bits_after_withdrawal"]
    ].sort_values(["stored_bits_after_withdrawal", "phenotype_bits_end_of_pulse"], ascending=False)
    keep.to_csv(TAB + "/I3_screen_information.csv", index=False)
    return keep, len(D6)


def timecourse():
    T = pd.read_csv(TAB + "/E4_pulse_kinetics_timecourse.csv")
    T = T[T.case.str.startswith("C0101_A1")].copy()
    T["time"] = np.where(T.phase == "clamp", T.t, T.t + 8.0) if T[T.phase != "clamp"].t.min() == 0 else T.t
    T["phenotype_bits"] = z_capacity(1 - T.P_prolif_on)
    T["committed_bits"] = z_capacity(T.expected_arrest.clip(0, 1))
    T[["case", "phase", "t", "time", "P_prolif_on", "expected_arrest", "phenotype_bits", "committed_bits"]].to_csv(
        TAB + "/I4_information_timecourse.csv", index=False
    )
    return T


def atlas():
    A = ATLAS.copy()
    A["capacity_bits"] = z_capacity(A.P_arrest_FP)
    A["capacity_lower"] = z_capacity((A.P_arrest_FP - 1.96 * A.SE.fillna(0)).clip(0, 1))
    A["capacity_upper"] = z_capacity((A.P_arrest_FP + 1.96 * A.SE.fillna(0)).clip(0, 1))
    A.to_csv(TAB + "/I5_atlas_capacity.csv", index=False)
    D = pd.read_csv(TAB + "/E1_duration_response.csv")
    rows = []
    for (att, pulse), g in D.groupby(["attractor", "pulse"]):
        g = g.sort_values("L")
        if g.P_arrest_FP.max() < A_HALF_BIT:
            continue
        k = int(np.argmax(g.P_arrest_FP.values >= A_HALF_BIT))
        if k == 0:
            lw = float(g.L.iloc[0])
            note = "at or below the shortest duration computed"
        else:
            x0, x1 = g.L.iloc[k - 1], g.L.iloc[k]
            y0, y1 = g.P_arrest_FP.iloc[k - 1], g.P_arrest_FP.iloc[k]
            lw = float(x0 + (A_HALF_BIT - y0) * (x1 - x0) / (y1 - y0))
            note = "linear interpolation"
        rows.append(
            dict(attractor=att, pulse=pulse, L_half_bit=lw, note=note, capacity_L16=z_capacity(float(g.P_arrest_FP.iloc[-1])))
        )
    W = pd.DataFrame(rows)
    W.to_csv(TAB + "/I5b_write_time.csv", index=False)
    return A, W


def schedules():
    S = pd.read_csv(TAB + "/D4b_time_window_crossover_fine.csv")
    for c in ["MYC_first", "concurrent", "CCND2_first"]:
        S[c + "_bits"] = z_capacity(S[c])
    S.to_csv(TAB + "/I6_schedule_capacity.csv", index=False)
    return S


def rates():
    R = pd.read_csv(TAB + "/D5_rate_heterogeneity_200draws.csv")
    cols = ["p27_pulse_L4", "p27_pulse_L8", "MYC_then_CCND2_L8", "CCND2_then_MYC_L8", "concurrent_L8"]
    out = []
    for c in cols:
        b = z_capacity(R[c].clip(0, 1))
        out.append(
            dict(
                schedule=c,
                arrest_median=R[c].median(),
                arrest_min=R[c].min(),
                bits_median=np.median(b),
                bits_q05=np.quantile(b, 0.05),
                bits_min=b.min(),
                bits_max=b.max(),
            )
        )
        R[c + "_bits"] = b
    rate_cols = [c for c in R.columns if c.startswith("rate_")]
    R["slowest_rate"] = R[rate_cols].min(axis=1)
    R.to_csv(TAB + "/I7_rate_capacity_draws.csv", index=False)
    pd.DataFrame(out).to_csv(TAB + "/I7b_rate_capacity_summary.csv", index=False)
    return pd.DataFrame(out), R


def lesions():
    F = pd.read_csv(TAB + "/F1_driver_lesion_summary.csv")
    for c in ["p27_ON", "p16_ON", "p27_ON_with_AKT2_OFF"]:
        F[c + "_bits"] = z_capacity(F[c].fillna(0).clip(0, 1))
        F.loc[F[c].isna(), c + "_bits"] = np.nan
    F.to_csv(TAB + "/I8_lesion_capacity.csv", index=False)
    return F


def tumours():
    L1 = pd.read_csv(TAB + "/I1_landscape_entropy.csv").set_index("context")
    M = MUT.copy()
    M["proliferative_context"] = M.Context.map(L1.n_proliferative_attractors > 0)
    ev = M[M.PTEN_mutated_PASS.notna()].copy()
    ev["PI3K_lesion"] = (ev.PTEN_mutated_PASS == 1) | (ev.PIK3CA_mutated_PASS == 1)
    rows = [
        dict(group="all tumours", n=len(M), in_proliferative_context=int(M.proliferative_context.sum())),
        dict(group="tumours with evaluable mutation calls", n=len(ev), in_proliferative_context=int(ev.proliferative_context.sum())),
    ]
    pro = ev[ev.proliferative_context]
    rows.append(
        dict(
            group="proliferative contexts, evaluable",
            n=len(pro),
            PI3K_lesion=int(pro.PI3K_lesion.sum()),
            PI3K_intact=int((~pro.PI3K_lesion).sum()),
            TP53_mutated=int((pro.TP53_mutated_PASS == 1).sum()),
        )
    )
    for s, g in pro.groupby("subtype"):
        rows.append(dict(group=f"proliferative contexts, {s}", n=len(g), PI3K_lesion=int(g.PI3K_lesion.sum()), PI3K_intact=int((~g.PI3K_lesion).sum())))
    for c, g in pro.groupby("Context"):
        rows.append(dict(group=f"context {c}", n=len(g), PI3K_lesion=int(g.PI3K_lesion.sum()), PI3K_intact=int((~g.PI3K_lesion).sum())))
    T = pd.DataFrame(rows)
    T.to_csv(TAB + "/I9_tumour_counts.csv", index=False)
    return T


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(f"arrest probability giving 0.5 bit: {A_HALF_BIT:.4f}; capacity at a = 0.5: {z_capacity(0.5):.4f} bit")
    D, MI = landscape()
    print(D.drop(columns=["pulses_at_least_half_bit_from_every_attractor"]).to_string())
    print(MI.round(4).to_string())
    print(duration().round(4).to_string())
    k, n = screen()
    print(n, "pulses;", len(k), "change Proliferation during exposure\n", k.round(4).to_string())
    T = timecourse()
    A, W = atlas()
    print(W.round(3).to_string())
    schedules()
    print(rates()[0].round(4).to_string())
    print(lesions()[["attractor", "lesion", "p27_ON_bits", "p16_ON_bits", "p27_ON_with_AKT2_OFF_bits"]].round(3).to_string())
    print(tumours().to_string())
