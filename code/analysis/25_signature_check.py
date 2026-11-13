"""Reproduction of the four-gene signature from the two cohort tables.
Manuscript: Results 2.1, Equation 1, Methods 4.3.

The script repeats, in Python, the stepwise Cox regression of bioinformatics/04_stepwise_cox.R: bidirectional selection
with the BIC penalty, starting from the candidate genes, in the training cohort. It then reports the coefficients, the
concordance index in the training and testing cohorts, and the log-rank test between risk groups defined by the
training median of the risk score.
Outputs: results/tables/G1_signature_coefficients.csv, results/tables/G1_signature_performance.csv."""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from paths import DATA, TAB

warnings.filterwarnings("ignore")

CANDIDATES = [
    "PLA2G10", "FSTL3", "B4GALT1", "ASB2", "SH3GL2", "HOXB6",
    "KLRB1", "ZAP70", "FASLG", "STK11", "FBXL16", "IRF5",
    "DUSP6", "SLC39A3", "ATP6V1C2", "TRPC1", "TSC22D3", "PTX3",
    "RET", "STAT6", "CD3D", "ANXA2", "ANG", "HLA-DMB",
]
PUBLISHED = {"KLRB1": -0.805056, "IRF5": 0.584826, "STAT6": -0.585980, "TSC22D3": 0.331543}


def read_cohort(name):
    d = pd.read_csv(f"{DATA}/signature/{name}")
    d.columns = ["sample", "time", "status"] + list(d.columns[3:])
    return d


def fit(data, genes):
    return CoxPHFitter().fit(data[["time", "status"] + genes], "time", "status")


def bic(data, genes):
    if not genes:
        null = CoxPHFitter().fit(data[["time", "status"]].assign(constant=0.0), "time", "status")
        return -2 * null.log_likelihood_
    return -2 * fit(data, genes).log_likelihood_ + np.log(len(data)) * len(genes)


def stepwise(data, candidates):
    """Bidirectional stepwise selection: at each step make the single removal or addition that lowers BIC most."""
    current = list(candidates)
    best = bic(data, current)
    while True:
        moves = [([g for g in current if g != gene], gene) for gene in current]
        moves += [(current + [gene], gene) for gene in candidates if gene not in current]
        scored = [(bic(data, genes), genes) for genes, _ in moves]
        value, genes = min(scored, key=lambda item: item[0])
        if value >= best - 1e-9:
            return current
        best, current = value, genes


if __name__ == "__main__":
    train = read_cohort("Training_set_raw.csv")
    test = read_cohort("Validation_set_raw.csv")

    selected = stepwise(train, [g for g in CANDIDATES if g in train.columns])
    model = fit(train, selected)

    coefficients = model.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    coefficients.columns = ["coefficient", "HR", "HR_lower_95", "HR_upper_95", "P"]
    coefficients["published_coefficient"] = [PUBLISHED.get(g, np.nan) for g in coefficients.index]
    coefficients.to_csv(TAB + "/G1_signature_coefficients.csv", index_label="gene")

    assert set(selected) == set(PUBLISHED), selected
    assert all(abs(model.params_[g] - PUBLISHED[g]) < 1e-4 for g in PUBLISHED)

    cutoff = float(np.median(model.predict_log_partial_hazard(train)))
    rows = []
    for name, cohort in (("training", train), ("testing", test)):
        score = model.predict_log_partial_hazard(cohort)
        high = score > cutoff
        test_result = logrank_test(cohort.time[high], cohort.time[~high], cohort.status[high], cohort.status[~high])
        rows.append({"cohort": name, "n": len(cohort), "deaths": int(cohort.status.sum()),
                     "C_index": concordance_index(cohort.time, -score, cohort.status),
                     "n_high_risk": int(high.sum()), "log_rank_P": test_result.p_value})
    pd.DataFrame(rows).to_csv(TAB + "/G1_signature_performance.csv", index=False)
    print("G1: selected", selected)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
