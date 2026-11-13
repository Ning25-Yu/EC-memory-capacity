"""Retrospective analysis of the published p16 pulse experiments (source data of Daigh et al., 2024).
Manuscript: Figure 6A-C, Figure S14 and Table S8.

Inputs (data/published_experiments/Daigh2024_p16/): the author-supplied source-data rows of Figures 1D, 1G-1J and 2J
and Extended Data Figure 2F, exported from Daigh2024_Source_Data.xlsx. The model prediction (eventual arrest after a
p16 pulse from the C0101 proliferative fixed point) is read from results/tables/E1_duration_response.csv.

No model is fitted to the experiments, and cycling-marker percentages are not converted into absorption probabilities.
Biological experiments, not cells, are the independent units.

Outputs (results/tables/Daigh2024_p16/):
  all_pulse_source_values.csv          every phospho-Rb / EdU value of Figures 1G-1J (59 values)
  pulse_group_summary.csv              mean, SD and pointwise 95% t interval per condition
  duration_effect_sizes.csv            1-, 3- and 5-day pulse means and their differences per panel
  observation_time_effects.csv         24 h versus 72 h means per pulse duration (descriptive)
  exact_trend_layout_sensitivity.csv   exact Jonckheere-Terpstra trend tests under two replicate layouts
  depletion_source_ratios.csv          release-phase re-entry ratios after si-p21, si-p27 or si-p53 (Figure 2J)
  depletion_effect_estimates.csv       geometric mean ratios, log-t intervals and exact sign-flip tests
  knockdown_validation_*.csv           residual target protein after knockdown (Extended Data Figure 2F)
  p16_*_degradation.csv                p16 reporter degradation after auxin addition (Figure 1D)
  model_p16_duration_predictions.csv   model arrest probability after a p16 pulse of duration L
  prediction_assessments.csv           qualitative comparison of model predictions and experiments"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import csv

import numpy as np

from paths import P16_DATA, P16_TAB, TAB
from statistical_methods import describe, exact_jt_blocked, exact_jt_independent, holm, log_ratio_analysis


def raw_rows(name):
    with open(os.path.join(P16_DATA, name), encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def write_csv(name, rows):
    assert rows, name
    with open(os.path.join(P16_TAB, name), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pulse_data():
    """Phospho-Rb and EdU positivity after 1-, 3- and 5-day p16 pulses (Figures 1G-1J)."""
    out = []
    for panel, readout, hours, n in [
        ("1G", "phospho_Rb", 24, 3),
        ("1H", "EdU", 24, 4),
        ("1I", "phospho_Rb", 72, 3),
        ("1J", "EdU", 72, 3),
    ]:
        name = f"Figure_{panel}_source_rows.csv"
        rows = raw_rows(name)
        header = rows[next(i for i, r in enumerate(rows) if "Cycling" in r)]
        n_rows = 0
        for ri, row in enumerate(rows):
            if not row or not row[0].startswith("Rep"):
                continue
            n_rows += 1
            for col, condition in enumerate(header):
                if col == 0 or not condition:
                    continue
                if condition.endswith("day"):
                    role, duration = "pulse_duration", int(condition.split()[0])
                elif condition == "Cycling":
                    role, duration = "cycling_control", 0
                else:
                    role, duration = "unreleased_control", 5
                value = float(row[col])
                assert 0 <= value <= 100
                out.append(
                    dict(
                        panel=panel,
                        readout=readout,
                        nominal_readout_hours=hours,
                        condition=condition,
                        pulse_duration_days=duration,
                        role=role,
                        replicate_id_in_source=row[0],
                        positive_cells_percent=value,
                        original_file=name,
                        original_csv_row=ri + 1,
                        original_csv_column=col + 1,
                    )
                )
        assert n_rows == n
    assert len(out) == 59
    write_csv("all_pulse_source_values.csv", out)

    summary = []
    for panel in ["1G", "1H", "1I", "1J"]:
        for condition in dict.fromkeys(r["condition"] for r in out if r["panel"] == panel):
            rows = [r for r in out if r["panel"] == panel and r["condition"] == condition]
            r = rows[0]
            summary.append(
                dict(
                    panel=panel,
                    readout=r["readout"],
                    nominal_readout_hours=r["nominal_readout_hours"],
                    condition=condition,
                    pulse_duration_days=r["pulse_duration_days"],
                    role=r["role"],
                    **describe([x["positive_cells_percent"] for x in rows]),
                    units="Percentage points on the original positive-cell scale",
                )
            )
    write_csv("pulse_group_summary.csv", summary)

    effects = []
    for panel in ["1G", "1H", "1I", "1J"]:
        s = {r["condition"]: r for r in summary if r["panel"] == panel}
        means = [s[f"{d} day"]["mean"] for d in [1, 3, 5]]
        values = [
            [r["positive_cells_percent"] for r in out if r["panel"] == panel and r["condition"] == f"{d} day"]
            for d in [1, 3, 5]
        ]
        effects.append(
            dict(
                panel=panel,
                readout=s["1 day"]["readout"],
                hours=s["1 day"]["nominal_readout_hours"],
                mean_1_day=means[0],
                mean_3_day=means[1],
                mean_5_day=means[2],
                difference_5_minus_1_pp=means[2] - means[0],
                relative_reduction_5_vs_1=1 - means[2] / means[0],
                mean_cycling=s["Cycling"]["mean"],
                ratio_of_means_5_day_vs_cycling=means[2] / s["Cycling"]["mean"],
                descending_group_means=means[0] > means[1] > means[2],
                completely_separated_duration_groups=(
                    min(values[0]) > max(values[1]) and min(values[1]) > max(values[2])
                ),
            )
        )
    write_csv("duration_effect_sizes.csv", effects)

    changes = []
    for readout in ["phospho_Rb", "EdU"]:
        for d in [1, 3, 5]:
            rs = [
                r
                for r in summary
                if r["readout"] == readout and r["role"] == "pulse_duration" and r["pulse_duration_days"] == d
            ]
            a = next(r for r in rs if r["nominal_readout_hours"] == 24)
            b = next(r for r in rs if r["nominal_readout_hours"] == 72)
            changes.append(
                dict(
                    readout=readout,
                    pulse_duration_days=d,
                    mean_24h=a["mean"],
                    mean_72h=b["mean"],
                    difference_72h_minus_24h_pp=b["mean"] - a["mean"],
                    interpretation="Descriptive comparison of separate panels, not a single-cell recovery rate",
                )
            )
    write_csv("observation_time_effects.csv", changes)
    return out, summary, effects


def trend_sensitivity(raw):
    """Exact one-sided trend tests (1 > 3 > 5 days) under independent-group and matched-row layouts."""
    result = []
    for panel in ["1G", "1H", "1I", "1J"]:
        groups = [
            [r["positive_cells_percent"] for r in raw if r["panel"] == panel and r["condition"] == f"{d} day"]
            for d in [1, 3, 5]
        ]
        for layout, test in [
            ("independent_groups", exact_jt_independent),
            ("matched_rows_if_valid", exact_jt_blocked),
        ]:
            result.append(dict(panel=panel, layout_assumption=layout, **test(groups)))
    for layout in ["independent_groups", "matched_rows_if_valid"]:
        subset = [r for r in result if r["layout_assumption"] == layout]
        for r, p in zip(subset, holm([r["p_one_sided"] for r in subset])):
            r["Holm_adjusted_p_4_endpoints"] = p
    write_csv("exact_trend_layout_sensitivity.csv", result)
    return result


def depletion_data():
    """Release-phase re-entry after si-p21, si-p27 or si-p53 (Figure 2J) and knockdown efficiency (Extended Data 2F)."""
    rows = raw_rows("Figure_2J_source_rows.csv")
    header = rows[next(i for i, r in enumerate(rows) if "Nts" in r)]
    raw = []
    for ri, row in enumerate(rows):
        if not row or not row[0].startswith("Rep"):
            continue
        for ci, condition in enumerate(header):
            if ci == 0 or not condition:
                continue
            raw.append(
                dict(
                    condition=condition,
                    source_replicate=row[0],
                    reentry_ratio_to_Nts=float(row[ci]),
                    source_csv_row=ri + 1,
                    source_csv_column=ci + 1,
                    Nts_is_normalization_reference=condition == "Nts",
                )
            )
    assert len(raw) == 16
    write_csv("depletion_source_ratios.csv", raw)

    estimates = []
    for condition in ["si-p21", "si-p27", "si-p53"]:
        values = [r["reentry_ratio_to_Nts"] for r in raw if r["condition"] == condition]
        r = dict(condition=condition, **log_ratio_analysis(values))
        if r["log_t95_lower"] > 1:
            r["directional_assessment"] = "supports increased re-entry"
        elif r["log_t95_upper"] < 1:
            r["directional_assessment"] = "opposite direction"
        else:
            r["directional_assessment"] = "undetermined; no clear support for increased re-entry"
        estimates.append(r)
    p_t = holm([r["log_t_p_two_sided"] for r in estimates])
    p_flip = holm([r["exact_signflip_p_two_sided"] for r in estimates])
    for r, a, b in zip(estimates, p_t, p_flip):
        r.update(log_t_Holm_p_3_conditions=a, signflip_Holm_p_3_conditions=b)
    write_csv("depletion_effect_estimates.csv", estimates)

    rows = raw_rows("Extended_Data_Fig_2F_source_rows.csv")
    hi = next(i for i, r in enumerate(rows) if "Nts" in r)
    header = rows[hi]
    protein = []
    for ri, row in enumerate(rows[hi + 1 :], hi + 1):
        if len(row) < 5 or not row[1]:
            continue
        for ci, condition in enumerate(header):
            if ci == 0 or not condition:
                continue
            protein.append(
                dict(
                    condition=condition,
                    source_row=ri + 1,
                    relative_target_protein=float(row[ci]),
                    source_csv_column=ci + 1,
                )
            )
    assert len(protein) == 12
    write_csv("knockdown_validation_source_values.csv", protein)
    summary = []
    for condition in ["si-p21", "si-p27", "si-p53"]:
        y = [r["relative_target_protein"] for r in protein if r["condition"] == condition]
        summary.append(
            dict(
                condition=condition,
                source_values=len(y),
                mean_residual_protein=float(np.mean(y)),
                minimum=min(y),
                maximum=max(y),
                mean_fraction_reduced=1 - float(np.mean(y)),
            )
        )
    write_csv("knockdown_validation_summary.csv", summary)
    return estimates


def input_degradation():
    """p16 reporter fluorescence after auxin addition (Figure 1D): published median and the 50 displayed cells."""
    rows = raw_rows("Figure_1D_source_rows.csv")
    times = next(r for r in rows if r and r[0] == "Time Since Auxin Addition (hr)")
    medians = next(r for r in rows if r and r[0] == "Median log2(mRuby-p16) (A.U.)")
    median = [
        dict(hours_since_auxin=float(t), published_median_log2_fluorescence=float(v), source_csv_column=ci)
        for ci, (t, v) in enumerate(zip(times[1:], medians[1:]), 2)
        if t
    ]
    cells = []
    cell_rows = [(ri, row) for ri, row in enumerate(rows, 1) if row and row[0].startswith("Cell ")]
    assert len(cell_rows) == 50
    for ri, row in cell_rows:
        for ci, t in enumerate(times[1:], 1):
            if not t:
                continue
            value = float(row[ci]) if ci < len(row) and row[ci] else ""
            cells.append(
                dict(
                    source_cell=row[0],
                    hours_since_auxin=float(t),
                    log2_fluorescence=value,
                    source_csv_row=ri,
                    source_csv_column=ci + 1,
                )
            )
    assert len(median) == 183 and len(cells) == 9150
    write_csv("p16_published_median_degradation.csv", median)
    write_csv("p16_displayed_cell_degradation.csv", cells)


def model_predictions():
    """Model arrest probability after a p16 (CDKN2A ON) pulse from the C0101 proliferative fixed point (table E1)."""
    with open(os.path.join(TAB, "E1_duration_response.csv"), encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["pulse"] == "CDKN2A=1" and r["attractor"] == "C0101_A1"]
    rows = sorted(rows, key=lambda r: float(r["L"]))
    values = [float(r["P_arrest_FP"]) for r in rows]
    assert all(b > a for a, b in zip(values, values[1:]))
    write_csv("model_p16_duration_predictions.csv", rows)


def main():
    raw, summary, effects = pulse_data()
    trends = trend_sensitivity(raw)
    estimates = depletion_data()
    input_degradation()
    model_predictions()
    p27 = next(r for r in estimates if r["condition"] == "si-p27")
    write_csv(
        "prediction_assessments.csv",
        [
            dict(
                id="H1",
                prediction="Longer p16 pulses give lower post-release cycling readouts",
                assessment=(
                    "consistent (directional, across systems)"
                    if all(r["descending_group_means"] for r in effects)
                    else "mixed or inconsistent"
                ),
                scope="No time calibration or model fitting",
            ),
            dict(
                id="H2",
                prediction="Suppression persists to the later observation window",
                assessment=(
                    "consistent (relative suppression still present at 72 h)"
                    if all(r["ratio_of_means_5_day_vs_cycling"] < 1 for r in effects if r["hours"] == 72)
                    else "undetermined"
                ),
                scope="Finite-window population readouts, not irreversible single-cell absorption",
            ),
            dict(
                id="H3",
                prediction="p27 depletion increases re-entry during release",
                assessment=p27["directional_assessment"],
                scope="Partial whole-protein depletion in a different cellular context, not a selective edge deletion",
            ),
            dict(
                id="H4",
                prediction="Specific p27 outputs to MYC and CCND2 are necessary for memory",
                assessment="undetermined",
                scope="Neither output was selectively perturbed",
            ),
            dict(
                id="H5",
                prediction="Exact arrest probability in C0101 after L = 8",
                assessment="undetermined",
                scope="Marker-positive percentages and experimental days are not model state probabilities or model time",
            ),
        ],
    )
    print(
        f"values: {len(raw)} | groups: {len(summary)} | trend tests: {len(trends)} | "
        f"si-p27 geometric mean ratio {p27['geometric_mean_ratio']:.3f} "
        f"(95% CI {p27['log_t95_lower']:.3f}-{p27['log_t95_upper']:.3f})"
    )


if __name__ == "__main__":
    main()
