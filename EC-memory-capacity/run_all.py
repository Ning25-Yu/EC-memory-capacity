"""Rebuild the result tables and the figures of the manuscript, in order.

    python run_all.py              all analyses, then all figures (about 1 h on two CPU cores)
    python run_all.py --new        only the closed forms and information measures (scripts 30 and 31), then the figures
    python run_all.py --figures    figures only, from the result tables already in results/tables (under 1 min)

Analyses write to results/tables and the figure scripts to results/figures.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(ROOT, "code", "analysis")
FIGURES = os.path.join(ROOT, "code", "figures")

UPSTREAM_STEPS = [
    "00_check_interaction_evidence.py",
    "01_attractor_repertoire.py",
    "02_context_phenotype_probabilities.py",
    "03_canalisation_and_trajectories.py",
    "05_native_switch_exact.py",
    "06_time_window_crossover.py",
    "07_full_network_pulse_screen.py",
    "08_context_mechanism_scope.py",
    "09_controllability_atlas.py",
    "10_edge_ablation.py",
    "11_return_edge_combinations.py",
    "14_post_withdrawal_kinetics.py",
    "21_p16_pulse_published_data.py",
    "23_robustness_checks.py",
    "24_driver_lesions.py",
    "25_signature_check.py",
]
NEW_STEPS = ["30_closed_forms.py", "31_information_measures.py"]
FIGURE_STEPS = [
    "figure1_landscape.py",
    "figure2_pulse.py",
    "figure3_dynamics.py",
    "figure4_atlas.py",
    "figure5_structure.py",
    "figure_appendix.py",
]


def run(folder, script):
    start = time.time()
    print(f"==> {os.path.relpath(os.path.join(folder, script), ROOT)}", flush=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")  # fixed hash seed: reproducible set order
    subprocess.run([sys.executable, script], cwd=folder, env=env, check=True)
    print(f"    done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    if "--figures" not in sys.argv:
        if "--new" not in sys.argv:
            for step in UPSTREAM_STEPS:
                run(ANALYSIS, step)
        for step in NEW_STEPS:
            run(ANALYSIS, step)
    for step in FIGURE_STEPS:
        run(FIGURES, step)
    print("finished")
