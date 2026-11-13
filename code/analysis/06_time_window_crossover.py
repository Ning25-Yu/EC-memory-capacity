"""Fine-grid equal-budget schedule comparison on the exact C0101 128-state CTMC, and the crossover L*.
Manuscript: Figure 3D (fine schedule grid and the crossover L*).

Uses run_schedule from 05_native_switch_exact.py (exact matrix exponentials; no sampling).
Output: results/tables/D4b_time_window_crossover_fine.csv (+ crossover written to results/tables/D4c_time_window_crossover_Lstar.csv)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import importlib.util

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from paths import TAB

spec = importlib.util.spec_from_file_location(
    "nat", os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_native_switch_exact.py")
)
nat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nat)
MYC_first = lambda L: nat.run_schedule([({"MYC": 0}, L), ({"CCND2": 0}, L)])
concurrent = lambda L: nat.run_schedule([({"MYC": 0, "CCND2": 0}, L), ({}, L)])
CCND2_first = lambda L: nat.run_schedule([({"CCND2": 0}, L), ({"MYC": 0}, L)])

if __name__ == "__main__":
    grid = np.concatenate([np.arange(0.25, 2.0 + 1e-9, 0.0625), np.arange(2.5, 8.0 + 1e-9, 0.5)])
    D = pd.DataFrame(
        {
            "L": grid,
            "MYC_first": [MYC_first(L) for L in grid],
            "concurrent": [concurrent(L) for L in grid],
            "CCND2_first": [CCND2_first(L) for L in grid],
        }
    )
    D.to_csv(os.path.join(TAB, "D4b_time_window_crossover_fine.csv"), index=False)
    Ls = brentq(lambda L: MYC_first(L) - concurrent(L), 0.5, 3.0, xtol=1e-12)
    pd.DataFrame(
        [{"L_star": Ls, "arrest_at_L_star": concurrent(Ls), "method": "brentq on exact CTMC, xtol 1e-12"}]
    ).to_csv(os.path.join(TAB, "D4c_time_window_crossover_Lstar.csv"), index=False)
    print(f"D4b: {len(D)} grid points; L* = {Ls:.10f}")
