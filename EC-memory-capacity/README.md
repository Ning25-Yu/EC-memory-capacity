# EC-memory-capacity

Network, interaction evidence, code and result tables for "Memory Capacity of Transient Interventions in a Boolean
Regulatory Network of Endometrial Cancer".

Xiaowei Liu, Ning Yu, Xin He, Kehu Yang, Yongxiu Yang

This repository reproduces every number, figure and table of the manuscript from the inputs in `data/`, and provides
the literature evidence for each of the 168 interactions of the network (`data/network_evidence/`).

## Quick start

```bash
pip install -r requirements.txt   # or: conda env create -f environment.yml
python run_all.py --figures    # redraw the figures from the result tables in results/tables (under 1 minute)
python run_all.py --new        # recompute the closed forms and the information measures, then redraw the figures
python run_all.py              # recompute everything (about 1 hour on two CPU cores)
```

Python 3.11 or later. `cmaboss` (MaBoSS) and `mpbn` are needed only for scripts 01 to 03; on Windows,
`conda install -c colomoto cmaboss` is the easiest way to install MaBoSS. Each script can also be run on its own.

## Folder structure

```
run_all.py                 runs every step in order
data/
  EC_network.bnet          the 37-node Boolean network (logical rules; Tables A4-A6)
  network_evidence/        literature evidence for each of the 168 interactions (CSV) and its references (CSV, BibTeX)
  TCGA_UCEC_*.csv          input context of each tumour, driver mutations, expression of the network genes
  signature/               the two cohort subsets used for Table A1
  published_experiments/   source data of the p16 induction experiment (Daigh et al., 2024)
code/
  lib/                     network model, exact Markov-chain engine, information measures, figure style,
                           network drawing with horizontal and vertical connectors (orthonet.py, netdraw.py)
  analysis/                analysis scripts; each writes tables to results/tables
  figures/                 one script per figure; writes results/figures
results/
  tables/                  result tables
  figures/                 Figure1-Figure5 and FigureA1-FigureA3 (PDF and 600 dpi PNG)
```

## Analysis scripts

| Script | What it does | Tables | Manuscript |
|---|---|---|---|
| `00_check_interaction_evidence.py` | checks the evidence table against the logical rules | - | Appendix D |
| `01_attractor_repertoire.py` | attractors of the 16 input contexts (minimal trap spaces, exact asynchronous dynamics) | A1, A2 | Section 3.1, Fig. A1 |
| `02_context_phenotype_probabilities.py` | basin weights of each attractor (MaBoSS) | A3, QC | Fig. 1b,c |
| `03_canalisation_and_trajectories.py` | nodes fixed by the inputs | A6 | Fig. 1b |
| `05_native_switch_exact.py` | exact chain of the seven-node switch: duration responses, schedules, rate heterogeneity | D1-D5 | Fig. 2b, 3b,c |
| `06_time_window_crossover.py` | fine schedule grid and crossover L* | D4b, D4c | Fig. 3b |
| `07_full_network_pulse_screen.py` | 60 pulses on the full network in C0101 | D6 | Fig. 2d |
| `08_context_mechanism_scope.py` | p27, p16 and p21 pulses from every proliferative attractor; simulator validation | D7 | Table A3 |
| `09_controllability_atlas.py` | all single-node pulses from all proliferative attractors; duration responses | E1 | Fig. 4, Table 1 |
| `10_edge_ablation.py` | deletion of each of the 168 interactions | E2 | Fig. 5a, Table A3 |
| `11_return_edge_combinations.py` | subsets of the four inhibitory inputs to p27 | E2b | Fig. 5a, Table A3 |
| `14_post_withdrawal_kinetics.py` | exact time courses during and after the pulse | E4 | Fig. 3a |
| `21_p16_pulse_published_data.py` | published p16 induction experiment | `Daigh2024_p16/` | Section 3.7 |
| `23_robustness_checks.py` | transition rates, synchronous updating, random deletion of interactions | R1 | Table A3 |
| `24_driver_lesions.py` | permanent lesions (AKT2 ON, MYC ON, p21 OFF, AKT2 OFF); lesion frequencies in TCGA | F1, F2 | Fig. 5b,c |
| `25_signature_check.py` | Cox model of the four input genes | G1 | Table A1 |
| `30_closed_forms.py` | committor of the switch and closed-form arrest probabilities in exact rational arithmetic | H1, H2 | Eqs. (5), (6), Appendix B |
| `31_information_measures.py` | history entropy, memory capacity (Z-channel), visible and committed information, tumour counts | I1-I9 | all figures, Table 1 |

`code/lib/infotheory.py` holds the capacity formula; `code/lib/exact_engine.py` the exact pulse engine.

## Reproducibility

Every random step uses a fixed seed and `run_all.py` fixes `PYTHONHASHSEED`. The exact results (tables D, E, F, H, I)
are reproduced exactly. The MaBoSS simulations of scripts 02 and 03 are not bit-for-bit reproducible, so basin
weights change in the third decimal place on a rerun; the tables in `results/tables` are those reported.

## Data sources

TCGA-UCEC (Genomic Data Commons; TCGA PanCancer Atlas through cBioPortal) and the source data of
Daigh et al., Nat. Commun. 2024, 15, 9181.

## Interaction evidence

`data/network_evidence/interaction_evidence.csv` lists, for every regulator and target, the sign of the interaction,
the supporting study, the experimental system and the DOI. `references.bib` holds the same studies in BibTeX format.
See `data/network_evidence/README.md`.

## How to cite

Please cite the article and this repository (see `CITATION.cff`; GitHub shows a "Cite this repository" button).

## Licence

Code: MIT (see `LICENSE`). The TCGA-derived tables and the published source data remain under the terms of their sources.
