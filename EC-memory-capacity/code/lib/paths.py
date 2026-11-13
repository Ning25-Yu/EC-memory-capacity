"""Folder locations used by every script in this repository.

All paths are derived from the location of this file, so the scripts can be run from any working directory.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inputs
DATA = os.path.join(ROOT, "data")
DOWNLOADS = os.path.join(
    DATA, "downloads"
)  # raw cBioPortal / MSigDB files (created by 00_download_public_data.py)
P16_DATA = os.path.join(DATA, "published_experiments", "Daigh2024_p16")

# Outputs
TAB = os.path.join(ROOT, "results", "tables")
IN_TAB = TAB  # tables written by earlier steps are read from the same folder
P16_TAB = os.path.join(TAB, "Daigh2024_p16")
FIGURES = os.path.join(ROOT, "results", "figures")  # manuscript figures

for _folder in (TAB, P16_TAB, FIGURES):
    os.makedirs(_folder, exist_ok=True)


def need(path):
    """Return the path if it exists; otherwise stop with a clear message."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"required input missing: {os.path.relpath(path, ROOT)}")
    return path
