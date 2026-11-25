"""Gene set shared by the STRING v11.0 and HuRI universes (used by the shared-gene model runs and 09_sparsity_control.py).
Output: data/processed/shared_huri_v11.csv (column gene, alphabetical).  Usage: python 03c_shared_huri.py"""
import pandas as pd

import os
from pathlib import Path

# ============================================================== helpers
# Paths, settings and functions this step needs (written out here so the script runs on its own).

# --------------------------------------------------------------------------- paths
# Data/{raw,labels,processed} sit in the project root, one level above this step's folder.
PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HOSTNET_DATA", PROJECT / "Data"))
RAW, LABELS, PROC = DATA / "raw", DATA / "labels", DATA / "processed"

for _d in (RAW, LABELS, PROC):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

s = pd.read_csv(PROC / "features_string_v11.csv", index_col=0).index
h = pd.read_csv(PROC / "features_huri.csv", index_col=0).index
out = pd.DataFrame({"gene": sorted(set(s) & set(h))})
out.to_csv(PROC / "shared_huri_v11.csv", index=False)
print(f"shared STRING v11 / HuRI genes: {len(out)}")
