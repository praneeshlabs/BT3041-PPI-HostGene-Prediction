"""Control for the HuRI comparison: is the weak guilt-by-association (GBA) result on HuRI due to the
network being LITERATURE-FREE, or simply due to HuRI being much SPARSER?

On the genes present in both networks, the STRING v11.0 (>=700) edge set is randomly thinned to the
number of HuRI edges among those genes (R draws), and GBA is evaluated with identical labels and folds.
If thinned-STRING GBA stays far above HuRI GBA, sparsity does not explain the gap; if it falls to HuRI's
level, the HuRI result says little about literature bias.

Also reports, for both networks on the shared genes: edges, mean degree, share of positives with >=1
positive neighbour, and positive-positive edge enrichment over degree-matched random gene sets.
Output: Results/sparsity_control.csv.   Usage: python 09_sparsity_control.py [--R 30]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

import os
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# ============================================================== helpers
# Paths, settings and functions this step needs (written out here so the script runs on its own).

# --------------------------------------------------------------------------- paths
# Data/{raw,labels,processed} sit in the project root, one level above this step's folder.
PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HOSTNET_DATA", PROJECT / "Data"))
RAW, LABELS, PROC = DATA / "raw", DATA / "labels", DATA / "processed"

# This step writes its outputs to the Results/ folder next to this script (HOSTNET_RESULTS overrides it).
RESULTS = Path(os.environ.get("HOSTNET_RESULTS", Path(__file__).resolve().parent / "Results"))

# --------------------------------------------------------------------------- config
CFG = {
    "cells": ["Calu3", "A549-ACE2", "NHBE"],  # names as they appear in GEO sample names
    "main_cell": "Calu3",
    "min_basemean": 10,            # expressed-gene filter, applied to the main cell line
    "string_min_score": 700,       # STRING combined score threshold
    "string_exp_min_score": 400,   # STRING *experimental channel only* threshold
    "restart": 0.5,                # RWR restart probability
    "de_padj": 0.05,
    "de_lfc": 1.0,
    "betweenness_cutoff": None,    # e.g. 4 -> much faster, approximate betweenness
    "seed": 42,
    # PubMed IDs are assigned roughly chronologically. Entrez-dated 2019-12-29..31 records sit at
    # PMID ~31.885M, 2020-01-01..03 at ~31.893M (checked with E-utilities, 2026-11-24), so this keeps
    # papers indexed before 2020.
    "pmid_cutoff": 31_893_000,
}
URLS = {
    # GEO GSE147507 (Blanco-Melo et al. 2020, Cell) raw counts
    "counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147507/suppl/GSE147507_RawReadCounts_Human.tsv.gz",
    # STRING v11.0 (Jan 2019, pre-COVID -> no COVID text-mining leakage)
    "string_links": "https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz",
    "string_detailed": "https://stringdb-static.org/download/protein.links.detailed.v11.0/9606.protein.links.detailed.v11.0.txt.gz",
    "string_info": "https://stringdb-static.org/download/protein.info.v11.0/9606.protein.info.v11.0.txt.gz",
    # HuRI (Luck et al. 2020) systematic Y2H map, Ensembl gene IDs
    "huri": "https://interactome-atlas.org/data/HuRI.tsv",  # bare domain: the www. host has a certificate name mismatch
    # NCBI gene info (symbol harmonisation, Ensembl/Entrez maps) and publication counts
    "gene_info": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
    "gene2pubmed": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz",
}


def raw_path(key: str) -> Path:
    return RAW / URLS[key].rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- gene symbols
def symbol_maps():
    """Return (alias->official symbol, ENSG->symbol, Entrez->symbol), cached."""
    cache = PROC / "symbol_maps.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    gi = pd.read_csv(raw_path("gene_info"), sep="\t", dtype=str,
                     usecols=["GeneID", "Symbol", "Synonyms", "dbXrefs"])
    official = {s.upper(): s for s in gi.Symbol}
    syn, ambiguous = {}, set()
    for s, syns in zip(gi.Symbol, gi.Synonyms):
        if syns == "-" or pd.isna(syns):
            continue
        for a in syns.split("|"):
            a = a.upper()
            if a in official:
                continue
            if a in syn and syn[a] != s:
                ambiguous.add(a)
            syn[a] = s
    for a in ambiguous:
        syn.pop(a, None)
    alias = {**syn, **official}  # official symbols always win
    ens = {}
    for s, x in zip(gi.Symbol, gi.dbXrefs):
        if pd.isna(x):
            continue
        for t in x.split("|"):
            if t.startswith("Ensembl:"):
                ens[t[len("Ensembl:"):]] = s
    entrez = dict(zip(gi.GeneID, gi.Symbol))
    maps = (alias, ens, entrez)
    cache.write_bytes(pickle.dumps(maps))
    return maps


def harmonise(symbols, alias):
    return [alias.get(str(s).strip().upper()) for s in symbols]


def load_labels(name: str, alias) -> set:
    """Read data/labels/<name>.csv (a column called gene/symbol/PreyGene, else first column)."""
    path = LABELS / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - see README section 'Labels'.")
    df = pd.read_csv(path)
    col = next((c for c in df.columns
                if c.lower() in ("gene", "symbol", "gene_symbol", "preygene", "genename")),
               df.columns[0])
    raw = df[col].dropna().astype(str).str.strip()
    mapped = {g for g in harmonise(raw, alias) if g}
    print(f"  labels '{name}': {raw.nunique()} listed, {len(mapped)} mapped to official symbols")
    return mapped


def rwr(W, P0, restart: float, tol: float = 1e-9, maxit: int = 300):
    """Random walk with restart; P0 is n x k (one column per seed vector)."""
    P0 = np.asarray(P0, dtype=float)
    if P0.ndim == 1:
        P0 = P0[:, None]
    P0 = P0 / np.maximum(P0.sum(0, keepdims=True), 1e-12)
    P = P0.copy()
    for _ in range(maxit):
        Pn = (1 - restart) * (W @ P) + restart * P0
        if np.abs(Pn - P).max() < tol:
            return Pn
        P = Pn
    return P


# --------------------------------------------------------------------------- folds
def make_folds(y, groups, scheme: str, n_rep: int = 5, n_splits: int = 5, seed: int = 42):
    """Return int array [n_rep, n] of fold ids. scheme 'gene' = stratified gene-level;
    anything else = stratified GROUP k-fold on `groups` (whole groups stay together)."""
    n = len(y)
    out = np.zeros((n_rep, n), dtype=np.int8)
    for r in range(n_rep):
        if scheme == "gene":
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
            split = cv.split(np.zeros(n), y)
        else:
            cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r)
            split = cv.split(np.zeros(n), y, groups)
        for k, (_, te) in enumerate(split):
            out[r, te] = k
    return out


def gba_oof(W, y, folds, restart=None):
    """Guilt-by-association ranker: RWR seeded with the TRAINING positives only; the score of a test
    gene is its stationary visit probability. No test label ever seeds a test score. [n_rep, n]."""
    restart = CFG["restart"] if restart is None else restart
    out = np.zeros(folds.shape, dtype=np.float32)
    for r in range(folds.shape[0]):
        K = int(folds[r].max()) + 1
        P0 = np.zeros((len(y), K))
        for k in range(K):
            P0[(folds[r] != k) & (y == 1), k] = 1.0
        P = rwr(W, P0, restart)
        for k in range(K):
            te = folds[r] == k
            out[r, te] = P[te, k]
    return out


# --------------------------------------------------------------------------- weighted metrics
class ScoreSet:
    """Scores [n_rep, n] with sort orders cached so a weighted metric is O(n) per resample."""

    def __init__(self, scores, y):
        s = np.atleast_2d(np.asarray(scores, float))
        self.y = np.asarray(y, float)
        self.s = s
        self.order, self.ends = [], []
        for row in s:
            o = np.argsort(-row, kind="stable")
            srt = row[o]
            ends = np.r_[np.where(np.diff(srt) != 0)[0], len(srt) - 1]   # last index of each tie block
            self.order.append(o)
            self.ends.append(ends)

    def ap(self, w=None):
        """Average precision (step-wise, ties handled per threshold like sklearn); mean over repeats."""
        vals = []
        for o, e in zip(self.order, self.ends):
            yy = self.y[o]
            ww = np.ones_like(yy) if w is None else w[o]
            tp = np.cumsum(ww * yy)[e]
            fp = np.cumsum(ww * (1 - yy))[e]
            tot = tp[-1]
            if tot <= 0:
                vals.append(np.nan)
                continue
            prec = tp / np.maximum(tp + fp, 1e-12)
            dr = np.diff(tp, prepend=0.0) / tot
            vals.append(float((prec * dr).sum()))
        return float(np.nanmean(vals))

    def auroc(self, w=None):
        vals = []
        for o, e in zip(self.order, self.ends):
            yy = self.y[o]
            ww = np.ones_like(yy) if w is None else w[o]
            cp, cn = np.cumsum(ww * yy)[e], np.cumsum(ww * (1 - yy))[e]
            P_g, N_g = np.diff(cp, prepend=0.0), np.diff(cn, prepend=0.0)
            Pt, Nt = cp[-1], cn[-1]
            if Pt <= 0 or Nt <= 0:
                vals.append(np.nan)
                continue
            neg_after = Nt - cn
            vals.append(float((P_g * (neg_after + 0.5 * N_g)).sum() / (Pt * Nt)))
        return float(np.nanmean(vals))

    def p_at_k(self, k=100):
        v = [self.y[o[:k]].mean() for o in self.order]
        return float(np.mean(v))

    def ap_by_repeat(self):
        out = []
        for o, e in zip(self.order, self.ends):
            yy = self.y[o]
            tp, fp = np.cumsum(yy)[e], np.cumsum(1 - yy)[e]
            out.append(float(((tp / (tp + fp)) * np.diff(tp, prepend=0.0) / tp[-1]).sum()))
        return np.array(out)

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")


def W_from_edges(ei, ej, n):
    A = sp.coo_matrix((np.ones(len(ei)), (ei, ej)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return (A @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr(), A, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=30)
    ap.add_argument("--labels", default="gordon")
    args = ap.parse_args()
    seed = CFG["seed"]
    rng = np.random.default_rng(seed)
    alias, _, _ = symbol_maps()
    shared = pd.read_csv(PROC / "shared_huri_v11.csv").gene.tolist()
    idx = {g: i for i, g in enumerate(shared)}
    n = len(shared)
    y = np.array([g in load_labels(args.labels, alias) for g in shared], int)

    def edges(name):
        E = pd.read_csv(PROC / f"edges_{name}.csv")
        E = E[E.a.isin(idx) & E.b.isin(idx)]
        return np.array([idx[a] for a in E.a]), np.array([idx[b] for b in E.b])

    ev, hv = edges("string_v11"), edges("huri")
    G = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0).reindex(shared)
    grp = pd.factorize(G.complex_group.astype(object).where(G.complex_group.notna(), pd.Series([f"s_{x}" for x in shared], index=G.index)))[0]
    rows = []
    for scheme in ("gene", "complex_group"):
        folds = make_folds(y, grp, scheme, 5, 5, seed)
        res = {}
        for name, (i, j) in (("STRING v11.0 (all shared edges)", ev), ("HuRI", hv)):
            W, A, deg = W_from_edges(i, j, n)
            ap_ = ScoreSet(gba_oof(W, y, folds), y).ap()
            has_pos_nb = float(((A @ y) > 0)[y == 1].mean())
            rows.append(dict(scheme=scheme, network=name, edges=len(i), mean_degree=2 * len(i) / n, GBA_AUPRC=ap_, GBA_sd=np.nan,
                             frac_pos_with_pos_neighbour=has_pos_nb, prevalence=y.mean()))
        m = len(hv[0])
        vals, hasn = [], []
        for _ in range(args.R):
            keep = rng.choice(len(ev[0]), m, replace=False)
            W, A, _ = W_from_edges(ev[0][keep], ev[1][keep], n)
            vals.append(ScoreSet(gba_oof(W, y, folds), y).ap())
            hasn.append(float(((A @ y) > 0)[y == 1].mean()))
        rows.append(dict(scheme=scheme, network=f"STRING v11.0 randomly thinned to {m} edges (HuRI size)", edges=m,
                         mean_degree=2 * m / n, GBA_AUPRC=float(np.mean(vals)), GBA_sd=float(np.std(vals, ddof=1)),
                         frac_pos_with_pos_neighbour=float(np.mean(hasn)), prevalence=y.mean()))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "sparsity_control.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"shared genes {n}, positives {int(y.sum())}")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
