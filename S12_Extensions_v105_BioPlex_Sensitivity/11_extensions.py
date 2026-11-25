"""Extensions requested after the main analysis (all from raw/processed files; outputs prefixed ext_).

  A  STRING v10.5 (2017) anchor for the vintage curve: v10.5 / v11.0 / v11.5 / v12.0 at combined >= 700,
     evaluated on the genes with >= 1 edge in ALL four (identical genes, labels, folds).
  B  Assay circularity (Gordon = AP-MS in HEK293T; BioPlex = AP-MS in HEK293T and HCT116):
     B1 GBA on BioPlex-293T vs BioPlex-HCT116 (same assay, other cell line) on shared genes;
     B2 STRING v11 >= 700 with every edge also seen in BioPlex-293T removed vs the same number of RANDOM edges removed.
     If HEK293T AP-MS edges drive the signal, B1 favours 293T for Gordon (not CRISPR) and B2 hurts more than random.
  C  Sensitivity of the guilt-by-association (GBA) result, STRING v11:
     C1 RWR restart r in {0.15, 0.3, 0.5, 0.7, 0.85}; C2 combined-score cut-off {500, 600, 700, 800, 900}
     (on genes with >= 1 edge at 900, so all cut-offs share genes); C3 DE cut-offs for the RQ1 depletion test.
Metric: GBA AUPRC, complex-group 5x5 CV (seeds = training positives only); paired complex-cluster bootstrap.
Scope note: only the GBA ranker and a degree baseline are re-run (the topology arm N was flat in every earlier
network comparison and needs betweenness per network, which this time budget does not allow).
Usage: python 11_extensions.py [--B 500]
"""
import argparse
import gzip
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

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


# --------------------------------------------------------------------------- bootstrap
def boot_weights(n, codes, B, kind, rng):
    """[B, n] integer weight matrix. kind='gene': resample genes. kind='group': resample whole groups
    (codes = integer group id per gene); each gene inherits its group's multiplicity."""
    if kind == "gene":
        idx = rng.integers(0, n, size=(B, n))
        W = np.zeros((B, n), dtype=np.float32)
        for b in range(B):
            W[b] = np.bincount(idx[b], minlength=n)
        return W
    G = int(codes.max()) + 1
    cnt = np.zeros((B, G), dtype=np.float32)
    draw = rng.integers(0, G, size=(B, G))
    for b in range(B):
        cnt[b] = np.bincount(draw[b], minlength=G)
    return cnt[:, codes]


def summarize_diff(obs, boot, margin=0.01):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(boot, [5, 95])
    p = 2 * min((np.sum(boot <= 0) + 1) / (len(boot) + 1), (np.sum(boot >= 0) + 1) / (len(boot) + 1))
    return dict(diff=obs, ci_lo=lo, ci_hi=hi, p_boot=min(p, 1.0),
                equivalent_pm_margin=bool(lo90 > -margin and hi90 < margin),   # TOST at alpha=0.05
                boot_sd=float(boot.std(ddof=1)))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")
SV = RAW / "string_versions"
SEED = CFG["seed"]


def W_of(edges, genes):
    idx = {g: i for i, g in enumerate(genes)}
    e = edges[edges.a.isin(idx) & edges.b.isin(idx) & (edges.a != edges.b)]
    i, j = e.a.map(idx).values, e.b.map(idx).values
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(len(genes),) * 2)
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return (A @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr(), A, deg


def string_raw(links, mapping, cut):
    """Raw STRING links -> symbol edge list with score >= cut (undirected, deduplicated)."""
    L = pd.read_csv(links, sep=" ")
    L = L[L.combined_score >= cut]
    L["a"], L["b"] = L.protein1.map(mapping), L.protein2.map(mapping)
    L = L.dropna(subset=["a", "b"])
    L = L[L.a != L.b]
    L[["a", "b"]] = np.sort(L[["a", "b"]].values, axis=1)
    return L.groupby(["a", "b"], as_index=False).combined_score.max()


def map_v11(alias):
    info = pd.read_csv(RAW / "9606.protein.info.v11.0.txt.gz", sep="\t")
    return dict(zip(info.iloc[:, 0], info.preferred_name.map(lambda s: alias.get(s, alias.get(str(s).upper())))))


def map_v105(alias, universe):
    A = pd.read_csv(SV / "9606.protein.aliases.v10.5.txt.gz", sep="\t", comment=None, header=0,
                    names=["pid", "alias", "source"])
    A = A[A.source.str.contains("BioMart_HUGO|Ensembl_HGNC", na=False)]
    A["sym"] = A.alias.map(lambda s: alias.get(s))
    A = A[A.sym.isin(universe)]
    return A.groupby("pid").sym.agg(lambda s: s.value_counts().index[0]).to_dict()


class Evaluator:
    def __init__(self, genes, labels, alias, groups, B, rng):
        self.genes = list(genes)
        self.y = {}
        for lab in labels:
            pos = load_labels(lab, alias)
            self.y[lab] = np.array([g in pos for g in self.genes], int)
        cg = groups.complex_group.reindex(self.genes).astype(object)
        cg = cg.where(cg.notna(), pd.Series([f"s_{g}" for g in self.genes], index=self.genes))
        self.codes = pd.factorize(cg)[0]
        self.folds = {lab: make_folds(y, self.codes, "complex_group", 5, 5, SEED) for lab, y in self.y.items()}
        self.Wb = boot_weights(len(self.genes), self.codes, B, "group", rng)

    def gba(self, edges, lab, restart=None):
        W, A, deg = W_of(edges, self.genes)
        S = ScoreSet(gba_oof(W, self.y[lab], self.folds[lab], restart), self.y[lab])
        D = ScoreSet(np.tile(deg + 1e-9 * np.random.default_rng(0).random(len(deg)), (5, 1)), self.y[lab])
        return S, D, A, deg

    def boot(self, S):
        return S.ap(), np.array([S.ap(w) for w in self.Wb])


def pos_edges(A, deg, y, n_perm=300, seed=SEED):
    rng = np.random.default_rng(seed)
    P = np.where(y == 1)[0]
    obs = A[P][:, P].sum() / 2
    bins = pd.qcut(pd.Series(deg).rank(method="first"), 20, labels=False).values
    by = {b: np.where(bins == b)[0] for b in np.unique(bins)}
    null = np.array([A[(S := np.array([rng.choice(by[bins[v]]) for v in P]))][:, S].sum() / 2 for _ in range(n_perm)])
    return obs, null.mean(), obs / max(null.mean(), 1e-9), (1 + (null >= obs).sum()) / (n_perm + 1)


def paired_rows(res, pairs, tag, lab):
    out = []
    for a, b in pairs:
        r = summarize_diff(res[a][0] - res[b][0], res[a][1] - res[b][1])
        out.append(dict(part=tag, labels=lab, a=a, b=b, **r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    alias, _, _ = symbol_maps()
    universe = pd.read_csv(PROC / "features_string_v11.csv", index_col=0).index
    U = set(universe)
    groups = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0)
    labs = ["gordon", "crispr"]
    summ, comps = [], []

    # ---------------- A: vintage curve with a v10.5 anchor
    E105 = string_raw(SV / "9606.protein.links.v10.5.txt.gz", map_v105(alias, U), 700)
    nets = {"v10.5": E105[["a", "b"]], "v11.0": pd.read_csv(PROC / "edges_string_v11.csv"),
            "v11.5": pd.read_csv(PROC / "edges_string_v115.csv"), "v12.0": pd.read_csv(PROC / "edges_string_v120.csv")}
    nets = {k: v[v.a.isin(U) & v.b.isin(U)] for k, v in nets.items()}
    shared = sorted(set.intersection(*(set(v.a) | set(v.b) for v in nets.values())))
    print(f"[A] v10.5 edges >=700 in universe: {len(nets['v10.5'])}; genes with >=1 edge in all 4 versions: {len(shared)}  ({time.time()-t0:.0f}s)", flush=True)
    ev = Evaluator(shared, labs, alias, groups, args.B, rng)
    for lab in labs:
        res = {}
        for k, e in nets.items():
            S, D, A, deg = ev.gba(e, lab)
            res[k] = ev.boot(S)
            o, ex, fold, p = pos_edges(A, deg, ev.y[lab])
            summ.append(dict(part="A_vintage", labels=lab, network=k, n_genes=len(shared), n_pos=int(ev.y[lab].sum()),
                             edges=int(A.nnz / 2), prevalence=ev.y[lab].mean(), GBA_AUPRC=res[k][0],
                             GBA_ci_lo=np.percentile(res[k][1], 2.5), GBA_ci_hi=np.percentile(res[k][1], 97.5),
                             degree_AUPRC=D.ap(), pos_pos_edges=o, pos_pos_expected=ex, pos_pos_fold=fold, pos_pos_p=p))
        comps += paired_rows(res, [("v11.0", "v10.5"), ("v11.5", "v10.5"), ("v12.0", "v10.5"), ("v12.0", "v11.0")], "A_vintage", lab)
    print(f"[A] done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- B: assay circularity
    bp = {}
    for cell, fn in (("BioPlex-293T", "BioPlex_293T_Network_10K_Dec_2019.tsv"), ("BioPlex-HCT116", "BioPlex_HCT116_Network_5.5K_Dec_2019.tsv")):
        T = pd.read_csv(RAW / "bioplex" / fn, sep="\t")
        T = pd.DataFrame({"a": T.SymbolA.map(lambda s: alias.get(s)), "b": T.SymbolB.map(lambda s: alias.get(s))}).dropna()
        T = T[T.a.isin(U) & T.b.isin(U) & (T.a != T.b)]
        T[["a", "b"]] = np.sort(T[["a", "b"]].values, axis=1)
        bp[cell] = T.drop_duplicates()
    sharedB = sorted((set(bp["BioPlex-293T"].a) | set(bp["BioPlex-293T"].b)) & (set(bp["BioPlex-HCT116"].a) | set(bp["BioPlex-HCT116"].b)))
    print(f"[B1] BioPlex edges in universe: 293T {len(bp['BioPlex-293T'])}, HCT116 {len(bp['BioPlex-HCT116'])}; shared genes {len(sharedB)}", flush=True)
    evB = Evaluator(sharedB, labs, alias, groups, args.B, rng)
    for lab in labs:
        res = {}
        for k, e in bp.items():
            S, D, A, deg = evB.gba(e, lab)
            res[k] = evB.boot(S)
            o, ex, fold, p = pos_edges(A, deg, evB.y[lab])
            summ.append(dict(part="B1_bioplex_cell", labels=lab, network=k, n_genes=len(sharedB), n_pos=int(evB.y[lab].sum()),
                             edges=int(A.nnz / 2), prevalence=evB.y[lab].mean(), GBA_AUPRC=res[k][0],
                             GBA_ci_lo=np.percentile(res[k][1], 2.5), GBA_ci_hi=np.percentile(res[k][1], 97.5),
                             degree_AUPRC=D.ap(), pos_pos_edges=o, pos_pos_expected=ex, pos_pos_fold=fold, pos_pos_p=p))
        comps += paired_rows(res, [("BioPlex-293T", "BioPlex-HCT116")], "B1_bioplex_cell", lab)
    # B2: remove BioPlex-293T edges from STRING v11 vs remove the same number of random STRING edges
    S11 = nets["v11.0"].copy()
    S11[["a", "b"]] = np.sort(S11[["a", "b"]].values, axis=1)
    key = S11.a + "|" + S11.b
    inBP = key.isin(set(bp["BioPlex-293T"].a + "|" + bp["BioPlex-293T"].b))
    m = int(inBP.sum())
    evS = Evaluator(list(universe), labs, alias, groups, args.B, rng)
    print(f"[B2] STRING v11 edges also in BioPlex-293T: {m} of {len(S11)}", flush=True)
    for lab in labs:
        res = {"STRING v11 (full)": evS.boot(evS.gba(S11, lab)[0]), "minus BioPlex-293T edges": evS.boot(evS.gba(S11[~inBP], lab)[0])}
        rnd = []
        for r in range(5):
            drop = np.random.default_rng(SEED + r).choice(len(S11), m, replace=False)
            keep = np.ones(len(S11), bool); keep[drop] = False
            rnd.append(evS.boot(evS.gba(S11[keep], lab)[0]))
        res["minus random edges (mean of 5)"] = (np.mean([x[0] for x in rnd]), np.mean([x[1] for x in rnd], axis=0))
        for k, v in res.items():
            summ.append(dict(part="B2_remove_bioplex", labels=lab, network=k, n_genes=len(universe), n_pos=int(evS.y[lab].sum()),
                             edges=len(S11) - (m if k != "STRING v11 (full)" else 0), prevalence=evS.y[lab].mean(), GBA_AUPRC=v[0],
                             GBA_ci_lo=np.percentile(v[1], 2.5), GBA_ci_hi=np.percentile(v[1], 97.5),
                             GBA_sd_random=np.std([x[0] for x in rnd], ddof=1) if k.startswith("minus random") else np.nan))
        comps += paired_rows(res, [("minus BioPlex-293T edges", "STRING v11 (full)"), ("minus random edges (mean of 5)", "STRING v11 (full)"),
                                   ("minus BioPlex-293T edges", "minus random edges (mean of 5)")], "B2_remove_bioplex", lab)
    print(f"[B] done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- C1: restart; C2: cut-off
    for lab in labs:
        res = {}
        for r in (0.15, 0.3, 0.5, 0.7, 0.85):
            res[f"r={r}"] = evS.boot(evS.gba(S11, lab, restart=r)[0])
            summ.append(dict(part="C1_restart", labels=lab, network=f"r={r}", n_genes=len(universe), n_pos=int(evS.y[lab].sum()),
                             prevalence=evS.y[lab].mean(), GBA_AUPRC=res[f"r={r}"][0],
                             GBA_ci_lo=np.percentile(res[f"r={r}"][1], 2.5), GBA_ci_hi=np.percentile(res[f"r={r}"][1], 97.5)))
        comps += paired_rows(res, [(f"r={r}", "r=0.5") for r in (0.15, 0.3, 0.7, 0.85)], "C1_restart", lab)
    E11 = string_raw(RAW / "9606.protein.links.v11.0.txt.gz", map_v11(alias), 500)
    E11 = E11[E11.a.isin(U) & E11.b.isin(U)]
    g900 = sorted(set(E11[E11.combined_score >= 900].a) | set(E11[E11.combined_score >= 900].b))
    evC = Evaluator(g900, labs, alias, groups, args.B, rng)
    print(f"[C2] genes with >=1 edge at >=900: {len(g900)}", flush=True)
    for lab in labs:
        res = {}
        for cut in (500, 600, 700, 800, 900):
            e = E11[E11.combined_score >= cut]
            S, D, A, deg = evC.gba(e, lab)
            res[f"cut={cut}"] = evC.boot(S)
            summ.append(dict(part="C2_cutoff", labels=lab, network=f"cut={cut}", n_genes=len(g900), n_pos=int(evC.y[lab].sum()),
                             edges=int(A.nnz / 2), prevalence=evC.y[lab].mean(), GBA_AUPRC=res[f"cut={cut}"][0],
                             GBA_ci_lo=np.percentile(res[f"cut={cut}"][1], 2.5), GBA_ci_hi=np.percentile(res[f"cut={cut}"][1], 97.5),
                             degree_AUPRC=D.ap()))
        comps += paired_rows(res, [(f"cut={c}", "cut=700") for c in (500, 600, 800, 900)], "C2_cutoff", lab)
    print(f"[C1/C2] done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- C3: DE cut-offs for the RQ1 test (both tails, hypergeometric, universe background)
    de_rows = []
    for cell in ("Calu3", "A549-ACE2"):
        de = pd.read_csv(PROC / f"de_{cell}.csv", index_col=0).reindex(universe)
        for lab in labs:
            pos = load_labels(lab, alias) & U
            for padj in (0.01, 0.05, 0.1):
                for lfc in (0.0, 0.5, 1.0):
                    for direction in ("up", "down"):
                        sel = (de.padj < padj) & ((de.log2FoldChange > lfc) if direction == "up" else (de.log2FoldChange < -lfc))
                        n, K, N = int(sel.sum()), len(pos), len(universe)
                        k = int(sel[sel].index.isin(pos).sum())
                        de_rows.append(dict(cell=cell, labels=lab, padj=padj, lfc=lfc, direction=direction, n_de=n, k=k,
                                            expected=n * K / N, ratio=k / max(n * K / N, 1e-9),
                                            p_enrich=hypergeom.sf(k - 1, N, K, n), p_deplete=hypergeom.cdf(k, N, K, n)))
    DE = pd.DataFrame(de_rows)
    DE["q_deplete"] = bh(DE.p_deplete.values)
    DE["q_enrich"] = bh(DE.p_enrich.values)

    Sdf, Cdf = pd.DataFrame(summ), pd.DataFrame(comps)
    Cdf["q_BH"] = np.nan
    for _, idx in Cdf.groupby(["part", "labels"]).groups.items():
        Cdf.loc[idx, "q_BH"] = bh(Cdf.loc[idx, "p_boot"].values)
    Sdf.to_csv(RESULTS / "ext_summary.csv", index=False)
    Cdf.to_csv(RESULTS / "ext_paired.csv", index=False)
    DE.to_csv(RESULTS / "ext_de_sensitivity.csv", index=False)
    pd.set_option("display.width", 240); pd.set_option("display.max_rows", 300)
    print(Sdf.drop(columns=[c for c in Sdf.columns if c.startswith("GBA_sd")]).round(4).to_string(index=False))
    print(Cdf[["part", "labels", "a", "b", "diff", "ci_lo", "ci_hi", "p_boot", "q_BH"]].round(4).to_string(index=False))
    print(DE.round(4).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
