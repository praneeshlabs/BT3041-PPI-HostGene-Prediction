"""Step S7 - transfer tests.

A. CELL-LINE TRANSFER (Calu-3 -> A549-ACE2). Genes are held out as well as the features swapped
   (training genes use Calu-3 features; test genes use A549-ACE2 features), and the comparison is
   restricted to genes that are actually TESTED in both lines (baseMean >= 10 in both) - the original
   pipeline filled the other line's missing fold changes with 0, which is informative missingness.
   Reported on the common gene set: Calu3->Calu3, Calu3->A549, A549->A549.
   NHBE is dropped (only 89 DE genes; the ranking is noise).

B. LABEL TRANSFER (Gordon -> CRISPR). Models are trained on Gordon labels (genes held out by CV) and
   scored against the CRISPR labels on the held-out genes, EXCLUDING genes that are Gordon positives
   (otherwise the 16 shared genes would be counted as transfer). Compared with a model trained on the
   CRISPR labels directly (same evaluation genes) and with baselines.

Outputs: Results/transfer.csv.  Usage: python 07_transfer.py [--B 1000]
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import os
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


# --------------------------------------------------------------------------- graph helpers
def load_network(ppi: str):
    """Return (feature table F indexed by gene, igraph Graph with the same vertex order)."""
    import igraph as ig
    F = pd.read_csv(PROC / f"features_{ppi}.csv", index_col=0)
    E = pd.read_csv(PROC / f"edges_{ppi}.csv")
    idx = {g: i for i, g in enumerate(F.index)}
    g = ig.Graph(n=len(F), edges=[(idx[a], idx[b]) for a, b in zip(E.a, E.b)], directed=False)
    g.vs["name"] = list(F.index)
    return F, g


TOPO = ["log_degree", "log_betweenness", "closeness", "pagerank", "coreness", "clustering"]

# --------------------------------------------------------------------------- the three arms
# Expression features are cell-line specific (columns stored as <name>__<cell>).
EXPR = ["lfc", "abs_lfc", "nlp", "logbm"]
EXPR_RESP = ["lfc", "abs_lfc", "nlp"]          # infection response only
EXPR_ABUND = ["logbm"]                          # abundance / detectability only

# Pure topology: computed from the graph alone, contains no expression and no labels.
# Propagated |logFC| ("prop") mixes expression WITH the network, so it is put only in the
# combined arm - otherwise the network-only arm would secretly contain expression.
ARMS = {
    "E": EXPR,                      # expression only
    "N": TOPO,                      # network only
    "E+N": EXPR + TOPO + ["prop"],  # expression + network (+ network-propagated expression)
}
CELL_SPECIFIC = set(EXPR + ["prop", "propn"])

# Extended arms for Step 4 (the three headline arms above are unchanged). "nuisance" = what a
# network has to beat to count: abundance (logbm, already in E), fame (log_pubs_pre2020), hubness.
ARMS_EXT = {
    **ARMS,
    "E_resp": EXPR_RESP,
    "E_abund": EXPR_ABUND,
    "E+N_norm": EXPR + TOPO + ["propn"],                       # sensitivity: degree-normalised propagation
    "E+nuis": EXPR + ["log_pubs_pre2020", "log_degree"],       # expression + fame + hubness
    "E+N+nuis": EXPR + TOPO + ["prop", "log_pubs_pre2020"],    # ... + the rest of the network
    "nuis": ["logbm", "log_pubs_pre2020", "log_degree"],       # nuisance-only baseline model
}


def build_X(F: pd.DataFrame, arm: str, cell: str) -> pd.DataFrame:
    """Feature matrix for one arm, using the expression columns of the given cell line."""
    cols = {c: (F[f"{c}__{cell}"] if c in CELL_SPECIFIC else F[c]) for c in ARMS_EXT[arm]}
    return pd.DataFrame(cols, index=F.index).astype(float)


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


# --------------------------------------------------------------------------- models
def make_model(name: str, seed: int):
    if name == "LR":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    raise ValueError(name)


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


def baseline_set(score, y, n_rep):
    """A repeat-independent ranker (degree, pubs, |lfc| ...) as a ScoreSet."""
    return ScoreSet(np.tile(np.asarray(score, float), (n_rep, 1)), y)


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

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")


def cross_oof(Xtr, Xte, y, folds, seed):
    """Fit on Xtr[train fold], predict Xte[test fold] (different feature matrices, same genes held out)."""
    out = np.zeros(folds.shape, dtype=np.float32)
    for r in range(folds.shape[0]):
        for k in np.unique(folds[r]):
            te = folds[r] == k
            m = make_model("LR", seed + r).fit(Xtr[~te], y[~te])
            out[r, te] = m.predict_proba(Xte[te])[:, 1]
    return out


def ci(S, W):
    v = np.array([S.ap(w) for w in W])
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    seed, rng = CFG["seed"], np.random.default_rng(CFG["seed"])
    alias, _, _ = symbol_maps()
    F, _ = load_network("string_v11")
    G = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0).reindex(F.index)
    grp = G.complex_group.astype(object).where(G.complex_group.notna(), pd.Series([f"single_{x}" for x in F.index], index=F.index))
    gor, cri = (F.index.isin(load_labels(n, alias)).astype(int) for n in ("gordon", "crispr"))
    rows = []

    # ---------------- A. cell-line transfer
    a549 = pd.read_csv(PROC / "de_A549-ACE2.csv", index_col=0).reindex(F.index)
    T = (a549.baseMean >= CFG["min_basemean"]).values & np.isfinite(a549.pvalue.values)
    yT, codes = gor[T], pd.factorize(grp[T])[0]
    print(f"A. genes tested in both lines: {T.sum()} of {len(F)}; Gordon positives {yT.sum()} (prev {yT.mean():.4f})")
    for scheme in ("gene", "complex_group"):
        folds = make_folds(yT, codes, scheme, 5, 5, seed)
        W = boot_weights(T.sum(), codes, args.B, "group", rng)
        for arm in ("E", "E+N"):
            Xc, Xa = (build_X(F, arm, c).values[T] for c in ("Calu3", "A549-ACE2"))
            for name, Xtr, Xte in (("Calu3->Calu3", Xc, Xc), ("Calu3->A549-ACE2", Xc, Xa), ("A549-ACE2->A549-ACE2", Xa, Xa)):
                S = ScoreSet(cross_oof(Xtr, Xte, yT, folds, seed), yT)
                lo, hi = ci(S, W)
                rows.append(dict(part="A cell-line", scheme=scheme, arm=arm, setting=name, n=int(T.sum()), n_pos=int(yT.sum()),
                                 prevalence=yT.mean(), AUPRC=S.ap(), ci_lo=lo, ci_hi=hi, lift=S.ap() / yT.mean(), AUROC=S.auroc()))
        for nm, sc in (("pubs_pre2020", F.log_pubs_pre2020.values[T]), ("degree", F.log_degree.values[T])):
            S = baseline_set(sc, yT, 5)
            rows.append(dict(part="A cell-line", scheme=scheme, arm="baseline", setting=nm, n=int(T.sum()), n_pos=int(yT.sum()),
                             prevalence=yT.mean(), AUPRC=S.ap(), lift=S.ap() / yT.mean(), AUROC=S.auroc()))

    # ---------------- B. Gordon -> CRISPR
    ev = gor == 0                      # evaluation genes: Gordon positives removed
    yE = cri[ev]
    print(f"B. evaluation genes {ev.sum()}, CRISPR positives among them {yE.sum()} (prev {yE.mean():.4f}); "
          f"CRISPR positives removed because they are Gordon positives: {int(((cri == 1) & (gor == 1)).sum())}")
    codesE = pd.factorize(grp[ev])[0]
    for scheme in ("gene", "complex_group"):
        fG = make_folds(gor, pd.factorize(grp)[0], scheme, 5, 5, seed)
        fC = make_folds(cri, pd.factorize(grp)[0], scheme, 5, 5, seed)
        W = boot_weights(int(ev.sum()), codesE, args.B, "group", rng)
        for arm in ("E", "N", "E+N", "nuis"):
            X = build_X(F, arm if arm != "nuis" else "nuis", "Calu3").values
            for name, y_train, folds in (("train Gordon -> eval CRISPR", gor, fG), ("train CRISPR -> eval CRISPR", cri, fC)):
                sc = cross_oof(X, X, y_train, folds, seed)[:, ev]
                S = ScoreSet(sc, yE)
                lo, hi = ci(S, W)
                rows.append(dict(part="B label", scheme=scheme, arm=arm, setting=name, n=int(ev.sum()), n_pos=int(yE.sum()),
                                 prevalence=yE.mean(), AUPRC=S.ap(), ci_lo=lo, ci_hi=hi, lift=S.ap() / yE.mean(), AUROC=S.auroc()))
        for nm, sc in (("pubs_pre2020", F.log_pubs_pre2020.values), ("degree", F.log_degree.values), ("abundance", F["logbm__Calu3"].values)):
            S = baseline_set(sc[ev], yE, 5)
            rows.append(dict(part="B label", scheme=scheme, arm="baseline", setting=nm, n=int(ev.sum()), n_pos=int(yE.sum()),
                             prevalence=yE.mean(), AUPRC=S.ap(), lift=S.ap() / yE.mean(), AUROC=S.auroc()))
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS / "transfer.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.drop(columns=["n", "n_pos", "prevalence"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
