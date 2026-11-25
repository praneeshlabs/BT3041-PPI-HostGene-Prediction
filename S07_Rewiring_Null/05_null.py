"""Step 5 - degree-preserving rewiring null (Maslov-Sneppen) for H4.

Question: is the network's predictive signal due to its specific wiring, or only to its degree sequence?
Each rewiring keeps every gene's degree EXACTLY (asserted) and destroys all specific wiring. For each of
R rewired graphs we recompute ALL topology features and the propagated |lfc| on the rewired graph,
refit the logistic-regression arms on the SAME folds as the real analysis, and record

    AUPRC(N), AUPRC(E+N), gain = AUPRC(E+N) - AUPRC(E)   (E is untouched by rewiring)
    AUPRC(GBA)  guilt-by-association ranker seeded with training positives

Empirical p (one-sided, "real network better than degree-preserving random") = (1 + #null >= obs) / (R + 1).
With R=50 the smallest possible p is 0.0196; R >= 200 gives 0.005.
Usage: python 05_null.py --ppi string_v11 --labels gordon --R 200 --procs 10
"""
import argparse
import random
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import os
import pickle
import scipy.sparse as sp
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


def graph_matrices(g):
    n = g.vcount()
    el = np.array(g.get_edgelist())
    A = sp.coo_matrix((np.ones(len(el)), (el[:, 0], el[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    W = (A @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()  # column-stochastic
    return A, W, deg


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


TOPO = ["log_degree", "log_betweenness", "closeness", "pagerank", "coreness", "clustering"]


def topo_features(g) -> pd.DataFrame:
    n = g.vcount()
    deg = np.array(g.degree(), dtype=float)
    bet = np.array(g.betweenness(directed=False, cutoff=CFG["betweenness_cutoff"]), dtype=float)
    bet = bet / max((n - 1) * (n - 2) / 2, 1)
    return pd.DataFrame({
        "degree": deg,
        "log_degree": np.log1p(deg),
        "log_betweenness": np.log10(bet + 1e-9),
        "closeness": np.nan_to_num(np.array(g.closeness(), dtype=float)),
        "pagerank": np.array(g.pagerank(), dtype=float) * n,
        "coreness": np.array(g.coreness(), dtype=float),
        "clustering": np.nan_to_num(np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)),
    }, index=g.vs["name"])


def prop_features(W, F: pd.DataFrame, cells) -> pd.DataFrame:
    """Network-propagated |logFC| per cell line (label-free, so no leakage)."""
    P0 = np.column_stack([F[f"abs_lfc__{c}"].fillna(0).values for c in cells])
    P = rwr(W, P0, CFG["restart"]) * len(F)
    # Raw RWR scores are partly a degree proxy (the walk piles up on hubs even when the seeds carry
    # no topology information; Spearman ~0.54 with degree here). The ratio to the diffusion of a
    # UNIFORM seed divides out what topology alone would deliver -> degree-normalised propagation.
    U = rwr(W, np.ones((len(F), 1)), CFG["restart"]) * len(F)
    out = pd.DataFrame(P, index=F.index, columns=[f"prop__{c}" for c in cells])
    for j, c in enumerate(cells):
        out[f"propn__{c}"] = P[:, j] / np.maximum(U[:, 0], 1e-12)
    return out


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


def oof_scores(model_name, X, y, folds, seed=42):
    """Out-of-fold probability for every gene, for each repeat -> [n_rep, n]."""
    X = np.asarray(X, float)
    out = np.zeros(folds.shape, dtype=np.float32)
    for r in range(folds.shape[0]):
        for k in np.unique(folds[r]):
            te = folds[r] == k
            m = make_model(model_name, seed + r).fit(X[~te], y[~te])
            out[r, te] = m.predict_proba(X[te])[:, 1]
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
G = {}


def init(ppi, labels, schemes, seed):
    alias, _, _ = symbol_maps()
    F, g = load_network(ppi)
    pos = load_labels(labels, alias)
    y = F.index.isin(pos).astype(int)
    grp = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0).reindex(F.index)["complex_group"]
    grp = grp.astype(object).where(grp.notna(), pd.Series([f"single_{x}" for x in F.index], index=F.index))
    codes = pd.factorize(grp)[0]
    folds = {s: make_folds(y, codes, s, 5, 5, seed) for s in schemes}
    G.update(F=F, g=g, y=y, folds=folds, schemes=schemes, seed=seed, ppi=ppi)


def evaluate(g, F, y, folds, schemes, seed, cell):
    """LR arms N and E+N + GBA on graph g (real or rewired); E is returned for the gain."""
    _, W, _ = graph_matrices(g)
    topo = topo_features(g)
    Fr = F.copy()
    for c in TOPO:
        Fr[c] = topo[c].values
    Fr[f"prop__{cell}"] = prop_features(W, Fr, [cell])[f"prop__{cell}"].values
    out = {}
    XN, XEN, XE = (build_X(Fr, a, cell).values for a in ("N", "E+N", "E"))
    for s in schemes:
        out[f"N|{s}"] = ScoreSet(oof_scores("LR", XN, y, folds[s], seed), y).ap()
        out[f"E+N|{s}"] = ScoreSet(oof_scores("LR", XEN, y, folds[s], seed), y).ap()
        out[f"GBA|{s}"] = ScoreSet(gba_oof(W, y, folds[s]), y).ap()
        if "E_real" not in G:   # E has no network features: same on every graph, computed once per worker
            pass
        out[f"E|{s}"] = ScoreSet(oof_scores("LR", XE, y, folds[s], seed), y).ap()
        out[f"gain|{s}"] = out[f"E+N|{s}"] - out[f"E|{s}"]
    return out


def one_rewire(i):
    import igraph as ig
    seed = G["seed"] + 1000 + i
    rnd = random.Random(seed)
    ig.set_random_number_generator(rnd)
    g2 = G["g"].copy()
    g2.rewire(n=10 * g2.ecount(), mode="simple")
    assert sorted(g2.degree()) == sorted(G["g"].degree()) and g2.degree() == G["g"].degree(), "degree sequence changed"
    t0 = time.time()
    r = evaluate(g2, G["F"], G["y"], G["folds"], G["schemes"], G["seed"], CFG["main_cell"])
    r.update(rewiring=i, seconds=time.time() - t0,
             edges_shared_with_real=len({tuple(sorted(e)) for e in g2.get_edgelist()} & {tuple(sorted(e)) for e in G["g"].get_edgelist()}) / g2.ecount())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi", default="string_v11")
    ap.add_argument("--labels", default="gordon")
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--schemes", nargs="+", default=["gene", "complex_group"])
    args = ap.parse_args()
    seed = CFG["seed"]
    init(args.ppi, args.labels, args.schemes, seed)
    cell = CFG["main_cell"]
    t0 = time.time()
    obs = evaluate(G["g"], G["F"], G["y"], G["folds"], args.schemes, seed, cell)
    print(f"observed values ({time.time() - t0:.0f}s):", {k: round(v, 4) for k, v in obs.items()})
    with Pool(args.procs, initializer=init, initargs=(args.ppi, args.labels, args.schemes, seed)) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(one_rewire, range(args.R))):
            rows.append(r)
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{args.R} rewirings done, {time.time() - t0:.0f}s elapsed, last {r['seconds']:.0f}s each", flush=True)
                pd.DataFrame(rows).to_csv(RESULTS / f"null_rewiring_raw__{args.ppi}__{args.labels}.csv", index=False)
    N = pd.DataFrame(rows).sort_values("rewiring")
    N.to_csv(RESULTS / f"null_rewiring_raw__{args.ppi}__{args.labels}.csv", index=False)
    summ = []
    for key, o in obs.items():
        if key.startswith("E|"):
            continue
        v = N[key].values
        summ.append(dict(ppi=args.ppi, labels=args.labels, quantity=key, observed=o, null_mean=v.mean(),
                         null_sd=v.std(ddof=1), z=(o - v.mean()) / v.std(ddof=1), R=len(v),
                         p_emp_one_sided=(1 + (v >= o).sum()) / (len(v) + 1), min_possible_p=1 / (len(v) + 1)))
    S = pd.DataFrame(summ)
    S.to_csv(RESULTS / f"null_summary__{args.ppi}__{args.labels}.csv", index=False)
    print(S.round(5).to_string(index=False))
    print("mean fraction of edges shared between a rewired graph and the real graph:", round(N.edges_shared_with_real.mean(), 4))


if __name__ == "__main__":
    main()
