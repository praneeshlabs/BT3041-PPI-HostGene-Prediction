"""Step 4 - arms, baselines, repeated CV (gene-level AND group-aware), paired bootstrap.

For one network and one label set it evaluates
  arms (logistic regression 'LR' and random forest 'RF'):
      E, E_resp, E_abund, N, E+N, E+N_norm, E+nuis, E+N+nuis, nuis        (definitions: ARMS_EXT below)
  baselines (no model):  random, |lfc|, degree, pubs (date-frozen), pubs (live), abundance, GBA-RWR
under CV schemes
      gene            stratified 5x5 gene-level folds                           (the guide's scheme)
      complex_group   stratified 5x5 GROUP folds, groups = CORUM-based clusters  (primary: complexes stay together)
      community_fine  stratified 5x5 GROUP folds, groups = fine Leiden modules    (strictest)
and writes  Results/models/{summary,comparisons}__<tag>.csv  and  oof__<tag>.npz.

Primary estimand (pre-specified in plan.md / guide Part 11): AUPRC(E+N) - AUPRC(E), LR, complex_group CV.
H3 nested test: AUPRC(E+N+nuis) - AUPRC(E+nuis), where nuis = abundance + date-frozen publications + degree.
Equivalence margin delta = 0.01 AUPRC (= 36% of the Gordon prevalence 0.028; chosen AFTER seeing the
first pilot gain of +0.001, so it is a post-hoc margin and is labelled as such in the report).
Usage: python 04_models.py --ppi string_v11 --labels gordon [--B 1000] [--subset genes.csv --tag name]
"""
import argparse
import time
import warnings

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


def paired_boot(a: ScoreSet, b: ScoreSet, W, metric="ap"):
    """Distribution of metric(a) - metric(b) over bootstrap weight rows (same resample for both)."""
    f = (lambda s, w: s.ap(w)) if metric == "ap" else (lambda s, w: s.auroc(w))
    return np.array([f(a, w) - f(b, w) for w in W])


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
OUT = RESULTS / "models"
OUT.mkdir(parents=True, exist_ok=True)

ARMS_LR = ["E", "E_resp", "E_abund", "N", "E+N", "E+N_norm", "E+nuis", "E+N+nuis", "nuis"]
ARMS_RF = ["E", "N", "E+N", "E+nuis", "E+N+nuis"]

# (id, A, B, question)   keys: "<model>:<arm>" or "base:<name>" or "gba"
def comparison_list(model):
    m = model
    return [
        ("E+N - E", f"{m}:E+N", f"{m}:E", "H3 (unnested): does the network add to expression?"),
        ("E+N+nuis - E+nuis", f"{m}:E+N+nuis", f"{m}:E+nuis", "H3 (nested): network beyond response, abundance, fame, hubness"),
        ("E+N_norm - E", f"{m}:E+N_norm", f"{m}:E", "same as E+N-E with degree-normalised propagation"),
        ("N - E", f"{m}:N", f"{m}:E", "H2: network alone vs expression alone"),
        ("N - pubs_pre2020", f"{m}:N", "base:log_pubs_pre2020", "H0': network vs date-frozen fame baseline"),
        ("N - pubs_live", f"{m}:N", "base:log_pubs_live", "same, with the live (time-contaminated) count"),
        ("N - degree", f"{m}:N", "base:log_degree", "network beyond hub-ness"),
        ("N - nuis", f"{m}:N", f"{m}:nuis", "network vs nuisance-only model"),
        ("E+N - pubs_pre2020", f"{m}:E+N", "base:log_pubs_pre2020", "combined vs fame"),
        ("E - |lfc|", f"{m}:E", "base:abs_lfc", "does the E arm beat 'just take DE genes'?"),
        ("E_resp - random", f"{m}:E_resp", "base:random", "infection response alone vs chance"),
        ("E_abund - E_resp", f"{m}:E_abund", f"{m}:E_resp", "abundance vs response"),
        ("E - E_abund", f"{m}:E", f"{m}:E_abund", "does response add to abundance?"),
        ("GBA - N", "gba", f"{m}:N", "label-propagation vs topology features"),
        ("GBA - pubs_pre2020", "gba", "base:log_pubs_pre2020", "label-propagation vs fame"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi", default="string_v11")
    ap.add_argument("--labels", default="gordon")
    ap.add_argument("--schemes", nargs="+", default=["gene", "complex_group", "community_fine"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no_rf", action="store_true")
    ap.add_argument("--subset", default=None, help="csv with column gene: evaluate on these genes only")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--restart", type=float, default=None, help="RWR restart probability for GBA (sensitivity sweep; default CFG[restart])")
    args = ap.parse_args()
    if args.restart is not None:
        CFG["restart"] = args.restart
    cell, seed = CFG["main_cell"], CFG["seed"]
    tag = args.tag or f"{args.ppi}__{args.labels}"
    rng = np.random.default_rng(seed)

    alias, _, _ = symbol_maps()
    F, g = load_network(args.ppi)
    _, W, _ = graph_matrices(g)
    pos = load_labels(args.labels, alias)
    keep = np.ones(len(F), bool)
    if args.subset:
        keep = F.index.isin(pd.read_csv(args.subset).gene)
    # CANONICAL GENE ORDER (alphabetical). Different networks list their genes in different orders; folds
    # are generated from the row order, so without this two networks evaluated on the same genes would get
    # DIFFERENT splits and cross-network comparisons would not be paired on identical folds.
    sel = np.where(keep)[0]
    sel = sel[np.argsort(F.index.values[sel], kind="stable")]
    genes = F.index[sel]
    y = F.index.isin(pos)[sel].astype(int)
    n, npos = len(genes), int(y.sum())
    print(f"[{tag}] genes={n} positives={npos} prevalence={npos / n:.4f}")

    G = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0).reindex(genes)
    for c in G.columns:
        G[c] = G[c].astype(object).where(G[c].notna(), pd.Series([f"single_{x}" for x in genes], index=genes))
    codes = {c: pd.factorize(G[c])[0] for c in G.columns}
    Wsub_idx = sel

    base_scores = {
        "random": None,
        "abs_lfc": F[f"abs_lfc__{cell}"].values[sel],
        "log_degree": F.log_degree.values[sel],
        "log_pubs_pre2020": F.log_pubs_pre2020.values[sel],
        "log_pubs_live": F.log_pubs.values[sel],
        "logbm": F[f"logbm__{cell}"].values[sel],
    }
    summary, comps = [], []
    oof_store = {}
    for scheme in args.schemes:
        t0 = time.time()
        grp = codes["complex_group"] if scheme == "gene" else codes[scheme]
        folds = make_folds(y, grp, scheme, args.repeats, 5, seed)
        S = {}
        for model, arms in (("LR", ARMS_LR), ("RF", [] if args.no_rf else ARMS_RF)):
            for arm in arms:
                X = build_X(F, arm, cell).values[sel]
                S[f"{model}:{arm}"] = ScoreSet(oof_scores(model, X, y, folds, seed), y)
        for name, sc in base_scores.items():
            if name == "random":
                S["base:random"] = ScoreSet(rng.random((args.repeats, n)), y)
            else:
                S[f"base:{name}"] = baseline_set(sc, y, args.repeats)
        # GBA ranker on the FULL graph, seeded only with training positives that are in the evaluation set
        yfull = np.zeros(len(F), int)
        yfull[Wsub_idx] = y
        fold_full = np.full((args.repeats, len(F)), -1, dtype=np.int8)
        fold_full[:, Wsub_idx] = folds
        # genes outside the subset get a fold id that never seeds: handled by giving them fold 0 but y=0
        fold_full[fold_full < 0] = 0
        gb = gba_oof(W, yfull, fold_full)[:, Wsub_idx]
        S["gba"] = ScoreSet(gb, y)
        for k, v in S.items():
            oof_store[f"{scheme}|{k}"] = v.s.astype(np.float32)
        print(f"  scheme {scheme}: fitted {len(S)} score sets in {time.time() - t0:.0f}s")

        # ---- bootstrap: genes and groups (complex clusters as the unit of independence)
        Wg = boot_weights(n, None, args.B, "gene", rng)
        Wc = boot_weights(n, codes["complex_group"], args.B, "group", rng)
        boots = {kind: {k: np.array([v.ap(w) for w in Wk]) for k, v in S.items()}
                 for kind, Wk in (("gene", Wg), ("group", Wc))}
        for k, v in S.items():
            model, _, arm = k.partition(":")
            lo, hi = np.percentile(boots["group"][k], [2.5, 97.5])
            lo_g, hi_g = np.percentile(boots["gene"][k], [2.5, 97.5])
            summary.append(dict(ppi=args.ppi, labels=args.labels, subset=args.subset is not None, scheme=scheme,
                                name=k, n=n, n_pos=npos, prevalence=npos / n, AUPRC=v.ap(),
                                AUPRC_sd_over_repeats=float(v.ap_by_repeat().std(ddof=1)) if len(v.order) > 1 else np.nan,
                                ci_group_lo=lo, ci_group_hi=hi, ci_gene_lo=lo_g, ci_gene_hi=hi_g,
                                AUROC=v.auroc(), lift=v.ap() / (npos / n), P_at_100=v.p_at_k(100)))
        for model in ("LR", "RF"):
            if args.no_rf and model == "RF":
                continue
            for cid, a, b, q in comparison_list(model):
                if a not in S or b not in S:
                    continue
                obs = S[a].ap() - S[b].ap()
                for kind in ("gene", "group"):
                    d = boots[kind][a] - boots[kind][b]
                    r = summarize_diff(obs, d)
                    comps.append(dict(ppi=args.ppi, labels=args.labels, subset=args.subset is not None, scheme=scheme,
                                      model=model, comparison=cid, bootstrap_unit=kind, question=q,
                                      mde80=2.8 * r["boot_sd"], **r))
        print(f"  scheme {scheme}: done in {time.time() - t0:.0f}s")

    Sdf = pd.DataFrame(summary)
    Cdf = pd.DataFrame(comps)
    Cdf["q_BH"] = np.nan
    for _, idx in Cdf.groupby(["scheme", "model", "bootstrap_unit"]).groups.items():
        Cdf.loc[idx, "q_BH"] = bh(Cdf.loc[idx, "p_boot"].values)     # family = the comparisons within one cell
    Cdf["primary"] = ((Cdf.comparison == "E+N - E") & (Cdf.model == "LR") & (Cdf.scheme == "complex_group")
                      & (Cdf.bootstrap_unit == "group"))
    Sdf.to_csv(OUT / f"summary__{tag}.csv", index=False)
    Cdf.to_csv(OUT / f"comparisons__{tag}.csv", index=False)
    np.savez_compressed(OUT / f"oof__{tag}.npz", genes=np.array(genes), y=y, **oof_store)
    pd.set_option("display.width", 220)
    show = Sdf[Sdf.name.str.startswith(("LR:", "base:", "gba"))].pivot_table(index="name", columns="scheme", values="AUPRC")
    print("\nAUPRC (prevalence %.4f):\n" % (npos / n), show.round(4).to_string())
    print("\nKey comparisons (group bootstrap, LR):")
    key = Cdf[(Cdf.model == "LR") & (Cdf.bootstrap_unit == "group")]
    print(key[["scheme", "comparison", "diff", "ci_lo", "ci_hi", "q_BH", "equivalent_pm_margin"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
