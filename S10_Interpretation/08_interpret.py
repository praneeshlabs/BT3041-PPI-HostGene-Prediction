"""Step S8 - interpretation, all computed from the out-of-fold scores of 04_models.py.

  1. Confound-matched benchmark: stratified AUROC within bins of (abundance x date-frozen fame). If an arm
     carries signal beyond "abundant, famous genes are more often labelled", it must discriminate positives
     from negatives INSIDE bins.
  2. Publication terciles: lift per tercile of date-frozen publication count (does it work for understudied genes?).
  3. P@k and calibration of the combined model.
  4. Top predictions: unlabelled genes ranked by out-of-fold score, flagged by how many labelled neighbours /
     co-complex members they have; the FILTERED list keeps only genes with none (i.e. not mere complex completion).
  5. Over-representation analysis of the top-100 against the UNIVERSE background (not the genome).
  6. Housekeeping test of the H1 story ("binders are stable housekeeping machinery"), adjusted for abundance/fame/degree.
  7. Standardised logistic coefficients of E+N and a TreeSHAP summary of the forest.

Usage: python 08_interpret.py [--tag string_v11__gordon] [--scheme complex_group]
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import os
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ============================================================== helpers
# Paths, settings and functions this step needs (written out here so the script runs on its own).

# --------------------------------------------------------------------------- paths
# Data/{raw,labels,processed} sit in the project root, one level above this step's folder.
PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HOSTNET_DATA", PROJECT / "Data"))
RAW, LABELS, PROC = DATA / "raw", DATA / "labels", DATA / "processed"
STEP_DIRS = {
    "download": "S01_Data_Download",
    "labels": "S02_Label_Sets",
    "de": "S03_Differential_Expression",
    "network": "S04_Network_Features",
    "groups": "S05_Communities_And_CV_Groups",
    "models": "S06_Prediction_Models",
    "null": "S07_Rewiring_Null",
    "panel": "S08_STRING_Versions_And_Channels",
    "transfer": "S09_Transfer",
    "interpret": "S10_Interpretation",
    "sparsity": "S11_HuRI_Sparsity_Control",
    "extensions": "S12_Extensions_v105_BioPlex_Sensitivity",
    "figures": "S13_Report_Figures",
}


def step_dir(key: str) -> Path:
    return PROJECT / STEP_DIRS[key]


def step_results(key: str) -> Path:
    """Results folder of another step (for scripts that read an earlier step's outputs)."""
    return step_dir(key) / "Results"


# This step writes its outputs to the Results/ folder next to this script (HOSTNET_RESULTS overrides it).
RESULTS = Path(os.environ.get("HOSTNET_RESULTS", Path(__file__).resolve().parent / "Results"))
FIG = RESULTS / "figures"

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


def savefig(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)   # created only by steps that draw figures
    out = FIG / name
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  saved figure {out.with_suffix('.png').name}")


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
ARMS = ["E", "N", "E+N", "E+nuis", "E+N+nuis", "nuis"]


def strat_auc(sets, ybin, w=None):
    """Mantel-Haenszel-style stratified AUROC: bin AUROCs weighted by (#pos x #neg)."""
    num = den = 0.0
    for S, idx in sets:
        ww = None if w is None else w[idx]
        yy = ybin[idx]
        P = yy.sum() if ww is None else (ww * yy).sum()
        Nn = (1 - yy).sum() if ww is None else (ww * (1 - yy)).sum()
        if P <= 0 or Nn <= 0:
            continue
        num += S.auroc(ww) * P * Nn
        den += P * Nn
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="string_v11__gordon")
    ap.add_argument("--scheme", default="complex_group")
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()
    cell, seed = CFG["main_cell"], CFG["seed"]
    rng = np.random.default_rng(seed)
    alias, _, _ = symbol_maps()
    z = np.load(step_results("models") / "models" / f"oof__{args.tag}.npz", allow_pickle=True)
    genes, y = z["genes"], z["y"]
    F, g = load_network("string_v11")
    F = F.loc[genes]
    lab = args.tag.split("__")[-1]
    G = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0).reindex(genes)
    codes = pd.factorize(G.complex_group.astype(object).where(G.complex_group.notna(), pd.Series([f"single_{x}" for x in genes], index=genes)))[0]
    S = {a: ScoreSet(z[f"{args.scheme}|LR:{a}"], y) for a in ARMS}
    S["GBA"] = ScoreSet(z[f"{args.scheme}|gba"], y)
    S["pubs_pre2020"] = ScoreSet(z[f"{args.scheme}|base:log_pubs_pre2020"], y)
    S["abundance"] = ScoreSet(z[f"{args.scheme}|base:logbm"], y)
    prev = y.mean()
    out = {}

    # ---- 1. confound-matched benchmark
    bm, pub = F[f"logbm__{cell}"].values, F.log_pubs_pre2020.values
    bins = (pd.qcut(pd.Series(bm).rank(method="first"), 5, labels=False).values * 5
            + pd.qcut(pd.Series(pub).rank(method="first"), 5, labels=False).values)
    W = boot_weights(len(y), codes, args.B, "group", rng)
    rows = []
    for name in list(S):
        sets = []
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            if y[idx].sum() >= 1 and (1 - y[idx]).sum() >= 1:
                sets.append((ScoreSet(S[name].s[:, idx], y[idx]), idx))
        full = ScoreSet(S[name].s, y)
        obs = strat_auc(sets, y)
        boot = np.array([strat_auc(sets, y, w) for w in W])
        rows.append(dict(name=name, AUROC_overall=full.auroc(), AUROC_matched=obs,
                         ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
    MB = pd.DataFrame(rows)
    MB.to_csv(RESULTS / f"matched_benchmark__{args.tag}.csv", index=False)
    print("1. Confound-matched benchmark (strata = abundance x pubs_pre2020, 5x5):\n", MB.round(3).to_string(index=False))

    # ---- 2. publication terciles
    ter = pd.qcut(pd.Series(pub).rank(method="first"), 3, labels=["low", "mid", "high"]).values
    rows = []
    for name in ["E", "N", "E+N", "GBA", "nuis"]:
        for t in ("low", "mid", "high"):
            idx = np.where(ter == t)[0]
            St = ScoreSet(S[name].s[:, idx], y[idx])
            Wt = boot_weights(len(idx), pd.factorize(codes[idx])[0], args.B, "group", rng)
            v = np.array([St.ap(w) for w in Wt])
            rows.append(dict(arm=name, tercile=t, n_pos=int(y[idx].sum()), prevalence=y[idx].mean(), AUPRC=St.ap(),
                             lift=St.ap() / y[idx].mean(), lift_lo=np.percentile(v, 2.5) / y[idx].mean(),
                             lift_hi=np.percentile(v, 97.5) / y[idx].mean(), AUROC=St.auroc()))
    TE = pd.DataFrame(rows)
    TE.to_csv(RESULTS / f"publication_terciles__{args.tag}.csv", index=False)
    print("\n2. Lift by publication tercile:\n", TE.pivot(index="arm", columns="tercile", values="lift")[["low", "mid", "high"]].round(2).to_string())

    # ---- 3. P@k and calibration
    rows = [dict(arm=a, **{f"P@{k}": S[a].p_at_k(k) for k in (50, 100, 200, 500)}) for a in S]
    PK = pd.DataFrame(rows)
    PK["chance"] = prev
    PK.to_csv(RESULTS / f"precision_at_k__{args.tag}.csv", index=False)
    print("\n3. Precision@k (chance = %.3f):\n" % prev, PK.round(3).to_string(index=False))
    sc = S["E+N"].s.mean(0)
    dec = pd.qcut(pd.Series(sc).rank(method="first"), 10, labels=False)
    cal = pd.DataFrame({"mean_score": pd.Series(sc).groupby(dec).mean(), "observed_rate": pd.Series(y).groupby(dec).mean()})
    cal["brier_note"] = ""
    cal.to_csv(RESULTS / f"calibration_E+N__{args.tag}.csv")

    # ---- 4. top predictions and their neighbourhoods
    import scipy.sparse as sp
    el = np.array(g.get_edgelist())
    Fall = load_network("string_v11")[0]
    idx_all = {n: i for i, n in enumerate(Fall.index)}
    A = sp.coo_matrix((np.ones(len(el)), (el[:, 0], el[:, 1])), shape=(len(Fall),) * 2).tocsr()
    A = A + A.T
    posv = np.zeros(len(Fall))
    posv[[idx_all[x] for x in genes[y == 1]]] = 1
    pos_nbrs = pd.Series(A @ posv, index=Fall.index).reindex(genes).values
    c = pd.read_csv(RAW / "corum_human_complexes.txt", sep="\t", dtype=str)
    gset, posset = set(genes), set(genes[y == 1])
    co_pos = {}
    for sub in c.subunits_gene_name.dropna():
        m = {s for s in harmonise([x.strip() for x in sub.split(";")], alias) if s} & gset
        n_pos = len(m & posset)
        for x in m:
            co_pos[x] = max(co_pos.get(x, 0), n_pos - (1 if x in posset else 0))
    tops = {}
    for name in ("E+N", "GBA", "E"):
        s = S[name].s.mean(0)
        df = pd.DataFrame({"gene": genes, "score": s, "label": y, "n_pos_neighbours": pos_nbrs,
                           "max_pos_in_same_complex": [co_pos.get(x, 0) for x in genes],
                           "log_pubs_pre2020": pub, "abundance": bm}).sort_values("score", ascending=False)
        un = df[df.label == 0].head(100).copy()
        un["trivial_complex_completion"] = (un.n_pos_neighbours > 0) | (un.max_pos_in_same_complex > 0)
        un.to_csv(RESULTS / f"top100_unlabelled_{name}__{args.tag}.csv", index=False)
        filt = df[(df.label == 0) & (df.n_pos_neighbours == 0) & (df.max_pos_in_same_complex == 0)].head(100)
        filt.to_csv(RESULTS / f"top100_novel_filtered_{name}__{args.tag}.csv", index=False)
        tops[name] = (un, filt)
        print(f"\n4. {name}: of the top-100 unlabelled genes, {int((un.n_pos_neighbours > 0).sum())} have >=1 labelled neighbour, "
              f"{int((un.max_pos_in_same_complex > 0).sum())} share a CORUM complex with a labelled gene, "
              f"{int(un.trivial_complex_completion.sum())} are 'trivial'. Top 12 filtered: {list(filt.gene.head(12))}")

    # ---- 5. ORA vs universe background
    try:
        import gseapy as gp
        rows = []
        for name in ("E+N", "GBA"):
            for kind, lst in (("top100", list(tops[name][0].gene)), ("top100_filtered", list(tops[name][1].gene))):
                er = gp.enrichr(gene_list=lst, gene_sets=["GO_Biological_Process_2021", "Reactome_2022"], background=list(genes), outdir=None)
                r = er.results.sort_values("Adjusted P-value").head(8)
                r.insert(0, "set", f"{name}:{kind}")
                rows.append(r[[c for c in ["set", "Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value"] if c in r.columns]])
        ORA = pd.concat(rows)
        ORA.to_csv(RESULTS / f"enrichment_top100__{args.tag}.csv", index=False)
        print("\n5. ORA (background = universe), best term per list:\n", ORA.groupby("set").head(2).to_string(index=False)[:2500])
    except Exception as e:
        print("\n5. ORA skipped:", repr(e)[:300])

    # ---- 6. housekeeping test
    hk = set(harmonise(pd.read_csv(RAW / "housekeeping_EisenbergLevanon2013.txt", sep="\t", header=None)[0].str.strip(), alias))
    isHK = np.isin(genes, list(hk))
    rows = []
    for name in ("gordon", "crispr"):
        yy = np.isin(genes, list(load_labels(name, alias)))
        a, b, c_, d = (isHK & yy).sum(), (isHK & ~yy).sum(), (~isHK & yy).sum(), (~isHK & ~yy).sum()
        OR, p = fisher_exact([[a, b], [c_, d]])
        rows.append(dict(labels=name, HK_among_pos=a / yy.sum(), HK_among_unlabelled=b / (~yy).sum(), odds_ratio=OR, p=p))
    import statsmodels.api as sm
    X = pd.DataFrame({"HK": isHK.astype(float), "abundance": bm, "log_pubs_pre2020": pub, "log_degree": F.log_degree.values})
    Xs = X.copy()
    for col in ("abundance", "log_pubs_pre2020", "log_degree"):      # standardise covariates; HK stays 0/1
        Xs[col] = (X[col] - X[col].mean()) / X[col].std()
    m = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0)
    HK = pd.DataFrame(rows)
    HK.loc[len(HK)] = dict(labels=f"{lab} adjusted (logit, HK coefficient OR)", odds_ratio=float(np.exp(m.params["HK"])), p=float(m.pvalues["HK"]))
    HK.to_csv(RESULTS / f"housekeeping_test__{args.tag}.csv", index=False)
    print("\n6. Housekeeping:\n", HK.round(4).to_string(index=False), f"\n   HK genes in universe: {isHK.sum()} ({isHK.mean():.1%})")

    # ---- 7. coefficients + SHAP
    Xe = build_X(F, "E+N", cell)
    Z = StandardScaler().fit_transform(Xe)
    lr = make_model("LR", seed).fit(Xe.values, y)
    coef = pd.Series(lr[-1].coef_[0], index=Xe.columns).sort_values()
    coef.to_csv(RESULTS / f"lr_coefficients_E+N__{args.tag}.csv")
    print("\n7. LR (standardised) coefficients, E+N:\n", coef.round(3).to_string())
    try:
        import shap
        rf = make_model("RF", seed).fit(Xe.values, y)
        sv = shap.TreeExplainer(rf).shap_values(Xe.values[: 4000])
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        imp = pd.Series(np.abs(sv).mean(0), index=Xe.columns).sort_values()
        imp.to_csv(RESULTS / f"rf_shap_E+N__{args.tag}.csv")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        coef.plot.barh(ax=ax[0]); ax[0].set_title("Logistic coefficients (standardised)")
        imp.plot.barh(ax=ax[1]); ax[1].set_title("Random forest |SHAP| (first 4000 genes)")
        savefig(fig, "17_feature_importance")
    except Exception as e:
        print("   SHAP skipped:", repr(e)[:200])

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = MB.set_index("name")
    order = ["abundance", "pubs_pre2020", "nuis", "E", "N", "E+N", "E+nuis", "E+N+nuis", "GBA"]
    ax[0].errorbar(range(len(order)), t.loc[order].AUROC_matched, yerr=[t.loc[order].AUROC_matched - t.loc[order].ci_lo, t.loc[order].ci_hi - t.loc[order].AUROC_matched], fmt="o", capsize=3, label="within-stratum")
    ax[0].plot(range(len(order)), t.loc[order].AUROC_overall, "x", c="tab:red", label="overall")
    ax[0].axhline(0.5, c="k", lw=0.5); ax[0].set_xticks(range(len(order)), order, rotation=60, fontsize=7); ax[0].legend(frameon=False)
    ax[0].set_ylabel("AUROC"); ax[0].set_title("Confound-matched benchmark")
    for a, mk in (("E", "o"), ("N", "s"), ("E+N", "^"), ("GBA", "d")):
        tt = TE[TE.arm == a].set_index("tercile").loc[["low", "mid", "high"]]
        ax[1].errorbar(range(3), tt.lift, yerr=[tt.lift - tt.lift_lo, tt.lift_hi - tt.lift], marker=mk, capsize=3, label=a)
    ax[1].axhline(1, c="k", lw=0.5); ax[1].set_xticks(range(3), ["low", "mid", "high"]); ax[1].set_xlabel("publication tercile (pre-2020)")
    ax[1].set_ylabel("lift"); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title("Does it work for understudied genes?")
    savefig(fig, "15_matched_and_terciles")


if __name__ == "__main__":
    main()
