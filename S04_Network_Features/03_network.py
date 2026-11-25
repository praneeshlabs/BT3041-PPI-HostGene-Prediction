"""Step 3 - build the PPI network, define the gene universe, compute features for the
three arms (E, N, E+N), and run label-level statistics (DE enrichment among positives,
degree / publication bias, single-feature AUROCs).

Outputs (data/processed/):
    features_<ppi>.csv          all features + covariates, one row per gene in the universe
    edges_<ppi>.csv             edges of the universe graph
    X_<arm>_<ppi>.csv           ready-to-model matrices for E, N and E+N (main cell line)

Usage:
    python 03_network.py --ppi string_v11        # main analysis (STRING v11.0, combined >= 700)
    python 03_network.py --ppi string_v11_exp    # experimental channel only
    python 03_network.py --ppi huri              # HuRI (systematic Y2H, low study bias)
    python 03_network.py --ppi biogrid           # BioGRID physical (file placed manually)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

import os
import pickle
import scipy.sparse as sp
from pathlib import Path

# ============================================================== helpers
# Paths, settings and functions this step needs (written out here so the script runs on its own).

# --------------------------------------------------------------------------- paths
# Data/{raw,labels,processed} sit in the project root, one level above this step's folder.
PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("HOSTNET_DATA", PROJECT / "Data"))
RAW, LABELS, PROC = DATA / "raw", DATA / "labels", DATA / "processed"

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


def available_labels():
    return sorted(p.stem for p in LABELS.glob("*.csv"))


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


# --------------------------------------------------------------------------- publications + feature table
def load_pubs(entrez) -> pd.DataFrame:
    """Publication counts per gene symbol: n_pubs (live file, all years) and n_pubs_pre2020
    (papers with PubMed ID <= CFG['pmid_cutoff'], i.e. Entrez-dated before 2020-01-01).
    The live gene2pubmed file keeps growing and contains post-COVID papers about the label genes
    themselves, so the date-frozen count is the one consistent with the pre-pandemic network."""
    cache = PROC / "pubs.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    live, frozen = {}, {}
    for chunk in pd.read_csv(raw_path("gene2pubmed"), sep="	", dtype=str, chunksize=2_000_000):
        chunk.columns = [c.lstrip("#") for c in chunk.columns]
        chunk = chunk[chunk.tax_id == "9606"].drop_duplicates(["GeneID", "PubMed_ID"])
        for gid, n in chunk.GeneID.value_counts().items():
            live[gid] = live.get(gid, 0) + n
        old = chunk[chunk.PubMed_ID.astype(np.int64) <= CFG["pmid_cutoff"]]
        for gid, n in old.GeneID.value_counts().items():
            frozen[gid] = frozen.get(gid, 0) + n
    s = pd.DataFrame({"n_pubs": pd.Series({entrez[g]: n for g, n in live.items() if g in entrez}),
                      "n_pubs_pre2020": pd.Series({entrez[g]: n for g, n in frozen.items() if g in entrez})})
    s = s.fillna(0).groupby(level=0).sum()
    s.to_csv(cache)
    return s


def assemble_features(g, de: dict, cells, entrez) -> pd.DataFrame:
    """Full gene x feature table for the universe graph g: topology, per-cell expression features,
    propagated |lfc| (raw and degree-normalised) and publication covariates."""
    genes = g.vs["name"]
    F = topo_features(g)
    for c in cells:
        r = de[c].reindex(genes)
        F[f"lfc__{c}"] = r.log2FoldChange.fillna(0).values
        F[f"abs_lfc__{c}"] = F[f"lfc__{c}"].abs()
        F[f"nlp__{c}"] = -np.log10(r.pvalue.clip(lower=1e-300)).fillna(0).values
        F[f"logbm__{c}"] = np.log1p(r.baseMean.fillna(0)).values
        F[f"stat__{c}"] = r.stat.fillna(0).values
        F[f"padj__{c}"] = r.padj.fillna(1).values
    _, W, _ = graph_matrices(g)
    F = F.join(prop_features(W, F, cells))
    pubs = load_pubs(entrez)
    F["n_pubs"] = pubs.n_pubs.reindex(genes).fillna(0).values
    F["log_pubs"] = np.log1p(F.n_pubs)
    F["n_pubs_pre2020"] = pubs.n_pubs_pre2020.reindex(genes).fillna(0).values
    F["log_pubs_pre2020"] = np.log1p(F.n_pubs_pre2020)
    return F

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers


# --------------------------------------------------------------------------- edges
def edges_string(alias, exp_only=False):
    info = pd.read_csv(raw_path("string_info"), sep="\t", header=0).iloc[:, :2]
    info.columns = ["pid", "name"]
    pid2sym = dict(zip(info.pid, harmonise(info.name, alias)))
    if exp_only:
        df = pd.read_csv(raw_path("string_detailed"), sep=" ",
                         usecols=["protein1", "protein2", "experimental"])
        df = df[df.experimental >= CFG["string_exp_min_score"]]
    else:
        df = pd.read_csv(raw_path("string_links"), sep=" ")
        df = df[df.combined_score >= CFG["string_min_score"]]
    return pd.DataFrame({"a": df.protein1.map(pid2sym), "b": df.protein2.map(pid2sym)})


def edges_huri(ens):
    df = pd.read_csv(raw_path("huri"), sep="\t", header=None, dtype=str).iloc[:, :2]
    df.columns = ["a", "b"]
    df = df[df.a.str.startswith("ENSG") & df.b.str.startswith("ENSG")]
    strip = lambda s: s.str.split(".").str[0]
    return pd.DataFrame({"a": strip(df.a).map(ens), "b": strip(df.b).map(ens)})


def edges_biogrid(alias):
    files = sorted(RAW.glob("BIOGRID-ORGANISM-Homo_sapiens*.tab3.txt"))
    if not files:
        raise FileNotFoundError("Place BIOGRID-ORGANISM-Homo_sapiens-*.tab3.txt in data/raw/")
    df = pd.read_csv(files[-1], sep="\t", dtype=str, low_memory=False)
    df = df[(df["Experimental System Type"] == "physical")
            & (df["Organism ID Interactor A"] == "9606") & (df["Organism ID Interactor B"] == "9606")]
    return pd.DataFrame({"a": harmonise(df["Official Symbol Interactor A"], alias),
                         "b": harmonise(df["Official Symbol Interactor B"], alias)})


def clean_edges(E):
    E = E.dropna()
    E = E[E.a != E.b]
    E = pd.DataFrame(np.sort(E[["a", "b"]].values, axis=1), columns=["a", "b"])
    return E.drop_duplicates().reset_index(drop=True)


# --------------------------------------------------------------------------- main
def main():
    import igraph as ig

    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi", default="string_v11",
                    choices=["string_v11", "string_v11_exp", "huri", "biogrid"])
    args = ap.parse_args()
    alias, ens, entrez = symbol_maps()
    cells = [c for c in CFG["cells"] if (PROC / f"de_{c}.csv").exists()]
    main_cell = CFG["main_cell"]
    de = {c: pd.read_csv(PROC / f"de_{c}.csv", index_col=0) for c in cells}

    # ---- network
    print(f"Building network: {args.ppi}")
    E = {"string_v11": lambda: edges_string(alias),
         "string_v11_exp": lambda: edges_string(alias, exp_only=True),
         "huri": lambda: edges_huri(ens),
         "biogrid": lambda: edges_biogrid(alias)}[args.ppi]()
    E = clean_edges(E)
    nodes = set(E.a) | set(E.b)
    print(f"  raw network: {len(nodes)} genes, {len(E)} edges")

    # ---- gene universe = expressed (main cell line) AND in network; then largest component
    expressed = set(de[main_cell].index[de[main_cell].baseMean >= CFG["min_basemean"]])
    U = expressed & nodes
    E = E[E.a.isin(U) & E.b.isin(U)]
    g = ig.Graph.TupleList(E.itertuples(index=False), directed=False)
    try:
        g = g.connected_components().giant()
    except AttributeError:
        g = g.clusters().giant()
    genes = g.vs["name"]
    print(f"  expressed: {len(expressed)} | expressed & in network: {len(U)} | "
          f"largest component: {g.vcount()} genes, {g.ecount()} edges")

    # ---- features (same code as the leakage panel: assemble_features)
    print("Computing topology features (betweenness/closeness can take a few minutes) ...")
    F = assemble_features(g, de, cells, entrez)
    F.to_csv(PROC / f"features_{args.ppi}.csv")
    el = g.get_edgelist()
    pd.DataFrame({"a": [genes[i] for i, _ in el], "b": [genes[j] for _, j in el]}).to_csv(
        PROC / f"edges_{args.ppi}.csv", index=False)
    print(f"  saved features_{args.ppi}.csv ({F.shape[0]} genes x {F.shape[1]} columns)")

    # ---- the three arms, written as ready-to-model matrices (main cell line)
    for arm in ARMS:   # the three headline arms; Step 4 builds the extended arms on the fly
        X = build_X(F, arm, main_cell)
        assert np.isfinite(X.values).all(), f"non-finite values in arm {arm}"
        X.to_csv(PROC / f"X_{arm.replace('+', 'plus')}_{args.ppi}.csv")
        print(f"  arm {arm:4s}: {X.shape[1]} features -> {list(X.columns)}")

    # ---- sanity checks
    assert F.index.is_unique, "duplicate genes in universe"
    assert (F.degree > 0).all(), "isolated node in giant component?"
    print(f"  publication counts: {(F.n_pubs == 0).sum()} genes with 0 papers (imputed as 0, not NaN)")

    # ---- feature correlation (Spearman): shows how much topology overlaps study bias
    cols = (["lfc__" + main_cell, "abs_lfc__" + main_cell, "nlp__" + main_cell,
             "logbm__" + main_cell, "prop__" + main_cell, "propn__" + main_cell] + ARMS["N"] + ["log_pubs", "log_pubs_pre2020"])
    C = F[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [c.replace(f"__{main_cell}", "") for c in cols]
    ax.set_xticks(range(len(cols)), short, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), short, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="w" if abs(C.values[i, j]) > 0.6 else "k")
    fig.colorbar(im, label="Spearman rho")
    ax.set_title(f"Feature correlations ({args.ppi}, {main_cell})")
    savefig(fig, f"08_feature_correlation_{args.ppi}")

    # ---- degree distribution
    deg = F.degree.values
    vals, cnt = np.unique(deg, return_counts=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.loglog(vals, cnt / cnt.sum(), "o", ms=3)
    ax.set_xlabel("Degree k")
    ax.set_ylabel("P(k)")
    ax.set_title(f"Degree distribution ({args.ppi})")
    savefig(fig, f"06_degree_distribution_{args.ppi}")

    # ---- label statistics
    rows, funnel, auc_rows = [], [], []
    labs = available_labels()
    if not labs:
        print("No label files in data/labels/ - skipping label statistics.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(4 * len(labs) + 2, 4))
    for li, name in enumerate(labs):
        pos = load_labels(name, alias)
        funnel.append(dict(labels=name, ppi=args.ppi, mapped=len(pos),
                           expressed=len(pos & expressed), in_network=len(pos & nodes),
                           in_universe=len(pos & set(genes))))
        y = F.index.isin(pos)
        # degree and publication bias
        for ax, col, lab in ((axes[0], "log_degree", "log(1+degree)"),
                             (axes[1], "log_pubs", "log(1+#publications)")):
            data = [F.loc[~y, col], F.loc[y, col]]
            ax.violinplot(data, positions=[2 * li, 2 * li + 1], showmedians=True)
            p = mannwhitneyu(data[1], data[0]).pvalue
            ax.text(2 * li + 0.5, max(d.max() for d in data), f"MWU p={p:.1e}",
                    ha="center", va="bottom", fontsize=7)
            ax.set_ylabel(lab)
        rows.append(dict(labels=name, test="degree pos vs unlabeled (MWU)",
                         p=mannwhitneyu(F.loc[y, "degree"], F.loc[~y, "degree"]).pvalue))
        rows.append(dict(labels=name, test="pubs pos vs unlabeled (MWU)",
                         p=mannwhitneyu(F.loc[y, "n_pubs"], F.loc[~y, "n_pubs"]).pvalue))
        # descriptive single-feature AUROC (no model, no fitting -> no leakage)
        for col in cols:
            auc = roc_auc_score(y, F[col])
            auc_rows.append(dict(labels=name, feature=col.replace(f"__{main_cell}", ""),
                                 AUROC=auc, rho_with_log_pubs=spearmanr(F[col], F.log_pubs)[0]))
        # DE enrichment among positives (hypergeometric + Fisher OR), universe = network genes
        N, K = len(F), int(y.sum())
        for c in cells:
            sig = (F[f"padj__{c}"] < CFG["de_padj"]) & (F[f"abs_lfc__{c}"] > CFG["de_lfc"])
            n, k = int(sig.sum()), int((sig & y).sum())
            OR, _ = fisher_exact([[k, K - k], [n - k, N - K - n + k]])
            # BOTH tails are stored: enrichment (upper) and depletion (lower). The H1 finding is a
            # depletion, which the upper-tail p alone (~1) cannot show.
            rows.append(dict(labels=name, test=f"DE enrichment {c}", N=N, K=K, n=n, k=k,
                             expected=round(n * K / N, 1), odds_ratio=OR,
                             p=hypergeom.sf(k - 1, N, K, n), p_depletion=hypergeom.cdf(k, N, K, n)))
            # threshold-free: preranked GSEA of the positive set on the Wald statistic
            try:
                import gseapy as gp
                rnk = F[f"stat__{c}"].sort_values(ascending=False)
                pre = gp.prerank(rnk=rnk, gene_sets={name: sorted(pos & set(genes))},
                                 min_size=5, max_size=len(genes), permutation_num=1000,
                                 outdir=None, seed=CFG["seed"], threads=4, verbose=False)
                r = pre.res2d.iloc[0]
                # permutation p cannot be below 1/(nperm+1); report that floor instead of 0.0
                p_raw = float(r["NOM p-val"])
                rows.append(dict(labels=name, test=f"GSEA prerank {c}", NES=float(r["NES"]),
                                 p=max(p_raw, 1 / 1001), p_note="p<0.001 (permutation floor 1/1001)" if p_raw < 1 / 1001 else ""))
            except Exception as e:
                print(f"  (GSEA prerank skipped for {c}: {e})")
    names = [f"{n}\n{t}" for n in labs for t in ("unlab.", "pos.")]
    for ax in axes:
        ax.set_xticks(range(len(names)), names, fontsize=7)
    axes[0].set_title("Degree: positives vs unlabeled")
    axes[1].set_title("Study bias: publications")
    savefig(fig, f"07_label_bias_{args.ppi}")

    A = pd.DataFrame(auc_rows)
    A.to_csv(RESULTS / f"single_feature_auroc_{args.ppi}.csv", index=False)
    P = A.pivot(index="feature", columns="labels", values="AUROC").loc[[c.replace(f"__{main_cell}", "") for c in cols]]
    fig, ax = plt.subplots(figsize=(1.5 + 1.4 * P.shape[1], 5))
    im = ax.imshow(P.values, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(P.shape[1]), P.columns)
    ax.set_yticks(range(P.shape[0]), P.index, fontsize=8)
    fig.colorbar(im, label="AUROC (0.5 = no signal)")
    ax.set_title("Single-feature AUROC vs labels")
    savefig(fig, f"09_single_feature_auroc_{args.ppi}")

    out = RESULTS / f"label_tests_{args.ppi}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(funnel).to_csv(RESULTS / f"label_funnel_{args.ppi}.csv", index=False)
    print(pd.DataFrame(funnel).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
