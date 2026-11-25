"""Step S3 - communities, CV groups, module test and gene-feature-space structure.

Why this step exists
  * Genes are NOT independent rows: members of a protein complex share near-identical network
    neighbourhoods. Step 4 therefore also cross-validates on GROUPS (whole complexes / modules held
    out together). This script builds those groups.
  * "Do the positives form a module?" (Menche 2015 / Gysi 2021 premise) is tested directly with a
    degree-matched largest-connected-component test.
  * Patterns in reduced-dimensional space: PCA/UMAP of the gene x feature matrix, clusters in that space
    and their tie to metadata (label prevalence, fame, abundance, DE status).

Outputs
  data/processed/groups_<ppi>.csv            gene, community_coarse, community_fine, complex_group
  Results/community_enrichment.csv         label enrichment per coarse Leiden community (BH)
  Results/group_dependence.csv             how often positives share a group (per scheme)
  Results/module_lcc_test.csv              degree-matched LCC test of the positive sets
  Results/feature_space_clusters.csv       k-means clusters of the PCA space, with metadata
  figures 10-12
Usage: python 03b_groups.py [--ppi string_v11]
"""
import argparse
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import os
import pickle
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

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def leiden(g, resolution, seed):
    import igraph as ig
    random.seed(seed)
    ig.set_random_number_generator(random)
    return np.array(g.community_leiden(objective_function="modularity", resolution=resolution,
                                       n_iterations=-1).membership)


def corum_groups(genes, alias, seed):
    """Cluster genes by CORUM co-membership. Merging complexes that share a subunit gives one giant
    component (3,298 of 10,722 genes), which is useless as a CV group - so the co-complex graph is
    split with Leiden. Genes in no complex are their own group."""
    import igraph as ig
    c = pd.read_csv(RAW / "corum_human_complexes.txt", sep="\t", dtype=str)
    gset, idx = set(genes), {g: i for i, g in enumerate(genes)}
    edges = {}
    for sub in c.subunits_gene_name.dropna():
        m = sorted({s for s in harmonise([x.strip() for x in sub.split(";")], alias) if s} & gset)
        if len(m) < 2:
            continue
        w = 1.0 / (len(m) - 1)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                k = (idx[m[i]], idx[m[j]])
                edges[k] = edges.get(k, 0.0) + w
    members = sorted({a for e in edges for a in e})
    sub_idx = {a: i for i, a in enumerate(members)}
    g = ig.Graph(n=len(members), edges=[(sub_idx[a], sub_idx[b]) for a, b in edges], directed=False)
    g.es["weight"] = list(edges.values())
    random.seed(seed)
    ig.set_random_number_generator(random)
    mem = g.community_leiden(objective_function="modularity", weights="weight", resolution=1.0,
                             n_iterations=-1).membership
    grp = {genes[a]: f"cx{m}" for a, m in zip(members, mem)}
    return pd.Series([grp.get(x, f"single_{x}") for x in genes], index=genes)


def lcc_size(g, vids):
    return max((len(c) for c in g.induced_subgraph(vids).connected_components()), default=0)


def module_test(g, F, pos, n_perm, rng):
    """Largest connected component among the positives vs random gene sets matched on degree."""
    deg = np.array(g.degree())
    names = np.array(g.vs["name"])
    pv = np.where(np.isin(names, list(pos)))[0]
    obs = lcc_size(g, pv.tolist())
    bins = pd.qcut(pd.Series(deg).rank(method="first"), 20, labels=False).values
    by_bin = {b: np.where(bins == b)[0] for b in np.unique(bins)}
    null = []
    for _ in range(n_perm):
        draw = [rng.choice(by_bin[bins[v]]) for v in pv]
        null.append(lcc_size(g, draw))
    null = np.array(null)
    return dict(n_pos=len(pv), lcc_obs=obs, null_mean=null.mean(), null_sd=null.std(ddof=1),
                z=(obs - null.mean()) / null.std(ddof=1), p_emp=(1 + (null >= obs).sum()) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppi", default="string_v11")
    ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()
    seed = CFG["seed"]
    rng = np.random.default_rng(seed)
    alias, _, _ = symbol_maps()
    F, g = load_network(args.ppi)
    genes = list(F.index)
    labels = {n: load_labels(n, alias) & set(genes) for n in ("gordon", "crispr", "crispr_min2")}

    # ---- groups ----------------------------------------------------------------------------
    coarse = leiden(g, 1.0, seed)
    sizes = {}
    for res in (1, 2, 5, 10, 20):
        m = leiden(g, res, seed)
        sizes[res] = (len(set(m)), int(np.median(np.bincount(m))))
    print("Leiden resolution -> (n communities, median size):", sizes)
    fine_res = min(sizes, key=lambda r: abs(sizes[r][0] - 200))   # ~200 communities
    fine = leiden(g, fine_res, seed)
    print(f"coarse res=1: {len(set(coarse))} communities; fine res={fine_res}: {len(set(fine))} communities")
    cx = corum_groups(genes, alias, seed)
    G = pd.DataFrame({"community_coarse": coarse, "community_fine": fine, "complex_group": cx.values}, index=genes)
    G.to_csv(PROC / f"groups_{args.ppi}.csv")

    dep = []
    for scheme in G.columns:
        for name, pos in labels.items():
            gs = G.loc[list(pos), scheme]
            full = G[scheme].value_counts()
            shared = np.mean([full[x] > 1 and (gs == x).sum() > 1 for x in gs])
            dep.append(dict(scheme=scheme, labels=name, n_groups=G[scheme].nunique(),
                            frac_pos_sharing_group_with_another_pos=shared,
                            largest_group=int(full.max())))
    dep = pd.DataFrame(dep)
    dep.to_csv(RESULTS / "group_dependence.csv", index=False)
    print(dep.round(3).to_string(index=False))

    # ---- community enrichment ---------------------------------------------------------------
    rows = []
    N = len(genes)
    for name, pos in labels.items():
        y = pd.Series(G.index.isin(pos), index=G.index)
        sub = []
        for cid, idx in G.groupby("community_coarse").groups.items():
            n, k, K = len(idx), int(y[idx].sum()), int(y.sum())
            sub.append(dict(labels=name, community=cid, size=n, positives=k, expected=n * K / N,
                            fold=(k / max(n * K / N, 1e-9)), p=hypergeom.sf(k - 1, N, K, n)))
        sub = pd.DataFrame(sub)
        sub["q_BH"] = bh(sub.p.values)
        rows.append(sub)
    CE = pd.concat(rows)
    CE.to_csv(RESULTS / "community_enrichment.csv", index=False)
    print("\nCommunities enriched for positives (q<0.05):")
    print(CE[CE.q_BH < 0.05].sort_values("q_BH").round(4).to_string(index=False))

    # ---- module (LCC) test --------------------------------------------------------------------
    mt = []
    for name, pos in labels.items():
        r = module_test(g, F, pos, args.perm, rng)
        mt.append(dict(labels=name, **r))
    mt = pd.DataFrame(mt)
    mt.to_csv(RESULTS / "module_lcc_test.csv", index=False)
    print("\nDegree-matched LCC test:\n", mt.round(3).to_string(index=False))

    # ---- reduced-dimensional structure of the gene x feature matrix --------------------------
    X = build_X(F, "E+N", CFG["main_cell"])
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=6, random_state=seed).fit(Z)
    S = pca.transform(Z)
    load = pd.DataFrame(pca.components_.T, index=X.columns, columns=[f"PC{i+1}" for i in range(6)])
    print("\nPCA explained variance:", np.round(pca.explained_variance_ratio_, 3))
    print(load.iloc[:, :3].round(2).to_string())
    y_g = F.index.isin(labels["gordon"])
    pc_auc = {f"PC{i+1}": roc_auc_score(y_g, S[:, i]) for i in range(6)}
    print("AUROC of each PC for Gordon:", {k: round(v, 3) for k, v in pc_auc.items()})

    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(Z)
    sub = rng.choice(len(Z), 3000, replace=False)
    ks = list(range(2, 9))
    sil = [silhouette_score(S[sub, :5], KMeans(k, n_init=10, random_state=seed).fit_predict(S[sub, :5])) for k in ks]
    kbest = ks[int(np.argmax(sil))]
    km = KMeans(kbest, n_init=20, random_state=seed).fit_predict(S[:, :5])
    meta = pd.DataFrame({"cluster": km, "gordon": y_g, "crispr": F.index.isin(labels["crispr"]),
                         "log_pubs_pre2020": F.log_pubs_pre2020.values, "logbm": F[f"logbm__{CFG['main_cell']}"].values,
                         "log_degree": F.log_degree.values, "abs_lfc": F[f"abs_lfc__{CFG['main_cell']}"].values,
                         "DE": ((F[f"padj__{CFG['main_cell']}"] < CFG["de_padj"]) & (F[f"abs_lfc__{CFG['main_cell']}"] > CFG["de_lfc"])).values})
    cl = meta.groupby("cluster").agg(n=("gordon", "size"), gordon_rate=("gordon", "mean"), crispr_rate=("crispr", "mean"),
                                     DE_rate=("DE", "mean"), median_log_pubs=("log_pubs_pre2020", "median"),
                                     median_logbm=("logbm", "median"), median_log_degree=("log_degree", "median"))
    cl["fold_gordon"] = cl.gordon_rate / meta.gordon.mean()
    cl["p_gordon"] = [hypergeom.sf(int(meta[meta.cluster == c].gordon.sum()) - 1, N, int(meta.gordon.sum()), int((meta.cluster == c).sum())) for c in cl.index]
    cl["q_BH"] = bh(cl.p_gordon.values)
    cl.to_csv(RESULTS / "feature_space_clusters.csv")
    print(f"\nk-means on PC1-5, silhouette-chosen k={kbest} (sil: {dict(zip(ks, np.round(sil, 3)))})")
    print(cl.round(3).to_string())

    # ---- figures --------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    ax = axes[0, 0]
    ax.scatter(emb[:, 0], emb[:, 1], s=2, c="lightgrey", rasterized=True)
    ax.scatter(emb[y_g, 0], emb[y_g, 1], s=9, c="tab:red", label=f"Gordon (n={y_g.sum()})")
    yc = F.index.isin(labels["crispr"])
    ax.scatter(emb[yc, 0], emb[yc, 1], s=9, c="tab:blue", label=f"CRISPR (n={yc.sum()})")
    ax.legend(frameon=False, fontsize=7, markerscale=2)
    ax.set_title("UMAP of 11-feature E+N matrix: labels")
    for ax, col, t in ((axes[0, 1], F.log_pubs_pre2020, "log(1+pubs, pre-2020)"),
                       (axes[1, 0], F[f"logbm__{CFG['main_cell']}"], "log baseMean (abundance)")):
        h = ax.scatter(emb[:, 0], emb[:, 1], s=2, c=col, cmap="viridis", rasterized=True)
        fig.colorbar(h, ax=ax, label=t)
        ax.set_title(f"UMAP coloured by {t}")
    ax = axes[1, 1]
    ax.scatter(emb[:, 0], emb[:, 1], s=2, c=km, cmap="tab10", rasterized=True)
    ax.set_title(f"k-means clusters (k={kbest}) on PC1-5")
    for a in axes.ravel():
        a.set_xlabel("UMAP1"); a.set_ylabel("UMAP2")
    savefig(fig, "10_feature_space_umap")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(range(1, 7), pca.explained_variance_ratio_)
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("variance explained"); axes[0].set_title("PCA of gene x feature matrix")
    im = axes[1].imshow(load.iloc[:, :3].values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    axes[1].set_yticks(range(len(load)), load.index, fontsize=7); axes[1].set_xticks(range(3), ["PC1", "PC2", "PC3"])
    fig.colorbar(im, ax=axes[1], label="loading"); axes[1].set_title("Loadings")
    axes[2].bar(range(1, 7), [pc_auc[f"PC{i}"] - 0.5 for i in range(1, 7)], bottom=0.5)
    axes[2].axhline(0.5, c="k", lw=0.5); axes[2].set_xlabel("PC"); axes[2].set_ylabel("AUROC for Gordon")
    axes[2].set_title("Which axis separates positives?")
    savefig(fig, "11_pca_feature_space")

    top = CE[CE.labels == "gordon"].sort_values("fold", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(CE[CE.labels == "gordon"]["size"], CE[CE.labels == "gordon"].fold, s=18, c=np.where(CE[CE.labels == "gordon"].q_BH < 0.05, "tab:red", "grey"))
    ax.axhline(1, c="k", lw=0.5); ax.set_xscale("log")
    ax.set_xlabel("community size"); ax.set_ylabel("Gordon enrichment (obs/exp)")
    ax.set_title(f"Leiden communities (red: q<0.05, n={(top.q_BH < 0.05).sum()})")
    savefig(fig, "12_community_enrichment")


if __name__ == "__main__":
    main()
