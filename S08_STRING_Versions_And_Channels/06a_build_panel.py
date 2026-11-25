"""Step S2 - network panel for RQ4 (how much apparent network signal is literature artefact?).

Builds, with the SAME universe rules as 03_network.py (expressed in Calu-3 AND in the network, largest
connected component) and the same feature code (assemble_features below):

  A. STRING database VINTAGE, combined score >= 700 (plus density-matched versions *_m) (v10.5 could not be downloaded - the archive host
     returns 404/403 - so the dose-response has 3 points):
        string_v11   (Jan 2019, pre-COVID)   - already built by 03_network.py
        string_v115  (Aug 2021; text mining covers PubMed to Jul 2020, i.e. first ~6 months of COVID)
        string_v120  (Jul 2023; text mining and databases cover 3+ years of COVID papers)
  B. STRING v11.0 EVIDENCE CHANNELS, each thresholded to the SAME number of edges as the combined>=700
     network inside the expressed-gene universe (otherwise differences reflect density, not information):
        v11_experiments, v11_database, v11_textmining, v11_coexpression

Also writes the shared gene sets on which all networks are compared (identical genes, labels, folds):
  data/processed/shared_versions.csv   genes in the universe of all three vintages
  data/processed/shared_all.csv        genes in the universe of string_v11 and the four channel networks
Usage: python 06a_build_panel.py [--procs 6]
"""
import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

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

for _d in (RAW, LABELS, PROC):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")
VER = RAW / "string_versions"
LINKS = {"string_v115": ("9606.protein.links.v11.5.txt.gz", "9606.protein.info.v11.5.txt.gz"),
         "string_v120": ("9606.protein.links.v12.0.txt.gz", "9606.protein.info.v12.0.txt.gz")}
CHANNELS = {"v11_experiments": "experimental", "v11_database": "database",
            "v11_textmining": "textmining", "v11_coexpression": "coexpression"}


def clean(E):
    E = E.dropna()
    E = E[E.a != E.b]
    return pd.DataFrame(np.sort(E[["a", "b"]].values, axis=1), columns=["a", "b"]).drop_duplicates().reset_index(drop=True)


def pid_map(info_path, alias):
    info = pd.read_csv(info_path, sep="\t", header=0).iloc[:, :2]
    info.columns = ["pid", "name"]
    return dict(zip(info.pid, harmonise(info.name, alias)))


def build_one(args):
    """Build one network -> features_<name>.csv + edges_<name>.csv; returns summary row."""
    import igraph as ig
    name, E, expressed = args
    t0 = time.time()
    alias, ens, entrez = symbol_maps()
    cells = [c for c in CFG["cells"] if (PROC / f"de_{c}.csv").exists()]
    de = {c: pd.read_csv(PROC / f"de_{c}.csv", index_col=0) for c in cells}
    U = expressed & (set(E.a) | set(E.b))
    E = E[E.a.isin(U) & E.b.isin(U)]
    g = ig.Graph.TupleList(E.itertuples(index=False), directed=False).connected_components().giant()
    genes = g.vs["name"]
    F = assemble_features(g, de, cells, entrez)
    F.to_csv(PROC / f"features_{name}.csv")
    el = g.get_edgelist()
    pd.DataFrame({"a": [genes[i] for i, _ in el], "b": [genes[j] for _, j in el]}).to_csv(PROC / f"edges_{name}.csv", index=False)
    print(f"  {name}: {g.vcount()} genes, {g.ecount()} edges ({time.time() - t0:.0f}s)", flush=True)
    return dict(network=name, genes=g.vcount(), edges=g.ecount(), mean_degree=2 * g.ecount() / g.vcount())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    alias, _, _ = symbol_maps()
    de = pd.read_csv(PROC / f"de_{CFG['main_cell']}.csv", index_col=0)
    expressed = set(de.index[de.baseMean >= CFG["min_basemean"]])

    jobs = []
    # --- A. vintages. STRING re-calibrated its scores: the same cut-off (>=700) keeps 260k edges in
    # v11.0 but only 168k (v11.5) / 148k (v12.0) inside the universe, so the fixed-threshold networks
    # differ in DENSITY. We therefore also build density-matched versions (top-M edges by combined score,
    # M = edges of the v12.0 universe network before the largest-component step).
    def links(lk, inf):
        p2s = pid_map(inf, alias)
        df = pd.read_csv(lk, sep=" ")
        d = pd.DataFrame({"a": df.protein1.map(p2s), "b": df.protein2.map(p2s), "score": df.combined_score}).dropna()
        d = d[(d.a != d.b) & d.a.isin(expressed) & d.b.isin(expressed)]
        d[["a", "b"]] = np.sort(d[["a", "b"]].values, axis=1)
        return d.sort_values("score", ascending=False).drop_duplicates(["a", "b"]).reset_index(drop=True)

    V = {"string_v11": links(raw_path("string_links"), raw_path("string_info"))}
    for name, (lk, inf) in LINKS.items():
        V[name] = links(VER / lk, VER / inf)
    M_ver = int((V["string_v120"].score >= CFG["string_min_score"]).sum())
    for name, d in V.items():
        n700 = int((d.score >= CFG["string_min_score"]).sum())
        print(f"{name}: {n700} edges at combined >= 700 inside the expressed universe (pre-LCC)")
        if name != "string_v11":
            jobs.append((name, d[d.score >= CFG["string_min_score"]][["a", "b"]], expressed))
        jobs.append((f"{name}_m", d.head(M_ver)[["a", "b"]], expressed))
        print(f"   density-matched {name}_m: top {M_ver} edges, score >= {int(d.head(M_ver).score.min())}")

    # --- B. channels of v11.0, edge-count matched
    p2s = pid_map(raw_path("string_info"), alias)
    det = pd.read_csv(raw_path("string_detailed"), sep=" ",
                      usecols=["protein1", "protein2", "experimental", "database", "textmining", "coexpression", "combined_score"])
    det["a"], det["b"] = det.protein1.map(p2s), det.protein2.map(p2s)
    det = det.dropna(subset=["a", "b"])
    det = det[det.a.isin(expressed) & det.b.isin(expressed) & (det.a != det.b)]
    det[["a", "b"]] = np.sort(det[["a", "b"]].values, axis=1)
    det = det.drop_duplicates(["a", "b"])
    M = int((det.combined_score >= CFG["string_min_score"]).sum())
    print(f"edges of combined>=700 inside the expressed universe (pre-LCC): M = {M}")
    thr = {}
    for name, col in CHANNELS.items():
        top = det.sort_values(col, ascending=False, kind="stable").head(M)
        thr[name] = int(top[col].min())
        print(f"  {name}: top {M} edges -> channel score >= {thr[name]} (max possible 999); zero-score edges included: {(top[col] == 0).sum()}")
        jobs.append((name, top[["a", "b"]].reset_index(drop=True), expressed))
    pd.DataFrame({"network": list(thr), "channel_score_threshold": list(thr.values()), "edges_requested": M}).to_csv(
        PROC / "panel_channel_thresholds.csv", index=False)

    with Pool(min(args.procs, len(jobs))) as pool:
        rows = pool.map(build_one, jobs)
    rows.insert(0, dict(network="string_v11", genes=pd.read_csv(PROC / "features_string_v11.csv", usecols=[0]).shape[0],
                        edges=pd.read_csv(PROC / "edges_string_v11.csv").shape[0], mean_degree=np.nan))
    P = pd.DataFrame(rows)
    P.to_csv(PROC / "panel_networks.csv", index=False)
    print(P.to_string(index=False))

    idx = {n: set(pd.read_csv(PROC / f"features_{n}.csv", usecols=[0]).iloc[:, 0]) for n in P.network}
    sv = set.intersection(*(idx[n] for n in ("string_v11", "string_v115", "string_v120", "string_v11_m", "string_v115_m", "string_v120_m")))
    sa = set.intersection(*(idx[n] for n in ["string_v11"] + list(CHANNELS)))   # channels are compared on string_v11 + 4 channels
    pd.DataFrame({"gene": sorted(sv)}).to_csv(PROC / "shared_versions.csv", index=False)
    pd.DataFrame({"gene": sorted(sa)}).to_csv(PROC / "shared_all.csv", index=False)
    print(f"shared gene sets: versions {len(sv)}, all networks {len(sa)}")


if __name__ == "__main__":
    main()
