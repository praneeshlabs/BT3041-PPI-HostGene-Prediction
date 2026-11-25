"""Step S6 - leakage dose-response and channel decomposition (RQ4), the project's headline analysis.

For every network of the panel (built by 06a_build_panel.py) the identical pipeline (04_models.py:
same arms, same folds, same labels, same genes) is run on the SHARED gene set, then collected into

  leakage_curve.csv         AUPRC / lift of N, E+N, GBA (+ pubs baseline) per STRING vintage, per label set
  channel_decomposition.csv per evidence channel: AUPRC, lift, topology~fame correlation, positive-positive
                            edge enrichment
  positive_edge_enrichment.csv   observed vs degree-matched expected number of edges BETWEEN positives
                            (the most direct leakage diagnostic: a post-2019 network that has absorbed the
                            answer will wire the labelled genes to each other far more than v11.0 does)
and figures 13 (leakage curve) and 14 (channel panel).

Falsification control: the same panel is evaluated against a PRE-2019 label set (HIV host-dependency
factors) if data/labels/hiv_control.csv exists. A rise with STRING vintage that also appears for the HIV
labels is generic database growth, not COVID leakage.
Usage: python 06b_panel_analysis.py [--labels gordon [hiv_control ...]] [--B 1000]
"""
import argparse
import os
import subprocess
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

import pickle
from pathlib import Path

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


def graph_matrices(g):
    n = g.vcount()
    el = np.array(g.get_edgelist())
    A = sp.coo_matrix((np.ones(len(el)), (el[:, 0], el[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    W = (A @ sp.diags(1.0 / np.maximum(deg, 1))).tocsr()  # column-stochastic
    return A, W, deg

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")
MODELS = RESULTS / "models"
VERSIONS = ["string_v11", "string_v115", "string_v120"]
VERSIONS_M = ["string_v11_m", "string_v115_m", "string_v120_m"]   # density-matched (same number of edges)
CHANNELS = ["v11_experiments", "v11_database", "v11_textmining", "v11_coexpression"]
NICE = {"string_v11": "v11.0 (2019)", "string_v115": "v11.5 (2021)", "string_v120": "v12.0 (2023)",
        "v11_experiments": "experiments", "v11_database": "database", "v11_textmining": "textmining",
        "v11_coexpression": "coexpression"}


def run_models(net, labels, subset, tag, B):
    out = MODELS / f"comparisons__{tag}.csv"
    if out.exists():
        return
    cmd = [sys.executable, str(step_dir("models") / "04_models.py"), "--ppi", net, "--labels", labels, "--subset", str(PROC / subset),
           "--tag", tag, "--schemes", "gene", "complex_group", "--no_rf", "--B", str(B)]
    print("  running", " ".join(cmd[2:]), flush=True)
    env = dict(os.environ, HOSTNET_RESULTS=str(RESULTS))   # panel models are this step's results
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stderr[-1500:])
        raise SystemExit(f"04_models failed for {tag}")


def pos_edge_enrichment(net, pos, shared, n_perm=1000, seed=42):
    F, g = load_network(net)
    A, _, deg = graph_matrices(g)
    keep = np.where(F.index.isin(shared))[0]
    names = np.array(F.index)
    A = A.tocsr()
    P = np.where(np.isin(names, list(pos)) & np.isin(names, list(shared)))[0]
    obs = A[P][:, P].sum() / 2
    rng = np.random.default_rng(seed)
    bins = pd.qcut(pd.Series(deg[keep]).rank(method="first"), 20, labels=False).values
    by = {b: keep[bins == b] for b in np.unique(bins)}
    bin_of = dict(zip(keep, bins))
    null = np.array([A[(S := np.array([rng.choice(by[bin_of[v]]) for v in P]))][:, S].sum() / 2 for _ in range(n_perm)])
    return dict(n_pos=len(P), pos_pos_edges=obs, null_mean=null.mean(), null_sd=null.std(ddof=1),
                fold=obs / max(null.mean(), 1e-9), z=(obs - null.mean()) / null.std(ddof=1),
                p_emp=(1 + (null >= obs).sum()) / (n_perm + 1))


def topology_fame(net, shared, pos):
    F = pd.read_csv(PROC / f"features_{net}.csv", index_col=0).loc[lambda d: d.index.isin(shared)]
    y = F.index.isin(pos)
    # partial Spearman of degree with fame controlling for abundance, via rank residuals
    r = F.rank()
    def resid(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)
    part = spearmanr(resid(r.log_degree, r[f"logbm__{CFG['main_cell']}"]), resid(r.log_pubs_pre2020, r[f"logbm__{CFG['main_cell']}"]))[0]
    return dict(rho_degree_pubs=spearmanr(F.log_degree, F.log_pubs_pre2020)[0], partial_rho_degree_pubs_given_abundance=part,
                n_genes=len(F), mean_degree=float(np.expm1(F.log_degree).mean()))


def collect(net, labels, tag):
    S = pd.read_csv(MODELS / f"summary__{tag}.csv")
    C = pd.read_csv(MODELS / f"comparisons__{tag}.csv")
    return S, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["gordon"])
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    alias, _, _ = symbol_maps()
    shared_all = set(pd.read_csv(PROC / "shared_all.csv").gene)
    shared_ver = set(pd.read_csv(PROC / "shared_versions.csv").gene)
    rows, comps, pe_rows, ch_rows = [], [], [], []
    for lab in args.labels:
        pos = load_labels(lab, alias)
        for net in VERSIONS + VERSIONS_M + CHANNELS:
            # every network is evaluated ONLY on the genes shared by its comparison group:
            #   vintages (fixed + density-matched)  -> shared_versions
            #   v11.0 combined + the 4 channels      -> shared_all
            scopes = []
            if net in VERSIONS + VERSIONS_M:
                scopes.append(("shared_versions", "shared_versions.csv", shared_ver))
            if net in CHANNELS + ["string_v11"]:
                scopes.append(("shared_all", "shared_all.csv", shared_all))
            for scope, subset, shared in scopes:
                tag = f"panel_{net}__{lab}__{scope}"
                run_models(net, lab, subset, tag, args.B)
                S, C = collect(net, lab, tag)
                S["network"], S["scope"], S["labels_set"] = net, scope, lab
                C["network"], C["scope"], C["labels_set"] = net, scope, lab
                rows.append(S)
                comps.append(C)
                pe = pos_edge_enrichment(net, pos, shared)
                pe_rows.append(dict(network=net, labels=lab, scope=scope, **pe))
                if lab == args.labels[0]:
                    ch_rows.append(dict(network=net, scope=scope, **topology_fame(net, shared, pos)))
    S = pd.concat(rows)
    C = pd.concat(comps)
    PE = pd.DataFrame(pe_rows)
    PE.to_csv(RESULTS / "positive_edge_enrichment.csv", index=False)
    S.to_csv(RESULTS / "panel_summary_all.csv", index=False)
    C.to_csv(RESULTS / "panel_comparisons_all.csv", index=False)

    pick = S[S.name.isin(["LR:N", "LR:E+N", "LR:E+N+nuis", "LR:E+nuis", "LR:E", "gba", "base:log_pubs_pre2020", "base:log_degree"])]
    leak = pick[pick.network.isin(VERSIONS + VERSIONS_M)]
    leak.to_csv(RESULTS / "leakage_curve.csv", index=False)
    chan = pick[pick.network.isin(CHANNELS + ["string_v11"]) & (pick.scope == "shared_all")].merge(
        pd.DataFrame(ch_rows), on=["network", "scope"], how="left").merge(
        PE[PE.labels == args.labels[0]][["network", "scope", "fold", "z"]].rename(columns={"fold": "pos_pos_edge_fold", "z": "pos_pos_edge_z"}),
        on=["network", "scope"], how="left")
    chan.to_csv(RESULTS / "channel_decomposition.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nPositive-positive edge enrichment (observed / degree-matched expected):")
    print(PE.round(3).to_string(index=False))
    for lab in args.labels:
        for vs, title in ((VERSIONS, "fixed threshold >= 700"), (VERSIONS_M, "density-matched")):
            t = leak[(leak.labels_set == lab) & (leak.scheme == "complex_group") & (leak.scope == "shared_versions") & leak.network.isin(vs)]
            print(f"\nLeakage curve, {lab}, {title}, complex_group CV, shared genes (AUPRC):")
            print(t.pivot(index="name", columns="network", values="AUPRC")[vs].round(4).to_string())

    # ---- figures
    fig, axes = plt.subplots(len(args.labels), 2, figsize=(10.5, 4 * len(args.labels)), squeeze=False)
    for r, lab in enumerate(args.labels):
        for ax, vs, title in ((axes[r, 0], VERSIONS, "fixed threshold >= 700"), (axes[r, 1], VERSIONS_M, "density-matched (same #edges)")):
            for nm, col, mk in (("gba", "tab:red", "o"), ("LR:N", "tab:blue", "s"), ("LR:E+N", "tab:green", "^"), ("base:log_pubs_pre2020", "grey", "x")):
                t = leak[(leak.labels_set == lab) & (leak.scheme == "complex_group") & (leak.scope == "shared_versions") & (leak.name == nm)].set_index("network").loc[vs]
                ax.errorbar(range(3), t.AUPRC, yerr=[t.AUPRC - t.ci_group_lo, t.ci_group_hi - t.AUPRC], marker=mk, c=col, capsize=3,
                            label=nm.replace("base:", "").replace("LR:", ""))
            ax.axhline(leak[leak.labels_set == lab].prevalence.iloc[0], c="k", ls=":", lw=0.8)
            ax.set_xticks(range(3), ["v11.0 (2019)", "v11.5 (2021)", "v12.0 (2023)"])
            ax.set_ylabel("AUPRC (complex-group CV)"); ax.set_title(f"{lab}: {title}"); ax.legend(frameon=False, fontsize=7)
    savefig(fig, "13_leakage_curve")

    t = chan[(chan.scheme == "complex_group") & (chan.name == "LR:N") & (chan.labels_set == args.labels[0])].set_index("network").loc[["string_v11"] + CHANNELS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(range(5), t.lift); axes[0].axhline(1, c="k", lw=0.5); axes[0].set_ylabel("lift of N arm (AUPRC/prevalence)")
    axes[1].bar(range(5), t.rho_degree_pubs); axes[1].set_ylabel("Spearman(degree, pubs pre-2020)")
    axes[2].bar(range(5), t.pos_pos_edge_fold); axes[2].axhline(1, c="k", lw=0.5); axes[2].set_ylabel("positive-positive edges obs/exp")
    for ax in axes:
        ax.set_xticks(range(5), ["combined >=700"] + [NICE[c] for c in CHANNELS], fontsize=7, rotation=30, ha="right")
    fig.suptitle("Evidence channels of STRING v11.0, matched edge counts", y=1.02)
    savefig(fig, "14_channel_panel")


if __name__ == "__main__":
    main()
