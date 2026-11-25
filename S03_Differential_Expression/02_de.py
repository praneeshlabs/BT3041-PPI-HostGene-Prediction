"""Step 2 - expression EDA + differential expression (PyDESeq2) per cell line.

Outputs: data/processed/de_<cell>.csv, data/processed/samples.csv, EDA figures.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pickle
import re
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


def cell_key(name) -> str:
    """'Calu-3', 'Calu3', 'calu_3' -> 'calu3' so names match robustly."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


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

for _d in (RAW, LABELS, PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- samples
def parse_samples(columns):
    """GSE147507 sample names look like 'Series7_Calu3_Mock_1' or
    'Series6_A549-ACE2_SARS-CoV-2_2'. Treated arms (e.g. '_Rux_') are excluded."""
    rows = []
    for c in columns:
        t = c.split("_")
        if len(t) != 4:
            continue
        series, cell, cond, rep = t
        cl = cell_key(cond)
        if cl == "mock":
            cond2 = "mock"
        elif cl in ("sarscov2", "scov2", "cov2"):
            cond2 = "sars2"
        else:
            continue
        rows.append(dict(sample=c, series=series, cell=cell, condition=cond2, rep=rep))
    return pd.DataFrame(rows).set_index("sample")


def select_samples(meta):
    wanted = {cell_key(c): c for c in CFG["cells"]}
    meta = meta[meta.cell.map(cell_key).isin(wanted)].copy()
    meta["cell_cfg"] = meta.cell.map(lambda c: wanted[cell_key(c)])
    # keep only series that contain BOTH mock and infected samples
    ok = meta.groupby(["cell_cfg", "series"]).condition.transform("nunique") == 2
    meta = meta[ok]
    missing = set(CFG["cells"]) - set(meta.cell_cfg)
    if missing:
        print(f"  WARNING: no usable mock+infected series for {missing}")
    print("\nSamples used (CHECK THIS against the GEO page):")
    print(meta.groupby(["cell_cfg", "series", "condition"]).size().to_string(), "\n")
    return meta


# --------------------------------------------------------------------------- DESeq2
def run_deseq(counts, meta):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    meta = meta[["condition", "series"]].copy()
    factors = (["series"] if meta.series.nunique() > 1 else []) + ["condition"]
    try:  # pydeseq2 >= 0.5 (formula API)
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~" + " + ".join(factors),
                           quiet=True)
    except TypeError:  # older pydeseq2
        dds = DeseqDataSet(counts=counts, metadata=meta, design_factors=factors,
                           ref_level=["condition", "mock"], quiet=True)
    dds.deseq2()
    ds = DeseqStats(dds, contrast=["condition", "sars2", "mock"], quiet=True)
    ds.summary()
    res = ds.results_df.copy()
    res["lfc_raw"] = res["log2FoldChange"]
    res["lfc_shrunk"] = False
    for coeff in ("condition[T.sars2]", "condition_sars2_vs_mock"):
        try:
            ds.lfc_shrink(coeff=coeff)
            res["log2FoldChange"] = ds.results_df["log2FoldChange"]
            res["lfc_shrunk"] = True
            break
        except Exception:
            continue
    if not res["lfc_shrunk"].iloc[0]:
        print("  (LFC shrinkage unavailable in this pydeseq2 version - using raw LFC)")
    return res


# --------------------------------------------------------------------------- plots
def logcpm(counts):
    return np.log2(counts.div(counts.sum(axis=1), axis=0) * 1e6 + 1)


def eda_plots(counts, meta, de):
    cells = list(dict.fromkeys(meta.cell_cfg))
    colors = dict(zip(cells, plt.cm.tab10.colors))

    # 1. library sizes
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(meta)), 3.5))
    lib = counts.sum(axis=1) / 1e6
    ax.bar(range(len(lib)), lib, color=[colors[c] for c in meta.cell_cfg],
           hatch=None, edgecolor=["k" if c == "sars2" else "none" for c in meta.condition])
    ax.set_xticks(range(len(lib)), meta.index, rotation=90, fontsize=6)
    ax.set_ylabel("Library size (M reads)")
    ax.set_title("Library sizes (black edge = SARS-CoV-2)")
    savefig(fig, "01_library_sizes")

    # 2. PCA on log-CPM, top 2000 variable genes
    from sklearn.decomposition import PCA
    L = logcpm(counts)
    L = L.loc[:, L.var().sort_values(ascending=False).index[:2000]]
    pca = PCA(2).fit(L - L.mean())
    Z = pca.transform(L - L.mean())
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for c in cells:
        for cond, mk in (("mock", "o"), ("sars2", "^")):
            m = ((meta.cell_cfg == c) & (meta.condition == cond)).values
            ax.scatter(Z[m, 0], Z[m, 1], c=[colors[c]], marker=mk, s=60, edgecolor="k",
                       label=f"{c} {cond}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("PCA (log-CPM, top 2000 variable genes)")
    savefig(fig, "02_pca")

    # 3. volcano per cell line
    fig, axes = plt.subplots(1, len(de), figsize=(4.2 * len(de), 4), squeeze=False)
    for ax, (c, res) in zip(axes[0], de.items()):
        r = res.dropna(subset=["padj"])
        sig = (r.padj < CFG["de_padj"]) & (r.log2FoldChange.abs() > CFG["de_lfc"])
        ax.scatter(r.log2FoldChange[~sig], -np.log10(r.padj[~sig]), s=3, c="lightgrey")
        ax.scatter(r.log2FoldChange[sig], -np.log10(r.padj[sig]), s=4,
                   c=np.where(r.log2FoldChange[sig] > 0, "tab:red", "tab:blue"))
        for g in r[sig].nsmallest(8, "padj").index:
            ax.annotate(g, (r.log2FoldChange[g], -np.log10(r.padj[g])), fontsize=6)
        ax.set_title(f"{c}: {sig.sum()} DE genes")
        ax.set_xlabel("log2 fold change (infected / mock)")
        ax.set_ylabel("-log10 adj. p")
    savefig(fig, "03_volcano")

    # 4. heatmap of top DE genes in the main cell line
    main = CFG["main_cell"]
    if main in de:
        r = de[main].dropna(subset=["padj"])
        top = r[r.log2FoldChange.abs() > CFG["de_lfc"]].nsmallest(40, "padj").index
        sm = meta.index[meta.cell_cfg == main]
        H = logcpm(counts.loc[sm])[top]
        H = (H - H.mean()) / (H.std() + 1e-9)
        H = H.loc[meta.loc[sm].sort_values("condition").index]
        fig, ax = plt.subplots(figsize=(4 + 0.25 * len(sm), 8))
        im = ax.imshow(H.T.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_yticks(range(len(top)), top, fontsize=6)
        ax.set_xticks(range(len(H)), [meta.condition[s] for s in H.index], rotation=90, fontsize=7)
        fig.colorbar(im, ax=ax, label="z-score (log-CPM)")
        ax.set_title(f"Top 40 DE genes, {main}")
        savefig(fig, "04_heatmap_top_de")

    # 5. DE overlap across cell lines (UpSet if available)
    sets = {c: set(r[(r.padj < CFG["de_padj"]) & (r.log2FoldChange.abs() > CFG["de_lfc"])].index)
            for c, r in de.items()}
    try:
        from upsetplot import UpSet, from_contents
        fig = plt.figure(figsize=(6, 4))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            UpSet(from_contents(sets), subset_size="count", show_counts=True).plot(fig=fig)
        fig.suptitle("Overlap of DE genes across cell lines")
        savefig(fig, "05_de_overlap_upset")
    except Exception as e:
        print(f"  (UpSet plot skipped: {e}); writing pairwise Jaccard instead")
        names = list(sets)
        J = np.array([[len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1) for b in names]
                      for a in names])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(J, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        fig.colorbar(im, label="Jaccard")
        savefig(fig, "05_de_overlap_jaccard")


# --------------------------------------------------------------------------- main
def main():
    alias, _, _ = symbol_maps()
    print("Loading counts ...")
    counts = pd.read_csv(raw_path("counts"), sep="\t", index_col=0)
    sym = harmonise(counts.index, alias)
    n0 = len(counts)
    counts.index = [s if s else None for s in sym]
    counts = counts[counts.index.notna()].groupby(level=0).sum()
    print(f"  {n0} rows -> {len(counts)} official gene symbols")

    meta = select_samples(parse_samples(counts.columns))
    counts = counts[meta.index].T.round().astype(int)  # samples x genes
    meta.to_csv(PROC / "samples.csv")

    de = {}
    for c in dict.fromkeys(meta.cell_cfg):
        m = meta[meta.cell_cfg == c]
        X = counts.loc[m.index]
        X = X.loc[:, X.sum(axis=0) >= 10]
        print(f"DESeq2: {c} ({len(m)} samples, {X.shape[1]} genes)")
        res = run_deseq(X, m)
        res.to_csv(PROC / f"de_{c}.csv")
        de[c] = res
        sig = (res.padj < CFG["de_padj"]) & (res.log2FoldChange.abs() > CFG["de_lfc"])
        print(f"  {sig.sum()} DE genes (padj<{CFG['de_padj']}, |LFC|>{CFG['de_lfc']})")
    eda_plots(counts, meta, de)


if __name__ == "__main__":
    main()
