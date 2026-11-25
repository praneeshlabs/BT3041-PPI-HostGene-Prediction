"""Step 0 - build label sets from the two supplementary spreadsheets (read programmatically, never
re-saved in Excel, so no SEPT2/MARCH1 date corruption is introduced by us).

Inputs (data/labels/source/):
    gordon_2020_ST2.xlsx   Gordon et al. 2020 Nature, Supplementary Table 2 (332 high-confidence preys)
    baggen_2021_ST.xlsx    Baggen et al. 2021 Nat Genet, Supplementary Table 2 ("Top 50+ genes from the
                           different screens", harmonised by the authors across published screens)
Outputs (data/labels/):
    gordon.csv         332 AP-MS binders (PreyGene)
    crispr.csv         union of the six HUMAN-cell screens (Vero E6 excluded: non-human)
    crispr_min2.csv    genes hit in >= 2 of the six screens (stricter sensitivity set)
    source/crispr_screens.csv  long table gene x screen, with cell line
    hiv_control.csv    OPTIONAL pre-2019 control label set (see build_hiv_control; written only if
                       data/labels/source/hiv_*.csv are present)

DEFINITION (stated openly): the CRISPR rule is "top ~50 genes per screen as ranked in Baggen's Suppl.
Table 2". That is a ranking cut-off, not each paper's own significance threshold.
"""
import pandas as pd

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

for _d in (RAW, LABELS, PROC):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

SRC = LABELS / "source"
CELL = {"Baggen": "Huh7", "Daniloski": "A549-ACE2", "Wang": "Huh7.5.1-ACE2/TMPRSS2",
        "Schneider": "Huh7.5", "Zhu Sdel": "A549-ACE2", "Heaton": "A549-ACE2", "Wei Vero": "Vero E6 (monkey)"}


def main():
    alias, _, _ = symbol_maps()

    g = pd.read_excel(SRC / "gordon_2020_ST2.xlsx", header=1)
    genes = g["PreyGene"].dropna().astype(str).str.strip().drop_duplicates()
    pd.DataFrame({"gene": genes}).to_csv(LABELS / "gordon.csv", index=False)
    mapped = [x for x in harmonise(genes, alias) if x]
    print(f"gordon: {len(genes)} listed, {len(set(mapped))} map to official symbols")

    d = pd.read_excel(SRC / "baggen_2021_ST.xlsx", sheet_name="Suppl Table 2", header=None)
    t = d.iloc[3:, [1, 2]].dropna()
    t.columns = ["gene", "source"]
    t = t[t.source.isin(CELL)].astype(str)
    t["gene"] = t.gene.str.strip()
    t["cell_line"] = t.source.map(CELL)
    t = t.drop_duplicates(["gene", "source"])
    t["official"] = harmonise(t.gene, alias)
    t.to_csv(SRC / "crispr_screens.csv", index=False)  # kept in source/ so it is not picked up as a label set
    human = t[t.source != "Wei Vero"].dropna(subset=["official"])
    print("per-screen genes (human):", human.source.value_counts().to_dict())
    allg = sorted(human.official.unique())
    n_scr = human.groupby("official").source.nunique()
    pd.DataFrame({"gene": allg}).to_csv(LABELS / "crispr.csv", index=False)
    pd.DataFrame({"gene": sorted(n_scr[n_scr >= 2].index)}).to_csv(LABELS / "crispr_min2.csv", index=False)
    print(f"crispr: {len(allg)} unique genes in 6 human screens; >=2 screens: {(n_scr >= 2).sum()}")
    both = sorted(set(allg) & {x for x in mapped})
    print(f"Gordon & CRISPR overlap: {len(both)} -> {both}")


if __name__ == "__main__":
    main()
