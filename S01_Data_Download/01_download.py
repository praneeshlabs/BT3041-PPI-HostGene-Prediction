"""Step 1 - download public data (expression, PPI, gene info, publication counts).

Usage:
    python 01_download.py                 # core files (counts, STRING v11 combined, gene info, pubs)
    python 01_download.py --extra huri    # + HuRI robustness network
    python 01_download.py --extra string_exp   # + STRING detailed (experimental channel)
Labels (Gordon 2020, CRISPR screens) and BioGRID must be fetched by hand - see README.
"""
import argparse
import sys

import requests
from tqdm import tqdm

import os
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

for _d in (RAW, LABELS, PROC):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

CORE = ["counts", "string_links", "string_info", "gene_info", "gene2pubmed"]
EXTRA = {"huri": ["huri"], "string_exp": ["string_detailed"]}


def fetch(key):
    out = raw_path(key)
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} already present")
        return
    url = URLS[key]
    print(f"[get ] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, leave=False) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], choices=list(EXTRA))
    args = ap.parse_args()
    keys = CORE + [k for e in args.extra for k in EXTRA[e]]
    failed = []
    for k in keys:
        try:
            fetch(k)
        except Exception as e:  # keep going, report at the end
            print(f"[fail] {k}: {e}")
            failed.append(k)
    print("\nManual steps (see README):")
    print(f"  * put label tables in {LABELS}/ as gordon.csv and crispr.csv (one column 'gene')")
    print(f"  * optional: BioGRID 'BIOGRID-ORGANISM-Homo_sapiens-*.tab3.txt' into {RAW}/")
    if failed:
        print(f"\nFailed downloads: {failed} - download those URLs in a browser into {RAW}/")
        sys.exit(1)


if __name__ == "__main__":
    main()
