# Can protein interaction networks predict SARS-CoV-2 host genes?

BT3041 (Analysis and Interpretation of Biological Data) course project.

We take 10,722 genes that are expressed in Calu-3 lung cells (infected vs mock RNA-seq, GEO GSE147507) and sit in the STRING v11.0 network. We then ask whether network features can pick out known SARS-CoV-2 host genes. The host genes are the Gordon 2020 AP-MS binders and the top hits of published CRISPR screens.

## Research questions and what we found

**RQ1. Are host genes switched on (up-regulated) by infection?**
No. Host genes are fewer than expected among up-regulated genes: in Calu-3, 19 binders against 39 expected (p = 1.2e-4). This holds at all nine DE cut-offs we tried. In A549-ACE2 cells they are more common than expected among down-regulated genes.

**RQ2. Does network position predict host genes, or only the number of partners (degree)?**
Standard network features (degree, betweenness, PageRank and so on) do no better than a gene's publication count. Scrambling the wiring while keeping each gene's degree barely changes them. So they mostly measure degree and how well studied a gene is.
Guilt-by-association (GBA) works differently: a gene scores high when it is close to known host genes. GBA reaches 2 to 3 times the chance level, and scrambling the wiring removes most of its signal, so GBA depends on the real wiring.

**RQ3. Does the network add anything beyond expression, abundance and fame?**
No, not as model features. Adding network features to the expression model changes AUPRC by +0.0012 (95% CI -0.0016 to +0.0040). That is equivalent to zero within our +/-0.01 margin. GBA beats the publication-count baseline by +0.025 for Gordon binders and +0.035 for CRISPR hits. It still works after matching genes on abundance and fame, and after the BioPlex control for assay overlap.

**RQ4. How much of the network signal comes from the literature (STRING evidence type and release date)?**
GBA gets better in newer STRING releases (Gordon AUPRC 0.054 in v11.0, 0.064 in v11.5, 0.072 in v12.0). The older v10.5 to v11.0 step changed nothing. This fits newer releases having absorbed COVID-19 papers, but we cannot prove it. Text-mined edges are the most tied to publication count, but the differences between evidence channels are not significant after correction.

## What each step does

| Step | What it does |
|---|---|
| `S01_Data_Download` | Downloads the public data files (RNA-seq counts, STRING v11.0, HuRI, gene info, publication counts). |
| `S02_Label_Sets` | Builds the host-gene lists (Gordon binders, CRISPR hits) from the papers' supplementary tables. |
| `S03_Differential_Expression` | Finds genes that change with infection in each cell line (PyDESeq2), with RNA-seq quality checks. |
| `S04_Network_Features` | Computes network features for each gene and tests whether host genes are up- or down-regulated. |
| `S05_Communities_And_CV_Groups` | Groups genes into protein complexes and network communities, used to keep related genes together in cross-validation. |
| `S06_Prediction_Models` | Trains and compares the expression, network and combined models, the baselines and GBA. |
| `S07_Rewiring_Null` | Scrambles the network 200 times (keeping each gene's degree) to see which signals need the real wiring. |
| `S08_STRING_Versions_And_Channels` | Repeats the models on STRING v11.5 and v12.0 and on each STRING evidence type. |
| `S09_Transfer` | Checks whether a model trained on one cell line or label set works on another. |
| `S10_Interpretation` | Matches genes on abundance and fame, lists top-ranked genes and explains the models. |
| `S11_HuRI_Sparsity_Control` | Checks whether HuRI does worse only because it has fewer edges than STRING. |
| `S12_Extensions_v105_BioPlex_Sensitivity` | Adds STRING v10.5, the BioPlex assay control and parameter sweeps. |
| `S13_Report_Figures` | Makes the report figures and the main results table from the earlier steps' outputs. |

Each step folder holds its script(s) and a `Results/` folder with the outputs we got.

## How to run

```bash
pip install -r requirements.txt
bash run_all.sh                                   # all steps in order (about 6-7 h on a 12-core laptop)
python S06_Prediction_Models/04_models.py --ppi string_v11 --labels gordon    # or one step at a time
```

- Every script runs on its own. There is no shared module to import.
- Run the steps in order the first time, because later steps read earlier steps' outputs. `run_all.sh` lists the order and the arguments.
- Run the commands from this folder. Each step writes to its own `Results/` folder, and intermediate tables go to `Data/processed/`.
- All random steps use seed 42.

## Data

- `S01_Data_Download` fetches most input files into `Data/raw/`.
- Already included: the label tables (`Data/labels/`), CORUM complexes, BioPlex networks and the housekeeping gene list (`Data/raw/`).
- Download by hand into `Data/raw/string_versions/` (too large for this repository): `9606.protein.links.v11.5.txt.gz`, `9606.protein.info.v11.5.txt.gz`, `9606.protein.links.v12.0.txt.gz`, `9606.protein.info.v12.0.txt.gz`, `9606.protein.links.v10.5.txt.gz` and `9606.protein.aliases.v10.5.txt.gz`. These come from STRING. We took v10.5 from an Internet Archive copy, because the original v10.5 site blocks scripted downloads.
