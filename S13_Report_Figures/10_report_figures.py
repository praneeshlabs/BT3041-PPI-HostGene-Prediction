"""Report figures F1-F9 and table T1, built ONLY from the Results folders of steps S04-S12 (no numbers typed by hand).

F1 response (RQ1)           F2 arms vs baselines (RQ2)      F3 forest of paired differences (RQ3)
F4 rewiring null (RQ2)      F5 matched benchmark/P@k/terciles (copy of fig 15)
F6 leakage curve (copy of fig 13)   F7 paired deltas across vintages and channels   F8 channel panel (copy of fig 14)
F9 extensions: v10.5 anchor, BioPlex control, restart / cut-off / DE sensitivity (from 11_extensions.py)
T1 main results table -> Results/T1_main_results.csv / .md
Usage: python 10_report_figures.py
"""
import shutil
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from pathlib import Path

# ============================================================== helpers
# Paths, settings and functions this step needs (written out here so the script runs on its own).

# --------------------------------------------------------------------------- paths
# Data/{raw,labels,processed} sit in the project root, one level above this step's folder.
PROJECT = Path(__file__).resolve().parents[1]
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


def savefig(fig, name: str):
    FIG.mkdir(parents=True, exist_ok=True)   # created only by steps that draw figures
    out = FIG / name
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  saved figure {out.with_suffix('.png').name}")

for _d in (RESULTS,):
    _d.mkdir(parents=True, exist_ok=True)
# ============================================================== end of helpers

warnings.filterwarnings("ignore")
M = step_results("models") / "models"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"gordon": "#1f77b4", "crispr": "#d95f02"}
LAB = {"gordon": "Gordon AP-MS", "crispr": "CRISPR (6 screens)"}


def summ(tag, scheme="complex_group"):
    s = pd.read_csv(M / f"summary__{tag}.csv")
    return s[s.scheme == scheme].set_index("name")


def comps(tag, scheme="complex_group"):
    c = pd.read_csv(M / f"comparisons__{tag}.csv")
    return c[(c.scheme == scheme) & (c.model == "LR") & (c.bootstrap_unit == "group")].set_index("comparison")


def f1():
    lt = pd.read_csv(step_results("network") / "label_tests_string_v11.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.1, 1, 1.4]})
    # a) DE enrichment: observed vs expected upregulated binders
    rows = []
    for lab in ("gordon", "crispr"):
        for line in ("Calu3", "A549-ACE2"):
            r = lt[(lt.labels == lab) & (lt.test == f"DE enrichment {line}")].iloc[0]
            rows.append((lab, line, r.k, r.expected, r.p_depletion))
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r[3] for r in rows], 0.4, color="lightgrey", label="expected by chance")
    ax[0].bar(x + 0.2, [r[2] for r in rows], 0.4, color=[C[r[0]] for r in rows], label="observed")
    for i, r in enumerate(rows):
        ax[0].text(i + 0.2, r[2] + 0.8, f"p={r[4]:.3g}", ha="center", fontsize=6.5)
    ax[0].set_xticks(x, [f"{LAB[r[0]].split()[0]}\n{r[1]}" for r in rows], fontsize=7)
    ax[0].set_ylabel("host factors among upregulated genes"); ax[0].set_title("a  Host factors are not induced\n(lower-tail p = depletion)", fontsize=9)
    ax[0].set_ylim(0, 54); ax[0].legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    # b) NES
    nes = [(lab, line, lt[(lt.labels == lab) & (lt.test == f"GSEA prerank {line}")].NES.iloc[0]) for lab in ("gordon", "crispr") for line in ("Calu3", "A549-ACE2")]
    ax[1].barh(range(4), [n[2] for n in nes], color=[C[n[0]] for n in nes])
    ax[1].axvline(0, c="k", lw=0.6)
    ax[1].set_yticks(range(4), [f"{LAB[n[0]].split()[0]} {n[1]}" for n in nes], fontsize=7); ax[1].invert_yaxis()
    ax[1].set_xlabel("GSEA NES (negative = host factors rank low)"); ax[1].set_title("b  Prerank GSEA", fontsize=9)
    # c) what the E arm is made of
    s = summ("string_v11__gordon")
    items = [("random", "base:random"), ("|log2FC| ranking", "base:abs_lfc"), ("response only (E_resp)", "LR:E_resp"),
             ("abundance only", "base:logbm"), ("expression (E)", "LR:E")]
    y = np.arange(len(items))
    ax[2].barh(y, [s.loc[k].AUPRC for _, k in items], color=["lightgrey", "lightgrey", "#9ecae1", "#9ecae1", C["gordon"]])
    ax[2].errorbar([s.loc[k].AUPRC for _, k in items], y, xerr=[[s.loc[k].AUPRC - s.loc[k].ci_group_lo for _, k in items], [s.loc[k].ci_group_hi - s.loc[k].AUPRC for _, k in items]],
                   fmt="none", c="k", capsize=2, lw=0.8)
    ax[2].axvline(s.loc["base:random"].prevalence, c="k", ls=":", lw=0.8)
    ax[2].set_yticks(y, [i[0] for i in items], fontsize=7); ax[2].invert_yaxis()
    ax[2].set_xlabel("AUPRC, complex-group CV (Gordon)"); ax[2].set_title("c  The E arm is abundance, not response", fontsize=9)
    fig.tight_layout()
    savefig(fig, "F1_response")


def f2():
    order = [("random", "base:random"), ("degree", "base:log_degree"), ("pubs (pre-2020)", "base:log_pubs_pre2020"), ("abundance", "base:logbm"),
             ("N", "LR:N"), ("E", "LR:E"), ("E+N", "LR:E+N"), ("nuisance only", "LR:nuis"), ("GBA", "gba")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, lab in zip(axes, ("gordon", "crispr")):
        s = summ(f"string_v11__{lab}")
        y = np.arange(len(order))
        v = np.array([s.loc[k].AUPRC for _, k in order])
        ax.barh(y, v, color=[C[lab] if k in ("LR:N", "LR:E", "LR:E+N", "gba") else "lightgrey" for _, k in order])
        ax.errorbar(v, y, xerr=[v - [s.loc[k].ci_group_lo for _, k in order], [s.loc[k].ci_group_hi for _, k in order] - v], fmt="none", c="k", capsize=2, lw=0.8)
        ax.axvline(s.iloc[0].prevalence, c="k", ls=":", lw=0.8)
        ax.set_yticks(y, [o[0] for o in order]); ax.invert_yaxis()
        ax.set_xlabel("AUPRC (complex-group CV, 95% cluster bootstrap)")
        ax.set_title(f"{LAB[lab]}  (prevalence {s.iloc[0].prevalence:.3f}, {int(s.iloc[0].n_pos)} positives)", fontsize=9)
    fig.tight_layout()
    savefig(fig, "F2_arms")


def f3():
    spec = [("E+N - E", "network added to expression (primary)"), ("E+N+nuis - E+nuis", "network added to nuisance model"),
            ("N - degree", "N vs degree alone"), ("N - pubs_pre2020", "N vs frozen publications"),
            ("GBA - N", "GBA vs N"), ("GBA - pubs_pre2020", "GBA vs frozen publications")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for j, lab in enumerate(("gordon", "crispr")):
        c = comps(f"string_v11__{lab}")
        for i, (k, _) in enumerate(spec):
            r = c.loc[k]
            yy = i + (j - 0.5) * 0.3
            ax.errorbar(r["diff"], yy, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o", c=C[lab], capsize=2, label=LAB[lab] if i == 0 else None)
    ax.axvspan(-0.01, 0.01, color="lightgrey", alpha=0.5, lw=0, label="equivalence band ±0.01 (post hoc)")
    ax.axvline(0, c="k", lw=0.7)
    ax.set_yticks(range(len(spec)), [s[1] for s in spec]); ax.invert_yaxis()
    ax.set_xlabel("difference in AUPRC (95% cluster-bootstrap CI), complex-group CV")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    savefig(fig, "F3_forest")


def f4():
    raw = pd.read_csv(step_results("null") / "null_rewiring_raw__string_v11__gordon.csv")
    sm = pd.read_csv(step_results("null") / "null_summary__string_v11__gordon.csv").set_index("quantity")
    qs = [("N|complex_group", "N arm"), ("gain|complex_group", "E+N − E gain"), ("GBA|complex_group", "GBA")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for a, (q, t) in zip(ax, qs):
        a.hist(raw[q], bins=25, color="lightgrey")
        a.axvline(sm.loc[q].observed, c="crimson", lw=1.5)
        a.set_title(f"{t}\np = {sm.loc[q].p_emp_one_sided:.3g}, z = {sm.loc[q].z:.1f}", fontsize=9)
        a.set_xlabel("AUPRC (200 degree-preserving rewirings)")
    ax[0].set_ylabel("rewirings")
    fig.tight_layout()
    savefig(fig, "F4_rewiring")


def f7():
    R = pd.read_csv(step_results("panel") / "panel_paired_tests.csv")
    rows = []
    for lab in ("gordon", "crispr"):
        for fam, a, b in (("fixed", "string_v120", "string_v11"), ("matched", "string_v120_m", "string_v11_m")):
            for m in ("gba", "LR:N", "gain"):
                rows.append((lab, f"v12.0 − v11.0 ({'fixed' if fam == 'fixed' else 'matched'})", m, R[(R.labels == lab) & (R.family == fam) & (R.a == a) & (R.b == b) & (R.metric == m)].iloc[0]))
        for m in ("gba", "LR:N"):
            rows.append((lab, "textmining − experiments", m, R[(R.labels == lab) & (R.family == "channels") & (R.a == "v11_textmining") & (R.b == "v11_experiments") & (R.metric == m)].iloc[0]))
    names = {"gba": "GBA", "LR:N": "N arm", "gain": "E+N − E"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    contrasts = list(dict.fromkeys(r[1] for r in rows))
    ylabels = [(c, m) for c in contrasts for m in ("gba", "LR:N", "gain") if any(r[1] == c and r[2] == m for r in rows)]
    for ax, lab in zip(axes, ("gordon", "crispr")):
        for i, (c, m) in enumerate(ylabels):
            r = [x for x in rows if x[0] == lab and x[1] == c and x[2] == m][0][3]
            ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o", c=C[lab] if m == "gba" else "grey", capsize=2)
            ax.text(ax.get_xlim()[1] if False else 0.0, i, "", fontsize=6)
            ax.annotate(f"q={r.q_BH:.2g}", (r.ci_hi, i), xytext=(3, -2), textcoords="offset points", fontsize=6)
        ax.axvline(0, c="k", lw=0.7)
        ax.set_yticks(range(len(ylabels)), [f"{c}: {names[m]}" for c, m in ylabels], fontsize=7); ax.invert_yaxis()
        ax.xaxis.set_major_locator(plt.MaxNLocator(5)); ax.set_xlabel("paired AUPRC difference (95% cluster-bootstrap CI)")
        ax.set_title(LAB[lab], fontsize=9)
    fig.suptitle("Across STRING vintages only GBA shifts consistently; the N arm never does (uncontrolled: no pre-2019 control)", fontsize=9, y=1.0)
    fig.tight_layout()
    savefig(fig, "F7_paired_deltas")


def t1():
    rows = []
    for lab in ("gordon", "crispr"):
        s, c = summ(f"string_v11__{lab}"), comps(f"string_v11__{lab}")
        for nm, k in (("random", "base:random"), ("degree", "base:log_degree"), ("frozen pubs", "base:log_pubs_pre2020"), ("abundance", "base:logbm"),
                      ("E", "LR:E"), ("N", "LR:N"), ("E+N", "LR:E+N"), ("nuisance only", "LR:nuis"), ("GBA", "gba")):
            r = s.loc[k]
            rows.append(dict(labels=lab, row=nm, AUPRC=r.AUPRC, CI=f"{r.ci_group_lo:.3f}–{r.ci_group_hi:.3f}", lift=r.lift, AUROC=r.AUROC, P_at_100=r.P_at_100))
        for k in ("E+N - E", "E+N+nuis - E+nuis", "N - pubs_pre2020", "GBA - pubs_pre2020"):
            r = c.loc[k]
            rows.append(dict(labels=lab, row=f"Δ {k}", AUPRC=r["diff"], CI=f"{r.ci_lo:.4f}–{r.ci_hi:.4f}", lift=np.nan, AUROC=np.nan, P_at_100=np.nan, q_BH=r.q_BH))
    T = pd.DataFrame(rows)
    T.round(4).to_csv(RESULTS / "T1_main_results.csv", index=False)
    with open(RESULTS / "T1_main_results.md", "w", encoding="utf-8") as f:
        Tm = T.round(3).fillna("").astype(str)
        f.write("| " + " | ".join(Tm.columns) + " |\n|" + "---|" * len(Tm.columns) + "\n")
        for _, r in Tm.iterrows():
            f.write("| " + " | ".join(r) + " |\n")
    print("  wrote T1_main_results.csv / .md")


def f9():
    """Extensions (11_extensions.py): v10.5 anchor, BioPlex assay control, restart / cut-off / DE sensitivity."""
    S = pd.read_csv(step_results("extensions") / "ext_summary.csv")
    DE = pd.read_csv(step_results("extensions") / "ext_de_sensitivity.csv")
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))

    def line(a, part, order, xl, title):
        for lab in ("gordon", "crispr"):
            t = S[(S.part == part) & (S.labels == lab)].set_index("network").loc[order]
            a.errorbar(range(len(order)), t.GBA_AUPRC, yerr=[t.GBA_AUPRC - t.GBA_ci_lo, t.GBA_ci_hi - t.GBA_AUPRC],
                       marker="o", c=C[lab], capsize=3, label=LAB[lab])
            a.axhline(t.prevalence.iloc[0], c=C[lab], ls=":", lw=0.8)
        a.set_xticks(range(len(order)), xl, fontsize=7.5)
        a.set_ylabel("GBA AUPRC (complex-group CV)"); a.set_title(title, fontsize=9)

    vers = ["v10.5", "v11.0", "v11.5", "v12.0"]
    line(ax[0, 0], "A_vintage", vers, ["v10.5\n(2017)", "v11.0\n(2019)", "v11.5\n(2021)", "v12.0\n(2023)"],
         "a  STRING vintage incl. pre-2020 v10.5\n(identical genes and folds; dotted = prevalence)")
    ax[0, 0].legend(frameon=False, fontsize=7)
    for lab in ("gordon", "crispr"):
        t = S[(S.part == "A_vintage") & (S.labels == lab)].set_index("network").loc[vers]
        ax[0, 1].plot(range(4), t.pos_pos_fold, marker="s", c=C[lab], label=LAB[lab])
    ax[0, 1].axhline(1, c="k", lw=0.5)
    ax[0, 1].set_xticks(range(4), ["v10.5", "v11.0", "v11.5", "v12.0"]); ax[0, 1].set_ylabel("positive-positive edges, obs / degree-matched exp")
    ax[0, 1].set_title("b  Labelled genes wired to each other", fontsize=9); ax[0, 1].legend(frameon=False, fontsize=7)
    # c) BioPlex controls
    rows = [("B1_bioplex_cell", "BioPlex-293T", "BioPlex\n293T"), ("B1_bioplex_cell", "BioPlex-HCT116", "BioPlex\nHCT116"),
            ("B2_remove_bioplex", "STRING v11 (full)", "STRING\nfull"), ("B2_remove_bioplex", "minus BioPlex-293T edges", "STRING −\n293T edges"),
            ("B2_remove_bioplex", "minus random edges (mean of 5)", "STRING −\nrandom edges")]
    w = 0.38
    for j, lab in enumerate(("gordon", "crispr")):
        v = [S[(S.part == p) & (S.labels == lab) & (S.network == n)].iloc[0] for p, n, _ in rows]
        x = np.arange(len(rows)) + (j - 0.5) * w
        ax[0, 2].bar(x, [r.GBA_AUPRC for r in v], w, color=C[lab], label=LAB[lab])
        ax[0, 2].errorbar(x, [r.GBA_AUPRC for r in v], yerr=[[r.GBA_AUPRC - r.GBA_ci_lo for r in v], [r.GBA_ci_hi - r.GBA_AUPRC for r in v]],
                          fmt="none", c="k", capsize=2, lw=0.7)
    ax[0, 2].axvline(1.5, c="grey", lw=0.6, ls="--")
    ax[0, 2].set_xticks(range(len(rows)), [r[2] for r in rows], fontsize=7)
    ax[0, 2].set_ylabel("GBA AUPRC"); ax[0, 2].set_title("c  Assay circularity: same-cell AP-MS gives no edge;\nremoving HEK293T edges = removing random edges", fontsize=9)
    ax[0, 2].legend(frameon=False, fontsize=7)
    line(ax[1, 0], "C1_restart", [f"r={r}" for r in (0.15, 0.3, 0.5, 0.7, 0.85)], ["0.15", "0.3", "0.5*", "0.7", "0.85"],
         "d  RWR restart probability (* = used)")
    line(ax[1, 1], "C2_cutoff", [f"cut={c}" for c in (500, 600, 700, 800, 900)], ["500", "600", "700*", "800", "900"],
         "e  STRING combined-score cut-off (* = used)")
    # f) DE sensitivity: enrichment ratio of Gordon binders among up / down genes, all 9 cut-offs
    g = DE[DE.labels == "gordon"].copy()
    g["setting"] = g.padj.astype(str) + " / " + g.lfc.astype(str)
    for cell, mk in (("Calu3", "o"), ("A549-ACE2", "^")):
        for d, col in (("up", "#b2182b"), ("down", "#2166ac")):
            t = g[(g.cell == cell) & (g.direction == d)]
            ax[1, 2].plot(range(len(t)), t.ratio, marker=mk, c=col, ls="-" if cell == "Calu3" else "--", lw=0.8, label=f"{cell} {d}")
    ax[1, 2].axhline(1, c="k", lw=0.6)
    ax[1, 2].set_xticks(range(9), g[(g.cell == "Calu3") & (g.direction == "up")].setting, fontsize=6.5, rotation=45, ha="right")
    ax[1, 2].set_xlabel("padj cut-off / |log2FC| cut-off", fontsize=8)
    ax[1, 2].set_ylabel("observed / expected Gordon binders"); ax[1, 2].legend(frameon=False, fontsize=6.5, ncol=2)
    ax[1, 2].set_title("f  DE cut-off sweep: binders depleted among UP,\nenriched among DOWN genes", fontsize=9)
    fig.tight_layout()
    savefig(fig, "F9_extensions")


def main():
    for fn in (f1, f2, f3, f4, f7, f9, t1):
        fn()
    for step, src, dst in (("interpret", "15_matched_and_terciles", "F5_matched_terciles"), ("panel", "13_leakage_curve", "F6_leakage"),
                           ("panel", "14_channel_panel", "F8_channels")):
        for ext in (".png", ".pdf"):
            shutil.copy(step_results(step) / "figures" / f"{src}{ext}", FIG / f"{dst}{ext}")
    print("  copied F5, F6, F8")


if __name__ == "__main__":
    main()
