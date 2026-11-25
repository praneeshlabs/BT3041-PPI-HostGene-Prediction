"""Step S6b - PAIRED tests behind the leakage curve and the channel decomposition.

The marginal CIs of the panel are wide (about 250 positives), so a visible rise across STRING versions
means nothing until the paired difference is tested. Every network was evaluated on the same genes with
identical folds (same labels/groups -> identical make_folds), so the out-of-fold scores in
Results/models/oof__panel_*.npz can be compared pairwise with a complex-cluster bootstrap.

  vintage tests : metric(v11.5) - metric(v11.0), metric(v12.0) - metric(v11.0), metric(v12.0) - metric(v11.5)
                  for GBA, N, E+N, and for gain = E+N - E; fixed threshold and density-matched.
  channel tests : GBA / N / E+N for every channel versus the experiments channel and versus textmining.
Output: Results/panel_paired_tests.csv.  Usage: python 06c_panel_tests.py [--labels gordon [crispr]] [--B 2000]
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import os
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


def summarize_diff(obs, boot, margin=0.01):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(boot, [5, 95])
    p = 2 * min((np.sum(boot <= 0) + 1) / (len(boot) + 1), (np.sum(boot >= 0) + 1) / (len(boot) + 1))
    return dict(diff=obs, ci_lo=lo, ci_hi=hi, p_boot=min(p, 1.0),
                equivalent_pm_margin=bool(lo90 > -margin and hi90 < margin),   # TOST at alpha=0.05
                boot_sd=float(boot.std(ddof=1)))


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
M = RESULTS / "models"
VERS = ["string_v11", "string_v115", "string_v120"]
VERS_M = ["string_v11_m", "string_v115_m", "string_v120_m"]
CH = ["v11_experiments", "v11_database", "v11_textmining", "v11_coexpression", "string_v11"]


def load(net, lab, scope):
    z = np.load(M / f"oof__panel_{net}__{lab}__{scope}.npz", allow_pickle=True)
    return z


def metric_boot(z, key, W, y):
    S = ScoreSet(z[key], y)
    return S.ap(), np.array([S.ap(w) for w in W])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["gordon", "crispr"])
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    G = pd.read_csv(PROC / "groups_string_v11.csv", index_col=0)
    rows = []
    for lab in args.labels:
        for scope, groups in (("shared_versions", [("fixed", VERS), ("matched", VERS_M)]), ("shared_all", [("channels", CH)])):
            z0 = load(groups[0][1][0], lab, scope)
            genes, y = z0["genes"], z0["y"]
            cg = G.complex_group.reindex(genes).astype(object)
            cg = cg.where(cg.notna(), pd.Series([f"s_{x}" for x in genes], index=genes))
            W = boot_weights(len(y), pd.factorize(cg)[0], args.B, "group", rng)
            for gname, nets in groups:
                Z = {n: load(n, lab, scope) for n in nets}
                for n in nets:
                    assert (Z[n]["genes"] == genes).all()
                res = {}
                for n in nets:
                    for k in ("gba", "LR:N", "LR:E+N", "LR:E"):
                        res[(n, k)] = metric_boot(Z[n], f"complex_group|{k}", W, y)
                    res[(n, "gain")] = (res[(n, "LR:E+N")][0] - res[(n, "LR:E")][0], res[(n, "LR:E+N")][1] - res[(n, "LR:E")][1])
                pairs = []
                if gname in ("fixed", "matched"):
                    a, b, c = nets
                    pairs = [(b, a), (c, a), (c, b)]
                else:
                    pairs = [(n, "v11_experiments") for n in nets if n != "v11_experiments"] + [(n, "v11_textmining") for n in nets if n not in ("v11_textmining", "v11_experiments")]
                for hi, lo in pairs:
                    for k in ("gba", "LR:N", "LR:E+N", "gain"):
                        obs = res[(hi, k)][0] - res[(lo, k)][0]
                        r = summarize_diff(obs, res[(hi, k)][1] - res[(lo, k)][1])
                        rows.append(dict(labels=lab, family=gname, a=hi, b=lo, metric=k, **r))
    R = pd.DataFrame(rows)
    R["q_BH"] = np.nan
    for _, idx in R.groupby(["labels", "family", "metric"]).groups.items():
        R.loc[idx, "q_BH"] = bh(R.loc[idx, "p_boot"].values)
    R.to_csv(RESULTS / "panel_paired_tests.csv", index=False)
    pd.set_option("display.width", 230)
    print(R[["labels", "family", "a", "b", "metric", "diff", "ci_lo", "ci_hi", "p_boot", "q_BH"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
