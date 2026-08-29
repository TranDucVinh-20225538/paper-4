"""
Plan Section 6: complete the reporting of the analyses that stay exploratory.

Two are kept in the manuscript but under-reported, and the plan asks for both to
be given enough detail that a reader can judge them rather than take the verdict.

Kendall's tau. Configurations sharing a seed are dependent, so the pooled table
is exploratory. The manuscript describes a whole-seed permutation sensitivity
for condition number only; the plan asks for the same treatment of LID and
spectral slope, so the reporting is symmetric. The permutation shuffles which
seed's outcome vector accompanies which seed's geometry vector, which breaks the
geometry-outcome link while leaving the within-seed configuration structure
intact. Five seeds give 5! = 120 arrangements, so the p-value grid is 1/120 =
0.0083 and nothing finer can be resolved; effect size and the permutation
distribution are reported rather than a verdict.

Mixed-effects interactions. The manuscript reports three significant p-values
out of sixteen and describes the coefficients as small, which a reader cannot
check. All sixteen are reported here with coefficient, standard error, raw and
Holm-adjusted p, predictor scale and convergence status.

    python3 PhaseB/analysis/exploratory_reporting.py
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kendalltau

GEOMETRY = ["lid_mean", "spectral_decay_slope", "condition_number", "per_direction_variance_w"]
SCORERS = ["maha_auroc", "cosine_auroc", "knn10_auroc", "kde_auroc"]


def holm(pvals):
    order = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def whole_seed_permutation(df, g, s):
    """Permute which seed's outcome vector accompanies which seed's geometry."""
    seeds = sorted(df.seed.unique())
    by_seed = {k: v.sort_values("config") for k, v in df.groupby("seed")}
    obs = kendalltau(df[g], df[s]).statistic
    null = []
    for perm in itertools.permutations(seeds):
        gv = np.concatenate([by_seed[k][g].to_numpy() for k in seeds])
        yv = np.concatenate([by_seed[p][s].to_numpy() for p in perm])
        null.append(kendalltau(gv, yv).statistic)
    null = np.array(null)
    return obs, null, float((np.abs(null) >= abs(obs) - 1e-12).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PhaseB/outputs/matched_input.csv"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/exploratory"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    # the tau population is the intervention-family ladder, one outcome vector per seed
    lad = df[df.arm.isin(["Intervention", "Conventional"])].copy()
    print(f"tau population {len(lad)} configurations, {lad.seed.nunique()} seeds, "
          f"{5 * 4 * 3 * 2:d} arrangements\n")

    rows = []
    for g, s in itertools.product(GEOMETRY, SCORERS):
        obs, null, p = whole_seed_permutation(lad, g, s)
        rows.append({"geometry": g, "scorer": s, "tau": obs, "perm_p": p,
                     "null_sd": null.std(), "null_p95_abs": np.percentile(np.abs(null), 95)})
    tau = pd.DataFrame(rows)
    tau["perm_p_holm"] = holm(tau.perm_p.to_numpy())
    tau.to_csv(args.out / "kendall_whole_seed_permutation.csv", index=False)

    print("Kendall tau with whole-seed permutation, all four metrics reported alike")
    print(f"{'geometry':<26}{'scorer':<14}{'tau':>8}{'perm p':>9}{'Holm':>8}"
          f"{'null |tau| P95':>16}")
    for _, r in tau.iterrows():
        print(f"{r.geometry:<26}{r.scorer:<14}{r.tau:+8.3f}{r.perm_p:9.4f}"
              f"{r.perm_p_holm:8.4f}{r.null_p95_abs:16.3f}")
    print(f"\nfinest resolvable p is 1/120 = {1/120:.4f}; "
          f"{(tau.perm_p <= 1/120 + 1e-9).sum()} of 16 reach it\n")

    # ---- mixed-effects interactions, all sixteen ----
    d = df.copy()
    d["type_bin"] = (d.arm == "Intervention").astype(int)
    rows = []
    for g, s in itertools.product(GEOMETRY, SCORERS):
        sub = d[["seed", "type_bin", g, s]].dropna().rename(columns={g: "G", s: "Y"})
        sub["G"] = (sub.G - sub.G.mean()) / sub.G.std(ddof=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("Y ~ type_bin * G", sub, groups=sub["seed"]).fit(method="lbfgs")
        k = "type_bin:G"
        rows.append({"geometry": g, "scorer": s, "coef": m.params[k], "se": m.bse[k],
                     "p": m.pvalues[k], "converged": bool(m.converged),
                     "re_var": float(m.cov_re.iloc[0, 0]), "n": len(sub)})
    mx = pd.DataFrame(rows)
    mx["p_holm"] = holm(mx.p.to_numpy())
    mx.to_csv(args.out / "mixed_effects_interactions.csv", index=False)

    print("Mixed-effects interaction type_bin:G, standardized G, seed random intercept")
    print(f"{'geometry':<26}{'scorer':<14}{'coef':>11}{'SE':>10}{'p':>9}{'Holm':>8}  conv")
    for _, r in mx.iterrows():
        print(f"{r.geometry:<26}{r.scorer:<14}{r.coef:+11.5f}{r.se:10.5f}{r.p:9.4f}"
              f"{r.p_holm:8.4f}  {'y' if r.converged else 'N'}")
    print(f"\n{int(mx.converged.sum())}/16 converged; "
          f"{(mx.p_holm < .05).sum()} significant after Holm")
    print(f"largest |coef| {mx.coef.abs().max():.5f} AUROC per SD of the metric")
    print(f"\nwrote two tables to {args.out}")


if __name__ == "__main__":
    main()
