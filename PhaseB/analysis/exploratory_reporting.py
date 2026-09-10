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
import math
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


EXHAUSTIVE_CAP = 200_000   # 5 seeds = 120, 7 seeds = 5040, 8 = 40320
N_SAMPLED = 20_000         # floor 5e-5, below Holm's .0031 at m=16


def whole_seed_permutation(df, g, s):
    """Permute which seed's outcome vector accompanies which seed's geometry."""
    seeds = sorted(df.seed.unique())
    by_seed = {k: v.sort_values("config") for k, v in df.groupby("seed")}
    obs = kendalltau(df[g], df[s]).statistic
    gv = np.concatenate([by_seed[k][g].to_numpy() for k in seeds])

    # k! is enumerable at five seeds and not at ten. Below the cap every
    # arrangement is used and the floor is exactly 1/k!; above it, arrangements
    # are sampled and the floor is 1/n_sample. Which was used is returned, so a
    # reported p is never read against the wrong floor.
    n_exact = math.factorial(len(seeds))
    if n_exact <= EXHAUSTIVE_CAP:
        arrangements = itertools.permutations(seeds)
        n_used, exhaustive = n_exact, True
    else:
        rng = np.random.default_rng(0)
        arrangements = (tuple(rng.permutation(seeds)) for _ in range(N_SAMPLED))
        n_used, exhaustive = N_SAMPLED, False

    null = np.array([kendalltau(gv, np.concatenate(
        [by_seed[p][s].to_numpy() for p in perm])).statistic
        for perm in arrangements])
    b = int((np.abs(null) >= abs(obs) - 1e-12).sum())
    # Enumeration already contains the observed arrangement, so b/k! is the
    # p-value. Sampling does not, and the naive b/n can return exactly zero;
    # the plus-one estimator (b+1)/(n+1) is the valid one and puts the floor at
    # 1/(n+1). Reporting b/n would overstate significance at the extreme.
    p = b / n_used if exhaustive else (b + 1) / (n_used + 1)
    return obs, null, float(p), n_used, exhaustive


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
        obs, null, p, n_perm, exhaustive = whole_seed_permutation(lad, g, s)
        rows.append({"geometry": g, "scorer": s, "tau": obs, "perm_p": p,
                     "null_sd": null.std(), "null_p95_abs": np.percentile(np.abs(null), 95),
                     "n_perm": n_perm, "exhaustive": exhaustive})
    tau = pd.DataFrame(rows)
    tau["perm_p_holm"] = holm(tau.perm_p.to_numpy())
    tau.to_csv(args.out / "kendall_whole_seed_permutation.csv", index=False)

    print("Kendall tau with whole-seed permutation, all four metrics reported alike")
    print(f"{'geometry':<26}{'scorer':<14}{'tau':>8}{'perm p':>9}{'Holm':>8}"
          f"{'null |tau| P95':>16}")
    for _, r in tau.iterrows():
        print(f"{r.geometry:<26}{r.scorer:<14}{r.tau:+8.3f}{r.perm_p:9.4f}"
              f"{r.perm_p_holm:8.4f}{r.null_p95_abs:16.3f}")
    floor = (1.0 / tau.n_perm.iloc[0] if tau.exhaustive.iloc[0]
             else 1.0 / (tau.n_perm.iloc[0] + 1))
    kind = "exhaustive" if tau.exhaustive.iloc[0] else "sampled"
    holm_first = 0.05 / len(tau)
    print(f"\n{len(tau)} associations, {kind} permutation over {tau.n_perm.iloc[0]:,} "
          f"arrangements at {lad.seed.nunique()} seeds")
    denom = tau.n_perm.iloc[0] + (0 if tau.exhaustive.iloc[0] else 1)
    print(f"floor 1/{denom:,} = {floor:.2e}; Holm's first threshold "
          f"{holm_first:.5f} -> {'reachable' if floor < holm_first else 'UNREACHABLE'}")
    print(f"{(tau.perm_p <= floor + 1e-12).sum()} of {len(tau)} attain the floor; "
          f"{(tau.perm_p_holm < .05).sum()} survive Holm\n")

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
