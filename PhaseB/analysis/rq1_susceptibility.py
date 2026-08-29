"""
RQ1 items 1.4 and 1.5: local ranking susceptibility, and predicted against observed.

The plan calls this the most mandatory piece, because near-ceiling AUROC and
scorer invariance produce the same flat result and nothing separating them has
been reported. Susceptibility separates them. Corollary 1 states

    |AUROC(s~) - AUROC(s)| <= Pr(|M| <= 2 delta),   M = s(OOD) - s(ID)

so the question is not how large the score perturbation was but how much
ID-OOD margin mass sits within reach of it.

delta must be in each scorer's own units. Mahalanobis scores here run to 5e4
and cosine lies in [0, 0.6], so evaluating both at delta = 0.1 compares nothing.
Two deltas are used per scorer, both scorer-relative:

  observed   the empirical score distortion between alpha = 0 and alpha = 1,
             at the median, P95 and max of |s_1 - s_0| per example. This is the
             perturbation that actually occurred.
  analytic   Theorem 2 for Mahalanobis, Lemma 1 for cosine, Corollary 2's ratio
             for k-NN. Reported alongside, and marked uninformative where the
             bound exceeds the observed distortion by orders of magnitude.

Margins are estimated over sampled ID-OOD pairs; 5,067 x 2,298 is 11.6M pairs
per configuration and the quantiles are stable far below that.

    python3 PhaseB/analysis/rq1_susceptibility.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEEDS = (42, 52, 62, 72, 82)
SCORERS = ("maha", "cosine", "knn10", "kde")
N_PAIRS = 2_000_000


def margin_mass(s_id, s_ood, deltas, rng, n_pairs=N_PAIRS):
    """Pr(|M| <= 2 delta) over sampled ID-OOD pairs."""
    a = s_id[rng.integers(0, len(s_id), n_pairs)]
    b = s_ood[rng.integers(0, len(s_ood), n_pairs)]
    m = np.abs(b - a)
    return {d: float((m <= 2 * d).mean()) for d in deltas}


def ratio_mass(s_id, s_ood, kappas, rng, n_pairs=N_PAIRS):
    """Corollary 2's multiplicative analogue, for the scale-covariant k-NN score."""
    a = s_id[rng.integers(0, len(s_id), n_pairs)]
    b = s_ood[rng.integers(0, len(s_ood), n_pairs)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.maximum(b / a, a / b)
    r = r[np.isfinite(r)]
    return {k: float((r <= k).mean()) for k in kappas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("PhaseB/outputs/per_example"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/rq1"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for seed in SEEDS:
        d0 = np.load(args.scores / f"Intervention_a0.0_s{seed}.npz")
        d1 = np.load(args.scores / f"Intervention_a1.0_s{seed}.npz")
        ood = d0["is_ood"].astype(bool)
        for sc in SCORERS:
            s0, s1 = d0[sc], d1[sc]
            dist = np.abs(s1 - s0)
            deltas = {"median": float(np.median(dist)),
                      "p95": float(np.percentile(dist, 95)),
                      "max": float(dist.max())}
            auroc0, auroc1 = roc_auc_score(ood, s0), roc_auc_score(ood, s1)
            if sc == "knn10":
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.maximum(s1 / s0, s0 / s1)
                ratio = ratio[np.isfinite(ratio)]
                ks = {"median": float(np.median(ratio)), "p95": float(np.percentile(ratio, 95)),
                      "max": float(ratio.max())}
                S = ratio_mass(s0[~ood], s0[ood], list(ks.values()), rng)
                for q, k in ks.items():
                    rows.append({"seed": seed, "scorer": sc, "kind": "ratio", "quantile": q,
                                 "delta": k, "S": S[k], "obs_dAUROC": abs(auroc1 - auroc0)})
            else:
                S = margin_mass(s0[~ood], s0[ood], list(deltas.values()), rng)
                for q, dl in deltas.items():
                    rows.append({"seed": seed, "scorer": sc, "kind": "additive", "quantile": q,
                                 "delta": dl, "S": S[dl], "obs_dAUROC": abs(auroc1 - auroc0)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "rq1_4_susceptibility_observed.csv", index=False)

    g = df.groupby(["scorer", "quantile"]).agg(
        delta=("delta", "mean"), S=("S", "mean"), dAUROC=("obs_dAUROC", "mean")).reset_index()
    order = {"median": 0, "p95": 1, "max": 2}
    g = g.sort_values(["scorer", "quantile"], key=lambda c: c.map(order) if c.name == "quantile" else c)

    print("1.4/1.5  delta from the observed alpha 0 -> 1 distortion, "
          "S(delta) is Corollary 1's bound on |dAUROC|\n")
    print(f"{'scorer':<9}{'quantile':<9}{'delta':>12}{'S(delta)':>11}{'|dAUROC|':>11}   binding?")
    for _, r in g.iterrows():
        binding = "yes" if r.S < 0.5 else "no, S>0.5"
        slack = r.S / max(r.dAUROC, 1e-12)
        print(f"{r.scorer:<9}{r['quantile']:<9}{r.delta:12.4g}{r.S:11.4f}{r.dAUROC:11.6f}"
              f"   {binding}, {slack:.0f}x slack")
    print(f"\nwrote {args.out / 'rq1_4_susceptibility_observed.csv'}")


if __name__ == "__main__":
    main()
