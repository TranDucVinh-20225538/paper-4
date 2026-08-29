"""
Phase 2 of the plan: geometry-matched inference for RQ3.

The A/B/C comparison the manuscript reports is a single approximate match on one
property, with no balance diagnostics and no common-support check. This builds
the matched design the plan asks for, from the same retained data.

Following Section 6 of the plan:

  G = (LID, spectral slope). Condition number is a nuisance metric here, not a
  matching variable: the manipulation check shows the normalization operator
  moves it by -2.7e5 while leaving both shape metrics unchanged, so it behaves
  as a scale descriptor and matching on it would match on scale.

  Standardization uses the conventional arm's distribution, not the pooled one,
  so the metric is not centred using the outcome-bearing arms.

  Pairs are formed only across different intervention identities. Matching never
  sees an AUROC.

  Caliper 0.25 preregistered, with 0.15 and 0.35 as sensitivity. Each observation
  is used at most once.

  Balance is checked by standardized mean difference, target |SMD| < 0.10 and
  acceptable to 0.20. If balance fails the answer is inconclusive, not a
  loosened caliper.

Uncertainty clusters on seed throughout. A negative control permutes intervention
identity within matched sets, keeping G and Y, to confirm the pipeline does not
manufacture a residual identity effect.

    python3 PhaseB/analysis/rq3_geometry_matching.py
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

G_COLS = ["lid_mean", "spectral_decay_slope"]
SCORERS = ["maha_auroc", "cosine_auroc", "knn10_auroc", "kde_auroc"]
CALIPERS = (0.15, 0.25, 0.35)


def standardize(df, ref_arm="Conventional"):
    ref = df[df.arm == ref_arm]
    out = df.copy()
    for c in G_COLS:
        mu, sd = ref[c].mean(), ref[c].std(ddof=1)
        out[f"z_{c}"] = (df[c] - mu) / (sd if sd > 0 else 1.0)
    return out


def greedy_match(df, caliper):
    """Nearest available cross-identity partner within the caliper, each row used once."""
    Z = df[[f"z_{c}" for c in G_COLS]].to_numpy()
    ids = df.arm.to_numpy()
    n = len(df)
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    same = ids[:, None] == ids[None, :]
    d = np.where(same, np.inf, d)
    np.fill_diagonal(d, np.inf)

    pairs, used = [], set()
    for i, j in sorted(((i, j) for i in range(n) for j in range(i + 1, n)), key=lambda p: d[p]):
        if d[i, j] > caliper or i in used or j in used:
            continue
        pairs.append((i, j))
        used.update((i, j))
    return pairs


def smd(a, b):
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


def cluster_bootstrap(diffs, seeds, n, rng):
    uniq = np.unique(seeds)
    return np.array([np.mean(np.concatenate([diffs[seeds == s] for s in rng.choice(uniq, len(uniq))]))
                     for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PhaseB/outputs/matched_input.csv"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/rq3"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = standardize(pd.read_csv(args.data)).reset_index(drop=True)
    print(f"{len(df)} observations, {df.arm.nunique()} intervention identities, "
          f"{df.seed.nunique()} seeds\n")

    rows = []
    for cal in CALIPERS:
        pairs = greedy_match(df, cal)
        if not pairs:
            print(f"caliper {cal}: no pairs within the caliper")
            continue
        i, j = map(np.array, zip(*pairs))
        A, B = df.iloc[i], df.iloc[j]

        bal = {c: smd(A[f"z_{c}"].to_numpy(), B[f"z_{c}"].to_numpy()) for c in G_COLS}
        worst = max(abs(v) for v in bal.values())
        verdict = ("balanced" if worst < 0.10 else
                   "acceptable" if worst < 0.20 else "FAILS -- inconclusive")
        print(f"caliper {cal}: {len(pairs)} pairs, |SMD| max {worst:.3f} ({verdict})")
        print("   " + "  ".join(f"{c} {v:+.3f}" for c, v in bal.items()))
        print("   identity pairings: " +
              ", ".join(f"{a}-{b}: {n}" for (a, b), n in
                        pd.Series(list(zip(A.arm, B.arm))).value_counts().items()))

        seeds = A.seed.to_numpy()
        for sc in SCORERS:
            d = A[sc].to_numpy() - B[sc].to_numpy()
            boot = cluster_bootstrap(d, seeds, args.n_boot, rng)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"caliper": cal, "n_pairs": len(pairs), "scorer": sc,
                         "worst_smd": worst, "delta": d.mean(), "lo": lo, "hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
        print()

    res = pd.DataFrame(rows)
    res.to_csv(args.out / "rq3_matched_contrasts.csv", index=False)
    print(f"{'caliper':>8}{'n':>5}  {'scorer':<14}{'delta':>10}{'95% CI':>24}  excl 0")
    for _, r in res.iterrows():
        print(f"{r.caliper:>8}{r.n_pairs:>5}  {r.scorer:<14}{r.delta:+10.5f}"
              f"   [{r.lo:+.5f}, {r.hi:+.5f}]  {'yes' if r.excludes_zero else 'no'}")

    # negative control at the preregistered caliper
    pairs = greedy_match(df, 0.25)
    if pairs:
        i, j = map(np.array, zip(*pairs))
        print("\nnegative control: identity permuted within matched sets, "
              f"{args.n_boot} permutations")
        for sc in SCORERS:
            obs = (df.iloc[i][sc].to_numpy() - df.iloc[j][sc].to_numpy()).mean()
            null = np.array([np.mean(np.where(rng.random(len(i)) < 0.5, 1, -1) *
                                     (df.iloc[i][sc].to_numpy() - df.iloc[j][sc].to_numpy()))
                             for _ in range(args.n_boot)])
            p = float((np.abs(null) >= abs(obs)).mean())
            print(f"   {sc:<14} observed {obs:+.5f}   permutation p {p:.4f}")
    print(f"\nwrote {args.out / 'rq3_matched_contrasts.csv'}")


if __name__ == "__main__":
    main()
