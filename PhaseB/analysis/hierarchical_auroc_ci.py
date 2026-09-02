"""
Hierarchical paired uncertainty for directed AUROC, per the plan's Section 5.

The manuscript reports A/B/C means over five seeds with no interval, and the
paired contrasts are read descriptively because nothing quantifies their
uncertainty. The data have two levels -- seed, and examples within seed -- and
the plan is explicit that the independent replicate is the seed. Resampling all
7,365 images as if independent would understate the interval; resampling seeds
alone would ignore the finite evaluation set.

So both, in that order: resample the five seeds with replacement, then within
each drawn seed resample ID and OOD examples with replacement, recompute
directed AUROC, and take the paired contrast on the same replicate. Pairing
inside the replicate is what makes the contrast interval narrower than the
difference of two marginal intervals, which is the point of doing it this way.

Configurations follow the manuscript: A is alpha=1 gamma=0, B is alpha=0.25
gamma=0, C is alpha=1 gamma=0.01, the last one geometry-pulled toward B's
spectral slope.

    python3 PhaseB/analysis/hierarchical_auroc_ci.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate_per_example_scores import load, displacement, score_all  # noqa: E402

SEEDS = (42, 52, 62, 72, 82)
SCORERS = ("maha", "cosine", "knn10", "kde")
CONFIGS = {"A": (1.0, 0.0), "B": (0.25, 0.0), "C": (1.0, 0.01)}


def temper(fit_pop, Z, gamma):
    """z' = mu + U diag(lambda^(-gamma/2)) U^T (z - mu), spectrum from the fit population."""
    mu = fit_pop.mean(0)
    S = np.cov((fit_pop - mu).T.astype(np.float64))
    lam, U = np.linalg.eigh(S)
    A = (U @ np.diag(np.clip(lam, 1e-12, None) ** (-gamma / 2)) @ U.T).astype(np.float32)
    return mu + (Z - mu) @ A.T


def fast_auroc(scores, is_ood):
    """Rank-based AUROC; called millions of times, so not roc_auc_score."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = is_ood.sum()
    n_neg = len(is_ood) - n_pos
    return (ranks[is_ood].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def build(root: Path, cache: Path):
    """Per-example scores for A, B, C across seeds, cached."""
    cache.mkdir(parents=True, exist_ok=True)
    Ztr, Zev, ytr, is_ood = load(root)
    out = {}
    for seed in SEEDS:
        dz = displacement(root, "Intervention", seed)
        Dtr, Dev = dz(Ztr), dz(Zev)
        for name, (a, g) in CONFIGS.items():
            f = cache / f"{name}_s{seed}.npz"
            if f.exists():
                out[(name, seed)] = dict(np.load(f))
                continue
            Ta, Ea = Ztr + a * Dtr, Zev + a * Dev
            if g > 0:
                Ea, Ta = temper(Ta, Ea, g), temper(Ta, Ta, g)
            sc, _, _ = score_all(Ta, ytr, Ea)
            np.savez_compressed(f, is_ood=is_ood, **sc)
            out[(name, seed)] = {"is_ood": is_ood, **sc}
            print(f"  scored {name} seed {seed}", flush=True)
    return out, is_ood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--cache", type=Path, default=Path("PhaseB/outputs/per_example_abc"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/rq3"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data, is_ood = build(args.root, args.cache)
    idx_id = np.flatnonzero(~is_ood.astype(bool))
    idx_ood = np.flatnonzero(is_ood.astype(bool))
    rng = np.random.default_rng(args.seed)

    # point estimate on the observed data, computed once and unresampled
    point = {sc: {c: float(np.mean([fast_auroc(data[(c, sd)][sc], is_ood.astype(bool))
                                    for sd in SEEDS]))
                  for c in CONFIGS} for sc in SCORERS}

    reps = {sc: {c: np.empty(args.n_boot) for c in CONFIGS} for sc in SCORERS}
    for b in range(args.n_boot):
        drawn = rng.choice(SEEDS, len(SEEDS))
        ii = idx_id[rng.integers(0, len(idx_id), len(idx_id))]
        oo = idx_ood[rng.integers(0, len(idx_ood), len(idx_ood))]
        take = np.concatenate([ii, oo])
        lab = np.concatenate([np.zeros(len(ii), bool), np.ones(len(oo), bool)])
        for sc in SCORERS:
            for c in CONFIGS:
                vals = [fast_auroc(data[(c, s)][sc][take], lab) for s in drawn]
                reps[sc][c][b] = np.mean(vals)

    rows = []
    for sc in SCORERS:
        for c in CONFIGS:
            v = reps[sc][c]
            # estimate is the point estimate, not the bootstrap mean: the latter
            # carries Monte Carlo noise of order 2e-4 here, which is 10% of the
            # cosine C-A contrast and larger than the k-NN one
            rows.append({"scorer": sc, "quantity": c, "estimate": point[sc][c],
                         "boot_mean": v.mean(),
                         "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)})
        for a, b_ in (("C", "A"), ("C", "B")):
            d = reps[sc][a] - reps[sc][b_]
            rows.append({"scorer": sc, "quantity": f"{a}-{b_}",
                         "estimate": point[sc][a] - point[sc][b_],
                         "boot_mean": d.mean(),
                         "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "hierarchical_auroc_ci.csv", index=False)

    print(f"\nhierarchical bootstrap, {args.n_boot} replicates, seeds then examples\n")
    print(f"{'scorer':<9}{'quantity':<10}{'estimate':>11}{'95% CI':>26}")
    for _, r in df.iterrows():
        print(f"{r.scorer:<9}{r.quantity:<10}{r.estimate:11.5f}   [{r.lo:+.5f}, {r.hi:+.5f}]")
    print(f"\nwrote {args.out / 'hierarchical_auroc_ci.csv'}")


if __name__ == "__main__":
    main()
