"""
Paired utility non-inferiority for RQ2, per the plan's Section 3.2.

The plan's objection to the current Gate 0 is right: a threshold of 0.225
against a chance level of 0.125 does not support the word "preserving". A model
could fall from 0.75 to 0.30 and still pass. The claim and the evidence do not
match.

They can be made to match without new data. Both arms were run at the same
doses and seeds, so the utility question has a paired answer:

    D_r = U(intervention, r) - U(conventional, r)

Seed is the independent replicate, so the bootstrap resamples seeds and keeps
every observation belonging to a resampled seed, rather than resampling the
twenty configuration-seed rows as if they were independent.

This script reports the paired differences and the interval. It does not pick
a non-inferiority margin: the plan requires that to be justified in advance,
which is a decision about what utility loss matters clinically, not one the
data can make. It prints what margin the interval would clear.

    python3 PhaseB/analysis/utility_noninferiority.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UTILITY = "id_task_balanced_accuracy_mean"


def paired(matrix: Path) -> pd.DataFrame:
    d = pd.read_csv(matrix)
    a = d[d.arm == "Intervention"][["config", "seed", UTILITY]]
    b = d[d.arm == "Conventional"][["config", "seed", UTILITY]]
    m = a.merge(b, on=["config", "seed"], suffixes=("_int", "_conv"))
    if m.empty:
        raise SystemExit("no configuration-seed pairs shared by the two arms")
    m["D"] = m[f"{UTILITY}_int"] - m[f"{UTILITY}_conv"]
    return m


def seed_bootstrap(D: np.ndarray, seeds: np.ndarray, n: int, rng) -> np.ndarray:
    """Resample seeds, not rows: rows sharing a seed are not independent."""
    uniq = np.unique(seeds)
    return np.array([np.mean(np.concatenate([D[seeds == s] for s in rng.choice(uniq, len(uniq))]))
                     for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=Path("PhaseB/outputs/raw_run_matrix.csv"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = paired(args.matrix)
    print(f"{len(m)} pairs matched on (dose, seed), {m.seed.nunique()} seeds\n")
    print(f"{'dose':<12}{'mean D':>10}{'min':>10}{'max':>10}")
    for c, g in m.groupby("config"):
        print(f"{c:<12}{g.D.mean():+10.4f}{g.D.min():+10.4f}{g.D.max():+10.4f}")

    boot = seed_bootstrap(m.D.to_numpy(), m.seed.to_numpy(), args.n_boot,
                          np.random.default_rng(args.seed))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    one_sided = np.percentile(boot, 5)
    print(f"\noverall mean D {m.D.mean():+.4f}")
    print(f"95% CI (seed bootstrap)      [{lo:+.4f}, {hi:+.4f}]")
    print(f"one-sided 95% lower bound    {one_sided:+.4f}")
    print(f"\nNon-inferiority holds for any margin dU > {abs(one_sided):.4f}. "
          "Choosing dU is a prior decision about tolerable utility loss, not one "
          "this script makes.")

    d = pd.read_csv(args.matrix)
    print(f"\nFor contrast, Gate 0 asks only for > 0.225; observed utility runs "
          f"{d[UTILITY].min():.4f} to {d[UTILITY].max():.4f}.")


if __name__ == "__main__":
    main()
