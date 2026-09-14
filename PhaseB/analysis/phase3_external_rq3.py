"""
Phase 3 -- RQ3 geometry matching on each external pair.

Same design as the primary analysis, same code: standardize, greedy_match, smd
and multiway_cluster_bootstrap are imported from rq3_geometry_matching rather than
copied, so a difference between pairs is a difference in the data and not in my
transcription.

alpha = 0 is dropped, as in the primary matched input: with the adapter not yet
applied all three arms coincide there, so a cross-identity match at alpha = 0
matches a configuration against itself.

Each external pair yields 60 observations (3 arms x 4 alphas x 5 seeds) against
the primary's 45, because the external run has the full Adaptation ladder rather
than the single rung that survived the primary merge.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from rq3_geometry_matching import (  # noqa: E402
    G_COLS, SCORERS, CALIPERS, standardize, greedy_match, smd,
    multiway_cluster_bootstrap,
)

OUT = ROOT / "outputs" / "phase3"
N_BOOT = 10_000


def main():
    au = pd.read_csv(OUT / "external_auroc.csv")
    ge = pd.read_csv(OUT / "external_geometry.csv")
    df0 = au.merge(ge, on=["pair", "arm", "seed", "alpha"], how="inner")
    df0 = df0[df0.alpha > 0].reset_index(drop=True)
    print(f"merged {len(df0)} observations across {df0.pair.nunique()} pairs\n")

    rng = np.random.default_rng(0)
    rows, neg = [], []
    for pair in sorted(df0.pair.unique()):
        d = standardize(df0[df0.pair == pair].reset_index(drop=True))
        print(f"===== {pair}: {len(d)} observations, {d.arm.nunique()} identities, "
              f"{d.seed.nunique()} seeds =====")
        for cal in CALIPERS:
            pairs = greedy_match(d, cal)
            if not pairs:
                print(f"  caliper {cal}: no pairs within the caliper")
                continue
            i, j = map(np.array, zip(*pairs))
            A, B = d.iloc[i], d.iloc[j]
            bal = {c: smd(A[f"z_{c}"].to_numpy(), B[f"z_{c}"].to_numpy()) for c in G_COLS}
            worst = max(abs(v) for v in bal.values())
            verdict = ("balanced" if worst < 0.10 else
                       "acceptable" if worst < 0.20 else "FAILS -- inconclusive")
            print(f"  caliper {cal}: {len(pairs)} pairs, |SMD| max {worst:.3f} ({verdict})"
                  f"   " + "  ".join(f"{c.split('_')[0]} {v:+.3f}" for c, v in bal.items()))
            sa, sb = A.seed.to_numpy(), B.seed.to_numpy()
            for sc in SCORERS:
                dd = A[sc].to_numpy() - B[sc].to_numpy()
                boot = multiway_cluster_bootstrap(dd, sa, sb, N_BOOT, rng)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                rows.append(dict(pair=pair, caliper=cal, n_pairs=len(pairs), scorer=sc,
                                 worst_smd=worst, delta=dd.mean(), lo=lo, hi=hi,
                                 excludes_zero=bool(lo > 0 or hi < 0)))

        # negative control at the preregistered caliper: permute identity within
        # matched sets, so any surviving signal is identity and not geometry
        pairs = greedy_match(d, 0.25)
        if pairs:
            i, j = map(np.array, zip(*pairs))
            A, B = d.iloc[i], d.iloc[j]
            for sc in SCORERS:
                obs = (A[sc].to_numpy() - B[sc].to_numpy()).mean()
                null = np.array([np.mean(np.where(rng.random(len(i)) < 0.5, 1, -1)
                                         * (A[sc].to_numpy() - B[sc].to_numpy()))
                                 for _ in range(N_BOOT)])
                neg.append(dict(pair=pair, scorer=sc, observed=obs,
                                p=float((np.abs(null) >= abs(obs)).mean())))
        print()

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "external_rq3_matched.csv", index=False)
    ncr = pd.DataFrame(neg)
    ncr.to_csv(OUT / "external_rq3_negative_control.csv", index=False)

    print("=== matched contrasts at the preregistered caliper 0.25 ===")
    p = res[res.caliper == 0.25]
    for _, r in p.iterrows():
        print(f"  {r.pair:<20}{r.scorer:<14}{r.delta:+9.5f}  "
              f"[{r.lo:+.5f}, {r.hi:+.5f}]  {'excludes 0' if r.excludes_zero else ''}")
    print("\n=== negative control (identity permuted within matched sets) ===")
    for _, r in ncr.iterrows():
        print(f"  {r.pair:<20}{r.scorer:<14}p={r.p:.3f}")
    print(f"\nwrote external_rq3_matched.csv and external_rq3_negative_control.csv")


if __name__ == "__main__":
    main()
