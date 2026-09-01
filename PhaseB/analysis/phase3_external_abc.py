"""
Phase 3 -- the A/B/C configurations on the external pairs.

Tables 4 and 5 of the manuscript rest on three configurations: A is the full
intervention, B a quarter of it, C the full intervention plus spectral tempering
at gamma = 0.01. The contrasts C-A and C-B are what the paper reads. Those were
measured against PAD-UFES only, and PAD-UFES is now the one domain where
anything replicates, so the contrasts are worth re-measuring where the domain
did not participate in calibration.

Everything is imported from hierarchical_auroc_ci and generate_per_example_scores
-- the configuration definitions, the tempering, the scorers and the fast AUROC.
Only the OOD half of the evaluation population changes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from hierarchical_auroc_ci import CONFIGS, SEEDS, SCORERS, temper, fast_auroc  # noqa: E402
from generate_per_example_scores import load, displacement, score_all  # noqa: E402

OUT = ROOT / "outputs" / "phase3"
CACHE = OUT / "per_example_abc"
PAIRS = {"P1_derm7pt_derm": "derm7pt_derm",
         "P2_derm7pt_clinic": "derm7pt_clinic",
         "P3_ph2": "ph2"}
N_BOOT = 5000


def cluster_boot(per_seed, rng, n=N_BOOT):
    """Seeds then examples, the two levels the design actually has."""
    seeds = list(per_seed)
    return np.array([np.mean([per_seed[s] for s in rng.choice(seeds, len(seeds))])
                     for _ in range(n)])


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    Ztr, Zev, ytr, is_ood0 = load(ROOT)
    isic = Zev[is_ood0 == 0]
    rng = np.random.default_rng(0)
    rows = []

    for pair, stem in PAIRS.items():
        ext = np.load(OUT / f"external_{stem}.npz")["embeddings"].astype(np.float32)
        Zp = np.vstack([isic, ext])
        y = np.r_[np.zeros(len(isic), bool), np.ones(len(ext), bool)]
        print(f"\n[abc] {pair}: {len(isic)} ID vs {len(ext)} OOD", flush=True)

        auc = {}
        for seed in SEEDS:
            dz = displacement(ROOT, "Intervention", seed)
            Dtr, Dev = dz(Ztr), dz(Zp)
            for name, (a, g) in CONFIGS.items():
                f = CACHE / f"{pair}_{name}_s{seed}.npz"
                if f.exists():
                    sc = dict(np.load(f))
                else:
                    Ta, Ea = Ztr + a * Dtr, Zp + a * Dev
                    if g > 0:
                        Ea, Ta = temper(Ta, Ea, g), temper(Ta, Ta, g)
                    sc, _, _ = score_all(Ta, ytr, Ea)
                    np.savez_compressed(f, is_ood=y, **sc)
                for s in SCORERS:
                    auc[(name, seed, s)] = fast_auroc(sc[s], y)
            print(f"   seed {seed}: " + "  ".join(
                f"{n}/{s[:3]}={auc[(n,seed,s)]:.4f}" for n in ["A", "C"]
                for s in ["cosine"]), flush=True)

        for s in SCORERS:
            for q, (x, z) in {"C-A": ("C", "A"), "C-B": ("C", "B")}.items():
                per = {sd: auc[(x, sd, s)] - auc[(z, sd, s)] for sd in SEEDS}
                b = cluster_boot(per, rng)
                lo, hi = np.percentile(b, [2.5, 97.5])
                rows.append(dict(pair=pair, scorer=s, quantity=q,
                                 estimate=np.mean(list(per.values())),
                                 lo=lo, hi=hi,
                                 excludes_zero=bool(lo > 0 or hi < 0)))
            for n in CONFIGS:
                rows.append(dict(pair=pair, scorer=s, quantity=n,
                                 estimate=np.mean([auc[(n, sd, s)] for sd in SEEDS]),
                                 lo=np.nan, hi=np.nan, excludes_zero=False))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "external_abc_hierarchical.csv", index=False)
    print("\n=== C-A and C-B on the external pairs ===")
    for _, r in df[df.quantity.isin(["C-A", "C-B"])].iterrows():
        print(f"  {r.pair:<20}{r.scorer:<8}{r.quantity}  {r.estimate:+.5f}  "
              f"[{r.lo:+.5f}, {r.hi:+.5f}]  {'excludes 0' if r.excludes_zero else ''}")


if __name__ == "__main__":
    main()
