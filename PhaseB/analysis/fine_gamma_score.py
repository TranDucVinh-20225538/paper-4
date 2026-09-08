"""
Score the fine-gamma configurations, so they can enter the matched analysis.

Geometry for these was computed first and the grid was fixed from it; this
script runs afterwards and only supplies the outcome side. Same scorers, same
fit population -- the 16,211 adapted training embeddings -- as everywhere else.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from hierarchical_auroc_ci import temper  # noqa: E402
from generate_per_example_scores import load, displacement, score_all  # noqa: E402

OUT = ROOT / "outputs" / "fine_gamma"


def main():
    geo = pd.read_csv(OUT / "geometry_partial.csv")
    Ztr, Zev, ytr, is_ood = load(ROOT)
    y = is_ood.astype(bool)
    part = OUT / "scores_partial.csv"
    prev = pd.read_csv(part) if part.is_file() else pd.DataFrame()
    done = set(map(tuple, prev[["arm", "seed", "alpha", "gamma"]].to_numpy())) if len(prev) else set()
    print(f"[score] {len(geo)} configurations, {len(done)} already done")

    rows = []
    for (arm, seed), grp in geo.groupby(["arm", "seed"]):
        dz = displacement(ROOT, arm, int(seed))
        Dtr, Dev = dz(Ztr), dz(Zev)
        for _, r in grp.iterrows():
            if (arm, seed, r.alpha, r.gamma) in done:
                continue
            Ta, Ea = Ztr + r.alpha * Dtr, Zev + r.alpha * Dev
            E, T = temper(Ta, Ea, r.gamma), temper(Ta, Ta, r.gamma)
            sc, _, _ = score_all(T, ytr, E)
            rec = dict(arm=arm, config=r.config, alpha=r.alpha, alpha_num=r.alpha,
                       gamma=r.gamma, seed=int(seed),
                       lid_mean=r.lid_mean, spectral_decay_slope=r.spectral_decay_slope)
            for k, v in sc.items():
                rec[f"{k}_auroc"] = roc_auc_score(y, v)
            rows.append(rec)
            pd.DataFrame(rows).to_csv(part, index=False)
        if rows:
            print(f"  {arm} s{seed}: cos={rows[-1]['cosine_auroc']:.4f} "
                  f"maha={rows[-1]['maha_auroc']:.4f}", flush=True)

    df = pd.concat([prev, pd.DataFrame(rows)]) if len(prev) else pd.DataFrame(rows)
    df = df.drop_duplicates(["arm", "seed", "alpha", "gamma"])
    df.to_csv(OUT / "fine_gamma_scored.csv", index=False)
    print(f"\n[score] wrote fine_gamma_scored.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
