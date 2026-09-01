"""
Phase 3 -- build one matched-input table per external pair.

Emits CSVs in the same schema as PhaseB/outputs/matched_input.csv so that
rq3_geometry_matching.py and rq3_residual_model.py run on them unchanged. That
is the point: the primary and external results then come from literally the same
code, and a difference between them cannot be a difference in how I invoked it.

alpha = 0 is dropped, as in the primary input -- with the adapter not applied the
three arms coincide there, so a cross-identity match at alpha = 0 would match a
configuration against itself.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase3"


def main():
    au = pd.read_csv(OUT / "external_auroc.csv")
    ge = pd.read_csv(OUT / "external_geometry.csv")
    df = au.merge(ge, on=["pair", "arm", "seed", "alpha"], how="inner")
    missing = set(map(tuple, au[["pair", "arm", "seed", "alpha"]].to_numpy())) - \
              set(map(tuple, df[["pair", "arm", "seed", "alpha"]].to_numpy()))
    if missing:
        print(f"[build] WARNING: {len(missing)} scored configurations have no "
              f"geometry and are dropped")
    df = df[df.alpha > 0].copy()
    df["config"] = "alpha" + df.alpha.astype(str)
    df["alpha_num"] = df.alpha

    for pair, d in df.groupby("pair"):
        cols = ["arm", "config", "alpha", "alpha_num", "seed",
                "lid_mean", "spectral_decay_slope",
                "maha_auroc", "cosine_auroc", "knn10_auroc", "kde_auroc"]
        p = OUT / f"matched_input_{pair}.csv"
        d[cols].to_csv(p, index=False)
        print(f"[build] {p.name}: {len(d)} rows, "
              f"{d.arm.nunique()} identities, {d.seed.nunique()} seeds, "
              f"LID {d.lid_mean.min():.2f}-{d.lid_mean.max():.2f}")


if __name__ == "__main__":
    main()
