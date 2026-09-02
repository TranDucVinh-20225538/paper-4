"""
Fine gamma grid, the conditional item in the advisor's Stage 2.

The condition is met: common support is thin. At the preregistered caliper only
a fifth of Adaptation's configurations have any cross-identity neighbour, and at
caliper .15 none do. Adaptation sits at high spectral slope, and the
preregistered gamma ladder is positive-only, which lowers slope -- so it moves
away from the region that needs filling, and no amount of refinement within the
registered range fills it. A pilot confirmed this before the grid below was
chosen: the closest configuration reachable with gamma >= 0 is at distance .95,
against a caliper of .25.

PROTOCOL DEVIATION, recorded in 24_PreRegistration.md section 13. The grid here
extends to small NEGATIVE gamma, which sharpens the spectrum rather than
whitening it. This is inside the theory -- kappa_2(A_gamma) = kappa_2(Sigma)^(|gamma|/2)
takes the absolute value -- but outside the registered ladder.

The extension was chosen on geometry alone, before any outcome was computed for
these configurations. This script computes geometry only. It has no access to
AUROC and no code path that reaches a scorer, which is what makes that claim
checkable rather than asserted.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from lid_spectral_decay import compute_lid_spectral_diagnostics  # noqa: E402
from hierarchical_auroc_ci import temper  # noqa: E402
from generate_per_example_scores import SEEDS, load, displacement  # noqa: E402

OUT = ROOT / "outputs" / "fine_gamma"
GAMMAS = (-0.006, -0.005, -0.004, -0.003)
ALPHAS = (0.25, 0.5, 0.75)
ARMS = ("Intervention", "Conventional")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / "geometry_partial.csv"
    prev = pd.read_csv(part) if part.is_file() else pd.DataFrame()
    done = set(map(tuple, prev[["arm", "seed", "alpha", "gamma"]].to_numpy())) if len(prev) else set()
    if done:
        print(f"[grid] resuming, {len(done)} already measured")

    Ztr, Zev, ytr, _ = load(ROOT)
    y = pd.read_csv(ROOT / "assets/reference_embeddings/ReferenceEmbedding/metadata.csv"
                    ).label_idx.to_numpy()
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            dz = displacement(ROOT, arm, seed)
            Dtr, Dev = dz(Ztr), dz(Zev)
            for a in ALPHAS:
                Ta, Ea = Ztr + a * Dtr, Zev + a * Dev
                for g in GAMMAS:
                    if (arm, seed, a, g) in done:
                        continue
                    E, T = temper(Ta, Ea, g), temper(Ta, Ta, g)
                    d = compute_lid_spectral_diagnostics(E, y, num_classes=8, k=50)
                    rows.append(dict(arm=arm, config=f"a{a}_g{g}", alpha=a, alpha_num=a,
                                     gamma=g, seed=seed, lid_mean=d.lid_mean,
                                     spectral_decay_slope=d.spectral_slope))
                    pd.DataFrame(rows).to_csv(part, index=False)
            if rows:
                print(f"  {arm} s{seed}: lid={rows[-1]['lid_mean']:.3f} "
                      f"slope={rows[-1]['spectral_decay_slope']:.5f}", flush=True)

    df = pd.concat([prev, pd.DataFrame(rows)]) if len(prev) else pd.DataFrame(rows)
    df = df.drop_duplicates(["arm", "seed", "alpha", "gamma"])
    df.to_csv(OUT / "fine_gamma_geometry.csv", index=False)
    print(f"\n[grid] wrote fine_gamma_geometry.csv ({len(df)} configurations)")


if __name__ == "__main__":
    main()
