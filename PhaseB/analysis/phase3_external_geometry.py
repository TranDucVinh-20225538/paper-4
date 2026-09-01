"""
Phase 3 -- geometry metrics for the external pairs.

LID and the within-class spectral-decay slope are computed on the full adapted
evaluation population, ID and OOD together, exactly as stage4_canonical_adaptation
does it. That means they are NOT reusable across pairs: swapping the OOD half
changes the population the geometry describes, so each pair gets its own.

compute_lid_spectral_diagnostics is imported rather than reimplemented, k=50 as
in the primary run. External label vectors are the locked mapping's, so classes
absent from a given external set are simply empty -- which already occurs in the
primary run, where PAD-UFES has no DF or VASC.

alpha = 0 is skipped, for a reason worth recording. The within-class scatter of
the raw embeddings has exactly one direction of essentially zero variance: the
class-centred data's smallest singular value is 3.95e-7 against 5.47e-3 for the
next, so the matrix is numerically rank 1023, not 1024. On the primary
population that eigenvalue lands at +5e-16 and the positive-definiteness guard
lets it through; on PH2 it lands at -9e-17 and the guard fires. Same
degeneracy, opposite side of floating-point zero.

It is not harmless where it passes. spectral_decay_slope fits log(lambda)
against index over the full range, so an eigenvalue at 1e-16 contributes a point
some 21 log units below its neighbour and drags the fit: the alpha = 0 slope is
inflated by 2.2%, against 0.3% at alpha > 0 where the adapter lifts the
degenerate direction to ~1e-6. For scale, the manuscript discusses slope
differences of about 0.1%.

Skipping alpha = 0 costs this phase nothing -- every downstream analysis already
drops it, because with the adapter unapplied all three arms coincide there. It
does not fix the primary run's alpha = 0 rows, which feed the exploratory
analyses and carry the same 2.2% inflation. That is flagged, not silently
patched, because changing the slope definition changes published numbers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from lid_spectral_decay import compute_lid_spectral_diagnostics  # noqa: E402
from generate_per_example_scores import ALPHAS, ARMS, SEEDS, load, displacement  # noqa: E402

OUT = ROOT / "outputs" / "phase3"
PART = OUT / "external_geometry_partial.csv"
PAIR_FILE = {"P1_derm7pt_derm": "derm7pt_derm",
             "P2_derm7pt_clinic": "derm7pt_clinic",
             "P3_ph2": "ph2"}


def main():
    Ztr, Zev, ytr, is_ood = load(ROOT)
    A = ROOT / "assets/reference_embeddings"
    meta = pd.read_csv(A / "ReferenceEmbedding/metadata.csv")
    isic = Zev[is_ood == 0]
    y_isic = meta.label_idx.to_numpy()[is_ood == 0]
    print(f"[geom] ID {isic.shape}, labels {np.bincount(y_isic, minlength=8)}")

    prev = pd.read_csv(PART) if PART.is_file() else pd.DataFrame()
    done = set(map(tuple, prev[["pair", "arm", "seed", "alpha"]].to_numpy())) if len(prev) else set()
    if done:
        print(f"[geom] resuming, {len(done)} configurations already measured")

    rows = []
    for pair, stem in PAIR_FILE.items():
        d = np.load(OUT / f"external_{stem}.npz", allow_pickle=True)
        Zp = np.vstack([isic, d["embeddings"].astype(np.float32)])
        yp = np.r_[y_isic, d["labels"]].astype(np.int64)
        print(f"\n[geom] {pair}: {Zp.shape}, labels {np.bincount(yp, minlength=8)}")
        for arm in ARMS:
            for seed in SEEDS:
                D = displacement(ROOT, arm, seed)(Zp)
                for a in ALPHAS:
                    if a == 0.0 or (pair, arm, seed, a) in done:
                        continue
                    g = compute_lid_spectral_diagnostics(Zp + a * D, yp,
                                                         num_classes=8, k=50)
                    rows.append(dict(pair=pair, arm=arm, seed=seed, alpha=a,
                                     lid_mean=g.lid_mean,
                                     spectral_decay_slope=g.spectral_slope))
                    # written per row: a crash on a later configuration must not
                    # cost the ones already measured
                    pd.DataFrame(rows).to_csv(PART, index=False)
                if rows:
                    print(f"   {arm} s{seed}: lid={rows[-1]['lid_mean']:.3f} "
                          f"slope={rows[-1]['spectral_decay_slope']:.5f}", flush=True)
    df = pd.DataFrame(rows)
    if PART.is_file() and len(prev):
        df = pd.concat([prev, df]).drop_duplicates(["pair", "arm", "seed", "alpha"])
    df.to_csv(OUT / "external_geometry.csv", index=False)
    print(f"\n[geom] wrote external_geometry.csv ({len(df)} rows)")
    print(df.groupby("pair")[["lid_mean", "spectral_decay_slope"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
