"""
Phase 3 -- score the locked external domains on the frozen configurations.

The ID side is unchanged: the same 5,067 held-out ISIC embeddings, adapted by
the same adapters. Only the OOD side is swapped, so any difference between pairs
is a property of the external domain and not of the pipeline.

The scorers are imported from generate_per_example_scores rather than
reimplemented, so they are the same functions that produced the primary results
-- including the fit population, which is the 16,211 adapted training
embeddings. Fitting on the evaluation set instead gives AUROC near 0.99997,
flat in alpha, and fails silently.

Nothing here is tuned on the external data. See LOCKED_PROTOCOL.md section 6.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from generate_per_example_scores import (  # noqa: E402
    ALPHAS, ARMS, SEEDS, load, displacement, score_all,
)

OUT = ROOT / "outputs" / "phase3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    alphas = [float(a) for a in args.alphas.split(",") if a]
    print(f"[score] arms={arms} alphas={alphas}")

    Ztr, Zev, ytr, is_ood_orig = load(ROOT)
    isic = Zev[is_ood_orig == 0]          # the 5,067 held-out ID images
    pad = Zev[is_ood_orig == 1]           # PAD-UFES-20, kept as pair P0
    print(f"[score] ID {isic.shape}, PAD {pad.shape}, train {Ztr.shape}")

    pairs = {}
    for f in sorted(OUT.glob("external_*.npz")):
        name = f.stem.replace("external_", "")
        pairs[{"derm7pt_derm": "P1_derm7pt_derm",
               "derm7pt_clinic": "P2_derm7pt_clinic",
               "ph2": "P3_ph2"}[name]] = np.load(f)["embeddings"].astype(np.float32)

    excluded = set()
    fl = OUT / "overlap_screen_flagged.csv"
    if fl.is_file() and fl.stat().st_size > 40:
        excluded = set(map(tuple, pd.read_csv(fl)[["dataset", "case"]].to_numpy()))
        print(f"[score] overlap screen excludes {len(excluded)} images")

    rows = []
    for pair, ood in pairs.items():
        Zev_p = np.vstack([isic, ood])
        y = np.r_[np.zeros(len(isic), int), np.ones(len(ood), int)]
        print(f"\n[score] {pair}: {len(isic)} ID vs {len(ood)} OOD")
        for arm in arms:
            for seed in SEEDS:
                D_tr = displacement(ROOT, arm, seed)(Ztr)
                D_ev = displacement(ROOT, arm, seed)(Zev_p)
                for a in alphas:
                    dest = OUT / "per_example" / f"{pair}_{arm}_a{a}_s{seed}.npz"
                    if dest.is_file():
                        continue
                    s, _, _ = score_all(Ztr + a * D_tr, ytr, Zev_p + a * D_ev)
                    rec = dict(pair=pair, arm=arm, seed=seed, alpha=a,
                               n_ood=len(ood))
                    for k, v in s.items():
                        rec[f"{k}_auroc"] = roc_auc_score(y, v)
                    rows.append(rec)
                    np.savez_compressed(dest, is_ood=y, **s)
                    print(f"   {arm} s{seed} a{a}: "
                          + "  ".join(f"{k[:4]}={rec[k+'_auroc']:.4f}"
                                      for k in ["maha", "cosine", "knn10", "kde"]),
                          flush=True)
    df = pd.DataFrame(rows)
    prev = OUT / "external_auroc.csv"
    if prev.is_file():
        df = pd.concat([pd.read_csv(prev), df]).drop_duplicates(
            ["pair", "arm", "seed", "alpha"], keep="last")
    df.to_csv(prev, index=False)
    print(f"\n[score] wrote external_auroc.csv ({len(df)} rows)")
    print(df.groupby("pair")[["maha_auroc", "cosine_auroc",
                              "knn10_auroc", "kde_auroc"]].mean().round(4).to_string())


if __name__ == "__main__":
    (OUT / "per_example").mkdir(parents=True, exist_ok=True)
    main()
