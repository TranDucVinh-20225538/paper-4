"""
Second control: does the primary cosine effect survive at Derm7pt's sample size?

The external pairs have 983 OOD images against PAD-UFES's 2,298. If the effect
vanishes simply because there are fewer images, that is a power story and the
external null says nothing about the domain. So: subsample PAD to 983, refit
nothing, rescore nothing, and rerun the same matched contrast.

Subsampling is stratified on the OOD label distribution and repeated, because a
single draw would confound the answer with the luck of that draw. The ID side is
left whole -- it is whole in the external pairs too.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from rq3_geometry_matching import (  # noqa: E402
    G_COLS, SCORERS, standardize, greedy_match, cluster_bootstrap,
)

OUT = ROOT / "outputs" / "phase3"
N_TARGET, N_DRAWS, N_BOOT = 983, 40, 4000


def main():
    base = pd.read_csv(OUT / "matched_input_P0_pad_ufes.csv")
    meta = pd.read_csv(ROOT / "assets/reference_embeddings/ReferenceEmbedding/metadata.csv")
    y_ood = meta.label_idx.to_numpy()[(meta.domain != "isic").to_numpy()]
    rng = np.random.default_rng(0)

    # one stratified index set per draw, shared across every configuration so the
    # contrast is computed on a single consistent population each time
    draws = []
    for _ in range(N_DRAWS):
        keep = []
        for c in np.unique(y_ood):
            idx = np.where(y_ood == c)[0]
            k = max(1, int(round(len(idx) * N_TARGET / len(y_ood))))
            keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
        draws.append(np.concatenate(keep))
    print(f"[sub] {N_DRAWS} draws of {len(draws[0])} OOD images from {len(y_ood)}")

    out = []
    for di, keep in enumerate(draws):
        d = base.copy()
        for _, r in base.iterrows():
            f = ROOT / "outputs" / "per_example" / f"{r.arm}_a{r.alpha}_s{int(r.seed)}.npz"
            s = np.load(f)
            ood_all = s["is_ood"].astype(bool)
            sel = np.r_[np.where(~ood_all)[0], np.where(ood_all)[0][keep]]
            yy = ood_all[sel]
            for sc in ["maha", "cosine", "knn10", "kde"]:
                d.loc[r.name, f"{sc}_auroc"] = roc_auc_score(yy, s[sc][sel])
        dz = standardize(d.reset_index(drop=True))
        pairs = greedy_match(dz, 0.25)
        if not pairs:
            continue
        i, j = map(np.array, zip(*pairs))
        A, B = dz.iloc[i], dz.iloc[j]
        for sc in SCORERS:
            dd = A[sc].to_numpy() - B[sc].to_numpy()
            boot = cluster_bootstrap(dd, A.seed.to_numpy(), N_BOOT, rng)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append(dict(draw=di, scorer=sc, n_pairs=len(pairs), delta=dd.mean(),
                            lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0)))
        if di % 10 == 0:
            print(f"  draw {di}: {len(pairs)} pairs", flush=True)

    r = pd.DataFrame(out)
    r.to_csv(OUT / "p0_subsample_control.csv", index=False)
    print("\n=== PAD subsampled to Derm7pt's 983 OOD images ===")
    g = r.groupby("scorer").agg(delta=("delta", "mean"), sd=("delta", "std"),
                                ci_width=("hi", "mean"), frac_excl=("excludes_zero", "mean"))
    g["ci_width"] = r.groupby("scorer").apply(lambda x: (x.hi - x.lo).mean())
    for sc, row in g.iterrows():
        print(f"  {sc:<14}delta {row.delta:+.5f} (sd {row.sd:.5f})  "
              f"mean CI width {row.ci_width:.5f}  "
              f"excludes 0 in {row.frac_excl*100:.0f}% of draws")


if __name__ == "__main__":
    main()
