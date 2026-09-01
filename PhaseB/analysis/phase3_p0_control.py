"""
Positive control: run the Phase 3 pipeline on PAD-UFES and see if it returns the
primary result.

The external pairs produce nothing, and that null contradicts the primary run's
cosine contrast. Before believing the data rather than the code, the pipeline
has to demonstrate it can still find the effect where the effect is known to be.
So: same scoring path, same geometry path, same RQ3 code, pointed at the OOD set
the primary result came from.

The control mirrors the primary input's structure rather than the external
pairs': the primary run scored only one Adaptation rung, partial-FT at alpha=1,
so that arm contributes five rows and the table is 45, not 60. Using all sixty
here would test a different design against the primary's number and prove
nothing. Configurations with no per-example file are skipped rather than
assumed.

What this must reproduce is the finding -- a cosine contrast near -0.0085 whose
interval excludes zero.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from lid_spectral_decay import compute_lid_spectral_diagnostics  # noqa: E402
from generate_per_example_scores import ALPHAS, ARMS, SEEDS, load, displacement  # noqa: E402

OUT = ROOT / "outputs" / "phase3"
DEST = OUT / "matched_input_P0_pad_ufes.csv"


def main():
    Ztr, Zev, ytr, is_ood = load(ROOT)
    y = pd.read_csv(ROOT / "assets/reference_embeddings/ReferenceEmbedding/metadata.csv"
                    ).label_idx.to_numpy()
    ood = is_ood.astype(bool)
    print(f"[p0] population {Zev.shape}, {(~ood).sum()} ID vs {ood.sum()} OOD")

    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            D = displacement(ROOT, arm, seed)(Zev)
            for a in ALPHAS:
                if a == 0.0:            # dropped downstream, and rank-deficient there
                    continue
                Za = Zev + a * D
                g = compute_lid_spectral_diagnostics(Za, y, num_classes=8, k=50)
                # scores come from the per-example files the primary run produced,
                # so this control cannot pass by rescoring differently
                f = ROOT / "outputs" / "per_example" / f"{arm}_a{a}_s{seed}.npz"
                if not f.is_file():
                    continue          # primary scored only partial-FT for Adaptation
                s = np.load(f)
                rec = dict(arm=arm, config=f"alpha{a}", alpha=a, alpha_num=a, seed=seed,
                           lid_mean=g.lid_mean, spectral_decay_slope=g.spectral_slope)
                for sc in ["maha", "cosine", "knn10", "kde"]:
                    rec[f"{sc}_auroc"] = roc_auc_score(s["is_ood"].astype(bool), s[sc])
                rows.append(rec)
                pd.DataFrame(rows).to_csv(DEST, index=False)
            if not rows:
                continue
            print(f"  {arm} s{seed}: lid={rows[-1]['lid_mean']:.3f} "
                  f"slope={rows[-1]['spectral_decay_slope']:.5f} "
                  f"cos={rows[-1]['cosine_auroc']:.5f}", flush=True)
    pd.DataFrame(rows).to_csv(DEST, index=False)
    print(f"\n[p0] wrote {DEST.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
