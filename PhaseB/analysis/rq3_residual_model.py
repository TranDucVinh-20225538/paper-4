"""
Residual intervention effect and out-of-sample geometry prediction, plan Section 6.2.

The matched contrast answers whether performance differs once geometry is
matched. This asks the complementary question on the full population: after
geometry is in the model, does knowing which intervention produced a
configuration still buy anything?

Two models per scorer, both with a seed random effect:

    Y ~ LID + slope + (1|seed)
    Y ~ LID + slope + identity + (1|seed)

In-sample p-values on the identity terms are reported but are not the criterion.
The plan is explicit that the comparison should be out-of-sample, so the models
are also refit leave-one-seed-out and scored on the held-out seed. With five
seeds that is five folds, and the fold-to-fold spread is reported rather than a
single mean, because five numbers averaged hide more than they show.

Predictors are standardized on the training folds only, so the held-out seed
does not contribute to its own centring.

The plan's reading rule: if identity improves held-out prediction stably, the
conclusion available is that the measured geometry is insufficient to account
for the remaining variation. Not that intervention history is causal.

    python3 PhaseB/analysis/rq3_residual_model.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

G = ["lid_mean", "spectral_decay_slope"]
SCORERS = ["maha_auroc", "cosine_auroc", "knn10_auroc", "kde_auroc"]


def zscore(train, test, cols):
    tr, te = train.copy(), test.copy()
    for c in cols:
        mu, sd = train[c].mean(), train[c].std(ddof=1)
        sd = sd if sd > 0 else 1.0
        tr[c] = (train[c] - mu) / sd
        te[c] = (test[c] - mu) / sd
    return tr, te


def fit(df, formula):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, df, groups=df["seed"]).fit(method="lbfgs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("PhaseB/outputs/matched_input.csv"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/rq3"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    print(f"{len(df)} observations, identities {sorted(df.arm.unique())}, "
          f"{df.seed.nunique()} seeds\n")

    rows = []
    for sc in SCORERS:
        d = df[["seed", "arm", sc] + G].rename(columns={sc: "Y", "arm": "identity"}).dropna()
        dz, _ = zscore(d, d, G)

        geo = fit(dz, "Y ~ lid_mean + spectral_decay_slope")
        full = fit(dz, "Y ~ lid_mean + spectral_decay_slope + C(identity)")
        ident = {k: (full.params[k], full.pvalues[k])
                 for k in full.params.index if k.startswith("C(identity)")}
        # statsmodels returns nan for MixedLM.aic under lbfgs; compute it from the
        # log-likelihood, which is finite and the fits do converge
        aic = lambda m: 2 * (len(m.params) + 1) - 2 * m.llf
        collinearity = np.corrcoef(dz.lid_mean, dz.spectral_decay_slope)[0, 1]

        # leave-one-seed-out held-out error
        errs = {"geometry": [], "geometry+identity": []}
        for s in sorted(d.seed.unique()):
            tr, te = d[d.seed != s], d[d.seed == s]
            trz, tez = zscore(tr, te, G)
            for name, f in (("geometry", "Y ~ lid_mean + spectral_decay_slope"),
                            ("geometry+identity",
                             "Y ~ lid_mean + spectral_decay_slope + C(identity)")):
                m = fit(trz, f)
                pred = m.predict(tez)          # fixed effects only: held-out seed is unseen
                errs[name].append(float(np.mean(np.abs(tez.Y.to_numpy() - pred.to_numpy()))))

        g_mae, f_mae = np.array(errs["geometry"]), np.array(errs["geometry+identity"])
        improved = int((f_mae < g_mae).sum())
        rows.append({"scorer": sc, "aic_geometry": aic(geo), "aic_full": aic(full),
                     "converged": bool(geo.converged and full.converged),
                     "y_var": float(d.Y.var()), "G_collinearity": float(collinearity),
                     "max_identity_effect": max(abs(v[0]) for v in ident.values()),
                     "min_identity_p": min(v[1] for v in ident.values()),
                     "loso_mae_geometry": g_mae.mean(), "loso_mae_full": f_mae.mean(),
                     "folds_improved": improved, "n_folds": len(g_mae)})

        print(f"{sc}   (var(Y) {d.Y.var():.2e}, LID-slope r {collinearity:+.3f})")
        print(f"   in-sample: largest |identity effect| "
              f"{max(abs(v[0]) for v in ident.values()):.5f} AUROC, "
              f"smallest p {min(v[1] for v in ident.values()):.4f}")
        print(f"   AIC {aic(geo):.1f} -> {aic(full):.1f}  (computed from llf; "
              f"statsmodels reports nan under lbfgs)")
        print(f"   leave-one-seed-out MAE: geometry {g_mae.mean():.5f}"
              f"   +identity {f_mae.mean():.5f}"
              f"   improved in {improved}/{len(g_mae)} folds")
        print(f"   per-fold geometry  {np.array2string(g_mae, precision=5)}")
        print(f"   per-fold +identity {np.array2string(f_mae, precision=5)}\n")

    res = pd.DataFrame(rows)
    res.to_csv(args.out / "rq3_residual_model.csv", index=False)
    print(f"wrote {args.out / 'rq3_residual_model.csv'}")


if __name__ == "__main__":
    main()
