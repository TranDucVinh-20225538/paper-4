"""Export per-class AM/GM anisotropy a_c and correlate with the KDE gate violation.

a_c = σ̄_c / det(Σ_c)^(1/(2d))  is computed on the *unrotated* train embeddings.
Under an isotropic within-class covariance a_c = 1 and a rotation cannot move σ̄_c
relative to c; the gap a_c - 1 is the mechanism-matched predictor of how much
mean-std Scott bandwidth (hence kde AUROC) will move under ortho_linear.

κ₂(Σ_c) is reported alongside at bounds.REL_CUTOFF (relative rank cutoff) as a
descriptor only — it depends on the cutoff; a_c does not.

    python3 PhaseC/analysis/export_kde_anisotropy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bounds  # noqa: E402
import scorers  # noqa: E402
from confirm_kde_bandwidth import BACKBONES, ID_SET, OUT  # noqa: E402
from run_audit import load_embeddings  # noqa: E402
from train_adapters import EMB_ROOT  # noqa: E402

REL_CUTOFF = bounds.REL_CUTOFF


def per_class_anisotropy(Ztr, ytr):
    Z, y = np.asarray(Ztr, np.float64), np.asarray(ytr)
    cls, sig, n_c = scorers.class_sigma_bar(Z, y)
    _, det_scale, _, _ = scorers.class_det_scale(Z, y, classes=cls)
    d = Z.shape[1]
    rows = []
    for i, c in enumerate(cls):
        C = Z[y == c]
        a = float(sig[i] / det_scale[i]) if det_scale[i] > 0 else float("inf")
        # κ₂ at the recorded relative rank cutoff
        if len(C) > 1:
            cov = np.cov(C, rowvar=False)
            lam = np.linalg.eigvalsh(cov)[::-1]
            lam_max = float(lam[0])
            keep = lam >= REL_CUTOFF * lam_max
            lam_k = lam[keep]
            kappa = float(lam_k[0] / lam_k[-1]) if len(lam_k) >= 2 and lam_k[-1] > 0 else float("nan")
            n_kept = int(keep.sum())
        else:
            kappa, n_kept, lam_max = float("nan"), 0, float("nan")
        rows.append({
            "class": int(c), "n_c": int(n_c[i]), "dim": d,
            "sigma_bar": float(sig[i]), "det_scale": float(det_scale[i]),
            "a_c": a, "kappa2": kappa, "kappa2_n_kept": n_kept,
            "kappa2_relative_cutoff": REL_CUTOFF, "lam_max": lam_max,
        })
    return pd.DataFrame(rows)


def main():
    out = OUT
    class_rows, bb_rows = [], []
    for bb in BACKBONES:
        Ztr, ytr, _, _ = load_embeddings(bb, ID_SET, EMB_ROOT)
        df = per_class_anisotropy(Ztr, ytr)
        df.insert(0, "id_set", ID_SET)
        df.insert(0, "backbone", bb)
        class_rows.append(df)
        bb_rows.append({
            "backbone": bb, "id_set": ID_SET,
            "a_c_mean": float(df.a_c.mean()), "a_c_max": float(df.a_c.max()),
            "a_c_min": float(df.a_c.min()),
            "kappa2_mean": float(df.kappa2.mean()), "kappa2_max": float(df.kappa2.max()),
            "kappa2_relative_cutoff": REL_CUTOFF, "n_classes": len(df), "dim": int(df.dim.iloc[0]),
        })
        print(f"[a_c] {bb}: mean a_c={bb_rows[-1]['a_c_mean']:.4g}  max a_c={bb_rows[-1]['a_c_max']:.4g}  "
              f"mean κ₂={bb_rows[-1]['kappa2_mean']:.4g}", flush=True)

    class_df = pd.concat(class_rows, ignore_index=True)
    bb_df = pd.DataFrame(bb_rows)
    class_path = out / "kde_anisotropy_per_class.csv"
    bb_path = out / "kde_anisotropy_per_backbone.csv"
    class_df.to_csv(class_path, index=False, float_format="%.10g")
    bb_df.to_csv(bb_path, index=False, float_format="%.10g")

    # Join to the 15 ortho_linear gate cells
    gate = pd.read_csv(out / "kde_det_gate.csv")
    sigma = pd.read_csv(out / "kde_sigma_bar.csv")
    # per (backbone, seed): max |dAUROC| kde and mean |log(σ̄_ratio/c)|
    cfg = (gate[gate.scorer == "kde"]
           .groupby(["backbone", "seed"], as_index=False)
           .agg(max_abs_dauroc=("abs_delta_auroc", "max"),
                mean_abs_log_dev=("mean_abs_log_sigma_ratio_over_c", "first")))
    # σ̄ deviation also from sigma file (same)
    cfg = cfg.merge(bb_df[["backbone", "a_c_mean", "a_c_max", "kappa2_mean", "kappa2_max"]],
                    on="backbone", how="left")

    rho_a_dev = spearmanr(cfg["a_c_mean"], cfg["mean_abs_log_dev"])
    rho_a_auc = spearmanr(cfg["a_c_mean"], cfg["max_abs_dauroc"])
    rho_amax_auc = spearmanr(cfg["a_c_max"], cfg["max_abs_dauroc"])
    rho_k_auc = spearmanr(cfg["kappa2_mean"], cfg["max_abs_dauroc"])
    # backbone-level (n=5): clearer for the ordering claim
    bb_level = (cfg.groupby("backbone", as_index=False)
                .agg(a_c_mean=("a_c_mean", "first"), a_c_max=("a_c_max", "first"),
                     kappa2_mean=("kappa2_mean", "first"),
                     mean_abs_log_dev=("mean_abs_log_dev", "mean"),
                     max_abs_dauroc=("max_abs_dauroc", "max")))
    rho5_a_dev = spearmanr(bb_level["a_c_mean"], bb_level["mean_abs_log_dev"])
    rho5_a_auc = spearmanr(bb_level["a_c_mean"], bb_level["max_abs_dauroc"])

    summary = {
        "relative_cutoff_kappa2": REL_CUTOFF,
        "n_cells": int(len(cfg)),
        "spearman_a_c_mean_vs_mean_abs_log_sigma_dev": {
            "rho": float(rho_a_dev.statistic), "p": float(rho_a_dev.pvalue), "n": int(len(cfg)),
        },
        "spearman_a_c_mean_vs_max_abs_dauroc_kde": {
            "rho": float(rho_a_auc.statistic), "p": float(rho_a_auc.pvalue), "n": int(len(cfg)),
        },
        "spearman_a_c_max_vs_max_abs_dauroc_kde": {
            "rho": float(rho_amax_auc.statistic), "p": float(rho_amax_auc.pvalue), "n": int(len(cfg)),
        },
        "spearman_kappa2_mean_vs_max_abs_dauroc_kde": {
            "rho": float(rho_k_auc.statistic), "p": float(rho_k_auc.pvalue), "n": int(len(cfg)),
        },
        "backbone_level_n5": {
            "spearman_a_c_mean_vs_mean_abs_log_sigma_dev": {
                "rho": float(rho5_a_dev.statistic), "p": float(rho5_a_dev.pvalue),
            },
            "spearman_a_c_mean_vs_max_abs_dauroc_kde": {
                "rho": float(rho5_a_auc.statistic), "p": float(rho5_a_auc.pvalue),
            },
            "rows": bb_level.to_dict(orient="records"),
        },
    }
    (out / "notes" / "kde_anisotropy_predictor.json").write_text(json.dumps(summary, indent=1) + "\n")
    cfg_path = out / "kde_anisotropy_gate_join.csv"
    cfg.to_csv(cfg_path, index=False, float_format="%.10g")
    print(f"[a_c] wrote {class_path}")
    print(f"[a_c] wrote {bb_path}")
    print(f"[a_c] wrote {cfg_path}")
    print(f"[a_c] 15-cell Spearman a_c_mean ↔ |log(σ̄_ratio/c)| ρ={rho_a_dev.statistic:.4f} p={rho_a_dev.pvalue:.3g}")
    print(f"[a_c] 15-cell Spearman a_c_mean ↔ max|ΔAUROC|kde ρ={rho_a_auc.statistic:.4f} p={rho_a_auc.pvalue:.3g}")
    print(f"[a_c] 15-cell Spearman κ₂_mean ↔ max|ΔAUROC|kde ρ={rho_k_auc.statistic:.4f} p={rho_k_auc.pvalue:.3g}")
    print(f"[a_c] 5-bb Spearman a_c_mean ↔ max|ΔAUROC|kde ρ={rho5_a_auc.statistic:.4f} p={rho5_a_auc.pvalue:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
