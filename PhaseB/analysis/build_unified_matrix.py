"""
Joins the Paper 3 and Paper 4 run matrices under one column vocabulary.

The two studies do not measure the same things, and the point of this file is to
make that legible rather than to hide it. Three traps are handled explicitly,
because a naive rename would create false equivalences that are easy to compute
with and wrong:

  1. Directed AUROC is not comparable across the two as a raw number. Paper 3's
     Mahalanobis sits near 0.402 under the higher-distance-is-more-OOD
     convention, which that study is explicit is an inverted ordering rather
     than an absent signal. Paper 4's sits near 0.998. Both are kept in
     *_auroc_directed, and an orientation-free companion, max(a, 1-a), is added
     for the three scorers the studies share, so a cross-study statement has a
     column it can defensibly use.

  2. Probe columns measure different quantities. Paper 3 reports probe AUROC
     for three classifier families; Paper 4 reports mean domain-probe accuracy.
     They keep separate names. Nothing merges them.

  3. k-NN is evaluated at k = 1, 10, 50 in Paper 3 and only k = 10 in Paper 4.
     Only knn10 is shared; the others stay study-specific.

Geometry overlaps in one metric, condition number, and even there the scales
differ by three orders of magnitude because the representations differ in
dimension and provenance. The column is shared; comparing its values across
studies is not licensed by that.

Neither source file is modified. Paper 3's matrix is published inside the
archived supplement, so rewriting it would put the repository and the archive
out of step.

    python3 PhaseB/analysis/build_unified_matrix.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import data_roots

SHARED_SCORERS = {"mahalanobis": "ood_mahalanobis", "cosine": "ood_cosine", "knn10": "ood_knn10"}


def orientation_free(a: pd.Series) -> pd.Series:
    """max(a, 1-a): separability with the score direction discarded."""
    return a.where(a >= 0.5, 1.0 - a)


def from_paper3(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    out = pd.DataFrame({
        "study": "paper3",
        "unit_id": d.checkpoint_id,
        "seed": d.seed,
        "condition": d.rung,
        "dose_kind": "lambda_orth",
        "dose_value": d.lambda_orth,
        "in_primary_population": True,
        "geom_condition_number": d.condition_number,
        "geom_fisher_hl": d.fisher_hl,
        "geom_fisher_scalar": d.fisher_scalar,
        "geom_mardia_b": d.mardia_b,
        "geom_mardia_z": d.mardia_z,
        "ood_mahalanobis_auroc_directed": d.mahalanobis_auroc,
        "ood_cosine_auroc_directed": d.cosine_auroc,
        "ood_knn10_auroc_directed": d.knn_10_auroc,
        "ood_knn1_auroc_directed": d.knn_1_auroc,
        "ood_knn50_auroc_directed": d.knn_50_auroc,
        "probe_lr_auroc": d.probe_lr_auroc,
        "probe_svm_auroc": d.probe_svm_auroc,
        "probe_rf_auroc": d.probe_rf_auroc,
    })
    return out


def from_paper4(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    return pd.DataFrame({
        "study": "paper4",
        "unit_id": d.arm + "_" + d.config.astype(str) + "_s" + d.seed.astype("Int64").astype(str),
        "seed": d.seed,
        "condition": d.arm + "/" + d.config.astype(str),
        "dose_kind": "alpha",
        "dose_value": d.alpha,
        "in_primary_population": d.in_c2_population,
        "geom_condition_number": d.condition_number,
        "geom_lid_mean": d.lid_mean,
        "geom_spectral_decay_slope": d.spectral_decay_slope,
        "geom_per_direction_variance_w": d.per_direction_variance_w,
        "ood_mahalanobis_auroc_directed": d.maha_auroc,
        "ood_cosine_auroc_directed": d.cosine_auroc,
        "ood_knn10_auroc_directed": d.knn_k10_auroc,
        "ood_kde_auroc_directed": d.kde_auroc,
        "probe_domain_accuracy": d.domain_probe_accuracy_mean,
        "id_task_balanced_accuracy": d.id_task_balanced_accuracy_mean,
        "gate0_pass": d.gate0_pass,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper3", type=Path,
                    default=data_roots.root("paper3") / "results" / "checkpoint_results_matrix.csv")
    ap.add_argument("--paper4", type=Path, default=Path("PhaseB/outputs/raw_run_matrix.csv"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/unified_run_matrix.csv"))
    args = ap.parse_args()

    df = pd.concat([from_paper3(args.paper3), from_paper4(args.paper4)], ignore_index=True)

    for short, prefix in SHARED_SCORERS.items():
        df[f"{prefix}_separability"] = orientation_free(df[f"{prefix}_auroc_directed"])

    lead = ["study", "unit_id", "seed", "condition", "dose_kind", "dose_value",
            "in_primary_population"]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + sorted(rest)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} rows: " + ", ".join(f"{k} {v}" for k, v in df.study.value_counts().items()))
    shared = [c for c in rest if df.groupby("study")[c].count().gt(0).all()]
    only3 = [c for c in rest if df[df.study == "paper3"][c].notna().any()
             and not df[df.study == "paper4"][c].notna().any()]
    only4 = [c for c in rest if df[df.study == "paper4"][c].notna().any()
             and not df[df.study == "paper3"][c].notna().any()]
    print(f"\nshared by both studies ({len(shared)}): {', '.join(shared)}")
    print(f"\npaper 3 only ({len(only3)}): {', '.join(only3)}")
    print(f"\npaper 4 only ({len(only4)}): {', '.join(only4)}")

    print("\ndirected vs orientation-free, mean by study:")
    for prefix in SHARED_SCORERS.values():
        g = df.groupby("study")[[f"{prefix}_auroc_directed", f"{prefix}_separability"]].mean()
        print(f"  {prefix}: " + "  ".join(
            f"{s} {r.iloc[0]:.3f}->{r.iloc[1]:.3f}" for s, r in g.iterrows()))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
