"""
Stage 0 (C1) retrospective driver -- 01_stage0_retrospective.md.

For each of Paper 3's 13 primary-ladder checkpoints (runA_grl x5, runB_orth1
x5, runB x3 -- lambda_orth = 0, 1, 5), computes LID and within-class
spectral-decay slope on the already-extracted E1 embeddings
(paper-3/results/e1_embeddings/*.npz), forms the composite
P = LID * |slope|, and tests Kendall's tau (exact) against Mahalanobis-AUROC
(paper-3/results/e2_auroc.csv) -- the C1 gate test specified in
22_Experimental_Design.md Section 6 and 24_PreRegistration.md Sections 5/8.

Reads paper-3/results/ only. Writes only under Paper4/PhaseB/outputs/.
Never modifies anything under paper-3/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from lid_spectral_decay import compute_lid_spectral_diagnostics

PAPER3_RESULTS = Path(__file__).resolve().parents[3] / "paper-3" / "results"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

PRIMARY_RUNGS = ["runA_grl", "runB_orth1", "runB"]  # lambda_orth = 0, 1, 5
COMMON_SEEDS = {42, 52, 62}  # all three rungs have these; runB has ONLY these


def load_checkpoint_list() -> pd.DataFrame:
    """13 (rung, seed) rows from e1_geometry_metrics.csv -- the authoritative checkpoint list."""
    geom = pd.read_csv(PAPER3_RESULTS / "e1_geometry_metrics.csv")
    geom = geom[geom["rung"].isin(PRIMARY_RUNGS)].copy()
    unexpected = set(geom["rung"].unique()) - set(PRIMARY_RUNGS)
    if unexpected:
        raise ValueError(f"Unexpected rung(s) in e1_geometry_metrics.csv: {unexpected}")
    return geom[["rung", "seed", "checkpoint_path", "condition_number"]]


def compute_lid_spectral_for_all(checkpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in checkpoints.iterrows():
        npz_path = PAPER3_RESULTS / "e1_embeddings" / f"{row['rung']}_s{row['seed']}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Expected E1 embedding file not found: {npz_path}")
        data = np.load(npz_path)
        features, labels = data["features"], data["labels"]
        num_classes = int(np.unique(labels).size)

        diag = compute_lid_spectral_diagnostics(features, labels, num_classes=num_classes)
        rows.append(
            {
                "rung": row["rung"],
                "seed": int(row["seed"]),
                "checkpoint_path": row["checkpoint_path"],
                "condition_number": row["condition_number"],
                "lid_mean": diag.lid_mean,
                "lid_k": diag.lid_k,
                "spectral_slope": diag.spectral_slope,
                "composite_lid_x_slope": diag.composite,
                "n_samples": diag.n_samples,
                "feat_dim": diag.feat_dim,
                "num_classes": diag.num_classes,
            }
        )
        print(
            f"[stage0] {row['rung']:<11s} s{row['seed']:<3d} "
            f"LID={diag.lid_mean:8.4f}  |slope|={diag.spectral_slope:.6f}  "
            f"P={diag.composite:.6f}  kappa={row['condition_number']:.2f}"
        )
    return pd.DataFrame(rows)


def join_auroc(geometry: pd.DataFrame) -> pd.DataFrame:
    auroc = pd.read_csv(PAPER3_RESULTS / "e2_auroc.csv")[["rung", "seed", "auroc", "fpr95"]]
    merged = geometry.merge(auroc, on=["rung", "seed"], how="inner", validate="one_to_one")
    if len(merged) != len(geometry):
        raise ValueError(
            f"AUROC join dropped rows: {len(geometry)} geometry rows -> {len(merged)} merged rows. "
            "e2_auroc.csv does not cover the same 13 checkpoints as e1_geometry_metrics.csv."
        )
    return merged


def kendall_exact(x: np.ndarray, y: np.ndarray) -> dict:
    tau, p = kendalltau(x, y, method="exact")
    return {"tau": float(tau), "p_exact": float(p), "n": int(len(x))}


def run_c1_tests(merged: pd.DataFrame) -> pd.DataFrame:
    full = merged
    common = merged[merged["seed"].isin(COMMON_SEEDS)]

    candidates = {
        "composite_lid_x_slope": "PRIMARY (C1 pre-registered test)",
        "lid_mean": "secondary (composite decomposition)",
        "spectral_slope": "secondary (composite decomposition)",
        "condition_number": "secondary (Paper 3 continuity metric)",
    }

    results = []
    for metric, role in candidates.items():
        r_full = kendall_exact(full[metric].to_numpy(), full["auroc"].to_numpy())
        r_common = kendall_exact(common[metric].to_numpy(), common["auroc"].to_numpy())
        results.append(
            {
                "metric": metric,
                "role": role,
                "target": "auroc",
                "tau_full": r_full["tau"],
                "p_full_exact": r_full["p_exact"],
                "n_full": r_full["n"],
                "tau_common_seed": r_common["tau"],
                "p_common_seed_exact": r_common["p_exact"],
                "n_common_seed": r_common["n"],
            }
        )
        print(
            f"[stage0] {metric:<24s} tau_full={r_full['tau']:+.4f} (p={r_full['p_exact']:.6f}, n={r_full['n']})  "
            f"tau_common={r_common['tau']:+.4f} (p={r_common['p_exact']:.6f}, n={r_common['n']})  [{role}]"
        )
    return pd.DataFrame(results)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = load_checkpoint_list()
    print(f"[stage0] {len(checkpoints)} primary-ladder checkpoints "
          f"({', '.join(f'{r}: n={len(checkpoints[checkpoints.rung==r])}' for r in PRIMARY_RUNGS)})\n")

    geometry = compute_lid_spectral_for_all(checkpoints)
    geometry.to_csv(OUTPUT_DIR / "stage0_geometry_metrics.csv", index=False)

    merged = join_auroc(geometry)
    merged.to_csv(OUTPUT_DIR / "stage0_geometry_auroc_merged.csv", index=False)

    print("\n=== C1 gate: Kendall's tau (exact) vs. Mahalanobis-AUROC ===")
    c1_results = run_c1_tests(merged)
    c1_results.to_csv(OUTPUT_DIR / "stage0_c1_kendall_tau.csv", index=False)

    primary = c1_results[c1_results["metric"] == "composite_lid_x_slope"].iloc[0]
    summary = {
        "gate": "Stage 0 / C1",
        "primary_metric": "composite_lid_x_slope",
        "tau_full": primary["tau_full"],
        "p_full_exact": primary["p_full_exact"],
        "n_full": int(primary["n_full"]),
        "tau_common_seed": primary["tau_common_seed"],
        "p_common_seed_exact": primary["p_common_seed_exact"],
        "n_common_seed": int(primary["n_common_seed"]),
    }
    with open(OUTPUT_DIR / "stage0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[stage0] Outputs written to {OUTPUT_DIR}/")
    print(f"[stage0] Primary C1 result: tau_full={summary['tau_full']:+.4f} "
          f"(exact p={summary['p_full_exact']:.6f}, n={summary['n_full']})")


if __name__ == "__main__":
    main()
