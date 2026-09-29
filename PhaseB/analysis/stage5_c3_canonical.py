"""
C3 canonical run (22_Experimental_Design.md SS6/SS7) -- paired comparison of scorer-AUROC
degradation at matched (normalized) geometry vs. matched (nominal) intervention strength.

Three configs per seed, Intervention arm, 5 seeds:
  A. "strong-natural"     alpha=1.0, gamma=0     -- frozen, AlphaLadder/reliability_scorers_results.json
  B. "weak-natural"       alpha=0.25, gamma=0    -- frozen, same source
  C. "strong-pulled"      alpha=1.0, gamma=0.01  -- NEW, computed here. gamma=0.01 locked from
                          stage5_c3_gamma_calibration.py's single-seed calibration (slope matched to
                          B's natural level: 0.006746 vs target 0.006751, d=-0.000005; LID not
                          matched at this gamma, d=-0.60, recorded not forced -- same discipline as
                          the manipulation check's own non-monotonic LID finding).

Two paired comparisons (across 5 seeds, per scorer), degradation = Reference-arm baseline AUROC -
config AUROC:
  1. Nominal-strength-matched, geometry different: degradation_A vs degradation_C (both alpha=1.0).
     If geometry mediates, these should differ despite identical nominal strength.
  2. Geometry-matched, nominal-strength different: degradation_B vs degradation_C (slope-matched,
     alpha 0.25 vs 1.0). If geometry mediates, these should NOT differ despite different nominal
     strength.

Both paired t-test and Wilcoxon signed-rank reported (normality not pre-committed, SS6). Holm-
Bonferroni across all 8 tests (4 scorers x 2 comparisons), per SS6's multiple-comparisons rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(data_roots.root("csg")))

from stage4_adapter import BottleneckAdapter  # noqa: E402
from stage4_train_intervention import TRAIN_DIR, load_eval_population, compute_gate0, compute_gate1_measurement  # noqa: E402
from stage5_c3_spectral_tempering_check import fit_spectral_basis, apply_spectral_tempering  # noqa: E402
from groupB_mahalanobis_family import (  # noqa: E402
    cosine_centroid_scores, compute_knn_scores, density_kde_scores, auroc_fpr95, K_VALUES, PRIMARY_K, NUM_CLASSES, REG_EPS,
)
from src.utils import ood_metrics  # noqa: E402

import data_roots

SEEDS = [42, 52, 62, 72, 82]
R = 16
GAMMA_C = 0.01  # locked, stage5_c3_gamma_calibration.py
INTERVENTION_DIR = Path(__file__).resolve().parents[1] / "InterventionArm_Canonical"
ALPHA_LADDER_DIR = Path(__file__).resolve().parents[1] / "AlphaLadder"
GROUPB_DIR = Path(__file__).resolve().parents[1] / "GroupBExecution"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "stage5_c3_results.json"

SCORERS = ["maha_auroc", "cosine_auroc", "knn_k10_auroc", "kde_auroc"]


@torch.no_grad()
def delta(adapter: BottleneckAdapter, z: np.ndarray) -> np.ndarray:
    adapter.eval()
    return adapter.delta(torch.tensor(z, dtype=torch.float32)).numpy()


def load_reference_baseline() -> dict:
    maha = json.loads((GROUPB_DIR / "maha_results.json").read_text())
    cosine = json.loads((GROUPB_DIR / "cosine_results.json").read_text())
    knn = json.loads((GROUPB_DIR / "knn_results.json").read_text())
    kde = json.loads((GROUPB_DIR / "kde_results.json").read_text())
    return {
        "maha_auroc": maha["auroc"],
        "cosine_auroc": cosine["auroc"],
        "knn_k10_auroc": knn["primary_result"]["auroc"],
        "kde_auroc": kde["auroc"],
    }


def load_frozen_alpha(alpha_target: float) -> dict:
    d = json.loads((ALPHA_LADDER_DIR / "reliability_scorers_results.json").read_text())
    rows = [r for r in d["results"]["Intervention"] if r["alpha"] == alpha_target]
    return {r["seed"]: r for r in rows}


def score(z_train_a: np.ndarray, y_train: np.ndarray, z_eval_a: np.ndarray, id_mask: np.ndarray, ood_mask: np.ndarray) -> dict:
    z_id, z_ood = z_eval_a[id_mask], z_eval_a[ood_mask]
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(z_train_a, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS)
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
    maha_auroc, maha_fpr95 = auroc_fpr95(s_id, s_ood)
    cos_auroc, cos_fpr95 = auroc_fpr95(cosine_centroid_scores(z_id, means), cosine_centroid_scores(z_ood, means))
    knn = compute_knn_scores(z_train_a, z_id, z_ood, K_VALUES)
    knn_auroc, knn_fpr95 = auroc_fpr95(*knn[PRIMARY_K])
    kde_auroc, kde_fpr95 = auroc_fpr95(density_kde_scores(z_train_a, y_train, z_id), density_kde_scores(z_train_a, y_train, z_ood))
    return {"maha_auroc": maha_auroc, "maha_fpr95": maha_fpr95, "cosine_auroc": cos_auroc, "cosine_fpr95": cos_fpr95,
            "knn_k10_auroc": knn_auroc, "knn_k10_fpr95": knn_fpr95, "kde_auroc": kde_auroc, "kde_fpr95": kde_fpr95}


def holm_bonferroni(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((n - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted.tolist()


def main() -> None:
    z_train_raw = np.load(TRAIN_DIR / "embeddings.npy").astype(np.float32)
    y_train = pd.read_csv(TRAIN_DIR / "metadata.csv")["label_idx"].to_numpy().astype(np.int64)
    z_eval_raw, meta_eval = load_eval_population()
    id_mask = (meta_eval["domain"] == "isic").to_numpy()
    ood_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()

    mu, eigvecs, eigvals = fit_spectral_basis(z_train_raw)
    baseline = load_reference_baseline()
    frozen_A = load_frozen_alpha(1.0)   # strong-natural
    frozen_B = load_frozen_alpha(0.25)  # weak-natural
    print(f"[c3-canonical] Reference baseline: {baseline}")

    rows = []
    for seed in SEEDS:
        ckpt = torch.load(INTERVENTION_DIR / f"adapter_seed{seed}.pt", map_location="cpu", weights_only=True)
        adapter = BottleneckAdapter(r=R)
        adapter.load_state_dict(ckpt["adapter"])
        dz_train = delta(adapter, z_train_raw)
        dz_eval = delta(adapter, z_eval_raw)

        z_train_pre = z_train_raw + 1.0 * dz_train
        z_eval_pre = z_eval_raw + 1.0 * dz_eval

        z_train_C = apply_spectral_tempering(z_train_pre, mu, eigvecs, eigvals, GAMMA_C)
        z_eval_C = apply_spectral_tempering(z_eval_pre, mu, eigvecs, eigvals, GAMMA_C)

        gate0_C = compute_gate0(z_eval_C, meta_eval)
        gate1_C = compute_gate1_measurement(z_eval_C, meta_eval)
        scores_C = score(z_train_C, y_train, z_eval_C, id_mask, ood_mask)

        row = {
            "seed": seed,
            "A_strong_natural": frozen_A[seed],
            "B_weak_natural": frozen_B[seed],
            "C_strong_pulled": {**scores_C, "gate0_pass": gate0_C["gate0_pass"],
                                 "lid_mean": gate1_C["lid_mean"], "spectral_decay_slope": gate1_C["spectral_decay_slope"]},
        }
        rows.append(row)
        print(f"[c3-canonical] seed={seed}: C gate0={'PASS' if gate0_C['gate0_pass'] else 'FAIL'} "
              f"LID={gate1_C['lid_mean']:.4f} slope={gate1_C['spectral_decay_slope']:.6f} "
              f"Maha={scores_C['maha_auroc']:.4f} Cos={scores_C['cosine_auroc']:.4f} "
              f"kNN10={scores_C['knn_k10_auroc']:.4f} KDE={scores_C['kde_auroc']:.4f}")

    gate0_all_pass = all(r["C_strong_pulled"]["gate0_pass"] for r in rows)

    # degradation = baseline - config AUROC, per scorer, per seed
    deg = {"A": {s: [] for s in SCORERS}, "B": {s: [] for s in SCORERS}, "C": {s: [] for s in SCORERS}}
    for r in rows:
        for s in SCORERS:
            deg["A"][s].append(baseline[s] - r["A_strong_natural"][s])
            deg["B"][s].append(baseline[s] - r["B_weak_natural"][s])
            deg["C"][s].append(baseline[s] - r["C_strong_pulled"][s])

    tests = []
    for s in SCORERS:
        a, b, c = np.array(deg["A"][s]), np.array(deg["B"][s]), np.array(deg["C"][s])

        # Comparison 1: nominal-matched (A vs C), geometry different
        d_ac = a - c
        if np.allclose(d_ac, 0.0):
            t1 = {"comparison": "nominal_matched_A_vs_C", "scorer": s, "degenerate": True,
                  "note": "A and C degradation identical across all seeds", "t_p": None, "wilcoxon_p": None}
        else:
            t_stat, t_p = stats.ttest_rel(a, c)
            try:
                w_stat, w_p = stats.wilcoxon(d_ac)
            except ValueError as e:
                w_stat, w_p = None, None
            t1 = {"comparison": "nominal_matched_A_vs_C", "scorer": s, "degenerate": False,
                  "mean_degradation_A": float(a.mean()), "mean_degradation_C": float(c.mean()),
                  "mean_diff": float(d_ac.mean()), "t_stat": float(t_stat), "t_p": float(t_p),
                  "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
                  "wilcoxon_p": float(w_p) if w_p is not None else None}
        tests.append(t1)

        # Comparison 2: geometry-matched (B vs C), nominal different
        d_bc = b - c
        if np.allclose(d_bc, 0.0):
            t2 = {"comparison": "geometry_matched_B_vs_C", "scorer": s, "degenerate": True,
                  "note": "B and C degradation identical across all seeds", "t_p": None, "wilcoxon_p": None}
        else:
            t_stat, t_p = stats.ttest_rel(b, c)
            try:
                w_stat, w_p = stats.wilcoxon(d_bc)
            except ValueError as e:
                w_stat, w_p = None, None
            t2 = {"comparison": "geometry_matched_B_vs_C", "scorer": s, "degenerate": False,
                  "mean_degradation_B": float(b.mean()), "mean_degradation_C": float(c.mean()),
                  "mean_diff": float(d_bc.mean()), "t_stat": float(t_stat), "t_p": float(t_p),
                  "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
                  "wilcoxon_p": float(w_p) if w_p is not None else None}
        tests.append(t2)

    # Holm-Bonferroni across all 8 t-test p-values (primary statistic)
    pvals = [t["t_p"] for t in tests if not t["degenerate"]]
    idxs = [i for i, t in enumerate(tests) if not t["degenerate"]]
    if pvals:
        adjusted = holm_bonferroni(pvals)
        for i, p_adj in zip(idxs, adjusted):
            tests[i]["t_p_holm"] = p_adj

    for t in tests:
        if t["degenerate"]:
            print(f"[c3-canonical] {t['comparison']} {t['scorer']}: DEGENERATE ({t['note']})")
        else:
            print(f"[c3-canonical] {t['comparison']} {t['scorer']}: mean_diff={t['mean_diff']:+.4f} "
                  f"t_p={t['t_p']:.4f} (holm={t.get('t_p_holm', float('nan')):.4f}) wilcoxon_p={t['wilcoxon_p']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({
        "gamma_C": GAMMA_C,
        "reference_baseline": baseline,
        "gate0_all_pass_config_C": gate0_all_pass,
        "raw_rows": rows,
        "degradation": deg,
        "tests": tests,
    }, indent=2))
    print(f"\n[c3-canonical] gate0_all_pass(C)={gate0_all_pass}. Results written to {OUTPUT_PATH} -- NOT frozen (pending review)")


if __name__ == "__main__":
    main()
