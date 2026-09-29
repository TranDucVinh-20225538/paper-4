"""
AR-02 manipulation check -- fixed-gamma Spectral Tempering.

Tests whether AR-02's selected mechanism (Li, Eustratiadis, Kanoulas, arXiv:2603.19339,
instantiated with a fixed, manually-swept gamma per AR-02.md's locked Step-2 rationale) actually
moves LID and/or within-class spectral-decay slope, before C3 is rerun on top of it -- same
discipline as the original (failed) beta-scaled l2-normalization check
(outputs/stage5_c3_manipulation_check.json).

Transform: z' = mu + U diag(lambda_i^(-gamma/2)) U^T (z - mu), where (mu, U, lambda) come from a
single eigendecomposition of the global covariance of the ISIC-only training population
(TRAIN_DIR, matching this project's established Mahalanobis-fitting population -- the same
population stage4_fix_adaptation_scorers.py's bugfix pinned reliability-scorer fitting to).
gamma=0 -> identity (no change, omitted below as a trivial no-op, same convention the alpha-ladder
and the original manipulation check both used for their own baseline point); gamma=1 -> full ZCA
whitening. Fixed, pre-specified ladder: gamma in {.25,.5,.75,1.0} -- per AR-02.md, chosen in advance,
not tuned by looking at results.

Applied post-hoc to the Intervention arm's alpha=1.0 checkpoints (5 seeds), not the full 5x4
alpha-cross the original manipulation check used. Justified scope reduction, not a silent one: the
original check crossed all 4 alphas because beta was defined per-checkpoint as whatever value
normalized that checkpoint's own norm to a fixed target, so beta's effective magnitude depended on
alpha. Gamma has no such dependency -- its effect on a fixed eigenbasis is the same operation
regardless of which alpha the base checkpoint sits at. What Gate 1 (EA-03) actually requires is
reproducible dose dependence across seeds; alpha=1.0 (the canonical, full-strength intervention
checkpoint, already the reference point stage4_train_intervention.py's own width-selection loop
uses) tests exactly that with 5 seeds x 4 gammas = 20 post configs, at a runtime this project's own
LID cKDTree cost (~90s/call at n=7365, d=1024, k=50 -- confirmed by direct timing, not assumed) makes
tractable; the full 4-alpha cross would have taken hours, not minutes, for no addition to what the
gate criterion checks.

Gate 1 pass criterion (EA-03, locked): at least one of LID/slope shows reproducible dose dependence
across the ladder. Gate 0 must still pass post-transform.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(data_roots.root("paper3_scripts")))
sys.path.insert(0, str(data_roots.root("csg")))

from stage4_adapter import BottleneckAdapter  # noqa: E402
from stage4_train_intervention import load_eval_population, compute_gate0, compute_gate1_measurement, TRAIN_DIR  # noqa: E402
from geometry_diagnostics import condition_number  # noqa: E402
from src.utils import ood_metrics  # noqa: E402

import data_roots

SEEDS = [42, 52, 62, 72, 82]
ALPHAS = [1.0]  # canonical full-strength checkpoint only -- see module docstring for justification
GAMMAS = [0.25, 0.5, 0.75, 1.0]
R = 16
REG_EPS = 1e-5
INTERVENTION_DIR = Path(__file__).resolve().parents[1] / "InterventionArm_Canonical"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "stage5_c3_spectral_tempering_check.json"


def fit_spectral_basis(z_train: np.ndarray, reg_eps: float = REG_EPS):
    mu = z_train.mean(axis=0).astype(np.float64)
    centered = z_train.astype(np.float64) - mu
    cov = (centered.T @ centered) / len(z_train)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals = np.clip(eigvals, reg_eps, None)
    return mu, eigvecs, eigvals


def apply_spectral_tempering(z: np.ndarray, mu, eigvecs, eigvals, gamma: float) -> np.ndarray:
    centered = z.astype(np.float64) - mu
    coords = centered @ eigvecs
    scale = eigvals ** (-gamma / 2.0)
    z_t = (coords * scale) @ eigvecs.T + mu
    return z_t.astype(np.float32)


def kappa_of(z: np.ndarray, labels: np.ndarray) -> float:
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(z, labels, num_classes=8, reg_eps=REG_EPS)
    return float(condition_number(precision))


@torch.no_grad()
def delta_z(adapter: BottleneckAdapter, z: np.ndarray) -> np.ndarray:
    adapter.eval()
    return adapter.delta(torch.tensor(z, dtype=torch.float32)).numpy()


def main() -> None:
    z_eval, meta_eval = load_eval_population()
    labels = meta_eval["label_idx"].to_numpy().astype(np.int64)
    z_train_isic = np.load(TRAIN_DIR / "embeddings.npy").astype(np.float32)

    mu, eigvecs, eigvals = fit_spectral_basis(z_train_isic)
    print(f"[spectral-tempering-check] global covariance spectrum: "
          f"lambda_max={eigvals.max():.4f} lambda_min={eigvals.min():.6f} "
          f"(floor {REG_EPS}) n_dims={len(eigvals)}")

    results = []
    for seed in SEEDS:
        ckpt = torch.load(INTERVENTION_DIR / f"adapter_seed{seed}.pt", map_location="cpu", weights_only=True)
        adapter = BottleneckAdapter(r=R)
        adapter.load_state_dict(ckpt["adapter"])
        dz = delta_z(adapter, z_eval)

        for alpha in ALPHAS:
            z_pre = z_eval + alpha * dz

            gate0_pre = compute_gate0(z_pre, meta_eval)
            gate1_pre = compute_gate1_measurement(z_pre, meta_eval)
            kappa_pre = kappa_of(z_pre, labels)

            for gamma in GAMMAS:
                z_post = apply_spectral_tempering(z_pre, mu, eigvecs, eigvals, gamma)

                gate0_post = compute_gate0(z_post, meta_eval)
                gate1_post = compute_gate1_measurement(z_post, meta_eval)
                kappa_post = kappa_of(z_post, labels)

                row = {
                    "seed": seed, "alpha": alpha, "gamma": gamma,
                    "LID_pre": gate1_pre["lid_mean"], "LID_post": gate1_post["lid_mean"],
                    "slope_pre": gate1_pre["spectral_decay_slope"], "slope_post": gate1_post["spectral_decay_slope"],
                    "kappa_pre": kappa_pre, "kappa_post": kappa_post,
                    "gate0_pass_pre": gate0_pre["gate0_pass"], "gate0_pass_post": gate0_post["gate0_pass"],
                    "domain_probe_acc_post": gate0_post["domain_probe_accuracy_mean"],
                    "id_task_bal_acc_post": gate0_post["id_task_balanced_accuracy_mean"],
                }
                results.append(row)
                print(f"[spectral-tempering-check] seed={seed} alpha={alpha} gamma={gamma}: "
                      f"LID {row['LID_pre']:.4f}->{row['LID_post']:.4f} (d={row['LID_post']-row['LID_pre']:+.4f}) "
                      f"slope {row['slope_pre']:.6f}->{row['slope_post']:.6f} (d={row['slope_post']-row['slope_pre']:+.6f}) "
                      f"kappa {row['kappa_pre']:.1f}->{row['kappa_post']:.1f} "
                      f"Gate0={'PASS' if row['gate0_pass_post'] else 'FAIL'}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[spectral-tempering-check] {len(results)} rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
