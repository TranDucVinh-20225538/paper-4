"""
LID and within-class spectral-decay-slope diagnostics for Stage 0 (C1).

Implements exactly the two geometric quantities Prediction 1
(`Scientific_Hypothesis.md`) names and `08_Mechanism_Map.md` Mechanism 1
sources from arXiv:2510.15202 ("Dissecting Mahalanobis: How Feature Geometry
and Normalization Shape OOD Detection"), Appendix A/B and Eq. 6:

  1. Local Intrinsic Dimensionality (LID) -- Levina-Bickel MLE estimator,
     k=50, dataset-average.
  2. Within-class spectral-decay slope -- log-eigenvalue-vs-rank linear fit
     on the within-class scatter matrix S_w, full range, magnitude reported.
  3. Composite P = LID * |slope| (Eq. 6, m(beta)*|s(beta)| with no
     beta-transform applied -- Stage 0 does not include the paper's
     normalization sweep, see 01_stage0_retrospective.md).

No other geometry metric is added here. Condition number is reused as-is
from paper-3/results/e1_geometry_metrics.csv, not recomputed -- per
Vinh's Protocol Execution Mode instruction, this module does not introduce
metrics beyond what Prediction 1 and C1 (02_Claim_Matrix.md) specify.

This module reads only already-extracted (features, labels) arrays. It does
not load checkpoints or datasets, and it does not import from or modify
paper-3/ or CSG-Skin/ -- Paper 4 owns this code independently, per
paper-3/CLAUDE.md's scope boundary (paper-3's own geometry_metric_audit.md
locked a different, closed 3-metric set for Paper 3's own E1; this is a
separate, Paper-4-owned computation over the same read-only npz inputs).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree

LID_K = 50  # arXiv:2510.15202 Appendix A: k fixed at 50 for the main composite metric.


class LidSpectralDiagnostics(NamedTuple):
    lid_mean: float
    lid_k: int
    spectral_slope: float  # magnitude |b|, log(lambda_i) = a + b*i
    composite: float  # lid_mean * spectral_slope
    n_samples: int
    feat_dim: int
    num_classes: int


def local_intrinsic_dimensionality(features: np.ndarray, k: int = LID_K) -> np.ndarray:
    """
    Levina-Bickel (2004) MLE LID estimator, per-point:

        LID_k(z) = -[ (1/k) * sum_{j=1}^k log(r_j(z) / r_k(z)) ]^{-1}

    r_j(z) is the distance from z to its j-th nearest neighbor WITHIN THE
    SAME feature set (self excluded). Returns one LID value per row of
    `features`; the "dataset-average LID" the composite uses is this array's
    mean, taken by the caller.

    cKDTree over raw (untransformed) features -- same space
    e1_geometry_metrics.csv's condition number and the Mahalanobis estimator
    both already operate in, per this module's docstring.
    """
    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n <= k:
        raise ValueError(f"n_samples={n} must exceed k={k} for LID estimation.")

    tree = cKDTree(features)
    # k+1 neighbors requested: index 0 is the point itself (distance 0), dropped below.
    distances, _ = tree.query(features, k=k + 1)
    r = distances[:, 1:]  # (n, k), r[:, 0] = r_1 ... r[:, k-1] = r_k, self-distance dropped

    r_k = r[:, -1:]  # (n, 1)
    # Guard against coincident points (r_j == 0) producing log(0); such
    # points are geometrically degenerate duplicates, not a modeling choice.
    with np.errstate(divide="ignore"):
        log_ratios = np.log(np.clip(r / r_k, a_min=1e-300, a_max=None))
    mean_log_ratio = np.mean(log_ratios, axis=1)  # (n,)

    lid = np.full(n, np.nan, dtype=np.float64)
    valid = mean_log_ratio < 0  # log(r_j/r_k) <= 0 always; strictly < 0 unless all k tied at r_k
    lid[valid] = -1.0 / mean_log_ratio[valid]
    return lid


def within_class_scatter(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    S_w = (1/N) * sum_c sum_{i: y_i=c} (z_i - mu_c)(z_i - mu_c)^T

    Unregularized by construction (arXiv:2510.15202 Appendix B, Eq. 27) --
    deliberately NOT the regularized precision matrix
    compute_mahalanobis_params_from_arrays fits, since regularization would
    distort exactly the tail-eigenvalue structure this metric measures.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    n, d = features.shape

    s_w = np.zeros((d, d), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            raise ValueError(f"Class {c} has zero samples -- num_classes/labels mismatch.")
        centered = features[mask] - features[mask].mean(axis=0)
        s_w += centered.T @ centered
    return s_w / n


def spectral_decay_slope(s_w: np.ndarray) -> float:
    """
    Sort eigenvalues of S_w descending, fit log(lambda_i) = a + b*i by
    ordinary least squares over the FULL range i=1..d (arXiv:2510.15202
    Appendix A: "full-range fit across all d dimensions"). Returns |b|.
    """
    eigvals = np.linalg.eigvalsh(s_w)
    eigvals = np.sort(eigvals)[::-1]
    if np.any(eigvals <= 0):
        raise ValueError(
            "S_w has non-positive eigenvalue(s); spectral-decay slope requires a strictly "
            "positive-definite within-class scatter matrix (unexpected at this project's n>>d)."
        )

    d = eigvals.shape[0]
    i = np.arange(1, d + 1, dtype=np.float64)
    log_lambda = np.log(eigvals)

    # OLS: log_lambda = a + b*i
    design = np.vstack([np.ones_like(i), i]).T
    (a, b), *_ = np.linalg.lstsq(design, log_lambda, rcond=None)
    return float(abs(b))


def compute_lid_spectral_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    k: int = LID_K,
) -> LidSpectralDiagnostics:
    """Single entry point: LID (dataset-average) x |within-class spectral-decay slope|."""
    lid_values = local_intrinsic_dimensionality(features, k=k)
    lid_mean = float(np.nanmean(lid_values))

    s_w = within_class_scatter(features, labels, num_classes)
    slope = spectral_decay_slope(s_w)

    n_samples, feat_dim = features.shape
    return LidSpectralDiagnostics(
        lid_mean=lid_mean,
        lid_k=k,
        spectral_slope=slope,
        composite=lid_mean * slope,
        n_samples=n_samples,
        feat_dim=feat_dim,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    # Smoke test only -- synthetic data, no Paper 3 checkpoints touched.
    rng = np.random.default_rng(42)
    n_classes, feat_dim, per_class = 8, 16, 2000

    true_means = rng.normal(scale=3.0, size=(n_classes, feat_dim))
    synth_features = np.concatenate(
        [rng.normal(loc=true_means[c], scale=1.0, size=(per_class, feat_dim)) for c in range(n_classes)]
    )
    synth_labels = np.repeat(np.arange(n_classes), per_class)

    diagnostics = compute_lid_spectral_diagnostics(synth_features, synth_labels, num_classes=n_classes)
    print(diagnostics)
