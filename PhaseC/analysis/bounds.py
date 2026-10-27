"""Numerical evaluation of the invariance bounds the audit compares against what was observed.

Two scorers, two bounds, both evaluated from quantities retained at transform
time (operators.py meta) and checked against the score change actually seen.

Theorem 2 (ridge-regularised Mahalanobis under an affine map). The scorer fits a
pooled covariance and adds a ridge eps I. After z -> A(z - mu) + mu + b the
refit covariance is C + eps I with C = A Sigma A^T, whereas exact affine
invariance would need A (Sigma + eps I) A^T = C + eps A A^T. The two quadratic
forms on the residual y = A(z - mu_c) of the nearest class are
    q_B = y^T B^{-1} y,  B = C + eps I          (what the refit scorer computes)
    q_D = y^T D^{-1} y,  D = C + eps A A^T      (= the score before the map)
and since B^{-1} - D^{-1} = eps B^{-1}(A A^T - I) D^{-1},
    |q_B - q_D| <= eps ||y||^2 ||I - A A^T||_2 / [(lmin(C) + eps)(lmin(C) + eps sigma_min(A)^2)].
Sigma must be the unregularised pooled within-class covariance (divisor N - K)
for q_B to equal the refit scorer's score exactly; the global covariance of the fit
population is accepted and gives an approximation. When Sigma is rank
deficient lmin(C) = 0 and the bound is loose by design. theorem2_evaluate also
reports a bound that is valid for the *minimum over classes*, which the
single-class form is not.

Lemma 2 (the scorer as implemented: cosine to class centroids, measured from the
origin). Both the evaluation vector and the centroids move; the relative
displacement of a vector v under T(v) = A(v - mu) + mu + b is
    r(v) = ||(A - I)(v - mu) + b|| / ||v||        (b = 0 in the brief)
and for centroid mu_j the same with mu_j in place of v. The numerical form used
here, as specified in the brief, is
    predicted = min(2, max_v r(v) + max_j r_mu_j),
a triangle-style bound on the change of a cosine score in [0, 2]; it is a
retained-quantity form, not the kappa_2 statement of Lemma 1. Lemma 1 is also
evaluated for comparison: with kappa = cond(A), eta = (kappa^2 - 1)/(kappa^2 + 1),
bound = min(2, 2 eta / (1 - eta)). The observed change is |s_after - s_before| per
point; tightness = observed / predicted, reported for the max and the median
(a bound that holds has tightness <= 1; tightness near 0 is a loose bound).

Spectra of class-centred data use a relative cutoff: eigenvalues below
relative_cutoff * lambda_max (default 1e-6) are dropped, and the cutoff and
the number kept are recorded, so rank is decided by a ratio and not by a
machine-dependent absolute floor.

No command line.
"""
from __future__ import annotations

import numpy as np

try:  # imported as a package (PhaseC.analysis) or as a flat directory on sys.path
    from .scorers import cosine_scores
except ImportError:
    from scorers import cosine_scores

__all__ = ["REL_CUTOFF", "pooled_within_class_cov", "class_centred_spectrum",
           "theorem2_bound", "theorem2_evaluate", "lemma1_kappa_bound",
           "lemma2_cosine_bound"]

REL_CUTOFF = 1e-6
EPS = 1e-5


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def pooled_within_class_cov(Z, y):
    """Unregularised pooled within-class covariance, sum_c Xc^T Xc / (N - K); and class means."""
    Z, y = _f64(Z), np.asarray(y)
    keep = y >= 0
    Z, y = Z[keep], y[keep]
    classes = np.unique(y)
    means = np.stack([Z[y == c].mean(0) for c in classes])
    S = np.zeros((Z.shape[1],) * 2)
    for c, m in zip(classes, means):
        Xc = Z[y == c] - m
        S += Xc.T @ Xc
    return S / (len(Z) - len(classes)), classes, means


def class_centred_spectrum(Z, y=None, relative_cutoff=REL_CUTOFF):
    """Eigendecomposition of the class-centred scatter with a relative rank cutoff.

    With y, each class is centred on its own mean (pooled within-class
    covariance); without, Z is centred on its global mean. Eigenvalues below
    relative_cutoff * lambda_max are dropped. Returns lam and U (descending, kept
    only), and lam_max, cutoff, n_kept, n_total, relative_cutoff.
    """
    if y is None:
        P = _f64(Z)
        S = np.cov((P - P.mean(0)).T)
    else:
        S, _, _ = pooled_within_class_cov(Z, y)
    lam, U = np.linalg.eigh(S)
    lam, U = lam[::-1], U[:, ::-1]
    lam_max = float(lam[0])
    cutoff = relative_cutoff * lam_max
    keep = lam >= cutoff
    return {"lam": lam[keep], "U": U[:, keep], "lam_max": lam_max, "cutoff": cutoff,
            "n_kept": int(keep.sum()), "n_total": int(len(lam)), "relative_cutoff": relative_cutoff}


def _sym_min_eig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.T))[0])


def theorem2_bound(A, Sigma, y, eps=EPS):
    """Theorem 2 inputs, predicted bound and observed |q_B - q_D| for residuals y (n x d).

    y[i] = A (z_i - mu_c(i)) for the class the point is assigned to. Returns a
    dict holding every input (A, Sigma, eps, singular values, lmin(C), the
    norms) and per-point arrays q_B, q_D, observed, predicted, tightness, plus
    their maxima. lmin(C) is clipped at 0 (it is >= 0 in exact arithmetic) and
    the raw value kept.
    """
    A, Sigma, y = _f64(A), _f64(Sigma), np.atleast_2d(_f64(y))
    d = A.shape[0]
    I = np.eye(d)
    AAT = A @ A.T
    C = A @ Sigma @ A.T
    B = C + eps * I
    D = C + eps * AAT
    lmin_raw = _sym_min_eig(C)
    lmin = max(lmin_raw, 0.0)
    sv = np.linalg.svd(A, compute_uv=False)
    smin = float(sv[-1])
    gap = float(np.linalg.norm(I - AAT, 2))
    qB = np.einsum("ij,ij->i", y, np.linalg.solve(B, y.T).T)
    qD = np.einsum("ij,ij->i", y, np.linalg.solve(D, y.T).T)
    observed = np.abs(qB - qD)
    y2 = np.einsum("ij,ij->i", y, y)
    denom = (lmin + eps) * (lmin + eps * smin ** 2)
    predicted = eps * y2 * gap / denom
    with np.errstate(divide="ignore", invalid="ignore"):
        tight = np.where(predicted > 0, observed / predicted, np.nan)
    return {
        "A": A, "Sigma": Sigma, "eps": eps, "singular_values": sv, "sigma_min": smin,
        "I_minus_AAT_2": gap, "lambda_min_C": lmin, "lambda_min_C_raw": lmin_raw,
        "y_sq_norm": y2, "q_B": qB, "q_D": qD, "observed": observed,
        "predicted": predicted, "tightness": tight,
        "observed_max": float(observed.max()), "predicted_max": float(predicted.max()),
        "bound_holds": bool(np.all(observed <= predicted * (1 + 1e-9) + 1e-12)),
        "tightness_max": float(np.nanmax(tight)) if np.isfinite(tight).any() else float("nan"),
        "tightness_median": float(np.nanmedian(tight)) if np.isfinite(tight).any() else float("nan"),
    }


def theorem2_evaluate(A, Sigma, Z_eval, means, eps=EPS):
    """Theorem 2 for a whole evaluation set, with the min over classes handled honestly.

    For each point the nearest class is chosen under D (the pre-map score, i.e.
    the score the unmapped scorer assigns) and the single-class bound is
    evaluated there: result["single_class"]. Because the scorer takes a minimum
    and |min_c a_c - min_c b_c| <= max_c |a_c - b_c|, a bound valid for the
    reported score change is the per-point maximum of the class bounds, and the
    observed change of the actual min scores is compared with it:
    result["min_over_classes"] (observed_score, predicted, tightness). means are
    the pre-map class means; Z_eval the pre-map evaluation vectors; the
    translation b cancels in the residuals.
    """
    A, Sigma = _f64(A), _f64(Sigma)
    Z, M = _f64(Z_eval), _f64(means)
    d = A.shape[0]
    AAT = A @ A.T
    C = A @ Sigma @ A.T
    B = C + eps * np.eye(d)
    D = C + eps * AAT
    Ys = np.stack([(Z - m) @ A.T for m in M])                    # K x n x d
    qB = np.stack([np.einsum("ij,ij->i", Y, np.linalg.solve(B, Y.T).T) for Y in Ys])
    qD = np.stack([np.einsum("ij,ij->i", Y, np.linalg.solve(D, Y.T).T) for Y in Ys])
    nearest = qD.argmin(0)
    y_near = Ys[nearest, np.arange(len(Z))]
    single = theorem2_bound(A, Sigma, y_near, eps)
    lmin = single["lambda_min_C"]
    factor = eps * single["I_minus_AAT_2"] / ((lmin + eps) * (lmin + eps * single["sigma_min"] ** 2))
    pred_all = factor * np.einsum("kij,kij->ki", Ys, Ys)
    pred_min = pred_all.max(0)
    obs_min = np.abs(qB.min(0) - qD.min(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tight = np.where(pred_min > 0, obs_min / pred_min, np.nan)
    return {
        "single_class": single, "nearest_class": nearest,
        "min_over_classes": {
            "observed_score": obs_min, "predicted": pred_min, "tightness": tight,
            "observed_max": float(obs_min.max()), "predicted_max": float(pred_min.max()),
            "bound_holds": bool(np.all(obs_min <= pred_min * (1 + 1e-9) + 1e-12)),
            "tightness_max": float(np.nanmax(tight)) if np.isfinite(tight).any() else float("nan"),
            "tightness_median": float(np.nanmedian(tight)) if np.isfinite(tight).any() else float("nan")},
        "score_before": qD.min(0), "score_after": qB.min(0),
    }


def lemma1_kappa_bound(A=None, singular_values=None):
    """Lemma 1 (kappa_2 form): eta = (k^2 - 1)/(k^2 + 1), bound = min(2, 2 eta / (1 - eta)).

    kappa = sigma_max / sigma_min of A. Note 2 eta / (1 - eta) = k^2 - 1.
    """
    sv = _f64(singular_values) if singular_values is not None else np.linalg.svd(_f64(A), compute_uv=False)
    kappa = float(sv.max() / sv.min()) if sv.min() > 0 else float("inf")
    if not np.isfinite(kappa):
        return {"kappa": kappa, "eta": 1.0, "bound": 2.0}
    eta = (kappa ** 2 - 1.0) / (kappa ** 2 + 1.0)
    bound = min(2.0, 2.0 * eta / (1.0 - eta)) if eta < 1 else 2.0
    return {"kappa": kappa, "eta": float(eta), "bound": float(bound)}


def _rel_disp(V, mu, A, b):
    """||(A - I)(v - mu) + b|| / ||v||, NaN where ||v|| = 0."""
    V = _f64(V)
    disp = (V - mu) @ (A - np.eye(A.shape[0])).T
    if b is not None:
        disp = disp + b
    n = np.linalg.norm(V, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, np.linalg.norm(disp, axis=1) / n, np.nan), int((n == 0).sum())


def lemma2_cosine_bound(Z_eval, means, mu, A, b=None, s_before=None, s_after=None):
    """Lemma 2 numerical form for the origin-based cosine-to-centroid scorer.

    Z_eval and means are pre-transform; the map is T(v) = A (v - mu) + mu + b.
    If s_before / s_after are not given they are computed with the cosine scorer
    on the original and mapped vectors and centroids (the mapped class means of
    an affine map are the map of the means). Zero-norm vectors or centroids
    are excluded from the maxima and counted (n_zero_norm_*).

    predicted_bound = min(2, max_v r(v) + max_j r_mu_j) (global);
    predicted_per_point = min(2, r(v_i) + max_j r_mu_j). Also returns the Lemma 1
    kappa bound, observed per-point and max/median, and tightness against each.
    """
    A, mu = _f64(A), _f64(mu)
    Z, M = _f64(Z_eval), _f64(means)
    bb = None if b is None else _f64(b)
    r_v, n0_v = _rel_disp(Z, mu, A, bb)
    r_mu, n0_m = _rel_disp(M, mu, A, bb)
    r_v_max = float(np.nanmax(r_v)) if np.isfinite(r_v).any() else float("nan")
    r_mu_max = float(np.nanmax(r_mu)) if np.isfinite(r_mu).any() else float("nan")
    predicted = min(2.0, r_v_max + r_mu_max)
    pred_pt = np.minimum(2.0, r_v + r_mu_max)

    def T(V):
        out = (V - mu) @ A.T + mu
        return out if bb is None else out + bb

    if s_before is None:
        s_before = cosine_scores(Z, M)
    if s_after is None:
        s_after = cosine_scores(T(Z), T(M))
    obs = np.abs(np.asarray(s_after, dtype=np.float64) - np.asarray(s_before, dtype=np.float64))
    l1 = lemma1_kappa_bound(A)
    with np.errstate(divide="ignore", invalid="ignore"):
        tight_pt = np.where(pred_pt > 0, obs / pred_pt, np.nan)
    return {
        "r_v": r_v, "r_mu": r_mu, "r_v_max": r_v_max, "r_mu_max": r_mu_max,
        "n_zero_norm_vectors": n0_v, "n_zero_norm_centroids": n0_m,
        "predicted_bound": predicted, "predicted_per_point": pred_pt,
        "lemma1": l1, "lemma1_bound": l1["bound"],
        "observed_per_point": obs, "observed_max": float(obs.max()),
        "observed_median": float(np.median(obs)),
        "tightness_max": float(obs.max() / predicted) if predicted > 0 else float("nan"),
        "tightness_median": float(np.median(obs) / predicted) if predicted > 0 else float("nan"),
        "tightness_per_point": tight_pt,
        "bound_holds": bool(obs.max() <= predicted * (1 + 1e-9) + 1e-12),
        "lemma1_bound_holds": bool(obs.max() <= l1["bound"] * (1 + 1e-9) + 1e-12),
    }
