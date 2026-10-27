"""The OOD scorers Phase C audits, self-contained.

Phase B's score_all imports the Mahalanobis fit from the CSG-SKin repository, so
it only runs where that repository is configured. Phase C audits other
backbones and benchmarks and must not depend on it, so every scorer is
reimplemented here from the formulas Phase B used. Higher always means more
out-of-distribution.

    maha    min over classes of the squared Mahalanobis distance to the class
            mean, one shared within-class covariance
            sum_c (Xc - mc)^T (Xc - mc) / (N - K) + 1e-5 I
    cosine  1 - max_c cos(z, mu_c), mu_c the class means fit for maha. Origin
            based: the angle is measured from the coordinate origin, not from
            the fit mean, so it is not affine-invariant (bounds.py, Lemma 2)
    knn10   Euclidean distance to the 10th nearest training neighbour
    kde     -max_c log p_c(z), per-class Gaussian KDE with Scott's rule using
            the arithmetic mean of per-dimension stds (σ̄_c) and a bandwidth
            floor of 1e-6. σ̄_c is *not* rotation-invariant: a rotation
            preserves total variance but redistributes it across axes, and the
            mean of square roots is not invariant to that redistribution, so
            h_c (and density ranks) can move under T(z)=cQz.
    kde_det same KDE with h_c = det(Σ_c)^(1/(2d)) · n_c^(-1/(d+4)), floored.
            det(Σ) is rotation-invariant and scales as c^(2d) under T(z)=cQz,
            so h_c scales exactly by c and the scorer is similarity-invariant.
            Kept alongside kde to show that the invariance group depends on the
            bandwidth formula, not only on the functional form of the scorer.
    rmd     Relative Mahalanobis (Ren et al. 2021): the class-conditional
            distance minus the distance to one background Gaussian fit on all
            training points (its own full covariance, same ridge). The
            background term does not depend on the class, so
            rmd = min_c MD_c - MD_bg.

All linear algebra is float64 whatever the input dtype; inputs may be float32
and are not modified. The ridge is recorded in the returned dict rather than
assumed by the reader.

fast_auroc and fpr_at_95_tpr are the two metrics the audit reads, with OOD as
the positive class. Nothing here has a command line.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "REG_EPS", "PRIMARY_K", "KDE_BW_FLOOR", "SCORER_NAMES",
    "fit_class_gaussian", "mahalanobis_min_sq", "cosine_scores", "knn_scores",
    "class_sigma_bar", "class_det_scale", "kde_scores", "kde_det_scores",
    "relative_mahalanobis_scores", "score_all",
    "fast_auroc", "fpr_at_95_tpr", "fpr_at_tpr",
]

REG_EPS = 1e-5
PRIMARY_K = 10
KDE_BW_FLOOR = 1e-6
SCORER_NAMES = ("maha", "cosine", "knn10", "kde", "kde_det", "rmd")
_CHUNK = 2048


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _chol_lower(S):
    c, low = cho_factor(S, lower=True, check_finite=True)
    return np.tril(c)


def _sq_maha(X, mean, L):
    """||L^{-1}(x - mean)||^2 for every row of X, L the lower Cholesky of the covariance."""
    R = solve_triangular(L, (X - mean).T, lower=True, check_finite=False)
    return np.einsum("ij,ij->j", R, R)


def fit_class_gaussian(Ztr, ytr, eps=REG_EPS):
    """Class means and the shared pooled covariance, ridge added.

    Pooled covariance = sum_c (Xc - mc)^T (Xc - mc) / (N - K) + eps I, with N the
    number of labelled points and K the number of classes present. Labels < 0
    (unlabelled) are ignored. Returns (classes, means, cov, precision).
    """
    Z, y = _f64(Ztr), np.asarray(ytr)
    keep = y >= 0
    Z, y = Z[keep], y[keep]
    classes = np.unique(y)
    K, N, d = len(classes), len(Z), Z.shape[1]
    if N <= K:
        raise ValueError(f"need more points ({N}) than classes ({K})")
    means = np.stack([Z[y == c].mean(0) for c in classes])
    scatter = np.zeros((d, d))
    for c, m in zip(classes, means):
        Xc = Z[y == c] - m
        scatter += Xc.T @ Xc
    cov = scatter / (N - K) + eps * np.eye(d)
    precision = cho_solve(cho_factor(cov, lower=True), np.eye(d))
    return classes, means, cov, precision


def mahalanobis_min_sq(Zev, means, cov):
    """min over classes of the squared Mahalanobis distance, shared covariance."""
    L = _chol_lower(cov)
    X = _f64(Zev)
    d2 = np.stack([_sq_maha(X, m, L) for m in means], axis=1)
    return d2.min(axis=1)


def cosine_scores(Zev, means):
    """1 - max_c cos(z, mu_c); zero-norm rows get score 1 (cosine 0), never NaN."""
    X, M = _f64(Zev), _f64(means)
    xn = np.linalg.norm(X, axis=1, keepdims=True)
    mn = np.linalg.norm(M, axis=1, keepdims=True)
    U = np.divide(X, xn, out=np.zeros_like(X), where=xn > 0)
    V = np.divide(M, mn, out=np.zeros_like(M), where=mn > 0)
    return (1.0 - U @ V.T).min(axis=1)


def knn_scores(Ztr, Zev, k=PRIMARY_K):
    """Euclidean distance to the k-th nearest training neighbour (brute force, exact)."""
    if len(Ztr) < k:
        raise ValueError(f"need at least k={k} training points, got {len(Ztr)}")
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(_f64(Ztr))
    dist, _ = nn.kneighbors(_f64(Zev))
    return dist[:, k - 1]


def class_sigma_bar(Ztr, ytr, classes=None):
    """Per-class σ̄_c = mean_j std_j(Z_c) (ddof=1), the scale used by kde's Scott rule.

    Returns (classes, sigma_bar, n_c). Under a pure scale T(z)=c z every σ̄_c
    multiplies by c; under a rotation the total variance is preserved but σ̄_c
    generally moves because the arithmetic mean of per-axis stds is not
    invariant to redistributing variance across axes.
    """
    Z, y = _f64(Ztr), np.asarray(ytr)
    cls = np.unique(y[y >= 0]) if classes is None else np.asarray(classes)
    sig, ns = [], []
    for c in cls:
        C = Z[y == c]
        ns.append(len(C))
        sig.append(float(C.std(axis=0, ddof=1).mean()) if len(C) > 1 else 0.0)
    return cls, np.asarray(sig, dtype=np.float64), np.asarray(ns, dtype=np.int64)


def class_det_scale(Ztr, ytr, classes=None):
    """Per-class det(Σ_c)^(1/(2d)), the rotation-invariant scale used by kde_det.

    Σ_c is the class sample covariance (np.cov, ddof=1). Returns
    (classes, det_scale, n_c, logdet). Under T(z)=c Q z, det_scale multiplies
    by exactly c (sign of slogdet must stay +1).
    """
    Z, y = _f64(Ztr), np.asarray(ytr)
    cls = np.unique(y[y >= 0]) if classes is None else np.asarray(classes)
    d = Z.shape[1]
    scales, ns, logdets = [], [], []
    for c in cls:
        C = Z[y == c]
        ns.append(len(C))
        if len(C) <= 1:
            scales.append(0.0)
            logdets.append(float("-inf"))
            continue
        sign, logdet = np.linalg.slogdet(np.cov(C, rowvar=False))
        logdets.append(float(logdet) if sign > 0 else float("-inf"))
        scales.append(float(np.exp(logdet / (2.0 * d))) if sign > 0 else 0.0)
    return (cls, np.asarray(scales, dtype=np.float64), np.asarray(ns, dtype=np.int64),
            np.asarray(logdets, dtype=np.float64))


def _kde_from_bandwidths(Ztr, y, Q, cls, bandwidths, chunk=_CHUNK):
    """Shared KDE evaluation given per-class isotropic bandwidths h_c."""
    d = Ztr.shape[1]
    qq = (Q * Q).sum(1)
    best = np.full(len(Q), -np.inf)
    for c, h in zip(cls, bandwidths):
        C = Ztr[y == c]
        n = len(C)
        h = max(float(h), KDE_BW_FLOOR)
        cc = (C * C).sum(1)
        const = -np.log(n) - 0.5 * d * np.log(2 * np.pi) - d * np.log(h)
        for s in range(0, len(Q), chunk):
            sl = slice(s, s + chunk)
            D2 = qq[sl, None] + cc[None, :] - 2.0 * (Q[sl] @ C.T)
            np.maximum(D2, 0.0, out=D2)
            lp = logsumexp(-D2 / (2.0 * h * h), axis=1) + const
            best[sl] = np.maximum(best[sl], lp)
    return -best


def kde_scores(Ztr, ytr, Zev, classes=None, chunk=_CHUNK):
    """Per-class Gaussian KDE, Scott's rule with σ̄_c, score = -max_c log p_c(z).

    h_c = max(mean_j std_j(Z_c) * n_c^(-1/(d+4)), 1e-6). Squared distances come
    from the BLAS expansion |q|^2 + |c|^2 - 2 q.c (Phase B measured agreement
    with cdist to 2e-11), computed in chunks of evaluation rows so the
    n_eval x n_class matrix never has to fit in memory. Identical formula to
    Phase B's kde_scores; only the class set is read from the labels instead of
    being fixed at eight. Not similarity-invariant — see class_sigma_bar.
    """
    Ztr, y = _f64(Ztr), np.asarray(ytr)
    Q = _f64(Zev)
    d = Ztr.shape[1]
    cls = np.unique(y[y >= 0]) if classes is None else np.asarray(classes)
    _, sig, ns = class_sigma_bar(Ztr, y, classes=cls)
    hs = np.maximum(sig * (ns.astype(np.float64) ** (-1.0 / (d + 4))), KDE_BW_FLOOR)
    return _kde_from_bandwidths(Ztr, y, Q, cls, hs, chunk=chunk)


def kde_det_scores(Ztr, ytr, Zev, classes=None, chunk=_CHUNK):
    """Per-class Gaussian KDE with rotation-invariant Scott scale det(Σ_c)^(1/(2d)).

    h_c = max(det(Σ_c)^(1/(2d)) * n_c^(-1/(d+4)), 1e-6). Under T(z)=c Q z every
    h_c scales by c, squared distances scale by c^2, and the density ranking is
    unchanged — the similarity-invariant counterpart of kde_scores.
    """
    Ztr, y = _f64(Ztr), np.asarray(ytr)
    Q = _f64(Zev)
    d = Ztr.shape[1]
    cls = np.unique(y[y >= 0]) if classes is None else np.asarray(classes)
    _, scale, ns, _ = class_det_scale(Ztr, y, classes=cls)
    hs = np.maximum(scale * (ns.astype(np.float64) ** (-1.0 / (d + 4))), KDE_BW_FLOOR)
    return _kde_from_bandwidths(Ztr, y, Q, cls, hs, chunk=chunk)


def relative_mahalanobis_scores(Ztr, ytr, Zev, eps=REG_EPS):
    """Relative Mahalanobis: min_c MD_c(z) - MD_bg(z), higher = more OOD.

    The class term is the shared-covariance squared distance of maha. The
    background Gaussian is one mean and one full covariance (divisor N - 1, same
    ridge) over every labelled training point. Returns (scores, bg_mean, bg_precision).
    """
    Z, y = _f64(Ztr), np.asarray(ytr)
    keep = y >= 0
    Zl = Z[keep]
    _, means, cov, _ = fit_class_gaussian(Z, y, eps)
    bg_mean = Zl.mean(0)
    bg_cov = np.cov(Zl.T) + eps * np.eye(Zl.shape[1])
    L0 = _chol_lower(bg_cov)
    X = _f64(Zev)
    md_bg = _sq_maha(X, bg_mean, L0)
    md_cls = mahalanobis_min_sq(X, means, cov)
    bg_precision = cho_solve(cho_factor(bg_cov, lower=True), np.eye(Zl.shape[1]))
    return md_cls - md_bg, bg_mean, bg_precision


def score_all(Ztr, ytr, Zev, eps=REG_EPS):
    """All scorers fit on (Ztr, ytr) and applied to Zev.

    Returns a dict:
      scores        {maha, cosine, knn10, kde, kde_det, rmd} -> float64 arrays over Zev
      classes, means, precision (shared, ridge included), cov
      bg_mean, bg_precision   the Relative Mahalanobis background Gaussian
      eps, n_train, n_classes
    Phase B returned (scores, means, precision); the dict carries the same three
    plus what the bounds module needs.
    """
    classes, means, cov, precision = fit_class_gaussian(Ztr, ytr, eps)
    maha = mahalanobis_min_sq(Zev, means, cov)
    rmd, bg_mean, bg_precision = relative_mahalanobis_scores(Ztr, ytr, Zev, eps)
    scores = {
        "maha": maha,
        "cosine": cosine_scores(Zev, means),
        "knn10": knn_scores(Ztr, Zev),
        "kde": kde_scores(Ztr, ytr, Zev, classes=classes),
        "kde_det": kde_det_scores(Ztr, ytr, Zev, classes=classes),
        "rmd": rmd,
    }
    return {
        "scores": scores, "classes": classes, "means": means, "cov": cov,
        "precision": precision, "bg_mean": bg_mean, "bg_precision": bg_precision,
        "eps": eps, "n_train": int((np.asarray(ytr) >= 0).sum()), "n_classes": len(classes),
    }


def fast_auroc(scores, is_ood):
    """Rank-based AUROC, OOD positive, ties given the average rank.

    Phase B's version broke ties by position, which is harmless for continuous
    scores and wrong for the quantised ones some benchmarks produce.
    """
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(is_ood).astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fpr_at_tpr(scores, is_ood, tpr=0.95):
    """FPR of the ID points at the loosest threshold that still keeps TPR >= tpr.

    Threshold = the (1 - tpr) lower quantile of the OOD scores ('lower'
    interpolation, so TPR is never below the target); a point is flagged OOD
    when its score >= threshold.
    """
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(is_ood).astype(bool)
    if pos.all() or not pos.any():
        return float("nan")
    thr = np.quantile(s[pos], 1.0 - tpr, method="lower")
    return float(np.mean(s[~pos] >= thr))


def fpr_at_95_tpr(scores, is_ood):
    return fpr_at_tpr(scores, is_ood, 0.95)
