"""The transformations the audit applies to an embedding space, grouped by what should be invariant.

A flat AUROC curve under an intervention can mean the scorer is protected by an
invariance, or that nothing reordered because nothing was in reach (reach.py).
To tell those apart Phase C needs interventions whose effect on each scorer is
known in advance. These operators are those interventions. Each returns
(Z_out, meta): Z_out in float32, and a meta dict that keeps everything the
bounds in bounds.py consume (A, b, mu, singular values, ||I - A A^T||_2,
sigma_min(A), the spectrum a whitening used, the floors applied), so no bound
has to refit or rediscover a matrix after the fact.

Convention for every affine operator:  T(z) = A (z - mu) + mu + b.
mu is the centre the operator acts about (the fit-population mean unless
stated), b a residual translation (zero unless asked for). meta["A"] is the
float64 matrix; Z_out is computed in float64 and stored as float32.

Groups (GROUPS below). random_orthogonal_scaled is in every one, as the common
baseline whose flat response any scorer in the group should reproduce.
    similarity   random_orthogonal_scaled
    spectral     random_orthogonal_scaled, spectral_temper, zca_whiten
    affine       random_orthogonal_scaled, spectral_temper, zca_whiten, general_affine
    outside      radial_l2_normalise (negative control), residual_nonlinear,
                 apply_adapter. Not affine, so no invariance is claimed; a
                 scorer that stays flat under these is flat for another reason.

temper is copied from Phase B's hierarchical_auroc_ci.py; spectral_temper is
the same map computed in float64 with its spectrum retained. apply_adapter
follows Phase B's displacement (z + alpha * (W2 GELU(W1 z + b1) + b2)), tanh-form
GELU included, so Phase C reproduces the numbers Phase B scored.

No command line; imported by the run scripts.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "GROUPS", "EIG_FLOOR", "ZCA_ALPHAS", "gelu", "temper", "affine_meta",
    "random_orthogonal_scaled", "spectral_temper", "zca_whiten", "general_affine",
    "radial_l2_normalise", "residual_nonlinear", "apply_adapter",
]

EIG_FLOOR = 1e-12
ZCA_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

GROUPS = {
    "similarity": ("random_orthogonal_scaled",),
    "spectral": ("random_orthogonal_scaled", "spectral_temper", "zca_whiten"),
    "affine": ("random_orthogonal_scaled", "spectral_temper", "zca_whiten", "general_affine"),
    "outside": ("radial_l2_normalise", "residual_nonlinear", "apply_adapter"),
}


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def gelu(x):
    """Tanh-form GELU exactly as Phase B's generate_per_example_scores.py writes it."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _gelu_exact(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def temper(fit_pop, Z, gamma):
    """z' = mu + U diag(lambda^(-gamma/2)) U^T (z - mu), spectrum from the fit population.

    Verbatim from Phase B's hierarchical_auroc_ci.py (float32 A). Kept so Phase B
    numbers can be reproduced bit-for-bit; new code should use spectral_temper.
    """
    mu = fit_pop.mean(0)
    S = np.cov((fit_pop - mu).T.astype(np.float64))
    lam, U = np.linalg.eigh(S)
    A = (U @ np.diag(np.clip(lam, 1e-12, None) ** (-gamma / 2)) @ U.T).astype(np.float32)
    return mu + (Z - mu) @ A.T


def affine_meta(name, A, b, mu, **extra):
    """The retained quantities for an affine map T(z) = A (z - mu) + mu + b."""
    A = _f64(A)
    d = A.shape[0]
    sv = np.linalg.svd(A, compute_uv=False)
    I_minus_AAT = float(np.linalg.norm(np.eye(d) - A @ A.T, 2))
    meta = {
        "operator": name, "A": A, "b": None if b is None else _f64(b), "mu": _f64(mu),
        "singular_values": sv, "sigma_max": float(sv[0]), "sigma_min": float(sv[-1]),
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "I_minus_AAT_2": I_minus_AAT,
        "I_minus_A_2": float(np.linalg.norm(np.eye(d) - A, 2)),
        "linear": True,
    }
    meta.update(extra)
    return meta


def _apply_affine(Z, A, b, mu):
    Zc = _f64(Z) - mu
    out = Zc @ A.T + mu
    if b is not None:
        out = out + b
    return out.astype(np.float32)


def _haar_orthogonal(d, rng):
    """Haar-distributed orthogonal matrix: QR of a Gaussian with the sign of R's diagonal fixed."""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def random_orthogonal_scaled(Z, rng, scale_range=(0.5, 2.0), mu=None, b_scale=0.0):
    """T(z) = s Q (z - mu) + mu + b, Q Haar-orthogonal, s log-uniform in scale_range.

    mu defaults to the mean of Z (centred form; cosine is *not* invariant — see Lemma 2).
    Pass zeros (or use `random_orthogonal_scaled_linear`) to rotate about the origin —
    the only centre under which an origin-based cosine is exactly invariant. b is zero
    unless b_scale > 0, in which case it is a random direction of length
    b_scale * median ||z - mu||. meta carries A = sQ, so ||I - A A^T||_2 =
    |1 - s^2| and every singular value equals s.
    """
    rng = _rng(rng)
    Z64 = _f64(Z)
    d = Z64.shape[1]
    centered = mu is None
    mu = Z64.mean(0) if centered else _f64(mu)
    lo, hi = scale_range
    s = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    A = s * _haar_orthogonal(d, rng)
    b = None
    if b_scale > 0:
        u = rng.standard_normal(d)
        b = u / np.linalg.norm(u) * b_scale * np.median(np.linalg.norm(Z64 - mu, axis=1))
    out = _apply_affine(Z64, A, b, mu)
    return out, affine_meta("random_orthogonal_scaled", A, b, mu, scale=s, scale_range=tuple(scale_range),
                            centered_at_mean=centered)


def random_orthogonal_scaled_linear(Z, rng, scale_range=(0.5, 2.0)):
    """Pure linear similarity: T(z) = s Q z (mu = 0, no translation).

    This is the code-correctness gate: maha, knn10, cosine (and kde_det) must
    be exactly invariant. The mean-std `kde` scorer is an explained exception
    (σ̄_c is not rotation-invariant; see confirm_kde_bandwidth.py). Distinct
    from `random_orthogonal_scaled` which centres at the train mean.
    """
    rng = _rng(rng)
    Z64 = _f64(Z)
    d = Z64.shape[1]
    lo, hi = scale_range
    s = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    A = s * _haar_orthogonal(d, rng)
    mu = np.zeros(d, dtype=np.float64)
    out = (Z64 @ A.T).astype(np.float32)          # T(z) = A z, no centering
    return out, affine_meta("random_orthogonal_scaled_linear", A, None, mu, scale=s,
                            scale_range=tuple(scale_range), centered_at_mean=False)


def _spectrum(fit_pop, floor, rel_floor):
    P = _f64(fit_pop)
    mu = P.mean(0)
    S = np.cov((P - mu).T)
    lam, U = np.linalg.eigh(S)
    eff_floor = max(floor, rel_floor * float(lam[-1]))
    return mu, lam, U, eff_floor


def spectral_temper(fit_pop, Z, gamma, floor=EIG_FLOOR, rel_floor=0.0):
    """z' = mu + U diag(max(lam, floor)^(-gamma/2)) U^T (z - mu); the map of temper(), spectrum kept.

    gamma = 0 is the identity, gamma = 1 full ZCA whitening. The floor is the
    absolute 1e-12 Phase B used; rel_floor (default 0, off) raises it to
    rel_floor * lam_max for rank-deficient spectra. Both the floor requested and
    the one effectively applied are recorded, with how many eigenvalues it
    touched. meta: lam (ascending), U, mu, A, gamma, floor, n_floored.
    """
    mu, lam, U, eff = _spectrum(fit_pop, floor, rel_floor)
    lam_c = np.clip(lam, eff, None)
    A = (U * lam_c ** (-gamma / 2.0)) @ U.T
    out = _apply_affine(Z, A, None, mu)
    return out, affine_meta(
        "spectral_temper", A, None, mu, gamma=float(gamma), lam=lam, U=U, floor=floor,
        rel_floor=rel_floor, floor_applied=eff, n_floored=int((lam < eff).sum()),
        lam_max=float(lam[-1]))


def zca_whiten(fit_pop, Z, alpha, floor=EIG_FLOOR, rel_floor=0.0):
    """T_alpha = (1 - alpha) I + alpha Sigma^{-1/2}, centred at the fit mean.

    alpha in {0, .25, .5, .75, 1} (ZCA_ALPHAS). alpha = 0 is the identity and
    alpha = 1 equals spectral_temper(gamma = 1). Unlike tempering, intermediate
    alpha interpolates the matrix, not the exponent. meta as spectral_temper,
    plus alpha.
    """
    mu, lam, U, eff = _spectrum(fit_pop, floor, rel_floor)
    lam_c = np.clip(lam, eff, None)
    W = (U * lam_c ** -0.5) @ U.T
    d = W.shape[0]
    A = (1.0 - alpha) * np.eye(d) + alpha * W
    out = _apply_affine(Z, A, None, mu)
    return out, affine_meta(
        "zca_whiten", A, None, mu, alpha=float(alpha), lam=lam, U=U, floor=floor,
        rel_floor=rel_floor, floor_applied=eff, n_floored=int((lam < eff).sum()),
        lam_max=float(lam[-1]))


def general_affine(Z, rng, cond_target=10.0, mu=None, b_scale=1.0):
    """Random invertible A with condition number exactly cond_target, and a random shift.

    A = U diag(s) V^T with U, V Haar-orthogonal (QR of Gaussians) and s
    log-uniform in [1/sqrt(kappa), sqrt(kappa)], the two endpoints pinned so that
    cond(A) = kappa and the geometric mean of the extreme singular values is 1.
    b is a random direction of length b_scale * median ||z - mu||.
    """
    rng = _rng(rng)
    Z64 = _f64(Z)
    d = Z64.shape[1]
    if cond_target < 1:
        raise ValueError("cond_target must be >= 1")
    mu = Z64.mean(0) if mu is None else _f64(mu)
    lo, hi = -0.5 * np.log(cond_target), 0.5 * np.log(cond_target)
    logs = rng.uniform(lo, hi, size=d)
    if d >= 2:
        logs[0], logs[-1] = lo, hi
    s = np.exp(logs)
    A = (_haar_orthogonal(d, rng) * s) @ _haar_orthogonal(d, rng).T
    u = rng.standard_normal(d)
    b = u / np.linalg.norm(u) * b_scale * np.median(np.linalg.norm(Z64 - mu, axis=1))
    out = _apply_affine(Z64, A, b, mu)
    return out, affine_meta("general_affine", A, b, mu, cond_target=float(cond_target), b_scale=b_scale)


def radial_l2_normalise(Z, target_norm=None, fit_pop=None):
    """z' = target_norm * z / ||z||. Outside every group: a negative control.

    Radial rescaling is a different scale for every point, so it is not an affine
    map and no scorer here is protected against it. target_norm defaults to the
    median ||z|| of fit_pop, which must then be supplied; it is never taken from
    Z, so evaluation data cannot choose its own scale. Zero-norm rows are left at
    zero and counted.
    """
    if target_norm is None:
        if fit_pop is None:
            raise ValueError("pass target_norm, or fit_pop to take its median norm")
        target_norm = float(np.median(np.linalg.norm(_f64(fit_pop), axis=1)))
    Z64 = _f64(Z)
    n = np.linalg.norm(Z64, axis=1, keepdims=True)
    out = np.divide(Z64 * target_norm, n, out=np.zeros_like(Z64), where=n > 0)
    meta = {"operator": "radial_l2_normalise", "A": None, "b": None, "mu": None,
            "target_norm": float(target_norm), "n_zero_norm": int((n == 0).sum()),
            "norm_ratio_min": float(target_norm / n[n > 0].max()) if (n > 0).any() else None,
            "norm_ratio_max": float(target_norm / n[n > 0].min()) if (n > 0).any() else None,
            "linear": False}
    return out.astype(np.float32), meta


def residual_nonlinear(Z, mu, W, t):
    """z -> z + t tanh(W (z - mu)), a bounded nonlinear residual.

    W is d x d, or low rank: a tuple (U, V) with U d x r and V r x d giving
    W = U V, or a single d x r array U read as the symmetric low-rank W = U U^T.
    The residual is bounded by |t| per coordinate and the Jacobian is
    I + t diag(sech^2) W, so ||J - I||_2 <= |t| ||W||_2 and sigma_min(J) >=
    1 - |t| ||W||_2; both are retained as the quantities an affine bound would
    have taken from A.
    """
    Z64, mu = _f64(Z), _f64(mu)
    d = Z64.shape[1]
    if isinstance(W, tuple):
        U, V = _f64(W[0]), _f64(W[1])
    else:
        W = _f64(W)
        if W.shape == (d, d):
            U, V = W, None
        elif W.ndim == 2 and W.shape[0] == d:
            U, V = W, W.T
        else:
            raise ValueError(f"W must be d x d or d x r with d={d}, got {W.shape}")
    Zc = Z64 - mu
    pre = Zc @ U.T if V is None else (Zc @ V.T) @ U.T
    R = t * np.tanh(pre)
    w_norm = float(np.linalg.norm(U, 2) if V is None else np.linalg.norm(U @ V, 2))
    meta = {"operator": "residual_nonlinear", "A": None, "b": None, "mu": mu, "t": float(t),
            "W_spectral_norm": w_norm, "jacobian_dev_bound": abs(t) * w_norm,
            "jacobian_sigma_min_lower": 1.0 - abs(t) * w_norm,
            "residual_norm_median": float(np.median(np.linalg.norm(R, axis=1))),
            "residual_norm_max": float(np.linalg.norm(R, axis=1).max()), "linear": False}
    return (Z64 + R).astype(np.float32), meta


def _arr(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return _f64(x)


def apply_adapter(Z, adapter_state, alpha, gelu_approximate=True):
    """z + alpha * up(gelu(down(z))): the residual bottleneck adapter of Phase B.

    adapter_state holds down.weight (r x d), down.bias, up.weight (d x r),
    up.bias, as numpy arrays or torch tensors (no torch import here); a
    checkpoint dict with the weights under "adapter" is accepted. alpha = 1 is
    the trained adapter, 0 the identity. gelu_approximate=True is the tanh form
    Phase B scored with; False uses the exact erf form the module trains with
    (the two differ by about 1e-3 in the activation).
    """
    sd = adapter_state.get("adapter", adapter_state)
    W1, b1 = _arr(sd["down.weight"]), _arr(sd["down.bias"])
    W2, b2 = _arr(sd["up.weight"]), _arr(sd["up.bias"])
    act = gelu if gelu_approximate else _gelu_exact
    Z64 = _f64(Z)
    delta = act(Z64 @ W1.T + b1) @ W2.T + b2
    out = Z64 + alpha * delta
    dn = np.linalg.norm(delta, axis=1)
    meta = {"operator": "apply_adapter", "A": None, "b": None, "mu": None, "alpha": float(alpha),
            "gelu_approximate": bool(gelu_approximate), "bottleneck": int(W1.shape[0]),
            "delta_norm_median": float(np.median(dn)), "delta_norm_max": float(dn.max()),
            "displacement_ratio_median": float(np.median(
                alpha * dn / np.maximum(np.linalg.norm(Z64, axis=1), 1e-300))),
            "linear": False}
    return out.astype(np.float32), meta
