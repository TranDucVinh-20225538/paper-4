"""
RQ1 items 1.1 to 1.4: the numerical content behind the analytical results.

RQ1 asks to what quantitative extent each scorer can register the operator's
geometric change. The manuscript answers the "which" with theorems; this
supplies the "how much", which symbolic bounds alone do not give.

  1.1  covariance spectrum per seed: lambda_min, lambda_max, condition number
       raw and after a numerical spectral floor. Reported both ways because a
       lambda_min near 1e-17 makes the raw condition number a statement about
       floating point rather than about geometry.
  1.2  kappa_2(A_gamma) = kappa_2(Sigma)^(gamma/2) and the bounds it governs:
       Lemma 1's cosine distortion 2*eta/(1-eta) with
       eta = (k^2-1)/(k^2+1), and Proposition 3's k-NN ratio threshold.
  1.3  Theorem 2's ridge-induced Mahalanobis deviation evaluated on the actual
       residual distribution, at the median, P95, P99 and max, rather than at
       the maximum alone, which gives a bound too loose to be informative.
  1.4  local ranking susceptibility: over ID-OOD pairs, the margin
       M = s(OOD) - s(ID), and S(delta) = Pr(|M| <= 2 delta), the mass a
       perturbation of size delta can reorder. Corollary 1 makes this the
       bridge from a score bound to an AUROC bound. For k-NN the multiplicative
       analogue of Corollary 2 is used instead, since that score is
       scale-covariant.

Pairs are subsampled for the margin distribution: 5,067 x 2,298 is 11.6M pairs
per configuration, and the quantiles are stable well below that.

    python3 PhaseB/analysis/rq1_numerical_bounds.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from generate_per_example_scores import displacement, load  # noqa: E402

SEEDS = (42, 52, 62, 72, 82)
GAMMAS = (0.01, 0.25, 0.50, 0.75, 1.00)
FLOOR = 1e-10
RIDGE = 1e-5


def spectrum(Z, floor=FLOOR):
    mu = Z.mean(0)
    S = np.cov((Z - mu).T.astype(np.float64))
    lam = np.linalg.eigvalsh(S)
    lam_pos = np.clip(lam, floor, None)
    return {"lambda_min": lam.min(), "lambda_max": lam.max(),
            "kappa_raw": lam.max() / max(lam.min(), 1e-300),
            "kappa_floored": lam_pos.max() / lam_pos.min(),
            "n_below_floor": int((lam < floor).sum())}


def transformation_bounds(kappa_sigma, gamma):
    """kappa_2(A_gamma), Lemma 1's cosine bound, Proposition 3's k-NN threshold."""
    k = kappa_sigma ** (gamma / 2)
    eta = (k ** 2 - 1) / (k ** 2 + 1)
    cosine = min(2.0, 2 * eta / (1 - eta)) if eta < 1 else 2.0
    return {"kappa_A": k, "eta_A": eta, "cosine_bound": cosine, "knn_ratio_threshold": k}


def ridge_bound(residual_sq, kappa_A, lam_min_C, eps=RIDGE):
    """Theorem 2 with A = A_gamma, so ||I - AA^T||_2 and sigma_min(A) follow from kappa_A."""
    sig_min = 1.0 / kappa_A if kappa_A >= 1 else kappa_A
    inorm = abs(1 - sig_min ** 2)
    denom = (lam_min_C + eps) * (lam_min_C + eps * sig_min ** 2)
    return eps * residual_sq * inorm / denom if denom > 0 else np.inf


def susceptibility(s_id, s_ood, deltas, rng, n_pairs=2_000_000, additive=True):
    """S(delta) over sampled ID-OOD pairs, additive margin or multiplicative ratio."""
    i = rng.integers(0, len(s_id), n_pairs)
    j = rng.integers(0, len(s_ood), n_pairs)
    a, b = s_id[i], s_ood[j]
    if additive:
        m = np.abs(b - a)
        return {d: float((m <= 2 * d).mean()) for d in deltas}, m
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.maximum(b / a, a / b)
    r = r[np.isfinite(r)]
    return {d: float((r <= d).mean()) for d in deltas}, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--scores", type=Path, default=Path("PhaseB/outputs/per_example"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/rq1"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Ztr, Zev, ytr, is_ood = load(args.root)

    # ---- 1.1 spectra, on the intervened training population at each dose ----
    rows = []
    for seed in SEEDS:
        dz = displacement(args.root, "Intervention", seed)
        Dtr = dz(Ztr)
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            rows.append({"seed": seed, "alpha": a, **spectrum(Ztr + a * Dtr)})
    spec = pd.DataFrame(rows)
    spec.to_csv(args.out / "rq1_1_spectra.csv", index=False)
    print("1.1 covariance spectrum, Intervention arm")
    print(f"    lambda_min  {spec.lambda_min.min():.3e} to {spec.lambda_min.max():.3e}")
    print(f"    kappa raw     median {spec.kappa_raw.median():.3e}")
    print(f"    kappa floored median {spec.kappa_floored.median():.3e}")
    print(f"    eigenvalues below {FLOOR:g}: {spec.n_below_floor.min()} to {spec.n_below_floor.max()}")

    # ---- 1.2 transformation bounds across the gamma ladder ----
    k_sigma = spec[spec.alpha == 1.0].kappa_floored.median()
    tb = pd.DataFrame([{"gamma": g, **transformation_bounds(k_sigma, g)} for g in GAMMAS])
    tb.to_csv(args.out / "rq1_2_transformation_bounds.csv", index=False)
    print(f"\n1.2 bounds from kappa_2(Sigma) = {k_sigma:.3e} (floored, alpha=1)")
    print(tb.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---- 1.3 residual distribution and Theorem 2 at quantiles ----
    lam_min_C = spec[spec.alpha == 1.0].lambda_min.median()
    res = []
    for seed in SEEDS:
        f = args.scores / f"Intervention_a1.0_s{seed}.npz"
        y2 = np.load(f)["maha"]          # squared Mahalanobis residual q(y)
        for g in GAMMAS:
            kA = k_sigma ** (g / 2)
            for name, v in [("median", np.median(y2)), ("p95", np.percentile(y2, 95)),
                            ("p99", np.percentile(y2, 99)), ("max", y2.max())]:
                res.append({"seed": seed, "gamma": g, "quantile": name, "residual_sq": v,
                            "theorem2_bound": ridge_bound(v, kA, lam_min_C)})
    r3 = pd.DataFrame(res)
    r3.to_csv(args.out / "rq1_3_ridge_bounds.csv", index=False)
    print("\n1.3 Theorem 2 bound by residual quantile, median over seeds")
    print(r3.groupby(["gamma", "quantile"]).theorem2_bound.median().unstack()
          [["median", "p95", "p99", "max"]].to_string(float_format=lambda v: f"{v:.3g}"))

    # ---- 1.4 local ranking susceptibility ----
    deltas_add = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ratios = [1.0001, 1.001, 1.01, 1.1]
    sus = []
    for seed in SEEDS:
        d = np.load(args.scores / f"Intervention_a1.0_s{seed}.npz")
        ood = d["is_ood"].astype(bool)
        for scorer in ("maha", "cosine", "knn10", "kde"):
            s = d[scorer]
            if scorer == "knn10":
                S, _ = susceptibility(s[~ood], s[ood], ratios, rng, additive=False)
                sus += [{"seed": seed, "scorer": scorer, "kind": "ratio",
                         "threshold": k, "S": v} for k, v in S.items()]
            else:
                S, _ = susceptibility(s[~ood], s[ood], deltas_add, rng)
                sus += [{"seed": seed, "scorer": scorer, "kind": "additive",
                         "threshold": k, "S": v} for k, v in S.items()]
    s4 = pd.DataFrame(sus)
    s4.to_csv(args.out / "rq1_4_susceptibility.csv", index=False)
    print("\n1.4 local ranking susceptibility, mean over seeds")
    print(s4.groupby(["scorer", "kind", "threshold"]).S.mean().unstack(fill_value=np.nan)
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote four tables to {args.out}")


if __name__ == "__main__":
    main()
