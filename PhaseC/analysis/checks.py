"""The two pre-registered feasibility and construction checks of the audit.

check2_feasibility asks whether a seed-level permutation test can reject at all.
With k seeds the smallest attainable exact p-value is 1/k!; after Holm
correction over m tests the first (smallest-p) test must clear alpha/m. If
1/k! > alpha/m, no outcome of the experiment could be significant and the test
is not run as a test. For larger k the exact permutation distribution is
sampled instead, and the floor becomes 1/(n_sample + 1) (the add-one estimator).

check3_construction_target asks whether the suppressed direction is the one the
construction was built to hide: the projection of the unit ID-to-OOD centroid
difference on the (unit) suppressed direction. Near 1 means the intervention
removed exactly the axis that separates the two populations, so a large
response is the construction working rather than a discovery; near 0 means the
suppressed axis is unrelated to the separation.

No command line.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["K10_N_SAMPLE", "check2_feasibility", "check3_construction_target"]

K10_N_SAMPLE = 20_000


def check2_feasibility(k_seeds, m_tests, alpha=0.05, n_sample=None):
    """Exact floor 1/k!, Holm first threshold alpha/m, and whether the floor clears it.

    Returns floor, holm_first, reachable (floor <= holm_first), and
    min_k_for_reachable, the smallest k whose exact floor would clear. If
    n_sample is given (e.g. K10_N_SAMPLE for k = 10, where 10! = 3,628,800
    permutations are sampled rather than enumerated) the sampled floor
    1/(n_sample + 1) and its reachability are added.
    """
    if k_seeds < 1 or m_tests < 1:
        raise ValueError("k_seeds and m_tests must be >= 1")
    floor = 1.0 / math.factorial(int(k_seeds))
    holm_first = alpha / m_tests
    k_min = next(k for k in range(1, 400) if 1.0 / math.factorial(k) <= holm_first)
    out = {"k_seeds": int(k_seeds), "m_tests": int(m_tests), "alpha": alpha,
           "n_permutations": math.factorial(int(k_seeds)), "floor": floor,
           "holm_first": holm_first, "reachable": bool(floor <= holm_first),
           "min_k_for_reachable": k_min}
    if n_sample is not None:
        sf = 1.0 / (n_sample + 1)
        out.update({"n_sample": int(n_sample), "sampled_floor": sf,
                    "sampled_reachable": bool(sf <= holm_first)})
    return out


def check3_construction_target(mu_id, mu_ood, w_suppressed):
    """projection = ((mu_ood - mu_id) / ||mu_ood - mu_id||) . (w / ||w||).

    w is normalised here, so the value is a cosine in [-1, 1]; its norm is
    returned. Sign is kept (the direction w is defined up to sign, so read
    abs_projection for the target test). Degenerate inputs give NaN, not an error.
    """
    mu_id, mu_ood, w = (np.asarray(v, dtype=np.float64).ravel() for v in (mu_id, mu_ood, w_suppressed))
    gap = mu_ood - mu_id
    gn, wn = float(np.linalg.norm(gap)), float(np.linalg.norm(w))
    if gn == 0 or wn == 0:
        proj = float("nan")
    else:
        proj = float((gap / gn) @ (w / wn))
    return {"projection": proj, "abs_projection": abs(proj), "gap_norm": gn, "w_norm": wn,
            "angle_deg": float(np.degrees(np.arccos(np.clip(abs(proj), 0, 1)))) if np.isfinite(proj) else float("nan")}
