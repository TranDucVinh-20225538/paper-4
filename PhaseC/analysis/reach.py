"""Can the scorer respond at all, and did it? Reach, realized reordering, and the reading of a flat AUROC.

Three quantities, kept apart because conflating them is the error the audit exists to prevent.

S, the ranking reach: the fraction of ID-OOD score pairs close enough that a
score perturbation of size delta could reverse them. Additive form: pairs with
|s_ood - s_id| <= 2 delta. Ratio form, for scale-covariant scorers such as kNN
distance: pairs with max(s_ood/s_id, s_id/s_ood) <= delta, as in flatcheck's
susceptibility.py. S bounds the AUROC change a perturbation of size delta can
cause; it is usually vacuous at the largest delta, so it is read comparatively.

F, the realized reordering: of the same kind of sampled pairs, the fraction
whose directed order (is the OOD score above the ID score?) differs before and
after the intervention. |dAUROC| <= F always, since AUROC is the probability of
the OOD-above-ID order; F is what actually happened, S is what could have.

check1_reading puts (dAUROC, S, F) together:
    moved        AUROC changed by at least tau
    invariant    flat, reach was available (S high), yet almost nothing reordered
                 (F small): the scorer declined to move
    insensitive  flat because almost no pair was within reach (S low)
    cancelling   flat on net but pairs did reorder (F large): flips offset
    inconsistent |dAUROC| > F beyond sampling error: a bug or a stale input

Pair sampling is with replacement, 2,000,000 pairs by default, ID index drawn
before OOD index from one generator, as in flatcheck. Reimplemented here so
PhaseC does not import flatcheck. No command line.
"""
from __future__ import annotations

import numpy as np

__all__ = ["N_PAIRS", "sample_pairs", "ranking_reach", "ranking_reach_detail",
           "realized_reordering_F", "realized_reordering_detail", "check1_reading"]

N_PAIRS = 2_000_000


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def sample_pairs(n_id, n_ood, n_pairs, rng):
    """Index pairs (i, j), i into the ID scores, j into the OOD scores, with replacement."""
    rng = _rng(rng)
    return rng.integers(0, n_id, n_pairs), rng.integers(0, n_ood, n_pairs)


def ranking_reach_detail(s_id, s_ood, delta, kind="additive", n_pairs=N_PAIRS, rng=0,
                         ratio_power=1.0):
    """S plus the bookkeeping: pairs used, pairs where the ratio was undefined.

    additive: S = mean(|s_ood - s_id| <= 2 delta).
    ratio:    S = mean(max(s_ood/s_id, s_id/s_ood) <= delta ** ratio_power).
    flatcheck compares the ratio to delta itself (ratio_power = 1, the default,
    so numbers match); the threshold that mirrors the additive 2 delta for a
    multiplicative perturbation of factor delta on each score is delta^2
    (ratio_power = 2). Ratio pairs with a non-positive score have no ratio; they
    are counted as out of reach and in the denominator (flatcheck drops them and
    renormalises), and n_invalid reports how many.
    """
    if kind not in ("additive", "ratio"):
        raise ValueError("kind must be 'additive' or 'ratio'")
    a = np.asarray(s_id, dtype=np.float64).ravel()
    b = np.asarray(s_ood, dtype=np.float64).ravel()
    i, j = sample_pairs(a.size, b.size, n_pairs, rng)
    x, y = a[i], b[j]
    n_invalid = 0
    if kind == "additive":
        hit = np.abs(y - x) <= 2.0 * delta
    else:
        valid = (x > 0) & (y > 0)
        n_invalid = int((~valid).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.maximum(y / x, x / y)
        hit = valid & (r <= delta ** ratio_power)
    return {"S": float(hit.mean()), "delta": float(delta), "kind": kind,
            "n_pairs": int(n_pairs), "n_invalid": n_invalid, "ratio_power": ratio_power}


def ranking_reach(s_id, s_ood, delta, kind="additive", n_pairs=N_PAIRS, rng=0):
    """S(delta): fraction of sampled ID-OOD pairs within reach of a perturbation of size delta."""
    return ranking_reach_detail(s_id, s_ood, delta, kind, n_pairs, rng)["S"]


def realized_reordering_detail(s_id_before, s_ood_before, s_id_after, s_ood_after,
                               n_pairs=N_PAIRS, rng=0):
    """F and its breakdown. The same sampled (i, j) pairs are read before and after.

    The directed order of a pair is sign(s_ood - s_id) in {-1, 0, +1}. F counts
    every change of that sign, including entering or leaving an exact tie
    (F_tie); F_flip counts only pairs that crossed from one strict order to the
    other, which is the part of F that can move AUROC by a full unit per pair.
    """
    ib, ob = (np.asarray(v, dtype=np.float64).ravel() for v in (s_id_before, s_ood_before))
    ia, oa = (np.asarray(v, dtype=np.float64).ravel() for v in (s_id_after, s_ood_after))
    if ib.size != ia.size or ob.size != oa.size:
        raise ValueError("before and after scores must cover the same examples")
    i, j = sample_pairs(ib.size, ob.size, n_pairs, rng)
    sb = np.sign(ob[j] - ib[i])
    sa = np.sign(oa[j] - ia[i])
    changed = sb != sa
    flip = (sb * sa) < 0
    return {"F": float(changed.mean()), "F_flip": float(flip.mean()),
            "F_tie": float((changed & ~flip).mean()), "n_pairs": int(n_pairs),
            "se": float(np.sqrt(changed.mean() * (1 - changed.mean()) / n_pairs))}


def realized_reordering_F(s_id_before, s_ood_before, s_id_after, s_ood_after,
                          n_pairs=N_PAIRS, rng=0):
    """Fraction of sampled ID-OOD pairs whose directed order changed."""
    return realized_reordering_detail(s_id_before, s_ood_before, s_id_after, s_ood_after,
                                      n_pairs, rng)["F"]


def check1_reading(delta_auroc, S, F, tau=0.002, S_high=0.02, F0=1e-4, n_pairs=N_PAIRS):
    """Classify a (dAUROC, S, F) triple. See the module docstring for the labels.

    Order of tests: non-finite input; consistency (|dAUROC| <= F + 4 se); moved
    (|dAUROC| >= tau); then, for flat responses, F > F0 -> cancelling, else
    S >= S_high -> invariant, else insensitive. Thresholds tau, S_high and F0 are
    the registered defaults; they are returned so the reading is auditable.
    """
    out = {"tau": tau, "S_high": S_high, "F0": F0, "delta_auroc": delta_auroc, "S": S, "F": F}
    if not all(np.isfinite([delta_auroc, S, F])):
        return {**out, "label": "undefined", "flat": None, "reason": "non-finite input"}
    ad = abs(delta_auroc)
    se = np.sqrt(max(F * (1 - F), 0.0) / n_pairs)
    if ad > F + 4 * se + 1e-12:
        return {**out, "label": "inconsistent", "flat": ad < tau,
                "reason": f"|dAUROC|={ad:.3g} exceeds F={F:.3g}; AUROC cannot move more than the "
                          "fraction of pairs that reordered"}
    if ad >= tau:
        return {**out, "label": "moved", "flat": False,
                "reason": f"|dAUROC|={ad:.3g} >= tau={tau}"}
    if F > F0:
        return {**out, "label": "cancelling", "flat": True,
                "reason": f"flat on net but F={F:.3g} > F0={F0}: reorderings offset"}
    if S >= S_high:
        return {**out, "label": "invariant", "flat": True,
                "reason": f"flat, reach S={S:.3g} >= {S_high} but F={F:.3g} <= {F0}: "
                          "the scorer was in reach and did not reorder"}
    return {**out, "label": "insensitive", "flat": True,
            "reason": f"flat and S={S:.3g} < {S_high}: almost no pair was within reach"}
