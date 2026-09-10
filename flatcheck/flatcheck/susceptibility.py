"""Can the scorer respond at all?

A scorer whose ranking does not move under an intervention may be protected by
an invariance, or may be unable to move because almost no in-distribution /
out-of-distribution pair sits within reach of the perturbation. Both produce the
same flat AUROC curve. This module separates them.

S(delta) = Pr(|M| <= 2*delta), where M is the score gap over ID-OOD pairs, is an
upper bound on the AUROC change a perturbation of size delta can produce. The
bound is valid at delta_max and is usually vacuous there, so its use is
comparative: two scorers evaluated at their own observed perturbation magnitudes
have commensurable S, and an order-of-magnitude gap between them separates a
scorer that declined to move from one that could not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["susceptibility", "Susceptibility"]

_N_PAIRS = 2_000_000


@dataclass(frozen=True)
class Susceptibility:
    """Result of one susceptibility computation."""

    delta: float
    S: float
    observed_delta_auroc: float | None
    quantile: str
    n_pairs: int
    kind: str          # "additive" or "ratio"

    @property
    def bound_holds(self) -> bool | None:
        """Whether S bounds the observed change. Only meaningful at delta_max."""
        if self.observed_delta_auroc is None:
            return None
        return self.S >= self.observed_delta_auroc

    @property
    def vacuous(self) -> bool:
        """S near 1 bounds nothing: it permits any AUROC change."""
        return self.S > 0.5

    def __str__(self) -> str:
        b = "" if self.observed_delta_auroc is None else \
            f"  observed |dAUROC| {self.observed_delta_auroc:.6f}"
        v = "  (vacuous, S>0.5)" if self.vacuous else ""
        sym = "kappa" if self.kind == "ratio" else "delta"
        return f"{sym}[{self.quantile}]={self.delta:.6g}  S={self.S:.6f}{b}{v}"


def _pair_stat(scores_id, scores_ood, rng, kind, n_pairs=_N_PAIRS):
    """Sampled ID-OOD pair statistic: |difference|, or the symmetric ratio."""
    a = np.asarray(scores_id, float).ravel()
    b = np.asarray(scores_ood, float).ravel()
    a = a[rng.integers(0, a.size, n_pairs)]
    b = b[rng.integers(0, b.size, n_pairs)]
    if kind == "additive":
        return np.abs(b - a)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.maximum(b / a, a / b)
    return r[np.isfinite(r)]


def susceptibility(scores_id, scores_ood, perturbation, *, quantile="median",
                   kind="additive", observed_delta_auroc=None, seed=0,
                   n_pairs=_N_PAIRS):
    """Local ranking susceptibility of one scorer.

    Parameters
    ----------
    scores_id, scores_ood
        Scores on the two populations, before the intervention. Higher must mean
        more out-of-distribution; flip the sign if your scorer runs the other way.
    perturbation
        Either a scalar delta, or the per-example score displacement caused by
        the intervention (``scores_after - scores_before``), in which case delta
        is taken at ``quantile`` of its absolute value.
    quantile
        Which delta to use when ``perturbation`` is a vector: 'median', 'p95' or
        'max'. Only 'max' gives a valid bound; the others are descriptive and
        are what make two scorers comparable.
    kind
        'additive' for a scorer whose perturbation shifts scores, 'ratio' for one
        that scales them. A k-nearest-neighbour distance is scale-covariant, so
        its perturbation is multiplicative and the additive form understates it;
        pass ``kind="ratio"`` and give ``perturbation`` as the ratio
        ``max(after/before, before/after)``, or as the raw pair and let the
        quantile be taken of that.
    observed_delta_auroc
        Optional. The AUROC change actually observed, so the result can report
        whether the bound holds.

    Returns
    -------
    Susceptibility
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(perturbation, dtype=float)
    if p.ndim == 0:
        delta, q = float(p), "given"
    else:
        d = np.abs(p.ravel()) if kind == "additive" else p.ravel()
        d = d[np.isfinite(d)]
        q = quantile
        delta = {"median": np.median, "p95": lambda x: np.percentile(x, 95),
                 "max": np.max}[quantile](d)
        delta = float(delta)

    if kind not in ("additive", "ratio"):
        raise ValueError("kind must be 'additive' or 'ratio'")
    stat = _pair_stat(scores_id, scores_ood, rng, kind, n_pairs)
    S = float(np.mean(stat <= (2 * delta if kind == "additive" else delta)))
    return Susceptibility(delta=delta, S=S,
                          observed_delta_auroc=(None if observed_delta_auroc is None
                                                else float(abs(observed_delta_auroc))),
                          quantile=q, n_pairs=int(stat.size), kind=kind)
