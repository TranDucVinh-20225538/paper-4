"""Was the operator built from the domain it is evaluated on?

Nuisance removal, domain-adversarial training and shortcut suppression commonly
define their target direction from a contrast between the in-distribution
population and a reference shifted population -- typically the difference of
their means. Where that reference is also an evaluation domain, the intervention
is trained to suppress the direction along which that domain separates, and any
effect measured against it is a measurement of the operator against its own
construction target rather than evidence about a mechanism.

The check is one inner product and no sample size enters it. A projection near
one identifies the construction target; projections near zero identify domains
the operator does not point at.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["alignment", "Alignment", "nuisance_direction"]


@dataclass(frozen=True)
class Alignment:
    name: str
    projection: float

    @property
    def is_construction_target(self) -> bool:
        return self.projection > 0.95

    @property
    def verdict(self) -> str:
        if self.projection > 0.95:
            return "construction target -- effects here are not mechanism evidence"
        if self.projection > 0.30:
            return "substantially aligned"
        if self.projection > 0.10:
            return "weakly aligned"
        return "near-orthogonal -- the operator does not point at this domain"

    def __str__(self) -> str:
        return f"{self.name:<24}|cos| = {self.projection:.4f}   {self.verdict}"


def nuisance_direction(embeddings_id, embeddings_reference):
    """The usual construction: normalised difference of population means."""
    w = (np.asarray(embeddings_reference, float).mean(0)
         - np.asarray(embeddings_id, float).mean(0))
    n = np.linalg.norm(w)
    if n == 0:
        raise ValueError("the two populations have identical means")
    return w / n


def alignment(direction, embeddings_id, domains):
    """Project each evaluation domain's separation onto the suppressed direction.

    Parameters
    ----------
    direction
        The direction the intervention suppresses. Need not be normalised.
    embeddings_id
        In-distribution embeddings, or their mean.
    domains
        Mapping of name -> embeddings (or mean) for each evaluation domain.

    Returns
    -------
    list[Alignment], ordered by projection descending.

    Notes
    -----
    A domain used to construct ``direction`` returns 1.0 by definition. That is
    the point: seeing it confirms the check is measuring what it should.
    """
    w = np.asarray(direction, float).ravel()
    w = w / np.linalg.norm(w)
    mu_id = np.asarray(embeddings_id, float)
    mu_id = mu_id.mean(0) if mu_id.ndim > 1 else mu_id

    out = []
    for name, emb in domains.items():
        mu = np.asarray(emb, float)
        mu = mu.mean(0) if mu.ndim > 1 else mu
        v = mu - mu_id
        nv = np.linalg.norm(v)
        if nv == 0:
            raise ValueError(f"domain {name!r} has the in-distribution mean")
        out.append(Alignment(name=name, projection=float(abs((v / nv) @ w))))
    return sorted(out, key=lambda a: -a.projection)
