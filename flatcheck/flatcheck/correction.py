"""Can the design reject at all?

A permutation test attains no p-value below 1/N, where N is the number of
arrangements it can form. A step-down correction over m tests has a first
threshold of alpha/m. Where the floor lies above that threshold no test in the
family can clear the first step, whatever the data show, and the family's null
describes the design rather than the phenomenon.

The check is one division and costs nothing when the analysis plan is written.
It is uncomfortable to discover afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["reachable", "Reachability", "max_family_size", "min_units"]


@dataclass(frozen=True)
class Reachability:
    floor: float
    threshold: float
    n_arrangements: int
    n_tests: int
    alpha: float
    exhaustive: bool

    @property
    def ok(self) -> bool:
        return self.floor < self.threshold

    def __str__(self) -> str:
        kind = "exhaustive" if self.exhaustive else "sampled"
        verdict = "reachable" if self.ok else "UNREACHABLE"
        return (f"{kind} permutation, {self.n_arrangements:,} arrangements: "
                f"floor {self.floor:.2e} vs Holm threshold {self.threshold:.2e} "
                f"-> {verdict}")


def reachable(n_units=None, n_tests=1, alpha=0.05, *, n_arrangements=None,
              paired=False):
    """Whether a permutation family can produce a corrected rejection.

    Parameters
    ----------
    n_units
        Number of exchangeable units permuted -- seeds, subjects, folds. For a
        whole-unit permutation this gives n_units! arrangements; with
        ``paired=True`` it gives 2**n_units, the sign-flip count.
    n_tests
        Size of the correction family.
    n_arrangements
        Give this instead of ``n_units`` when the permutation is sampled rather
        than enumerated. The floor is then 1/n_arrangements.

    Returns
    -------
    Reachability

    Examples
    --------
    >>> reachable(n_units=5, n_tests=16).ok
    False
    >>> reachable(n_arrangements=20_000, n_tests=16).ok
    True
    """
    if n_arrangements is None:
        if n_units is None:
            raise ValueError("give n_units or n_arrangements")
        n_arrangements = 2 ** n_units if paired else math.factorial(n_units)
        exhaustive = True
    else:
        exhaustive = False
    return Reachability(floor=1.0 / n_arrangements, threshold=alpha / n_tests,
                        n_arrangements=int(n_arrangements), n_tests=int(n_tests),
                        alpha=float(alpha), exhaustive=exhaustive)


def max_family_size(n_units, alpha=0.05, *, paired=False):
    """Largest correction family this many units can support."""
    n = 2 ** n_units if paired else math.factorial(n_units)
    return int(alpha * n)


def min_units(n_tests, alpha=0.05, *, paired=False):
    """Fewest units needed before a family of this size becomes reachable."""
    k = 1
    while True:
        n = 2 ** k if paired else math.factorial(k)
        if 1.0 / n < alpha / n_tests:
            return k
        k += 1
        if k > 200:
            raise RuntimeError("no feasible unit count below 200")
