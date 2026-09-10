"""Three checks on whether a flat evaluation result carries information.

A representation intervention that leaves a distance-based scorer's AUROC flat
admits at least three readings, and an aggregate summary cannot tell them apart:
the scorer may be invariant, it may be unable to respond to anything, or the
analysis may be unable to reject whatever it observes. A fourth applies when the
curve is *not* flat -- the apparent response may be an artifact of evaluating an
operator on the domain it was constructed from.

    from flatcheck import susceptibility, reachable, alignment

Each check is independent of the others and none needs more data than a study
already has.
"""
from .susceptibility import susceptibility, Susceptibility
from .correction import reachable, Reachability, max_family_size, min_units
from .alignment import alignment, Alignment, nuisance_direction

__version__ = "0.1.0"
__all__ = ["susceptibility", "Susceptibility",
           "reachable", "Reachability", "max_family_size", "min_units",
           "alignment", "Alignment", "nuisance_direction"]
