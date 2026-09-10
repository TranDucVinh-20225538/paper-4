"""The package must reproduce the published numbers, or it is not the same tool.

Each test names the table it checks. Run from the Paper 4 repository root; the
tests skip cleanly where the artifacts are absent so the package remains usable
on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flatcheck import alignment, nuisance_direction, reachable, susceptibility  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PER_EXAMPLE = ROOT / "PhaseB" / "outputs" / "per_example"
NUISANCE = ROOT / "PhaseB" / "NuisanceDirection" / "nuisance_direction.npz"
REF = ROOT / "PhaseB" / "assets" / "reference_embeddings"

needs_data = pytest.mark.skipif(not PER_EXAMPLE.is_dir(),
                                reason="Paper 4 artifacts not present")


def test_five_seeds_cannot_clear_sixteen_tests():
    """Section 4, second condition: the arithmetic that closes the family."""
    r = reachable(n_units=5, n_tests=16)
    assert not r.ok
    assert r.floor == pytest.approx(1 / 120)
    assert r.threshold == pytest.approx(0.05 / 16)


def test_ten_seeds_do_clear_it():
    """Section 6.3: what the seed extension bought."""
    assert reachable(n_units=10, n_tests=16).ok
    assert reachable(n_arrangements=20_000, n_tests=16).ok


def _paper_S(scorer, seeds=(42, 52, 62, 72, 82)):
    vals = []
    for seed in seeds:
        a = np.load(PER_EXAMPLE / f"Intervention_a0.0_s{seed}.npz")
        b = np.load(PER_EXAMPLE / f"Intervention_a1.0_s{seed}.npz")
        ood = a["is_ood"].astype(bool)
        if scorer == "knn10":
            with np.errstate(divide="ignore", invalid="ignore"):
                pert = np.maximum(b[scorer] / a[scorer], a[scorer] / b[scorer])
            kind = "ratio"
        else:
            pert, kind = b[scorer] - a[scorer], "additive"
        vals.append(susceptibility(a[scorer][~ood], a[scorer][ood], pert,
                                   kind=kind, seed=0).S)
    return float(np.mean(vals))


@needs_data
@pytest.mark.parametrize("scorer,published", [
    ("maha", 0.000065), ("knn10", 0.007253),
    ("kde", 0.018551), ("cosine", 0.047818),
])
def test_susceptibility_matches_table(scorer, published):
    """Table 5, within Monte Carlo error.

    Both the package and the study sample two million pairs, from different
    random streams, so agreement is to sampling error and not to the digit. The
    tolerance is relative because Mahalanobis's S is a rare-event rate -- 6.5e-5
    of two million pairs is about 130 of them -- where a fixed absolute
    tolerance would be either vacuous or unmeetable.
    """
    assert _paper_S(scorer) == pytest.approx(published, rel=0.20)


@needs_data
def test_scorer_ordering_is_the_claim():
    """The ordering is what the paper argues from; it must be exact."""
    s = {k: _paper_S(k) for k in ("maha", "knn10", "kde", "cosine")}
    assert s["maha"] < s["knn10"] < s["kde"] < s["cosine"]


@needs_data
def test_separation_is_orders_of_magnitude():
    """Section 6 reports a factor of roughly 735 between the extremes."""
    ratio = _paper_S("cosine") / _paper_S("maha")
    assert 400 < ratio < 1400


@needs_data
def test_mahalanobis_is_insensitive_not_invariant():
    """The distinction the whole framework exists to make."""
    a = np.load(PER_EXAMPLE / "Intervention_a0.0_s42.npz")
    b = np.load(PER_EXAMPLE / "Intervention_a1.0_s42.npz")
    ood = a["is_ood"].astype(bool)
    s = {k: susceptibility(a[k][~ood], a[k][ood], b[k] - a[k], seed=0)
         for k in ("maha", "cosine")}  # both additive: this compares like with like
    # Mahalanobis moves further in score space and less in rank space.
    assert s["maha"].delta > s["cosine"].delta
    assert s["maha"].S < s["cosine"].S / 100


@needs_data
def test_alignment_finds_the_construction_target():
    """Section 6: 1.000 for the pair the direction was built from."""
    w = np.load(NUISANCE, allow_pickle=True)["w"]
    import pandas as pd
    ev = np.load(REF / "ReferenceEmbedding" / "embeddings.npy")
    meta = pd.read_csv(REF / "ReferenceEmbedding" / "metadata.csv")
    isic = ev[(meta.domain == "isic").to_numpy()]
    pad = ev[(meta.domain == "pad_ufes").to_numpy()]
    (a,) = alignment(w, isic, {"PAD-UFES": pad})
    assert a.projection == pytest.approx(1.0, abs=1e-3)
    assert a.is_construction_target


def test_nuisance_direction_round_trip():
    """Building the direction then projecting the source domain returns one."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 64))
    b = rng.normal(size=(500, 64)) + 3.0
    w = nuisance_direction(a, b)
    (r,) = alignment(w, a, {"b": b})
    assert r.projection == pytest.approx(1.0, abs=1e-9)


def test_vacuous_bound_is_flagged():
    """S near one permits any change and the result says so."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=2000), rng.normal(size=2000) + 0.2
    assert susceptibility(a, b, 10.0).vacuous
