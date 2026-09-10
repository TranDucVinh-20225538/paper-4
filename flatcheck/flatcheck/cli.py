"""Command line entry point: run the checks on a CSV of per-example scores.

    flatcheck scores.csv --score maha --after maha_after

The CSV needs one row per evaluation example, a column marking which rows are
out-of-distribution, and the score before the intervention. Give the score after
it as well and the perturbation is taken from the pair.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .correction import reachable
from .susceptibility import susceptibility


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flatcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="one row per evaluation example")
    ap.add_argument("--score", required=True, help="score column, before the intervention")
    ap.add_argument("--after", help="score column after it; perturbation is the difference")
    ap.add_argument("--is-ood", default="is_ood", help="0/1 column marking the shifted population")
    ap.add_argument("--kind", default="additive", choices=["additive", "ratio"],
                    help="ratio for a scale-covariant score such as a k-NN distance")
    ap.add_argument("--quantile", default="median", choices=["median", "p95", "max"])
    ap.add_argument("--seeds", type=int, help="units your permutation exchanges")
    ap.add_argument("--tests", type=int, help="size of the correction family")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        print("reading a CSV needs pandas; the library itself does not", file=sys.stderr)
        return 2

    df = pd.read_csv(a.csv)
    for col in (a.score, a.is_ood) + ((a.after,) if a.after else ()):
        if col not in df.columns:
            print(f"column {col!r} not in {a.csv}", file=sys.stderr)
            return 2

    ood = df[a.is_ood].to_numpy().astype(bool)
    s0 = df[a.score].to_numpy(float)
    if a.after is None:
        print("no --after column: susceptibility needs a perturbation, skipping it\n")
    else:
        s1 = df[a.after].to_numpy(float)
        if a.kind == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                pert = np.maximum(s1 / s0, s0 / s1)
        else:
            pert = s1 - s0
        for q in ("median", "max"):
            r = susceptibility(s0[~ood], s0[ood], pert, kind=a.kind, quantile=q)
            print(r)
        print("\nonly the max row is a valid bound; the median row is descriptive")
        print("and is the one that compares two scorers against each other.\n")

    if a.seeds and a.tests:
        r = reachable(n_units=a.seeds, n_tests=a.tests, alpha=a.alpha)
        print(r)
        if not r.ok:
            print("a null from this family describes the design, not the phenomenon.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
