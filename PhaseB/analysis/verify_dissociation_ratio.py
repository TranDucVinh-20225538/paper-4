"""
Recomputes the comparison figure Results cites from Paper 3, from that study's
own per-checkpoint matrix rather than from a reading of its figure.

The debt this settles: Paper 3 reports a directed AUROC near 0.402 for
Mahalanobis, and its final version is explicit that this reflects an inverted
ordering rather than an absent signal. The probe-minus-Mahalanobis gap
therefore takes two values depending on the convention, and the ratio against
the present study's 1.6e-3 changes with it. Both are reported in Results; this
script is where the numbers come from.

    python3 PhaseB/analysis/verify_dissociation_ratio.py --matrix <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import data_roots

PROBES = ["probe_lr_auroc", "probe_svm_auroc", "probe_rf_auroc"]
THIS_STUDY_GAP = 1.6e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path,
                    default=data_roots.root("paper3") / "results" / "checkpoint_results_matrix.csv")
    ap.add_argument("--scorer", default="mahalanobis_auroc")
    args = ap.parse_args()

    d = pd.read_csv(args.matrix)
    directed = d[args.scorer]
    oriented = directed.map(lambda a: max(a, 1 - a))

    print(f"n = {len(d)} checkpoints   directed {directed.mean():.4f}   "
          f"orientation-free {oriented.mean():.4f}\n")
    print(f"{'probe':<20} {'directed gap':>13} {'ratio':>7} "
          f"{'free gap':>10} {'ratio':>7}")
    for p in PROBES:
        gd, gf = (d[p] - directed).mean(), (d[p] - oriented).mean()
        print(f"{p:<20} {gd:13.4f} {gd/THIS_STUDY_GAP:7.0f} "
              f"{gf:10.4f} {gf/THIS_STUDY_GAP:7.0f}")

    print("\nResults cites the linear probes: 0.333 directed (ratio ~200), "
          "0.137 orientation-free (ratio ~86).")


if __name__ == "__main__":
    main()
