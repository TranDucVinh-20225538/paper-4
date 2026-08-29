"""
Checks the empirical numbers in Sections 4.1 and 4.2 against the stored artifacts.

Every figure the manuscript reports for RQ2 and RQ3 should be recomputable from
the frozen outputs. This script recomputes them and prints the comparison, so a
disagreement is visible rather than discovered by a reviewer. It asserts nothing
and fails on nothing: the point is to show both numbers side by side.

    python3 PhaseB/analysis/verify_manuscript_numbers.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import pandas as pd

CFG = ("A_strong_natural", "B_weak_natural", "C_strong_pulled")
SCORERS = (("maha_auroc", "Mahalanobis"), ("cosine_auroc", "Cosine"),
           ("knn_k10_auroc", "k-NN"), ("kde_auroc", "KDE"))


def gates(matrix: Path):
    d = pd.read_csv(matrix)
    iv = d[d.arm == "Intervention"]
    p, t = iv.domain_probe_accuracy_mean, iv.id_task_balanced_accuracy_mean
    print(f"4.1 gate margins over {len(iv)} Intervention checkpoints")
    print(f"    domain-probe accuracy   {p.min():.4f} to {p.max():.4f}   threshold 0.738")
    print(f"    ID balanced accuracy    {t.min():.4f} to {t.max():.4f}   threshold 0.225")


def manipulation(root: Path):
    t = json.loads((root / "outputs/stage5_c3_spectral_tempering_check.json").read_text())
    print("\n4.1 tempering, mean over five seeds")
    for g in sorted({r["gamma"] for r in t}):
        rs = [r for r in t if r["gamma"] == g]
        ds = st.mean((r["slope_post"] - r["slope_pre"]) * 1e3 for r in rs)
        dl = [r["LID_post"] - r["LID_pre"] for r in rs]
        print(f"    gamma {g:.2f}   d_slope {ds:+.2f}e-3   d_LID mean {st.mean(dl):+.2f}"
              f"   range {min(dl):+.2f} to {max(dl):+.2f}")

    m = json.loads((root / "outputs/stage5_c3_manipulation_check.json").read_text())
    dk = [r["kappa_post"] - r["kappa_pre"] for r in m]
    dl = [r["LID_post"] - r["LID_pre"] for r in m]
    ds = [r["slope_post"] - r["slope_pre"] for r in m]
    print(f"\n4.1 normalization (negative control)")
    print(f"    d_kappa mean {st.mean(dk):+.3e}")
    print(f"    d_LID   mean {st.mean(dl):+.3e}  SD {st.stdev(dl):.3e}")
    print(f"    d_slope mean {st.mean(ds):+.3e}  SD {st.stdev(ds):.3e}")


def config_c(root: Path):
    c3 = json.loads((root / "outputs/stage5_c3_results.json").read_text())
    ladder = json.loads((root / "AlphaLadder/alpha_ladder_results.json").read_text())
    B = {r["seed"]: r["gate1_measurement"] for r in ladder["results"]["Intervention"]
         if r["alpha"] == 0.25}
    rows = [(r["seed"], r["C_strong_pulled"], B[r["seed"]]) for r in c3["raw_rows"]]

    print("\n4.1 configuration C against its slope target B, per seed")
    for s, c, b in rows:
        print(f"    seed {s}   slope C {c['spectral_decay_slope']:.6e}"
              f"   slope B {b['spectral_decay_slope']:.6e}"
              f"   d_LID {c['lid_mean'] - b['lid_mean']:+.4f}")
    print(f"    mean      slope C {st.mean(c['spectral_decay_slope'] for _, c, _ in rows):.6e}"
          f"   slope B {st.mean(b['spectral_decay_slope'] for _, _, b in rows):.6e}"
          f"   d_LID {st.mean(c['lid_mean'] - b['lid_mean'] for _, c, b in rows):+.4f}")


def abc(root: Path):
    c3 = json.loads((root / "outputs/stage5_c3_results.json").read_text())
    ref = c3["reference_baseline"]
    print("\n4.2 Table 5, reference AUROC and headroom")
    for k, name in SCORERS:
        print(f"    {name:<12} {ref[k]:.4f}   headroom {1 - ref[k]:.4f}")

    print("\n4.2 Table 4 means, and the paired contrasts")
    for k, name in SCORERS:
        a, b, c = (st.mean(r[x][k] for r in c3["raw_rows"]) for x in CFG)
        print(f"    {name:<12} A {a:.4f}  B {b:.4f}  C {c:.4f}"
              f"   C-A {c - a:+.4f}   C-B {c - b:+.4f}")
    print("    note: C-A from full precision, not from the rounded table entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--matrix", type=Path, default=Path("PhaseB/outputs/raw_run_matrix.csv"))
    args = ap.parse_args()
    gates(args.matrix)
    manipulation(args.root)
    config_c(args.root)
    abc(args.root)


if __name__ == "__main__":
    main()
