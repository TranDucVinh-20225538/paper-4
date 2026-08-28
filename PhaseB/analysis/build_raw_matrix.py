"""
Rebuilds the per-run matrix underlying the C2 correlation, and checks it against
the published result.

Why this exists: outputs/stage5_c2_correlation_v2.json stores the twenty
(geometry metric, scorer) Kendall correlations but not the thirty rows they were
computed from, and no stage5_c2 script is present in this repository (D-053
records the same gap for the confirmatory analysis). The inputs do survive as
frozen artifacts, so the matrix is recoverable. This script recovers it, writes
it as one tidy CSV, and then recomputes all twenty correlations from that CSV
and compares them to the stored values. A mismatch means the CSV is not the
population the paper analysed, and the script stops rather than emit it.

Population, per the stored sources_included: Intervention at alpha in
{0.25, 0.5, 0.75, 1.0} and Adaptation at linear-probe and partial-FT, five seeds
each, thirty rows. Conventional and Adaptation full-adapter-FT are written to
the CSV as well, flagged out of population, so the excluded arms are visible
rather than silently absent.

    python3 PhaseB/analysis/build_raw_matrix.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import kendalltau

GEOMETRY = ["lid_mean", "spectral_decay_slope", "condition_number", "per_direction_variance_w"]
SCORERS = ["maha_auroc", "cosine_auroc", "knn_k10_auroc", "kde_auroc", "domain_probe_accuracy_mean"]
PUBLISHED_NAMES = {
    "lid_mean": "LID", "spectral_decay_slope": "spectral_decay_slope",
    "condition_number": "condition_number", "per_direction_variance_w": "per_direction_variance_w",
    "maha_auroc": "Mahalanobis", "cosine_auroc": "Cosine", "knn_k10_auroc": "kNN10",
    "kde_auroc": "KDE", "domain_probe_accuracy_mean": "probe_accuracy(companion)",
}


def load(root: Path):
    j = lambda p: json.loads((root / p).read_text())
    return (j("AlphaLadder/alpha_ladder_results.json"),
            j("AlphaLadder/geometry_completion_results.json"),
            j("AlphaLadder/reliability_scorers_results.json"),
            j("AdaptationArm_Canonical/manifest.json"))


def build(root: Path) -> list[dict]:
    ladder, geom, scor, adapt = load(root)
    rows = []

    for arm in ("Intervention", "Conventional"):
        gate = {(r["seed"], r["alpha"]): r for r in ladder["results"][arm]}
        kappa = {(r["seed"], r["alpha"]): r for r in geom["results"][arm]}
        sc = {(r["seed"], r["alpha"]): r for r in scor["results"][arm]}
        for key, s in sc.items():
            g, k = gate.get(key), kappa.get(key)
            if g is None or k is None:
                raise SystemExit(f"{arm} {key}: geometry present in one source and absent in another")
            rows.append({
                "arm": arm, "config": f"alpha{key[1]}", "alpha": key[1], "seed": key[0],
                "in_c2_population": arm == "Intervention",
                "lid_mean": g["gate1_measurement"]["lid_mean"],
                "spectral_decay_slope": g["gate1_measurement"]["spectral_decay_slope"],
                "condition_number": k["condition_number"],
                "per_direction_variance_w": k["per_direction_variance_w"],
                "domain_probe_accuracy_mean": g["gate0"]["domain_probe_accuracy_mean"],
                "id_task_balanced_accuracy_mean": g["gate0"]["id_task_balanced_accuracy_mean"],
                "gate0_pass": g["gate0"]["gate0_pass"],
                **{s_: s[s_] for s_ in ("maha_auroc", "cosine_auroc", "knn_k10_auroc", "kde_auroc")},
            })

    for rung, entries in adapt["rungs"].items():
        for e in entries:
            rows.append({
                "arm": "Adaptation", "config": rung, "alpha": "", "seed": e.get("seed", ""),
                "in_c2_population": rung in ("linear-probe", "partial-FT"),
                **{g: e[g] for g in GEOMETRY},
                "domain_probe_accuracy_mean": e["gate0"]["domain_probe_accuracy_mean"],
                "id_task_balanced_accuracy_mean": e["gate0"]["id_task_balanced_accuracy_mean"],
                "gate0_pass": e["gate0"]["gate0_pass"],
                # full-adapter-FT carries no scorer results of its own: the manifest
                # records reused_from, and C2 excludes it for that reason. Emit the
                # row with blanks rather than dropping it, so the exclusion is visible.
                **{s: e.get(s, "") for s in ("maha_auroc", "cosine_auroc", "knn_k10_auroc", "kde_auroc")},
                "reused_from": e.get("reused_from", ""),
            })
    return rows


def check(rows, published):
    pop = [r for r in rows if r["in_c2_population"]]
    if len(pop) != published["n"]:
        raise SystemExit(f"population is {len(pop)} rows, stored result says n={published['n']}")
    stored = {(r["geometry_metric"], r["scorer"]): r for r in published["results"]}
    bad = []
    for g in GEOMETRY:
        for s in SCORERS:
            key = (PUBLISHED_NAMES[g], PUBLISHED_NAMES[s])
            if key not in stored:
                continue
            tau, _ = kendalltau([r[g] for r in pop], [r[s] for r in pop])
            if abs(tau - stored[key]["tau"]) > 1e-9:
                bad.append(f"{key}: rebuilt {tau:.6f} vs stored {stored[key]['tau']:.6f}")
    return len(pop), len(stored), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/raw_run_matrix.csv"))
    args = ap.parse_args()

    rows = build(args.root)
    published = json.loads((args.root / "outputs/stage5_c2_correlation_v2.json").read_text())
    n_pop, n_pairs, bad = check(rows, published)

    print(f"{len(rows)} rows built, {n_pop} in the C2 population")
    print(f"{n_pairs} stored correlations recomputed from the CSV")
    if bad:
        print("\nMISMATCH -- the rebuilt matrix is not the analysed population:")
        for b in bad:
            print(f"  {b}")
        raise SystemExit(1)
    print("every recomputed tau matches the stored value to 1e-9")

    cols = ["arm", "config", "alpha", "seed", "in_c2_population", *GEOMETRY,
            "maha_auroc", "cosine_auroc", "knn_k10_auroc", "kde_auroc",
            "domain_probe_accuracy_mean", "id_task_balanced_accuracy_mean", "gate0_pass",
            "reused_from"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
