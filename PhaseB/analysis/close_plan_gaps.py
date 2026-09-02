"""
The two items in the advisor's plan that were still open.

Common support (Stage 2, first item). The matched analysis reports how many
pairs it found, which is the overlap after the fact. What the plan asks for is
the check before it: do the intervention identities occupy a shared region of G
at all, or is the caliper doing the work of hiding that they do not? Reported as
the fraction of each identity's configurations having a cross-identity neighbour
within the caliper, per identity rather than pooled, because a pooled figure
hides an identity with no overlap at all.

FPR@95%TPR (classified recommended, cheap once raw scores exist). AUROC
integrates over every threshold; a deployed reliability estimator runs at one.
Reported at the operating point where 95% of shifted inputs are flagged.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from rq3_geometry_matching import G_COLS, standardize  # noqa: E402

OUT = ROOT / "outputs"
SCORERS = ("maha", "cosine", "knn10", "kde")
CALIPERS = (0.15, 0.25, 0.35)


def common_support(df, caliper):
    """Fraction of each identity's rows with a cross-identity neighbour inside the caliper."""
    Z = df[[f"z_{c}" for c in G_COLS]].to_numpy()
    ids = df.arm.to_numpy()
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    d[ids[:, None] == ids[None, :]] = np.inf
    has = (d <= caliper).any(1)
    return {a: float(has[ids == a].mean()) for a in np.unique(ids)}


def fpr_at_tpr(scores, is_ood, tpr=0.95):
    """FPR on ID at the threshold flagging `tpr` of OOD. Higher score = more OOD."""
    s, y = np.asarray(scores, float), np.asarray(is_ood, bool)
    thr = np.quantile(s[y], 1 - tpr)          # threshold catching tpr of OOD
    return float((s[~y] >= thr).mean())


def main():
    print("=== common support, per intervention identity ===")
    rows = []
    for name, path in [("primary", OUT / "matched_input.csv")] + \
                      [(p.stem.replace("matched_input_", ""), p)
                       for p in sorted((OUT / "phase3").glob("matched_input_P*.csv"))]:
        d = standardize(pd.read_csv(path).reset_index(drop=True))
        print(f"\n{name}  (n={len(d)})")
        for cal in CALIPERS:
            cs = common_support(d, cal)
            worst = min(cs.values())
            print(f"  caliper {cal}: " + "  ".join(f"{a} {v:.2f}" for a, v in cs.items())
                  + f"   worst {worst:.2f}")
            for a, v in cs.items():
                rows.append(dict(dataset=name, caliper=cal, identity=a, frac_supported=v))
    pd.DataFrame(rows).to_csv(OUT / "common_support.csv", index=False)

    print("\n\n=== FPR@95%TPR, Intervention arm, mean over 5 seeds ===")
    fr = []
    for label, d in [("PAD-UFES (calibration)", OUT / "per_example"),
                     ("external", OUT / "phase3" / "per_example")]:
        for a in ("0.0", "1.0"):
            pats = sorted(d.glob(f"*Intervention_a{a}_s*.npz"))
            groups = {}
            for f in pats:
                key = f.name.split("_Intervention")[0] if "_Intervention" in f.name else label
                groups.setdefault(key, []).append(f)
            for key, files in groups.items():
                vals = {sc: [] for sc in SCORERS}
                for f in files:
                    z = np.load(f)
                    for sc in SCORERS:
                        vals[sc].append(fpr_at_tpr(z[sc], z["is_ood"].astype(bool)))
                rec = dict(pair=key if key != label else "P0_pad_ufes", alpha=float(a))
                for sc in SCORERS:
                    rec[sc] = float(np.mean(vals[sc]))
                fr.append(rec)
    f = pd.DataFrame(fr).sort_values(["pair", "alpha"])
    f.to_csv(OUT / "fpr_at_95tpr.csv", index=False)
    print(f.round(4).to_string(index=False))
    print(f"\nwrote common_support.csv and fpr_at_95tpr.csv")


if __name__ == "__main__":
    main()
