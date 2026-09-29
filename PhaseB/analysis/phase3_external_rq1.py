"""
Phase 3 -- RQ1 susceptibility on each external pair, and the headroom verdict.

Same estimator as the primary analysis: margin_mass and ratio_mass are imported
from rq1_susceptibility rather than reimplemented, so the numbers are comparable
across pairs by construction and not by my care in copying.

Reports, per pair and scorer, the operating AUROC, the observed distortion at
median and max, S at each, and S/MDE -- the ratio below which the scorer cannot
express an effect as large as the smallest one the design can detect. The
max-delta row is the only one that is a valid bound; the median row is
descriptive and is labelled as such.

Then checks the three predictions registered in LOCKED_PROTOCOL.md section 8.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from rq1_susceptibility import margin_mass, ratio_mass, SEEDS_5, SCORERS  # noqa: E402

# The external domains were scored once, at the five registered seeds. D-2
# extended the seed set for the primary domain's permutation families and did
# not touch Phase 3, so importing the extended list here asks for scores that
# were never computed and were never claimed.
SEEDS = SEEDS_5

OUT = ROOT / "outputs" / "phase3"
PE = OUT / "per_example"


def hanley_mcneil_se(a, n1, n0):
    q1, q0 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q0 - a * a)) / (n1 * n0)
    return float(np.sqrt(max(v, 0.0)))


def main():
    rng = np.random.default_rng(0)
    pairs = sorted({p.name.split("_Intervention")[0]
                    for p in PE.glob("*_Intervention_a0.0_s*.npz")})
    if not pairs:
        raise SystemExit("no external per-example scores; run phase3_score_external.py")

    rows = []
    for pair in pairs:
        for seed in SEEDS:
            d0 = np.load(PE / f"{pair}_Intervention_a0.0_s{seed}.npz")
            d1 = np.load(PE / f"{pair}_Intervention_a1.0_s{seed}.npz")
            ood = d0["is_ood"].astype(bool)
            n1, n0 = int(ood.sum()), int((~ood).sum())
            for sc in SCORERS:
                s0, s1 = d0[sc], d1[sc]
                dist = np.abs(s1 - s0)
                a0, a1 = roc_auc_score(ood, s0), roc_auc_score(ood, s1)
                if sc == "knn10":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r = np.maximum(s1 / s0, s0 / s1)
                    r = r[np.isfinite(r)]
                    ks = {"median": float(np.median(r)), "max": float(r.max())}
                    S = ratio_mass(s0[~ood], s0[ood], list(ks.values()), rng)
                    items = [(q, k, S[k]) for q, k in ks.items()]
                else:
                    ds = {"median": float(np.median(dist)), "max": float(dist.max())}
                    S = margin_mass(s0[~ood], s0[ood], list(ds.values()), rng)
                    items = [(q, d, S[d]) for q, d in ds.items()]
                for q, d, sv in items:
                    rows.append(dict(pair=pair, seed=seed, scorer=sc, quantile=q,
                                     delta=d, S=sv, auroc=a0, auroc1=a1,
                                     obs_dAUROC=abs(a1 - a0), n_ood=n1, n_id=n0,
                                     se_hm=hanley_mcneil_se(a0, n1, n0)))

    df = pd.DataFrame(rows)

    # MDE from the variance the design actually shows, not from a sampling
    # formula that cannot see it. The comparison is paired within seed, so the
    # relevant spread is that of the alpha 0 -> 1 difference across seeds;
    # Hanley-McNeil is kept alongside as the reference it is, and where the two
    # disagree the observed one is the one that governs.
    per = df[df["quantile"] == "median"].drop_duplicates(["pair", "scorer", "seed"])
    sd = (per.assign(d=per.auroc1 - per.auroc)
             .groupby(["pair", "scorer"])
             .agg(sd_paired=("d", "std"), sd_level=("auroc", "std"),
                  se_hm=("se_hm", "mean")).reset_index())
    sd["mde_observed"] = 2.80 * sd.sd_paired / np.sqrt(len(SEEDS))
    sd["mde_hm"] = 2.80 * sd.se_hm / np.sqrt(len(SEEDS))
    sd["mde"] = sd[["mde_observed", "mde_hm"]].max(axis=1)
    df = df.merge(sd[["pair", "scorer", "mde", "mde_observed", "mde_hm",
                      "sd_paired", "sd_level"]], on=["pair", "scorer"])
    df["ratio"] = df.S / df.mde
    df.to_csv(OUT / "external_rq1_susceptibility.csv", index=False)
    print("=== MDE: observed paired spread vs Hanley-McNeil ===")
    print(sd[["pair", "scorer", "sd_level", "sd_paired", "mde_observed",
              "mde_hm"]].round(5).to_string(index=False))
    print()

    g = (df.groupby(["pair", "scorer", "quantile"])
           .agg(auroc=("auroc", "mean"), delta=("delta", "mean"), S=("S", "mean"),
                obs=("obs_dAUROC", "mean"), mde=("mde", "mean"),
                ratio=("ratio", "mean"), n_ood=("n_ood", "first")).reset_index())
    print("=== operating point and resolution, per pair ===\n")
    for pair in sorted(g.pair.unique()):
        sub = g[(g.pair == pair) & (g["quantile"] == "median")]
        print(f"{pair}   (n_ood = {int(sub.n_ood.iloc[0])})")
        for _, r in sub.iterrows():
            print(f"   {r.scorer:<7} AUROC={r.auroc:.5f}  S={r.S:.6f}  "
                  f"|dAUROC|={r.obs:.6f}  MDE={r.mde:.5f}  S/MDE={r.ratio:8.2f}")
        print()

    print("=== pre-registered predictions ===")
    a = g[(g["quantile"] == "median")].pivot_table(index="pair", columns="scorer",
                                                   values="auroc")
    r = g[(g["quantile"] == "median")].pivot_table(index="pair", columns="scorer",
                                                   values="ratio")
    print("\nAUROC by pair:\n" + a.round(5).to_string())
    print("\nS/MDE by pair:\n" + r.round(2).to_string())
    if "maha" in r.columns:
        print(f"\n(1) Mahalanobis S/MDE < 1 everywhere: "
              f"{bool((r['maha'] < 1).all())}  [max {r['maha'].max():.3f}]")
    if {"P0_pad_ufes", "P1_derm7pt_derm"} <= set(a.index):
        d = (a.loc["P1_derm7pt_derm"] - a.loc["P0_pad_ufes"])
        print(f"(2) P1 below P0 (headroom gained): {dict(d.round(5))}")
    if {"P1_derm7pt_derm", "P2_derm7pt_clinic"} <= set(a.index):
        d = (a.loc["P2_derm7pt_clinic"] - a.loc["P1_derm7pt_derm"])
        print(f"(3) P2 above P1 (within-lesion modality): {dict(d.round(5))}")
    print(f"\nwrote {OUT}/external_rq1_susceptibility.csv")


if __name__ == "__main__":
    main()
