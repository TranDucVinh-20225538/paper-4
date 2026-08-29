"""
Plan Section 3.4: per-seed geometry dose-response, with the figure it asks for.

The manuscript reports mean changes only. The plan asks for the per-seed values,
the paired effect against alpha = 0, intervals, a monotonic-trend test for
spectral slope, and LID described as reproducible but non-monotonic rather than
as controlled.

The figure is the one specified: dose on x, standardized metric on y, one line
per seed, with the mean and interval over seeds drawn on top. Standardization uses the
conventional arm across its whole ladder. Centring on alpha = 0 alone does not
work: the adapter is not applied there, so the five seeds coincide and the
between-seed SD is around 1e-15.

Paired contrasts are within seed, and the interval is a paired bootstrap over
the five seeds. With five seeds the interval is wide by construction and is
reported as such rather than leaned on.

    python3 PhaseB/analysis/geometry_dose_response.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

METRICS = {"lid_mean": "LID", "spectral_decay_slope": "spectral slope"}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_ladder(root: Path) -> pd.DataFrame:
    d = json.loads((root / "AlphaLadder/alpha_ladder_results.json").read_text())
    rows = []
    for arm in ("Intervention", "Conventional"):
        for r in d["results"][arm]:
            rows.append({"arm": arm, "seed": r["seed"], "alpha": r["alpha"],
                         **{k: r["gate1_measurement"][k] for k in METRICS}})
    return pd.DataFrame(rows)


def paired_bootstrap(diffs, n, rng):
    return np.array([np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/rq2"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lad = load_ladder(args.root)
    lad.to_csv(args.out / "geometry_per_seed.csv", index=False)
    # Standardize on the conventional arm across its whole ladder, not at alpha = 0.
    # At alpha = 0 the adapter is not applied, so all five seeds coincide and the
    # between-seed SD is ~1e-15; dividing by it puts the axis on a 1e14 scale.
    ref = lad[lad.arm == "Conventional"]

    rows = []
    for m in METRICS:
        mu, sd = ref[m].mean(), ref[m].std(ddof=1)
        sd = sd if sd > 0 else 1.0
        for arm in ("Intervention", "Conventional"):
            base = lad[(lad.arm == arm) & (lad.alpha == 0.0)].set_index("seed")[m]
            for a in ALPHAS:
                cur = lad[(lad.arm == arm) & (lad.alpha == a)].set_index("seed")[m]
                d = (cur - base).dropna()
                boot = paired_bootstrap(d.to_numpy(), args.n_boot, rng)
                rows.append({"metric": m, "arm": arm, "alpha": a, "n_seeds": len(d),
                             "mean_delta": d.mean(), "sd_delta": d.std(ddof=1),
                             "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5),
                             "z_mean": (cur.mean() - mu) / sd})
    eff = pd.DataFrame(rows)
    eff.to_csv(args.out / "geometry_dose_effects.csv", index=False)

    print("Paired change from alpha = 0, within seed, Intervention arm")
    print(f"{'metric':<22}{'alpha':>7}{'mean delta':>13}{'95% CI':>26}")
    for _, r in eff[(eff.arm == "Intervention") & (eff.alpha > 0)].iterrows():
        print(f"{r.metric:<22}{r.alpha:7.2f}{r.mean_delta:+13.5f}   [{r.lo:+.5f}, {r.hi:+.5f}]")

    print("\nMonotonic trend across the dose ladder, per seed (Kendall tau vs alpha)")
    for m in METRICS:
        taus = []
        for s, g in lad[lad.arm == "Intervention"].groupby("seed"):
            g = g.sort_values("alpha")
            taus.append(kendalltau(g.alpha, g[m]).statistic)
        taus = np.array(taus)
        same = "all five agree in sign" if np.all(np.sign(taus) == np.sign(taus[0])) else \
               "signs differ across seeds"
        print(f"   {METRICS[m]:<16} tau per seed {np.array2string(taus, precision=2)}"
              f"   {same}")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10.5, 4.2), sharex=True)
    for ax, (m, label) in zip(axes, METRICS.items()):
        mu, sd = ref[m].mean(), ref[m].std(ddof=1)
        sd = sd if sd > 0 else 1.0
        sub = lad[lad.arm == "Intervention"]
        for s, g in sub.groupby("seed"):
            g = g.sort_values("alpha")
            ax.plot(g.alpha, (g[m] - mu) / sd, color="0.72", lw=1, marker="o", ms=3, zorder=2)
        agg = sub.assign(z=(sub[m] - mu) / sd).groupby("alpha").z.agg(["mean", "std"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], color="tab:blue", lw=2,
                    marker="o", capsize=4, zorder=3, label="mean ± SD over seeds")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel(r"dose $\alpha$")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("standardized on the conventional arm")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Geometry dose-response, per seed and aggregate (Intervention arm)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"geometry_dose_response.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote tables and figure to {args.out}")


if __name__ == "__main__":
    main()
