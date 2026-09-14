"""
Every matched contrast the manuscript reports, from committed inputs.

Three of these tables were previously matched by hand and the results typed
straight in, so 51, 30, 21 and the five-versus-ten seed comparison existed only
as prose. This script recovers all of them.

Two axes are emitted rather than one answer, because the correction here is a
judgement and should be checkable. The `bootstrap` axis is the substantive one:
matching is blind to seed, so a pair can join configurations from two seeds,
and `one_way` -- what the manuscript was built on -- charges such a pair to the
seed of its first member alone. `multiway` admits a pair only when both of its
seeds are drawn, which is what "uncertainty clustered on seed" asks for. The
`matching` axis is a sensitivity: `within_seed` forbids cross-seed pairs
outright, which fixes the clustering by construction but strips the pool so
thin that covariate balance fails on the external domains.

The manuscript reports pooled matching with the multiway bootstrap.

    python3 PhaseB/analysis/rq3_augmented_matching.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from rq3_geometry_matching import (  # noqa: E402
    G_COLS, SCORERS, CALIPERS, standardize, greedy_match, smd,
    cluster_bootstrap, multiway_cluster_bootstrap,
)

OUT = ROOT / "outputs" / "rq3"
KEEP = ["arm", "seed", "registered"] + G_COLS + SCORERS
REGISTERED_SEEDS = [42, 52, 62, 72, 82]


def load_sets() -> dict[str, pd.DataFrame]:
    reg = pd.read_csv(ROOT / "outputs" / "matched_input.csv").assign(registered=True)
    fine = pd.read_csv(ROOT / "outputs" / "fine_gamma" / "fine_gamma_scored.csv").assign(registered=False)
    ten = pd.read_csv(ROOT / "outputs" / "seeds10" / "matched_input_10seed.csv").assign(registered=True)
    return {
        "registered": reg[KEEP],
        "augmented": pd.concat([reg[KEEP], fine[KEEP]], ignore_index=True),
        "seeds5": ten[ten.seed.isin(REGISTERED_SEEDS)][KEEP].reset_index(drop=True),
        "seeds10": ten[KEEP],
    }


def match(df: pd.DataFrame, caliper: float, mode: str) -> list[tuple[int, int]]:
    if mode == "pooled":
        return greedy_match(df, caliper)
    pairs = []
    for s in sorted(df.seed.unique()):
        sub = df[df.seed == s]
        idx = sub.index.to_numpy()
        pairs += [(idx[a], idx[b]) for a, b in greedy_match(sub.reset_index(drop=True), caliper)]
    return pairs


def contrast(A, B, label, caliper, matching, boot_kind, n_boot, rng) -> list[dict]:
    worst = max(abs(smd(A[f"z_{c}"].to_numpy(), B[f"z_{c}"].to_numpy())) for c in G_COLS)
    sa, sb = A.seed.to_numpy(), B.seed.to_numpy()
    rows = []
    for sc in SCORERS:
        d = A[sc].to_numpy() - B[sc].to_numpy()
        boot = (multiway_cluster_bootstrap(d, sa, sb, n_boot, rng) if boot_kind == "multiway"
                else cluster_bootstrap(d, sa, n_boot, rng))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"configuration_set": label, "matching": matching, "bootstrap": boot_kind,
                     "caliper": caliper, "n_pairs": len(A),
                     "n_cross_seed_pairs": int((sa != sb).sum()), "worst_smd": worst,
                     "scorer": sc, "delta": d.mean(), "lo": lo, "hi": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sets, rows = load_sets(), []
    print("  ".join(f"{k} {len(v)}" for k, v in sets.items()), "\n")

    for matching in ("pooled", "within_seed"):
        for boot_kind in ("multiway", "one_way"):
            rng = np.random.default_rng(args.seed)
            for name, raw in sets.items():
                df = standardize(raw).reset_index(drop=True)
                for cal in CALIPERS:
                    pairs = match(df, cal, matching)
                    if not pairs:
                        continue
                    i, j = map(np.array, zip(*pairs))
                    A, B = df.iloc[i], df.iloc[j]
                    rows += contrast(A, B, name, cal, matching, boot_kind, args.n_boot, rng)
                    if name == "augmented":
                        # entirely inside the extended regime iff neither member registered
                        t = ~((~A.registered.to_numpy()) & (~B.registered.to_numpy()))
                        rows += contrast(A[t], B[t], "augmented_touching_registered",
                                         cal, matching, boot_kind, args.n_boot, rng)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rq3_table8.csv", index=False)

    sub = res[(res.matching == "pooled") & (res.bootstrap == "multiway") & (res.caliper == 0.25)]
    print("=== reported: pooled matching, multiway bootstrap, caliper .25 ===")
    for name in ("registered", "augmented", "augmented_touching_registered", "seeds5", "seeds10"):
        g = sub[sub.configuration_set == name]
        if g.empty:
            continue
        r = g.iloc[0]
        print(f"{name:<32}{int(r.n_pairs):>3} pairs ({int(r.n_cross_seed_pairs)} cross-seed)"
              f"  |SMD| {r.worst_smd:.3f}")
        for _, x in g.iterrows():
            print(f"    {x.scorer:<14}{x.delta:+9.5f}  [{x.lo:+.5f}, {x.hi:+.5f}]"
                  f"  {'*' if x.excludes_zero else ''}")
    print(f"\nwrote {OUT / 'rq3_table8.csv'}")


if __name__ == "__main__":
    main()
