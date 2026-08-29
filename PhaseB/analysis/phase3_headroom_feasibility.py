"""
Phase 3 feasibility: how much headroom does an external domain need, and how big?

The advisor's objection to the present evidence is that PAD-UFES-20 produces
near-ceiling AUROC, so a scorer that does not respond and a scorer that cannot
respond look identical.  Adding a locked same-modality domain is only worth the
cost if headroom actually buys resolution.  That is answerable now, from the
scores we already have, without touching a new dataset.

Two quantities, computed separately because they pull in opposite directions:

  resolution   S(delta) is Corollary 1's ceiling on |dAUROC| for a score
               perturbation of size delta.  Near ceiling the ID/OOD margins sit
               far from zero, so S is tiny and the design cannot see a response
               however real it is.  Headroom moves margin mass toward zero and
               S rises.

  precision    A lower AUROC is also a noisier one.  Hanley-McNeil SE grows as
               the operating point leaves the ceiling, and the paired seed-level
               design adds a between-seed component on top.

The useful headroom is where the first outruns the second.  We sweep separability
by adding Gaussian noise to the observed scores -- a harder domain is one whose
ID and OOD score distributions overlap more -- and recompute S at the delta the
intervention actually produces.  delta is a property of the operator, not of the
domain, so it is held fixed across the sweep.  That is the load-bearing
assumption and it is stated rather than buried.
"""
import numpy as np, pandas as pd, glob, os, re, json

SCORERS = ["maha", "cosine", "knn10", "kde"]
OUT = "PhaseB/outputs/phase3"
RNG = np.random.default_rng(20260829)


def auroc(scores, is_ood):
    """Directed AUROC: higher score should mean OOD."""
    s, y = np.asarray(scores, float), np.asarray(is_ood, bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def margins(scores, is_ood, cap=4_000_000):
    """Pairwise ID-OOD score gaps |s_ood - s_id|, subsampled for tractability."""
    s, y = np.asarray(scores, float), np.asarray(is_ood, bool)
    a, b = s[y], s[~y]
    if a.size * b.size > cap:
        k = int(np.sqrt(cap))
        a = RNG.choice(a, min(k, a.size), replace=False)
        b = RNG.choice(b, min(k, b.size), replace=False)
    return np.abs(a[:, None] - b[None, :]).ravel()


def susceptibility(m, delta):
    """S(delta) = Pr(|M| <= 2 delta), Corollary 1's bound on |dAUROC|."""
    return float(np.mean(m <= 2 * delta))


def degrade_noise(s, y, target, iters=60):
    """Harder domain as added noise: widens the margin scale as it overlaps."""
    eps = RNG.standard_normal(s.size)
    lo, hi = 0.0, np.std(s) * 400
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if auroc(s + mid * eps, y) > target else (lo, mid)
    return s + ((lo + hi) / 2) * eps


def degrade_shrink(s, y, target, iters=60):
    """Harder domain as a smaller group gap: within-group spread untouched.

    The two models bracket the question.  Noise inflates the margin scale while
    it overlaps the groups; shrinkage leaves the scale alone.  A conclusion that
    survives both does not depend on which picture of 'harder' one holds.
    """
    s = s.copy()
    gap = s[y].mean() - s[~y].mean()
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        t = s.copy(); t[y] = t[y] - mid * gap
        lo, hi = (mid, hi) if auroc(t, y) > target else (lo, mid)
    t = s.copy(); t[y] = t[y] - ((lo + hi) / 2) * gap
    return t


def hanley_mcneil_se(a, n1, n0):
    q1, q0 = a / (2 - a), 2 * a * a / (1 + a)
    v = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q0 - a * a)) / (n1 * n0)
    return float(np.sqrt(max(v, 0.0)))


def load_pairs():
    """Observed delta per scorer: score distortion from alpha=0 to alpha=1, per seed."""
    out = {}
    for f in sorted(glob.glob("PhaseB/outputs/per_example/*_a1.0_s*.npz")):
        seed = re.search(r"_s(\d+)\.npz$", f).group(1)
        arm = os.path.basename(f).split("_a")[0]
        base = f"PhaseB/outputs/per_example/{arm}_a0.0_s{seed}.npz"
        if not os.path.exists(base):
            base = f"PhaseB/outputs/per_example/Conventional_a0.0_s{seed}.npz"
        if os.path.exists(base):
            out[(arm, seed)] = (np.load(f), np.load(base))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = load_pairs()
    if not pairs:
        raise SystemExit("no per-example score pairs found")

    targets = [0.9995, 0.99, 0.97, 0.95, 0.90, 0.85, 0.80]
    rows, mde_rows = [], []

    for sc in SCORERS:
        # delta the intervention actually produces, median over seeds
        ds = []
        for (arm, seed), (hi, lo) in pairs.items():
            ds.append(np.median(np.abs(hi[sc] - lo[sc])))
        delta = float(np.median(ds))

        # observed operating point, on the alpha=0 arm
        anyk = sorted(pairs)[0]
        base = pairs[anyk][1]
        s0, y0 = np.asarray(base[sc], float), np.asarray(base["is_ood"], bool)
        a_obs = auroc(s0, y0)
        s_obs = susceptibility(margins(s0, y0), delta)

        rows.append(dict(scorer=sc, model="observed", target_auroc=np.nan,
                         achieved_auroc=a_obs, delta=delta, S=s_obs, observed=True))

        for t in targets:
            if t >= a_obs:
                continue
            for name, fn in [("noise", degrade_noise), ("shrink", degrade_shrink)]:
                sn = fn(s0, y0, t)
                rows.append(dict(scorer=sc, model=name, target_auroc=t,
                                 achieved_auroc=auroc(sn, y0), delta=delta,
                                 S=susceptibility(margins(sn, y0), delta),
                                 observed=False))

        # precision side: SE of AUROC vs external-set size, at each operating point
        n_id = int((~y0).sum())
        for t in [a_obs] + [x for x in targets if x < a_obs]:
            for n_ext in [200, 500, 1000, 2000, 5000]:
                se = hanley_mcneil_se(t, n_ext, n_id)
                mde_rows.append(dict(scorer=sc, auroc=t, n_external=n_ext,
                                     se_single=se, se_5seed=se / np.sqrt(5),
                                     mde_80pct=2.80 * se / np.sqrt(5)))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/headroom_resolution.csv", index=False)
    md = pd.DataFrame(mde_rows)
    md.to_csv(f"{OUT}/external_size_mde.csv", index=False)

    print("=== resolution: S(delta) as the domain leaves the ceiling ===")
    print("(S is the ceiling on |dAUROC| the design can express at all)\n")
    piv = df.pivot_table(index="scorer", columns="target_auroc", values="S", dropna=False)
    obs = df[df.observed].set_index("scorer")
    print("  observed AUROC / S:")
    for sc in SCORERS:
        print(f"    {sc:<8} A={obs.loc[sc,'achieved_auroc']:.5f}   S={obs.loc[sc,'S']:.6f}"
              f"   delta={obs.loc[sc,'delta']:.4g}")
    print("\n  S at reduced separability:")
    print(piv.round(5).to_string())

    print("\n=== precision: MDE at 80% power, 5 seeds, paired ===")
    sub = md[md.scorer == "maha"]
    print(sub.pivot_table(index="auroc", columns="n_external",
                          values="mde_80pct").round(5).to_string())

    print("\n=== resolution ratio S / MDE at n=2000 ===")
    print("(below 1 the scorer cannot express an effect as large as the")
    print(" smallest one the design can detect -- dead on arrival)\n")
    mde2 = md[md.n_external == 2000].set_index(["scorer", "auroc"])["mde_80pct"]
    rr = []
    for _, r in df.iterrows():
        a = r.achieved_auroc
        key = min(mde2.loc[r.scorer].index, key=lambda x: abs(x - a))
        rr.append(dict(scorer=r.scorer, model=r.model,
                       auroc=round(a, 4), S=r.S,
                       mde=mde2.loc[(r.scorer, key)],
                       ratio=r.S / mde2.loc[(r.scorer, key)]))
    rrd = pd.DataFrame(rr)
    rrd.to_csv(f"{OUT}/resolution_ratio.csv", index=False)
    print(rrd.pivot_table(index=["scorer", "model"], columns="auroc",
                          values="ratio").round(1).to_string())
    print(f"\nwrote {OUT}/headroom_resolution.csv and {OUT}/external_size_mde.csv")


if __name__ == "__main__":
    main()
