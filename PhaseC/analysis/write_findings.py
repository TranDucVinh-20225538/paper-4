"""Write PhaseC/FINDINGS.md from the audit CSVs: five questions, in order, every number read from a file.

Why: the findings document must not contain a number that is not in an output CSV, so this script is the only
author of FINDINGS.md. It reads outputs/generalisation_audit.csv, check1_readings.csv, bounds_evaluation.csv,
check2_feasibility.json, check3_construction.csv and Phase B's seed-level susceptibility table (the PanDerm reach
reference), and writes tables and sentences computed from them. Where an output is absent the section says so
instead of inferring. Target length: four pages (it warns above ~230 lines).

The five questions as the pipeline implements them (the brief did not enumerate them):
    Q1  When a scorer's AUROC is flat, is it invariant, insensitive, or cancelling (Check 1)?
    Q2  Does Phase B's reach ordering of the scorers (PanDerm: maha << knn < kde < cosine) hold on other backbones?
    Q3  Do the affine bounds (Theorem 2 for maha, Lemma 2 for cosine) hold, and how tight are they?
    Q4  Does the audit discriminate: do the controls move, and does |dAUROC| <= F <= S(delta_max) hold everywhere?
    Q5  What do the adapters do across seeds and doses, is a seed-level test feasible (Check 2), and was the
        suppressed direction the one separating the OOD sets (Check 3)?

    python3 PhaseC/analysis/write_findings.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
PHASEC = HERE.parent
REPO = PHASEC.parent
sys.path.insert(0, str(HERE))

# kde = Phase B σ̄ bandwidth (published column). kde_det = addition with det(Σ)^(1/(2d));
# never treat kde_det as an erratum replacing kde (see notes/kde_bandwidth_mechanism.md).
SCORERS = ("maha", "cosine", "knn10", "kde", "kde_det", "rmd")
AFFINE = ("ortho_scale", "spectral_temper", "zca", "general_affine")
OUTSIDE = ("radial_l2", "residual_nonlinear", "adapter")
LABELS = ("moved", "invariant", "insensitive", "cancelling", "inconsistent")
CODE = {0: "undefined", 1: "moved", 2: "invariant", 3: "insensitive", 4: "cancelling", 5: "inconsistent"}
PHASEB_KIND = {"maha": "additive", "cosine": "additive", "kde": "additive", "knn10": "ratio"}   # as Phase B read them
PANDERM_QUOTED = {"maha": 6.5e-5, "cosine": 0.0478}      # the figures quoted in the brief
TAU = 0.002


def num(x, p=3):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x == 0:
        return "0"
    ax = abs(x)
    return f"{x:.{p - 1}e}" if (ax < 1e-3 or ax >= 1e4) else f"{x:.{p}g}"


def table(df: pd.DataFrame, index=True, p=3):
    if index:
        df = df.reset_index()
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(num(v, p) if not isinstance(v, str) else v for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def load(out: Path):
    a = pd.read_csv(out / "generalisation_audit.csv") if (out / "generalisation_audit.csv").is_file() else None
    c1 = pd.read_csv(out / "check1_readings.csv") if (out / "check1_readings.csv").is_file() else None
    b = pd.read_csv(out / "bounds_evaluation.csv") if (out / "bounds_evaluation.csv").is_file() else None
    c3 = pd.read_csv(out / "check3_construction.csv") if (out / "check3_construction.csv").is_file() else None
    c2 = json.loads((out / "check2_feasibility.json").read_text()) if (out / "check2_feasibility.json").is_file() else None
    return a, c1, b, c2, c3


def panderm_reference():
    p = REPO / "PhaseB" / "outputs" / "seeds10" / "rq1_4_susceptibility_observed.csv"
    if not p.is_file():
        return None
    d = pd.read_csv(p)
    m = d[d["quantile"] == "median"]
    out = m.groupby("scorer").S.median().to_dict()
    out["_mean"] = m.groupby("scorer").S.mean().to_dict()
    return out


def wide(a):
    w = a[a.ood_set != "ALL"].pivot_table(index=["backbone", "id_set", "ood_set", "operator", "dose", "scorer", "seed"],
                                          columns="quantity", values="value", aggfunc="first")
    return w.reset_index()


def is_control(w):
    """Dose-0 configurations are identity maps by construction: reported as controls, excluded from tallies."""
    return ((w.operator == "spectral_temper") & (w.dose == 0)) | ((w.operator == "zca") & (w.dose == 0)) | \
           ((w.operator == "adapter") & (w.dose == 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=PHASEC / "outputs")
    ap.add_argument("--findings", type=Path, default=PHASEC / "FINDINGS.md")
    ap.add_argument("--id-set", default="cifar10")
    ar = ap.parse_args(argv)
    a, c1, b, c2, c3 = load(ar.out)
    L = ["# Phase C findings: the flat-response audit across backbones", ""]
    if a is None:
        L += ["No `generalisation_audit.csv` yet: run `PhaseC/run_pipeline.sh`. Nothing to report."]
        ar.findings.write_text("\n".join(L) + "\n")
        print("[findings] no audit CSV; wrote stub")
        return 0

    a = a[a.id_set == ar.id_set]
    w = wide(a)
    w = w.assign(ctrl=is_control(w))
    wt = w[~w.ctrl & (w.operator != "identity")]
    done = a[a.quantity == "config_complete"]
    bbs = sorted(a.backbone.unique())
    L += [f"Source: `PhaseC/outputs/*.csv` (ID set `{ar.id_set}`); regenerated by `write_findings.py`. "
          f"Scorers refit on every transformed training set. Check 1 at tau={TAU}, S_high=0.02, F0=1e-4, reach read at "
          "S(delta_med) unless stated. dAUROC is against the identity baseline of the same backbone and OOD set.", ""]
    cov = done.groupby(["backbone", "operator"]).size().unstack(fill_value=0)
    ood_sets = sorted(w.ood_set.unique())
    L += [f"**Coverage.** Backbones: {', '.join(bbs)}. OOD rungs: {', '.join(ood_sets)}. Completed configurations per backbone: "
          + "; ".join(f"{k} {int(v)}" for k, v in done.groupby('backbone').size().items()) + ". "
          "A rung or backbone absent here was not extracted (dataset or checkpoint unavailable); nothing is extrapolated to it.", ""]

    # ------------------------------------------------------------------ Q1
    L += ["## Q1. When AUROC is flat, why? (Check 1)", ""]
    wt = wt.assign(label=wt.check1_code_Smed.map(CODE))
    rows = []
    for grp, names in (("affine", AFFINE), ("outside", OUTSIDE)):
        for sc in SCORERS:
            s = wt[wt.operator.isin(names) & (wt.scorer == sc)]
            if len(s) == 0:
                continue
            vc = s.label.value_counts(normalize=True)
            flat = s[s.label != "moved"]
            rows.append({"group": grp, "scorer": sc, "n": len(s), **{l: float(vc.get(l, 0.0)) for l in LABELS},
                         "flat_invariant_share": float((flat.label == "invariant").mean()) if len(flat) else np.nan})
    q1 = pd.DataFrame(rows)
    L += [table(q1, index=False, p=2), "",
          "Shares of configurations (dose > 0, all backbones, OOD rungs, seeds). `flat_invariant_share` is the share of *flat* "
          "readings (AUROC change < tau) labelled invariant rather than insensitive or cancelling."]
    if c1 is not None:
        c1 = c1[c1.id_set == ar.id_set]
        key = ["backbone", "id_set", "ood_set", "operator", "dose", "seed", "scorer", "S_basis"]
        prim = c1[(c1.tau == TAU) & (c1.S_high == 0.02)][key + ["label"]].rename(columns={"label": "primary"})
        m = c1.merge(prim, on=key)
        m = m[~(((m.operator.isin(["spectral_temper", "zca", "adapter"])) & (m.dose == 0)))]
        stab = m[(m.S_basis == "med")].assign(same=lambda d: d.label == d.primary).groupby("scorer").same.mean()
        flat_only = m[(m.S_basis == "med") & (m.primary != "moved")].assign(same=lambda d: d.label == d.primary).groupby("scorer").same.mean()
        sb = m[m.S_basis == "med"].groupby(["scorer"]).primary.apply(lambda s: (s == "invariant").mean())
        mx = m[m.S_basis == "max"].groupby(["scorer"]).primary.apply(lambda s: (s == "invariant").mean())
        st = pd.DataFrame({"label_stable_over_grid": stab, "stable_flat_only": flat_only,
                           "invariant_share_S_med": sb, "invariant_share_S_max": mx}).reindex(SCORERS)
        L += ["", "Sensitivity over tau in [.001,.01] and S_high in [.005,.1] (20 cells): the fraction of cells whose label equals the "
              "registered one, and the invariant share when reach is read at delta_max instead of delta_med:", "", table(st, p=2)]
        L += ["", f"Averaged over scorers the invariant share is {num(float(sb.mean()), 2)} when reach is read at delta_med and "
              f"{num(float(mx.mean()), 2)} at delta_max; the median reading is the one used in the tables above."]
    L += [""]

    # ------------------------------------------------------------------ Q2
    L += ["## Q2. Does the PanDerm reach ordering generalise?", ""]
    ref = panderm_reference()
    ad1 = w[(w.operator == "adapter") & (w.dose == 1.0)]
    z1 = w[(w.operator == "zca") & (w.dose == 1.0)]

    def reach_table(sub, label):
        if len(sub) == 0:
            return None, None
        t = {}
        for bbn, g in sub.groupby("backbone"):
            t[bbn] = {sc: float(g[g.scorer == sc].S_med.median()) for sc in SCORERS}
        t = pd.DataFrame(t).reindex(SCORERS)
        if ref:
            t["PanDerm (Phase B; knn10 ratio form)"] = pd.Series({sc: ref.get(sc, np.nan) for sc in SCORERS})
        return t, label

    for sub, label in ((ad1, "adapter alpha=1 (median over seeds, OOD rungs)"), (z1, "ZCA alpha=1 (median over OOD rungs)")):
        t, _ = reach_table(sub, label)
        if t is None:
            L += [f"S(delta_med), {label}: not available.", ""]
            continue
        L += [f"S(delta_med), additive form, {label}:", "", table(t, p=2), ""]
        if "knn10" in sub.columns and sub.S_ratio_med_p1.notna().any():
            kr = sub[sub.scorer == "knn10"].groupby("backbone").S_ratio_med_p1.median()
            L += ["knn10 in its Phase B (ratio, power 1) form: " + ", ".join(f"{k} {num(v, 2)}" for k, v in kr.items())
                  + (f"; PanDerm {num(ref['knn10'], 2)}" if ref else "") + ".", ""]
    if ref:
        L += [f"PanDerm reference recomputed from `rq1_4_susceptibility_observed.csv` (median over 10 seeds of the median-delta rows): "
              f"maha {num(ref['maha'], 3)}, cosine {num(ref['cosine'], 3)}, kde {num(ref['kde'], 3)}, knn10 (ratio) {num(ref['knn10'], 3)}. "
              f"The brief quotes maha {PANDERM_QUOTED['maha']:.1e} and cosine {PANDERM_QUOTED['cosine']}; against the medians that is "
              f"{PANDERM_QUOTED['maha'] / ref['maha'] - 1:+.1%} and {PANDERM_QUOTED['cosine'] / ref['cosine'] - 1:+.1%}, against the seed means "
              f"(maha {num(ref['_mean']['maha'], 3)}, cosine {num(ref['_mean']['cosine'], 3)}) {PANDERM_QUOTED['maha'] / ref['_mean']['maha'] - 1:+.1%} and "
              f"{PANDERM_QUOTED['cosine'] / ref['_mean']['cosine'] - 1:+.1%}. The recomputed medians are used for comparison.", ""]
        order_ref = [s for s in ("maha", "knn10", "kde", "cosine")]
        conc = []
        for bbn, g in ad1.groupby("backbone"):
            vals = {}
            for sc in order_ref:
                col = "S_ratio_med_p1" if PHASEB_KIND[sc] == "ratio" else "S_med"
                vals[sc] = float(g[g.scorer == sc][col].median())
            tau_k = kendalltau([ref[s] for s in order_ref], [vals[s] for s in order_ref]).statistic
            conc.append((bbn, " < ".join(sorted(order_ref, key=vals.get)), tau_k))
        if conc:
            L += ["Ordering of maha, knn10 (ratio), kde, cosine by S(delta_med) of the alpha=1 adapter, vs PanDerm ("
                  + " < ".join(sorted(order_ref, key=ref.get)) + "):", ""]
            L += [table(pd.DataFrame(conc, columns=["backbone", "ordering (low to high reach)", "Kendall tau vs PanDerm"]), index=False, p=2), ""]
    base = w[w.operator == "identity"] if "auroc" in w else None
    lad = ad1.groupby(["ood_set", "scorer"]).S_med.median().unstack().reindex(columns=list(SCORERS))
    ida = a[(a.operator == "identity") & (a.quantity == "auroc") & (a.ood_set != "ALL")].groupby(["ood_set", "scorer"]).value.median().unstack().reindex(columns=list(SCORERS))
    # Per (backbone × OOD × scorer) identity AUROC — this is the difficulty axis, not near/far labels.
    ida_bb = a[(a.operator == "identity") & (a.quantity == "auroc") & (a.ood_set != "ALL")]
    if len(ida):
        # Order rungs by mean baseline AUROC (easy = high AUROC at the top of the ladder).
        rung_order = ida.mean(axis=1).sort_values(ascending=False).index.tolist()
        ida = ida.reindex(rung_order)
        lad = lad.reindex(rung_order) if len(lad) else lad
        L += ["Difficulty axis = measured identity-baseline AUROC (not OpenOOD near/far labels). "
              "Rungs ordered by mean AUROC over scorers (high AUROC = easiest / farthest). "
              "OpenOOD CIFAR-10 labels (informational): near = cifar100, tin; far = mnist, svhn, texture, places365.", "",
              "Baseline AUROC per rung × scorer (median over backbones):", "", table(ida, p=3), ""]
        if len(ida_bb):
            per = ida_bb.pivot_table(index=["backbone", "ood_set"], columns="scorer", values="value", aggfunc="median")
            per = per.reindex(columns=list(SCORERS))
            # Sort within each backbone by mean AUROC
            rows = []
            for bbn, g in per.groupby(level=0):
                means = g.mean(axis=1).sort_values(ascending=False)
                rows.append(g.reindex(means.index))
            per_sorted = pd.concat(rows)
            L += ["Baseline AUROC per backbone × OOD × scorer (the axis used when reporting reach vs difficulty):",
                  "", table(per_sorted, p=3), ""]
        if len(lad):
            L += ["S(delta_med) of the alpha=1 adapter per rung (median over backbones and seeds), "
                  "same difficulty order:", "", table(lad, p=2), ""]
            # Spearman of reach vs baseline difficulty per scorer
            if "auroc" in (base.columns if base is not None else []):
                join = ad1.merge(
                    ida_bb.rename(columns={"value": "auroc_baseline"})[["backbone", "ood_set", "scorer", "auroc_baseline"]],
                    on=["backbone", "ood_set", "scorer"], how="left")
                spear = []
                for sc, g in join.groupby("scorer"):
                    g = g.dropna(subset=["S_med", "auroc_baseline"])
                    if len(g) < 3:
                        continue
                    rho = spearmanr(g["auroc_baseline"], g["S_med"]).statistic
                    spear.append((sc, float(rho), len(g)))
                if spear:
                    L += ["Spearman rho(baseline AUROC, S_med) of the alpha=1 adapter (positive ⇒ easier rungs show higher reach):",
                          "", table(pd.DataFrame(spear, columns=["scorer", "spearman_rho", "n"]), index=False, p=3), ""]

    # ------------------------------------------------------------------ Q3
    L += ["## Q3. Do the affine bounds hold, and how tight?", ""]
    if b is not None and len(b[b.id_set == ar.id_set]):
        bb = b[b.id_set == ar.id_set]
        bb = bb[~((bb.operator.isin(["spectral_temper", "zca"])) & (bb.dose == 0))]
        g = bb.groupby(["bound", "backbone"]).agg(n=("bound_holds", "size"), holds=("bound_holds", "mean"),
                                                   tight_med=("tightness_max", "median"), tight_max=("tightness_max", "max"),
                                                   pred_over_obs=("predicted_max", lambda s: np.nan))
        po = (bb.predicted_max / bb.observed_max.replace(0, np.nan)).groupby([bb.bound, bb.backbone]).median()
        g["pred_over_obs_med"] = po
        g = g.drop(columns="pred_over_obs")
        L += [table(g, p=2), ""]
        t2 = bb[bb.bound == "theorem2_maha"]
        l2 = bb[bb.bound == "lemma2_cosine"]
        if len(t2):
            L += [f"Theorem 2 holds in {t2.bound_holds.mean():.1%} of {len(t2)} affine configurations; the observed change of the *refit* maha "
                  f"score stays under the predicted bound in {t2.actual_within_predicted.mean():.1%}; the refit and the closed-form q_B agree to a "
                  f"worst relative error of {num(t2.refit_vs_qB_max_rel.max(), 2)} (float32 rounding of the transformed points). "
                  f"By operator (median tightness = observed/predicted): "
                  + ", ".join(f"{k} {num(v, 2)}" for k, v in t2.groupby('operator').tightness_max.median().items()) + ".", ""]
        if len(l2):
            L += [f"Lemma 2 (cosine, retained-quantity form) holds in {l2.bound_holds.mean():.1%} of {len(l2)} configurations; the kappa-only "
                  f"Lemma 1 form holds in {l2.lemma1_bound_holds.mean():.1%} (it assumes the map acts about the origin, the audit's maps act about the fit mean, "
                  "so Lemma 1 is shown for comparison only). Median tightness by operator: "
                  + ", ".join(f"{k} {num(v, 2)}" for k, v in l2.groupby('operator').tightness_max.median().items()) + ".", ""]
    else:
        L += ["`bounds_evaluation.csv` not available.", ""]

    # ------------------------------------------------------------------ Q4
    L += ["## Q4. Does the audit discriminate? Controls and the inequality chain", ""]
    wt = wt.assign(abs_d=wt.delta_auroc.abs())
    sel = {"ortho_scale": wt.operator == "ortho_scale", "temper g=1": (wt.operator == "spectral_temper") & (wt.dose == 1.0),
           "zca a=1": (wt.operator == "zca") & (wt.dose == 1.0), "general_affine": wt.operator == "general_affine",
           "radial_l2": wt.operator == "radial_l2",
           "residual t=.5": (wt.operator == "residual_nonlinear") & (wt.dose == 0.5),
           "adapter a=1": (wt.operator == "adapter") & (wt.dose == 1.0)}
    resp = pd.DataFrame({k: wt[m].groupby("scorer").abs_d.median() for k, m in sel.items()}).reindex(SCORERS)
    L += ["Median |dAUROC| over backbones, OOD rungs and seeds (invariance predicts ~0 for the affine columns where a scorer is protected; "
          "the last three are outside the invariance groups and are the controls that should move):", "", table(resp, p=2), ""]
    ctrl = w[w.ctrl]
    if len(ctrl):
        L += [f"Dose-0 controls (gamma=0, alpha=0, adapter alpha=0 are identity maps): max |dAUROC| = {num(ctrl.delta_auroc.abs().max(), 2)} over {len(ctrl)} readings.", ""]
    chain = []
    for name, col in (("abs(dAUROC) <= F (+4 se + 3/n_pairs)", "check_dauroc_le_F"), ("F <= S(delta_max)", "check_F_le_Smax"),
                      ("F <= S_ratio(delta_ratio_max), where valid", "check_F_le_Sratio_max")):
        x = w[(w.operator != "identity")][col].dropna()
        chain.append({"inequality": name, "n": len(x), "pass_rate": float(x.mean()) if len(x) else np.nan, "failures": int((x == 0).sum())})
    L += [table(pd.DataFrame(chain), index=False, p=4), ""]
    bad = w[(w.operator != "identity") & (w.check_F_le_Smax == 0)]
    if len(bad):
        L += [f"{len(bad)} F > S(delta_max) readings: a violation of an exact inequality on identical pairs, which means a bug or a stale input; "
              "first rows: " + "; ".join(f"{r.backbone}/{r.ood_set}/{r.operator}/{r.dose}/{r.scorer}" for r in bad.head(5).itertuples()) + ".", ""]

    # ------------------------------------------------------------------ Q5
    L += ["## Q5. Adapters across seeds and doses; Check 2 and Check 3", ""]
    ad = w[(w.operator == "adapter") & (w.dose > 0)]
    if len(ad):
        per_seed = ad[ad.dose == 1.0].groupby(["backbone", "scorer", "seed"]).delta_auroc.mean().reset_index()
        s = per_seed.groupby(["backbone", "scorer"]).delta_auroc.agg(mean="mean", sd="std", n="size",
                                                                     flat_seeds=lambda v: float((v.abs() < TAU).mean()))
        L += ["dAUROC at alpha=1 (mean over OOD rungs per seed, then mean and sd over seeds; `flat_seeds` = share of seeds with |dAUROC| < tau):", "",
              table(s, p=2), ""]
        dr = ad.groupby(["scorer", "dose"]).delta_auroc.median().unstack().reindex(SCORERS)
        L += ["Dose response, median dAUROC over backbones, rungs and seeds:", "", table(dr, p=2), ""]
    else:
        L += ["No adapter configurations completed (run `train_adapters.py`, then the audit).", ""]
    if c2:
        p = c2["primary"]
        L += [f"**Check 2.** With k={p['k_seeds']} seeds the exact permutation floor is {num(p['floor'], 2)} (10! = {p['n_permutations']:,} permutations); "
              f"Holm's first threshold over the {p['m_tests']}-test family ({c2['family_definition']}) is {num(p['holm_first'], 2)}; "
              f"reachable: {p['reachable']}. Sampled at {p['n_sample']:,} permutations the floor is {num(p['sampled_floor'], 2)}, reachable: {p['sampled_reachable']}. "
              f"The smallest k that clears this family exactly is {p['min_k_for_reachable']}.", ""]
    if c3 is not None and len(c3[c3.id_set == ar.id_set]):
        c = c3[c3.id_set == ar.id_set]
        t = c.groupby(["backbone", "w_kind", c.ood_set == "ref"]).abs_projection.median().unstack()
        t.columns = ["OOD rungs" if not k else "reference (sanity)" for k in t.columns]
        L += ["**Check 3.** Median |cos| between the unit OOD mean shift (from the ID-train mean) and the suppressed direction w "
              "(`w_construction` = ID minus reference mean; `w_learned` = the direction the trained adapter moves the reference relative to ID). "
              "Near 1 means the adapter removes the axis that separates the OOD set, so a large response is the construction working; near 0 means unrelated:", "",
              table(t, p=2), ""]
        per = c[(c.w_kind == "w_construction") & (c.ood_set != "ref")].groupby("ood_set").abs_projection.median()
        L += ["By rung (w_construction, median over backbones and seeds): " + ", ".join(f"{k} {num(v, 2)}" for k, v in per.items()) + ".", ""]

    L += ["## Provenance", "",
          "Checkpoint and data hashes, seeds, package versions and GPU are in `PhaseC/outputs/manifest.json` (`write_manifest.py`). "
          "Seeds: adapters 42..132 (10), operators per `run_audit.OPERATOR_SEEDS`, pairs 2e6 with PAIR_SEED. The ten adapter seeds are replicates of "
          "one training recipe on one backbone, not independent datasets: claims about a backbone rest on seed variation, claims across backbones on five backbones.", ""]
    text = "\n".join(L) + "\n"
    ar.findings.write_text(text)
    n = text.count("\n")
    print(f"[findings] wrote {ar.findings} ({n} lines){'  WARNING: over ~4 pages' if n > 230 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
