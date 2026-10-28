"""The flat-response audit: for every operator, dose and scorer, what moved, what could have moved, and what did.

Why: a scorer whose AUROC stays flat under an intervention is either protected by an invariance, or was never
in reach of the perturbation (few ID-OOD score pairs close enough to flip), or flips offset. Reading AUROC alone cannot
tell these apart; this script measures, for each configuration, the three quantities that can (reach.py):
    S  how many pairs a perturbation of the observed size could reorder (ranking reach),
    F  how many actually reordered (realized reordering),
    dAUROC  the net change, which never exceeds F in absolute value, and F never exceeds S(delta_max).

For each (backbone, ID set) it
  1. loads the pinned embeddings (assets/embeddings/<backbone>/<id_set>/*.npz),
  2. for each operator/dose/seed transforms train and every evaluation population (ID test + each OOD rung),
  3. REFITS EVERY SCORER on the transformed train set (Theorem 1 is about the refit scorer, not the old one),
  4. records AUROC, dAUROC against the identity baseline, FPR@95TPR, referrals per 1000, delta_med / delta_max of
     |s~ - s|, S at both, F (with flip/tie split), the two inequality checks with their slack, the Check 1 reading at the
     registered thresholds and over a sensitivity grid, the ratio-margin reach variant, and
  5. for affine operators evaluates Theorem 2 (maha) and Lemma 2 (cosine) against the observed change.
Outputs are long-format and appended (header once, flock-guarded, one write per configuration, completion marker last)
so a killed run resumes where it stopped and backbones can run as parallel jobs:
    PhaseC/outputs/generalisation_audit.csv  backbone,id_set,ood_set,operator,dose,scorer,seed,quantity,value
    PhaseC/outputs/bounds_evaluation.csv     one row per affine configuration and bound, every input retained
    PhaseC/outputs/check1_readings.csv       label per (config, ood, scorer, S basis, tau, S_high) at F0 = 1e-4
    PhaseC/outputs/check2_feasibility.json   can a seed-level permutation test reject at all (exact floor vs Holm)
    PhaseC/outputs/check3_construction.csv   projection of each OOD mean shift on the adapter's suppressed direction

Seeds. OPERATOR_SEEDS below are explicit; random operators draw from default_rng([seed, OPERATOR_STREAM[op]]) so that two
operators sharing a seed number do not share a random stream. Pair sampling uses PAIR_SEED for every
(S, F) computation of every configuration, so S, F and the inequality F <= S(delta_max) are evaluated on identical pairs.

    python3 PhaseC/analysis/run_audit.py --backbone resnet50_imagenet --id-set cifar10
    python3 PhaseC/analysis/run_audit.py --backbone resnet50_imagenet --id-set cifar10 --quick --n-pairs 200000
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bounds  # noqa: E402
import checks  # noqa: E402
import operators as ops  # noqa: E402
import reach  # noqa: E402
import scorers  # noqa: E402
from train_adapters import ADAPTER_SEEDS, ADAPTER_ROOT, EMB_ROOT  # noqa: E402

PHASEC = HERE.parent
OUT = PHASEC / "outputs"
AUDIT_CSV, BOUNDS_CSV = "generalisation_audit.csv", "bounds_evaluation.csv"
CHECK1_CSV, CHECK3_CSV, CHECK2_JSON = "check1_readings.csv", "check3_construction.csv", "check2_feasibility.json"

# ----------------------------------------------------------------------------- registered grids and seeds
TEMPER_GAMMAS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
ZCA_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RESIDUAL_T = (0.05, 0.1, 0.25, 0.5)
ADAPTER_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
GENERAL_AFFINE_COND = 10.0
#: seeds per operator family (explicit; nothing is imported from Phase B)
OPERATOR_SEEDS = {
    "deterministic": [0],                  # spectral_temper, zca_whiten, radial_l2, identity
    "ortho_scale": [0, 1, 2],               # centred at train mean (Lemma 2 regime for cosine)
    "ortho_linear": [0, 1, 2],              # pure T(z)=c Q z; gate: maha/knn10/cosine pass; kde=σ̄ exception (see kde_det)
    "general_affine": [0, 1, 2, 3, 4],
    "residual_nonlinear": [0, 1, 2, 3, 4],
    "adapter": list(ADAPTER_SEEDS),
}
OPERATOR_STREAM = {"ortho_scale": 1, "ortho_linear": 4, "general_affine": 2, "residual_nonlinear": 3}
PAIR_SEED = 20260101            # pair sampling for S and F, every configuration
BOUNDS_SEED = 20260102          # the subsample of the evaluation set Theorem 2 is evaluated on
AUDIT_SUBSAMPLE_SEED = 20260103  # --max-ood / --max-train subsampling (smoke runs only)
THRESHOLDS = {"tau": 0.002, "S_high": 0.02, "F0": 1e-4}
SENS_TAU = (0.001, 0.002, 0.005, 0.01)
SENS_SHIGH = (0.005, 0.01, 0.02, 0.05, 0.1)
CHECK1_CODE = {"undefined": 0, "moved": 1, "invariant": 2, "insensitive": 3, "cancelling": 4, "inconsistent": 5}
BOUNDS_N_PER_POP = 600          # evaluation points per population (ID test, each OOD) for Theorem 2
RATIO_VALID_MIN = 0.99          # ratio reach is applicable when >= 99% of scores are positive before and after
RATIO_PREFERRED_SPEARMAN = 0.5  # ... and "preferred" when |s~-s| rises with s (multiplicative-looking perturbation)
TOL = 1e-12


# ----------------------------------------------------------------------------- output helpers
def _append(path: Path, df: pd.DataFrame):
    """Append with a single header, under an exclusive flock so parallel backbones do not interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, header=False, index=False, float_format="%.10g")
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                f.write(",".join(df.columns) + "\n")
            f.write(buf.getvalue())
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def completed_keys(path: Path, backbone, id_set):
    if not path.is_file():
        return set()
    done = set()
    for ch in pd.read_csv(path, usecols=["backbone", "id_set", "operator", "dose", "seed", "quantity"], chunksize=500_000):
        ch = ch[(ch.backbone == backbone) & (ch.id_set == id_set) & (ch.quantity == "config_complete")]
        done.update(zip(ch.operator, ch.dose.round(8), ch.seed))
    return done


# ----------------------------------------------------------------------------- data
def load_embeddings(backbone, id_set, root=EMB_ROOT, max_ood=None, max_train=None):
    d = Path(root) / backbone / id_set
    tr = np.load(d / "id_train.npz")
    Ztr, ytr = tr["embeddings"], tr["labels"]
    if max_train and len(Ztr) > max_train:
        idx = np.sort(np.random.default_rng(AUDIT_SUBSAMPLE_SEED).choice(len(Ztr), max_train, replace=False))
        Ztr, ytr = Ztr[idx], ytr[idx]
    te = np.load(d / "id_test.npz")
    Zte = te["embeddings"]
    if max_ood and len(Zte) > max_ood:
        Zte = Zte[np.sort(np.random.default_rng(AUDIT_SUBSAMPLE_SEED + 1).choice(len(Zte), max_ood, replace=False))]
    oods = {}
    for p in sorted(d.glob("ood_*.npz")):
        name = p.stem[4:]
        if name == "ref":
            continue                      # adapter construction only; never evaluated
        Z = np.load(p)["embeddings"]
        if max_ood and len(Z) > max_ood:
            Z = Z[np.sort(np.random.default_rng(AUDIT_SUBSAMPLE_SEED + 2).choice(len(Z), max_ood, replace=False))]
        oods[name] = Z
    return Ztr, ytr, Zte, oods


# ----------------------------------------------------------------------------- operators on (train, eval)
def affine_apply(Z, meta):
    """T(z) = A (z - mu) + mu + b from a meta dict, float64 internally, float32 out (same arithmetic as operators)."""
    out = (np.asarray(Z, np.float64) - meta["mu"]) @ meta["A"].T + meta["mu"]
    if meta["b"] is not None:
        out = out + meta["b"]
    return out.astype(np.float32)


def make_config_list(quick=False, only=None, adapter_available=True):
    """[(operator, dose, seed)] in a fixed order. identity first (the baseline every dAUROC is against)."""
    S = OPERATOR_SEEDS
    cfg = [("identity", 0.0, 0)]
    if quick:
        cfg += [("ortho_linear", 1.0, 0)]
        cfg += [("ortho_scale", 1.0, 0)]
        cfg += [("spectral_temper", g, 0) for g in (0.0, 0.5, 1.0)]
        cfg += [("zca", a, 0) for a in (0.5, 1.0)]
        cfg += [("general_affine", GENERAL_AFFINE_COND, 0), ("radial_l2", 1.0, 0)]
        cfg += [("residual_nonlinear", 0.25, 0)]
        if adapter_available:
            cfg += [("adapter", a, s) for a in (0.0, 1.0) for s in S["adapter"][:2]]
    else:
        cfg += [("ortho_linear", 1.0, s) for s in S["ortho_linear"]]
        cfg += [("ortho_scale", 1.0, s) for s in S["ortho_scale"]]
        cfg += [("spectral_temper", g, 0) for g in TEMPER_GAMMAS]
        cfg += [("zca", a, 0) for a in ZCA_ALPHAS]
        cfg += [("general_affine", GENERAL_AFFINE_COND, s) for s in S["general_affine"]]
        cfg += [("radial_l2", 1.0, 0)]
        cfg += [("residual_nonlinear", t, s) for t in RESIDUAL_T for s in S["residual_nonlinear"]]
        if adapter_available:
            cfg += [("adapter", a, s) for s in S["adapter"] for a in ADAPTER_ALPHAS]
    if only:
        cfg = [c for c in cfg if c[0] in only or c[0] == "identity"]
    return cfg


def transform(op, dose, seed, Ztr, E, adapter_state=None):
    """(Ztr_t, E_t, meta). Everything is fitted on Ztr only; E (ID test + OODs, concatenated) is only mapped."""
    if op == "identity":
        return Ztr, E, {"operator": "identity", "linear": True}
    if op in ("spectral_temper", "zca"):
        fn = (lambda Z: ops.spectral_temper(Ztr, Z, dose)) if op == "spectral_temper" else (lambda Z: ops.zca_whiten(Ztr, Z, dose))
        Xtr, meta = fn(Ztr)
        return Xtr, affine_apply(E, meta), meta
    if op == "ortho_scale":
        rng = np.random.default_rng([seed, OPERATOR_STREAM[op]])
        Xtr, meta = ops.random_orthogonal_scaled(Ztr, rng)          # mu = train mean (centred)
        return Xtr, affine_apply(E, meta), meta
    if op == "ortho_linear":
        rng = np.random.default_rng([seed, OPERATOR_STREAM[op]])
        Xtr, meta = ops.random_orthogonal_scaled_linear(Ztr, rng)   # T(z) = c Q z, mu = 0
        return Xtr, affine_apply(E, meta), meta
    if op == "general_affine":
        rng = np.random.default_rng([seed, OPERATOR_STREAM[op]])
        Xtr, meta = ops.general_affine(Ztr, rng, cond_target=dose)  # mu = train mean, b from train norms
        return Xtr, affine_apply(E, meta), meta
    if op == "radial_l2":
        Xtr, meta = ops.radial_l2_normalise(Ztr, fit_pop=Ztr)
        XE, _ = ops.radial_l2_normalise(E, target_norm=meta["target_norm"])
        return Xtr, XE, meta
    if op == "residual_nonlinear":
        rng = np.random.default_rng([seed, OPERATOR_STREAM[op]])
        Z64 = Ztr.astype(np.float64)
        mu = Z64.mean(0)
        rms = float(np.sqrt(Z64.var(0).mean()))
        W = ops._haar_orthogonal(Ztr.shape[1], rng) / rms           # ||W||_2 = 1/rms: pre-activation entries are O(1)
        t_abs = dose * rms                                          # dose t is relative to the feature scale
        Xtr, meta = ops.residual_nonlinear(Ztr, mu, W, t_abs)
        XE, _ = ops.residual_nonlinear(E, mu, W, t_abs)
        meta.update({"t_relative": float(dose), "feature_rms": rms})
        return Xtr, XE, meta
    if op == "adapter":
        Xtr, meta = ops.apply_adapter(Ztr, adapter_state, dose, gelu_approximate=False)
        XE, _ = ops.apply_adapter(E, adapter_state, dose, gelu_approximate=False)
        return Xtr, XE, meta
    raise ValueError(op)


# ----------------------------------------------------------------------------- per-config evaluation
def ratio_stats(s0, s1):
    ok = (s0 > 0) & (s1 > 0)
    if not ok.any():
        return {"valid_frac": 0.0, "med": float("nan"), "max": float("nan")}
    r = np.maximum(s1[ok] / s0[ok], s0[ok] / s1[ok])
    return {"valid_frac": float(ok.mean()), "med": float(np.median(r)), "max": float(r.max())}


def reading_rows(d_auroc, S, F, n_pairs):
    """Check 1 labels: the registered thresholds, and the sensitivity grid in tau and S_high (F0 fixed)."""
    out = {}
    for tau in SENS_TAU:
        for sh in SENS_SHIGH:
            out[(tau, sh)] = reach.check1_reading(d_auroc, S, F, tau=tau, S_high=sh, F0=THRESHOLDS["F0"], n_pairs=n_pairs)["label"]
    return out


def evaluate_population(s_before, s_after, n_id, sl, n_pairs, base_auroc, base_fpr):
    """All per-(OOD set, scorer) quantities. s_* are score vectors over E; the population is ID test + this OOD slice."""
    idx = np.r_[0:n_id, sl.start:sl.stop]
    b, a = s_before[idx], s_after[idx]
    is_ood = np.r_[np.zeros(n_id, bool), np.ones(sl.stop - sl.start, bool)]
    sb_id, sb_ood = s_before[:n_id], s_before[sl]
    sa_id, sa_ood = s_after[:n_id], s_after[sl]
    q = {}
    q["auroc"] = scorers.fast_auroc(a, is_ood)
    q["fpr95"] = scorers.fpr_at_95_tpr(a, is_ood)
    q["delta_auroc"] = q["auroc"] - base_auroc
    q["delta_fpr95"] = q["fpr95"] - base_fpr
    q["delta_referrals_per_1000"] = 1000.0 * q["delta_fpr95"]
    diff = np.abs(a - b)
    q["delta_med"], q["delta_max"] = float(np.median(diff)), float(diff.max())
    # Rank preservation over the evaluation population (ID test + this OOD). Spearman
    # of scores before vs after; discordant pairs among a PAIR_SEED sample of the same size
    # as the ID–OOD F sample. Reach cannot evidence invariance; this can.
    if np.ptp(b) == 0 and np.ptp(a) == 0:
        rho_rank = 1.0
    elif np.ptp(b) == 0 or np.ptp(a) == 0:
        rho_rank = float("nan")
    else:
        rho_rank = float(spearmanr(b, a).statistic)
    q["rank_spearman"] = rho_rank
    q["rank_one_minus_rho"] = float(1.0 - rho_rank) if np.isfinite(rho_rank) else float("nan")
    rng_r = np.random.default_rng(PAIR_SEED)
    ii = rng_r.integers(0, len(b), n_pairs)
    jj = rng_r.integers(0, len(b), n_pairs)
    db = np.sign(b[ii] - b[jj])
    da = np.sign(a[ii] - a[jj])
    untied = db != 0
    q["n_discordant_pairs"] = float(int(np.sum(untied & (db != da))))
    q["n_pairs_untied_before"] = float(int(untied.sum()))
    sr_med = reach.ranking_reach_detail(sb_id, sb_ood, q["delta_med"], "additive", n_pairs, PAIR_SEED)
    sr_max = reach.ranking_reach_detail(sb_id, sb_ood, q["delta_max"], "additive", n_pairs, PAIR_SEED)
    q["S_med"], q["S_max"] = sr_med["S"], sr_max["S"]
    fd = reach.realized_reordering_detail(sb_id, sb_ood, sa_id, sa_ood, n_pairs, PAIR_SEED)
    q["F"], q["F_flip"], q["F_tie"], q["F_se"] = fd["F"], fd["F_flip"], fd["F_tie"], fd["se"]
    # |dAUROC| <= F <= S(delta_max). The first is a population statement checked against a sampled F, so it is
    # given 4 se plus the rule-of-three resolution of a pair sample (a sample with zero reordered pairs only
    # bounds the true F by 3 / n_pairs); that same floor is added to F wherever F feeds a Check 1 reading.
    res = 3.0 / n_pairs
    q["F_resolution"] = res
    q["check_dauroc_le_F"] = float(abs(q["delta_auroc"]) <= q["F"] + 4 * fd["se"] + res + TOL)
    q["slack_F_minus_dauroc"] = q["F"] - abs(q["delta_auroc"])
    q["check_F_le_Smax"] = float(q["F"] <= q["S_max"] + TOL)
    q["slack_Smax_minus_F"] = q["S_max"] - q["F"]
    q["S_med_over_Smax"] = q["S_med"] / q["S_max"] if q["S_max"] > 0 else float("nan")
    # ratio-margin variant
    rs = ratio_stats(b, a)
    q["ratio_valid_frac"] = rs["valid_frac"]
    rho = spearmanr(diff, b).statistic if np.ptp(b) > 0 and diff.std() > 0 else float("nan")
    q["delta_level_spearman"] = float(rho)
    applicable = rs["valid_frac"] >= RATIO_VALID_MIN
    q["ratio_applicable"] = float(applicable)
    q["ratio_preferred"] = float(applicable and np.isfinite(rho) and rho >= RATIO_PREFERRED_SPEARMAN)
    if applicable:
        q["delta_ratio_med"], q["delta_ratio_max"] = rs["med"], rs["max"]
        for tag, p in (("", 2.0), ("_p1", 1.0)):
            q["S_ratio_med" + tag] = reach.ranking_reach_detail(sb_id, sb_ood, rs["med"], "ratio", n_pairs, PAIR_SEED, ratio_power=p)["S"]
            q["S_ratio_max" + tag] = reach.ranking_reach_detail(sb_id, sb_ood, rs["max"], "ratio", n_pairs, PAIR_SEED, ratio_power=p)["S"]
        if rs["valid_frac"] == 1.0:
            q["check_F_le_Sratio_max"] = float(q["F"] <= q["S_ratio_max"] + TOL)
    # Check 1 at the registered thresholds, on S(delta_med) and on S(delta_max)
    for basis, S in (("med", q["S_med"]), ("max", q["S_max"])):
        r = reach.check1_reading(q["delta_auroc"], S, q["F"] + res, tau=THRESHOLDS["tau"], S_high=THRESHOLDS["S_high"],
                                 F0=THRESHOLDS["F0"], n_pairs=n_pairs)
        q[f"check1_code_S{basis}"] = float(CHECK1_CODE[r["label"]])
    q["check1_flat"] = float(abs(q["delta_auroc"]) < THRESHOLDS["tau"])
    return q


def bounds_rows(op, dose, seed, Ztr, ytr, E, meta, base, after, n_id, oods_slices, rng_idx, n_pairs, ids):
    """Theorem 2 (maha) and Lemma 2 (cosine) for one affine configuration; one wide row each."""
    A, mu, b = meta["A"], meta["mu"], meta["b"]
    Sigma, _, means_pre = bounds.pooled_within_class_cov(Ztr, ytr)
    # Theorem 2 on a seeded subsample of the evaluation set (K x n x d residuals are held in memory)
    sub = rng_idx
    t2 = bounds.theorem2_evaluate(A, Sigma, E[sub].astype(np.float64), means_pre, eps=scorers.REG_EPS)
    sc, mo = t2["single_class"], t2["min_over_classes"]
    actual = np.abs(after["scores"]["maha"][sub] - base["scores"]["maha"][sub])
    refit_vs_qB = np.abs(after["scores"]["maha"][sub] - t2["score_after"])
    rows = [{
        **ids, "operator": op, "dose": dose, "seed": seed, "bound": "theorem2_maha", "n_eval": len(sub),
        "eps": scorers.REG_EPS, "sigma_max": meta["sigma_max"], "sigma_min": meta["sigma_min"], "cond": meta["cond"],
        "I_minus_AAT_2": meta["I_minus_AAT_2"], "I_minus_A_2": meta["I_minus_A_2"],
        "lambda_min_C": sc["lambda_min_C"], "lambda_min_C_raw": sc["lambda_min_C_raw"],
        "Sigma_trace": float(np.trace(Sigma)), "n_classes": len(means_pre), "n_train": len(Ztr), "dim": Ztr.shape[1],
        "single_class_observed_max": sc["observed_max"], "single_class_predicted_max": sc["predicted_max"],
        "single_class_tightness_max": sc["tightness_max"], "single_class_tightness_median": sc["tightness_median"],
        "single_class_holds": sc["bound_holds"],
        "predicted_max": mo["predicted_max"], "observed_max": mo["observed_max"],
        "tightness_max": mo["tightness_max"], "tightness_median": mo["tightness_median"], "bound_holds": mo["bound_holds"],
        "actual_refit_observed_max": float(actual.max()), "actual_refit_observed_median": float(np.median(actual)),
        "actual_within_predicted": bool(actual.max() <= mo["predicted_max"] * (1 + 1e-9) + 1e-12),
        "refit_vs_qB_max_abs": float(refit_vs_qB.max()),
        "base_vs_qD_max_rel": float((np.abs(base["scores"]["maha"][sub] - t2["score_before"])
                                     / np.maximum(np.abs(t2["score_before"]), 1e-12)).max()),
        "refit_vs_qB_max_rel": float((refit_vs_qB / np.maximum(np.abs(t2["score_after"]), 1e-12)).max()),
    }]
    s_b, s_a = base["scores"]["cosine"], after["scores"]["cosine"]
    l2 = bounds.lemma2_cosine_bound(E.astype(np.float64), base["means"], mu, A, b, s_before=s_b, s_after=s_a)
    rows.append({
        **ids, "operator": op, "dose": dose, "seed": seed, "bound": "lemma2_cosine", "n_eval": len(E),
        "sigma_max": meta["sigma_max"], "sigma_min": meta["sigma_min"], "cond": meta["cond"],
        "I_minus_AAT_2": meta["I_minus_AAT_2"], "I_minus_A_2": meta["I_minus_A_2"],
        "b_norm": 0.0 if b is None else float(np.linalg.norm(b)),
        "r_v_max": l2["r_v_max"], "r_mu_max": l2["r_mu_max"],
        "n_zero_norm_vectors": l2["n_zero_norm_vectors"], "n_zero_norm_centroids": l2["n_zero_norm_centroids"],
        "predicted_max": l2["predicted_bound"], "observed_max": l2["observed_max"], "observed_median": l2["observed_median"],
        "tightness_max": l2["tightness_max"], "tightness_median": l2["tightness_median"], "bound_holds": l2["bound_holds"],
        "lemma1_kappa": l2["lemma1"]["kappa"], "lemma1_eta": l2["lemma1"]["eta"], "lemma1_bound": l2["lemma1_bound"],
        "lemma1_bound_holds": l2["lemma1_bound_holds"],
    })
    return rows


def meta_rows(meta, op):
    keys = ("sigma_max", "sigma_min", "cond", "I_minus_AAT_2", "I_minus_A_2", "scale", "gamma", "alpha", "n_floored",
            "floor_applied", "lam_max", "target_norm", "n_zero_norm", "jacobian_dev_bound", "jacobian_sigma_min_lower",
            "residual_norm_median", "residual_norm_max", "delta_norm_median", "delta_norm_max",
            "displacement_ratio_median", "feature_rms", "W_spectral_norm")
    return {k: float(meta[k]) for k in keys if k in meta and meta[k] is not None and np.isscalar(meta[k])}


def run_config(op, dose, seed, Ztr, ytr, E, n_id, slices, base, base_metrics, n_pairs, ids, adapter_state, bounds_idx):
    t0 = time.perf_counter()
    Xtr, XE, meta = transform(op, dose, seed, Ztr, E, adapter_state)
    # sanity: the operator must have left shapes alone and produced finite numbers
    assert Xtr.shape == Ztr.shape and XE.shape == E.shape and np.isfinite(Xtr).all() and np.isfinite(XE).all(), op
    after = base if op == "identity" else scorers.score_all(Xtr, ytr, XE)      # refit ALL scorers on the transformed train set
    rows, c1 = [], []
    key = dict(operator=op, dose=dose, seed=seed)

    def add(ood, scorer, quantity, value):
        rows.append((ids["backbone"], ids["id_set"], ood, op, dose, scorer, seed, quantity, value))

    for k, v in meta_rows(meta, op).items():
        add("ALL", "ALL", k, v)
    for ood, sl in slices.items():
        for sc in scorers.SCORER_NAMES:
            s_b, s_a = base["scores"][sc], after["scores"][sc]
            if op == "identity":
                is_ood = np.r_[np.zeros(n_id, bool), np.ones(sl.stop - sl.start, bool)]
                idx = np.r_[0:n_id, sl.start:sl.stop]
                base_metrics[(ood, sc)] = (scorers.fast_auroc(s_b[idx], is_ood), scorers.fpr_at_95_tpr(s_b[idx], is_ood))
                add(ood, sc, "auroc", base_metrics[(ood, sc)][0])
                add(ood, sc, "fpr95", base_metrics[(ood, sc)][1])
                continue
            b_auc, b_fpr = base_metrics[(ood, sc)]
            q = evaluate_population(s_b, s_a, n_id, sl, n_pairs, b_auc, b_fpr)
            q["auroc_baseline"], q["fpr95_baseline"] = b_auc, b_fpr
            for k, v in q.items():
                add(ood, sc, k, v)
            F_read = q["F"] + q["F_resolution"]
            for (tau, sh), lab in reading_rows(q["delta_auroc"], q["S_med"], F_read, n_pairs).items():
                c1.append({**ids, "ood_set": ood, **key, "scorer": sc, "S_basis": "med", "tau": tau, "S_high": sh,
                           "F0": THRESHOLDS["F0"], "label": lab})
            for (tau, sh), lab in reading_rows(q["delta_auroc"], q["S_max"], F_read, n_pairs).items():
                c1.append({**ids, "ood_set": ood, **key, "scorer": sc, "S_basis": "max", "tau": tau, "S_high": sh,
                           "F0": THRESHOLDS["F0"], "label": lab})
    brows = []
    if op != "identity" and meta.get("linear") and meta.get("A") is not None:
        brows = bounds_rows(op, dose, seed, Ztr, ytr, E, meta, base, after, n_id, slices, bounds_idx, n_pairs, ids)
        for r in brows:
            tag = r["bound"]
            add("ALL", "maha" if tag.startswith("theorem2") else "cosine", f"{tag}_bound_holds", float(r["bound_holds"]))
            add("ALL", "maha" if tag.startswith("theorem2") else "cosine", f"{tag}_tightness_max", r["tightness_max"])
    add("ALL", "ALL", "config_seconds", time.perf_counter() - t0)
    add("ALL", "ALL", "config_complete", 1.0)         # last row: a killed write leaves no marker and the config is redone
    cols = ["backbone", "id_set", "ood_set", "operator", "dose", "scorer", "seed", "quantity", "value"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(c1), pd.DataFrame(brows)


# ----------------------------------------------------------------------------- checks 2 and 3
def write_check2(path, n_backbones=5, n_ood=5, n_scorers=len(scorers.SCORER_NAMES)):
    fam = n_backbones * n_ood * n_scorers
    designs = {}
    for k in (3, 5, 10):
        for label, m in (("per_backbone_ood", n_scorers), ("per_backbone", n_ood * n_scorers), ("whole_audit", fam), ("scorers_only", 1)):
            designs[f"k{k}_m{m}_{label}"] = checks.check2_feasibility(k, m, alpha=0.05, n_sample=checks.K10_N_SAMPLE if k == 10 else None)
    out = {"alpha": 0.05, "registered_k_seeds_adapter": len(ADAPTER_SEEDS), "registered_family_size_whole_audit": fam,
           "family_definition": f"{n_backbones} backbones x {n_ood} OOD rungs x {n_scorers} scorers (CIFAR-10 ID)",
           "primary": checks.check2_feasibility(len(ADAPTER_SEEDS), fam, 0.05, n_sample=checks.K10_N_SAMPLE),
           "designs": designs}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return out


def write_check3(backbone, id_set, Ztr, oods, out_path, adapter_root=ADAPTER_ROOT, emb_root=EMB_ROOT):
    d = Path(adapter_root) / backbone / id_set
    files = sorted(d.glob("adapter_seed*.pt"))
    if not files:
        print(f"[audit] check3: no adapters for {backbone}/{id_set}")
        return
    mu_id = Ztr.astype(np.float64).mean(0)
    pops = dict(oods)
    ref = Path(emb_root) / backbone / id_set / "ood_ref.npz"
    if ref.is_file():
        pops["ref"] = np.load(ref)["embeddings"]
    rows = []
    for f in files:
        ck = torch.load(f, map_location="cpu")
        seed = ck["meta"]["seed"]
        for kind in ("w_construction", "w_learned"):
            w = ck[kind].numpy()
            for name, Z in pops.items():
                r = checks.check3_construction_target(mu_id, Z.astype(np.float64).mean(0), w)
                rows.append({"backbone": backbone, "id_set": id_set, "ood_set": name, "adapter_seed": seed, "w_kind": kind, **r})
    df = pd.DataFrame(rows)
    if out_path.is_file():
        old = pd.read_csv(out_path)
        df = pd.concat([old[~((old.backbone == backbone) & (old.id_set == id_set))], df], ignore_index=True)
    df.to_csv(out_path, index=False, float_format="%.10g")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--id-set", default="cifar10")
    ap.add_argument("--quick", action="store_true", help="subset of operators/doses/seeds for a smoke test")
    ap.add_argument("--n-pairs", type=int, default=reach.N_PAIRS)
    ap.add_argument("--operators", nargs="*", default=None,
                    help="restrict to these operator families (identity is always run)")
    ap.add_argument("--max-ood", type=int, default=None, help="subsample every evaluation set to this many (smoke runs)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--emb-root", type=Path, default=EMB_ROOT)
    ap.add_argument("--adapter-root", type=Path, default=ADAPTER_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--redo", action="store_true", help="ignore completion markers")
    a = ap.parse_args(argv)

    ids = {"backbone": a.backbone, "id_set": a.id_set}
    try:
        Ztr, ytr, Zte, oods = load_embeddings(a.backbone, a.id_set, a.emb_root, a.max_ood, a.max_train)
    except FileNotFoundError as e:
        print(f"[audit] {a.backbone}/{a.id_set}: embeddings missing ({e}); nothing to do")
        return 0
    if not oods:
        print(f"[audit] {a.backbone}/{a.id_set}: no OOD embeddings found; nothing to do")
        return 0
    n_id = len(Zte)
    E = np.concatenate([Zte] + list(oods.values()))
    slices, o = {}, n_id
    for name, Z in oods.items():
        slices[name] = slice(o, o + len(Z))
        o += len(Z)
    print(f"[audit] {a.backbone}/{a.id_set}: train {Ztr.shape}, ID test {n_id}, OOD sets "
          + ", ".join(f"{k}={len(v)}" for k, v in oods.items()), flush=True)

    out = a.out
    write_check2(out / CHECK2_JSON, n_ood=len(oods))
    write_check3(a.backbone, a.id_set, Ztr, oods, out / CHECK3_CSV, a.adapter_root, a.emb_root)

    adir = a.adapter_root / a.backbone / a.id_set
    adapters = {int(p.stem.replace("adapter_seed", "")): p for p in sorted(adir.glob("adapter_seed*.pt"))} if adir.is_dir() else {}
    cfgs = make_config_list(a.quick, a.operators, adapter_available=bool(adapters))
    if not adapters:
        print(f"[audit] no adapters in {adir}: adapter operator skipped (run train_adapters.py)")
    cfgs = [c for c in cfgs if c[0] != "adapter" or c[2] in adapters]
    done = set() if a.redo else completed_keys(out / AUDIT_CSV, a.backbone, a.id_set)
    rng_b = np.random.default_rng(BOUNDS_SEED)
    bounds_idx = np.sort(np.concatenate([rng_b.choice(np.arange(sl.start, sl.stop), min(BOUNDS_N_PER_POP, sl.stop - sl.start), replace=False)
                                         for sl in [slice(0, n_id)] + list(slices.values())]))

    t_start = time.perf_counter()
    base_metrics = {}
    # the identity baseline is always (re)computed: every dAUROC and F is against it
    base = scorers.score_all(Ztr, ytr, E)
    df, _, _ = run_config("identity", 0.0, 0, Ztr, ytr, E, n_id, slices, base, base_metrics, a.n_pairs, ids, None, bounds_idx)
    if ("identity", 0.0, 0) not in done:
        _append(out / AUDIT_CSV, df)
    todo = [c for c in cfgs[1:] if (c[0], round(c[1], 8), c[2]) not in done]
    print(f"[audit] {len(cfgs) - 1} configurations, {len(todo)} to run", flush=True)
    for i, (op, dose, seed) in enumerate(todo):
        state = torch.load(adapters[seed], map_location="cpu") if op == "adapter" else None
        df, c1, br = run_config(op, dose, seed, Ztr, ytr, E, n_id, slices, base, base_metrics, a.n_pairs, ids, state, bounds_idx)
        if len(br):
            _append(out / BOUNDS_CSV, br)
        if len(c1):
            _append(out / CHECK1_CSV, c1)
        _append(out / AUDIT_CSV, df)               # completion marker is the last row of this write
        sec = df[df.quantity == "config_seconds"].value.iloc[0]
        d_auc = df[(df.quantity == "delta_auroc")].groupby("scorer").value.apply(lambda v: float(np.max(np.abs(v)))).to_dict()
        print(f"[audit] {i + 1}/{len(todo)} {op} dose={dose} seed={seed}  {sec:.0f}s  max|dAUROC| per scorer: "
              + " ".join(f"{k}={v:.4f}" for k, v in d_auc.items()), flush=True)
    print(f"[audit] done {a.backbone}/{a.id_set} in {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
