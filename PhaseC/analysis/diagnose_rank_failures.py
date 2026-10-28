"""Attribute γ<0 rank failures: ridge discrepancy (A) vs float32 storage (B).

Runs only on the failing configurations named in Follow-up 08:
  backbones {resnet18_cifar_openood, clip_vitb16}
  spectral_temper γ ∈ {-0.5, -0.25}
  scorers {maha, rmd}

Two experiments, same 2×2×6 OOD = 24 configs × 2 scorers = 48 readings each:

  float64_storage  keep T(Z) in float64 (drop the astype float32); eps = REG_EPS
  eps_sweep        store float32 as usual; eps ∈ {1e-7, 1e-5, 1e-3}

(A) ridge: refit uses A C Aᵀ + eps I instead of A C Aᵀ + eps A Aᵀ; grows with eps.
(B) float32: rounding of the stored cloud, amplified by cond(C+eps I); falls as eps rises.

Also: maha_noridge (eps=0, pinv) over clean affine on all five backbones — Theorem 1's estimator.

    python3 PhaseC/analysis/diagnose_rank_failures.py
    python3 PhaseC/analysis/diagnose_rank_failures.py --skip-noridge
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import operators as ops  # noqa: E402
import reach  # noqa: E402
import run_audit as audit  # noqa: E402
import scorers  # noqa: E402
from train_adapters import EMB_ROOT  # noqa: E402

PHASEC = HERE.parent
OUT = PHASEC / "outputs"
PAIR_SEED = audit.PAIR_SEED
N_PAIRS = reach.N_PAIRS

FAIL_BBS = ("resnet18_cifar_openood", "clip_vitb16")
FAIL_GAMMAS = (-0.5, -0.25)
FAIL_SCORERS = ("maha", "rmd")
EPS_SWEEP = (1e-7, 1e-5, 1e-3)
CLEAN_AFFINE = ("ortho_linear", "ortho_scale", "general_affine", "zca")
NORIDGE_BBS = (
    "clip_vitb16", "dinov2_vitb14", "resnet18_cifar_openood",
    "resnet50_imagenet", "vitb16_imagenet",
)


def rank_stats(b, a, n_pairs=N_PAIRS, pair_seed=PAIR_SEED):
    b = np.asarray(b, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    degenerate = bool(np.ptp(b) == 0 and np.ptp(a) == 0)
    one_sided = bool((np.ptp(b) == 0) != (np.ptp(a) == 0))
    if degenerate:
        rho = 1.0
    elif one_sided:
        rho = float("nan")
    else:
        rho = float(spearmanr(b, a).statistic)
    rng = np.random.default_rng(pair_seed)
    ii = rng.integers(0, len(b), n_pairs)
    jj = rng.integers(0, len(b), n_pairs)
    db = np.sign(b[ii] - b[jj])
    da = np.sign(a[ii] - a[jj])
    untied = db != 0
    # Counts pairs that were untied before and whose sign changed after — includes
    # before-untied → after-tied as well as flips. "Discordant" usually means flips only;
    # n_flipped restricts to da != 0.
    n_changed = int(np.sum(untied & (db != da)))
    n_flipped = int(np.sum(untied & (da != 0) & (db != da)))
    return {
        "rank_spearman": rho,
        "rank_one_minus_rho": float(1.0 - rho) if np.isfinite(rho) else float("nan"),
        "n_discordant_pairs": n_changed,   # legacy name: untied & sign-changed (incl. →tie)
        "n_flipped_pairs": n_flipped,
        "n_pairs_untied_before": int(untied.sum()),
        "n_eval": int(len(b)),
        "degenerate_constant": int(degenerate),
    }


def apply_affine(Z, meta, dtype=np.float32):
    """T(z) = A(z-mu)+mu+b in float64, cast to dtype (float32 = production; float64 = experiment B off)."""
    out = (np.asarray(Z, np.float64) - meta["mu"]) @ meta["A"].T + meta["mu"]
    if meta.get("b") is not None:
        out = out + meta["b"]
    return out.astype(dtype, copy=False)


def fit_cov(Ztr, ytr, eps):
    """Pooled cov with ridge eps. eps=0 → raw scatter/(N-K), no ridge."""
    Z, y = scorers._f64(Ztr), np.asarray(ytr)
    keep = y >= 0
    Z, y = Z[keep], y[keep]
    classes = np.unique(y)
    K, N, d = len(classes), len(Z), Z.shape[1]
    means = np.stack([Z[y == c].mean(0) for c in classes])
    scatter = np.zeros((d, d))
    for c, m in zip(classes, means):
        Xc = Z[y == c] - m
        scatter += Xc.T @ Xc
    cov = scatter / (N - K)
    if eps > 0:
        cov = cov + eps * np.eye(d)
    return classes, means, cov


#: Near-zero ridge for the Theorem 1 estimator. eps=0 + pinvh is correct but
#: O(n d²) per class via dense Prec; 1e-12 Cholesky is the practical stand-in
#: the brief allows and is what we report as maha_noridge.
NORIDGE_EPS = 1e-12


def maha_scores(Zev, means, cov, eps):
    """min_c Mahalanobis² via Cholesky of cov (already ridged by the caller)."""
    X = scorers._f64(Zev)
    L = scorers._chol_lower(cov)
    d2 = np.stack([scorers._sq_maha(X, m, L) for m in means], axis=1)
    return d2.min(axis=1)


def rmd_scores(Ztr, ytr, Zev, eps):
    """Relative Mahalanobis with the same ridge on class and background covs."""
    classes, means, cov = fit_cov(Ztr, ytr, eps)
    maha = maha_scores(Zev, means, cov, eps)
    Z, y = scorers._f64(Ztr), np.asarray(ytr)
    keep = y >= 0
    Z = Z[keep]
    bg_mean = Z.mean(0)
    bg_cov = np.cov(Z.T, ddof=1) + eps * np.eye(Z.shape[1])
    L = scorers._chol_lower(bg_cov)
    bg = scorers._sq_maha(scorers._f64(Zev), bg_mean, L)
    return maha - bg


def score_pair(Ztr, ytr, Zev, names, eps):
    out = {}
    if "maha" in names:
        _, means, cov = fit_cov(Ztr, ytr, eps)
        out["maha"] = maha_scores(Zev, means, cov, eps)
    if "rmd" in names:
        out["rmd"] = rmd_scores(Ztr, ytr, Zev, eps)
    return out


def temper_transform(Ztr, E, gamma, store_dtype):
    Xtr, meta = ops.spectral_temper(Ztr, Ztr, gamma)  # fit on Ztr; returns float32 by default
    # Re-apply at the requested storage dtype (production path uses float32).
    Xtr = apply_affine(Ztr, meta, dtype=store_dtype)
    XE = apply_affine(E, meta, dtype=store_dtype)
    return Xtr, XE, meta


def eval_population(s_b, s_a, n_id, slices):
    rows = []
    for ood, sl in slices.items():
        idx = np.r_[0:n_id, sl.start:sl.stop]
        st = rank_stats(s_b[idx], s_a[idx])
        rows.append({"ood_set": ood, **st})
    return rows


def run_fail_diagnostics(out_dir: Path):
    rows = []
    for bb in FAIL_BBS:
        print(f"[diag] loading {bb}/cifar10", flush=True)
        Ztr, ytr, Zte, oods = audit.load_embeddings(bb, "cifar10", EMB_ROOT)
        n_id = len(Zte)
        E = np.concatenate([Zte] + list(oods.values()))
        slices, o = {}, n_id
        for name, Z in oods.items():
            slices[name] = slice(o, o + len(Z))
            o += len(Z)

        # identity baseline at each eps (and float64 identity = same cloud)
        baselines = {}
        for eps in EPS_SWEEP:
            baselines[("f32", eps)] = score_pair(Ztr, ytr, E, FAIL_SCORERS, eps)
        baselines[("f64", scorers.REG_EPS)] = score_pair(
            Ztr.astype(np.float64), ytr, E.astype(np.float64), FAIL_SCORERS, scorers.REG_EPS)

        for gamma in FAIL_GAMMAS:
            # --- float32 storage + eps sweep (mechanism A vs B via eps) ---
            Xtr32, XE32, meta = temper_transform(Ztr, E, gamma, np.float32)
            for eps in EPS_SWEEP:
                t0 = time.perf_counter()
                after = score_pair(Xtr32, ytr, XE32, FAIL_SCORERS, eps)
                base = baselines[("f32", eps)]
                for sc in FAIL_SCORERS:
                    for r in eval_population(base[sc], after[sc], n_id, slices):
                        rows.append({
                            "experiment": "eps_sweep", "backbone": bb, "operator": "spectral_temper",
                            "dose": gamma, "seed": 0, "scorer": sc, "store_dtype": "float32",
                            "eps": eps, "sigma_min": meta["sigma_min"], "sigma_max": meta["sigma_max"],
                            "I_minus_AAT_2": meta["I_minus_AAT_2"], "cond": meta["cond"],
                            "seconds": time.perf_counter() - t0, **r,
                        })
                print(f"[diag] {bb} γ={gamma} f32 eps={eps:g}  {time.perf_counter()-t0:.1f}s", flush=True)

            # --- float64 storage, eps = REG_EPS (mechanism B off) ---
            Xtr64, XE64, meta64 = temper_transform(Ztr, E, gamma, np.float64)
            t0 = time.perf_counter()
            after64 = score_pair(Xtr64, ytr, XE64, FAIL_SCORERS, scorers.REG_EPS)
            base64 = baselines[("f64", scorers.REG_EPS)]
            for sc in FAIL_SCORERS:
                for r in eval_population(base64[sc], after64[sc], n_id, slices):
                    rows.append({
                        "experiment": "float64_storage", "backbone": bb, "operator": "spectral_temper",
                        "dose": gamma, "seed": 0, "scorer": sc, "store_dtype": "float64",
                        "eps": scorers.REG_EPS, "sigma_min": meta64["sigma_min"],
                        "sigma_max": meta64["sigma_max"], "I_minus_AAT_2": meta64["I_minus_AAT_2"],
                        "cond": meta64["cond"], "seconds": time.perf_counter() - t0, **r,
                    })
            print(f"[diag] {bb} γ={gamma} f64 eps={scorers.REG_EPS:g}  {time.perf_counter()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    path = out_dir / "rank_failure_attribution.csv"
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"[diag] wrote {path} ({len(df)} rows)", flush=True)
    return df


def run_maha_noridge(out_dir: Path):
    """Theorem 1 estimator: eps=0 pinv, clean affine only, all five backbones."""
    rows = []
    for bb in NORIDGE_BBS:
        print(f"[noridge] loading {bb}/cifar10", flush=True)
        Ztr, ytr, Zte, oods = audit.load_embeddings(bb, "cifar10", EMB_ROOT)
        n_id = len(Zte)
        E = np.concatenate([Zte] + list(oods.values()))
        slices, o = {}, n_id
        for name, Z in oods.items():
            slices[name] = slice(o, o + len(Z))
            o += len(Z)

        # baselines: ridged (audit) and noridge (eps=1e-12)
        base_ridge = score_pair(Ztr, ytr, E, ("maha",), scorers.REG_EPS)["maha"]
        base_noridge = score_pair(Ztr, ytr, E, ("maha",), NORIDGE_EPS)["maha"]
        print(f"[noridge] {bb} baselines ready", flush=True)

        cfgs = audit.make_config_list(False, list(CLEAN_AFFINE), adapter_available=False)
        cfgs = [c for c in cfgs if c[0] != "identity" and not (c[0] in ("zca",) and c[1] == 0.0)]
        for op, dose, seed in cfgs:
            t0 = time.perf_counter()
            Xtr, XE, meta = audit.transform(op, dose, seed, Ztr, E, None)
            after_ridge = score_pair(Xtr, ytr, XE, ("maha",), scorers.REG_EPS)["maha"]
            after_noridge = score_pair(Xtr, ytr, XE, ("maha",), NORIDGE_EPS)["maha"]
            sec = time.perf_counter() - t0
            for tag, sb, sa, eps in (
                ("maha", base_ridge, after_ridge, scorers.REG_EPS),
                ("maha_noridge", base_noridge, after_noridge, NORIDGE_EPS),
            ):
                for r in eval_population(sb, sa, n_id, slices):
                    rows.append({
                        "backbone": bb, "id_set": "cifar10", "operator": op, "dose": dose,
                        "seed": seed, "scorer": tag, "eps": eps, "seconds": sec, **r,
                    })
            def _worst(sb, sa):
                vals = []
                for sl in slices.values():
                    idx = np.r_[0:n_id, sl.start:sl.stop]
                    vals.append(1.0 - float(spearmanr(sb[idx], sa[idx]).statistic))
                return max(vals)
            print(f"[noridge] {bb} {op} dose={dose} seed={seed}  {sec:.1f}s  "
                  f"ridge max 1-ρ={_worst(base_ridge, after_ridge):.3e}  "
                  f"noridge max 1-ρ={_worst(base_noridge, after_noridge):.3e}",
                  flush=True)

    df = pd.DataFrame(rows)
    path = out_dir / "maha_noridge_clean_affine.csv"
    df.to_csv(path, index=False, float_format="%.10g")
    print(f"[noridge] wrote {path} ({len(df)} rows)", flush=True)
    return df


def summarise(attr: pd.DataFrame, noridge: pd.DataFrame | None, out_dir: Path):
    """Compact tables for FINDINGS; write JSON note."""
    note = {"attribution": {}, "noridge": {}}

    # Baseline production = float32 + eps=1e-5
    prod = attr[(attr.experiment == "eps_sweep") & (attr.eps == 1e-5)]
    f64 = attr[attr.experiment == "float64_storage"]
    # Compare max 1-ρ over ood×scorer for each (bb, gamma)
    def worst(df, keys):
        return (df.groupby(keys).rank_one_minus_rho.max()
                .reset_index().rename(columns={"rank_one_minus_rho": "worst_1_minus_rho"}))

    w_prod = worst(prod, ["backbone", "dose", "scorer"])
    w_f64 = worst(f64, ["backbone", "dose", "scorer"])
    cmp64 = w_prod.merge(w_f64, on=["backbone", "dose", "scorer"], suffixes=("_f32", "_f64"))
    note["attribution"]["float64_vs_float32"] = cmp64.to_dict(orient="records")

    # eps sweep: worst 1-ρ vs eps
    sweep = (attr[attr.experiment == "eps_sweep"]
             .groupby(["backbone", "dose", "scorer", "eps"]).rank_one_minus_rho.max()
             .reset_index())
    note["attribution"]["eps_sweep_worst"] = sweep.to_dict(orient="records")

    # Verdict: does 1-ρ rise with eps (A) or fall (B)? Does f64 collapse it (B)?
    verdict_lines = []
    for (bb, dose, sc), g in sweep.groupby(["backbone", "dose", "scorer"]):
        g = g.sort_values("eps")
        rhos = g.rank_one_minus_rho.values
        eps_trend = "rises_with_eps" if rhos[-1] > rhos[0] * 1.5 else (
            "falls_with_eps" if rhos[-1] < rhos[0] / 1.5 else "flat")
        row64 = cmp64[(cmp64.backbone == bb) & (cmp64.dose == dose) & (cmp64.scorer == sc)]
        if len(row64):
            r32 = float(row64.worst_1_minus_rho_f32.iloc[0])
            r64 = float(row64.worst_1_minus_rho_f64.iloc[0])
            f64_ratio = r64 / r32 if r32 > 0 else float("nan")
        else:
            r32 = r64 = f64_ratio = float("nan")
        verdict_lines.append({
            "backbone": bb, "dose": dose, "scorer": sc,
            "eps_trend": eps_trend, "worst_1m_eps1e7": float(rhos[0]),
            "worst_1m_eps1e5": float(rhos[1]), "worst_1m_eps1e3": float(rhos[2]),
            "worst_1m_f32": r32, "worst_1m_f64": r64, "f64_over_f32": f64_ratio,
        })
    note["attribution"]["per_config_verdict"] = verdict_lines

    # Global: median f64/f32 and Spearman of 1-ρ with eps
    if len(cmp64):
        note["attribution"]["median_f64_over_f32"] = float(
            (cmp64.worst_1_minus_rho_f64 / cmp64.worst_1_minus_rho_f32.replace(0, np.nan)).median())
    # Across sweep rows, correlate log(eps) with 1-ρ
    from scipy.stats import spearmanr as _sp
    if len(sweep) >= 3:
        rho_eps, _ = _sp(np.log10(sweep.eps), sweep.rank_one_minus_rho)
        note["attribution"]["spearman_logeps_vs_1minus_rho"] = float(rho_eps)

    # Decide
    med_ratio = note["attribution"].get("median_f64_over_f32", float("nan"))
    sp_eps = note["attribution"].get("spearman_logeps_vs_1minus_rho", float("nan"))
    if np.isfinite(med_ratio) and med_ratio < 0.2 and (not np.isfinite(sp_eps) or sp_eps <= 0):
        winner = "B_float32_rounding"
    elif np.isfinite(sp_eps) and sp_eps > 0.3 and (not np.isfinite(med_ratio) or med_ratio > 0.5):
        winner = "A_ridge_discrepancy"
    elif np.isfinite(med_ratio) and med_ratio < 0.2:
        winner = "B_float32_rounding"
    elif np.isfinite(sp_eps) and sp_eps > 0.3:
        winner = "A_ridge_discrepancy"
    else:
        winner = "mixed_or_inconclusive"
    note["attribution"]["winner"] = winner
    note["attribution"]["rule"] = (
        "A if 1−ρ rises with eps (Spearman logeps>0.3) and float64 storage leaves ≥50% of the residual; "
        "B if float64 storage collapses 1−ρ to <20% of float32 and eps trend is flat/falling."
    )

    if noridge is not None and len(noridge):
        g = (noridge.groupby(["scorer", "operator"])
             .agg(n=("rank_one_minus_rho", "size"),
                  max_1m=("rank_one_minus_rho", "max"),
                  median_1m=("rank_one_minus_rho", "median"),
                  max_discordant=("n_discordant_pairs", "max"),
                  median_discordant=("n_discordant_pairs", "median"),
                  n_degenerate=("degenerate_constant", "sum"))
             .reset_index())
        note["noridge"]["by_operator"] = g.to_dict(orient="records")
        for sc in ("maha", "maha_noridge"):
            s = noridge[noridge.scorer == sc]
            note["noridge"][sc] = {
                "n": int(len(s)),
                "max_1_minus_rho": float(s.rank_one_minus_rho.max()),
                "median_1_minus_rho": float(s.rank_one_minus_rho.median()),
                "max_n_discordant": int(s.n_discordant_pairs.max()),
                "median_n_discordant": float(s.n_discordant_pairs.median()),
                "frac_below_1e-3": float((s.rank_one_minus_rho > 1e-3).mean()),
                "n_degenerate": int(s.degenerate_constant.sum()),
            }

    path = out_dir / "notes" / "rank_failure_attribution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(note, indent=2) + "\n")
    print(f"[diag] wrote {path}", flush=True)
    print(f"[diag] WINNER={winner}  median_f64/f32={med_ratio}  spearman(logeps,1-ρ)={sp_eps}", flush=True)
    return note


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-noridge", action="store_true")
    ap.add_argument("--skip-attr", action="store_true")
    a = ap.parse_args(argv)
    attr = None
    noridge = None
    if not a.skip_attr:
        attr = run_fail_diagnostics(a.out)
    else:
        p = a.out / "rank_failure_attribution.csv"
        attr = pd.read_csv(p) if p.is_file() else None
    if not a.skip_noridge:
        noridge = run_maha_noridge(a.out)
    else:
        p = a.out / "maha_noridge_clean_affine.csv"
        noridge = pd.read_csv(p) if p.is_file() else None
    if attr is not None:
        summarise(attr, noridge, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
