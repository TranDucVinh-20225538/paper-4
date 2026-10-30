"""CIFAR-10 confirmatory subsample: hold K=10, cut n_c to 500 (N=5000).

Registered predictions (Follow-up 11), contradicting the mis-derived CIFAR-100 (iv):

  1. λ_min(C)/λ̄ falls on all five, by a bounded factor of order (1 − √(d/N))² —
     roughly 2× (d=512), 2.7× (d=768), 7.7× (d=2048). Marchenko–Pastur assumes
     iid draws; deep features are not — treat as order-of-magnitude.
  2. Regime assignment does not change: a factor 2–8× cannot cross a ten-order
     gap, so resnet18/clip/resnet50 stay computable and dinov2/vitb16 stay at
     the float32 floor.

Scope: λ_min/λ̄ + maha_noridge discordant counts on clean affine operators,
one seed set. No new extraction — subsample embeddings already on disk.
Not the 95-config grid.

    python -u PhaseC/analysis/subsample_nc500_regime.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bounds  # noqa: E402
from diagnose_rank_failures import (  # noqa: E402
    NORIDGE_EPS, fit_cov, maha_scores, rank_stats,
)
from run_audit import (  # noqa: E402
    OPERATOR_SEEDS, OUT, load_embeddings, transform,
)
from train_adapters import EMB_ROOT  # noqa: E402

ID_SET = "cifar10"
N_C_TARGET = 500
EPS2_F32 = float(np.finfo(np.float32).eps) ** 2
N_PAIRS = 2_000_000
PAIR_SEED = 20251027
CLEAN = ("ortho_linear", "ortho_scale", "general_affine", "zca")
BACKBONES = (
    "resnet18_cifar_openood", "resnet50_imagenet", "vitb16_imagenet",
    "clip_vitb16", "dinov2_vitb14",
)
# one seed per clean operator (tight confirmatory scope).
# zca is deterministic — seed list lives under OPERATOR_SEEDS["deterministic"].
SEEDS_ONE = {
    op: ((OPERATOR_SEEDS[op][0],) if op in OPERATOR_SEEDS
         else (OPERATOR_SEEDS["deterministic"][0],))
    for op in CLEAN
}


def subsample_train(Ztr, ytr, n_c, rng):
    idx = []
    for c in np.unique(ytr):
        ic = np.flatnonzero(ytr == c)
        take = min(n_c, len(ic))
        idx.append(rng.choice(ic, size=take, replace=False))
    idx = np.sort(np.concatenate(idx))
    return Ztr[idx], ytr[idx]


def lam_stats(Ztr, ytr):
    Sigma, _, _ = bounds.pooled_within_class_cov(Ztr, ytr)
    ev = np.linalg.eigvalsh(Sigma.astype(np.float64))
    lam_min = float(ev[0])
    lam_bar = float(np.trace(Sigma) / Sigma.shape[0])
    return {
        "lambda_min_C": lam_min,
        "lam_bar": lam_bar,
        "lam_min_over_lam_bar": lam_min / lam_bar,
        "dim": int(Sigma.shape[0]),
        "n_train": int(len(Ztr)),
        "n_classes": int(len(np.unique(ytr))),
        "n_c_mean": float(len(Ztr) / len(np.unique(ytr))),
    }


def mp_factor(d, N):
    """Order-of-magnitude Marchenko–Pastur edge ratio (1 − √(d/N))² for λ_min."""
    x = np.sqrt(d / N)
    return float((1.0 - x) ** 2) if x < 1 else float("nan")


def run_backbone(backbone, emb_root=EMB_ROOT):
    Ztr_full, ytr_full, Zte, oods = load_embeddings(backbone, ID_SET, emb_root)
    rng_sub = np.random.default_rng([2026, 11, hash(backbone) % 2**31])
    Ztr, ytr = subsample_train(Ztr_full, ytr_full, N_C_TARGET, rng_sub)
    n_id = len(Zte)
    E = np.concatenate([Zte] + list(oods.values()))

    full = lam_stats(Ztr_full, ytr_full)
    sub = lam_stats(Ztr, ytr)
    full["backbone"] = backbone
    full["setting"] = "full"
    sub["backbone"] = backbone
    sub["setting"] = f"n_c={N_C_TARGET}"
    sub["mp_factor_pred"] = mp_factor(sub["dim"], sub["n_train"])
    sub["ratio_fall_obs"] = full["lam_min_over_lam_bar"] / sub["lam_min_over_lam_bar"]
    sub["ratio_full"] = full["lam_min_over_lam_bar"]
    rows_lam = [full, {**sub, "mp_factor_pred": sub["mp_factor_pred"],
                       "ratio_fall_obs": sub["ratio_fall_obs"],
                       "ratio_full": sub["ratio_full"]}]

    # maha_noridge discordant on clean affine, subsampled train
    disc_rows = []
    _, means0, cov0 = fit_cov(Ztr, ytr, NORIDGE_EPS)
    base_scores = maha_scores(E, means0, cov0, NORIDGE_EPS)
    for op in CLEAN:
        dose = 10.0 if op == "general_affine" else 1.0
        for seed in SEEDS_ONE[op]:
            t0 = time.perf_counter()
            Xtr, XE, _meta = transform(op, dose, seed, Ztr, E, None)
            _, means1, cov1 = fit_cov(Xtr, ytr, NORIDGE_EPS)
            after = maha_scores(XE, means1, cov1, NORIDGE_EPS)
            st = rank_stats(base_scores, after, n_pairs=N_PAIRS, pair_seed=PAIR_SEED)
            disc_rows.append({
                "backbone": backbone, "operator": op, "dose": dose, "seed": seed,
                "n_discordant_pairs": int(st["n_discordant_pairs"]),
                "n_untied": int(st["n_pairs_untied_before"]),
                "rank_spearman": float(st["rank_spearman"]),
                "rank_one_minus_rho": float(st["rank_one_minus_rho"]),
                "seconds": time.perf_counter() - t0,
                "lam_min_over_lam_bar": sub["lam_min_over_lam_bar"],
                "regime": ("float32 floor" if sub["lam_min_over_lam_bar"] < EPS2_F32 * 100
                           else "computable"),
            })
            print(f"[nc500] {backbone} {op} seed={seed} disc={st['n_discordant_pairs']} "
                  f"1-ρ={st['rank_one_minus_rho']:.3e}  {time.perf_counter() - t0:.0f}s",
                  flush=True)
    return rows_lam, disc_rows


def write_note(lam_df, disc_df, path: Path):
    piv = lam_df.pivot_table(index="backbone", columns="setting",
                             values="lam_min_over_lam_bar")
    lines = [
        "# CIFAR-10 n_c=500 subsample — regime confirmatory",
        "",
        "Registered before the run (Follow-up 11). Hold K=10 on CIFAR-10; "
        f"subsample train to n_c={N_C_TARGET} (N=5·10³). This cuts the pooled "
        "estimator's sample size; CIFAR-100 did not (N unchanged).",
        "",
        "## Predictions",
        "",
        "1. λ_min/λ̄ falls on all five by ~(1−√(d/N))².",
        "2. Regime assignment unchanged (ten-order gap vs factor 2–8).",
        "",
        "## λ_min/λ̄",
        "",
    ]
    # rebuild a readable table
    sub = lam_df[lam_df.setting.str.startswith("n_c")].set_index("backbone")
    full = lam_df[lam_df.setting == "full"].set_index("backbone")
    lines += [
        "| backbone | ratio_full | ratio_n_c500 | fall_obs | MP_fall ≈1/(1−√(d/N))² | regime_full | regime_500 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for bb in sorted(full.index):
        rf = float(full.loc[bb, "lam_min_over_lam_bar"])
        rs = float(sub.loc[bb, "lam_min_over_lam_bar"])
        fall = rf / rs
        # (1−√(d/N))² is the MP edge multiplier on λ_min; expected fall factor is its reciprocal.
        mp_mult = mp_factor(int(sub.loc[bb, "dim"]), int(sub.loc[bb, "n_train"]))
        mp_fall = (1.0 / mp_mult) if mp_mult and mp_mult > 0 else float("nan")
        reg_f = "floor" if rf < EPS2_F32 * 100 else "computable"
        reg_s = "floor" if rs < EPS2_F32 * 100 else "computable"
        lines.append(
            f"| {bb} | {rf:.3e} | {rs:.3e} | {fall:.2f}× | {mp_fall:.2f}× | {reg_f} | {reg_s} |"
        )
    lines += ["", "## maha_noridge discordant (clean affine, one seed)", ""]
    g = (disc_df.groupby(["backbone", "operator"])
         .n_discordant_pairs.max().unstack(fill_value=-1))
    # Avoid pandas.to_markdown (needs optional tabulate).
    cols = ["backbone"] + list(g.columns)
    lines += ["| " + " | ".join(str(c) for c in cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
    for bb, row in g.iterrows():
        lines.append("| " + " | ".join([str(bb)] + [str(int(row[c])) for c in g.columns]) + " |")
    lines += ["", f"ε²_float32 = {EPS2_F32:.2e}.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    lam_rows, disc_rows = [], []
    for bb in BACKBONES:
        print(f"[nc500] === {bb} ===", flush=True)
        lr, dr = run_backbone(bb)
        lam_rows.extend(lr)
        disc_rows.extend(dr)
    lam_df = pd.DataFrame(lam_rows)
    disc_df = pd.DataFrame(disc_rows)
    lam_path = out / "nc500_lam_min_ratio.csv"
    disc_path = out / "nc500_maha_noridge_rank.csv"
    lam_df.to_csv(lam_path, index=False, float_format="%.10g")
    disc_df.to_csv(disc_path, index=False, float_format="%.10g")
    write_note(lam_df, disc_df, out / "notes" / "nc500_regime.md")
    print(f"[nc500] wrote {lam_path}")
    print(f"[nc500] wrote {disc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
