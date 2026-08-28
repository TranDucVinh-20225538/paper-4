"""
AR-02 / C3 -- mini gamma calibration, single seed, geometry-only.

Purpose: find the gamma value that pulls the STRONG checkpoint's (alpha=1.0) geometry down to match
the WEAK checkpoint's (alpha=0.25, gamma=0) NATURAL geometry -- so C3's matched-pair comparison can
hold nominal intervention strength (alpha, the preregistered independent variable, 22_Experimental_
Design.md SS5/SS6) fixed at 1.0 while geometry is pulled to the alpha=0.25 level, decoupling the two.

REVISED after a first attempt (calibrating on the alpha=0.25 checkpoint, targeting alpha=1.0's
natural slope) moved every tested gamma AWAY from target, not toward it: gamma (as validated,
gamma>=0 in AR-02's own manipulation check) only ever DECREASES spectral-decay slope. Natural slope
INCREASES with alpha (alpha=1.0's slope=0.006807 > alpha=0.25's slope=0.006751 for every seed,
frozen AlphaLadder data) -- so reaching alpha=1.0's higher slope from alpha=0.25's lower starting
point would require gamma to INCREASE slope, the opposite of what's validated. The only direction
consistent with the confirmed gamma-slope relationship is pulling the HIGHER-slope checkpoint
(alpha=1.0) DOWN to the LOWER-slope checkpoint's (alpha=0.25) level -- so this script now calibrates
gamma on the alpha=1.0 checkpoint instead, targeting alpha=0.25's natural values.

Per Vinh: no linear extrapolation to guess gamma -- run a narrow calibration grid directly. Single
seed (42), geometry only (LID + spectral-decay slope), no Gate 0, no scorer AUROC -- exactly the
probe -> lock -> canonical-run pattern already used for r-selection (stage4_train_intervention.py)
and lambda-selection (stage4_select_lambda.py).

Targets (seed 42, from frozen AlphaLadder/alpha_ladder_results.json, alpha=0.25, gamma=0):
  LID_target   = 14.9497
  slope_target = 0.006751
Starting point (seed 42, alpha=1.0, gamma=0, same frozen source): LID=14.2610, slope=0.006807.
The natural alpha=0.25-vs-1.0 gap in slope is ~0.00006 -- two orders of magnitude smaller than the
gamma=0.25 manipulation-check step (~0.0015), hence this narrow grid instead of reusing that ladder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_adapter import BottleneckAdapter  # noqa: E402
from stage4_train_intervention import load_eval_population, compute_gate1_measurement, TRAIN_DIR  # noqa: E402
from stage5_c3_spectral_tempering_check import fit_spectral_basis, apply_spectral_tempering, delta_z  # noqa: E402

SEED = 42
ALPHA = 1.0
GAMMAS = [0.005, 0.01, 0.02, 0.05]
R = 16
INTERVENTION_DIR = Path(__file__).resolve().parents[1] / "InterventionArm_Canonical"

LID_TARGET = 14.9497
SLOPE_TARGET = 0.006751
LID_START = 14.2610
SLOPE_START = 0.006807


def main() -> None:
    z_eval, meta_eval = load_eval_population()
    z_train_isic = np.load(TRAIN_DIR / "embeddings.npy").astype(np.float32)
    mu, eigvecs, eigvals = fit_spectral_basis(z_train_isic)

    ckpt = torch.load(INTERVENTION_DIR / f"adapter_seed{SEED}.pt", map_location="cpu", weights_only=True)
    adapter = BottleneckAdapter(r=R)
    adapter.load_state_dict(ckpt["adapter"])
    dz = delta_z(adapter, z_eval)
    z_pre = z_eval + ALPHA * dz

    print(f"[gamma-calibration] seed={SEED} alpha={ALPHA} (natural, gamma=0, being pulled down): "
          f"LID={LID_START:.4f} slope={SLOPE_START:.6f}")
    print(f"[gamma-calibration] target (seed={SEED} alpha=0.25, gamma=0, natural): "
          f"LID={LID_TARGET:.4f} slope={SLOPE_TARGET:.6f}")

    for gamma in GAMMAS:
        z_post = apply_spectral_tempering(z_pre, mu, eigvecs, eigvals, gamma)
        gate1 = compute_gate1_measurement(z_post, meta_eval)
        lid, slope = gate1["lid_mean"], gate1["spectral_decay_slope"]
        print(f"[gamma-calibration] gamma={gamma}: LID={lid:.4f} (target {LID_TARGET:.4f}, "
              f"d={lid-LID_TARGET:+.4f}) slope={slope:.6f} (target {SLOPE_TARGET:.6f}, "
              f"d={slope-SLOPE_TARGET:+.6f})")


if __name__ == "__main__":
    main()
