"""
Stage 4, Execution Plan v2 Step 7 -- geometry measurement suite completion.
Condition number + per-direction variance of w, at every (arm, seed, alpha) already computed by
the alpha-ladder (stage4_alpha_ladder.py) -- reuses the same adapted embeddings logic, no retraining.
LID + spectral-decay slope already exist (Gate 1's own measurement); not recomputed here.

condition_number: paper-3/geometry_diagnostics.py, reused (same call Stage 3.2 Group A used).
per_direction_variance: NOT computable on the Reference arm alone (flagged in
BaselineReport/geometry_metrics.json) -- computable now that w (the intervened direction) exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(data_roots.root("paper3_scripts")))
sys.path.insert(0, str(data_roots.root("csg")))

from stage4_adapter import BottleneckAdapter  # noqa: E402
from stage4_train_intervention import load_eval_population, NUISANCE_PATH  # noqa: E402
from geometry_diagnostics import condition_number  # noqa: E402  -- paper-3, reused
from src.utils.ood_metrics import compute_mahalanobis_params_from_arrays  # noqa: E402  -- CSG-Skin, reused

import data_roots

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
R = 16
ARMS = {
    "Intervention": Path(__file__).resolve().parents[1] / "InterventionArm_Canonical",
    "Conventional": Path(__file__).resolve().parents[1] / "ConventionalArm_Canonical",
}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "AlphaLadder"


@torch.no_grad()
def delta_z(adapter: BottleneckAdapter, z: np.ndarray) -> np.ndarray:
    adapter.eval()
    return adapter.delta(torch.tensor(z, dtype=torch.float32)).numpy()


def main() -> None:
    z_eval, meta_eval = load_eval_population()
    w = np.load(NUISANCE_PATH)["w"]
    labels = meta_eval["label_idx"].to_numpy().astype(np.int64)  # same 8-class grouping as Stage 3.2's own geometry_metrics.json

    all_results = {}
    for arm_name, arm_dir in ARMS.items():
        manifest = json.loads((arm_dir / "manifest.json").read_text())
        seeds = [r["seed"] for r in manifest["per_seed"]]
        arm_results = []
        for seed in seeds:
            ckpt = torch.load(arm_dir / f"adapter_seed{seed}.pt", map_location="cpu", weights_only=True)
            adapter = BottleneckAdapter(r=R)
            adapter.load_state_dict(ckpt["adapter"])
            dz = delta_z(adapter, z_eval)

            for alpha in ALPHAS:
                z_alpha = z_eval + alpha * dz
                _means, precision = compute_mahalanobis_params_from_arrays(z_alpha, labels, num_classes=8, reg_eps=1e-5)
                kappa = float(condition_number(precision))
                proj = z_alpha @ w
                per_dir_var = float(np.var(proj))

                print(f"[geometry] {arm_name} seed={seed} alpha={alpha}: "
                      f"condition_number={kappa:.4f}  per_direction_variance(w)={per_dir_var:.6f}")
                arm_results.append({
                    "seed": seed, "alpha": alpha,
                    "condition_number": kappa, "per_direction_variance_w": per_dir_var,
                })
        all_results[arm_name] = arm_results

    # Reference (alpha=0 is identical across arms/seeds by construction -- one reference value)
    _means_ref, precision_ref = compute_mahalanobis_params_from_arrays(z_eval, labels, num_classes=8, reg_eps=1e-5)
    kappa_ref = float(condition_number(precision_ref))
    per_dir_var_ref = float(np.var(z_eval @ w))
    print(f"\n[geometry] Reference (alpha=0, no adapter): condition_number={kappa_ref:.4f} "
          f"per_direction_variance(w)={per_dir_var_ref:.6f}")

    out_path = OUTPUT_DIR / "geometry_completion_results.json"
    out_path.write_text(json.dumps({
        "alphas": ALPHAS, "r": R,
        "reference_alpha0": {"condition_number": kappa_ref, "per_direction_variance_w": per_dir_var_ref},
        "results": all_results,
        "not_computed_here": "LID and spectral-decay slope already computed by Gate 1 (AlphaLadder/alpha_ladder_results.json), not repeated here. No reliability scorers (Mahalanobis/kNN/cosine/KDE) -- separate step, needs an adapted train-population fit.",
    }, indent=2))
    print(f"\n[geometry] results written to {out_path} -- NOT frozen (pending review)")


if __name__ == "__main__":
    main()
