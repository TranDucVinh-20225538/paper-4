"""
Stage 4, Execution Plan v2 Step 4 -- lambda_proj selection, r=16 fixed (Step 3 result).
Grid + stopping rule locked in Stage4_Hyperparameter_Selection_Rules.md SS3; Gate 1 criterion per
EA-02. No new grid values, no mid-run tuning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_train_intervention import (  # noqa: E402
    R_GRID, load_training_population, load_eval_population, train_adapter,
    apply_adapter, compute_gate0, compute_gate1_measurement, NUISANCE_PATH,
)

R_FIXED = 16  # Step 3 result
LAMBDA_GRID = [0.1, 1, 5]  # Stage4_Hyperparameter_Selection_Rules.md SS3, unchanged


def gate1_pass(measurement: dict, baseline: dict) -> bool:
    """EA-02: PASS iff at least one of LID/slope shows a measurable, correctly-directed
    (decreasing) change vs. the reference artifact. Same criterion applied in Step 3."""
    lid_changed = measurement["lid_mean"] < baseline["lid_mean"]
    slope_changed = measurement["spectral_decay_slope"] < baseline["spectral_decay_slope"]
    return bool(lid_changed or slope_changed)


def main() -> None:
    z_train, y_train = load_training_population()
    z_eval, meta_eval = load_eval_population()
    w = np.load(NUISANCE_PATH)["w"]
    baseline = json.loads((Path(__file__).resolve().parents[1] / "ReferenceArmBaseline" / "BaselineReport" / "geometry_metrics.json").read_text())
    print(f"[stage4-lambda] r={R_FIXED} (fixed, Step 3 result). baseline LID={baseline['lid_mean']:.4f} slope={baseline['spectral_decay_slope']:.6f}")

    results = []
    selected = None
    for lam in LAMBDA_GRID:
        print(f"\n[stage4-lambda] === lambda_proj={lam} ===")
        adapter, head = train_adapter(z_train, y_train, w, r=R_FIXED, lambda_proj=lam)
        z_adapted = apply_adapter(adapter, z_eval)

        gate0 = compute_gate0(z_adapted, meta_eval)
        gate1_meas = compute_gate1_measurement(z_adapted, meta_eval)
        g1_pass = gate1_pass(gate1_meas, baseline)
        print(f"[stage4-lambda] Gate 0: {'PASS' if gate0['gate0_pass'] else 'FAIL'} "
              f"(probe={gate0['domain_probe_accuracy_mean']:.4f}, id_task={gate0['id_task_balanced_accuracy_mean']:.4f})")
        print(f"[stage4-lambda] Gate 1 (EA-02): LID={gate1_meas['lid_mean']:.4f} "
              f"(delta={gate1_meas['lid_mean']-baseline['lid_mean']:+.4f}), "
              f"slope={gate1_meas['spectral_decay_slope']:.6f} "
              f"(delta={gate1_meas['spectral_decay_slope']-baseline['spectral_decay_slope']:+.6f}) "
              f"-> {'PASS' if g1_pass else 'FAIL'}")

        both_pass = gate0["gate0_pass"] and g1_pass
        results.append({"lambda_proj": lam, "gate0": gate0, "gate1_measurement": gate1_meas,
                         "gate1_pass": g1_pass, "both_pass": both_pass})
        if both_pass and selected is None:
            selected = lam
            print(f"[stage4-lambda] STOP -- lambda_proj={lam} is the smallest value passing both gates")
            break

    out_path = Path(__file__).resolve().parents[1] / "outputs" / "stage4_lambda_selection.json"
    out_path.write_text(json.dumps({
        "r_fixed": R_FIXED,
        "baseline_alpha0": {"lid_mean": baseline["lid_mean"], "spectral_decay_slope": baseline["spectral_decay_slope"]},
        "lambda_grid_results": results,
        "selected_lambda_proj": selected,
    }, indent=2))
    print(f"\n[stage4-lambda] selected_lambda_proj = {selected}")
    print(f"[stage4-lambda] results written to {out_path}")


if __name__ == "__main__":
    main()
