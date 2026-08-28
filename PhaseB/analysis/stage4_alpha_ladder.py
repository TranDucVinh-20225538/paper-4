"""
Stage 4, Execution Plan v2 Step 6 -- alpha-dose ladder.
z'(alpha) = z + alpha*delta_z, alpha in {0,.25,.5,.75,1}, from the ALREADY-TRAINED canonical
adapters (InterventionArm_Canonical/, ConventionalArm_Canonical/) -- no retraining, per
Stage4_Intervention_Method_Specification.md SS3. Applied to both arms (measurement-parity extension,
Stage4_Adapter_And_Control_Specification.md SS3). Gate 0 + Gate 1 (EA-02) computed per arm x seed x
alpha.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_adapter import BottleneckAdapter  # noqa: E402
from stage4_train_intervention import load_eval_population, compute_gate0, compute_gate1_measurement  # noqa: E402
from stage4_select_lambda import gate1_pass  # noqa: E402

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
    baseline = json.loads((Path(__file__).resolve().parents[1] / "ReferenceArmBaseline" / "BaselineReport" / "geometry_metrics.json").read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
                gate0 = compute_gate0(z_alpha, meta_eval)
                gate1_meas = compute_gate1_measurement(z_alpha, meta_eval)
                g1 = gate1_pass(gate1_meas, baseline)
                print(f"[alpha-ladder] {arm_name} seed={seed} alpha={alpha}: "
                      f"Gate0={'PASS' if gate0['gate0_pass'] else 'FAIL'} "
                      f"(probe={gate0['domain_probe_accuracy_mean']:.4f}, id_task={gate0['id_task_balanced_accuracy_mean']:.4f}) "
                      f"Gate1={'PASS' if g1 else 'FAIL'} (LID={gate1_meas['lid_mean']:.4f}, slope={gate1_meas['spectral_decay_slope']:.6f})")
                arm_results.append({
                    "seed": seed, "alpha": alpha, "gate0": gate0,
                    "gate1_measurement": gate1_meas, "gate1_pass": g1,
                })
        all_results[arm_name] = arm_results

    out_path = OUTPUT_DIR / "alpha_ladder_results.json"
    out_path.write_text(json.dumps({
        "alphas": ALPHAS, "r": R,
        "baseline_alpha0_reference": {"lid_mean": baseline["lid_mean"], "spectral_decay_slope": baseline["spectral_decay_slope"]},
        "results": all_results,
    }, indent=2))
    print(f"\n[alpha-ladder] all results written to {out_path} -- NOT frozen (pending review)")


if __name__ == "__main__":
    main()
