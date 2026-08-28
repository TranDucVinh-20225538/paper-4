"""
Stage 4 -- Canonical Intervention Arm, 5 seeds, r=16 / lambda_proj=0.1 fixed (Step 3/4 results).
NOT a hyperparameter probe -- this is the actual scientific artifact: trained adapter + adapted
embeddings + Gate 0/Gate 1 (EA-02) per seed and aggregated, frozen only if both gates pass at the
aggregate level. Kept explicitly separate from the probe runs (stage4_train_intervention.py,
stage4_select_lambda.py), which produced no frozen artifact.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_train_intervention import (  # noqa: E402
    load_training_population, load_eval_population, train_adapter,
    apply_adapter, compute_gate0, compute_gate1_measurement, NUISANCE_PATH,
)
from stage4_select_lambda import gate1_pass  # noqa: E402

R = 16
LAMBDA_PROJ = 0.1
SEEDS = [42, 52, 62, 72, 82]  # matches PROBE_SEEDS / SS5 Controls, multi-seed requirement

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "InterventionArm_Canonical"
NUISANCE_MANIFEST = Path(__file__).resolve().parents[1] / "NuisanceDirection" / "manifest.json"
TRAIN_MANIFEST = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceTrainEmbedding" / "manifest.json"
EVAL_MANIFEST = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceEmbedding" / "manifest.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    z_train, y_train = load_training_population()
    z_eval, meta_eval = load_eval_population()
    w = np.load(NUISANCE_PATH)["w"]
    baseline = json.loads((Path(__file__).resolve().parents[1] / "ReferenceArmBaseline" / "BaselineReport" / "geometry_metrics.json").read_text())
    print(f"[canonical] r={R} lambda_proj={LAMBDA_PROJ} (Step 3/4 results, fixed) -- {len(SEEDS)} seeds")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in SEEDS:
        print(f"\n[canonical] === seed={seed} ===")
        import stage4_train_intervention as m
        m.SEED = seed  # train_adapter reads module-level SEED for torch.manual_seed + loader shuffling
        adapter, head = train_adapter(z_train, y_train, w, r=R, lambda_proj=LAMBDA_PROJ)
        z_adapted = apply_adapter(adapter, z_eval)

        gate0 = compute_gate0(z_adapted, meta_eval)
        gate1_meas = compute_gate1_measurement(z_adapted, meta_eval)
        g1 = gate1_pass(gate1_meas, baseline)
        print(f"[canonical] Gate 0: {'PASS' if gate0['gate0_pass'] else 'FAIL'} "
              f"(probe={gate0['domain_probe_accuracy_mean']:.4f}, id_task={gate0['id_task_balanced_accuracy_mean']:.4f})")
        print(f"[canonical] Gate 1: LID={gate1_meas['lid_mean']:.4f} slope={gate1_meas['spectral_decay_slope']:.6f} "
              f"-> {'PASS' if g1 else 'FAIL'}")

        emb_path = OUTPUT_DIR / f"embeddings_seed{seed}.npy"
        np.save(emb_path, z_adapted.astype(np.float32))
        ckpt_path = OUTPUT_DIR / f"adapter_seed{seed}.pt"
        torch.save({"adapter": adapter.state_dict(), "head": head.state_dict()}, ckpt_path)

        per_seed.append({
            "seed": seed, "gate0": gate0, "gate1_measurement": gate1_meas, "gate1_pass": g1,
            "embeddings_file": emb_path.name, "checkpoint_file": ckpt_path.name,
        })

    # Aggregate, same style as Reference Arm's own 5-seed probe aggregation
    probe_acc_mean = float(np.mean([r["gate0"]["domain_probe_accuracy_mean"] for r in per_seed]))
    id_bal_acc_mean = float(np.mean([r["gate0"]["id_task_balanced_accuracy_mean"] for r in per_seed]))
    gate0_agg_pass = all(r["gate0"]["gate0_pass"] for r in per_seed)
    lid_mean = float(np.mean([r["gate1_measurement"]["lid_mean"] for r in per_seed]))
    slope_mean = float(np.mean([r["gate1_measurement"]["spectral_decay_slope"] for r in per_seed]))
    gate1_agg_pass = all(r["gate1_pass"] for r in per_seed)  # EA-02: "reproducible" = every seed agrees

    print(f"\n[canonical] === AGGREGATE (5 seeds) ===")
    print(f"[canonical] Gate 0: probe_acc_mean={probe_acc_mean:.4f} id_task_bal_acc_mean={id_bal_acc_mean:.4f} "
          f"-> {'PASS' if gate0_agg_pass else 'FAIL'} (all 5 seeds individually pass)")
    print(f"[canonical] Gate 1: LID_mean={lid_mean:.4f} (baseline {baseline['lid_mean']:.4f}) "
          f"slope_mean={slope_mean:.6f} (baseline {baseline['spectral_decay_slope']:.6f}) "
          f"-> {'PASS' if gate1_agg_pass else 'FAIL'} (reproducible = all 5 seeds individually pass)")

    both_pass = gate0_agg_pass and gate1_agg_pass
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest = {
        "artifact": "Canonical Intervention Arm (5 seeds)",
        "protocol_step": "Stage 4 Execution Plan v2, Step 4 (canonical, not a probe)",
        "hyperparameters": {"r": R, "lambda_proj": LAMBDA_PROJ, "seeds": SEEDS,
                             "source": "Step 3/4 hyperparameter selection, EA-02 Gate 1 criterion"},
        "adapter_architecture": "z' = z + W2.GELU(W1.z), bottleneck r=16, zero-init up-projection",
        "orthogonality_target": {"artifact": str(NUISANCE_MANIFEST), "formula": "cos(z',w)^2"},
        "training_population": {"isic_train_artifact": str(TRAIN_MANIFEST),
                                 "pad_ufes_from_artifact": str(EVAL_MANIFEST), "n_total": int(len(z_train))},
        "eval_population_artifact": str(EVAL_MANIFEST),
        "per_seed": per_seed,
        "aggregate": {
            "gate0_probe_accuracy_mean": probe_acc_mean,
            "gate0_id_task_balanced_accuracy_mean": id_bal_acc_mean,
            "gate0_pass": gate0_agg_pass,
            "gate1_lid_mean": lid_mean,
            "gate1_spectral_decay_slope_mean": slope_mean,
            "gate1_pass": gate1_agg_pass,
            "baseline_alpha0": {"lid_mean": baseline["lid_mean"], "spectral_decay_slope": baseline["spectral_decay_slope"]},
        },
        "both_gates_pass": both_pass,
        "not_computed_here": "No alpha-ladder, no Conventional arm, no measurement suite beyond Gate0/Gate1, no comparison, no claim.",
        "created_at": now,
        "status": "FROZEN" if both_pass else "NOT_FROZEN_GATE_FAILURE",
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if both_pass:
        sha_lines = []
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.name == "sha256.txt":
                continue
            sha_lines.append(f"{_sha256_file(f)}  {f.name}")
        (OUTPUT_DIR / "sha256.txt").write_text("\n".join(sha_lines) + "\n")
        for f in OUTPUT_DIR.iterdir():
            f.chmod(0o444)
        OUTPUT_DIR.chmod(0o555)
        print(f"\n[canonical] BOTH GATES PASS (aggregate). Frozen: {OUTPUT_DIR}")
    else:
        print(f"\n[canonical] GATE FAILURE at aggregate level -- NOT frozen. See {manifest_path}")


if __name__ == "__main__":
    main()
