"""
Stage 4 -- Canonical Conventional Arm, 5 seeds, r=16 fixed (matches Intervention exactly).
Stage4_Adapter_And_Control_Specification.md SS3: same architecture, bottleneck width, optimizer,
epochs, seeds as Intervention -- L_orth structurally ABSENT (loss = L_task only), not lambda_proj=0.
Isolates intervention type (C4): does ordinary task-only adapter fine-tuning also shift geometry,
independent of the orthogonality mechanism?
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_adapter import BottleneckAdapter, TaskHead, task_loss  # noqa: E402
from stage4_train_intervention import (  # noqa: E402
    load_training_population, load_eval_population, apply_adapter,
    compute_gate0, compute_gate1_measurement, EPOCHS, LR, BATCH_SIZE,
)
from stage4_select_lambda import gate1_pass  # noqa: E402

R = 16  # matches Intervention exactly, Stage4_Adapter_And_Control_Specification.md SS3
SEEDS = [42, 52, 62, 72, 82]  # matches Intervention exactly

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "ConventionalArm_Canonical"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_conventional(z_train: np.ndarray, y_train: np.ndarray, seed: int):
    """Identical to stage4_train_intervention.train_adapter's optimizer/loop, minus L_orth
    entirely -- not an ablation of lambda_proj, a structurally different loss."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    adapter = BottleneckAdapter(r=R).to(device)
    head = TaskHead().to(device)

    ds = TensorDataset(torch.tensor(z_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))

    params = list(adapter.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)

    for epoch in range(EPOCHS):
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            z_adapted = adapter(zb)
            logits = head(z_adapted)
            loss = task_loss(logits, yb)  # L_task only -- no L_orth term anywhere in this loop
            opt.zero_grad()
            loss.backward()
            opt.step()
    return adapter, head


def main() -> None:
    z_train, y_train = load_training_population()
    z_eval, meta_eval = load_eval_population()
    baseline = json.loads((Path(__file__).resolve().parents[1] / "ReferenceArmBaseline" / "BaselineReport" / "geometry_metrics.json").read_text())
    print(f"[conventional] r={R} (matches Intervention), loss=L_task only -- {len(SEEDS)} seeds")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in SEEDS:
        print(f"\n[conventional] === seed={seed} ===")
        adapter, head = train_conventional(z_train, y_train, seed)
        z_adapted = apply_adapter(adapter, z_eval)

        gate0 = compute_gate0(z_adapted, meta_eval)
        gate1_meas = compute_gate1_measurement(z_adapted, meta_eval)
        g1 = gate1_pass(gate1_meas, baseline)
        print(f"[conventional] Gate 0: {'PASS' if gate0['gate0_pass'] else 'FAIL'} "
              f"(probe={gate0['domain_probe_accuracy_mean']:.4f}, id_task={gate0['id_task_balanced_accuracy_mean']:.4f})")
        print(f"[conventional] Gate 1: LID={gate1_meas['lid_mean']:.4f} slope={gate1_meas['spectral_decay_slope']:.6f} "
              f"-> {'PASS' if g1 else 'FAIL'}")

        emb_path = OUTPUT_DIR / f"embeddings_seed{seed}.npy"
        np.save(emb_path, z_adapted.astype(np.float32))
        ckpt_path = OUTPUT_DIR / f"adapter_seed{seed}.pt"
        torch.save({"adapter": adapter.state_dict(), "head": head.state_dict()}, ckpt_path)

        per_seed.append({
            "seed": seed, "gate0": gate0, "gate1_measurement": gate1_meas, "gate1_pass": g1,
            "embeddings_file": emb_path.name, "checkpoint_file": ckpt_path.name,
        })

    probe_acc_mean = float(np.mean([r["gate0"]["domain_probe_accuracy_mean"] for r in per_seed]))
    id_bal_acc_mean = float(np.mean([r["gate0"]["id_task_balanced_accuracy_mean"] for r in per_seed]))
    gate0_agg_pass = all(r["gate0"]["gate0_pass"] for r in per_seed)
    lid_mean = float(np.mean([r["gate1_measurement"]["lid_mean"] for r in per_seed]))
    slope_mean = float(np.mean([r["gate1_measurement"]["spectral_decay_slope"] for r in per_seed]))
    gate1_agg_pass = all(r["gate1_pass"] for r in per_seed)

    print(f"\n[conventional] === AGGREGATE (5 seeds) ===")
    print(f"[conventional] Gate 0: probe_acc_mean={probe_acc_mean:.4f} id_task_bal_acc_mean={id_bal_acc_mean:.4f} "
          f"-> {'PASS' if gate0_agg_pass else 'FAIL'}")
    print(f"[conventional] Gate 1: LID_mean={lid_mean:.4f} (baseline {baseline['lid_mean']:.4f}) "
          f"slope_mean={slope_mean:.6f} (baseline {baseline['spectral_decay_slope']:.6f}) "
          f"-> {'PASS' if gate1_agg_pass else 'FAIL'}")

    # Gate 0 is the only PASS/FAIL requirement for freezing (Conventional isn't supposed to pass
    # Gate 1 -- it has no manipulation mechanism; Gate 1's role here is diagnostic for C4, not a
    # freeze precondition. This is a genuine interpretive choice, flagged in the report.)
    freeze_ok = gate0_agg_pass
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest = {
        "artifact": "Canonical Conventional Arm (5 seeds)",
        "protocol_step": "Stage 4 Execution Plan v2, Step 5",
        "hyperparameters": {"r": R, "seeds": SEEDS, "loss": "L_task only -- L_orth structurally absent",
                             "matches_intervention": "architecture, r, optimizer, LR, epochs, seeds -- all identical to InterventionArm_Canonical"},
        "adapter_architecture": "z' = z + W2.GELU(W1.z), bottleneck r=16, zero-init up-projection",
        "per_seed": per_seed,
        "aggregate": {
            "gate0_probe_accuracy_mean": probe_acc_mean,
            "gate0_id_task_balanced_accuracy_mean": id_bal_acc_mean,
            "gate0_pass": gate0_agg_pass,
            "gate1_lid_mean": lid_mean,
            "gate1_spectral_decay_slope_mean": slope_mean,
            "gate1_pass_diagnostic_only": gate1_agg_pass,
            "baseline_alpha0": {"lid_mean": baseline["lid_mean"], "spectral_decay_slope": baseline["spectral_decay_slope"]},
        },
        "freeze_precondition": "Gate 0 only -- Gate 1 is diagnostic for C4 (does task-only capacity also shift geometry?), not a freeze requirement, since Conventional has no manipulation mechanism to check",
        "not_computed_here": "No alpha-ladder, no measurement suite beyond Gate0/Gate1, no comparison, no claim.",
        "created_at": now,
        "status": "FROZEN" if freeze_ok else "NOT_FROZEN_GATE0_FAILURE",
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if freeze_ok:
        sha_lines = []
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.name == "sha256.txt":
                continue
            sha_lines.append(f"{_sha256_file(f)}  {f.name}")
        (OUTPUT_DIR / "sha256.txt").write_text("\n".join(sha_lines) + "\n")
        for f in OUTPUT_DIR.iterdir():
            f.chmod(0o444)
        OUTPUT_DIR.chmod(0o555)
        print(f"\n[conventional] Gate 0 PASS (freeze precondition). Frozen: {OUTPUT_DIR}")
    else:
        print(f"\n[conventional] Gate 0 FAILURE -- NOT frozen. See {manifest_path}")


if __name__ == "__main__":
    main()
