"""
Stage 4, Execution Plan v2 Step 3 -- bottleneck-width selection loop for the Intervention arm.

Trains BottleneckAdapter+TaskHead on cached frozen PanDerm features (no image/ViT forward pass
needed -- ReferenceTrainEmbedding + ReferenceEmbedding's PAD-UFES rows already contain them),
regularized against the frozen nuisance direction w (NuisanceDirection/), then evaluates Gate 0
(domain-leakage probe + ID task, reusing stage3_2_reference_arm_baseline.py's own functions) and
Gate 1 (LID + spectral-decay slope, reusing lid_spectral_decay.py) on the held-out ReferenceEmbedding
population at full adapter strength (alpha=1).

Grid and stopping rule locked in Stage4_Hyperparameter_Selection_Rules.md SS1: r in {16,32,64,128},
lambda_proj=0.1 fixed probe value, stop at first r that passes both gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_adapter import BottleneckAdapter, TaskHead, orthogonality_loss, task_loss  # noqa: E402
from lid_spectral_decay import compute_lid_spectral_diagnostics  # noqa: E402
from stage3_2_reference_arm_baseline import linear_probe, ece_from_proba  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings"
TRAIN_DIR = ASSETS / "ReferenceTrainEmbedding"
EVAL_DIR = ASSETS / "ReferenceEmbedding"
NUISANCE_PATH = Path(__file__).resolve().parents[1] / "NuisanceDirection" / "nuisance_direction.npz"

R_GRID = [16, 32, 64, 128]
LAMBDA_PROBE = 0.1  # fixed probe value for width selection, Stage4_Hyperparameter_Selection_Rules.md SS1
PROBE_SEEDS = [42, 52, 62, 72, 82]
SEED = 42
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 512

# Gate 0 thresholds, exactly as gate0_verdict.json's own rule (BaselineReport/, frozen)
MAJORITY_BASELINE_DOMAIN = 0.6879837067209776
CHANCE_BALANCED_ACC = 0.125


def load_training_population():
    z_isic = np.load(TRAIN_DIR / "embeddings.npy").astype(np.float32)
    meta_isic = pd.read_csv(TRAIN_DIR / "metadata.csv")
    y_isic = meta_isic["label_idx"].to_numpy().astype(np.int64)

    z_eval_all = np.load(EVAL_DIR / "embeddings.npy").astype(np.float32)
    meta_eval = pd.read_csv(EVAL_DIR / "metadata.csv")
    pad_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()
    z_pad = z_eval_all[pad_mask]
    y_pad = np.full(len(z_pad), -1, dtype=np.int64)  # CSG-Skin's own convention: PAD-UFES not lesion-labeled

    z = np.concatenate([z_isic, z_pad], axis=0)
    y = np.concatenate([y_isic, y_pad], axis=0)
    return z, y


def load_eval_population():
    z_eval = np.load(EVAL_DIR / "embeddings.npy").astype(np.float32)
    meta_eval = pd.read_csv(EVAL_DIR / "metadata.csv")
    return z_eval, meta_eval


def train_adapter(z_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, r: int, lambda_proj: float):
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    adapter = BottleneckAdapter(r=r).to(device)
    head = TaskHead().to(device)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    ds = TensorDataset(torch.tensor(z_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    params = list(adapter.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR)

    for epoch in range(EPOCHS):
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            z_adapted = adapter(zb)
            logits = head(z_adapted)
            loss = task_loss(logits, yb) + lambda_proj * orthogonality_loss(z_adapted, w_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return adapter, head


@torch.no_grad()
def apply_adapter(adapter: BottleneckAdapter, z: np.ndarray) -> np.ndarray:
    adapter.eval()
    z_t = torch.tensor(z, dtype=torch.float32)
    return adapter(z_t).numpy()


def compute_gate0(z_adapted: np.ndarray, meta_eval: pd.DataFrame) -> dict:
    domain_binary = (meta_eval["domain"] == "pad_ufes").to_numpy().astype(np.int64)
    probe_accs = []
    for seed in PROBE_SEEDS:
        result, _, _, _ = linear_probe(z_adapted, domain_binary, seed=seed)
        probe_accs.append(result["accuracy"])
    probe_acc_mean = float(np.mean(probe_accs))

    id_mask = (meta_eval["domain"] == "isic").to_numpy()
    y_id = meta_eval.loc[id_mask, "label_idx"].to_numpy().astype(np.int64)
    z_id = z_adapted[id_mask]
    bal_accs, eces = [], []
    for seed in PROBE_SEEDS:
        result, proba, y_test, _ = linear_probe(z_id, y_id, seed=seed)
        bal_accs.append(result["balanced_accuracy"])
        eces.append(ece_from_proba(proba, y_test))
    balanced_acc_mean = float(np.mean(bal_accs))

    criterion_1 = probe_acc_mean > MAJORITY_BASELINE_DOMAIN + 0.05
    criterion_2 = balanced_acc_mean > CHANCE_BALANCED_ACC + 0.10
    return {
        "domain_probe_accuracy_mean": probe_acc_mean,
        "criterion_1_pass": bool(criterion_1),
        "id_task_balanced_accuracy_mean": balanced_acc_mean,
        "id_task_ece_mean": float(np.mean(eces)),
        "criterion_2_pass": bool(criterion_2),
        "gate0_pass": bool(criterion_1 and criterion_2),
    }


def compute_gate1_measurement(z_adapted: np.ndarray, meta_eval: pd.DataFrame) -> dict:
    labels = meta_eval["label_idx"].to_numpy().astype(np.int64)  # same grouping as Stage 3.2's own geometry_metrics.json
    diag = compute_lid_spectral_diagnostics(z_adapted, labels, num_classes=8)
    return {
        "lid_mean": diag.lid_mean,
        "spectral_decay_slope": diag.spectral_slope,
        "composite": diag.composite,
        "n_samples": diag.n_samples,
    }


def main() -> None:
    print("[stage4-width] loading cached frozen embeddings (no ViT forward pass needed)")
    z_train, y_train = load_training_population()
    z_eval, meta_eval = load_eval_population()
    w = np.load(NUISANCE_PATH)["w"]
    print(f"[stage4-width] training population n={len(z_train)} (ISIC labeled + PAD-UFES unlabeled)")
    print(f"[stage4-width] eval population n={len(z_eval)}")

    # Baseline (alpha=0, i.e. no adapter) LID/slope -- from the frozen Reference Arm's own numbers,
    # not recomputed, since alpha=0 is defined as the untouched embedding (BaselineReport/geometry_metrics.json).
    baseline = json.loads((Path(__file__).resolve().parents[1] / "ReferenceArmBaseline" / "BaselineReport" / "geometry_metrics.json").read_text())
    print(f"[stage4-width] baseline (alpha=0) LID={baseline['lid_mean']:.4f} slope={baseline['spectral_decay_slope']:.6f}")

    results = []
    for r in R_GRID:
        print(f"\n[stage4-width] === r={r}, lambda_proj={LAMBDA_PROBE} (probe value) ===")
        adapter, head = train_adapter(z_train, y_train, w, r=r, lambda_proj=LAMBDA_PROBE)
        z_adapted = apply_adapter(adapter, z_eval)

        gate0 = compute_gate0(z_adapted, meta_eval)
        gate1 = compute_gate1_measurement(z_adapted, meta_eval)
        print(f"[stage4-width] Gate 0: domain_probe_acc={gate0['domain_probe_accuracy_mean']:.4f} "
              f"(criterion_1={'PASS' if gate0['criterion_1_pass'] else 'FAIL'}), "
              f"id_task_bal_acc={gate0['id_task_balanced_accuracy_mean']:.4f} "
              f"(criterion_2={'PASS' if gate0['criterion_2_pass'] else 'FAIL'}) "
              f"-> {'PASS' if gate0['gate0_pass'] else 'FAIL'}")
        print(f"[stage4-width] Gate 1 measurement: LID={gate1['lid_mean']:.4f} "
              f"(baseline {baseline['lid_mean']:.4f}, delta={gate1['lid_mean']-baseline['lid_mean']:+.4f}), "
              f"slope={gate1['spectral_decay_slope']:.6f} "
              f"(baseline {baseline['spectral_decay_slope']:.6f}, delta={gate1['spectral_decay_slope']-baseline['spectral_decay_slope']:+.6f}) "
              f"-- PASS/FAIL threshold not yet fixed, see report")

        results.append({"r": r, "gate0": gate0, "gate1_measurement": gate1})

    out_path = Path(__file__).resolve().parents[1] / "outputs" / "stage4_width_selection_probe.json"
    out_path.write_text(json.dumps({
        "lambda_proj_probe": LAMBDA_PROBE,
        "baseline_alpha0": {"lid_mean": baseline["lid_mean"], "spectral_decay_slope": baseline["spectral_decay_slope"]},
        "r_grid_results": results,
    }, indent=2))
    print(f"\n[stage4-width] raw results (not a gated artifact, not frozen) written to {out_path}")


if __name__ == "__main__":
    main()
