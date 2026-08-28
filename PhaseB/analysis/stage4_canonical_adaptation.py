"""Stage 4 -- Canonical Adaptation Arm. 3 rungs per 22_Experimental_Design.md SS3
("linear-probe / partial-FT / full-adapter-FT"), task-only loss (24_PreRegistration.md line 58:
"ordinary fine-tuning" control, same as Conventional), Gate 0 only -- Gate 1's own text names only
Intervention/Conventional checkpoints, not Adaptation.

Rung capacities (execution-level parameterization, not literally numbered in the protocol beyond
the 3 named categories + a "3 rungs" fallback count in SS8):
  linear-probe   : no adapter at all, z unchanged, linear head only (identical construction to
                   Reference Arm's own ID-task probe)
  partial-FT     : bottleneck adapter, r=8 (half of Intervention/Conventional's r=16), task-only loss
  full-adapter-FT: r=16, task-only loss -- IDENTICAL condition to ConventionalArm_Canonical/, whose
                   already-frozen 5-seed data is reused as this rung rather than retrained
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/Users/cubo/Research/paper-3/scripts")
sys.path.insert(0, "/Users/cubo/Research/CSG-SKin")

from stage4_adapter import BottleneckAdapter, TaskHead, task_loss
from stage4_train_intervention import (
    load_training_population, load_eval_population, compute_gate0,
    EPOCHS, LR, BATCH_SIZE, TRAIN_DIR,
)
from geometry_diagnostics import condition_number
from src.utils import ood_metrics
from groupB_mahalanobis_family import cosine_centroid_scores, compute_knn_scores, density_kde_scores, auroc_fpr95, K_VALUES, PRIMARY_K, NUM_CLASSES, REG_EPS
from lid_spectral_decay import compute_lid_spectral_diagnostics

SEEDS = [42, 52, 62, 72, 82]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "AdaptationArm_Canonical"
NUISANCE_PATH = Path(__file__).resolve().parents[1] / "NuisanceDirection" / "nuisance_direction.npz"
CONVENTIONAL_DIR = Path(__file__).resolve().parents[1] / "ConventionalArm_Canonical"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_linear_head(z_train, y_train, seed):
    torch.manual_seed(seed)
    head = TaskHead()
    ds = TensorDataset(torch.tensor(z_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for zb, yb in loader:
            loss = task_loss(head(zb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def train_adapter_task_only(z_train, y_train, seed, r):
    torch.manual_seed(seed)
    adapter = BottleneckAdapter(r=r)
    head = TaskHead()
    ds = TensorDataset(torch.tensor(z_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=LR)
    for _ in range(EPOCHS):
        for zb, yb in loader:
            z_adapted = adapter(zb)
            loss = task_loss(head(z_adapted), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return adapter, head


def measure(z_train_a, y_train, z_eval_a, meta_eval, w):
    labels = meta_eval["label_idx"].to_numpy().astype(np.int64)
    id_mask = (meta_eval["domain"] == "isic").to_numpy()
    ood_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()

    gate0 = compute_gate0(z_eval_a, meta_eval)
    lid_diag = compute_lid_spectral_diagnostics(z_eval_a, labels, num_classes=8, k=50)
    means_g, precision_g = ood_metrics.compute_mahalanobis_params_from_arrays(z_eval_a, labels, num_classes=8, reg_eps=1e-5)
    kappa = float(condition_number(precision_g))
    pdv = float(np.var(z_eval_a @ w))

    z_id, z_ood = z_eval_a[id_mask], z_eval_a[ood_mask]
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(z_train_a, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS)
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
    maha_auroc, maha_fpr95 = auroc_fpr95(s_id, s_ood)
    cos_auroc, cos_fpr95 = auroc_fpr95(cosine_centroid_scores(z_id, means), cosine_centroid_scores(z_ood, means))
    knn = compute_knn_scores(z_train_a, z_id, z_ood, K_VALUES)
    knn_auroc, knn_fpr95 = auroc_fpr95(*knn[PRIMARY_K])
    kde_auroc, kde_fpr95 = auroc_fpr95(density_kde_scores(z_train_a, y_train, z_id), density_kde_scores(z_train_a, y_train, z_ood))

    return {
        "gate0": gate0,
        "lid_mean": lid_diag.lid_mean, "spectral_decay_slope": lid_diag.spectral_slope,
        "condition_number": kappa, "per_direction_variance_w": pdv,
        "maha_auroc": maha_auroc, "maha_fpr95": maha_fpr95,
        "cosine_auroc": cos_auroc, "cosine_fpr95": cos_fpr95,
        "knn_k10_auroc": knn_auroc, "knn_k10_fpr95": knn_fpr95,
        "kde_auroc": kde_auroc, "kde_fpr95": kde_fpr95,
    }


def main():
    z_train, y_train = load_training_population()
    z_eval, meta_eval = load_eval_population()
    w = np.load(NUISANCE_PATH)["w"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rungs = {}

    # Rung 1: linear-probe -- no adapter, z unchanged
    print("=== Rung: linear-probe (no adapter) ===")
    rung_results = []
    for seed in SEEDS:
        head = train_linear_head(z_train, y_train, seed)
        m = measure(z_train, y_train, z_eval, meta_eval, w)  # z unchanged for both train/eval
        m["seed"] = seed
        rung_results.append(m)
        print(f"  seed={seed}: Gate0={'PASS' if m['gate0']['gate0_pass'] else 'FAIL'} LID={m['lid_mean']:.3f} Maha={m['maha_auroc']:.4f}")
    rungs["linear-probe"] = rung_results

    # Rung 2: partial-FT, r=8, task-only
    print("=== Rung: partial-FT (r=8, task-only) ===")
    rung_results = []
    for seed in SEEDS:
        adapter, head = train_adapter_task_only(z_train, y_train, seed, r=8)
        with torch.no_grad():
            adapter.eval()
            z_train_a = adapter(torch.tensor(z_train, dtype=torch.float32)).numpy()
            z_eval_a = adapter(torch.tensor(z_eval, dtype=torch.float32)).numpy()
        m = measure(z_train_a, y_train, z_eval_a, meta_eval, w)
        m["seed"] = seed
        rung_results.append(m)
        torch.save({"adapter": adapter.state_dict(), "head": head.state_dict()}, OUTPUT_DIR / f"partialFT_adapter_seed{seed}.pt")
        print(f"  seed={seed}: Gate0={'PASS' if m['gate0']['gate0_pass'] else 'FAIL'} LID={m['lid_mean']:.3f} Maha={m['maha_auroc']:.4f}")
    rungs["partial-FT"] = rung_results

    # Rung 3: full-adapter-FT -- reuse ConventionalArm_Canonical/ (identical condition: r=16, task-only)
    print("=== Rung: full-adapter-FT (reusing ConventionalArm_Canonical/, r=16 task-only, identical condition) ===")
    conv_manifest = json.loads((CONVENTIONAL_DIR / "manifest.json").read_text())
    rung_results = []
    for r in conv_manifest["per_seed"]:
        rung_results.append({
            "seed": r["seed"], "gate0": r["gate0"],
            "lid_mean": r["gate1_measurement"]["lid_mean"], "spectral_decay_slope": r["gate1_measurement"]["spectral_decay_slope"],
            "condition_number": None, "per_direction_variance_w": None,  # not computed for Conventional at alpha=1 endpoint separately; see AlphaLadder geometry_completion for alpha=1.0 values
            "reused_from": str(CONVENTIONAL_DIR / "manifest.json"),
        })
    rungs["full-adapter-FT"] = rung_results
    print(f"  reused {len(rung_results)} seeds from ConventionalArm_Canonical/manifest.json (no retraining)")

    gate0_all_pass = all(r["gate0"]["gate0_pass"] for rung in ["linear-probe", "partial-FT"] for r in rungs[rung])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        "artifact": "Canonical Adaptation Arm (3 rungs x 5 seeds)",
        "protocol_step": "Stage 4 -- Adaptation arm, 22_Experimental_Design.md SS3",
        "rung_definitions": {
            "linear-probe": "no adapter, frozen z, linear task head only",
            "partial-FT": "bottleneck adapter r=8, task-only loss (no orthogonality term)",
            "full-adapter-FT": "bottleneck adapter r=16, task-only loss -- identical condition to ConventionalArm_Canonical/, reused not retrained",
        },
        "loss": "L_task only, no orthogonality term -- ordinary fine-tuning control, per 24_PreRegistration.md line 58",
        "gates": "Gate 0 only -- Gate 1 not applicable (22_Experimental_Design.md SS3.5 names only Intervention/Conventional checkpoints)",
        "rungs": rungs,
        "gate0_pass_linear_probe_and_partial_ft": gate0_all_pass,
        "created_at": now,
        "status": "FROZEN" if gate0_all_pass else "NOT_FROZEN_GATE0_FAILURE",
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    if gate0_all_pass:
        sha_lines = []
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.name == "sha256.txt":
                continue
            sha_lines.append(f"{sha256_file(f)}  {f.name}")
        (OUTPUT_DIR / "sha256.txt").write_text("\n".join(sha_lines) + "\n")
        for f in OUTPUT_DIR.iterdir():
            f.chmod(0o444)
        OUTPUT_DIR.chmod(0o555)
        print(f"\nGate 0 PASS (linear-probe + partial-FT). Frozen: {OUTPUT_DIR}")
    else:
        print(f"\nGate 0 FAILURE -- NOT frozen. See {manifest_path}")


if __name__ == "__main__":
    main()
