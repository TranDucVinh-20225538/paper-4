"""Stage 4 Step 7 -- Mahalanobis/kNN/cosine/KDE on adapted embeddings, alpha in {.25,.5,.75,1}, both arms.
Fits on adapted ISIC-train, evaluates on adapted held-out ID/OOD. Reuses groupB_mahalanobis_family.py's
exact functions."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/Users/cubo/Research/CSG-SKin")
from stage4_adapter import BottleneckAdapter
from stage4_train_intervention import TRAIN_DIR, load_eval_population
from groupB_mahalanobis_family import (
    cosine_centroid_scores, compute_knn_scores, density_kde_scores, auroc_fpr95, K_VALUES, PRIMARY_K, NUM_CLASSES, REG_EPS,
)
from src.utils import ood_metrics

ALPHAS = [0.25, 0.5, 0.75, 1.0]
R = 16
ARMS = {"Intervention": Path("InterventionArm_Canonical"), "Conventional": Path("ConventionalArm_Canonical")}

z_train_raw = np.load(TRAIN_DIR / "embeddings.npy").astype(np.float32)
y_train = pd.read_csv(TRAIN_DIR / "metadata.csv")["label_idx"].to_numpy().astype(np.int64)
z_eval_raw, meta_eval = load_eval_population()
id_mask = (meta_eval["domain"] == "isic").to_numpy()
ood_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()

@torch.no_grad()
def delta(adapter, z):
    adapter.eval()
    return adapter.delta(torch.tensor(z, dtype=torch.float32)).numpy()

results = {}
for arm, arm_dir in ARMS.items():
    manifest = json.loads((arm_dir / "manifest.json").read_text())
    seeds = [r["seed"] for r in manifest["per_seed"]]
    arm_results = []
    for seed in seeds:
        ckpt = torch.load(arm_dir / f"adapter_seed{seed}.pt", map_location="cpu", weights_only=True)
        adapter = BottleneckAdapter(r=R)
        adapter.load_state_dict(ckpt["adapter"])
        dz_train = delta(adapter, z_train_raw)
        dz_eval = delta(adapter, z_eval_raw)

        for alpha in ALPHAS:
            z_train_a = z_train_raw + alpha * dz_train
            z_eval_a = z_eval_raw + alpha * dz_eval
            z_id, z_ood = z_eval_a[id_mask], z_eval_a[ood_mask]

            means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(z_train_a, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS)
            s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
            s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
            maha_auroc, maha_fpr95 = auroc_fpr95(s_id, s_ood)

            cos_auroc, cos_fpr95 = auroc_fpr95(cosine_centroid_scores(z_id, means), cosine_centroid_scores(z_ood, means))

            knn = compute_knn_scores(z_train_a, z_id, z_ood, K_VALUES)
            knn_primary_auroc, knn_primary_fpr95 = auroc_fpr95(*knn[PRIMARY_K])

            kde_id = density_kde_scores(z_train_a, y_train, z_id)
            kde_ood = density_kde_scores(z_train_a, y_train, z_ood)
            kde_auroc, kde_fpr95 = auroc_fpr95(kde_id, kde_ood)

            row = {"seed": seed, "alpha": alpha,
                   "maha_auroc": maha_auroc, "maha_fpr95": maha_fpr95,
                   "cosine_auroc": cos_auroc, "cosine_fpr95": cos_fpr95,
                   "knn_k10_auroc": knn_primary_auroc, "knn_k10_fpr95": knn_primary_fpr95,
                   "kde_auroc": kde_auroc, "kde_fpr95": kde_fpr95}
            arm_results.append(row)
            print(f"[reliability] {arm} seed={seed} alpha={alpha}: "
                  f"Maha={maha_auroc:.4f} Cos={cos_auroc:.4f} kNN10={knn_primary_auroc:.4f} KDE={kde_auroc:.4f}")
    results[arm] = arm_results

Path("AlphaLadder/reliability_scorers_results.json").write_text(json.dumps({"alphas": ALPHAS, "results": results}, indent=2))
print("done")
