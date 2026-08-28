"""Fix: recompute Adaptation arm's reliability scorers using ISIC-only training population
(bug: previous run used load_training_population()'s combined ISIC+PAD-UFES set, where
label=-1 rows corrupt compute_mahalanobis_params_from_arrays via negative indexing)."""

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
from groupB_mahalanobis_family import cosine_centroid_scores, compute_knn_scores, density_kde_scores, auroc_fpr95, K_VALUES, PRIMARY_K, NUM_CLASSES, REG_EPS
from src.utils import ood_metrics

OUTPUT_DIR = Path("AdaptationArm_Canonical")
SEEDS = [42, 52, 62, 72, 82]

z_train_isic = np.load(TRAIN_DIR / "embeddings.npy").astype(np.float32)  # ISIC-only, no PAD-UFES
y_train_isic = pd.read_csv(TRAIN_DIR / "metadata.csv")["label_idx"].to_numpy().astype(np.int64)
z_eval, meta_eval = load_eval_population()
id_mask = (meta_eval["domain"] == "isic").to_numpy()
ood_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()

def score(z_train_a, y_train, z_eval_a):
    z_id, z_ood = z_eval_a[id_mask], z_eval_a[ood_mask]
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(z_train_a, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS)
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
    maha_auroc, maha_fpr95 = auroc_fpr95(s_id, s_ood)
    cos_auroc, cos_fpr95 = auroc_fpr95(cosine_centroid_scores(z_id, means), cosine_centroid_scores(z_ood, means))
    knn = compute_knn_scores(z_train_a, z_id, z_ood, K_VALUES)
    knn_auroc, knn_fpr95 = auroc_fpr95(*knn[PRIMARY_K])
    kde_auroc, kde_fpr95 = auroc_fpr95(density_kde_scores(z_train_a, y_train, z_id), density_kde_scores(z_train_a, y_train, z_ood))
    return {"maha_auroc": maha_auroc, "maha_fpr95": maha_fpr95, "cosine_auroc": cos_auroc, "cosine_fpr95": cos_fpr95,
            "knn_k10_auroc": knn_auroc, "knn_k10_fpr95": knn_fpr95, "kde_auroc": kde_auroc, "kde_fpr95": kde_fpr95}

manifest = json.loads((OUTPUT_DIR / "manifest.json").read_text())

# Rung 1: linear-probe -- z unchanged, fixed scores (identical across seeds, as before -- correctly so)
fixed_scores = score(z_train_isic, y_train_isic, z_eval)
print("linear-probe (corrected):", {k: round(v, 4) for k, v in fixed_scores.items()})
for r in manifest["rungs"]["linear-probe"]:
    r.update(fixed_scores)

# Rung 2: partial-FT -- reload saved adapters, re-transform, refit correctly
for i, seed in enumerate(SEEDS):
    ckpt = torch.load(OUTPUT_DIR / f"partialFT_adapter_seed{seed}.pt", map_location="cpu", weights_only=True)
    adapter = BottleneckAdapter(r=8)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()
    with torch.no_grad():
        z_train_a = adapter(torch.tensor(z_train_isic, dtype=torch.float32)).numpy()
        z_eval_a = adapter(torch.tensor(z_eval, dtype=torch.float32)).numpy()
    s = score(z_train_a, y_train_isic, z_eval_a)
    manifest["rungs"]["partial-FT"][i].update(s)
    print(f"partial-FT seed={seed} (corrected): Maha={s['maha_auroc']:.4f} Cos={s['cosine_auroc']:.4f} kNN10={s['knn_k10_auroc']:.4f} KDE={s['kde_auroc']:.4f}")

manifest["bugfix_note"] = ("Reliability scorers originally fit on load_training_population()'s combined "
                            "ISIC+PAD-UFES set (label=-1 for PAD-UFES), which corrupts "
                            "compute_mahalanobis_params_from_arrays via negative indexing (label=-1 -> "
                            "class_means[-1], i.e. class 7's mean, instead of exclusion). Refit on "
                            "ISIC-only training population (TRAIN_DIR, matching stage4_reliability_scorers.py's "
                            "already-correct approach). Gate 0 and geometry metrics were unaffected -- not recomputed.")

manifest_path = OUTPUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
print("\nmanifest updated (not yet frozen)")
