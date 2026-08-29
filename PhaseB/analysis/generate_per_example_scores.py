"""
Per-example scores for every configuration, the input Phase 1 of the plan needs.

All four scorers. Mahalanobis is the genuine CSG-SKin implementation, imported
rather than reimplemented. Cosine and k-NN follow groupB_mahalanobis_family.py
exactly, including the detail that cosine reuses the class means fit for
Mahalanobis.

KDE is computed directly instead of through sklearn's KernelDensity, which does
not compute reliably at d = 1024: its kd_tree and ball_tree paths disagree with
each other, and both return values pinned near the theoretical ceiling for
points that are not near any training point. The direct form here matches
sklearn exactly in one dimension, on synthetic 1024-dimensional data, and on
queries that coincide with a training point, and it is the same formula --
Gaussian kernel, Scott's rule per class, floor 1e-6. See docs/kde_reliability.md.

Mahalanobis is the genuine CSG-SKin implementation, imported rather than
reimplemented, so the numbers are the ones the study already reported. Cosine
and k-NN follow groupB_mahalanobis_family.py exactly, including the detail that
cosine reuses the class means fit for Mahalanobis.

Fit population is the adapted training set of 16,211; evaluation is the adapted
5,067 held-out ISIC against 2,298 PAD-UFES. Getting those two the wrong way
round gives AUROC near 0.99997 and reproduces nothing.

    python3 PhaseB/analysis/generate_per_example_scores.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "/Users/cubo/Research/CSG-SKin")
from src.utils.ood_metrics import (  # noqa: E402
    compute_mahalanobis_params_from_arrays,
    mahalanobis_min_squared_distances,
)

NUM_CLASSES, REG_EPS, PRIMARY_K = 8, 1e-5, 10
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = {"Intervention": "adapter", "Conventional": "adapter", "Adaptation": "partialFT_adapter"}
SEEDS = (42, 52, 62, 72, 82)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def load(root: Path):
    A = root / "assets/reference_embeddings"
    Ztr = np.load(A / "ReferenceTrainEmbedding/embeddings.npy").astype(np.float32)
    Zev = np.load(A / "ReferenceEmbedding/embeddings.npy").astype(np.float32)
    ytr = pd.read_csv(A / "ReferenceTrainEmbedding/metadata.csv").label_idx.to_numpy()
    dom = pd.read_csv(A / "ReferenceEmbedding/metadata.csv").domain.to_numpy()
    return Ztr, Zev, ytr, (dom != "isic").astype(int)


def displacement(root: Path, arm: str, seed: int):
    sd = torch.load(root / f"{arm}Arm_Canonical/{ARMS[arm]}_seed{seed}.pt",
                    map_location="cpu", weights_only=False)["adapter"]
    W1, b1 = sd["down.weight"].numpy(), sd["down.bias"].numpy()
    W2, b2 = sd["up.weight"].numpy(), sd["up.bias"].numpy()
    return lambda Z: (gelu(Z @ W1.T + b1) @ W2.T + b2).astype(np.float32)


def kde_scores(Ztr_a, ytr, Zev_a):
    """Per-class Gaussian KDE, Scott's rule, score = -max_c log p_c(z).

    Squared distances via the BLAS expansion rather than cdist: 26x faster and
    agreeing to 2e-11 here, which matters because the sklearn path this replaces
    took sixteen minutes per configuration."""
    d = Ztr_a.shape[1]
    logp = np.empty((len(Zev_a), NUM_CLASSES))
    Q = Zev_a.astype(np.float64)
    qq = (Q * Q).sum(1)[:, None]
    for c in range(NUM_CLASSES):
        zc = Ztr_a[ytr == c]
        h = max(zc.std(axis=0, ddof=1).mean() * (len(zc) ** (-1.0 / (d + 4))), 1e-6)
        C = zc.astype(np.float64)
        D2 = qq + (C * C).sum(1)[None, :] - 2 * (Q @ C.T)
        logp[:, c] = (logsumexp(-D2 / (2 * h ** 2), axis=1) - np.log(len(zc))
                      - 0.5 * d * np.log(2 * np.pi) - d * np.log(h))
    return -logp.max(axis=1)


def score_all(Ztr_a, ytr, Zev_a):
    means, precision = compute_mahalanobis_params_from_arrays(Ztr_a, ytr, NUM_CLASSES, REG_EPS)
    maha = mahalanobis_min_squared_distances(Zev_a, means, precision)

    zu = Zev_a / np.linalg.norm(Zev_a, axis=1, keepdims=True)
    mu = means / np.linalg.norm(means, axis=1, keepdims=True)
    cosine = (1.0 - zu @ mu.T).min(axis=1)

    nn = NearestNeighbors(n_neighbors=PRIMARY_K).fit(Ztr_a)
    dist, _ = nn.kneighbors(Zev_a)
    knn = dist[:, PRIMARY_K - 1]

    kde = kde_scores(Ztr_a, ytr, Zev_a)
    return {"maha": maha, "cosine": cosine, "knn10": knn, "kde": kde}, means, precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--out", type=Path, default=Path("PhaseB/outputs/per_example"))
    args = ap.parse_args()

    Ztr, Zev, ytr, is_ood = load(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []

    for arm in ARMS:
        alphas = ALPHAS if arm != "Adaptation" else (1.0,)
        for seed in SEEDS:
            dz = displacement(args.root, arm, seed)
            Dtr, Dev = dz(Ztr), dz(Zev)
            for a in alphas:
                scores, means, precision = score_all(Ztr + a * Dtr, ytr, Zev + a * Dev)
                tag = f"{arm}_a{a}_s{seed}"
                np.savez_compressed(args.out / f"{tag}.npz", is_ood=is_ood, **scores)
                row = {"arm": arm, "alpha": a, "seed": seed}
                for name, s in scores.items():
                    row[f"{name}_auroc"] = roc_auc_score(is_ood, s)
                rows.append(row)
                print(f"  {tag:<30} " + "  ".join(f"{k} {v:.6f}" for k, v in row.items()
                                                  if k.endswith("auroc")), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "auroc_from_per_example.csv", index=False)
    print(f"\n{len(df)} configurations, per-example scores in {args.out}")


if __name__ == "__main__":
    main()
