"""
Regenerates per-example scores from the retained artifacts.

The plan marks RQ1 items 1.3 and 1.4 mandatory: the distribution of ||y||^2 with
Mahalanobis bounds at median/P95/P99/max, and score margins with
neighbour-distance ratios for a susceptibility ECDF. None of that survives in
the stored outputs, which keep aggregate AUROC only -- the manuscript says as
much. But nothing new has to be run: the frozen reference embeddings and the
trained adapters are enough to reproduce the scoring exactly, so this stays
inside the no-new-data constraint.

Protocol recovered from the artifacts, and worth stating because getting it
wrong silently inflates every number: the Mahalanobis parameters are fit on the
adapted TRAINING population (16,211 ISIC), and scoring runs on the adapted
evaluation population (5,067 held-out ISIC against 2,298 PAD-UFES). Fitting and
scoring on the same 5,067 gives AUROC near 0.99997 and reproduces nothing.

Verified against the stored values for the Intervention arm, seed 42:

    alpha 0.00 -> 0.998385   (matches reference_baseline maha_auroc)
    alpha 0.25 -> 0.998692   (stored 0.998693)
    alpha 1.00 -> 0.998826   (stored 0.998826)

    python3 PhaseB/analysis/rescore_per_example.py --arm Intervention --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

TRAIN = "assets/reference_embeddings/ReferenceTrainEmbedding"
EVAL = "assets/reference_embeddings/ReferenceEmbedding"
RIDGE = 1e-5


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def displacement(root: Path, arm: str, seed: int):
    """Delta z = W2 GELU(W1 z) + b2, from the trained adapter."""
    name = "partialFT_adapter" if arm == "Adaptation" else "adapter"
    sd = torch.load(root / f"{arm}Arm_Canonical/{name}_seed{seed}.pt",
                    map_location="cpu", weights_only=False)["adapter"]
    W1, b1 = sd["down.weight"].numpy(), sd["down.bias"].numpy()
    W2, b2 = sd["up.weight"].numpy(), sd["up.bias"].numpy()
    return lambda Z: gelu(Z @ W1.T + b1) @ W2.T + b2


def mahalanobis_scores(Ztr, ytr, Zev, ridge=RIDGE):
    """Per-example minimum squared Mahalanobis distance, parameters fit on Ztr."""
    classes = np.unique(ytr)
    means = np.stack([Ztr[ytr == c].mean(0) for c in classes]).astype(np.float64)
    scatter = np.zeros((Ztr.shape[1],) * 2)
    for m, c in zip(means, classes):
        D = Ztr[ytr == c].astype(np.float64) - m
        scatter += D.T @ D
    P = np.linalg.inv(scatter / len(Ztr) + ridge * np.eye(Ztr.shape[1]))
    X = Zev.astype(np.float64)
    # (X-m) @ P * (X-m) summed: BLAS gemm rather than a three-operand einsum
    return np.stack([(((X - m) @ P) * (X - m)).sum(1) for m in means]).min(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PhaseB"))
    ap.add_argument("--arm", default="Intervention")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--out", type=Path, default=None, help="write per-example scores as .npz")
    args = ap.parse_args()

    Ztr = np.load(args.root / TRAIN / "embeddings.npy").astype(np.float32)
    Zev = np.load(args.root / EVAL / "embeddings.npy").astype(np.float32)
    ytr = pd.read_csv(args.root / TRAIN / "metadata.csv").label_idx.to_numpy()
    dom = pd.read_csv(args.root / EVAL / "metadata.csv").domain.to_numpy()
    is_ood = (dom != "isic").astype(int)

    dz = displacement(args.root, args.arm, args.seed)
    Dtr, Dev = dz(Ztr), dz(Zev)

    print(f"{args.arm} seed {args.seed}: fit on {len(Ztr)}, score {(~is_ood.astype(bool)).sum()} "
          f"ID against {is_ood.sum()} OOD")
    out = {}
    for a in args.alphas:
        s = mahalanobis_scores(Ztr + a * Dtr, ytr, Zev + a * Dev)
        out[f"alpha{a}"] = s
        print(f"  alpha {a:.2f}   AUROC {roc_auc_score(is_ood, s):.6f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, is_ood=is_ood, **out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
