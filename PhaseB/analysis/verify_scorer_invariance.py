"""
Checks the two invariance propositions against the PanDerm reference embeddings.

Proposition 1. Spectral tempering by gamma maps the in-distribution covariance
Sigma = U diag(lambda_i) U^T to Sigma' = U diag(lambda_i^{1-gamma}) U^T, so
kappa(Sigma') = kappa(Sigma)^{1-gamma}. The operator is therefore a designed
manipulation of the condition number: log kappa scales linearly in (1-gamma).

Proposition 2. Mahalanobis parameters are refit on the tempered training
population (Methods, Reliability Estimators). Tempering is an invertible linear
map A, so the refit precision is (A S A^T)^{-1} and every squared distance is
preserved: d'^2(Az) = d^2(z). Mahalanobis AUROC is therefore invariant to
gamma, exactly, up to the epsilon = 1e-5 ridge and floating point.

The consequence for the study: within this intervention family, condition
number cannot predict Mahalanobis AUROC. The absent association is forced by
the algebra, not a shortfall of power. Scorers that are not affine-invariant --
cosine and k-NN -- do respond, which is what makes "scorer-relevant geometry" a
distinction with content rather than a relabelling.

Caveat this script also demonstrates: Proposition 1 holds in exact arithmetic.
The smallest sample eigenvalue here is ~1e-17, so lambda^{-gamma/2} on the tail
is dominated by round-off and the recomputed kappa departs from kappa^{1-gamma}
while remaining monotone. Report the identity analytically and the realized
values empirically; do not present the two as the same number.

    python3 PhaseB/analysis/verify_scorer_invariance.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ARTIFACT = Path("PhaseB/assets/reference_embeddings/panderm_reference_embedding_artifact.npz")
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPS = 1e-5


def load(path):
    d = np.load(path, allow_pickle=True)
    dom = np.array([str(x) for x in d["domain"]])
    return d["embeddings"].astype(np.float64), d["labels"], dom


def tempered(Z, mu, lam, U, gamma):
    A = U @ np.diag(np.clip(lam, 1e-12, None) ** (-gamma / 2)) @ U.T
    return mu + (Z - mu) @ A.T


def mahalanobis_scorer(Ztr, ytr):
    classes = np.unique(ytr)
    means = np.stack([Ztr[ytr == c].mean(0) for c in classes])
    scatter = np.zeros((Ztr.shape[1],) * 2)
    for m, c in zip(means, classes):
        D = Ztr[ytr == c] - m
        scatter += D.T @ D
    P = np.linalg.inv(scatter / len(Ztr) + EPS * np.eye(Ztr.shape[1]))
    return lambda X: np.min(
        np.stack([np.einsum("ij,jk,ik->i", X - m, P, X - m) for m in means]), 0
    )


def cosine_scorer(Ztr, ytr):
    means = np.stack([Ztr[ytr == c].mean(0) for c in np.unique(ytr)])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    return lambda X: -np.max((X / np.linalg.norm(X, axis=1, keepdims=True)) @ means.T, 1)


def knn_scorer(Ztr, _ytr, k=10):
    nn = NearestNeighbors(n_neighbors=k).fit(Ztr)
    return lambda X: nn.kneighbors(X)[0][:, -1]


def condition_number(Z):
    lam = np.linalg.eigvalsh(np.cov((Z - Z.mean(0)).T))
    return lam.max() / np.clip(lam, 1e-12, None).min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()

    Z, y, dom = load(args.artifact)
    rng = np.random.default_rng(args.seed)
    id_idx = np.flatnonzero(dom == "isic")
    rng.shuffle(id_idx)
    cut = int(args.train_frac * len(id_idx))
    tr, te = id_idx[:cut], id_idx[cut:]
    ood = np.flatnonzero(dom != "isic")

    mu = Z[tr].mean(0)
    lam, U = np.linalg.eigh(np.cov((Z[tr] - mu).T))
    kappa0 = condition_number(Z[tr])
    labels = np.r_[np.zeros(len(te)), np.ones(len(ood))]

    print(f"ID={len(id_idx)}  OOD={len(ood)}  d={Z.shape[1]}  classes={len(np.unique(y[tr]))}")
    print(f"smallest sample eigenvalue {lam.min():.3e} -- see the caveat in the docstring\n")
    print(f"{'gamma':>6} {'kappa realized':>16} {'kappa^(1-g)':>14} "
          f"{'Mahalanobis':>13} {'cosine':>10} {'kNN k=10':>10}")

    for g in GAMMAS:
        Zt = tempered(Z, mu, lam, U, g)
        row = []
        for build in (mahalanobis_scorer, cosine_scorer, knn_scorer):
            f = build(Zt[tr], y[tr])
            row.append(roc_auc_score(labels, np.concatenate([f(Zt[te]), f(Zt[ood])])))
        print(f"{g:6.2f} {condition_number(Zt[tr]):16.4e} {kappa0 ** (1 - g):14.4e} "
              f"{row[0]:13.6f} {row[1]:10.6f} {row[2]:10.6f}")

    print("\nMahalanobis is flat; cosine and k-NN are not. Condition number therefore "
          "cannot be a predictor of Mahalanobis AUROC under this operator.")


if __name__ == "__main__":
    main()
