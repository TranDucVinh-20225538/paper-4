"""
Group B execution -- Mahalanobis, kNN, Cosine-to-centroid, KDE.

Authorized by Vinh, 2026-08-05, per GroupBExecution/execution_contract.md:
"Mahalanobis-family execution: AUTHORIZED (sau khi tạo Group B Execution
Contract)." Runs exactly the four scorers the contract locks -- fit on
ReferenceTrainEmbedding (z_train, y_train), evaluate on ReferenceEmbedding
(z_id = ISIC-test, z_ood = PAD-UFES) -- and only those four. Energy, ViM,
and Relative Mahalanobis are explicitly out of scope (03c_groupB_readiness.md)
and are not touched by this script.

Reuse discipline:
  - Mahalanobis fit/score + FPR@95 are genuine imports, unmodified, from
    CSG-SKin/src/utils/ood_metrics.py (compute_mahalanobis_params_from_arrays,
    mahalanobis_min_squared_distances, fpr_at_95_tpr) -- the same functions
    paper-3/scripts/extract_auroc_e2.py itself imports.
  - Cosine-to-centroid and kNN reproduce, formula-for-formula, paper-3/
    scripts/extract_auroc_e2.py::cosine_centroid_scores and ::compute_knn_scores.
    Not imported directly: that file's own module-level imports pull in
    paper-3-local `_repo_paths` and Lightning checkpoint-loading classes
    (BaselineResNet50/CSGLiteLightning/EffB3SingleLightning) that have
    nothing to do with these two pure-numpy formulas and would couple Paper4
    to paper-3's own script structure for no reason. Reproduced verbatim
    instead, cited at each function.
  - KDE reproduces paper-3/scripts/extract_e2_8_extra_scorers.py::
    density_kde_scores verbatim, same reasoning.

No cross-arm comparison (there is no second arm yet). No Tier-3 language.
Output: one JSON per scorer in GroupBExecution/, per the execution contract's
table -- NOT frozen (matches BaselineReport's precedent: Tier 1/2 analysis
outputs stay unfrozen until Vinh reviews them, unlike primary embedding
artifacts, which freeze immediately on Gate PASS).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, NearestNeighbors

CSG_ROOT = data_roots.root("csg")
sys.path.insert(0, str(CSG_ROOT))
from src.utils import ood_metrics  # noqa: E402 -- genuine import, unmodified

import data_roots

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings"
TRAIN_DIR = ASSETS / "ReferenceTrainEmbedding"
EVAL_DIR = ASSETS / "ReferenceEmbedding"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "GroupBExecution"
CONTRACT_PATH = OUTPUT_DIR / "execution_contract.md"

NUM_CLASSES = 8
REG_EPS = 1e-5  # paper-3/scripts/extract_auroc_e2.py's own deliberate choice, not CSG-SKin's 1e-3 default
K_VALUES = (1, 10, 50)  # paper-3/scripts/extract_auroc_e2.py::K_VALUES
PRIMARY_K = 10


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def _verify_and_load(artifact_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Load embeddings + metadata, re-checking SHA256 against the artifact's own frozen manifest
    before use -- a frozen artifact is trusted only after independent re-verification here, not
    assumed from freeze-time."""
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    for fname in ("embeddings.npy", "metadata.csv"):
        recorded = manifest["files"][fname]["sha256"]
        actual = _sha256_file(artifact_dir / fname)
        if actual != recorded:
            raise RuntimeError(
                f"{artifact_dir / fname}: SHA256 mismatch (manifest={recorded[:16]}..., "
                f"actual={actual[:16]}...) -- frozen artifact integrity check failed."
            )
    embeddings = np.load(artifact_dir / "embeddings.npy")
    metadata = pd.read_csv(artifact_dir / "metadata.csv")
    assert embeddings.shape[0] == len(metadata), "embeddings/metadata row count mismatch"
    return embeddings, metadata, manifest


def cosine_centroid_scores(z: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Verbatim reproduction of paper-3/scripts/extract_auroc_e2.py::cosine_centroid_scores.
    score(z) = min_c [1 - cosine_similarity(z, mean_c)]; `means` is the SAME per-class means
    array already fit for Mahalanobis -- isolates one variable (drops covariance/precision)."""
    z_unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    means_unit = means / np.linalg.norm(means, axis=1, keepdims=True)
    cosine_sim = z_unit @ means_unit.T
    return (1.0 - cosine_sim).min(axis=1)


def compute_knn_scores(z_train: np.ndarray, z_id: np.ndarray, z_ood: np.ndarray,
                        k_values: tuple[int, ...]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Verbatim reproduction of paper-3/scripts/extract_auroc_e2.py::compute_knn_scores.
    Pooled k-NN distance (Sun et al. 2022-style), all classes pooled, no covariance."""
    k_max = max(k_values)
    index = NearestNeighbors(n_neighbors=k_max).fit(z_train)
    dist_id, _ = index.kneighbors(z_id)
    dist_ood, _ = index.kneighbors(z_ood)
    return {k: (dist_id[:, k - 1], dist_ood[:, k - 1]) for k in k_values}


def density_kde_scores(z_train: np.ndarray, y_train: np.ndarray, z_query: np.ndarray) -> np.ndarray:
    """Verbatim reproduction of paper-3/scripts/extract_e2_8_extra_scorers.py::density_kde_scores.
    Per-class Gaussian-kernel KDE, Scott's-rule bandwidth, score = -max_c[log p_c(z)]."""
    d = z_train.shape[1]
    log_probs = np.empty((z_query.shape[0], NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        z_c = z_train[y_train == c]
        n_c = len(z_c)
        sigma_bar = z_c.std(axis=0, ddof=1).mean()
        bandwidth = max(sigma_bar * (n_c ** (-1.0 / (d + 4))), 1e-6)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(z_c)
        log_probs[:, c] = kde.score_samples(z_query)
    return -log_probs.max(axis=1)


def auroc_fpr95(s_id: np.ndarray, s_ood: np.ndarray) -> tuple[float, float]:
    """y=1 for OOD, higher score = more OOD-like -- same convention as every prior AUROC
    computation in this project and in Papers 1-3."""
    y = np.concatenate([np.zeros(len(s_id), dtype=np.int64), np.ones(len(s_ood), dtype=np.int64)])
    scores = np.concatenate([s_id, s_ood])
    auroc = float(roc_auc_score(y, scores))
    fpr95 = ood_metrics.fpr_at_95_tpr(y, scores)
    return auroc, fpr95


def main() -> None:
    z_train_all, meta_train, manifest_train = _verify_and_load(TRAIN_DIR)
    y_train = meta_train["label_idx"].to_numpy().astype(np.int64)
    n_train = len(meta_train)
    print(f"[groupB] ReferenceTrainEmbedding verified: n={n_train}, sha256 OK")

    z_eval_all, meta_eval, manifest_eval = _verify_and_load(EVAL_DIR)
    id_mask = (meta_eval["domain"] == "isic").to_numpy()
    ood_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()
    z_id, z_ood = z_eval_all[id_mask], z_eval_all[ood_mask]
    n_id, n_ood = int(id_mask.sum()), int(ood_mask.sum())
    print(f"[groupB] ReferenceEmbedding verified: n_id={n_id}, n_ood={n_ood}, sha256 OK")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csg_commit = _git_commit(CSG_ROOT)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def provenance(scorer: str, hyperparameters: dict, extra: dict) -> dict:
        return {
            "scorer": scorer,
            "protocol_step": "Stage 3.2 Group B (Mahalanobis-family)",
            "execution_contract": str(CONTRACT_PATH),
            "produced_by_script": str(Path(__file__).resolve()),
            "fit_artifact": {"path": str(TRAIN_DIR), "manifest_status": manifest_train["status"], "n": n_train},
            "eval_artifact": {"path": str(EVAL_DIR), "manifest_status": manifest_eval["status"],
                               "n_id": n_id, "n_ood": n_ood},
            "csg_skin_commit": csg_commit,
            "hyperparameters": hyperparameters,
            "created_at": now,
            "tier": "Tier 1 (Measurement) -- single-arm raw scores, no cross-arm comparison, no claim test",
            "not_computed_here": "No Energy, no ViM, no Relative Mahalanobis, no cross-arm comparison, "
                                  "no interpretation beyond the raw AUROC/FPR95 numbers themselves.",
            **extra,
        }

    # ---------------- Mahalanobis ----------------
    means, precision = ood_metrics.compute_mahalanobis_params_from_arrays(
        z_train_all, y_train, num_classes=NUM_CLASSES, reg_eps=REG_EPS
    )
    s_id = ood_metrics.mahalanobis_min_squared_distances(z_id, means, precision)
    s_ood = ood_metrics.mahalanobis_min_squared_distances(z_ood, means, precision)
    maha_auroc, maha_fpr95 = auroc_fpr95(s_id, s_ood)
    maha_result = provenance("mahalanobis", {"num_classes": NUM_CLASSES, "reg_eps": REG_EPS},
                              {"auroc": maha_auroc, "fpr95": maha_fpr95})
    (OUTPUT_DIR / "maha_results.json").write_text(json.dumps(maha_result, indent=2))
    print(f"[groupB] Mahalanobis  AUROC={maha_auroc:.4f}  FPR95={maha_fpr95:.4f}")

    # ---------------- Cosine-to-centroid (reuses Mahalanobis's means) ----------------
    cos_id = cosine_centroid_scores(z_id, means)
    cos_ood = cosine_centroid_scores(z_ood, means)
    cos_auroc, cos_fpr95 = auroc_fpr95(cos_id, cos_ood)
    cos_result = provenance("cosine_to_centroid", {"means_source": "reused from the Mahalanobis fit above, same call, not refit"},
                             {"auroc": cos_auroc, "fpr95": cos_fpr95})
    (OUTPUT_DIR / "cosine_results.json").write_text(json.dumps(cos_result, indent=2))
    print(f"[groupB] Cosine       AUROC={cos_auroc:.4f}  FPR95={cos_fpr95:.4f}")

    # ---------------- kNN ----------------
    knn_scores = compute_knn_scores(z_train_all, z_id, z_ood, K_VALUES)
    knn_per_k = {}
    for k, (d_id, d_ood) in knn_scores.items():
        auroc_k, fpr95_k = auroc_fpr95(d_id, d_ood)
        knn_per_k[str(k)] = {"auroc": auroc_k, "fpr95": fpr95_k}
        print(f"[groupB] kNN k={k:<3d} AUROC={auroc_k:.4f}  FPR95={fpr95_k:.4f}")
    knn_result = provenance(
        "knn",
        {"k_values": list(K_VALUES), "primary_k": PRIMARY_K, "reference_pool": "z_train, pooled across all classes"},
        {"results_by_k": knn_per_k, "primary_result": knn_per_k[str(PRIMARY_K)]},
    )
    (OUTPUT_DIR / "knn_results.json").write_text(json.dumps(knn_result, indent=2))

    # ---------------- KDE ----------------
    kde_id = density_kde_scores(z_train_all, y_train, z_id)
    kde_ood = density_kde_scores(z_train_all, y_train, z_ood)
    kde_auroc, kde_fpr95 = auroc_fpr95(kde_id, kde_ood)
    kde_result = provenance("kde", {"kernel": "gaussian", "bandwidth_rule": "Scott's rule per class, floor 1e-6"},
                             {"auroc": kde_auroc, "fpr95": kde_fpr95})
    (OUTPUT_DIR / "kde_results.json").write_text(json.dumps(kde_result, indent=2))
    print(f"[groupB] KDE          AUROC={kde_auroc:.4f}  FPR95={kde_fpr95:.4f}")

    print("\n[groupB] DONE. Four output artifacts written to GroupBExecution/ -- NOT frozen "
          "(pending Vinh's review, same precedent as BaselineReport/). No Energy, no ViM, no "
          "Relative Mahalanobis, no cross-arm comparison computed.")


if __name__ == "__main__":
    main()
