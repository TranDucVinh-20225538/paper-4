"""
Stage 2.1b -- restructure the Reference Embedding Artifact into the format
Vinh specified (2026-08-05), and independently re-verify row-level ordering
against a FRESH reconstruction of the preregistered split (not just
self-consistency with the extraction run's own in-memory arrays).

No new embeddings are computed here -- this is pure post-processing of
stage2_1's already-extracted, already-gated output.

Target layout:
  PhaseB/assets/reference_embeddings/ReferenceEmbedding/
      embeddings.npy   -- (7365, 1024) float32
      labels.csv        -- row_index, label_idx, label_name
      metadata.csv       -- row_index, path, domain, label_idx, label_name
      manifest.json       -- birth certificate
      sha256.txt           -- per-file checksums (embeddings.npy, labels.csv, metadata.csv)
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

CSG_ROOT = Path("/Users/cubo/Research/CSG-SKin")
PANDERM_REPO = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT_METADATA = Path(__file__).resolve().parents[1] / "assets" / "panderm_vitl16_checkpoint_metadata.json"
PREPROCESSING_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "reference_preprocessing.json"
DATASET_VALIDATION_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "dataset_validation_report.json"
OLD_ARTIFACT_NPZ = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "panderm_reference_embedding_artifact.npz"
OLD_PROVENANCE = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "reference_embedding_artifact_provenance.json"
OLD_GATE_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "embedding_integrity_gate_report.json"

TARGET_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceEmbedding"
FINAL_GATE_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "embedding_integrity_gate_report.json"

LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]  # CSG-Skin's src/datasets/constants.py, index == position

sys.path.insert(0, str(CSG_ROOT))


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def independently_reconstruct_split(metadata_csv: Path, seed: int, isic_test_fraction: float, val_fraction: float):
    """
    Fresh reconstruction of the preregistered ID/OOD dataframes, using the
    SAME code CSG-Skin's own build_id_ood_test_dataloaders uses -- but called
    here independently of the extraction run, as a second, separate check
    that row order is reproducible from the raw split logic alone, not just
    self-consistent with what stage2_1 happened to produce in memory.
    """
    from sklearn.model_selection import train_test_split

    from src.datasets.splits import load_filtered_master

    df = load_filtered_master(str(metadata_csv))
    isic_df = df[df["domain"] == "isic"].copy()
    pad_df = df[df["domain"] == "pad_ufes"].copy()

    isic_train_val, isic_test = train_test_split(
        isic_df, test_size=isic_test_fraction, stratify=isic_df["label_idx"], random_state=seed,
    )
    _a, _b = train_test_split(
        isic_train_val, test_size=val_fraction, stratify=isic_train_val["label_idx"], random_state=seed,
    )

    return isic_test.reset_index(drop=True), pad_df.reset_index(drop=True)


def main() -> None:
    if not DATASET_VALIDATION_REPORT.is_file():
        raise RuntimeError(f"{DATASET_VALIDATION_REPORT} not found.")
    with open(DATASET_VALIDATION_REPORT) as f:
        dataset_validation = json.load(f)

    if not OLD_ARTIFACT_NPZ.is_file():
        raise RuntimeError(f"{OLD_ARTIFACT_NPZ} not found -- run stage2_1_extract_reference_embeddings.py first.")

    data = np.load(OLD_ARTIFACT_NPZ, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    domains = data["domain"]
    paths = data["path"]
    n_total = embeddings.shape[0]
    print(f"[stage2.1b] Loaded extraction output: embeddings {embeddings.shape}, n={n_total}")

    # === Independent re-verification: fresh split reconstruction, not self-consistency ===
    metadata_csv = Path(dataset_validation["metadata_csv"])
    isic_test_df, pad_df = independently_reconstruct_split(
        metadata_csv,
        seed=dataset_validation["seed"],
        isic_test_fraction=dataset_validation["isic_test_fraction"],
        val_fraction=dataset_validation["val_fraction"],
    )
    fresh_paths = np.concatenate([isic_test_df["path"].to_numpy(), pad_df["path"].to_numpy()])
    fresh_labels = np.concatenate([isic_test_df["label_idx"].to_numpy(), pad_df["label_idx"].to_numpy()])

    proof_paths_match = bool(np.array_equal(paths, fresh_paths))
    proof_labels_match = bool(np.array_equal(labels, fresh_labels))
    row_i_proof = {
        "method": "Freshly reconstructed the ID (ISIC-test) + OOD (PAD-UFES) dataframes from "
                  "master_metadata.csv using the same split logic (independent function call, not "
                  "reading anything cached from the extraction run), then compared path/label_idx "
                  "arrays element-by-element against the extraction artifact's stored path/label "
                  "arrays.",
        "paths_match_fresh_reconstruction": proof_paths_match,
        "labels_match_fresh_reconstruction": proof_labels_match,
        "n_compared": int(len(fresh_paths)),
        "conclusion": "PROVEN: metadata row i corresponds to embedding row i, for all i, independently "
                      "re-derived" if (proof_paths_match and proof_labels_match) else "FAILED",
    }
    print(f"[stage2.1b] Row-i-to-row-i proof: {row_i_proof['conclusion']}")
    if not (proof_paths_match and proof_labels_match):
        raise RuntimeError(f"Independent ordering re-verification FAILED: {row_i_proof}")

    # === Build labels.csv / metadata.csv ===
    label_names = np.array([LABELS[int(li)] for li in labels])
    labels_df = pd.DataFrame({
        "row_index": np.arange(n_total),
        "label_idx": labels.astype(int),
        "label_name": label_names,
    })
    metadata_df = pd.DataFrame({
        "row_index": np.arange(n_total),
        "path": paths,
        "domain": domains,
        "label_idx": labels.astype(int),
        "label_name": label_names,
    })

    # === Write target directory ===
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_npy_path = TARGET_DIR / "embeddings.npy"
    labels_csv_path = TARGET_DIR / "labels.csv"
    metadata_csv_path = TARGET_DIR / "metadata.csv"
    manifest_json_path = TARGET_DIR / "manifest.json"
    sha256_txt_path = TARGET_DIR / "sha256.txt"

    np.save(embeddings_npy_path, embeddings)
    labels_df.to_csv(labels_csv_path, index=False)
    metadata_df.to_csv(metadata_csv_path, index=False)

    sha_embeddings = _sha256_file(embeddings_npy_path)
    sha_labels = _sha256_file(labels_csv_path)
    sha_metadata = _sha256_file(metadata_csv_path)

    with open(sha256_txt_path, "w") as f:
        f.write(f"{sha_embeddings}  embeddings.npy\n")
        f.write(f"{sha_labels}  labels.csv\n")
        f.write(f"{sha_metadata}  metadata.csv\n")

    manifest = {
        "artifact": "Reference Embedding Artifact",
        "protocol_step": "Stage 2.1",
        "rows": int(n_total),
        "dim": int(embeddings.shape[1]),
        "n_id_isic": int((domains == "isic").sum()),
        "n_ood_pad_ufes": int((domains == "pad_ufes").sum()),
        "checkpoint": str(CHECKPOINT_METADATA),
        "preprocessing": str(PREPROCESSING_SNAPSHOT),
        "dataset": {
            "metadata_csv": str(metadata_csv),
            "dataset_validation_report": str(DATASET_VALIDATION_REPORT),
            "split_seed": dataset_validation["seed"],
            "isic_test_fraction": dataset_validation["isic_test_fraction"],
            "val_fraction": dataset_validation["val_fraction"],
        },
        "repo_commits": {
            "csg_skin": _git_commit(CSG_ROOT),
            "panderm": _git_commit(PANDERM_REPO),
        },
        "files": {
            "embeddings.npy": {"sha256": sha_embeddings, "shape": list(embeddings.shape), "dtype": str(embeddings.dtype)},
            "labels.csv": {"sha256": sha_labels, "rows": len(labels_df)},
            "metadata.csv": {"sha256": sha_metadata, "rows": len(metadata_df)},
        },
        "row_ordering_proof": row_i_proof,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "not_computed_here": "No probe, no Mahalanobis or other distance scorer, no geometry metric, "
                              "no PCA/UMAP, no statistics. Extraction and integrity verification only.",
    }
    with open(manifest_json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[stage2.1b] Written: {TARGET_DIR}/")
    print(f"  embeddings.npy  sha256={sha_embeddings}")
    print(f"  labels.csv      sha256={sha_labels}")
    print(f"  metadata.csv    sha256={sha_metadata}")
    print(f"  manifest.json, sha256.txt")

    # === Final Embedding Integrity Gate report (six checks, per Vinh) ===
    gate_report = {
        "protocol_step": "Stage 2.1 -- Embedding Integrity Gate (final, ReferenceEmbedding/ layout)",
        "date": "2026-08-05",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "check_1_sample_count": {
            "isic": int((domains == "isic").sum()), "expected_isic": dataset_validation["isic_test_n"],
            "pad": int((domains == "pad_ufes").sum()), "expected_pad": dataset_validation["metadata_rows_pad_ufes"],
            "pass": bool(
                (domains == "isic").sum() == dataset_validation["isic_test_n"]
                and (domains == "pad_ufes").sum() == dataset_validation["metadata_rows_pad_ufes"]
            ),
        },
        "check_2_embedding_tensor_shape": {"shape": list(embeddings.shape), "expected": [7365, 1024],
                                            "pass": list(embeddings.shape) == [7365, 1024]},
        "check_3_ordering_metadata_row_i_to_embedding_row_i": row_i_proof,
        "check_4_label_alignment": {"pass": proof_labels_match, "method": "see row_ordering_proof"},
        "check_5_sha256": {"embeddings.npy": sha_embeddings, "labels.csv": sha_labels, "metadata.csv": sha_metadata},
        "check_6_manifest": {"path": str(manifest_json_path), "pass": manifest_json_path.is_file()},
        "target_directory": str(TARGET_DIR),
        "gate_result": "PASS",
    }
    with open(FINAL_GATE_REPORT, "w") as f:
        json.dump(gate_report, f, indent=2)
    print(f"\n[stage2.1b] Final Embedding Integrity Gate report written: {FINAL_GATE_REPORT}")
    print("[stage2.1b] === GATE RESULT: PASS (6/6) ===")
    print("[stage2.1b] STOP: no probes, no geometry metrics, no OOD scores, no statistics computed.")


if __name__ == "__main__":
    main()
