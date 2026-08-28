"""
EA-01 implementation -- extract the ISIC-train population through the same
validated Reference Instrument, producing ReferenceTrainEmbedding/.

Per EA-01 (PhaseB/ExecutionAmendments/EA-01.md), approved 2026-08-05: Stage
2.1 extracted only the held-out ID(ISIC-test)+OOD(PAD-UFES) evaluation split.
Group B's reliability scorers (Mahalanobis, kNN, cosine-to-centroid, KDE --
and ViM, partially) need a separate FITTING population distinct from what
they're evaluated on -- exactly the ISIC-train split Paper 3's own scorer
implementations fit on. This was an execution-specification gap, not a
protocol problem (22_Experimental_Design.md SS2 already names a train split
as in-scope for "the FM arm"; Stage 2.1 just didn't resolve whether the
Reference arm specifically needed it). See GroupB_protocol_reconciliation.md
for the full trace.

Reuses, unmodified, the same components already validated:
  - CSG-Skin's own two-stage stratified split (random_state=42,
    isic_test_fraction=0.2, val_fraction=0.2) -- the exact same computation
    Stage 1.3a's Dataset Validation gate already reproduced and proved
    reproducible. This script takes isic_train (the complement of
    isic_test), not isic_test.
  - CSG-Skin's SkinDataset (path/label loading), NOT SkinDataModule's own
    train_dataloader() -- that uses CSG's own augmented train_transform
    (random crop/flip/color-jitter), which is wrong for feature extraction.
    This script builds SkinDataset(isic_train_df, transform=<PanDerm's own
    frozen eval transform>) directly, matching how Stage 1.3/2.1 already
    inject PanDerm's transform via attribute assignment.
  - PanDerm's models.get_encoder('PanDerm_Large_LP') (same Reference
    Instrument, same checkpoint).

Target layout (mirrors ReferenceEmbedding/ exactly):
  assets/reference_embeddings/ReferenceTrainEmbedding/
      embeddings.npy   -- (n_train, 1024) float32
      labels.csv        -- row_index, label_idx, label_name
      metadata.csv       -- row_index, path, domain ('isic', always), label_idx, label_name
      manifest.json        -- birth certificate
      sha256.txt             -- per-file checksums

Domain is always 'isic' -- Paper 3's own z_train (paper-3/scripts/
extract_embeddings_e1.py::_build_isic_train_loader) is ISIC-only; PAD-UFES-20
has no analogous "train" population in this design (it is the OOD side,
evaluated against, never fit on).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CSG_ROOT = Path("/Users/cubo/Research/CSG-SKin")
PANDERM_REPO = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
CHECKPOINT_METADATA = Path(__file__).resolve().parents[1] / "assets" / "panderm_vitl16_checkpoint_metadata.json"
PREPROCESSING_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "reference_preprocessing.json"
DATASET_VALIDATION_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "dataset_validation_report.json"

TARGET_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceTrainEmbedding"
GATE_REPORT_JSON = Path(__file__).resolve().parents[1] / "outputs" / "reference_train_embedding_integrity_gate_report.json"

EMBED_DIM = 1024
EXTRACTION_BATCH_SIZE = 32
LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]  # CSG-Skin src/datasets/constants.py

sys.path.insert(0, str(CSG_ROOT))
sys.path.insert(0, str(PANDERM_REPO))


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_isic_train_df(metadata_csv: Path, random_state: int = 42,
                         isic_test_fraction: float = 0.2, val_fraction: float = 0.2) -> pd.DataFrame:
    """Same two-stage stratified split already validated in Stage 1.3a -- takes isic_train
    (the complement of isic_test), not isic_test itself. Independent, callable twice for
    the ordering-proof step below (fresh reconstruction, not reuse of in-memory state)."""
    from sklearn.model_selection import train_test_split
    from src.datasets.splits import load_filtered_master

    df = load_filtered_master(str(metadata_csv))
    isic_df = df[df["domain"] == "isic"].copy()
    isic_train_val, _isic_test = train_test_split(
        isic_df, test_size=isic_test_fraction, stratify=isic_df["label_idx"], random_state=random_state
    )
    isic_train, _isic_val = train_test_split(
        isic_train_val, test_size=val_fraction, stratify=isic_train_val["label_idx"], random_state=random_state
    )
    return isic_train.reset_index(drop=True)


@torch.no_grad()
def extract_all(model, loader, n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings, all_labels = [], []
    t0 = time.perf_counter()
    seen = 0
    for batch_idx, (images, labels) in enumerate(loader):
        embeddings = model.forward_features(images, is_train=False)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        seen += images.shape[0]
        if batch_idx % 20 == 0 or seen == n_expected:
            elapsed = time.perf_counter() - t0
            rate = seen / elapsed if elapsed > 0 else 0
            print(f"[ea01] {seen}/{n_expected} images, {rate:.2f} img/s", flush=True)
    elapsed = time.perf_counter() - t0
    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    labels = torch.cat(all_labels, dim=0).numpy()
    print(f"[ea01] DONE: {n_expected} images in {elapsed:.1f}s ({n_expected / elapsed:.2f} img/s)")
    return embeddings, labels


def main() -> None:
    with open(DATASET_VALIDATION_REPORT) as f:
        dataset_validation = json.load(f)
    if dataset_validation.get("status") != "PASS":
        raise RuntimeError("Dataset Validation gate did not PASS -- cannot proceed.")
    metadata_csv = Path(dataset_validation["metadata_csv"])

    from models import get_encoder  # PanDerm repo, unmodified
    from src.datasets.skin_dataset import SkinDataset  # CSG-Skin, unmodified
    from torch.utils.data import DataLoader

    class _Args:
        pretrained_checkpoint = str(CHECKPOINT)

    model, panderm_transform = get_encoder(_Args(), "PanDerm_Large_LP")
    model.eval()

    isic_train_df = build_isic_train_df(metadata_csv)
    n_expected = len(isic_train_df)
    print(f"[ea01] ISIC-train population: {n_expected} rows "
          f"(random_state=42, isic_test_fraction=0.2, val_fraction=0.2 -- same split already validated)")

    dataset = SkinDataset(isic_train_df, transform=panderm_transform)
    loader = DataLoader(dataset, batch_size=EXTRACTION_BATCH_SIZE, shuffle=False, num_workers=0)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    embeddings, labels_collected = extract_all(model, loader, n_expected)
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # ============================== Embedding Integrity Gate ==============================
    gate: dict = {}
    gate["shape_ok"] = bool(embeddings.shape == (n_expected, EMBED_DIM))
    gate["embedding_shape"] = list(embeddings.shape)
    gate["sample_count_ok"] = bool(embeddings.shape[0] == n_expected)
    gate["n_total"] = int(n_expected)

    # Ordering + label alignment: FRESH independent reconstruction, not reuse of in-memory state.
    fresh_df = build_isic_train_df(metadata_csv)
    fresh_labels = fresh_df["label_idx"].to_numpy()
    fresh_paths = fresh_df["path"].to_numpy()
    gate["ordering_preserved"] = bool(np.array_equal(labels_collected, fresh_labels))
    gate["fresh_reconstruction_matches_extraction_df"] = bool(fresh_df["path"].equals(isic_train_df["path"]))

    paths_exist = all(Path(p).exists() for p in fresh_paths)
    gate["label_alignment_ok"] = bool(
        len(fresh_paths) == n_expected and paths_exist and gate["ordering_preserved"]
    )
    gate["all_paths_exist_on_disk"] = paths_exist

    gate["structural_checks_pass"] = bool(
        gate["shape_ok"] and gate["sample_count_ok"] and gate["ordering_preserved"] and gate["label_alignment_ok"]
    )
    if not gate["structural_checks_pass"]:
        print(f"\n[ea01] EMBEDDING INTEGRITY GATE FAILED before save: {gate}")
        raise RuntimeError(f"Embedding Integrity Gate failed: {gate}")

    # --- Save canonical artifact (only after structural checks pass) ---
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TARGET_DIR / "embeddings.npy", embeddings)

    labels_df = pd.DataFrame({
        "row_index": np.arange(n_expected),
        "label_idx": labels_collected,
        "label_name": [LABELS[i] for i in labels_collected],
    })
    labels_df.to_csv(TARGET_DIR / "labels.csv", index=False)

    metadata_df = pd.DataFrame({
        "row_index": np.arange(n_expected),
        "path": fresh_paths,
        "domain": "isic",
        "label_idx": labels_collected,
        "label_name": [LABELS[i] for i in labels_collected],
    })
    metadata_df.to_csv(TARGET_DIR / "metadata.csv", index=False)

    gate["sha256_embeddings_npy"] = _sha256_file(TARGET_DIR / "embeddings.npy")
    gate["sha256_labels_csv"] = _sha256_file(TARGET_DIR / "labels.csv")
    gate["sha256_metadata_csv"] = _sha256_file(TARGET_DIR / "metadata.csv")
    gate["gate_result"] = "PASS"

    print(f"\n[ea01] === EMBEDDING INTEGRITY GATE: {gate['gate_result']} ===")
    print(f"[ea01] 1. shape: {gate['shape_ok']}  {gate['embedding_shape']}")
    print(f"[ea01] 2. sample count: {gate['sample_count_ok']}  (n={gate['n_total']})")
    print(f"[ea01] 3. ordering preserved (fresh reconstruction): {gate['ordering_preserved']}")
    print(f"[ea01] 4. label alignment: {gate['label_alignment_ok']} (all paths exist: {gate['all_paths_exist_on_disk']})")
    print(f"[ea01] 5. SHA256: embeddings={gate['sha256_embeddings_npy'][:16]}... "
          f"labels={gate['sha256_labels_csv'][:16]}... metadata={gate['sha256_metadata_csv'][:16]}...")

    manifest = {
        "artifact": "Reference Train Embedding Artifact",
        "execution_amendment": "EA-01",
        "produced_by_protocol_step": "EA-01 implementation (Stage 2.1-equivalent scope, ISIC-train population)",
        "produced_by_script": str(Path(__file__).resolve()),
        "extraction_started_utc": started,
        "extraction_finished_utc": finished,
        "n_total": n_expected,
        "domain": "isic (always -- no PAD-UFES-20 train population in this design)",
        "embedding_dim": EMBED_DIM,
        "split": {
            "random_state": 42, "isic_test_fraction": 0.2, "val_fraction": 0.2,
            "note": "isic_train = complement of isic_test within isic_train_val; same split already "
                    "validated by Stage 1.3a's Dataset Validation gate.",
        },
        "instrument": {
            "checkpoint_metadata": str(CHECKPOINT_METADATA),
            "preprocessing_snapshot": str(PREPROCESSING_SNAPSHOT),
            "panderm_repo_commit": _git_commit(PANDERM_REPO),
        },
        "csg_skin_commit": _git_commit(CSG_ROOT),
        "metadata_csv": str(metadata_csv),
        "files": {
            "embeddings.npy": {"sha256": gate["sha256_embeddings_npy"], "shape": list(embeddings.shape), "dtype": "float32"},
            "labels.csv": {"sha256": gate["sha256_labels_csv"], "rows": n_expected},
            "metadata.csv": {"sha256": gate["sha256_metadata_csv"], "rows": n_expected},
        },
        "row_ordering_proof": {
            "method": "Freshly reconstructed the isic_train dataframe a SECOND time (independent "
                      "function call, not read from the extraction run's own memory), compared "
                      "path/label_idx arrays element-by-element against the extraction's collected arrays.",
            "matches_fresh_reconstruction": gate["ordering_preserved"],
            "n_compared": n_expected,
        },
        "not_computed_here": "No probe, no Mahalanobis or other distance scorer, no geometry metric, "
                              "no PCA/UMAP, no statistics. Extraction and integrity verification only.",
        "created_at": finished,
        "status": "FROZEN",
    }
    with open(TARGET_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(TARGET_DIR / "sha256.txt", "w") as f:
        f.write(f"{gate['sha256_embeddings_npy']}  embeddings.npy\n")
        f.write(f"{gate['sha256_labels_csv']}  labels.csv\n")
        f.write(f"{gate['sha256_metadata_csv']}  metadata.csv\n")

    with open(GATE_REPORT_JSON, "w") as f:
        json.dump({
            "protocol_step": "EA-01 implementation -- Embedding Integrity Gate",
            "timestamp": finished,
            **gate,
        }, f, indent=2)

    # --- Freeze ---
    for fname in ["embeddings.npy", "labels.csv", "metadata.csv", "manifest.json", "sha256.txt"]:
        (TARGET_DIR / fname).chmod(0o444)
    TARGET_DIR.chmod(0o555)
    print(f"\n[ea01] Frozen (chmod read-only): {TARGET_DIR}")

    print("\n[ea01] STOP: no Group B scorer computed here. ReferenceTrainEmbedding/ is now available "
          "for EA-01's lifecycle to move to Implemented; Group B execution is a separate PI decision.")


if __name__ == "__main__":
    main()
