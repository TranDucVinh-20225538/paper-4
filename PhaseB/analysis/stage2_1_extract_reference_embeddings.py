"""
Stage 2.1 -- extract embeddings for the COMPLETE preregistered ID + OOD
datasets, producing one frozen Reference Embedding Artifact.

Per Vinh (2026-08-05), verbatim: "Begin Stage 2.1 by extracting embeddings
for the complete preregistered ID and OOD datasets using the validated
Reference Instrument. Produce a single frozen Reference Embedding Artifact
together with full provenance, integrity checks (shape, ordering, label
alignment, sample count, SHA256), and no scientific analyses. Do not compute
probes, geometry metrics, OOD scores, or statistics. Stop immediately after
the Embedding Integrity Gate passes."

Reuses, unmodified:
  - CSG-Skin's SkinDataModule + build_id_ood_test_dataloaders (dataset/split).
  - PanDerm's models.get_encoder('PanDerm_Large_LP') (instrument).
Same components Stage 1.3's rerun already validated -- this script only
scales that same pipeline from a 16-image convenience batch to the complete
7,365-image ID+OOD population, and freezes the result instead of discarding
it.

This is NOT a cache. The output is treated like the PanDerm checkpoint
itself: frozen, provenance-tracked, integrity-gated before use. No
downstream measurement (probe, Mahalanobis, geometry, etc.) is computed
here -- those are separately-authorized, later steps that will read this
artifact, not recompute embeddings from images.
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
import torch

CSG_ROOT = Path("/Users/cubo/Research/CSG-SKin")
PANDERM_REPO = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
CHECKPOINT_METADATA = Path(__file__).resolve().parents[1] / "assets" / "panderm_vitl16_checkpoint_metadata.json"
PREPROCESSING_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "reference_preprocessing.json"
DATASET_VALIDATION_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "dataset_validation_report.json"

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings"
ARTIFACT_NPZ = ARTIFACT_DIR / "panderm_reference_embedding_artifact.npz"
PROVENANCE_JSON = ARTIFACT_DIR / "reference_embedding_artifact_provenance.json"
GATE_REPORT_JSON = Path(__file__).resolve().parents[1] / "outputs" / "embedding_integrity_gate_report.json"

EMBED_DIM = 1024
EXTRACTION_BATCH_SIZE = 32

sys.path.insert(0, str(CSG_ROOT))
sys.path.insert(0, str(PANDERM_REPO))


def _git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unavailable"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_dataset_validation_passed() -> dict:
    if not DATASET_VALIDATION_REPORT.is_file():
        raise RuntimeError(f"{DATASET_VALIDATION_REPORT} not found -- Stage 1.3a must PASS first.")
    with open(DATASET_VALIDATION_REPORT) as f:
        report = json.load(f)
    if report.get("status") != "PASS":
        raise RuntimeError(f"Dataset Validation gate did not PASS: {report}")
    return report


@torch.no_grad()
def extract_all(model, loader, domain_name: str) -> dict:
    """Extract embeddings for every image in loader, in loader (dataframe) order."""
    all_embeddings, all_labels = [], []
    n = len(loader.dataset)
    t0 = time.perf_counter()
    seen = 0
    for batch_idx, (images, labels) in enumerate(loader):
        embeddings = model.forward_features(images, is_train=False)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        seen += images.shape[0]
        if batch_idx % 10 == 0 or seen == n:
            elapsed = time.perf_counter() - t0
            rate = seen / elapsed if elapsed > 0 else 0
            print(f"[stage2.1] {domain_name}: {seen}/{n} images, {rate:.2f} img/s", flush=True)
    elapsed = time.perf_counter() - t0
    embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
    labels = torch.cat(all_labels, dim=0).numpy()
    print(f"[stage2.1] {domain_name}: DONE, {n} images in {elapsed:.1f}s ({n / elapsed:.2f} img/s)", flush=True)
    return {"embeddings": embeddings, "labels": labels, "elapsed": elapsed}


def main() -> None:
    dataset_validation = require_dataset_validation_passed()
    print(f"[stage2.1] Dataset Validation gate confirmed PASS "
          f"(timestamp={dataset_validation.get('timestamp')})")

    from models import get_encoder  # PanDerm repo, unmodified
    from src.datasets.skin_dataset import SkinDataModule  # CSG-Skin, unmodified
    from src.datasets.splits import build_id_ood_test_dataloaders  # CSG-Skin, unmodified

    class _Args:
        pretrained_checkpoint = str(CHECKPOINT)

    model, panderm_transform = get_encoder(_Args(), "PanDerm_Large_LP")
    model.eval()

    metadata_csv = Path(dataset_validation["metadata_csv"])
    dm = SkinDataModule(metadata_csv=str(metadata_csv), batch_size=EXTRACTION_BATCH_SIZE, num_workers=0)
    dm.eval_transform = panderm_transform  # attribute injection -- see 02_stage1_setup.md rationale

    id_loader, ood_loader = build_id_ood_test_dataloaders(dm)
    print(f"[stage2.1] ID (ISIC-test): {len(id_loader.dataset)} images")
    print(f"[stage2.1] OOD (PAD-UFES): {len(ood_loader.dataset)} images")

    extraction_started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    id_result = extract_all(model, id_loader, "ID (ISIC-test)")
    ood_result = extract_all(model, ood_loader, "OOD (PAD-UFES)")
    extraction_finished = datetime.datetime.now(datetime.timezone.utc).isoformat()

    embeddings = np.concatenate([id_result["embeddings"], ood_result["embeddings"]], axis=0)
    labels_collected = np.concatenate([id_result["labels"], ood_result["labels"]], axis=0)
    domains = np.array(["isic"] * len(id_result["labels"]) + ["pad_ufes"] * len(ood_result["labels"]))
    paths = np.concatenate([
        id_loader.dataset.data["path"].to_numpy(),
        ood_loader.dataset.data["path"].to_numpy(),
    ])
    n_total = embeddings.shape[0]

    # ============================== Embedding Integrity Gate ==============================
    gate: dict = {}

    # 1. Shape
    gate["shape_ok"] = bool(embeddings.shape == (n_total, EMBED_DIM))
    gate["embedding_shape"] = list(embeddings.shape)

    # 2. Sample count, cross-checked against Stage 1.3a's own Dataset Validation report
    expected_id = dataset_validation["isic_test_n"]
    expected_ood = dataset_validation["metadata_rows_pad_ufes"]
    gate["n_id"] = int(len(id_result["labels"]))
    gate["n_ood"] = int(len(ood_result["labels"]))
    gate["n_total"] = int(n_total)
    gate["sample_count_ok"] = bool(
        gate["n_id"] == expected_id
        and gate["n_ood"] == expected_ood
        and gate["n_total"] == expected_id + expected_ood
    )

    # 3. Ordering preserved: labels collected via DataLoader iteration must exactly match
    # label_idx read DIRECTLY from the dataframe (independent code path), same row order.
    df_labels = np.concatenate([
        id_loader.dataset.data["label_idx"].to_numpy(),
        ood_loader.dataset.data["label_idx"].to_numpy(),
    ])
    gate["ordering_preserved"] = bool(np.array_equal(labels_collected, df_labels))

    # 4. Label alignment: paths non-empty, all exist on disk, domain segmentation correct.
    paths_exist = all(Path(p).exists() for p in paths)
    domain_segmentation_ok = bool(
        (domains[:gate["n_id"]] == "isic").all() and (domains[gate["n_id"]:] == "pad_ufes").all()
    )
    gate["label_alignment_ok"] = bool(
        len(paths) == n_total and paths_exist and domain_segmentation_ok
    )
    gate["all_paths_exist_on_disk"] = paths_exist

    gate["structural_checks_pass"] = bool(
        gate["shape_ok"] and gate["sample_count_ok"] and gate["ordering_preserved"] and gate["label_alignment_ok"]
    )

    if not gate["structural_checks_pass"]:
        print(f"\n[stage2.1] EMBEDDING INTEGRITY GATE FAILED before save: {gate}")
        raise RuntimeError(f"Embedding Integrity Gate failed: {gate}")

    # --- Save the artifact (only after structural checks pass) ---
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        ARTIFACT_NPZ,
        embeddings=embeddings,
        labels=labels_collected,
        domain=domains,
        path=paths,
    )
    gate["sha256"] = _sha256(ARTIFACT_NPZ)
    gate["artifact_path"] = str(ARTIFACT_NPZ)
    gate["artifact_bytes"] = ARTIFACT_NPZ.stat().st_size
    gate["gate_result"] = "PASS"

    print(f"\n[stage2.1] === EMBEDDING INTEGRITY GATE: {gate['gate_result']} ===")
    print(f"[stage2.1] 1. shape: {gate['shape_ok']}  {gate['embedding_shape']}")
    print(f"[stage2.1] 2. sample count: {gate['sample_count_ok']}  "
          f"(id={gate['n_id']}, ood={gate['n_ood']}, total={gate['n_total']})")
    print(f"[stage2.1] 3. ordering preserved: {gate['ordering_preserved']}")
    print(f"[stage2.1] 4. label alignment: {gate['label_alignment_ok']} "
          f"(all paths exist on disk: {gate['all_paths_exist_on_disk']})")
    print(f"[stage2.1] 5. SHA256: {gate['sha256']}")

    with open(GATE_REPORT_JSON, "w") as f:
        json.dump({
            "protocol_step": "Stage 2.1 -- Embedding Integrity Gate",
            "date": "2026-08-05",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_validation_gate": dataset_validation,
            **gate,
        }, f, indent=2)
    print(f"[stage2.1] Gate report written: {GATE_REPORT_JSON}")

    # --- Provenance ---
    provenance = {
        "artifact": "Reference Embedding Artifact",
        "artifact_path": str(ARTIFACT_NPZ),
        "artifact_sha256": gate["sha256"],
        "artifact_bytes": gate["artifact_bytes"],
        "produced_by_protocol_step": "Stage 2.1",
        "produced_by_script": str(Path(__file__).resolve()),
        "extraction_started_utc": extraction_started,
        "extraction_finished_utc": extraction_finished,
        "n_id_isic": gate["n_id"],
        "n_ood_pad_ufes": gate["n_ood"],
        "n_total": gate["n_total"],
        "embedding_dim": EMBED_DIM,
        "extraction_batch_size": EXTRACTION_BATCH_SIZE,
        "id_throughput_images_per_sec": len(id_result["labels"]) / id_result["elapsed"],
        "ood_throughput_images_per_sec": len(ood_result["labels"]) / ood_result["elapsed"],
        "instrument": {
            "checkpoint_metadata": str(CHECKPOINT_METADATA),
            "preprocessing_snapshot": str(PREPROCESSING_SNAPSHOT),
            "panderm_repo_commit": _git_commit(PANDERM_REPO),
        },
        "dataset": {
            "csg_skin_commit": _git_commit(CSG_ROOT),
            "metadata_csv": str(metadata_csv),
            "dataset_validation_report": str(DATASET_VALIDATION_REPORT),
        },
        "embedding_integrity_gate_report": str(GATE_REPORT_JSON),
        "columns": {
            "embeddings": "(n_total, 1024) float32 -- CLS token, post-final-LayerNorm",
            "labels": "(n_total,) int -- CSG-Skin label_idx (0..7, see src/datasets/constants.py LABELS)",
            "domain": "(n_total,) str -- 'isic' (ID) or 'pad_ufes' (OOD)",
            "path": "(n_total,) str -- source image file path",
        },
        "not_computed_here": "No probe, no Mahalanobis or other distance scorer, no geometry metric, "
                              "no PCA/UMAP, no statistics. Extraction and integrity checks only.",
    }
    with open(PROVENANCE_JSON, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[stage2.1] Provenance written: {PROVENANCE_JSON}")

    print("\n[stage2.1] STOP per Vinh's scope: no probes, no geometry metrics, no OOD scores, "
          "no statistics computed. Reference Embedding Artifact frozen.")


if __name__ == "__main__":
    main()
