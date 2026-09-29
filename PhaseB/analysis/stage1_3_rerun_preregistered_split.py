"""
Stage 1.3 RE-RUN -- Batch Inference Validation, this time through the
PREREGISTERED dataset loader, not the legacy DST-Skin reader-study folder.

Per Vinh (PI review, 2026-08-05): the first Stage 1.3 run (see
stage1_3_batch_inference_validation.py) was an Engineering PASS but not a
Scientific PASS -- it sampled from DST-Skin/reader_images_96/, a Paper 2
reader-study convenience folder, not Papers 1/3's preregistered ISIC 2019 +
PAD-UFES-20 split. That run stays valid and stays in the Lab Notebook
unedited; this is a separate, additional run, not a correction of it.

Dataset loader: CSG-Skin's OWN code, imported and run unmodified --
`SkinDataModule` + `build_id_ood_test_dataloaders`
(CSG-SKin/src/datasets/skin_dataset.py, splits.py) -- the exact loader
Papers 1/3's own scripts use (paper-3/scripts/extract_auroc_e2.py). The only
injected difference is which `eval_transform` the datamodule carries: CSG's
own build_val_transform_robust() is swapped for PanDerm's official,
already-frozen transform (assets/reference_preprocessing.json) BEFORE
build_id_ood_test_dataloaders is called, via a plain attribute assignment
-- no CSG-Skin file is read from or edited, and this is not a code fork,
since build_id_ood_test_dataloaders reads datamodule.eval_transform
dynamically at call time.

master_metadata.csv is built by CSG-Skin's own, unmodified
src/datasets/preprocess_metadata.py, from:
  - ISIC 2019 Training Input images + GroundTruth CSV (official ISIC
    archive, downloaded this session -- see 26_Lab_Notebook.md for URLs,
    sizes, dates).
  - PAD-UFES-20 images + metadata.csv (confirmed byte-identical to
    DST-Skin/data/raw/pad_ufes20/, copied into CSG-Skin/data/ read-only).
Split: CSG-Skin's own SplitConfig defaults (random_state=42,
isic_test_fraction=0.2, val_fraction=0.2) -- the same defaults Papers 1-3's
own scripts use, per 22_Experimental_Design.md SS2's "same split seeds where
feasible."

Convenience batch: first 8 rows of the ISIC-test (ID) loader + first 8 rows
of the PAD-UFES (OOD) loader (batch_size=8, shuffle=False on both loaders --
a deterministic, reproducible "first N" sample, not a random one), combined
into one 16-image batch -- matching the original Stage 1.3 run's batch size
for direct comparability.

Same 5 gate criteria as the original Stage 1.3 run. Same scope discipline:
no embedding cache, no probe, no geometry metric, no scientific analysis.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

import data_roots

CSG_ROOT = data_roots.root("csg")
PANDERM_REPO = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
PREPROCESSING_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "reference_preprocessing.json"
OUTPUT_JSON = Path(__file__).resolve().parents[1] / "outputs" / "stage1_3_rerun_preregistered_split.json"
EMBED_DIM = 1024
CONVENIENCE_BATCH_SIZE_PER_DOMAIN = 8
ZERO_NORM_THRESHOLD = 1e-6

sys.path.insert(0, str(CSG_ROOT))
sys.path.insert(0, str(PANDERM_REPO))


def check_transform_matches_snapshot(eval_transform) -> None:
    with open(PREPROCESSING_SNAPSHOT) as f:
        snapshot = json.load(f)
    pipeline = snapshot["pipeline"]
    resize, crop, _to_tensor, normalize = eval_transform.transforms
    assert resize.size == pipeline[0]["size"]
    assert str(resize.interpolation).split(".")[-1].lower() == pipeline[0]["interpolation"]
    assert list(crop.size) == pipeline[1]["size"]
    assert list(normalize.mean) == pipeline[3]["mean"]
    assert list(normalize.std) == pipeline[3]["std"]
    print("[stage1.3-rerun] Live transform matches frozen assets/reference_preprocessing.json -- no drift.")


DATASET_VALIDATION_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "dataset_validation_report.json"


def require_dataset_validation_passed() -> dict:
    """
    Stage 1.3a (stage1_3a_dataset_validation.py) is a separate, standalone gate
    that must PASS -- and be reviewed at a PI checkpoint, per Vinh (2026-08-05)
    -- before this script runs at all. Rather than re-deriving those checks here
    (risking the two copies drifting apart), this reads that gate's own written
    report and refuses to proceed if it's missing or not a PASS. This is the
    dependency link between the two gates, not a re-validation.
    """
    if not DATASET_VALIDATION_REPORT.is_file():
        raise RuntimeError(
            f"{DATASET_VALIDATION_REPORT} not found -- run stage1_3a_dataset_validation.py "
            "first and get a PI checkpoint sign-off before running this script."
        )
    with open(DATASET_VALIDATION_REPORT) as f:
        report = json.load(f)
    if report.get("status") != "PASS":
        raise RuntimeError(f"Dataset Validation gate did not PASS: {report}")
    print(f"[stage1.3-rerun] Dataset Validation gate confirmed PASS "
          f"(from {DATASET_VALIDATION_REPORT}, timestamp={report.get('timestamp')})")
    return report


def main() -> None:
    from src.datasets.skin_dataset import SkinDataModule  # CSG-Skin, unmodified
    from src.datasets.splits import build_id_ood_test_dataloaders  # CSG-Skin, unmodified
    from models import get_encoder  # PanDerm repo, unmodified, same as Stage 1.2/1.3

    dataset_validation = require_dataset_validation_passed()  # PI-checkpointed gate; raises if missing/FAIL

    class _Args:
        pretrained_checkpoint = str(CHECKPOINT)

    model, panderm_transform = get_encoder(_Args(), "PanDerm_Large_LP")
    model.eval()
    check_transform_matches_snapshot(panderm_transform)

    metadata_csv = Path(dataset_validation["metadata_csv"])
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"{metadata_csv} not found (per Dataset Validation report).")

    dm = SkinDataModule(
        metadata_csv=str(metadata_csv),
        batch_size=CONVENIENCE_BATCH_SIZE_PER_DOMAIN,
        num_workers=0,
    )
    # Inject PanDerm's official, already-frozen preprocessing in place of CSG's own
    # CNN-oriented eval_transform. build_id_ood_test_dataloaders reads this attribute
    # dynamically at call time -- no CSG-Skin file touched.
    dm.eval_transform = panderm_transform

    print(f"[stage1.3-rerun] metadata_csv: {metadata_csv}")
    print(f"[stage1.3-rerun] split_config: random_state={dm.split_config.random_state}, "
          f"isic_test_fraction={dm.split_config.isic_test_fraction}, "
          f"val_fraction={dm.split_config.val_fraction} (CSG-Skin defaults, matching Papers 1-3)")

    id_loader, ood_loader = build_id_ood_test_dataloaders(dm)  # CSG-Skin's own function, unmodified
    print(f"[stage1.3-rerun] ID (ISIC-test) dataset size: {len(id_loader.dataset)}")
    print(f"[stage1.3-rerun] OOD (PAD-UFES) dataset size: {len(ood_loader.dataset)}")

    id_images, id_labels = next(iter(id_loader))
    ood_images, ood_labels = next(iter(ood_loader))
    print(f"[stage1.3-rerun] Convenience batch: {id_images.shape[0]} ID (ISIC) + "
          f"{ood_images.shape[0]} OOD (PAD-UFES) images")

    batch = torch.cat([id_images, ood_images], dim=0)
    domain_labels = ["isic"] * id_images.shape[0] + ["pad_ufes"] * ood_images.shape[0]
    class_labels = torch.cat([id_labels, ood_labels], dim=0).tolist()

    t0 = time.perf_counter()
    with torch.no_grad():
        embeddings = model.forward_features(batch, is_train=False)
    elapsed = time.perf_counter() - t0

    shape_ok = embeddings.shape == (batch.shape[0], EMBED_DIM)
    has_nan = torch.isnan(embeddings).any().item()
    has_inf = torch.isinf(embeddings).any().item()
    row_norms = embeddings.norm(dim=1)
    zero_rows = (row_norms < ZERO_NORM_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
    throughput = batch.shape[0] / elapsed if elapsed > 0 else float("nan")

    gate_pass = shape_ok and not has_nan and not has_inf and len(zero_rows) == 0

    summary = {
        "stage": "1.3 RE-RUN -- Batch Inference Validation via preregistered dataset loader",
        "date": "2026-08-05",
        "dataset_loader": "CSG-Skin SkinDataModule + build_id_ood_test_dataloaders (unmodified)",
        "metadata_csv": str(metadata_csv),
        "split_config": {
            "random_state": dm.split_config.random_state,
            "isic_test_fraction": dm.split_config.isic_test_fraction,
            "val_fraction": dm.split_config.val_fraction,
        },
        "dataset_validation_gate": dataset_validation,
        "id_dataset_size_isic_test": len(id_loader.dataset),
        "ood_dataset_size_pad_ufes": len(ood_loader.dataset),
        "convenience_batch": {
            "n_id_isic": id_images.shape[0],
            "n_ood_pad_ufes": ood_images.shape[0],
            "n_total": batch.shape[0],
            "domain_labels": domain_labels,
            "class_label_indices": class_labels,
        },
        "embedding_shape": list(embeddings.shape),
        "criterion_1_shape_always_Nx1024": shape_ok,
        "criterion_2_no_nan": not has_nan,
        "criterion_3_no_inf": not has_inf,
        "criterion_4_no_silent_failure": len(zero_rows) == 0,
        "zero_norm_indices": zero_rows,
        "criterion_5_throughput_images_per_sec": throughput,
        "forward_pass_seconds": elapsed,
        "gate_result": "PASS" if gate_pass else "FAIL",
        "note": "Embeddings held in memory for validation only, then discarded -- not cached to "
                "disk. No probe, no geometry metric, no scientific analysis performed.",
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[stage1.3-rerun] === GATE RESULT: {summary['gate_result']} ===")
    print(f"[stage1.3-rerun] 1. shape always Nx1024: {shape_ok}  (got {tuple(embeddings.shape)})")
    print(f"[stage1.3-rerun] 2. no NaN: {not has_nan}")
    print(f"[stage1.3-rerun] 3. no Inf: {not has_inf}")
    print(f"[stage1.3-rerun] 4. no silent failure: {len(zero_rows) == 0} ({len(zero_rows)} zero-norm rows)")
    print(f"[stage1.3-rerun] 5. throughput: {throughput:.3f} images/sec "
          f"({batch.shape[0]} images / {elapsed:.3f}s, CPU)")
    print(f"[stage1.3-rerun] Summary written to {OUTPUT_JSON}")
    print("[stage1.3-rerun] STOP: no embedding cache, no probing, no geometry metrics, "
          "no scientific analysis performed.")


if __name__ == "__main__":
    main()
