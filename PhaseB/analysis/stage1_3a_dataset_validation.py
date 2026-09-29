"""
Stage 1.3a -- Dataset Validation gate, standalone.

Per Vinh (2026-08-05): after the ISIC 2019 download and master_metadata.csv
build, do NOT run straight through to the encoder. Stop here, report, and
wait for a PI checkpoint before Stage 1.3's actual re-run
(stage1_3_rerun_preregistered_split.py) touches PanDerm at all.

This script deliberately does NOT import PanDerm's encoder or load its
checkpoint -- only `models.builder.get_eval_transforms`, a plain
torchvision transform constructor with no model weights, is used to attach
and verify the preprocessing. No 1.3GB checkpoint load happens here; that
stays inside the Stage 1.3 rerun script, on the other side of the PI
checkpoint.

Five checks (Vinh's list), all must pass:
  1. ISIC image count on disk.
  2. PAD-UFES-20 image count on disk.
  3. master_metadata.csv row count (by domain).
  4. Split reproducible at the pinned seed (random_state=42) -- recomputed
     twice from the same input, index sets compared.
  5. Preprocessing attached -- dm.eval_transform matches the frozen
     assets/reference_preprocessing.json snapshot.

Writes outputs/dataset_validation_report.json. Does not build any
DataLoader batch, does not sample any image, does not touch torch.nn.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import data_roots

CSG_ROOT = data_roots.root("csg")
PANDERM_REPO = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
PREPROCESSING_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "reference_preprocessing.json"
OUTPUT_JSON = Path(__file__).resolve().parents[1] / "outputs" / "dataset_validation_report.json"
PROTOCOL_VERSION = "24_PreRegistration.md frozen 2026-08-04 (session 9)"

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


def check_transform_matches_snapshot(eval_transform) -> bool:
    with open(PREPROCESSING_SNAPSHOT) as f:
        snapshot = json.load(f)
    pipeline = snapshot["pipeline"]
    try:
        resize, crop, _to_tensor, normalize = eval_transform.transforms
        assert resize.size == pipeline[0]["size"]
        assert str(resize.interpolation).split(".")[-1].lower() == pipeline[0]["interpolation"]
        assert list(crop.size) == pipeline[1]["size"]
        assert list(normalize.mean) == pipeline[3]["mean"]
        assert list(normalize.std) == pipeline[3]["std"]
        return True
    except AssertionError:
        return False


def main() -> None:
    from models.builder import get_eval_transforms  # PanDerm repo -- transform only, no model/checkpoint
    from src.datasets.skin_dataset import SkinDataModule  # CSG-Skin, unmodified
    from src.datasets.splits import load_filtered_master  # CSG-Skin, unmodified
    from sklearn.model_selection import train_test_split

    metadata_csv = CSG_ROOT / "data" / "master_metadata.csv"
    isic_input_dir = CSG_ROOT / "data" / "ISIC_2019_Training_Input"
    pad_images_dir = CSG_ROOT / "data" / "pad_ufes20" / "images"

    panderm_transform = get_eval_transforms(which_img_norm="imagenet", img_resize=256, center_crop=True)
    preprocessing_ok = check_transform_matches_snapshot(panderm_transform)

    dm = SkinDataModule(metadata_csv=str(metadata_csv), batch_size=8, num_workers=0)
    dm.eval_transform = panderm_transform  # attribute injection -- see 02_stage1_setup.md for the rationale

    n_isic_files = sum(1 for _ in isic_input_dir.glob("*.jpg")) if isic_input_dir.is_dir() else 0
    n_pad_files = sum(1 for _ in pad_images_dir.glob("*")) if pad_images_dir.is_dir() else 0

    df = load_filtered_master(str(metadata_csv))
    n_meta_isic = int((df["domain"] == "isic").sum())
    n_meta_pad = int((df["domain"] == "pad_ufes").sum())

    def _isic_test_indices():
        isic_df = df[df["domain"] == "isic"]
        _train_val, test = train_test_split(
            isic_df,
            test_size=dm.split_config.isic_test_fraction,
            stratify=isic_df["label_idx"],
            random_state=dm.split_config.random_state,
        )
        return frozenset(test.index.tolist())

    split_a = _isic_test_indices()
    split_b = _isic_test_indices()
    split_reproducible = split_a == split_b

    checks = {
        "isic_images": n_isic_files,
        "pad_images": n_pad_files,
        "metadata_rows": len(df),
        "metadata_rows_isic": n_meta_isic,
        "metadata_rows_pad_ufes": n_meta_pad,
        "seed": dm.split_config.random_state,
        "split_reproducible": split_reproducible,
        "isic_test_fraction": dm.split_config.isic_test_fraction,
        "val_fraction": dm.split_config.val_fraction,
        "isic_test_n": len(split_a),
        "preprocessing_attached": preprocessing_ok,
    }

    all_pass = (
        n_isic_files > 0
        and n_pad_files > 0
        and n_meta_isic > 0
        and n_meta_pad > 0
        and split_reproducible
        and preprocessing_ok
    )

    report = {
        "protocol_step": "Stage 1.3a -- Dataset Validation",
        "protocol_version": PROTOCOL_VERSION,
        "date": "2026-08-05",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "csg_skin_commit": _git_commit(CSG_ROOT),
        "panderm_repo_commit": _git_commit(PANDERM_REPO),
        "panderm_preprocessing_snapshot": str(PREPROCESSING_SNAPSHOT),
        "panderm_preprocessing_sha256": _sha256(PREPROCESSING_SNAPSHOT),
        "metadata_csv": str(metadata_csv),
        **checks,
        "status": "PASS" if all_pass else "FAIL",
    }

    print("=== Dataset Validation gate (standalone -- no encoder touched) ===")
    print(f"  [{'x' if n_isic_files > 0 else ' '}] ISIC images on disk: {n_isic_files}")
    print(f"  [{'x' if n_pad_files > 0 else ' '}] PAD-UFES images on disk: {n_pad_files}")
    print(f"  [{'x' if n_meta_isic > 0 and n_meta_pad > 0 else ' '}] metadata rows: "
          f"{len(df)} total ({n_meta_isic} isic, {n_meta_pad} pad_ufes)")
    print(f"  [{'x' if split_reproducible else ' '}] split reproducible at seed={dm.split_config.random_state} "
          f"(isic_test n={len(split_a)})")
    print(f"  [{'x' if preprocessing_ok else ' '}] preprocessing attached "
          f"(matches assets/reference_preprocessing.json)")
    print(f"\n  === STATUS: {report['status']} ===")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWritten: {OUTPUT_JSON}")

    if not all_pass:
        raise RuntimeError(f"Dataset Validation gate FAILED: {report}")

    print("\nSTOP per Vinh's PI-checkpoint requirement: no encoder loaded, no batch sampled, "
          "no forward pass run. Stage 1.3's actual rerun waits for explicit go-ahead.")


if __name__ == "__main__":
    main()
