"""
Stage 1.3 -- Batch Inference Validation.

Per Vinh (PI decision, 2026-08-05): "Run official PanDerm preprocessing and
encoder on a representative batch of images to validate stable batched
inference. Record tensor shape, NaN/Inf checks, failed-image count, and
throughput. Do not cache embeddings, run probes, compute geometry metrics,
or perform any scientific analysis. Stop immediately after the Batch
Inference Validation gate passes."

Five gate criteria (exact, from Vinh):
  1. Embedding shape is always N x 1024.
  2. No NaN.
  3. No Inf.
  4. No sample fails silently (image -> zero embedding without being counted
     as a failure).
  5. Throughput (images/sec) -- recorded, not benchmarked/optimized.

Model + preprocessing: same official recipe as Stage 1.2
(models.get_encoder(args, 'PanDerm_Large_LP')) -- not reimplemented. Before
running, the live transform is checked against the frozen
assets/reference_preprocessing.json snapshot, so any future repo drift is
caught rather than silently absorbed.

Explicitly NOT done here (per scope): no embedding cache written to disk, no
dataset loader/DataLoader infrastructure (that is a separate, later Stage 1
checklist item), no probe, no geometry metric, no comparison to Paper 3.
Embeddings live only in memory for the duration of the validation checks
below, then are discarded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
PREPROCESSING_SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "reference_preprocessing.json"
IMAGE_DIR = Path("/Users/cubo/Research/DST-Skin/reader_images_96")
OUTPUT_JSON = Path(__file__).resolve().parents[1] / "outputs" / "stage1_3_batch_inference_validation.json"
BATCH_SIZE = 16
EMBED_DIM = 1024
ZERO_NORM_THRESHOLD = 1e-6  # a genuine LayerNorm'd CLS embedding should never be ~0

sys.path.insert(0, str(REPO_ROOT))


def check_transform_matches_snapshot(eval_transform) -> None:
    with open(PREPROCESSING_SNAPSHOT) as f:
        snapshot = json.load(f)
    pipeline = snapshot["pipeline"]

    resize, crop, _to_tensor, normalize = eval_transform.transforms
    assert resize.size == pipeline[0]["size"], f"Resize size drifted: {resize.size} != {pipeline[0]['size']}"
    assert str(resize.interpolation).split(".")[-1].lower() == pipeline[0]["interpolation"], \
        f"Resize interpolation drifted: {resize.interpolation}"
    assert list(crop.size) == pipeline[1]["size"], f"CenterCrop size drifted: {crop.size}"
    assert list(normalize.mean) == pipeline[3]["mean"], f"Normalize mean drifted: {normalize.mean}"
    assert list(normalize.std) == pipeline[3]["std"], f"Normalize std drifted: {normalize.std}"
    print("[stage1.3] Live transform matches frozen assets/reference_preprocessing.json -- no drift.")


def main() -> None:
    from models import get_encoder  # official repo import, same as Stage 1.2

    class _Args:
        pretrained_checkpoint = str(CHECKPOINT)

    model, eval_transform = get_encoder(_Args(), "PanDerm_Large_LP")
    model.eval()
    check_transform_matches_snapshot(eval_transform)

    image_paths = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))
    n_isic = sum(1 for p in image_paths if p.name.startswith("ISIC"))
    n_pat = sum(1 for p in image_paths if p.name.startswith("PAT"))
    print(f"[stage1.3] Representative batch source: {IMAGE_DIR}")
    print(f"[stage1.3] {len(image_paths)} images ({n_isic} ISIC / dermoscopy, {n_pat} PAT / PAD-UFES-style clinical)")
    print(f"[stage1.3] Batch size: {BATCH_SIZE}")

    failed_images: list[dict] = []
    zero_norm_images: list[dict] = []
    batch_reports: list[dict] = []
    total_images_attempted = 0
    total_images_succeeded = 0
    total_wall_time = 0.0

    for batch_start in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]
        batch_tensors = []
        batch_valid_paths = []

        for p in batch_paths:
            total_images_attempted += 1
            try:
                image = Image.open(p).convert("RGB")
                tensor = eval_transform(image)
                batch_tensors.append(tensor)
                batch_valid_paths.append(p)
            except Exception as e:
                failed_images.append({"file": p.name, "stage": "load/preprocess", "error": str(e)})

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors, dim=0)

        t0 = time.perf_counter()
        with torch.no_grad():
            embeddings = model.forward_features(batch, is_train=False)
        elapsed = time.perf_counter() - t0
        total_wall_time += elapsed

        # --- Gate checks, per batch ---
        shape_ok = embeddings.shape == (len(batch_valid_paths), EMBED_DIM)
        has_nan = torch.isnan(embeddings).any().item()
        has_inf = torch.isinf(embeddings).any().item()
        row_norms = embeddings.norm(dim=1)
        zero_rows = (row_norms < ZERO_NORM_THRESHOLD).nonzero(as_tuple=True)[0].tolist()
        for idx in zero_rows:
            zero_norm_images.append({"file": batch_valid_paths[idx].name, "norm": row_norms[idx].item()})

        total_images_succeeded += len(batch_valid_paths)

        batch_report = {
            "batch_index": batch_start // BATCH_SIZE,
            "n_images": len(batch_valid_paths),
            "embedding_shape": list(embeddings.shape),
            "shape_ok": shape_ok,
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
            "n_zero_norm": len(zero_rows),
            "forward_pass_seconds": elapsed,
        }
        batch_reports.append(batch_report)
        print(
            f"[stage1.3] batch {batch_report['batch_index']:>2d}: n={batch_report['n_images']:>2d}  "
            f"shape={tuple(embeddings.shape)}  shape_ok={shape_ok}  nan={has_nan}  inf={has_inf}  "
            f"zero_norm={len(zero_rows)}  {elapsed:.3f}s"
        )

        del embeddings, batch  # explicitly not retained -- no embedding cache

    throughput = total_images_succeeded / total_wall_time if total_wall_time > 0 else float("nan")

    all_shapes_ok = all(b["shape_ok"] for b in batch_reports)
    any_nan = any(b["has_nan"] for b in batch_reports)
    any_inf = any(b["has_inf"] for b in batch_reports)
    any_zero_norm = len(zero_norm_images) > 0
    silent_failure = any_zero_norm  # a zero-embedding NOT already counted as a load/preprocess failure

    gate_pass = all_shapes_ok and not any_nan and not any_inf and not silent_failure

    summary = {
        "stage": "1.3 -- Batch Inference Validation",
        "date": "2026-08-05",
        "image_source": str(IMAGE_DIR),
        "n_images_total": len(image_paths),
        "n_images_attempted": total_images_attempted,
        "n_images_succeeded": total_images_succeeded,
        "n_images_failed_to_load": len(failed_images),
        "failed_images": failed_images,
        "batch_size": BATCH_SIZE,
        "n_batches": len(batch_reports),
        "criterion_1_shape_always_Nx1024": all_shapes_ok,
        "criterion_2_no_nan": not any_nan,
        "criterion_3_no_inf": not any_inf,
        "criterion_4_no_silent_failure": not silent_failure,
        "zero_norm_images": zero_norm_images,
        "criterion_5_throughput_images_per_sec": throughput,
        "total_forward_pass_wall_seconds": total_wall_time,
        "gate_result": "PASS" if gate_pass else "FAIL",
        "batches": batch_reports,
        "note": "Embeddings held in memory for validation only, then discarded -- not cached to disk, "
                "per Vinh's explicit Stage 1.3 scope.",
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[stage1.3] === GATE RESULT: {summary['gate_result']} ===")
    print(f"[stage1.3] 1. shape always Nx1024: {all_shapes_ok}")
    print(f"[stage1.3] 2. no NaN: {not any_nan}")
    print(f"[stage1.3] 3. no Inf: {not any_inf}")
    print(f"[stage1.3] 4. no silent failure: {not silent_failure} "
          f"({len(failed_images)} loud/counted load failures, {len(zero_norm_images)} zero-norm rows)")
    print(f"[stage1.3] 5. throughput: {throughput:.3f} images/sec "
          f"({total_images_succeeded} images / {total_wall_time:.3f}s forward-pass wall time, CPU)")
    print(f"[stage1.3] Summary written to {OUTPUT_JSON}")
    print("[stage1.3] STOP per Vinh's scope: no embedding cache, no probing, no geometry metrics, "
          "no scientific analysis performed.")


if __name__ == "__main__":
    main()
