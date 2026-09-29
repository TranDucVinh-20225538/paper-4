"""
Stage 3.0 -- Artifact Audit on the frozen Reference Embedding Artifact.

Per Vinh (2026-08-05): "Stage 3 khong nen bat dau bang Mahalanobis... Stage
3 nen la 3.0 Artifact Audit... Khong phan tich. Chi audit."

Five checks, all data-integrity / traceability, none interpretive:
  1. File-integrity re-verification (SHA256 of the frozen files vs manifest).
  2. No duplicate rows (paths, embedding vectors).
  3. No missing/corrupted samples (row count, NaN, Inf, zero-norm) re-checked
     from the on-disk frozen files, not from Stage 2.1's in-memory state.
  4. Random-sample visual inspection -- montage of source images sent to Vinh.
  5. Reproducibility spot-check -- re-run the same sampled images through the
     Reference Instrument fresh, compare to the stored embedding.

Explicitly out of scope: Mahalanobis, probe, LID, spectral-decay slope,
energy, ViM, KDE, or any hypothesis-testing statistic. This script does not
interpret embedding content, only whether the artifact is intact and
traceable to its source images.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import data_roots

CSG_ROOT = data_roots.root("csg")
PANDERM_REPO = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceEmbedding"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
AUDIT_REPORT = OUTPUT_DIR / "stage3_0_artifact_audit_report.json"
MONTAGE_PNG = OUTPUT_DIR / "stage3_0_sample_montage.png"

N_SAMPLES = 12
RANDOM_SEED = 42
REPRO_ATOL = 1e-4  # deterministic CPU eval-mode forward pass -- expect near-exact match

sys.path.insert(0, str(CSG_ROOT))
sys.path.insert(0, str(PANDERM_REPO))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_1_file_integrity() -> dict:
    with open(ARTIFACT_DIR / "manifest.json") as f:
        manifest = json.load(f)
    result = {}
    all_ok = True
    for fname in ["embeddings.npy", "labels.csv", "metadata.csv"]:
        recomputed = _sha256_file(ARTIFACT_DIR / fname)
        expected = manifest["files"][fname]["sha256"]
        ok = recomputed == expected
        all_ok = all_ok and ok
        result[fname] = {"expected": expected, "recomputed": recomputed, "match": ok}
    return {"pass": all_ok, "per_file": result, "manifest_status": manifest.get("status")}


def check_2_no_duplicates(embeddings: np.ndarray, metadata: pd.DataFrame) -> dict:
    n = len(metadata)
    dup_paths = int(metadata["path"].duplicated().sum())
    # Duplicate embedding rows: hash each row's bytes, count collisions.
    row_hashes = [hashlib.sha256(embeddings[i].tobytes()).hexdigest() for i in range(n)]
    n_unique_rows = len(set(row_hashes))
    dup_embeddings = n - n_unique_rows
    return {
        "pass": bool(dup_paths == 0 and dup_embeddings == 0),
        "n_rows": n,
        "duplicate_paths": dup_paths,
        "duplicate_embedding_rows": dup_embeddings,
    }


def check_3_no_missing_corrupted(embeddings: np.ndarray, metadata: pd.DataFrame, labels: pd.DataFrame) -> dict:
    n_rows_expected = 7365
    row_count_ok = embeddings.shape[0] == n_rows_expected == len(metadata) == len(labels)
    has_nan = bool(np.isnan(embeddings).any())
    has_inf = bool(np.isinf(embeddings).any())
    norms = np.linalg.norm(embeddings, axis=1)
    zero_norm_rows = int((norms < 1e-6).sum())
    return {
        "pass": bool(row_count_ok and not has_nan and not has_inf and zero_norm_rows == 0),
        "row_count": int(embeddings.shape[0]),
        "row_count_expected": n_rows_expected,
        "row_count_ok": bool(row_count_ok),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "zero_norm_rows": zero_norm_rows,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def build_montage(sample_rows: pd.DataFrame, out_path: Path) -> None:
    thumb_size = 180
    cols = 4
    rows = (len(sample_rows) + cols - 1) // cols
    label_h = 28
    canvas = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (_, row) in enumerate(sample_rows.iterrows()):
        r, c = divmod(i, cols)
        img = Image.open(row["path"]).convert("RGB").resize((thumb_size, thumb_size))
        x, y = c * thumb_size, r * (thumb_size + label_h)
        canvas.paste(img, (x, y))
        caption = f"row {row['row_index']} | {row['domain']} | {row['label_name']}"
        draw.rectangle([x, y + thumb_size, x + thumb_size, y + thumb_size + label_h], fill="black")
        draw.text((x + 4, y + thumb_size + 6), caption, fill="white")

    canvas.save(out_path)


@torch.no_grad()
def check_5_reproducibility_spotcheck(sample_rows: pd.DataFrame, stored_embeddings: np.ndarray) -> dict:
    from models import get_encoder  # PanDerm repo, unmodified -- same instrument as extraction

    class _Args:
        pretrained_checkpoint = str(CHECKPOINT)

    model, transform = get_encoder(_Args(), "PanDerm_Large_LP")
    model.eval()

    per_row = []
    all_close = True
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        image = Image.open(row["path"]).convert("RGB")
        x = transform(image).unsqueeze(0)
        fresh_embedding = model.forward_features(x, is_train=False).squeeze(0).numpy()
        stored = stored_embeddings[int(row["row_index"])]
        max_abs_diff = float(np.max(np.abs(fresh_embedding - stored)))
        cosine_sim = float(
            np.dot(fresh_embedding, stored) / (np.linalg.norm(fresh_embedding) * np.linalg.norm(stored))
        )
        row_ok = max_abs_diff < REPRO_ATOL
        all_close = all_close and row_ok
        per_row.append({
            "row_index": int(row["row_index"]),
            "path": row["path"],
            "max_abs_diff": max_abs_diff,
            "cosine_similarity": cosine_sim,
            "pass": row_ok,
        })
        print(f"[stage3.0] spot-check row {row['row_index']}: max_abs_diff={max_abs_diff:.2e}, "
              f"cosine_sim={cosine_sim:.8f}, pass={row_ok}")

    return {"pass": bool(all_close), "atol": REPRO_ATOL, "per_row": per_row}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[stage3.0] === Check 1: file-integrity re-verification ===")
    c1 = check_1_file_integrity()
    print(f"  pass={c1['pass']}")

    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
    labels = pd.read_csv(ARTIFACT_DIR / "labels.csv")
    metadata = pd.read_csv(ARTIFACT_DIR / "metadata.csv")

    print("[stage3.0] === Check 2: no duplicates ===")
    c2 = check_2_no_duplicates(embeddings, metadata)
    print(f"  pass={c2['pass']}  {c2}")

    print("[stage3.0] === Check 3: no missing/corrupted samples ===")
    c3 = check_3_no_missing_corrupted(embeddings, metadata, labels)
    print(f"  pass={c3['pass']}  {c3}")

    print(f"[stage3.0] === Check 4: random-sample visual inspection (n={N_SAMPLES}, seed={RANDOM_SEED}) ===")
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(len(metadata), size=N_SAMPLES, replace=False)
    sample_rows = metadata.iloc[sorted(sample_idx)].reset_index(drop=True)
    build_montage(sample_rows, MONTAGE_PNG)
    print(f"  Montage written: {MONTAGE_PNG}")
    c4 = {
        "pass": MONTAGE_PNG.is_file(),
        "n_samples": N_SAMPLES,
        "seed": RANDOM_SEED,
        "sampled_rows": sample_rows[["row_index", "domain", "label_name", "path"]].to_dict("records"),
        "montage_path": str(MONTAGE_PNG),
    }

    print("[stage3.0] === Check 5: reproducibility spot-check ===")
    c5 = check_5_reproducibility_spotcheck(sample_rows, embeddings)
    print(f"  pass={c5['pass']}")

    all_pass = c1["pass"] and c2["pass"] and c3["pass"] and c4["pass"] and c5["pass"]

    report = {
        "protocol_step": "Stage 3.0 -- Artifact Audit",
        "date": "2026-08-05",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifact_dir": str(ARTIFACT_DIR),
        "check_1_file_integrity": c1,
        "check_2_no_duplicates": c2,
        "check_3_no_missing_corrupted": c3,
        "check_4_random_sample_visual_inspection": c4,
        "check_5_reproducibility_spotcheck": c5,
        "gate_result": "PASS" if all_pass else "FAIL",
        "not_computed_here": "No probe, no Mahalanobis or other distance scorer, no geometry metric, "
                              "no PCA/UMAP, no hypothesis-testing statistic. Audit only.",
    }
    with open(AUDIT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[stage3.0] === ARTIFACT AUDIT: {report['gate_result']} ===")
    print(f"[stage3.0] Report written: {AUDIT_REPORT}")
    print("[stage3.0] STOP: no Mahalanobis, probe, geometry, or statistics computed.")


if __name__ == "__main__":
    main()
