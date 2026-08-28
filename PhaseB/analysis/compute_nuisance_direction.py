"""
Nuisance direction w -- Stage 4's first artifact-producing step (Execution Plan v2, Step 2).

Not adapter training. Computes the fixed target direction w = unit(mu_isic - mu_pad_ufes) that the
Intervention arm's adapter will later be regularized against (Stage4_Intervention_Method_Specification.md
SS1-2). Closed-form, deterministic, no optimization, no gradient, no training -- kept as its own
artifact per Vinh's explicit instruction (2026-08-05), separate from adapter code, so a later Gate 1
failure or Execution Amendment can be traced to direction estimation, adapter implementation, or
training individually rather than a single tangled pipeline.

Population: ReferenceTrainEmbedding (ISIC-train, n=16211) for mu_isic; ReferenceEmbedding's
PAD-UFES rows (n=2298) for mu_pad_ufes -- same split already verified in
Stage4_Intervention_Method_Specification.md SS1 (CSG-Skin's own precedent reuses the full PAD-UFES
population for domain-training-signal purposes; no new artifact/EA required).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings"
TRAIN_DIR = ASSETS / "ReferenceTrainEmbedding"
EVAL_DIR = ASSETS / "ReferenceEmbedding"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "NuisanceDirection"
CONTRACT_PATH = OUTPUT_DIR / "execution_contract.md"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_and_load(artifact_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Same integrity-first pattern as groupB_mahalanobis_family.py -- re-verify SHA256 against
    the artifact's own frozen manifest before use, not assumed intact."""
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


def main() -> None:
    z_train, meta_train, manifest_train = _verify_and_load(TRAIN_DIR)
    n_isic = len(meta_train)
    print(f"[nuisance] ReferenceTrainEmbedding verified: n={n_isic} (all ISIC), sha256 OK")

    z_eval, meta_eval, manifest_eval = _verify_and_load(EVAL_DIR)
    pad_mask = (meta_eval["domain"] == "pad_ufes").to_numpy()
    z_pad = z_eval[pad_mask]
    n_pad = int(pad_mask.sum())
    print(f"[nuisance] ReferenceEmbedding verified: n_pad_ufes={n_pad}, sha256 OK")

    mu_isic = z_train.astype(np.float64).mean(axis=0)
    mu_pad_ufes = z_pad.astype(np.float64).mean(axis=0)
    w_raw = mu_isic - mu_pad_ufes
    w_raw_norm = float(np.linalg.norm(w_raw))
    w = (w_raw / w_raw_norm).astype(np.float32)

    print(f"[nuisance] ||w_raw|| = {w_raw_norm:.6f}")
    print(f"[nuisance] ||w|| (post-normalization) = {float(np.linalg.norm(w)):.6f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "nuisance_direction.npz"
    np.savez(
        npz_path,
        w=w,
        w_raw=w_raw.astype(np.float32),
        mu_isic=mu_isic.astype(np.float32),
        mu_pad_ufes=mu_pad_ufes.astype(np.float32),
    )

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {
        "artifact": "Nuisance Direction",
        "protocol_step": "Stage 4 Execution Plan v2, Step 2",
        "execution_contract": str(CONTRACT_PATH),
        "produced_by_script": str(Path(__file__).resolve()),
        "method": "closed-form, deterministic -- no optimization, no gradient, no training",
        "formula": "w = unit(mean(ReferenceTrainEmbedding) - mean(ReferenceEmbedding[domain=='pad_ufes']))",
        "fit_artifact_isic": {"path": str(TRAIN_DIR), "manifest_status": manifest_train["status"], "n": n_isic},
        "fit_artifact_pad_ufes": {"path": str(EVAL_DIR), "manifest_status": manifest_eval["status"], "n": n_pad},
        "embedding_dim": int(w.shape[0]),
        "w_raw_norm": w_raw_norm,
        "w_norm_post_normalization": float(np.linalg.norm(w)),
        "files": {
            "nuisance_direction.npz": {"sha256": _sha256_file(npz_path)},
        },
        "not_computed_here": "No adapter, no training, no Gate 0, no Gate 1, no measurement suite. "
                              "Direction estimation only.",
        "created_at": now,
        "status": "PENDING_REVIEW",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUTPUT_DIR / "sha256.txt").write_text(
        f"{manifest['files']['nuisance_direction.npz']['sha256']}  nuisance_direction.npz\n"
    )

    print(f"\n[nuisance] DONE. Output written to {OUTPUT_DIR} -- NOT frozen "
          "(pending review + sanity check, same precedent as GroupBExecution/).")


if __name__ == "__main__":
    main()
