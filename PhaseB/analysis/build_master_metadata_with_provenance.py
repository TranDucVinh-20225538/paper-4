"""
Build master_metadata.csv via CSG-Skin's own, UNMODIFIED
src/datasets/preprocess_metadata.py, then freeze its provenance.

Per Vinh (2026-08-05): the build script itself, its output, and the exact
source data it consumed must all be identifiable later -- "which version of
master_metadata.csv was this." Runs preprocess_metadata.py as a subprocess
(their own CLI entry point, `if __name__ == "__main__": main()` -- treated
as a black box, not imported and re-driven line by line) so this script
never touches their code, only invokes it.

Writes assets/master_metadata_build_provenance.json.
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
BUILD_SCRIPT = CSG_ROOT / "src" / "datasets" / "preprocess_metadata.py"
OUTPUT_CSV = CSG_ROOT / "data" / "master_metadata.csv"
ISIC_INPUT_DIR = CSG_ROOT / "data" / "ISIC_2019_Training_Input"  # symlinks into Paper4/archive/
ARCHIVE_SOURCE = data_roots.root("archive")
GROUNDTRUTH_CSV = CSG_ROOT / "data" / "ISIC_2019_Training_GroundTruth.csv"
PROVENANCE_JSON = Path(__file__).resolve().parents[1] / "assets" / "master_metadata_build_provenance.json"


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


def main() -> None:
    csg_commit = _git_commit(CSG_ROOT)
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print(f"[build] Running {BUILD_SCRIPT} (CSG-Skin's own script, subprocess, unmodified)...")
    result = subprocess.run(
        [sys.executable, "-m", "src.datasets.preprocess_metadata"],
        cwd=str(CSG_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"preprocess_metadata.py failed (exit {result.returncode})")

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if not OUTPUT_CSV.is_file():
        raise FileNotFoundError(f"Expected output not found: {OUTPUT_CSV}")

    import pandas as pd

    df = pd.read_csv(OUTPUT_CSV)
    row_counts_by_domain = df["domain"].value_counts().to_dict() if "domain" in df.columns else {}

    provenance = {
        "artifact": "master_metadata.csv",
        "output_path": str(OUTPUT_CSV),
        "generated_by_script": str(BUILD_SCRIPT),
        "generated_by_command": "python -m src.datasets.preprocess_metadata (subprocess, unmodified)",
        "csg_skin_commit": csg_commit,
        "build_started_utc": started_at,
        "build_finished_utc": finished_at,
        "row_count": len(df),
        "row_count_by_domain": row_counts_by_domain,
        "output_sha256": _sha256(OUTPUT_CSV),
        "source_data": {
            "isic_2019_images": {
                "source": str(ARCHIVE_SOURCE),
                "note": (
                    "Official S3 download (isic-challenge-data.s3.amazonaws.com/2019/"
                    "ISIC_2019_Training_Input.zip) failed twice with connection resets "
                    "partway through a 9.1GB transfer. Vinh provided a local copy instead, "
                    "organized into per-diagnosis-class folders (Kaggle-style redistribution, "
                    "not the official flat archive layout) at this path. Symlinked into "
                    f"{ISIC_INPUT_DIR} (flat, matching preprocess_metadata.py's expected layout) "
                    "-- no files copied or modified, only linked."
                ),
                "verification_method": (
                    "Every image's folder-assigned label cross-checked against the official "
                    "ISIC_2019_Training_GroundTruth.csv (one-hot decoded) by image ID -- not "
                    "just a file-count or checksum match, but a full content-identity check "
                    "against ground truth."
                ),
                "verification_result": "25331/25331 images matched officially-downloaded "
                                        "GroundTruth exactly; 0 missing; 0 label mismatches.",
                "verification_groundtruth_csv": str(GROUNDTRUTH_CSV),
                "verification_groundtruth_csv_source": "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv",
            },
            "pad_ufes20_source": "DST-Skin/data/raw/pad_ufes20/ (confirmed byte-identical by MD5 to an earlier working copy under the author's home directory before copying)",
        },
        "stdout": result.stdout,
    }

    PROVENANCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_JSON, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\n[build] master_metadata.csv: {len(df)} rows, sha256={provenance['output_sha256']}")
    print(f"[build] Row counts by domain: {row_counts_by_domain}")
    print(f"[build] Provenance written: {PROVENANCE_JSON}")


if __name__ == "__main__":
    main()
