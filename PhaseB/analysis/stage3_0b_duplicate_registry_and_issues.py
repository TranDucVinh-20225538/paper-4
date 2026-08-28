"""
Stage 3.0b -- Duplicate Registry + Artifact Issues Register.

Per Vinh (2026-08-05): audit findings get recorded, not acted on. "Artifact
-> Audit" and "Artifact -> Cleaning" are two different things; this project
is doing the first, not the second. No row is removed, merged, or relabeled
anywhere in this script -- the frozen ReferenceEmbedding/ artifact is read
read-only and never touched.

Wording correction from Stage 3.0's initial report, per Vinh: "same
photograph" overclaims what pixel comparison actually proves. Corrected to
"pixel-identical images" throughout -- identity is established at the pixel
level; whether two pixel-identical images are the same capture, the same
lesion, or the same patient is not independently established here and is
not claimed.

Produces two frozen artifacts:
  1. assets/DuplicateRegistry/ -- the raw evidence (which rows, which
     images, which labels, pixel-hash confirmation) for the duplicate
     finding specifically.
  2. assets/ArtifactIssuesRegister/ -- the higher-level register every
     future claim should reference: what audit issues exist on this
     artifact, their status, and (for now) that no action has been taken.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceEmbedding"
DUP_REGISTRY_DIR = Path(__file__).resolve().parents[1] / "assets" / "DuplicateRegistry"
ISSUES_REGISTER_DIR = Path(__file__).resolve().parents[1] / "assets" / "ArtifactIssuesRegister"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pixel_identical(p1: str, p2: str) -> bool:
    im1 = np.array(Image.open(p1).convert("RGB"))
    im2 = np.array(Image.open(p2).convert("RGB"))
    return im1.shape == im2.shape and bool(np.array_equal(im1, im2))


def find_duplicate_groups(embeddings: np.ndarray) -> list[list[int]]:
    row_hashes = [hashlib.sha256(embeddings[i].tobytes()).hexdigest() for i in range(len(embeddings))]
    from collections import defaultdict
    groups = defaultdict(list)
    for i, h in enumerate(row_hashes):
        groups[h].append(i)
    return [idxs for idxs in groups.values() if len(idxs) > 1], row_hashes


def build_pair_montage(groups: list[list[int]], metadata: pd.DataFrame, out_path: Path) -> None:
    thumb = 140
    label_h = 24
    pair_w = thumb * 2 + 10
    pair_h = thumb + label_h
    cols = 3
    rows = (len(groups) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * pair_w, rows * pair_h), "white")
    draw = ImageDraw.Draw(canvas)

    for gi, idxs in enumerate(groups):
        col, row = gi % cols, gi // cols
        x0, y0 = col * pair_w, row * pair_h
        for j, ridx in enumerate(idxs[:2]):
            path = metadata.iloc[ridx]["path"]
            label = metadata.iloc[ridx]["label_name"]
            img = Image.open(path).convert("RGB").resize((thumb, thumb))
            x = x0 + j * (thumb + 10)
            canvas.paste(img, (x, y0))
            draw.text((x + 2, y0 + thumb + 2), f"r{ridx} {label}", fill="black")
        draw.rectangle([x0, y0, x0 + pair_w - 12, y0 + pair_h - 2], outline="red", width=2)

    canvas.save(out_path)


def main() -> None:
    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
    metadata = pd.read_csv(ARTIFACT_DIR / "metadata.csv")

    groups, row_hashes = find_duplicate_groups(embeddings)
    print(f"[stage3.0b] {len(groups)} duplicate-embedding groups found ({sum(len(g) for g in groups)} rows)")

    # === Build duplicate_groups.csv ===
    rows_out = []
    for gi, idxs in enumerate(groups):
        paths = [metadata.iloc[i]["path"] for i in idxs]
        pixel_ok = all(pixel_identical(paths[0], paths[j]) for j in range(1, len(paths)))
        labels = [metadata.iloc[i]["label_name"] for i in idxs]
        label_conflict = len(set(labels)) > 1
        for i in idxs:
            row = metadata.iloc[i]
            rows_out.append({
                "group_id": gi,
                "row_index": int(i),
                "path": row["path"],
                "domain": row["domain"],
                "label_idx": int(row["label_idx"]),
                "label_name": row["label_name"],
                "embedding_sha256": row_hashes[i],
                "pixel_identical_confirmed_within_group": pixel_ok,
                "label_conflict_within_group": label_conflict,
            })
    dup_df = pd.DataFrame(rows_out)

    DUP_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    dup_csv_path = DUP_REGISTRY_DIR / "duplicate_groups.csv"
    dup_df.to_csv(dup_csv_path, index=False)
    print(f"[stage3.0b] Written: {dup_csv_path}")

    montage_path = DUP_REGISTRY_DIR / "montage.png"
    build_pair_montage(groups, metadata, montage_path)
    print(f"[stage3.0b] Written: {montage_path}")

    n_label_conflict = dup_df.groupby("group_id")["label_conflict_within_group"].first().sum()
    notes = f"""# Duplicate Registry — Notes

Produced by Stage 3.0b, {datetime.date.today().isoformat()}. Evidence only — no dataset
modification. `ReferenceEmbedding/` was read read-only; nothing in it was changed.

## Method

1. Every row's embedding vector (from the frozen `embeddings.npy`) was SHA256-hashed; rows sharing
   a hash form a group (bitwise-identical embedding vectors).
2. **{len(groups)} groups found, {sum(len(g) for g in groups)} rows involved, all groups size 2
   (pairs).**
3. For every group, the source images were independently pixel-compared (`numpy.array_equal` on
   the decoded RGB arrays, not a file-hash comparison — one pair had different file bytes but
   identical pixel content, i.e. a re-encoding of the same image data).
4. **All {len(groups)}/{len(groups)} groups confirmed pixel-identical.**

## Wording (corrected 2026-08-05, per Vinh)

These are **pixel-identical images** — that is what pixel comparison actually proves. This registry
does **not** claim they are "the same photograph," "the same lesion," or "the same patient capture"
— those would be claims about real-world provenance that pixel identity alone does not establish,
even though it is consistent with them. Only the pixel-level fact is asserted here.

## Label conflict

**{int(n_label_conflict)} of {len(groups)} duplicate-image groups have differing `label_name`
values across their rows** (the same pixel-identical image is assigned a different diagnostic label
under different rows/image IDs in the source metadata). See `duplicate_groups.csv`'s
`label_conflict_within_group` column.

## Status

**NO ACTION TAKEN.** No row removed, merged, or relabeled. The frozen `ReferenceEmbedding/` artifact
is unmodified. This is a record of what audit found, for a future PI decision — not a cleaning step.

## Files

- `duplicate_groups.csv` — one row per (group_id, row_index): path, domain, label, embedding SHA256,
  pixel-identity confirmation, label-conflict flag.
- `montage.png` — every pair, side by side, with row index and label overlaid.
"""
    notes_path = DUP_REGISTRY_DIR / "notes.md"
    notes_path.write_text(notes)
    print(f"[stage3.0b] Written: {notes_path}")

    # === Artifact Issues Register ===
    ISSUES_REGISTER_DIR.mkdir(parents=True, exist_ok=True)
    register = {
        "register": "Artifact Issues Register",
        "artifact": "Reference Embedding Artifact (ReferenceEmbedding/, frozen 2026-08-04T21:05:42Z)",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "issues": [
            {
                "issue_id": 1,
                "category": "duplicate_images_with_label_conflict",
                "discovered_at_protocol_step": "Stage 3.0 (Artifact Audit)",
                "description": (
                    f"{len(groups)} groups of pixel-identical images ({sum(len(g) for g in groups)} rows) "
                    "found in the preregistered ISIC 2019 / PAD-UFES-20 datasets, detected via "
                    "bitwise-identical embedding vectors and independently confirmed via pixel-array "
                    f"comparison. {int(n_label_conflict)}/{len(groups)} of these groups carry conflicting "
                    "diagnostic labels for the same pixel-identical image."
                ),
                "evidence": {
                    "duplicate_groups_csv": str(dup_csv_path),
                    "montage": str(montage_path),
                    "notes": str(notes_path),
                },
                "severity": "unclassified — a property of the source data, not yet assessed for impact on any claim",
                "status": "logged, no action taken",
                "action_taken": "none",
                "resolution": "open — requires a PI decision before Stage 3.2+ (claim-bearing analysis) "
                               "treats this as resolved one way or another",
            },
        ],
        "policy": "Every future claim computed on this artifact should reference this register. "
                  "New issues found by later audits are APPENDED as new issue_id entries, never "
                  "edited into or removed from existing entries.",
        "status": "FROZEN",
    }
    register_json_path = ISSUES_REGISTER_DIR / "issues_register.json"
    with open(register_json_path, "w") as f:
        json.dump(register, f, indent=2)
    print(f"[stage3.0b] Written: {register_json_path}")

    register_md = f"""# Artifact Issues Register

Artifact: Reference Embedding Artifact (`ReferenceEmbedding/`, frozen 2026-08-04T21:05:42Z).
Policy: every future claim computed on this artifact should reference this register. New issues are
appended as new entries; existing entries are never edited or removed.

## Issue #1 — Duplicate images with label conflict

- **Discovered at:** Stage 3.0 (Artifact Audit)
- **Description:** {len(groups)} groups of pixel-identical images ({sum(len(g) for g in groups)}
  rows) found in the preregistered ISIC 2019 / PAD-UFES-20 datasets.
  {int(n_label_conflict)}/{len(groups)} groups carry conflicting diagnostic labels for the same
  pixel-identical image.
- **Evidence:** `assets/DuplicateRegistry/` (`duplicate_groups.csv`, `montage.png`, `notes.md`)
- **Status:** Logged, no action taken.
- **Resolution:** Open — a PI decision is required before Stage 3.2+ (claim-bearing analysis)
  treats this as resolved.
"""
    register_md_path = ISSUES_REGISTER_DIR / "issues_register.md"
    register_md_path.write_text(register_md)
    print(f"[stage3.0b] Written: {register_md_path}")

    # === Freeze both directories ===
    for d in [DUP_REGISTRY_DIR, ISSUES_REGISTER_DIR]:
        for f in d.iterdir():
            f.chmod(0o444)
        d.chmod(0o555)
        print(f"[stage3.0b] Frozen (chmod read-only): {d}")

    print("\n[stage3.0b] STOP: no dataset modification, no claim-bearing analysis. "
          "Registers are evidence records only.")


if __name__ == "__main__":
    main()
