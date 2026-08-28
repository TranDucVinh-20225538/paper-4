# Duplicate Registry — Notes

Produced by Stage 3.0b, 2026-08-05. Evidence only — no dataset
modification. `ReferenceEmbedding/` was read read-only; nothing in it was changed.

## Method

1. Every row's embedding vector (from the frozen `embeddings.npy`) was SHA256-hashed; rows sharing
   a hash form a group (bitwise-identical embedding vectors).
2. **17 groups found, 34 rows involved, all groups size 2
   (pairs).**
3. For every group, the source images were independently pixel-compared (`numpy.array_equal` on
   the decoded RGB arrays, not a file-hash comparison — one pair had different file bytes but
   identical pixel content, i.e. a re-encoding of the same image data).
4. **All 17/17 groups confirmed pixel-identical.**

## Wording (corrected 2026-08-05, per Vinh)

These are **pixel-identical images** — that is what pixel comparison actually proves. This registry
does **not** claim they are "the same photograph," "the same lesion," or "the same patient capture"
— those would be claims about real-world provenance that pixel identity alone does not establish,
even though it is consistent with them. Only the pixel-level fact is asserted here.

## Label conflict

**13 of 17 duplicate-image groups have differing `label_name`
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
