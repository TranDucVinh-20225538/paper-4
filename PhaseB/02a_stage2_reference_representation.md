# Stage 2 — Reference Representation

Introduced 2026-08-05, per Vinh's PI decision on Stage 1's completion. Source concepts:
`22_Experimental_Design.md` §3 (Reference arm — "frozen pretrained FM embeddings"), §3.5 (Gate 0).

**Framing (Vinh, verbatim intent):** the embedding this stage produces is not "an embedding cache."
It is a **primary research artifact** — every downstream measurement this project will ever compute
(Mahalanobis, probe, LID, spectral-decay slope, energy, ViM, KDE, cosine-to-centroid, etc.) is
computed *from* this one frozen artifact, not re-derived from images each time. It gets the same
treatment as the PanDerm checkpoint itself: identity frozen, provenance recorded, integrity gated,
never silently regenerated.

## Stage 2.1 — Extraction (current, authorized scope)

PI instruction, verbatim (2026-08-05): "Begin Stage 2.1 by extracting embeddings for the complete
preregistered ID and OOD datasets using the validated Reference Instrument. Produce a single frozen
Reference Embedding Artifact together with full provenance, integrity checks (shape, ordering,
label alignment, sample count, SHA256), and no scientific analyses. Do not compute probes, geometry
metrics, OOD scores, or statistics. Stop immediately after the Embedding Integrity Gate passes."

### Scope

- **Datasets:** the complete preregistered ID (ISIC-test, n=5,067) and OOD (PAD-UFES-20, n=2,298)
  splits — not a convenience batch. Same `CSG-Skin` `SkinDataModule` + `build_id_ood_test_dataloaders`
  loader, same `master_metadata.csv`, same seed=42 split already validated in Stage 1 (Dataset
  Validation gate, `dataset_validation_report.json`).
- **Instrument:** the exact Reference Instrument validated in Stage 1 — `models.get_encoder(args,
  'PanDerm_Large_LP')`, official PanDerm preprocessing (`assets/reference_preprocessing.json`), same
  checkpoint (`assets/panderm_vitl16_checkpoint_metadata.json`).
- **Output:** one frozen artifact containing, per image, in a fixed row order: the 1024-d embedding,
  class `label_idx`, domain (`isic`/`pad_ufes`), and source image path (for independent
  re-verification without re-running extraction).

### Embedding Integrity Gate (must pass before freeze; five checks, per Vinh)

1. **Shape** — embeddings array is exactly `(7365, 1024)`.
2. **Sample count** — 5,067 ID + 2,298 OOD = 7,365, matching Stage 1's Dataset Validation gate
   exactly (cross-checked against `dataset_validation_report.json`, not re-derived independently).
3. **Ordering preserved** — the embedding array's row order matches the labels/domain/path arrays'
   row order exactly (verified by construction: built in one lock-step loop, no reordering).
4. **Label alignment** — every collected `(label_idx, domain, path)` triple matches the underlying
   `SkinDataset.data` dataframe row at the same index, independently re-read from the dataframe, not
   just trusted from the collection loop.
5. **SHA256** — computed over the saved artifact file, frozen into the provenance record.

### Explicitly out of scope for Stage 2.1

No probe. No Mahalanobis or any other distance-based scorer. No geometry metric (condition number,
LID, spectral-decay slope, variance). No PCA/UMAP. No summary statistics beyond the five integrity
checks above (which are structural/bookkeeping checks, not inferential ones). No comparison to
Paper 3. Nothing here supports or contradicts the hypothesis — this stage produces the input data
those later, separately-authorized steps will consume.

## Result (executed 2026-08-05) — GATE PASS, 6/6

Extraction: 7,365 images (5,067 ID + 2,298 OOD), CPU, ~40 min wall time
(id: 3.13 img/s over 1616s; ood: 2.92 img/s over 788s — recorded for planning, not a benchmark
claim). Embedding Integrity Gate, final form (six checks, per Vinh's refined spec):

| # | Check | Result |
|---|---|---|
| 1 | Sample count (ISIC=5,067, PAD=2,298) matches protocol | **PASS** |
| 2 | Embedding tensor shape 7,365 × 1,024 | **PASS** |
| 3 | Ordering: metadata row *i* → embedding row *i*, no drift | **PASS** — proven by an *independent* re-derivation (fresh split reconstruction from `master_metadata.csv`, not read from the extraction run's own memory), compared element-by-element |
| 4 | Label alignment (image → label → embedding) | **PASS** — same independent comparison, `label_idx` array matched exactly |
| 5 | SHA256 (per file, not just the bundle) | **PASS** — `embeddings.npy`, `labels.csv`, `metadata.csv` each hashed separately, see `sha256.txt` |
| 6 | Manifest ("birth certificate") | **PASS** — `manifest.json` |

## Artifacts (canonical, frozen)

```
assets/reference_embeddings/ReferenceEmbedding/
    embeddings.npy   -- (7365, 1024) float32
    labels.csv        -- row_index, label_idx, label_name
    metadata.csv       -- row_index, path, domain, label_idx, label_name
    manifest.json        -- birth certificate (artifact/protocol_step/rows/dim/checkpoint/
                             preprocessing/dataset/repo commits/per-file SHA256/row-ordering
                             proof/created_at)
    sha256.txt             -- per-file checksums
```

Intermediate extraction-run record (kept, not canonical): `assets/reference_embeddings/
panderm_reference_embedding_artifact.npz` + `reference_embedding_artifact_provenance.json` — has
per-domain extraction throughput/timing not repeated in `manifest.json`.

Scripts: `analysis/stage2_1_extract_reference_embeddings.py` (extraction),
`analysis/stage2_1b_finalize_embedding_artifact.py` (restructure + independent re-verification).
Logs: `logs/stage2_1_run_2026-08-05.log`, `logs/stage2_1b_run_2026-08-05.log`. Gate report:
`outputs/embedding_integrity_gate_report.json`.

## Not yet authorized

Stage 2.2+ (if any further Reference Representation sub-steps are needed) and all of Stage 3
(probe/geometry/accessibility/reliability analysis on this artifact) — each requires its own PI
decision, per the pattern held since Stage 1. Every future computation on this data (Mahalanobis,
probe, LID, spectral decay, energy, ViM, KDE, etc.) reads from
`ReferenceEmbedding/embeddings.npy` + `metadata.csv` — none of it re-extracts from images.

## Related artifact: ReferenceTrainEmbedding (EA-01, executed 2026-08-05)

A second, distinct frozen embedding artifact — not part of Stage 2.1's original scope, produced
under `ExecutionAmendments/EA-01.md` after Group B's reconciliation note (see
`ReferenceArmBaseline/GroupB_protocol_reconciliation.md`) traced a fitting-population gap. Same
Reference Instrument, same integrity-gate discipline (shape, sample count, independent-reconstruction
ordering proof, label alignment, SHA256, manifest), same immutability treatment (chmod 444/555).
**It is not a second evaluation split and not a Stage 2.1 correction** — it exists solely so Group B's
training-population-dependent scorers have something to fit on. The two artifacts must not be
conflated; each is scope-locked in its own `manifest.json` (`population`/`purpose`/`not_for` fields).

| Artifact | Population | Used for |
|---|---|---|
| `ReferenceEmbedding` | Held-out ID (ISIC-test, n=5,067) + OOD (PAD-UFES, n=2,298) | Evaluation |
| `ReferenceTrainEmbedding` | ISIC 2019 train split (n=16,211) | Reference fitting (Group B scorers only) |

Gate result: PASS, 5/5 (shape `[16211, 1024]`; sample count n=16,211; ordering — independent fresh
split-reconstruction, matched; label alignment — all paths exist, matched; SHA256 per file). Script:
`analysis/ea01_extract_reference_train_embeddings.py`. Log: `logs/ea01_extraction_run_2026-08-05.log`.
Artifact: `assets/reference_embeddings/ReferenceTrainEmbedding/`.

Producing this artifact is infrastructure, not analysis: no Group B scorer has been computed here or
anywhere else. See `ExecutionAmendments/EA-01.md` for lifecycle status and the explicit PI stop
instruction pending artifact review.
