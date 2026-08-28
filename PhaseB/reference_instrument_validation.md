# Reference Instrument Validation — PanDerm ViT-L/16

Per Vinh (2026-08-05): PanDerm is not being treated as "a model" from this point forward, but as a
**Reference Instrument** for testing the locked hypothesis. This checklist is its validation
record, kept separate from the narrative logs (`26_Lab_Notebook.md`, `02_stage1_setup.md`) so the
instrument's status can be checked at a glance.

Scope: Stage 1.2 only — "instantiate the PanDerm encoder using the official repository
implementation, validate a single forward pass on one image, record embedding dimensionality and
preprocessing, then stop." No batch inference, no embedding cache, no probing, no scientific
analysis performed or implied by this checklist.

| Item | Status | Evidence |
|---|---|---|
| ✓ checkpoint | **PASS** | PanDerm ViT-L/16, `panderm_ll_data6_checkpoint-499.pth`, SHA256 `23405e92f853c6f7ac9da80ccf4cfde84d4feb22cd9d7b98c7e850e3875554b2`. Full identity: `assets/panderm_vitl16_checkpoint_metadata.md`. Unmodified since download. |
| ✓ architecture | **PASS** | Instantiated via the repo's own registered factory, `panderm_large_patch16_224()` (`external/PanDerm/classification/models/modeling_finetune.py`), called through the repo's own `models.get_encoder(args, 'PanDerm_Large_LP')` (`models/builder.py`) — the exact function+model-name their own `linear_eval.py` uses, and whose own reproduce script (`script/lp_reproduce.sh`) references this same checkpoint filename. VisionTransformer, patch 16, embed_dim 1024, depth 24, 16 heads, mlp_ratio 4 — matches the checkpoint's `encoder.*` structure exactly. No custom wrapper written. |
| ✓ forward pass | **PASS** | `model.forward_features(x, is_train=False)` — the same call the repo's own `extract_features.py::extract_features_from_dataloader` uses (not `model(x)`, which would route through an irrelevant, checkpoint-uninitialized 1000-way head left over from the class's `num_classes=1000` default — unused here, as it is in the repo's own extraction path). Output: no NaN, no Inf. |
| ✓ embedding dimension | **PASS** | **1024** (CLS token, post-final-LayerNorm) — output shape `(1, 1024)` for one image. Matches ViT-L's embed_dim. |
| ✓ preprocessing | **PASS** | Official pipeline, from `get_encoder`'s own return value, not re-derived: `Resize(256) → CenterCrop(224) → ToTensor() → Normalize(mean=(0.485,0.456,0.406), std=(0.228,0.224,0.225))` — ImageNet statistics (`get_eval_transforms(which_img_norm='imagenet')`). |

## Test conditions

- **Repo:** `PhaseB/external/PanDerm`, commit `fd7a80748ba7fc3e203fed88f909f4689d0d6f24` (same commit
  the checkpoint's identity record was frozen against).
- **Test image:** `DST-Skin/reader_images_96/ISIC_0034336.jpg` — a real image already in this
  project's own data (ISIC 2019 dermoscopy), not a synthetic/placeholder input. Original size
  600×450, RGB.
- **Environment:** `timm==0.9.16`, `open_clip_torch` installed to match the repo's own
  `requirements.txt` pin exactly (needed only because `models/builder.py` unconditionally imports
  both at module load time; neither package's code path is actually exercised by the
  `PanDerm_Large_LP` branch used here).
- **Script:** `analysis/stage1_2_encoder_forward_pass.py`. Run log:
  `logs/stage1_2_run_2026-08-05.log`.

## strict=False — explained, not assumed

The repo's own `PanDerm_Large_LP` loading branch (`models/builder.py`) uses
`model.load_state_dict(state_dict, strict=False)` after a bare `k.replace("encoder.", "")` key
strip. That is **their** choice for this checkpoint, reused as-is here — not a leniency this
project introduced. Independently verified (by set difference against the loaded model's own
`state_dict()` keys, not by a second load) what that leniency actually covers:

- **92 unexpected keys**, all under `decoder.*`, `mask_token`, `rd_pos_embed`, `regresser.*` —
  pretraining-only masked-image-modeling components (encoder + cross-attention regressor + pixel
  decoder), with no corresponding module in the plain `VisionTransformer` encoder. Exactly the 92
  = 458 total − 366 `encoder.*` tensors already predicted in
  `assets/panderm_vitl16_checkpoint_metadata.md` before this step ran — confirms that prediction
  rather than surprising it.
- **2 missing keys**: `head.weight`, `head.bias` — the `VisionTransformer`'s default
  `num_classes=1000` classification head, which the checkpoint (a pure self-supervised pretraining
  artifact) never had in the first place. Irrelevant here: this pipeline calls
  `forward_features()`, never `forward()`, so `head` is never evaluated.

No missing keys inside the `encoder.*` subtree — every encoder parameter the checkpoint saved has a
home in this model. Recorded here specifically so that a `Missing keys` / `Unexpected keys` message
seen later (e.g. during batch inference or a future re-run) is recognized as this same expected
pattern, not re-debugged from scratch.

## Explicitly not done in this validation (Stage 1.2)

Batch inference, embedding caching, any probe, any geometry/accessibility/reliability metric, any
comparison to Paper 3. Per Vinh: those wait for Stage 1.3 onward, gated on this checklist staying
green.

---

# Stage 1.3 — Batch Inference Validation (2026-08-05)

PI decision, verbatim: "Run official PanDerm preprocessing and encoder on a representative batch of
images to validate stable batched inference. Record tensor shape, NaN/Inf checks, failed-image
count, and throughput. Do not cache embeddings, run probes, compute geometry metrics, or perform
any scientific analysis. Stop immediately after the Batch Inference Validation gate passes."

**Gate result: PASS.**

| # | Criterion | Result |
|---|---|---|
| 1 | Embedding shape always N × 1024 | **PASS** — every batch: `(16, 1024)` |
| 2 | No NaN | **PASS** |
| 3 | No Inf | **PASS** |
| 4 | No sample fails silently | **PASS** — 0 load/preprocess failures, 0 zero-norm embeddings |
| 5 | Throughput | **5.208 images/sec** (96 images / 18.435s forward-pass wall time, CPU) |

## Test conditions

- **Representative batch:** all 96 images in `DST-Skin/reader_images_96/` — 26 ISIC 2019
  (dermoscopy) + 70 PAD-UFES-20-style (`PAT_*`, clinical smartphone) — a real mix of both domains
  this project's acquisition-shift testbed is built on, not a synthetic or single-domain sample.
- **Batching:** 6 batches of 16 (last batch full; 96 divides evenly), true batched forward passes
  (`model.forward_features(batch, is_train=False)` on a stacked `(16, 3, 224, 224)` tensor), not a
  python loop over single images.
- **Preprocessing drift check:** before running, the live `get_eval_transforms()` output was
  asserted against the frozen `assets/reference_preprocessing.json` snapshot (resize size,
  interpolation, crop size, normalize mean/std) — passed, no drift since Stage 1.2.
- **Script:** `analysis/stage1_3_batch_inference_validation.py`. Full per-batch report:
  `outputs/stage1_3_batch_inference_validation.json`. Run log:
  `logs/stage1_3_run_2026-08-05.log`.
- **Device:** CPU (no CUDA on this machine) — the 5.2 img/s figure is a CPU number, recorded for
  planning purposes only, not a benchmark claim.

## Explicitly not done in this validation (Stage 1.3)

No embeddings written to disk (held in memory per batch, discarded after the checks above — `del
embeddings, batch` after each batch, no accumulation). No dataset loader / `DataLoader`
infrastructure built (that remains a separate, later Stage 1 checklist item). No probe, no geometry
metric, no comparison to Paper 3, no scientific analysis of any kind.

**Status note (2026-08-05):** this run stays valid as an infrastructure smoke test, but per Vinh's
PI review it does not satisfy the preregistered validation gate (`DST-Skin/reader_images_96/` is a
Paper 2 reader-study folder, not the preregistered ISIC 2019 + PAD-UFES-20 split). Superseded, for
scientific-gate purposes, by the section below — not edited or removed, since it's still the correct
record of what it actually tested.

---

# Stage 1.3 RE-RUN — Batch Inference Validation, preregistered split (2026-08-05)

PI decision, verbatim: "Proceed to the preregistered Stage 1.3 rerun using the validated CSG-Skin
dataset loader and the official PanDerm preprocessing. Limit execution strictly to batch inference
validation. Produce no cached embeddings, no derived datasets, and no scientific analyses. Stop
immediately after the Reference Instrument Validation checklist is completed on the preregistered
split."

**Gate result: PASS — Scientific PASS, not just Engineering PASS.**

| # | Criterion | Result |
|---|---|---|
| 1 | Embedding shape always N × 1024 | **PASS** — `(16, 1024)` |
| 2 | No NaN | **PASS** |
| 3 | No Inf | **PASS** |
| 4 | No sample fails silently | **PASS** — 0 zero-norm embeddings |
| 5 | Throughput | **5.576 images/sec** (16 images / 2.870s forward-pass wall time, CPU) |

## What makes this the scientific version, not the engineering one

- **Dataset:** CSG-Skin's own `SkinDataModule` + `build_id_ood_test_dataloaders`
  (`src/datasets/skin_dataset.py`, `splits.py`), imported and called **unmodified** — the exact
  loader Papers 1–3's own scripts use. Preprocessing (PanDerm's frozen, official transform) attached
  via attribute injection, not a code fork — rationale spelled out in `02_stage1_setup.md`.
- **Split:** `master_metadata.csv` (27,629 rows — 25,331 ISIC 2019 + 2,298 PAD-UFES-20, matching the
  preregistered dataset specification's testbed size), CSG-Skin's own `SplitConfig` defaults
  (`random_state=42`, `isic_test_fraction=0.2`, `val_fraction=0.2`). ID (ISIC-test) dataset size
  **5,067** — independently matches Paper 3's own `results/e2_distances/*.npz` `s_id` array size
  exactly. OOD (PAD-UFES) dataset size **2,298**.
- **Gated on Stage 1.3a:** this script refuses to run unless `outputs/dataset_validation_report.json`
  shows `status: "PASS"` — a real dependency check (`require_dataset_validation_passed()`), not a
  convention. That report is embedded verbatim in this run's own output JSON, so the full chain is
  readable from one file.
- **Convenience batch:** first 8 rows of the ID loader + first 8 of the OOD loader
  (`shuffle=False` on both — deterministic, reproducible, not randomly sampled), one true 16-image
  batched forward pass.

## Test conditions

- **Script:** `analysis/stage1_3_rerun_preregistered_split.py`. Full output:
  `outputs/stage1_3_rerun_preregistered_split.json`. Run log: `logs/stage1_3_rerun_2026-08-05.log`.
- **Device:** CPU (no CUDA) — throughput recorded for planning only, not a benchmark claim.

## Explicitly not done in this run

No embedding cache, no derived dataset, no probe, no Mahalanobis, no geometry metric, no PCA, no
UMAP, no statistics of any kind — per Vinh's explicit boundary for this step. Infrastructure only.

## Artifact provenance (per Vinh's three-question rule, going forward)

For `outputs/stage1_3_rerun_preregistered_split.json`:
1. **Produced by which protocol step?** Stage 1.3 (rerun), gated on Stage 1.3a passing.
2. **Depends on which prior artifacts?** `dataset_validation_report.json` (Stage 1.3a),
   `master_metadata_build_provenance.json` (dataset build), `panderm_vitl16_checkpoint_metadata.json`
   (checkpoint identity), `reference_preprocessing.json` (preprocessing pin) — all four referenced or
   embedded in this run's output or its logged inputs.
3. **Reproducible from scratch without manual steps?** Yes, given the same inputs on disk
   (`CSG-SKin/data/master_metadata.csv`, the PanDerm checkpoint, the cloned PanDerm repo at its
   pinned commit): `python3 analysis/stage1_3a_dataset_validation.py && python3
   analysis/stage1_3_rerun_preregistered_split.py`, no manual intervention. The one non-scripted step
   upstream is the ISIC 2019 image source itself (provided locally after the official download failed
   twice — see `26_Lab_Notebook.md`), which is why `master_metadata_build_provenance.json` documents
   that provenance explicitly rather than assuming it away.
