# Stage 1 — Infrastructure

**Status: in progress.** Stage 0 gate reviewed (proceed as preregistered — `01_stage0_retrospective.md`).
Backbone dependency resolved (PanDerm). Checkpoint downloaded, verified, and validated as a
Reference Instrument through single-image inference, an infrastructure batch-inference smoke test,
AND — as of 2026-08-05 — a Scientific PASS batch-inference run through the actual preregistered
ISIC 2019 + PAD-UFES-20 dataset loader (`reference_instrument_validation.md` — all green). Remaining:
embedding extraction (full-scale), embedding cache, Reference-arm dry run — each gated on the
previous one passing, per Vinh's explicit sequencing.

No science happens in this stage — per Vinh's Phase B framing, this stage exists only to make sure
the harness works (Implementation Integrity Gate infrastructure, `22_Experimental_Design.md` §3.5
Gate 0), not to test anything.

## Checklist (source: `22_Experimental_Design.md` §2, §8; `16_Minimal_Sufficient_Experiment.md`
Week 1)

- [x] **Resolve backbone dependency — RESOLVED 2026-08-05: use PanDerm.** Per Vinh's decision rule
      (use PanDerm if accessible within a reasonable time, else fall back to CLIP ViT-B/16
      immediately, per `24_PreRegistration.md` §13's pre-specified trigger table): checked
      `github.com/SiyuanYan1/PanDerm` — checkpoints (`PanDerm` ViT-L/16, `PanDerm_Base` ViT-B/16,
      `DermLIP_PanDerm` ViT-B/16) are **direct-download**, no access-request form, no data-use
      agreement, no author email required. License: **CC-BY-NC-ND 4.0**, "may only be used for
      non-commercial academic research purposes with proper attribution" — matches this project
      exactly. Not a deviation (the trigger/response pair was already pre-registered); this is the
      resolution of an open dependency, not a change to the protocol.
  - `PanDerm` (ViT-L/16, primary candidate): https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view
  - `PanDerm_Base` (ViT-B/16): https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB/view
  - **ND (No-Derivatives) flag for later:** if the eventual paper's supplementary material would
    include the adapted/fine-tuned Intervention/Conventional/Adaptation-arm checkpoints, that may
    be in tension with "No Derivatives" — a Stage 3 publication-logistics question, not a Stage 1
    blocker; noted here so it isn't forgotten.
- [x] **Download checkpoint — DONE 2026-08-05.** PanDerm ViT-L/16, downloaded via `gdown` from the
      Google Drive link above to `PhaseB/assets/checkpoints/panderm_vitl16_checkpoint-499.pth`
      (1,418,820,654 bytes). SHA256, exact source URL, repo commit at download time, and download
      timestamp all frozen in `PhaseB/assets/panderm_vitl16_checkpoint_metadata.{md,json}` — per
      Vinh's requirement, not just "PanDerm" but a specific, checksummed, dated artifact. File not
      modified in any way after download (no conversion, no re-serialization).
- [x] **Verify checkpoint loads — DONE 2026-08-05.** `torch.load(..., map_location="cpu")` succeeds:
      raw state_dict, 458 tensors / 354,660,064 total params. `encoder.*` subtree (366 tensors /
      303,326,208 params) matches standard ViT-L/16 (24 blocks, dim 1024, patch 16, 197 tokens ⇒
      224×224 input) — this is the part relevant to downstream feature extraction. The remainder
      (`mask_token`, `rd_pos_embed`, `regresser.*`, `decoder.*`) are pretraining-only MIM
      objective components, expected to be absent from a downstream encoder-only load. Full detail:
      `PhaseB/assets/panderm_vitl16_checkpoint_metadata.md` §Structural verification.
- [x] **Instantiate encoder + single-image forward pass (Stage 1.2) — DONE 2026-08-05.** Used the
      *official* repo implementation only, no custom wrapper: cloned
      `github.com/SiyuanYan1/PanDerm` to `PhaseB/external/PanDerm` (same commit as the checkpoint
      metadata record), called `models.get_encoder(args, 'PanDerm_Large_LP')` — the exact
      function+model-name the repo's own `linear_eval.py` uses, and whose own
      `script/lp_reproduce.sh` references this project's exact checkpoint filename
      (`panderm_ll_data6_checkpoint-499.pth`). Forward pass via `model.forward_features(x,
      is_train=False)` (matching the repo's own `extract_features.py`), on one real project image
      (`DST-Skin/reader_images_96/ISIC_0034336.jpg`). Embedding shape **(1, 1024)**, no NaN/Inf.
      `strict=False` in the repo's own loading branch explained, not assumed — independently
      verified as 92 unexpected keys (`decoder.*`/`mask_token`/`rd_pos_embed`/`regresser.*`,
      pretraining-only, exactly matching the prediction already logged in the checkpoint metadata)
      and 2 missing keys (`head.weight`/`head.bias`, an unused 1000-way head never touched since
      `forward_features` is called, not `forward`). Full checklist:
      **`reference_instrument_validation.md`** (all five items PASS). Run log:
      `logs/stage1_2_run_2026-08-05.log`.
- [x] **Batch inference validation (Stage 1.3) — Engineering PASS 2026-08-05, Scientific PASS
      PENDING.** Representative batch (all 96 images in `DST-Skin/reader_images_96/`, both
      domains), 6 batches of 16, official encoder + official preprocessing (checked against the
      frozen `assets/reference_preprocessing.json` snapshot for drift — none found). All 5
      PI-specified criteria passed: shape always (16, 1024), no NaN, no Inf, no silent failures,
      throughput 5.208 img/s (CPU). No embeddings cached, no probe, no analysis. **Per Vinh's PI
      review: this proves the encoder pipeline is stable, but `reader_images_96/` is a legacy Paper
      2 reader-study folder, not the preregistered ISIC 2019 + PAD-UFES-20 split — does not satisfy
      the preregistered validation gate.** Stays valid as an infrastructure smoke test; full record
      unchanged at `reference_instrument_validation.md` §Stage 1.3,
      `outputs/stage1_3_batch_inference_validation.json`.
  - **Resolved — DONE 2026-08-05.** Official S3 download
    (`isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip`, 9.77 GB) failed twice
    with connection resets partway through (1.34 GB, then 1.55 GB — logged in `26_Lab_Notebook.md`
    with the actual `curl` errors, not silently retried into a false success). Vinh provided a local
    copy at `Paper4/archive/` (Kaggle-style, per-diagnosis-class folders, original `ISIC_XXXXXXX.jpg`
    filenames intact). **Independently verified, not just trusted:** every one of 25,331 images'
    folder-assigned label cross-checked against the officially-downloaded
    `ISIC_2019_Training_GroundTruth.csv` by image ID — 25,331/25,331 matched, 0 missing, 0
    mismatches. Symlinked (not copied) into `CSG-SKin/data/ISIC_2019_Training_Input/` — the flat
    layout `preprocess_metadata.py` expects. PAD-UFES-20 unchanged from the earlier
    byte-identical-verified copy. `master_metadata.csv` built by CSG-Skin's own, **unmodified**
    `src/datasets/preprocess_metadata.py`: **27,629 rows** (25,331 ISIC + 2,298 PAD-UFES-20) —
    exactly matching the preregistered dataset specification's testbed size (`22_Experimental_Design.md`
§2 / `Scientific_Hypothesis.md`'s Observation section — a protocol/data-scope fact, not a hypothesis
claim). Full build provenance (SHA256,
    row counts, timestamps, CSG-Skin commit, source verification method) frozen in
    `assets/master_metadata_build_provenance.json`.
- [x] **Dataset loader — resolved 2026-08-05: reuse CSG-Skin's own `SkinDataModule` +
      `build_id_ood_test_dataloaders`, unmodified.** ISIC 2019 (train split) + PAD-UFES-20, matching
      Papers 1–3's split protocol exactly (`22_Experimental_Design.md` §2) — same `SplitConfig`
      defaults (`random_state=42`, `isic_test_fraction=0.2`, `val_fraction=0.2`) their own scripts
      use, not re-derived. See `analysis/stage1_3_rerun_preregistered_split.py`.
- [x] **Dataset Validation gate (Stage 1.3a) — DONE 2026-08-05, PASS.** Standalone script
      (`analysis/stage1_3a_dataset_validation.py`), no encoder/checkpoint touched: 25,331 ISIC images
      on disk, 2,299 PAD-UFES images on disk, 27,629 metadata rows (25,331/2,298 by domain), split
      reproducible at seed=42 (recomputed twice, `isic_test_n=5067` — matching Paper 3's own
      `e2_distances` ID-split size exactly, a strong independent consistency check), preprocessing
      attached and matching the frozen snapshot. Self-describing report (`protocol_step`,
      `protocol_version`, `csg_skin_commit`, `panderm_repo_commit`, `panderm_preprocessing_sha256`)
      written to `outputs/dataset_validation_report.json`.
- [x] **Stage 1.3 rerun on the preregistered split — DONE 2026-08-05, GATE PASS (Scientific PASS).**
      Authorized after the Dataset Validation checkpoint. ID (ISIC-test) n=5,067, OOD (PAD-UFES)
      n=2,298; convenience batch = 8+8=16 (deterministic `shuffle=False` first-N sample). All 5
      criteria PASS: shape (16, 1024), no NaN, no Inf, no silent failure, 5.576 img/s (CPU). No
      cache, no probe, no geometry/PCA/UMAP/statistics — strictly infrastructure, per Vinh's explicit
      boundary. Full record: `reference_instrument_validation.md` §Stage 1.3 RE-RUN,
      `outputs/stage1_3_rerun_preregistered_split.json` (embeds the Stage 1.3a report verbatim, so
      the full chain is readable from one file).

  **Why attribute-injecting PanDerm's transform is not a protocol deviation (per Vinh's request
  this be spelled out, in case a reviewer later asks "why didn't you use the original
  preprocessing?"):**
  - Dataset split — **unchanged**. `build_id_ood_test_dataloaders(dm)` is CSG-Skin's own function,
    called with no modification to its code; it computes ID (ISIC-test) / OOD (PAD-UFES) membership
    exactly as Papers 1–3's own scripts do, from the same `master_metadata.csv` schema and the same
    `SplitConfig` defaults.
  - Sampling — **unchanged**. `shuffle=False` on both loaders, same as CSG-Skin's own construction;
    the convenience batch is a deterministic "first N" slice, not a different sampling scheme.
  - Labels — **unchanged**. `label_idx`/`domain` come from the same dataframe rows CSG-Skin's own
    code selects; nothing about class or domain assignment is touched.
  - **Only the image transform is replaced** — `dm.eval_transform` is a plain Python attribute that
    `build_id_ood_test_dataloaders` reads dynamically at call time (`SkinDataset(df,
    transform=datamodule.eval_transform)`); assigning it to PanDerm's own frozen, already-validated
    preprocessing (`assets/reference_preprocessing.json`) before calling that function is dependency
    injection, not a code change. CSG's own `build_val_transform_robust()` is tuned for *their* CNN
    (a different Reference Instrument); using it to feed PanDerm would silently mismatch the encoder
    under test against its own validated preprocessing pipeline — using PanDerm's own transform is
    the *correct* choice for testing PanDerm, not a shortcut around the correct one.
  - **No CSG-Skin source file is read from disk and re-written, no function body is copied or
    edited.** The only new code is Paper-4-owned: constructing `SkinDataModule`, setting one
    attribute, and calling their existing function.

  **Dataset Validation gate (per Vinh, run and must PASS before any encoder/forward-pass code
  executes):** ISIC image count on disk, PAD-UFES image count on disk, metadata row counts by
  domain, split reproducibility at the pinned seed (recomputed twice, index sets compared),
  preprocessing-attached check (`reference_preprocessing.json` match). Implemented as
  `validate_dataset()` in `analysis/stage1_3_rerun_preregistered_split.py`, raises before touching
  the encoder if any check fails.
- [ ] Feature extraction (forward pass only — no training in this stage).
- [ ] Embedding cache. **Per Vinh: its own separately-gated step, not bundled with extraction.**
- [ ] Reproduce baseline: confirm the Reference-arm pipeline runs end to end (Gate 0 dry run, not
      the Gate 0 pass/fail decision itself — that's Stage 2).

## Not in scope for this stage

Everything in `22_Experimental_Design.md` §3's Intervention/Conventional/Adaptation arms. No
adapter, no GRL, no orthogonality loss, no fine-tuning ladder. This stage only proves the pipe is
plumbed correctly.
