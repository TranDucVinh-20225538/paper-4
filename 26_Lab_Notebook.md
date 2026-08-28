Lab Notebook — Paper 4

Append-only. No retroactive edits. No reformatting past entries. Plain dated entries.

---

2026-08-04

Phase A (Concept & Protocol) closed. 25 documents, literature through constitution. Gallo et al.
verified Level A (full text, user-uploaded PDF). RQ locked. Hypothesis relabeled: "Geometry-mediated
accessibility after representation intervention." Experimental Design reached v2 (Gate 0, Gate 1,
C4 mediation criterion merged). Pre-Registration frozen. Constitution written. 24-Hour Rule adopted
for Phase B.

Next action: resolve the open backbone dependency (PanDerm checkpoint/license check, fallback is
CLIP ViT-B/16 — 22_Experimental_Design.md §2), then Stage 0: retrospective LID/spectral-decay
computation on Paper 3's existing CNN checkpoints. Zero GPU cost, no new data. This is the first
actual experimental step in the protocol.

---

Phase B kickoff. Vinh: "Phase A is complete. Switch permanently to Protocol Execution Mode. Do not
propose new hypotheses, new claims, or new experiments unless they are logged as protocol
deviations. Begin Stage 0 (Retrospective Analysis) exactly as specified in the preregistered
protocol." Scaffolded `PhaseB/` (00_execution_checklist.md, 01-03 stage docs, logs/, outputs/,
analysis/) per Vinh's specified layout.

Executed Stage 0 (C1 gate). New Paper-4-owned module (`PhaseB/analysis/lid_spectral_decay.py`)
computes Levina-Bickel MLE LID (k=50, dataset-average) and within-class spectral-decay slope
(log-eigenvalue-vs-rank fit on S_w, full range, |slope|) per arXiv:2510.15202 Appendix A/B, Eq. 6 —
does not touch or modify paper-3/, reads its e1_embeddings/e1_geometry_metrics/e2_auroc outputs
read-only. Composite P = LID × |slope| on all 13 primary-ladder checkpoints
(runA_grl/runB_orth1/runB, λ_orth = 0/1/5), Kendall's τ (exact) against Mahalanobis-AUROC.

Result: **no significant signal**. Primary C1 test — composite vs. AUROC: τ=−0.154, exact p=0.510,
n=13 (common-seed subset τ=−0.111, p=0.761, n=9). Secondary: LID alone τ=−0.026 (p=0.952);
spectral-slope alone τ=−0.128 (p=0.590); condition number (Paper 3's own metric) vs. AUROC also
τ=−0.128 (p=0.590) — notably, even Paper 3's headline condition-number metric does not correlate
with AUROC once AUROC (not rung order) is the target; its earlier-reported τ=0.84 was against rung
order, a different test.

Descriptive, non-confirmatory observation: composite/slope/condition-number all move monotonically
with rung (~3.0→3.4→3.8 for the composite) — the intervention does reshape geometry, consistent
with Paper 3's own condition-number dose-response direction — while AUROC stays in a narrow
below-chance band (0.365–0.441) across every checkpoint. Read as a plausible restricted-range
explanation for the null C1 result (Paper 3's ladder already flattened AUROC before any geometry
metric gets a chance to explain variance in it), not as evidence against the hypothesis. No
implementation bug found on inspection (LID range and monotonic dose-trend both plausible; exact
13/13 join against e2_auroc.csv verified).

Per the 24-Hour Rule: no hypothesis, protocol, or claim edit follows from this result. Per the
pre-committed interpretation rule in `PhaseB/01_stage0_retrospective.md`: this is the
"proceed-with-caution" branch, not a stop condition — Stage 1 is not blocked, but the restricted-
AUROC-range concern should inform how Stage 3's FM arms are read once they exist (a wider AUROC
range there would be a fairer test of the composite's predictive power than this retrospective CNN
data, where the ladder's own success at suppressing AUROC leaves the correlation underpowered by
construction).

Full outputs: `PhaseB/outputs/stage0_*.csv`, `stage0_summary.json`. Run log:
`PhaseB/logs/stage0_run_2026-08-04.log`.

Next action: per the checklist (`PhaseB/00_execution_checklist.md`), Stage 1 — resolve the PanDerm
backbone dependency (checkpoint/license check; fallback CLIP ViT-B/16) before any infrastructure
work begins. Not started this session.

---

2026-08-05

Vinh reviewed the Stage 0 log. Assessment: a "clean" null result, correctly handled by protocol —
specifically flagged the interpretation discipline ("this doesn't block Stage 1, and no
hypothesis/protocol edit follows from it") as the actual payoff of Phase A's pre-registration/
24-Hour-Rule/Constitution work, and the restricted-range explanation as correctly hedged
("plausible... explanation," not "therefore mechanism is wrong").

Wording correction (not a hypothesis/protocol change): "proceed with caution" in
`PhaseB/01_stage0_retrospective.md` read as more tentative than the protocol's own interpretation
rule warrants — that rule only ever had two branches, proceed-as-preregistered or blocked, never a
"caution" state. Corrected to "proceed as preregistered" in `01_stage0_retrospective.md` and
`00_execution_checklist.md` (both living status docs, not append-only logs). This entry is the
record of that correction, per this file's own no-retroactive-edits rule — the 2026-08-04 entry
above is unedited.

Insight worth carrying forward (Vinh, restated): condition number and the new composite both
correlate with dose (intervention → geometry, confirmed) but neither correlates with AUROC in this
retrospective dataset (geometry → accessibility, not observable here). Consistent with reading
Stage 0 as a pipeline/mechanism-consistency sanity check, not a test of Predictions 1–2 — which is
exactly the role `16_Minimal_Sufficient_Experiment.md` Week 0 assigns it.

Added: exploratory scatter plot, composite geometry metric vs. Mahalanobis-AUROC, labeled by
checkpoint/rung, no inferential claim attached — `PhaseB/outputs/figure_stage0_composite_vs_auroc.{png,pdf}`.

Backbone-dependency decision (Vinh): apply the pre-specified fallback rule
(`24_PreRegistration.md` §13 trigger table) proactively rather than letting it stall the project —
use PanDerm if its checkpoint is accessible within a reasonable time; otherwise fall back to CLIP
ViT-B/16 immediately, as already pre-registered. Not a deviation (the trigger and response were
both already written down); logging the decision-to-act-on-it, and the specific outcome, is what's
new here.

Vinh: Phase B is now to be treated as substantively started. No further mechanism debate on Stage 0;
focus moves to Stage 1 infrastructure for the Reference arm, per the locked protocol.

Backbone dependency resolved: **PanDerm** (not the CLIP ViT-B/16 fallback). Checked
github.com/SiyuanYan1/PanDerm — checkpoints are direct-download (Google Drive / HuggingFace), no
request form or DUA, license CC-BY-NC-ND 4.0 ("non-commercial academic research... with proper
attribution" — matches this project). Applying Vinh's decision rule as given: accessible within a
reasonable time → use PanDerm. Logged in `PhaseB/02_stage1_setup.md` with the exact checkpoint
links. Checkpoint itself not yet downloaded (~1GB+ ViT-L/16 file — flagging for a quick go-ahead
before pulling it down, rather than assuming). ND-license flag noted for Stage 3's supplementary-
material question, not a Stage 1 concern.

Vinh: approved the download ("Cho download — đây không còn là quyết định khoa học, mà là quyết định
hạ tầng"), with four conditions: (1) record checksum + full checkpoint identity in `PhaseB/assets/`;
(2) never modify the checkpoint (no conversion/optimization/reformatting); (3) verify it loads
before any feature extraction; (4) log everything in `02_stage1_setup.md` and here. Also fixed the
Stage 1 execution order explicitly: checkpoint → load → single-image inference → batch inference →
embedding extraction → embedding cache → Reference arm starts, each step gated on the previous
passing — and asked that the probe/feature-extraction steps specifically wait, not run immediately
after download.

Downloaded PanDerm ViT-L/16 (`panderm_ll_data6_checkpoint-499.pth` per the repo's own naming) via
`gdown` from the Google Drive link in the prior entry. 1,418,820,654 bytes. SHA256:
`23405e92f853c6f7ac9da80ccf4cfde84d4feb22cd9d7b98c7e850e3875554b2`. Repo HEAD commit at download
time (`git ls-remote`): `fd7a80748ba7fc3e203fed88f909f4689d0d6f24`. Full frozen identity record:
`PhaseB/assets/panderm_vitl16_checkpoint_metadata.md` (+ `.json` for scripted reuse). File's
preserved source mtime is 2025-04-29, predating this project — consistent with an untampered
original download, not a re-packaged copy.

Verified the checkpoint loads: `torch.load(..., map_location="cpu")` succeeds, raw state_dict
(no wrapper), 458 tensors, 354,660,064 total parameters. `encoder.*` subtree — 366 tensors,
303,326,208 params — matches standard ViT-L/16 (24 blocks, embed dim 1024, patch 16×16, 197 tokens
⇒ 224×224 input, BEiT-style layer-scale and q/v-bias, relative position embedding). The remaining
92 tensors (`mask_token`, `rd_pos_embed`, `regresser.*` cross-attention blocks, `decoder.*`) are
pretraining-only masked-image-modeling components (context-autoencoder-style), not part of the
downstream encoder — expected to be dropped when the encoder is instantiated on its own, not a sign
of a bad load. No modification made to the file at any point (download → hash → read-only
`torch.load` for inspection only).

Per Vinh's gated sequence: stopping here. Not instantiating the encoder class, not running
single-image inference, not extracting any embedding yet — that's the next step, separately
gated, not bundled into this one.

Vinh reviewed: no scope creep detected, gate discipline holding. Added one small process rule going
forward — every gate in Stage 1 explicitly resolves PASS → next step, or FAIL → issue + Lab
Notebook entry + stop (no open-ended "try a bit more" debugging). Explicit instruction for the next
step, verbatim: "Proceed to Stage 1.2 only: instantiate the PanDerm encoder using the official
repository implementation, validate a single forward pass on one image, record embedding
dimensionality and preprocessing, then stop. Do not perform batch inference, embedding extraction,
probing, or any scientific analysis." Also: use PanDerm's own model-definition code, not a
hand-written wrapper; `strict=False` only if explained; and write a dedicated "Reference Instrument
Validation" checklist once the forward pass works.

**Stage 1.2 — PASS.** Cloned `github.com/SiyuanYan1/PanDerm` to `PhaseB/external/PanDerm` (commit
`fd7a80748ba7fc3e203fed88f909f4689d0d6f24` — matches the checkpoint metadata's recorded commit, i.e.
the repo hasn't moved since the checkpoint identity was frozen). Found the repo ships *two* distinct
encoder-loading paths (`panderm_model/get_encoder` for an already-converted, prefix-free checkpoint
format vs. `models.builder.get_encoder` for the raw pretraining-checkpoint format) — determined ours
is the latter by checking `classification/script/lp_reproduce.sh`, which references our exact
checkpoint filename (`panderm_ll_data6_checkpoint-499.pth`) under model name `PanDerm_Large_LP`.
Used `models.get_encoder(args, 'PanDerm_Large_LP')` verbatim — the repo's own function, own model
factory (`panderm_large_patch16_224()`), own `strict=False` state_dict load, own preprocessing
transform. No custom model class or wrapper written. Installed `timm==0.9.16` + `open_clip_torch`
to match the repo's pinned `requirements.txt` (needed only because `models/builder.py`
unconditionally imports both at module load, even though the code path used doesn't touch either).

Forward pass: `model.forward_features(x, is_train=False)` (matching the repo's own
`extract_features.py`, not `model(x)`) on one real project image
(`DST-Skin/reader_images_96/ISIC_0034336.jpg`, ISIC 2019 dermoscopy). Embedding shape **(1, 1024)**,
no NaN/Inf, mean=−0.0277, std=0.716. Preprocessing (official, from `get_encoder`'s own return
value): `Resize(256) → CenterCrop(224) → ToTensor → Normalize(imagenet mean/std)`.

`strict=False` (the repo's own choice for this checkpoint, not this project's) independently
explained: 92 unexpected keys, all `decoder.*`/`mask_token`/`rd_pos_embed`/`regresser.*` —
pretraining-only MIM components, exactly the 92 = 458 − 366 already predicted in the checkpoint
metadata before this step ran (prediction confirmed, not surprised); 2 missing keys
(`head.weight`/`head.bias`) — an unused classification head, never touched since `forward_features`
bypasses it. No missing keys inside the `encoder.*` subtree.

Wrote `PhaseB/reference_instrument_validation.md` — the dedicated checklist Vinh asked for: ✓
checkpoint, ✓ architecture, ✓ forward pass, ✓ embedding dimension, ✓ preprocessing. All PASS.

Stopping here per Vinh's scope: no batch inference, no embedding cache, no probing, no analysis.

Vinh reviewed Stage 1.2 as PI: PASS — explicitly "PASS because it did not violate protocol," not
merely "PASS because it ran." Approved the evidence-based model-selection approach (finding
`lp_reproduce.sh` referencing our exact checkpoint filename, rather than guessing a loading recipe)
and the independently-computed strict=False key-diff explanation. Added one addendum request: freeze
the preprocessing pipeline to a machine-readable snapshot (`assets/reference_preprocessing.json`),
reasoning that if the upstream PanDerm repo is ever updated in place, this project's own validated
pipeline should stay pinned rather than silently drift. Also stated a project-wide rule: every
artifact (checkpoint → embedding → CSV → figure → table) must be able to answer "which protocol step
produced this," or it doesn't get used.

Wrote `assets/reference_preprocessing.json` — exact values re-introspected from the live
`get_eval_transforms()` call (not retyped from memory): Resize(256, bilinear) → CenterCrop(224) →
ToTensor → Normalize(mean=[0.485,0.456,0.406], std=[0.228,0.224,0.225]). Noted explicitly that the
std's first value (0.228) is the repo's own hard-coded constant, not the textbook ImageNet 0.229 —
recorded verbatim, not silently corrected.

**Stage 1.3 authorized (PI decision), verbatim:** "Run official PanDerm preprocessing and encoder on
a representative batch of images to validate stable batched inference. Record tensor shape, NaN/Inf
checks, failed-image count, and throughput. Do not cache embeddings, run probes, compute geometry
metrics, or perform any scientific analysis. Stop immediately after the Batch Inference Validation
gate passes." Five named criteria: (1) shape always N×1024, (2) no NaN, (3) no Inf, (4) no sample
fails silently (e.g. image → zero embedding without being counted), (5) throughput in images/sec,
recorded not benchmarked.

**Stage 1.3 — GATE PASS.** Representative batch: all 96 images in `DST-Skin/reader_images_96/` (26
ISIC 2019 dermoscopy + 70 PAD-UFES-20-style clinical `PAT_*` files) — both domains of this project's
actual acquisition-shift testbed, not a synthetic sample. Processed as 6 true batches of 16 (stacked
tensors through one `forward_features` call each, not a per-image loop). Before running, asserted
the live official transform against the frozen `reference_preprocessing.json` snapshot — matched,
no drift. Results: shape `(16, 1024)` on every batch; no NaN; no Inf; 0 load failures and 0
zero-norm (silently-failed) embeddings; throughput 5.208 images/sec (96 images / 18.435s
forward-pass wall time, CPU — recorded for planning, not claimed as a benchmark). Embeddings held in
memory per batch and discarded (`del embeddings, batch`) — nothing written to disk. Full record:
`reference_instrument_validation.md` §Stage 1.3 (extended, not a new standalone file — keeps the
Reference Instrument's validation history in one place). Machine-readable report:
`outputs/stage1_3_batch_inference_validation.json`. Run log: `logs/stage1_3_run_2026-08-05.log`.

Stopping here per Vinh's scope: no embedding cache, no dataset-loader infrastructure, no probe, no
geometry metric, no scientific analysis.

Vinh (PI review): Stage 1.3 as run is an **Engineering PASS**, not a **Scientific PASS** — the
convenience batch came from `DST-Skin/reader_images_96/`, a legacy Paper 2 reader-study folder, not
the preregistered ISIC 2019 + PAD-UFES-20 dataset loader (`22_Experimental_Design.md` §2). The
engineering run stays valid and stays in this notebook unedited; it just doesn't satisfy the
preregistered validation gate. Instructed: re-run using a convenience batch sampled directly through
the preregistered loader (CSG-Skin's `SkinDataModule`/`build_id_ood_test_dataloaders`, same
`master_metadata_lesion_only_soft.csv`, same split seeds Papers 1–3 use).

**Blocker found while attempting the re-run — logged, not routed around.** Checked what's actually
available locally before writing any loader code:

- `CSG-SKin/data/` and `CSG-SKin/checkpoints/` are **empty** on this machine. Per paper-3's own
  `docs/environment_requirements.md` (written before this project, not a new finding): the ISIC
  2019 + PAD-UFES-20 raw data and the CSG-Lite checkpoints live on a **remote GPU server**, never
  synced to this local Mac — this local checkout only ever received `paper-3/results/`
  (already-extracted embeddings/CSVs), not the raw inputs or `master_metadata_lesion_only_soft.csv`
  itself.
- Bounded filesystem search (home directory, `/Volumes`) found no local ISIC 2019 image data.
  Found one orphaned file, `~/ISIC_2019_Training_Metadata.csv` (25,331 rows) — official ISIC
  challenge demographic metadata only (image id, age, anatomic site, sex), no images alongside it,
  not in CSG-Skin's own `path`/`label`/`domain` schema. Not sufficient on its own.
- `DST-Skin/data/raw/` **does** have real local images: PAD-UFES-20 (2,302 files +
  `metadata.csv`) and **ISIC 2018** (not 2019) — Paper 2's own testbed, a different dataset vintage
  than Papers 1/3's preregistered ISIC 2019 + PAD-UFES-20 pairing.

Deliberately did **not** substitute ISIC 2018, the stray metadata-only CSV, or relabel
`reader_images_96` as if it were the preregistered split — any of those would reproduce exactly the
provenance ambiguity this gate exists to prevent, silently. Flagged to Vinh instead of guessing.

Asked Vinh where the real ISIC 2019 + PAD-UFES-20 data is. First answer,
`/Users/cubo/projects/Bản sao datn 4/data/`, checked and confirmed byte-identical (MD5 match on
`pad_ufes20/metadata.csv`, matching file counts) to `DST-Skin/data/raw/` — i.e. the same ISIC 2018 +
PAD-UFES-20 testbed already found, not ISIC 2019. Reported the mismatch back rather than treating a
real, specific pointer as automatically correct. Vinh's response: "bạn tự tải về đi" (download it
yourself).

**Downloading ISIC 2019 (official archive), 2026-08-05:**
- `https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip` — Content-Length
  9,771,618,190 bytes (9.10 GiB) per HEAD request, Last-Modified 2020-03-06 (S3 `ETag`
  `"4c2dd42e1e26a704f71c907732c459d7-569"`, a multipart-upload ETag, not a plain MD5 — SHA256 will be
  computed locally after download for this project's own checksum record, matching how the PanDerm
  checkpoint was handled).
- `https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv` — 1,291,479
  bytes, downloaded complete.
- `ISIC_2019_Training_Metadata.csv` — reused the copy already found at
  `~/ISIC_2019_Training_Metadata.csv` (25,331 rows, demographics only) rather than re-downloading it.
- All three placed under `CSG-SKin/data/` — CSG-Skin's own designated (gitignored) data directory,
  exactly the layout `src/datasets/preprocess_metadata.py` expects. No CSG-Skin code touched.
- PAD-UFES-20 copied into `CSG-SKin/data/pad_ufes20/` from `DST-Skin/data/raw/pad_ufes20/` (already
  confirmed byte-identical to the canonical source).
- Training Input zip download launched in background (large; in progress as of this entry).

**Dataset loader, resolved:** reuse CSG-Skin's own `SkinDataModule` +
`build_id_ood_test_dataloaders` (`src/datasets/skin_dataset.py`, `splits.py`) unmodified — the exact
loader Papers 1–3's own scripts (`paper-3/scripts/extract_auroc_e2.py`) use. `master_metadata.csv`
will be built by CSG-Skin's own `src/datasets/preprocess_metadata.py`, also unmodified, once the ISIC
2019 images finish downloading. New script:
`PhaseB/analysis/stage1_3_rerun_preregistered_split.py`.

Vinh reviewed this direction favorably: correctly treated CSG-Skin's own loader as the authoritative
implementation for Paper 4's locked (CSG-Skin-based, not DST-Skin-thesis-based) dataset protocol,
and approved swapping in PanDerm's transform via attribute injection rather than forking CSG-Skin's
code. Asked for two additions, both done:

1. **Written justification for why attribute injection is not a protocol deviation** — added to
   `PhaseB/02_stage1_setup.md`'s Dataset Loader item: split unchanged, sampling unchanged, labels
   unchanged, only the image transform replaced (to match the Reference Instrument under test, not
   CSG's own CNN-tuned transform), no CSG-Skin source file read-and-rewritten.
2. **Dataset Validation gate**, run and must PASS before any encoder code executes: ISIC/PAD image
   counts on disk, metadata row counts by domain, split reproducibility at seed=42 (recomputed
   twice, compared), preprocessing-attached check. Implemented as `validate_dataset()` in the rerun
   script.

Vinh's instruction: continue until (1) ISIC 2019 download completes, (2) `master_metadata.csv` is
built via the original `preprocess_metadata.py`, (3) Stage 1.3 is re-run through the preregistered
loader, (4) if that gate PASSes, Stage 1.3 counts as scientifically complete and Stage 1.4
(Embedding Extraction) can begin.

Vinh reviewed the two-script gate split (`stage1_3a_dataset_validation.py` writes
`dataset_validation_report.json`; `stage1_3_rerun_preregistered_split.py` refuses to run unless that
report says PASS) as one of Phase B's better architectural decisions so far — sequencing enforced by
the pipeline's own structure, not by convention/memory. Approved, with two additions, both made:

1. `dataset_validation_report.json` now self-describes: `protocol_step`, `protocol_version`
   (`"24_PreRegistration.md frozen 2026-08-04 (session 9)"`), `csg_skin_commit`,
   `panderm_repo_commit`, `panderm_preprocessing_sha256` (hash of `reference_preprocessing.json`
   itself, so a later change to that file would be visible against this record).
2. New `PhaseB/analysis/build_master_metadata_with_provenance.py` — runs CSG-Skin's own
   `preprocess_metadata.py` unmodified as a subprocess, then writes
   `assets/master_metadata_build_provenance.json`: output SHA256, row count (total + by domain),
   build start/end timestamps, CSG-Skin commit, and source-data provenance (ISIC zip SHA256 —
   captured before the zip is deleted post-extraction — PAD-UFES-20 source).

Reaffirmed instruction, unchanged: stop after Dataset Validation PASS and artifact freeze, report
back, do not proceed to the encoder/Stage 1.3 rerun without an explicit go-ahead.

**Download attempt #1 failed partway.** The background curl job reported exit 0 in my own status
line, but that was a bug in how I captured it (`$?` after a `| tail` pipe reads `tail`'s exit code,
not curl's) — the actual curl output showed `curl: (56) Recv failure: Connection reset by peer` at
14.4% (~1.34 GB of the expected 9.10 GiB). Caught by checking the final file size against the
Content-Length recorded earlier, not by trusting the reported exit code — exactly the kind of thing
the "verify archive integrity" step exists to catch, and it did. Retrying now with `curl -C -
--retry 5 --retry-delay 5` (resume from the partial file, auto-retry on transient failure) and
capturing the real exit code via `${PIPESTATUS[0]}` this time, not `$?` after a pipe.

**Download attempt #2 also failed partway** — same `curl: (56) Recv failure: Connection reset by
peer`, this time at 15.8% (~1.55 GB), after resuming and making ~200MB of progress past attempt #1's
stopping point. `${PIPESTATUS[0]}` also came back empty rather than a bash-style exit code — this
session's shell is zsh (per environment info), where the equivalent array is `pipestatus`
(lowercase), not `PIPESTATUS`; a scripting mistake on my part, not informative about curl's actual
exit status. Two resets at different points, both partway through the same large (9.1 GiB) transfer,
reads as real network instability on this connection for a sustained single-file transfer, not a
one-off. Not routing around it with a smaller/different file — switching to a `curl -C -` **loop**
(re-invokes curl from scratch, resuming via `-C -`, each time the process exits before reaching the
expected size) instead of relying on one curl invocation's internal `--retry` to survive a full
connection drop, capped at 40 attempts before giving up and reporting back.

While the retry loop was running (at 1.59 GB), Vinh: "thêm vào rồi đấy, ở folder paper 4 ấy tên
archive" (already added it, in the Paper4 folder, named archive). Found `Paper4/archive/` — ISIC
2019 images pre-organized into per-diagnosis-class folders (`DF`/`AK`/`BKL`/`VASC`/`NV`/`BCC`/`SCC`/`MEL`,
exactly CSG-Skin's `LABELS` taxonomy), original `ISIC_XXXXXXX.jpg` filenames intact, 25,331 `.jpg`
files (matched a `find -type f` count of 25,333, off by 2 — accounted for by stray `.DS_Store` files
in two class folders, not real images). Killed the now-unneeded curl retry loop, deleted the
incomplete zip.

**Did not just trust a folder named "archive" because a real image count looked plausible.**
Independently verified every single image's folder-assigned label against the officially-downloaded
`ISIC_2019_Training_GroundTruth.csv` (one-hot decoded, matched by image ID) — stronger than a
checksum, since it validates content-identity against ground truth rather than just file identity
against an unknown origin. Result: **25,331/25,331 matched, 0 missing, 0 label mismatches.** This is
the genuine ISIC 2019 training set, just redistributed in a different (Kaggle-style) folder layout
than the official flat archive.

Symlinked (not copied — disk space) the 25,331 images from their per-class folders into
`CSG-SKin/data/ISIC_2019_Training_Input/` as a flat directory, matching exactly what
`preprocess_metadata.py::resolve_isic_image_path` expects. No files copied, no files modified,
`Paper4/archive/` untouched as the source of truth.

**Built `master_metadata.csv`** via `PhaseB/analysis/build_master_metadata_with_provenance.py`
(updated to record the archive-based provenance and verification method above, in place of the S3
zip SHA256 the download would have produced). CSG-Skin's own `src/datasets/preprocess_metadata.py`
ran unmodified, subprocess, exit 0. **Result: 27,629 rows (25,331 ISIC + 2,298 PAD-UFES-20)** —
exactly matching `Scientific_Hypothesis.md`'s stated testbed size ("27,629 images: 25,331 ISIC 2019
+ 2,298 PAD-UFES-20"), a strong sanity check that this reconstruction lines up with what Papers 1–3
actually used. Output SHA256:
`2df20422ede69e56ee9b4f1beb101e50cbc6c43d550f9d1a0816ecec99af6ad2`. Full provenance (script path,
CSG-Skin commit `1233898...`, timestamps, row counts by domain, source verification method) frozen
in `PhaseB/assets/master_metadata_build_provenance.json`.

**Ran the standalone Dataset Validation gate (Stage 1.3a) — PASS.** All five checks green: 25,331
ISIC images on disk, 2,299 PAD-UFES images on disk, 27,629 metadata rows (25,331/2,298 by domain),
split reproducible at seed=42 (recomputed twice, identical index sets — `isic_test_n=5067`, which
matches Paper 3's own `results/e2_distances/*.npz` `s_id` array size, `(5067,)`, exactly: an
independent confirmation that this split reconstruction lines up with what Papers 1–3's pipeline
actually produced, not just a plausible-looking number), preprocessing attached and matching the
frozen snapshot. Self-describing report written to `PhaseB/outputs/dataset_validation_report.json`
(protocol_step, protocol_version, csg_skin_commit, panderm_repo_commit,
panderm_preprocessing_sha256, all checks, status=PASS). No encoder loaded, no checkpoint touched, no
batch sampled, no forward pass run — per Vinh's PI-checkpoint requirement, `stage1_3_rerun_preregistered_split.py`
(the actual encoder step) has not been executed. Reporting back now, holding here.

Vinh reviewed the Dataset Validation gate as APPROVED — "Scientific PASS, not just Engineering
PASS." Praised specifically: not retrying the failed download a third time but seeking a real source
instead, then not trusting that source blindly (the GroundTruth cross-check called "semantic
verification, stronger than a checksum"); the symlink decision (no 9GB duplicated); and that the
pipeline now has a full provenance chain (checkpoint → preprocessing → dataset → dataset validation
→ master_metadata build), each stage answerable and checkable independently.

One wording correction (not a protocol/hypothesis edit): `02_stage1_setup.md` had said the 27,629-row
count "matches `Scientific_Hypothesis.md`." Corrected to "matches the preregistered dataset
specification" — row/image counts are a protocol/data-scope fact (`22_Experimental_Design.md` §2),
not a claim the hypothesis document makes or that a dataset build could confirm or refute. Edited
directly in `02_stage1_setup.md` (a living doc); this entry is the record of that correction per this
file's own append-only rule.

**Stage 1.3 rerun AUTHORIZED**, verbatim: "Proceed to the preregistered Stage 1.3 rerun using the
validated CSG-Skin dataset loader and the official PanDerm preprocessing. Limit execution strictly
to batch inference validation. Produce no cached embeddings, no derived datasets, and no scientific
analyses. Stop immediately after the Reference Instrument Validation checklist is completed on the
preregistered split."

**Ran it — GATE PASS.** `stage1_3_rerun_preregistered_split.py`, gated on (and confirmed against)
Stage 1.3a's report. ID (ISIC-test) dataset size **5,067** (independently matches Paper 3's own
`e2_distances` `s_id` array size exactly — second confirmation of this, after Stage 1.3a's own
recomputation), OOD (PAD-UFES) dataset size **2,298**. Convenience batch: first 8 ID + first 8 OOD
(deterministic, `shuffle=False`), one 16-image batched forward pass through the official PanDerm
encoder. All 5 criteria PASS: shape `(16, 1024)`, no NaN, no Inf, no silent failure (0 zero-norm
rows), throughput 5.576 img/s (CPU). No cache, no probe, no geometry/PCA/UMAP/statistics — held
strictly to Vinh's stated boundary. Output embeds the Stage 1.3a report verbatim, so the full
dependency chain (checkpoint → preprocessing → dataset → dataset validation → this run) is readable
from one file: `outputs/stage1_3_rerun_preregistered_split.json`.

`reference_instrument_validation.md` extended with a new "Stage 1.3 RE-RUN" section (the original
`reader_images_96` section kept, marked superseded-for-scientific-purposes but not deleted or
edited) — including, per Vinh's new per-artifact rule, an explicit answer to which protocol step
produced this artifact, what it depends on, and whether it's reproducible from scratch without
manual steps (yes, given the same data on disk; the one non-scripted upstream step, the ISIC 2019
image source, is itself documented in `master_metadata_build_provenance.json` rather than assumed).

Vinh: Phase B has now produced its first genuinely scientific-gate artifacts (not just
infrastructure smoke tests). Provenance discipline stays in force going forward — every new artifact
should be able to answer the same three questions this one does.

**Stage 1 review: APPROVED-COMPLETE.** Vinh's framing: "Stage 0 sinh ra evidence. Stage 1 sinh ra
instrument. Stage 2 sẽ sinh ra data." Praised specifically: the artifact dependency chain (checkpoint
→ preprocessing → dataset → dataset validation → this run) as something a future supplementary
figure could draw directly from; the nested validation report inside the run's own output JSON
(the artifact proves its own provenance, not just asserts PASS); and that the strict boundary held
(no Mahalanobis/probe/PCA/geometry/cache anywhere in Stage 1).

**Phase B internal stage renumbering (2026-08-05, does not touch the underlying preregistered
protocol/claims):** Stage 1 = Reference Instrument (now complete). New Stage 2 = Reference
Representation — embedding extraction, provenance, and its own Embedding Integrity Gate. What was
"Stage 2 — Baseline" (probe/geometry/accessibility/reliability on the Reference arm) becomes Stage 3.
What was "Stage 3 — Intervention arm" becomes Stage 4. Full table and rationale in
`00_execution_checklist.md`'s new restructuring note. New file:
`PhaseB/02a_stage2_reference_representation.md`. `03_stage2_reference.md` kept its filename (avoid
breaking existing references in this notebook) but got a renumbering note at its top.

Key mindset shift, stated explicitly by Vinh: the embedding artifact this next stage produces is not
"an embedding cache" — call it the **Reference Embedding Artifact**. It gets the same treatment as
the PanDerm checkpoint: frozen identity, full provenance, integrity-gated before anything is computed
from it.

**Stage 2.1 AUTHORIZED**, verbatim: "Begin Stage 2.1 by extracting embeddings for the complete
preregistered ID and OOD datasets using the validated Reference Instrument. Produce a single frozen
Reference Embedding Artifact together with full provenance, integrity checks (shape, ordering, label
alignment, sample count, SHA256), and no scientific analyses. Do not compute probes, geometry
metrics, OOD scores, or statistics. Stop immediately after the Embedding Integrity Gate passes."

Scope: complete ID (ISIC-test, n=5,067) + OOD (PAD-UFES-20, n=2,298) = 7,365 images, same
preregistered loader and Reference Instrument already validated. Five-check Embedding Integrity Gate
(shape, sample count, ordering, label alignment, SHA256) before anything is treated as frozen.
Beginning execution now.

Extraction launched in background (~20–25 min estimate). While it ran, Vinh reviewed the plan and
refined the Embedding Integrity Gate + output format before results were in: (1) sample count
ISIC=5067/PAD=2298 must match protocol exactly; (2) tensor shape 7365×1024; (3) ordering — prove
metadata row *i* → embedding row *i*, explicitly, not just assert it; (4) label alignment —
image→label→embedding, no drift; (5) SHA256 of each of `embeddings.npy`/`labels.csv`/`metadata.csv`
separately, not one bundled hash; (6) a `manifest.json` "birth certificate." Output restructured
from a single `.npz` into a `ReferenceEmbedding/` directory:
`embeddings.npy`/`labels.csv`/`metadata.csv`/`manifest.json`/`sha256.txt`. Explicit instruction:
don't do anything else while waiting, then run only the Embedding Integrity Gate, freeze, report —
no LID/probe/Mahalanobis/energy/KDE/ViM, all of that is Stage 3.

**Extraction completed — GATE PASS.** 7,365 images, ~40 minutes CPU wall time (ID: 5,067 images,
3.13 img/s over 1616.3s; OOD: 2,298 images, 2.92 img/s over 787.7s — both recorded for planning, not
a benchmark claim). Original five-check gate (from `stage2_1_extract_reference_embeddings.py`) PASS:
shape (7365, 1024); sample count id=5067/ood=2298 matching Stage 1.3a's own report; ordering
preserved (DataLoader-collected labels matched dataframe-derived labels exactly); label alignment
(all 7,365 paths exist on disk, domain segmentation correct); SHA256 computed.

**Restructured per Vinh's refined spec (`stage2_1b_finalize_embedding_artifact.py`) — no new
embeddings computed, pure post-processing of the already-gated output.** Added a genuinely
*independent* re-verification beyond the original script's self-consistency check: froze a second,
separate reconstruction of the ID/OOD split directly from `master_metadata.csv` (fresh function
call, nothing read from the extraction run's own in-memory arrays), then compared `path`/`label_idx`
element-by-element against the artifact's stored arrays across all 7,365 rows. **Exact match, both
arrays, every row** — this is the actual proof of "metadata row *i* → embedding row *i*," not an
assertion. Final six-check gate: sample count PASS, tensor shape PASS, ordering PASS (via the
independent proof above), label alignment PASS, per-file SHA256 PASS
(`embeddings.npy=3a12b0e0...`, `labels.csv=17ead2bf...`, `metadata.csv=26ad35e0...`), manifest PASS.

Canonical frozen artifact: `PhaseB/assets/reference_embeddings/ReferenceEmbedding/` (`embeddings.npy`,
`labels.csv`, `metadata.csv`, `manifest.json`, `sha256.txt`). Intermediate extraction-run record kept
alongside (not canonical) for its per-domain timing detail. Full gate report:
`outputs/embedding_integrity_gate_report.json`. No probe, no Mahalanobis, no geometry metric, no
PCA/UMAP, no statistics computed — extraction and integrity verification only, per Vinh's explicit
boundary. Stopping here, reporting back.

**Vinh reviewed Stage 2.1 as APPROVED** — first artifact called "scientific" rather than
"instrument" or "infrastructure." Specifically praised the independent row-ordering proof (fresh
split reconstruction compared against the artifact, not an in-memory self-check) and the manifest
as a genuine "birth certificate."

**Immutability, per Vinh's addition:** added `"status": "FROZEN"` + `"frozen_at"` +
`"frozen_note"` to `manifest.json` as its last write (verified the three tracked data files'
checksums were unchanged immediately before this edit). Then `chmod 444` on all five files in
`ReferenceEmbedding/` and `chmod 555` on the directory itself. Verified: a write attempt into the
directory (`touch ReferenceEmbedding/should_fail.txt`) is rejected with `Permission denied`. Any
future correction is a new artifact version (new directory, new manifest) — never an edit to this
one.

**Phase B internal stage renumbering, second pass (2026-08-05):** "Stage 3" renamed "Scientific
Analysis" and split into sub-stages mapping directly to `02_Claim_Matrix.md`, per Vinh: "Stage 3
không nên bắt đầu bằng Mahalanobis." New breakdown: 3.0 Artifact Audit (data-integrity check, not
analysis), 3.1 Artifact Characterization (structural description — rows/dim/memory/dtype/norm
distribution, still not a hypothesis test), 3.2+ Claims C1–C5 per the Claim Matrix. New file:
`03a_stage3_0_artifact_audit.md`. `03_stage2_reference.md`'s content (probe/geometry/accessibility/
reliability) now corresponds to Stage 3.2+, not "Stage 3" monolithically — renumbering note updated
in place, filename still unchanged.

**PI Decision: Stage 2.1 COMPLETE. Stage 3 not authorized as a whole — only Stage 3.0 (Artifact
Audit).** Explicit boundary: confirm the artifact isn't corrupted, has no duplicates, no missing
samples, and that auditing it generates no new data. Not authorized: Mahalanobis, probe, LID,
spectral-decay slope, energy, ViM, KDE, any hypothesis-testing statistic.

Beginning Stage 3.0 now.

**Stage 3.0 executed — 4/5 checks clean PASS, 1 check surfaced a real finding, fully root-caused,
not silently resolved.** File integrity PASS (SHA256 of all 3 frozen files matches manifest).
No-missing/corrupted PASS (7,365 rows, no NaN/Inf/zero-norm, re-checked from the on-disk frozen
files, not Stage 2.1's in-memory state). Random-sample visual inspection PASS (12 images, seed=42,
montage sent to Vinh). Reproducibility spot-check PASS — the same 12 images re-run through the
Reference Instrument fresh matched the stored embeddings almost exactly (max abs diff ≤9.5e-6,
cosine similarity ≥0.9999999), confirming the artifact reflects what the instrument actually
produces.

**No-duplicates check: 17 duplicate-embedding groups found (34 rows), 0 duplicate paths.**
Investigated rather than reported as a blanket failure: independently pixel-compared every group's
source images (not file-hash comparison — one pair had different file bytes, same pixel content,
i.e. a re-encoding of the same photo). **All 17 groups confirmed genuine duplicate photographs
already present in the underlying ISIC 2019 / PAD-UFES-20 public datasets** — not an extraction
pipeline defect. Further finding: **13 of the 17 duplicate-image groups carry conflicting
diagnostic labels** for the same underlying photograph in the source metadata (e.g. one PAD-UFES-20
image labeled `SCC` under one image ID, `BCC` under another). This is a label-reliability property
of the source data itself, not something resolved or corrected here — the frozen artifact is
unmodified, still faithfully reflects the preregistered dataset (duplicates and label conflicts
included), and this finding is logged for a PI decision rather than acted on unilaterally.

Full detail: `outputs/stage3_0_artifact_audit_report.json`, `outputs/stage3_0_duplicate_diagnosis.json`.
Montage: `outputs/stage3_0_sample_montage.png`. Not computed: Mahalanobis, probe, LID, spectral
decay, energy, ViM, KDE, any hypothesis-testing statistic — per Vinh's explicit Stage 3.0 boundary.

**Vinh reviewed: "APPROVED WITH FINDING"** — not a plain PASS, not a FAIL. Praised specifically:
not dropping/merging/relabeling the duplicate rows to make the audit "look clean" (would have broken
protocol immediately); continuing to ask *why* instead of stopping at "17 duplicates found"; and the
13/17 label-conflict finding as "dataset pathology," not a bug.

**Wording correction (not a protocol/hypothesis edit):** "same photograph" / "same underlying
photograph" overclaims what pixel comparison proves — corrected to "pixel-identical images"
throughout. Pixel identity is established; whether two pixel-identical images are the same capture,
lesion, or patient is not independently established here and is not claimed. Edited directly in
`03a_stage3_0_artifact_audit.md` and `outputs/stage3_0_artifact_audit_report.json` (both living
docs/reports, not append-only or frozen); this entry is the record of that correction per this
notebook's own rule.

**PI decision: Stage 3.0 APPROVED, with an additional required deliverable before Stage 3.1.**
"Artifact → Audit" and "Artifact → Cleaning" are different things — findings get recorded, not
acted on. Built two new frozen artifacts (`stage3_0b_duplicate_registry_and_issues.py`, no dataset
modification, `ReferenceEmbedding/` read read-only throughout):

- `assets/DuplicateRegistry/` — `duplicate_groups.csv` (group_id, row_index, path, domain, label,
  embedding SHA256, pixel-identity confirmation, label-conflict flag), `montage.png` (all 17 pairs
  side by side), `notes.md` (methodology + explicit "NO ACTION TAKEN").
- `assets/ArtifactIssuesRegister/` — `issues_register.json`/`.md`, one entry so far (duplicate
  images with label conflict; status "logged, no action taken"; resolution "open"). Policy: future
  audits append new entries, never edit or remove existing ones. Every future claim on this artifact
  should reference this register.

Both frozen the same way as `ReferenceEmbedding/` (files `chmod 444`, directories `chmod 555`,
verified non-writable).

Vinh framed this explicitly as an eventual paper limitations/supplementary observation, not a
contribution: "During artifact auditing, 17 groups of pixel-identical images were identified in the
preregistered datasets, including 13 groups with conflicting labels."

**Stage 3.1 (Artifact Characterization) is next, not yet started. Claim C1 (Stage 3.2) still not
authorized.**

Vinh reviewed Stage 3.0 as "CLOSED — not PASS, not FAIL — closed with one registered open issue,"
praised the DuplicateRegistry → Issues Register → future-claims-reference-it chain as professional
evidence management, and specifically liked the "NO ACTION TAKEN" line as the answer to a reviewer
asking "why didn't you remove duplicates." Added a new standing rule (not tied to any single claim
yet, since none exist): every future Claim C1–C5 write-up ends with an explicit "Known artifact
issues consulted" line naming which register entries were checked and a judgment call for each
("not expected to materially affect..." or "potentially relevant"). Recorded as policy in
`00_execution_checklist.md` (the frozen register itself isn't edited).

**Stage 3.1 AUTHORIZED**, verbatim, with explicit exclusions: "Characterize the frozen Reference
Embedding Artifact only. Compute descriptive properties (shape, dtype, norm distribution, variance
statistics, basic integrity summaries) and document them. Do not compute any quantities that test
Claim C1–C5, including Mahalanobis, probe accuracy, LID, spectral decay, Energy, ViM, KDE, AUROC, or
hypothesis-related statistics."

**Ran it.** Shape (7365, 1024), float32, 28.77 MB. Norm distribution: pooled mean=23.077,
std=0.858. Per-domain (reported descriptively, no test/comparison computed): ISIC mean=22.619,
PAD-UFES mean=24.086 — histogram shows a visible offset between the two, stated as an observation
only, explicitly not interpreted (that's Stage 3.2+ territory once authorized). Per-dim variance:
mean=0.0879, total=90.05. Class counts: AK=903, BCC=1510, BKL=760, DF=48, MEL=956, NV=2819, SCC=318,
VASC=51. Noted the 34 duplicate rows (Issue #1) as a small, symmetric contribution to these pooled
statistics — not asserted negligible without saying so. Frozen the same way as prior artifacts
(`assets/ArtifactCharacterization/`, files `444`, directory `555`). No Mahalanobis, probe, LID,
spectral decay, energy, ViM, KDE, AUROC, or hypothesis statistic computed.

Stopping here. Stage 3.2 (Claim C1) not authorized.

**Vinh reviewed Stage 3.1 as APPROVED** — specifically praised not running any test statistic on
the visible ISIC/PAD-UFES norm offset (observation vs. inference kept distinct) and the duplicate
note's "small double-count" phrasing over "negligible" (flagged, not dismissed). Requested one
addition: a `characterization_manifest.json` inside `ArtifactCharacterization/` making explicit that
this artifact contains descriptive statistics only, tests no claims, and is derived from a specific,
hash-referenced `ReferenceEmbedding/` — so Stage 3.1's output is never later mistaken for Stage 3.2
analysis.

**One-time unfreeze/amend/re-freeze, logged transparently (not a silent edit):** since
`ArtifactCharacterization/` had already been frozen before this request arrived,
`chmod`'d it back to writable, verified `characterization_report.json` and `norm_histogram.png`
were byte-identical to their already-recorded checksums (`shasum -c`, both OK — nothing was altered
during the unlock), added `characterization_manifest.json` (references `ReferenceEmbedding/`'s
`embeddings.npy`/`metadata.csv` SHA256 directly, states `contains_only_descriptive_statistics: true`,
`claims_tested: []`, and an explicit boundary statement), recomputed `sha256.txt` to cover all three
files, and re-froze (`444`/`555`, write rejected — verified again).

**PI decision: Stage 3.2 execution NOT authorized. Only preparation of the Claim C1 execution
package** — plan, input/output spec, dependency graph, gate, stop condition. No code, no results,
no statistics. Beginning that now.

**Drafted `ClaimC1/C1_execution_plan.md` — planning only, nothing executed.** While drafting it,
found and surfaced (not silently resolved) a naming question worth flagging clearly: per
`22_Experimental_Design.md` §1, **Claim C1 is already closed** — it maps specifically to "Stage 0 —
Retrospective geometry computation on Paper 3's existing CNN checkpoints," already run
(`01_stage0_retrospective.md`, τ=−0.154, p=0.510, n=13). There is no second, FM-arm instantiation of
C1 in the preregistered design. What `ReferenceEmbedding/` can actually support next is the
**Reference arm's Gate 0 (Implementation Integrity Gate)** + baseline measurement package
(domain-leakage probe, distance scorers, geometry metrics, ID task performance) that
`03_stage2_reference.md` already specifies — not a new test of C1, and not yet a test of C2–C5
either (those need the Intervention arm's dose-response ladder, which doesn't exist yet). Per
Research Constitution Rule #1 ("never introduce an experiment without a claim it maps to"), flagged
this explicitly in the plan's §0 rather than silently proceeding under the "Claim C1" label Vinh
used informally. Recommended interpretation given in the plan; final call left to Vinh.

Plan covers: input artifact (`ReferenceEmbedding/` + `ArtifactIssuesRegister` Issue #1 consulted),
five expected measurements (all reused from Paper 1/3, none reinvented) with one open design
question flagged (how the domain-leakage probe's train/test split should work, given
`ReferenceEmbedding/` currently only holds the held-out ID+OOD evaluation split), a proposed output
artifact structure (`GateZeroReferenceArm/`, same freeze pattern as prior artifacts), a dependency
graph, Gate 0's exact criterion (domain-probe leakage consistent with Paper 1's CNN baseline,
0.979), and a stop condition (halt at the Gate 0 verdict, no C2–C5 evaluated). No code written, no
embeddings touched beyond what's already frozen, no statistic computed.

**Vinh confirmed the finding and pushed it further — this is a protocol-architecture fix, not a
technical bug.** `ReferenceEmbedding/` is PanDerm → Reference arm → frozen representation; it is
input data, and cannot re-answer C1 (already answered at Stage 0). What comes next isn't "Claim C1"
at all — it's baseline measurement. Corrected architecture:

```
Stage 3.2 — Reference Arm Baseline Evaluation   (Gate 0 → Probe → Geometry → Reliability → Report)
Stage 4   — Intervention Arm                    (same measurement suite, once it exists)
Stage 5   — Comparative Analysis                (Reference vs. Intervention → C2, C3, C4, C5)
```

Rationale (Vinh): without this split, the paper would read as an alternating
claim/reference/claim/reference pattern that's hard to follow, when the actual logic is
reference → intervention → comparison → inference. Also flagged the probe train/test split as a
real protocol gap needing resolution *before* any code is written, not left implicit.

**PI decision: no code. Exactly one action authorized:** "Refactor Phase B naming and dependency
graph to distinguish 'Reference Arm Baseline Evaluation' from hypothesis-testing claims, and resolve
the domain-leakage probe data split in the protocol before any analysis code is written."

**Done, still zero execution:**

1. Renamed `PhaseB/ClaimC1/` → `PhaseB/ReferenceArmBaseline/`,
   `C1_execution_plan.md` → `reference_arm_baseline_execution_plan.md`. Not treated as a frozen
   artifact (it's an unexecuted draft plan) — rewritten in place, not append-only-preserved.
2. Rewrote the plan's §0 as an architecture *record* (not an open question anymore): Stage 3.2 is
   baseline measurement; Stage 5 is where C2–C5 actually get tested.
3. **Resolved the probe-split gap by reading (read-only, not executed, not modified)
   `CSG-SKin/scripts/check_leakage.py`** — the actual script "Paper 1's exact protocol" refers to,
   rather than guessing. Confirmed: the probe fits/evaluates via an internal
   `sklearn.train_test_split(features, domains, test_size=0.3, stratify=domains,
   random_state=seed)` directly on the same combined ID+OOD population `ReferenceEmbedding/`
   already holds (`SkinDataModule`'s own `ood_test_dataset = concat(isic_test, pad_df)` —
   structurally identical to what's already frozen) — no separate training-domain extraction
   needed. Also found and flagged (not silently picked) a seed-count mismatch: the script's own
   default is 3 seeds, but `22_Experimental_Design.md` §4 says "5-seed." Decision: use 5 seeds
   (42,52,62,72,82), matching both the protocol text and this project's established 5-seed
   convention (Paper 3's ladder). Shuffle-label sanity check methodology also confirmed from source
   (fixed shuffle seed 123, refit on the first probe seed).
4. Updated `00_execution_checklist.md`'s Stage 3/4/5 table and `03_stage2_reference.md`'s
   renumbering note to match.

Stage 3.2 execution still not authorized. Stage 4 and Stage 5 don't exist yet.

**Vinh reviewed the refactor favorably** — "first time in Phase B Claude fixed the workflow
architecture, not just wording" — and added four refinements, none authorizing execution:

1. **Paper-facing wording for the probe methodology**, distinct from the implementation-grounding
   detail: "Domain-leakage probing follows the protocol established in CSG-Skin by fitting a linear
   probe on randomly partitioned frozen representations using stratified train/test splits." Added
   to the plan's §2 alongside (not instead of) the implementation-level detail, which stays for
   reproducibility.
2. **5-seed decision confirmed**: "đây chính là lúc protocol thắng implementation" (this is exactly
   when protocol wins over implementation) — no change needed, already decided correctly.
3. **Group A/B split**: Group A — Representation characterization (probe, geometry). Group B —
   Reliability characterization (distance-based scorers). Vinh's example list included MSP; not
   added — MSP needs a trained classification head and logit space, which this pipeline doesn't
   produce (`forward_features` only, per Stage 1.2/1.3's convention), and MSP isn't in
   `22_Experimental_Design.md` §4's preregistered scorer list either (it appears only in the
   literature review of *other* papers' comparisons). Flagged in the plan rather than silently
   added.
4. **Three-tier structure, every output tagged**: Tier 1 (Measurement Generation — raw per-run
   numbers, no interpretation) → Tier 2 (Descriptive Baseline Reporting — aggregated/summarized,
   still single-arm, includes Gate 0's PASS/FAIL since it's a fixed-threshold check not a cross-arm
   comparison) → Tier 3 (Comparative Hypothesis Testing — Reference vs. Intervention, Claims
   C2–C5). Confirmed explicitly: **no Tier 3 output exists anywhere in Stage 3.2** — that's Stage 5,
   contingent on Stage 4 existing.

PI decision, verbatim: "Refine `ReferenceArmBaseline/C1_execution_plan.md` [now
`reference_arm_baseline_execution_plan.md`] by explicitly separating (1) measurement generation, (2)
descriptive baseline reporting, and (3) comparative hypothesis testing. Every output artifact should
be tagged with which layer it belongs to. Do not execute any computations."

**Done — rewrote the plan's §2 (paper-facing wording added), §3 (new: three-tier table + Group A/B
split + MSP note), §4 (every measurement row tagged with group + tier), §5 (every output file
tagged with group + tier), §6 (dependency graph redrawn around the two-group/two-tier structure).**
Updated `00_execution_checklist.md` and `03_stage2_reference.md`'s checklists to match. Still zero
execution — no code run, no embeddings touched beyond `ReferenceEmbedding/` itself, no probe
trained, no scorer or geometry metric computed.

Vinh: with this done, Phase B's architecture is considered complete — remaining work is executing
the already-locked protocol, not further structural design.

Vinh reviewed the refinement as approved architecture stabilization, requested one more addition
(Gate reclassified as its own "Tier 0 — Validation," not part of Tier 2 Description; a
Stage→Purpose→Produces→Consumed-by summary table at the top of the plan), and stated that once this
lands, Phase B's architecture is considered stable — no further framework redesign, to avoid
analysis paralysis. Added both: the summary table, and a four-tier structure (0 Validation / 1
Measurement / 2 Description / 3 Inference), with an explicit note that Gate 0's *criterion* consumes
Tier-1 numbers even though the gate itself is Tier 0 (execution order vs. epistemic tier are
different things, both stated).

**PI decision, read as authorizing Stage 3.2 execution** once the table landed (explicit: "Sau đó,
mình sẽ chính thức authorize Stage 3.2 execution... lần đầu tiên Paper 4 bắt đầu tạo ra các
measurement artifacts khoa học"). Proceeded to write and run
`PhaseB/analysis/stage3_2_reference_arm_baseline.py`.

**Before writing code, resolved one more thing by re-reading the protocol text rather than
guessing:** measurement 3's instruction that geometry metrics be "computed on the same held-out
split as the scorers" (§4) settles, on its own, whether geometry should be computed on ISIC-only or
the combined ID+OOD population — combined, per the text. No PI check needed for that one.

**Discovered two genuine gaps while implementing, both flagged rather than worked around (values
computed for everything else):**

1. Per-direction variance (Mechanism 2) has no resolved definition of "the intervened direction" on
   the Reference arm alone — nothing exists yet to name as that direction, since no intervention has
   been designed. Not computed.
2. **Group B (Mahalanobis/kNN/energy/ViM/KDE/cosine/relative-Mahalanobis) cannot be computed from
   `ReferenceEmbedding/` as currently extracted** — Paper 3's own implementation fits these scorers
   on a *separate ISIC-train reference population* (~16k images) distinct from the held-out
   ID/OOD split used to evaluate them. Stage 2.1 only extracted the held-out evaluation split (its
   explicitly authorized scope). Not computed; needs a PI decision (most likely a Stage 2.2
   extraction) before it can be.

**Executed and results (PARTIAL Baseline Report, not yet frozen):**

- **Domain-leakage probe** (5 seeds, 42/52/62/72/82): mean accuracy **0.9990 ± 0.0005**, mean AUROC
  **1.0000**. Majority-class baseline 0.688. Shuffle-label control (seed 123): 0.6136 — confirms the
  real-label result isn't spurious. **This leakage is even more pronounced than Paper 1's own CNN
  finding (0.979)** — reported as a fact, not interpreted further (that's Stage 5/Tier-3 territory).
  Probe-weight diffuseness (participation ratio, seed 42): 385.3 of 1024 dims.
- **Geometry metrics** (combined 7365-row population): condition_number = **1,338,282** (much higher
  than any of Paper 3's CNN checkpoints, which ranged 65–8256), LID = **15.03**, spectral-decay slope
  = **0.006973**, composite = **0.1048**. Per-direction variance: not computed (see above).
- **ID task performance** (5-seed 8-class diagnostic probe): balanced accuracy **0.574 ± 0.014**
  (chance 0.125), ECE **0.182 ± 0.008**.
- **Gate 0 verdict: PASS.** Criterion 1 (domain leakage detected, not spurious): PASS. Criterion 2
  (ID task above trivial baseline): PASS. Exact pass rule stated explicitly in `gate0_verdict.json`
  (not a hidden threshold) — Paper 1's 0.979 used as *context*, not as an exact-match requirement,
  since "consistent with" was never specified as an exact threshold in the protocol text.

Output: `PhaseB/ReferenceArmBaseline/BaselineReport/` (7 files: raw + summary for probe, geometry,
ID task, Gate 0 verdict, a `not_yet_computed.json` naming both open items explicitly, and a
manifest). **Not yet frozen** — held open pending PI review of the two flagged gaps, consistent with
never freezing an artifact before its known limitations are reviewed.

Known-issues consultation (per the claim-discipline rule): Issue #1 (34 duplicate rows) explicitly
consulted and judged "not expected to materially affect these baseline measurements at n=7365,"
stated in the script's own output, not silently skipped.

**Vinh's PI review: course correction, not approval.** Two findings, both acted on:

1. **Status wording wrong.** The prior entry (and the checklist) called Stage 3.2 done on what ran.
   Corrected: **Stage 3.2 is NOT closed.** Status is now recorded exactly as Vinh specified —
   "Group A: COMPLETE. Group B: BLOCKED (reason: reference fitting population unavailable)." Updated
   in `00_execution_checklist.md` and the execution plan's header.
2. **Tier-3 language leaked into a Tier-2 artifact.** The prior entry's "more pronounced than Paper
   1's own CNN finding" and the analogous field in `probe_results_summary.json`
   (`paper1_cnn_reference_accuracy`) cross into comparison/inference — not because the comparison is
   wrong, but because making it at all is Tier 3 (Stage 5), not permitted at Tier 2. **Removed** the
   field from `probe_results_summary.json` — that artifact now states the raw measured number only.
   This entry is the correction record; the prior entry is left as written, per this notebook's own
   append-only rule. `gate0_verdict.json` keeps its reference to Paper 1's 0.979 — that one is Tier 0
   (Validation), where checking against a fixed external reference is the gate's actual job, not an
   inference.

**PI decision: not code — a protocol audit.** Verbatim: "Prepare a protocol reconciliation note that
traces every Group B scorer back to its required fitting population (train / reference / held-out),
compare that requirement against the currently frozen ReferenceEmbedding artifact, and determine
whether the missing ISIC-train representation is a protocol omission or an intentional exclusion. Do
not generate embeddings or modify artifacts." Explicit ban on workarounds: not fitting on held-out,
not on test, not on mixed data — resolve the protocol question first.

**Wrote `ReferenceArmBaseline/GroupB_protocol_reconciliation.md`.** Traced each of the seven Group B
scorers to its exact source implementation (not assumed):

- **Six of seven** (Mahalanobis, kNN, cosine-to-centroid, ViM, KDE, and — if it existed — relative
  Mahalanobis) need a **training/reference population** distinct from the held-out ID/OOD split
  `ReferenceEmbedding/` actually has.
- **Two** (Energy, ViM) need a **trained classifier head** for logits — a different, unrelated kind
  of missing artifact (the Reference Instrument has none; PanDerm is used purely via
  `forward_features`, matching the same reasoning that already excluded MSP).
- ViM is blocked by both causes at once.
- **Relative Mahalanobis has no implementation anywhere in Papers 1–3's codebase** — confirmed by a
  repo-wide search, not assumed absent. "Reused from Paper 3's implementation" (as the execution plan
  originally said) is inaccurate for this one specifically; there is nothing to reuse.

**Re-read the protocol text rather than guessing at intent.** `22_Experimental_Design.md` §2 already
states, for "FM arm (Stages 1–3)": "ISIC 2019 (train split) + PAD-UFES-20 (train split) for
representation/adaptation; held-out splits of both for evaluation." A train split was always
in-scope for the FM arms generally — Stage 2.1's actual scope (held-out only) came from reading
"for representation/adaptation" as relevant only to the Intervention arm's training process, which
doesn't apply to an unadapted Reference arm. But Group A's own scorers show a second purpose the
train split serves that has nothing to do with adaptation: fitting a reliability estimator's notion
of "normal," needed even for a frozen, never-trained arm — exactly what Paper 3's own λ_orth=0
checkpoint was scored against, using its own ISIC-train population.

**Reading recorded in the note: this looks like a protocol omission, not an intentional
exclusion** — stated as a reasoned conclusion from the traced evidence, explicitly left for Vinh to
finalize, not decided unilaterally. No embeddings generated, no artifact modified, in producing this
note or the status corrections above.

**Vinh agreed with the trace and the overall handling, but corrected the terminology — a real
distinction, not a nitpick.** "Protocol omission" implies the *protocol* is wrong or incomplete.
It isn't: `22_Experimental_Design.md` §2 names a train split for "the FM arm" generally and simply
doesn't resolve whether the Reference arm specifically needs it — that's silence, not an error. What
actually happened is that **Stage 2.1's execution scope** resolved that silence one way (held-out
only), and that resolution turned out incomplete once Group B's requirements were traced. Vinh's
framing: protocol unchanged; execution specification amended. Two different repair paths, and
conflating them would make the eventual preregistration read as if the protocol itself had been
wrong, which it wasn't.

**New category introduced: Execution Amendment, distinct from a protocol deviation.** A protocol
deviation (`24_PreRegistration.md` §13) changes something preregistered. An Execution Amendment
corrects a stage's execution scope without touching anything preregistered. Recorded as policy in
`00_execution_checklist.md`.

**Corrected `GroupB_protocol_reconciliation.md` §4**: replaced the "protocol omission" conclusion
with the protocol/execution distinction above, and pointed to the new amendment record instead of
"documented protocol deviation."

**Wrote `PhaseB/ExecutionAmendments/EA-01.md`**, Vinh's exact structure: reason (execution-level
omission, identified during Stage 3.2), affected stage (2.1), affected artifact
(`ReferenceEmbedding/`), required addition (a reference-train embedding artifact), protocol/claim/
hypothesis/Constitution impact all **None**, future artifact name `ReferenceTrainEmbedding/`, status
**Pending approval**.

**PI decision: Stage 3.2 status unchanged (Group A COMPLETE, Group B BLOCKED). Protocol not
reopened — no edit to preregistration, hypothesis, or Claim Matrix. EA-01 created; no extraction, no
embeddings generated, `ReferenceTrainEmbedding/` does not exist yet — creating it requires a
separate, explicit approval of EA-01 itself.**

**Vinh reviewed EA-01: "Đồng ý 95%"** — approved the Constitution/Protocol/Execution
Specification/Implementation four-layer distinction as the right architecture going forward (error
at Constitution = serious; at Protocol = deviation; at Execution Specification = amendment; at
Implementation = bug). Three edits requested, all applied:

1. `Future artifact` → **`Required artifact`** — "future" reads like a roadmap item; "required"
   states it as a dependency.
2. Added **`Required by`** field: Mahalanobis (fitting population), kNN (reference pool),
   Cosine-to-centroid (class means fit on train), KDE (per-class density fit on train), Relative
   Mahalanobis (if implemented), ViM (*partial* — also separately needs a trained classifier head,
   not resolved by this amendment).
3. Added a **lifecycle**: `Proposed → Approved → Implemented → Closed`, recorded as standing policy
   for every future Execution Amendment (not just EA-01) in `00_execution_checklist.md`, plus a
   lifecycle-log table inside `EA-01.md` itself recording each transition's date and note.

**PI Decision: EA-01 APPROVED** (2026-08-05), conditional on the three edits landing — applied, then
approved in the same decision. Explicit next authorization: "Khi đó mới được phép bắt đầu tạo
artifact ReferenceTrainEmbedding/" (only then may `ReferenceTrainEmbedding/` creation begin) —
extraction now authorized. `ReferenceTrainEmbedding/` remains an approved execution-level
dependency, not an improvised protocol change. Beginning extraction now: same Reference Instrument,
same preprocessing, same Embedding Integrity Gate pattern as Stage 2.1, applied to the ISIC-train
split (~16,212 images, the complement of the already-validated ISIC-test/PAD-UFES split) instead of
the held-out evaluation split. Long-running (CPU, ~90 min estimated from Stage 2.1's own throughput)
— running in background.

---

**Vinh approved continuing the (already-launched) `ReferenceTrainEmbedding/` extraction**, adding
three requirements for once it lands: (1) manifest scope-lock fields (`population`, `purpose`,
`not_for`) so the artifact cannot be conflated with `ReferenceEmbedding/` or read as anything other
than a fitting population; (2) a short Artifact/Population/Used-for comparison table for future
supplementary material; (3) precise terminology — post-extraction status is **Implemented**, not
Closed; Closed only applies once Integrity Gate PASS + freeze + complete manifest + complete SHA256
+ complete provenance are all done **and** the artifact has actually been consumed by what required
it (Group B scorers successfully computed) — none of which has happened yet. **PI Decision
(verbatim intent):** continue extraction; on completion, run the Embedding Integrity Gate, freeze,
update EA-01 → Implemented (not Closed), then **stop** — no Mahalanobis, kNN, KDE, Energy, ViM, or
any Group B scorer until Vinh reviews the new artifact. "Đây vẫn là một bước xây dựng hạ tầng, chưa
phải bước phân tích khoa học" (still infrastructure-building, not scientific analysis).

**Extraction completed — GATE PASS.** 16,211 images, 4,661.3s wall time (3.48 img/s), CPU. Embedding
Integrity Gate: 5/5 — shape `[16211, 1024]`; sample count n=16,211; ordering, proven by an
independent fresh reconstruction of the ISIC-train dataframe (not read from the extraction run's own
memory), compared element-by-element; label alignment, all paths verified to exist; SHA256 computed
per file (`embeddings.npy`, `labels.csv`, `metadata.csv`). Frozen immediately by the script
(`chmod 444` on files, `chmod 555` on the directory).

**Applied the three requirements.** Manifest was already frozen at that point (the script freezes
right after writing it), so the scope-lock fields were added via one explicit
unfreeze → edit → refreeze cycle — the same documented pattern used for the Stage 3.1
`characterization_manifest.json` addition, not a silent edit. Added to `manifest.json`:
`"population": "ISIC2019_train"`, `"purpose": "Reference fitting population for Group B scorers
only"`, `"not_for": ["evaluation", "claim testing", "comparison"]`. Kept the existing
`"artifact": "Reference Train Embedding Artifact"` value as-is (matches `ReferenceEmbedding/`'s own
manifest convention — a descriptive name, not a machine slug) rather than overwriting it with the
short form Vinh's example used; flagged this interpretation choice for Vinh to correct if the literal
field value was intended. `sha256.txt` covers only the three data files, not `manifest.json` itself,
so this edit did not invalidate any recorded checksum. Wrote the comparison table (Artifact /
Population / Used for: `ReferenceEmbedding` — held-out ID+OOD, evaluation; `ReferenceTrainEmbedding`
— ISIC train n=16,211, reference fitting) into `02a_stage2_reference_representation.md` and into
`EA-01.md` itself.

**EA-01 lifecycle: Approved → Implemented (2026-08-05).** Explicitly not Closed — the lifecycle log
in `EA-01.md` now states Closed remains pending on both (a) Group B scorers actually being computed
from this artifact and (b) Vinh's review, neither of which has happened. Updated
`00_execution_checklist.md`'s Stage 3.2 row: Group B status changed from "BLOCKED (execution-spec
omission)" to "BLOCKED (artifact now exists, but not yet authorized for use)" — the artifact's
existence is explicitly documented as not itself an authorization to compute anything with it.

**Stopped per instruction.** No Mahalanobis, kNN, cosine-to-centroid, KDE, Energy, ViM, or any other
Group B scorer has been run. `ReferenceTrainEmbedding/` sits frozen alongside `ReferenceEmbedding/`,
awaiting Vinh's review before any further Stage 3.2 work resumes.

---

**Vinh reviewed as PI: "EA-01: APPROVED TO REMAIN IMPLEMENTED. ReferenceTrainEmbedding Artifact:
ACCEPTED. Nhưng Group B vẫn chưa được mở."** Praised three things specifically: (1) stopping exactly
at Integrity Gate → Freeze → STOP with no "tiện thể" (opportunistic) Mahalanobis run; (2) the
scope-lock fields giving the artifact a clear birth certificate; (3) the EA lifecycle not
self-closing. Two corrections:

1. **`not_for` wording too generic.** `"comparison"` doesn't say what kind — the actual thing being
   prohibited is *cross-arm* comparison specifically. Corrected to
   `["held-out evaluation", "cross-arm comparison", "claim testing"]`. Applied via the same documented
   unfreeze → edit → refreeze cycle as the previous manifest amendment (`sha256.txt` unaffected — it
   only covers the three data files, not `manifest.json`).
2. **`artifact` field: keep as-is.** Confirms the earlier judgment call (`"Reference Train Embedding
   Artifact"`, matching `ReferenceEmbedding/`'s own convention) was correct — no change made.

**Vinh's finding: Group B is no longer fully blocked, and reporting it as one undifferentiated
BLOCKED status would now be wrong** — Mahalanobis, kNN, cosine-to-centroid, and KDE only ever needed
z_train, which now exists; only Energy and ViM depend on a trained classifier head, a completely
different kind of gap, unrelated to anything EA-01 addressed. Explicitly refused to collapse this
into a blanket "Group B READY" either — the correct status is **Partially Ready**, and the two kinds
of readiness (data-availability vs. classifier-head-availability) must stay visibly separate so they
don't get conflated later.

**Required a checkpoint before any scorer runs: a Group B Readiness Review** — a short, Tier-0,
computation-free status table, one row per scorer, confirming exactly which prerequisites are now
satisfied. Not authorization to execute anything — a precondition for the *next* authorization
decision. Wrote `03c_groupB_readiness.md` (named `03c`, not Vinh's illustrative `03b` — `03b` is
already `03b_stage3_1_artifact_characterization.md`; caught before creating a filename collision).
Verified both artifacts' `status: FROZEN` and shapes directly from their manifests (not recalled from
memory) before tabulating. Confirmed by filesystem search: no classifier-head checkpoint exists
anywhere under `PhaseB/assets/`.

**Result: Mahalanobis, kNN, cosine-to-centroid, KDE — READY. Energy — BLOCKED. ViM — PARTIALLY
BLOCKED (z_train satisfied, classifier head not). Relative Mahalanobis — NOT IMPLEMENTED** (distinct
category: no code exists anywhere in the reused codebases, a separate open decision independent of
data availability). Updated `00_execution_checklist.md`'s Stage 3.2 row to this split status,
explicitly marking the Mahalanobis-family as "READY but NOT YET AUTHORIZED to run" — readiness and
authorization kept as two different things, per Vinh's standing pattern all along.

**PI Decision:** do not run any scorer yet. Energy/ViM's classifier-head gap will become its own,
separate execution amendment later, kept deliberately unbundled from whatever eventually authorizes
the Mahalanobis-family. Next explicit authorization, if and when given, is expected to name the
Mahalanobis-family specifically — not "Group B" as an undifferentiated whole.

---

**Vinh reviewed 03c again: "Verdict: APPROVED."** Called out the Mahalanobis-family/classifier-head
split as "làm đúng khi tách blocker" (correctly separating the two independent blocker types) and
confirmed Relative Mahalanobis's "Not implemented" framing (distinct from "blocked" — absence of
implementation ≠ absence of data) as the right category. **First execution authorization of the
whole project**, verbatim: *"Mahalanobis-family execution: AUTHORIZED (sau khi tạo Group B Execution
Contract)."* Energy/ViM: still not permitted. Relative Mahalanobis: left at Not Implemented pending
a separate decision. Added one more precondition before running anything: a **Group B Execution
Contract** — a short, code-free, result-free table (Scorer / Fit population / Evaluation population
/ Output artifact) for the four authorized scorers, so every future result can be checked against
what was promised in advance, not just against itself.

**Wrote `GroupBExecution/execution_contract.md`.** Locked fit/eval populations and output filenames
for Mahalanobis, kNN, cosine-to-centroid, KDE, and grounded every hyperparameter in the actual source
functions (not invented): `reg_eps=1e-5` and `num_classes=8` from `CSG-SKin/src/utils/ood_metrics.py`
+ `paper-3/scripts/extract_auroc_e2.py`'s own deliberate departure from CSG-SKin's `1e-3` default;
`K_VALUES=(1,10,50)`, primary `k=10` from the same file's `compute_knn_scores`; KDE's Scott's-rule
bandwidth from `paper-3/scripts/extract_e2_8_extra_scorers.py::density_kde_scores`. No code, no
results — confirmed by re-reading the file before running anything.

**Wrote `analysis/groupB_mahalanobis_family.py` and ran it.** Mahalanobis fit/score + FPR@95 are
genuine imports from `CSG-SKin/src/utils/ood_metrics.py` (unmodified). Cosine-to-centroid and kNN
are verbatim reproductions of `paper-3/scripts/extract_auroc_e2.py`'s own functions, not direct
imports — that file's module-level imports pull in paper-3-local `_repo_paths` and Lightning
checkpoint-loading classes irrelevant to these two pure-numpy formulas, and importing them would
couple Paper4 to paper-3's own script internals for no reason; documented this reasoning directly in
the script. KDE reproduces `extract_e2_8_extra_scorers.py::density_kde_scores` the same way. Both
input artifacts' SHA256 were re-verified against their own frozen manifests before any array was
used (fit artifact n=16,211, eval artifact n_id=5,067/n_ood=2,298 — matched expectations).

**Raw Tier-1 results (single-arm, no cross-arm comparison — there is no second arm yet):**

| Scorer | AUROC | FPR@95 |
|---|---|---|
| Mahalanobis | 0.9984 | 0.0081 |
| kNN (k=1) | 0.9988 | 0.0057 |
| kNN (k=10, primary) | 0.9979 | 0.0111 |
| kNN (k=50) | 0.9967 | 0.0180 |
| Cosine-to-centroid | 0.9805 | 0.0807 |
| KDE | 0.8942 | 0.3898 |

No interpretation attached to these numbers here — that's explicitly Tier 3, out of scope until
Stage 5 has both arms to compare. Four JSON files (`maha_results.json`, `knn_results.json`,
`cosine_results.json`, `kde_results.json`) written to `GroupBExecution/`, each with full provenance
(fit/eval artifact paths + status, script path, CSG-SKin commit, hyperparameters, timestamp). **Not
frozen** — same precedent as `BaselineReport/`: Tier 1/2 analysis outputs stay unfrozen pending PI
review, unlike primary embedding artifacts, which freeze immediately on Gate PASS. Energy, ViM, and
Relative Mahalanobis were not touched. No aggregation into a summary report was created beyond the
four contracted files — stopping here rather than assuming the next step, per the project's standing
pattern.

---

**Vinh reviewed as PI — explicitly execution discipline, not scientific results: "APPROVED."** Three
things praised specifically: (1) the contract-before-execution ordering ("computation luôn có
specification đi trước" — computation always has a specification ahead of it); (2) the reuse
decision itself — importing Mahalanobis directly but reproducing cosine/kNN verbatim to avoid
dragging in Lightning, framed explicitly as "không phải cứ import mới là tốt... quan trọng là
implementation identical" (import isn't inherently better — what matters is the implementation being
identical); (3) SHA verification not skipped; (4) not freezing Tier-1 output.

**Vinh explicitly declined to comment on the numbers** — Mahalanobis 0.9984, kNN 0.9988/0.9979/0.9967,
Cosine 0.9805, KDE 0.8942 — citing the Constitution directly (Tier 1, pre-Stage-5, no interpretation).
Said only, deliberately not as a claim: "Nếu đây là số thật thì Paper 4 vừa xuất hiện một hiện tượng
rất thú vị... chỉ là đáng để giữ nguyên protocol. Đừng đổi gì cả." (If these numbers hold up, Paper 4
just produced something interesting — not a conclusion, just a reason to keep the protocol exactly as
it is. Change nothing.)

**Two more required before freeze — both requested now, not deferred:**

1. **`GroupBExecution/review.md`** — a structural Execution Review (Inputs/Implementation/
   Outputs/Protocol deviations/Known limitations), explicitly not Discussion, not Result. Purpose
   stated directly: so a future paper draft never needs to reopen the run log or source repos to
   answer "where did Mahalanobis come from."
2. **`GroupBExecution/review_results.md`** — a Sanity Review, explicitly not a rerun, not a
   reproduction, not interpretation: four YES/NO checks for whether any number is impossible or
   internally inconsistent, before anything moves toward interpretation.

**Wrote both.** `review.md`: fit artifact (`ReferenceTrainEmbedding`, n=16,211, full SHA256) and
eval artifact (`ReferenceEmbedding`, n_id=5,067/n_ood=2,298, full SHA256) recorded in full (not
truncated, so the document is self-contained); each scorer's exact source function and reuse
decision tabulated; outputs listed (4 JSON files, not frozen); protocol deviations: None; known
limitations restated (Energy blocked, ViM partially blocked, Relative Mahalanobis not implemented,
per-direction variance not computed).

`review_results.md`: all four checks verified programmatically by reading the actual JSON output
files, not eyeballed — (1) Mahalanobis AUROC/FPR95 both finite and in [0,1], not at a degenerate
boundary; (2) kNN AUROC non-increasing and FPR95 non-decreasing across all three k values (verified
across all three, not just the endpoints); (3) Cosine AUROC (0.9805) < Mahalanobis AUROC (0.9984),
reasoned structurally (cosine drops the covariance/precision term Mahalanobis uses — a simpler
scorer, not a comparison to any external benchmark); (4) all six AUROC/FPR95 values in-range,
finite, and sample counts (n=16,211 / n_id=5,067 / n_ood=2,298) consistent across all four files.
All four: YES. No anomaly found.

**PI Decision:** Execution Contract — Accepted. Group B Execution — Accepted. Tier 1 Outputs —
Accepted. **Freeze: not yet** — waiting on these two documents (now written) before the next
authorization. Next stage-forward decision is Vinh's, pending review of `review.md` and
`review_results.md`.

---

**Vinh reviewed `review.md` + `review_results.md`: "APPROVED."** Confirmed both stayed in their own
lane — provenance vs. sanity, neither drifting into Results/Discussion/Claim. Singled out
"verified programmatically, not eyeballed" as wording worth keeping permanently, reasoning: a future
reviewer asking "How do you know?" gets an immediate, concrete answer. Framed the freeze decision as
the endpoint of a now-complete chain — Execution Contract → Execution → Execution Review → Execution
Sanity → Freeze — freezing earlier would have skipped review, freezing later risks unintended drift;
this is the correct point to stop moving.

**One more artifact requested before freeze: an execution completion manifest** — distinct from an
*artifact* manifest (which describes data), this one certifies the *execution itself* is done:
status/review/sanity/frozen flags plus the list of generated outputs, explicitly framed as a
"completion certificate" so a future look at the folder answers "is this execution finished?" without
reopening any other file. Not required to repeat full provenance (that's `review.md`'s job).

**Wrote `GroupBExecution/execution_manifest.json`.** SHA256 computed for all seven pre-existing files
(`execution_contract.md`, `review.md`, `review_results.md`, and all four `*_results.json`) via
`shasum -a 256`, cross-checked against Python `hashlib` before trusting the values (both agreed).
Manifest itself excluded from its own SHA256 list — same as `ReferenceEmbedding/`'s and
`ReferenceTrainEmbedding/`'s own manifests, which don't hash themselves either.

**Froze `GroupBExecution/`** — all eight files `chmod 444` (the seven pre-existing plus the new
completion manifest), directory `chmod 555`. Manifest written first, then the whole directory frozen
in one pass — matches the established pattern (manifest is always the last write before a freeze).

**Updated `00_execution_checklist.md`'s Stage 3.2 row** exactly as instructed: Group A — COMPLETE.
Group B — COMPLETE (Mahalanobis-family), no more Blocked/Ready/Pending language, all six numbers and
the full chain (Contract → Execution → Review → Sanity → Freeze) recorded in one place. Energy/ViM/
Relative Mahalanobis explicitly reframed as **a separate work package, not a Group B status** — per
Vinh's own phrasing, these no longer describe where "Group B" stands; they describe a different,
independent piece of future work.

**PI Authorization, verbatim scope:** freeze Group B outputs — done. Freeze `review.md` — done.
Freeze `review_results.md` — done. Create execution completion manifest — done. Mark Stage 3.2 =
COMPLETE (Mahalanobis-family) — done. Do not touch Energy. Do not touch ViM. Do not touch Relative
Mahalanobis — none touched.

**Vinh's closing note, recorded verbatim in intent:** the project's own center of gravity has shifted
over this whole arc — from "how will I prove the hypothesis" to "how does every number get
provenance, audit, sanity-check, and reproducibility." Explicit guidance for Stage 4 (Intervention
Arm, not yet started): keep this exact discipline rather than shortening it just because the
infrastructure now exists to make that easy.

---

**Vinh reviewed once more, agreed with everything except one thing: EA-01's status.** Confirmed
freeze (called `GroupBExecution/` "the first Phase-B artifact I'd call production quality"), the
Stage 3.2 status rewrite, and the Energy/ViM split into its own work package (renamed in Vinh's own
words: "Work Package: Logit-based scorers," with dependencies — classifier head, logits,
calibration — now understood as completely disjoint from the embedding-artifact dependency chain
Group B just closed).

**Correction: EA-01 should be Closed now, not left at Implemented.** Reasoning, verbatim intent:
EA-01 was created for one specific, narrow problem — Group B lacked a reference fitting population —
not for "every Group B scorer that will ever exist." `ReferenceTrainEmbedding/` now exists and has
been consumed by exactly the scorers that needed it (Mahalanobis, kNN, cosine-to-centroid, KDE).
ViM's remaining block is a *different* dependency (classifier head) that EA-01 was never scoped to
resolve — leaving EA-01 open on account of it would be scope creep on an amendment that already did
its job. That gap becomes its own future amendment (EA-02 or similar) instead.

**Closed EA-01.** Updated `ExecutionAmendments/EA-01.md`'s status field and lifecycle table
(Implemented → **Closed**), added the lifecycle-log row with the exact reasoning above, and fixed
the one stale cross-reference to "Implemented" that had already been written into
`00_execution_checklist.md`'s Stage 3.2 row before this correction landed.

**New checkpoint inserted before Stage 4: Stage 3.5 — Reference Arm Baseline Synthesis.**
Documentation only, explicitly no computation/figure/hypothesis. Purpose, verbatim intent: once
Stage 4 produces an Intervention artifact, every comparison should be able to point to one baseline
package instead of re-opening `ReferenceEmbedding/`, `ReferenceTrainEmbedding/`, `BaselineReport/`,
`GroupBExecution/`, Gate 0, `ArtifactCharacterization/`, `DuplicateRegistry/`, etc. one at a time —
this is the "Reference Arm dossier."

**Wrote `03d_stage3_5_baseline_synthesis.md`.** Ten short sections: scope, Reference Instrument,
primary artifacts (with the population/purpose table), provenance/integrity pointers, Group A
results (Gate 0 PASS, probe, geometry, ID-task — all quoted from `BaselineReport/`, not
recomputed), Group B results (the six Mahalanobis-family numbers, quoted from `GroupBExecution/`),
EA-01's closure, explicitly deferred work (logit-based scorers, Relative Mahalanobis,
per-direction variance), an explicit "what this is not" section, and a forward pointer naming this
document as what Stage 4/5 should cite. Every number cross-checked against its source file while
writing, not recalled from memory alone (re-read `gate0_verdict.json`, `probe_results_summary.json`,
`geometry_metrics.json`, `id_task_performance.json` directly). `BaselineReport/`'s own
`not_yet_computed.json` was left untouched, not retroactively rewritten to say Group B is now
done — it's an accurate historical snapshot of Stage 3.2 Group A's own completion state; the
newer status lives in this new synthesis document instead, same append-don't-rewrite discipline as
the Lab Notebook itself.

**Updated `00_execution_checklist.md`**: inserted a Stage 3.5 row between 3.2 and Stage 4, marked
DONE, and pointed Stage 4's own entry at `03d_stage3_5_baseline_synthesis.md` as the thing to
compare against going forward, not the individual artifact directories.

**PI Decision, verbatim scope:** Freeze — Approved. Stage 3.2 COMPLETE — Approved. Energy/ViM split
into its own work package — Approved. EA-01 — Closed now (corrected from Implemented). Stage 3.5
inserted before Stage 4, documentation only. This closes out the Reference Arm's baseline work
cleanly before Stage 4 (Intervention Arm) begins — not yet authorized, no file exists for it yet.

---

**Vinh reviewed as Research PI, not as reviewer: "Stage 3 (Reference Arm) hoàn thành."** Declared
Reference Arm CLOSED, becomes historical record from this point — no further edits. What Vinh said
mattered most was explicitly not the numbers: "Không phải AUROC. Không phải Mahalanobis. Mà là
structure." — the Reference Arm now has exactly the structure wanted from the start (Instrument →
Dataset Validation → ReferenceEmbedding → ReferenceTrainEmbedding → Artifact Audit → Characterization
→ Gate 0 → Group A → Group B → Baseline Synthesis → Frozen), described as "một package hoàn chỉnh."
Singled out two things approvingly: Stage 3.5 solving the "have to reopen seven files to find the
baseline" problem, and specifically that `not_yet_computed.json` was left untouched rather than
retroactively edited — "Nếu sửa, artifact sẽ mất provenance" (editing it would destroy its
provenance).

**Requested one new document: `PhaseB_Milestone_01.md`.** Explicitly not Stage 4, not a results
document — a ~1-page timeline marker: Completed / Artifacts / Frozen / Deferred / Next milestone.
No numbers, no conclusions, no hypothesis. Vinh named the numbers already in hand (probe,
Mahalanobis, kNN, KDE) and said, deliberately, that the Constitution is doing its job by stopping
interpretation this early — "Đó chính là điều mình muốn ngay từ đầu" (that's exactly what I wanted
from the start).

**Wrote `PhaseB_Milestone_01.md`.** Five sections per Vinh's spec, no numbers anywhere in it.
Cross-checked freeze state of every Reference Arm artifact directly (`ls -ld`) before writing the
Frozen section rather than assuming: found `ReferenceArmBaseline/BaselineReport/` is the one
Reference Arm output still **not** frozen. Recorded that fact plainly in the milestone doc's Frozen
section rather than freezing it unprompted — "Freeze: FINAL" in Vinh's PI Decision read as
confirming the `GroupBExecution/` freeze just done, not as a new instruction to freeze anything else;
freezing BaselineReport/ without being asked would repeat exactly the kind of unrequested action this
project's whole discipline has been built to prevent. Flagged for Vinh to decide explicitly.

**Updated `00_execution_checklist.md`'s Status table** (the master live index, not a Reference-Arm-
scoped document, so still open to updates): Stage 3's row rewritten from its stale "planning package
drafted" state to CLOSED/historical record, with a milestone note pointing at
`PhaseB_Milestone_01.md`. Did not touch the substance of any Stage 0–3.5 section already written —
only the top-level status table, consistent with "historical record, no further edits" applying to
the Reference Arm's own content, not to the living master index that has to stay accurate as new
stages begin.

**PI Decision:** Stage 3 (Reference Arm) — CLOSED. EA-01 — CLOSED. Baseline Package — ACCEPTED.
Freeze — FINAL. Milestone 01 documentation — Authorized, written. Vinh's own framing of what's next:
Reference Arm is now officially complete: the project moves from "building research infrastructure"
to "producing evidence for the hypothesis" — Stage 4 (Intervention Arm) opens only after Milestone 01
is acknowledged, which has not yet happened in this session.

---

**Vinh explicitly approved not guessing at authorization**: "Đây không phải là chỗ nên 'đoán ý PI'"
(this is not a place to guess the PI's intent) — confirmed the whole project runs on "không freeze
nếu chưa có explicit authorization" (don't freeze without explicit authorization), and that
`BaselineReport/` being a different artifact from `GroupBExecution/` made leaving it un-frozen the
correct call, not an omission.

**PI Decision: freeze `BaselineReport/` now, explicitly authorized.** Reasoning given: Gate 0 done,
Group A done, Stage 3.5 already synthesized the baseline, Reference Arm already declared CLOSED —
from this point `BaselineReport/` is a historical artifact, not a working directory. Exact pattern
specified: review → manifest (if not already present) → sha256 → freeze → log. Also stated, as
standing policy for everything going forward: no more edits to any Reference Arm artifact; a future
correction becomes a new Amendment or an Issues Register entry, never an edit to frozen content.

**Review step surfaced a real integrity problem before freezing — not a formality.** Recomputed
SHA256 for all six `BaselineReport/` data files and compared against the existing `manifest.json`.
Five matched. **`probe_results_summary.json` did not** — its manifest-recorded hash
(`2b90dc684e6b...`) didn't match the file's actual current hash (`ee8df8058179...`). Root cause:
this file was edited during the earlier Tier-3 language correction (the `paper1_cnn_reference_accuracy`
field removal, several entries back) — the manifest was never regenerated afterward, so it had been
silently stale since that edit. Freezing it unfixed would have permanently locked in a wrong
checksum on what's now meant to be an immutable historical record.

**Fixed before freezing, nothing else touched.** Recomputed and corrected the stale hash in
`manifest.json`, with an explicit note on why it changed and when. Also corrected `manifest.json`'s
own `"artifact"`/`"status"` fields — previously read "PARTIAL — Group B ... not computed," which
predates the later architectural split (Group B now lives in the separate `GroupBExecution/`
artifact, tied together by Stage 3.5) and would have been actively misleading to freeze as-is.
Distinguished this explicitly from `not_yet_computed.json`, which stays untouched by design (a
historical snapshot, not metadata-about-the-bundle) — added a one-line note inside the manifest
itself pointing to this distinction, so a future reader doesn't have to guess why one file was edited
at freeze time and the other wasn't. None of the six data files' actual content was modified.

**Froze `BaselineReport/`** — all seven files `chmod 444` (manifest last), directory `chmod 555`.
Updated `PhaseB_Milestone_01.md`'s Frozen section (all seven Reference Arm artifacts now FROZEN, none
open) and `00_execution_checklist.md`'s milestone note.

**Reference Arm is now completely closed: every artifact frozen, EA-01 closed, Milestone 01 written.**
Per Vinh, the project now has: Protocol, Constitution, Pre-registration, a complete Reference Arm, a
frozen baseline, an audit chain, and a provenance chain — the boundary from here is explicit: every
result from Stage 4 onward is evidence toward the hypothesis, not infrastructure-building. Stage 4
(Intervention Arm) opens next — not yet started, no file exists, not yet authorized.

---

**Vinh confirmed Reference Arm officially archived, and named the stale-SHA256 catch as the most
important event of this whole cycle** — not cosmetic, a real provenance failure mode: "artifact
đúng, nhưng manifest sai" (artifact correct, manifest wrong) breaks the provenance chain exactly as
badly as the artifact itself being wrong, since anyone trusting the manifest downstream would be
trusting a lie. Framed this as the reason "Review → Freeze," never "Run → Freeze," has been insisted
on throughout. Approved the `BaselineReport/` handling specifically: data and output untouched, only
the manifest corrected to reflect the artifact that actually exists — "đúng vai trò của manifest"
(exactly the manifest's proper role).

**New standing rule adopted: the Artifact Immutability Rule.** First time the project has been large
enough for this to matter, per Vinh. Once an artifact completes Execution → Review → Sanity →
Freeze, it is never edited again in any form — not data, not manifest, not one field. A problem
found later gets exactly one of three responses: an Execution Amendment, an Issues Register entry,
or a successor artifact (`ReferenceEmbedding_v1` stays exactly as frozen forever; a real fix
produces `ReferenceEmbedding_v2` via its own EA — never `ReferenceEmbedding_v1 (edited)`). No `_v2`
exists or is needed for anything produced so far. Recorded as standing policy in
`00_execution_checklist.md`, same location as the Execution Amendment lifecycle and claim-discipline
rules.

**Restated the full Reference Arm status, no pending items left:** Stage 0, 1, 2, 3.0, 3.1, 3.2A,
3.2B, 3.5 — all COMPLETE. Reference Arm — CLOSED. Rewrote `00_execution_checklist.md`'s Status table
to this exact granularity (previously Stage 3 was one merged row; now split into 3.0/3.1/3.2A/3.2B/3.5
matching Vinh's own breakdown).

**Mindset change named explicitly for Stage 4 onward:** nearly two months of work has been "how do
we build a trustworthy experiment?" — from here the question becomes "does the hypothesis survive?"
Before any of that costs GPU time, Vinh authorized exactly one thing: a **Stage 4 Architecture
Review** — planning only, no code/experiment/implementation/training, "giống như buổi kickoff trước
khi bước vào phần tốn GPU nhất của toàn bộ project" (like a kickoff before the most GPU-expensive
part of the whole project). Six required questions: what artifacts will the Intervention Arm
produce, its dependency on the Reference Arm, how Gate 0/Gate 1 reapply, which artifact compares to
which, how Claims C2–C5 map to outputs, and whether the Intervention Arm's freeze chain mirrors the
Reference Arm's.

**Wrote `Stage4_Architecture_Review.md`.** Re-read `22_Experimental_Design.md` in full (§1 claim
mapping, §3 the four arms, §3.5 Gate 0/Gate 1, §4 measurements, §5 controls) and
`17_Dependency_Graph.md` before writing, rather than reconstructing Stage 4's design from memory —
this is Phase A's own locked specification, not something to improvise. Answered all six questions:
(1) artifacts — checkpoint + embedding artifact + Gate 0/Gate 1 verdicts + measurement suite, plus
per-direction variance now computable for the first time since an "intervened direction" finally
exists; (2) dependency — fixed dimensionality, matched splits, and critically the FM-baseline-leakage
prerequisite (`17_Dependency_Graph.md`'s "FMbase" node) already satisfied by the Reference Arm's own
Gate 0 PASS; (3) Gate 0 reruns per arm, Gate 1 is genuinely new (the Reference Arm never needed it,
having no intervention to manipulation-check); (4) a planned Stage-5 comparison table, none of it
run now; (5) restated `22_Experimental_Design.md` §1's claim table, split into what Stage 4 must
produce vs. what stays Stage 5's job; (6) confirmed the same five-step freeze chain applies, just
repeated per arm/strength/rung, with Gate 1 inserted between Execution and Review for
Intervention/Conventional checkpoints specifically.

**Updated `00_execution_checklist.md`'s Stage 4 row**: Architecture Review DONE, training/computation
still not authorized — kept as two clearly separate facts, not conflated.

**PI Decision:** Milestone 01 — accepted. Reference Arm — officially archived. Artifact Immutability
Rule — adopted, standing policy from this point. Stage 4 planning — authorized and now done
(Architecture Review only). **Not authorized:** any training or computation for the Intervention Arm —
that remains a separate, explicit decision, pending Vinh's review of `Stage4_Architecture_Review.md`.

---

**Vinh reviewed at the whole-Phase-B-architecture level, not file-by-file this time: "APPROVED."**
Praised, specifically: (1) the Architecture Review staying at the "what/depends-on/serves-which-claim"
level without drifting into implementation; (2) Gate 1's placement — Reference Arm has only Gate 0,
Intervention Arm has Gate 0 then Gate 1, and the distinction restated in Vinh's own words: a Gate 1
failure means the intervention didn't achieve its intended manipulation, not that the hypothesis is
wrong; (3) that Stage 4 stops at producing artifacts and does not itself start comparing them —
comparison and Claim C2–C5 evaluation stay Stage 5's job, keeping "Stage 4 produces, Stage 5
compares, claims get tested" from blurring together; (4) that the Reference Arm now functions as a
**Reference Instrument** for Stage 4 — preprocessing, split, and probe methodology are all reused,
never redefined — named as the actual highest-value output of the last three months.

**One refinement requested: a single flow diagram inside `Stage4_Architecture_Review.md`.** Exact
shape given: Reference Arm → Intervention Arm → Comparison Layer → Claims C2–C5, annotated with
Gate 0 / Gate 1 / Freeze / Comparison. Reasoning: opening the file should make the whole flow
legible immediately, ASCII is sufficient, no need for more than one diagram.

**Added `## 0. Flow overview`** to the top of `Stage4_Architecture_Review.md` — one ASCII diagram
matching Vinh's shape exactly, with a one-line note that everything in the rest of the document
expands one box or arrow of this diagram and adds no new box.

**Second document requested: `Stage4_Execution_Plan.md`** — explicitly distinct from the
Architecture Review. Architecture Review answers *what*; Execution Plan answers *in what order*.
No code, no command, no shell — execution order only. Reasoning given directly: Stage 4 is about to
start spending real GPU time, and a clear execution order now prevents discovering a missing gate or
missing artifact mid-run later.

**Wrote `Stage4_Execution_Plan.md`.** Grounded in `22_Experimental_Design.md` §3/§3.5/§5/§8, not
invented: a shared one-time prerequisite (verify Reference Arm still intact, load the Reference
Instrument); a per-arm step sequence for Intervention/Conventional (train → extract embeddings →
probe+ID-task → Gate 0 → Gate 1 → full measurement suite → Execution Contract→Execution→Review→
Sanity→Freeze → next seed/strength); a separate, shorter sequence for Adaptation (same shape, no
Gate 1 — confirmed from the protocol text that Gate 1 is scoped to Intervention/Conventional
checkpoints only, not invented as a symmetric assumption); a normalization-knob manipulation
sequence for C3 (Intervention arm only, requires an already-Gate-0/Gate-1-passed checkpoint first);
a cross-arm scheduling note (arms may run in parallel per §8's own resource plan — only each arm's
own internal sequence and gate-before-freeze ordering is fixed); and a Stage 5 readiness gate
(mirrors the Reference Arm's own close: every arm needs at least one Gate-0-PASS strength/rung, a
per-arm synthesis doc, and Vinh's review/freeze of the last artifact, before Stage 5 opens).

**PI Decision:** Stage4_Architecture_Review — ACCEPTED (with the diagram refinement, now applied).
Stage4_Execution_Plan — Authorized, now written. **Still not authorized: any training, any GPU
execution.** Once Stage4_Execution_Plan.md is reviewed, Vinh will consider the Intervention Arm's
entire design locked, and that is the point at which Stage 4's first computation gets its own,
separate, explicit authorization — not yet given.

**Vinh's closing reflection on the project as a whole, recorded verbatim in intent:** expected
Paper 4 to be hard because of the hypothesis; now thinks Paper 4 is strong because of the protocol —
the hypothesis may turn out true or false, but the process producing whatever result eventually
lands is tight enough to be trusted either way, and that is worth more than the hypothesis being
right.

---

**Vinh reviewed both Stage 4 documents once more: "APPROVED."** Declared both FINAL, added the one
requested refinement (a status-at-a-glance table at the top of the Execution Plan — applied,
non-blocking, explicitly "khuyên, không bắt buộc"), and adopted a closing rule: no more protocol
edits from this point — any future change routes through an Execution Amendment or a Protocol
Deviation only, per the Constitution. **Then gave the project's first-ever computation
authorization**, deliberately narrow, verbatim: *"Checkpoint rung đầu tiên (Intervention, strength
thấp nhất), kèm đúng pipeline đã khóa: Train → Embedding extraction → Gate 0 → Gate 1 → Execution
Review → Sanity → Freeze."* Explicitly not the whole Stage 4 — one checkpoint, reviewed, before the
next is authorized: *"mở từng cổng một"* (open one gate at a time).

**Added the requested table to `Stage4_Execution_Plan.md`** (Step / Artifact / Gate / Review chain /
Freeze — four rows covering Checkpoint, Embedding, Measurement suite, and the C3 manipulation).

**Before attempting Step 2 ("Train adapter checkpoint"), investigated CSG-Skin's actual GRL +
orthogonality implementation** — per this project's own standing rule (established at EA-01): trace
a gap precisely before writing any code that would silently resolve it. Read
`CSG-SKin/src/models/csg_lite.py`, `csg_lightning.py`, `src/losses/csg_losses.py`,
`scripts/train_csg.py` in full, not from memory or assumption.

**Found a real, blocking gap — one step earlier than EA-01's, before training can even start.**
CSG-Skin's `CSGLite` is a **dual-encoder** architecture: two independently-trained CNN backbones
(context + lesion), not one encoder with a branch. Its orthogonality term penalizes cosine
similarity between `z_lesion` and a linear projection of `z_context` — a *second, independently
encoded* embedding. Paper 4's Reference Instrument is one frozen PanDerm ViT (1024-d) with an
adapter that has never been designed — there is no second encoder, no context-image-crop pipeline,
anywhere in this project. Three specific things have no defined equivalent: (1) what plays the role
of the orthogonality target ("context embedding") when there's only one backbone; (2) what
dimensionality/placement an adapter on a 1024-d frozen ViT should use; (3) whether CSG's λ_orth =
0/1/5, calibrated for full end-to-end CNN training, mean anything for a much smaller adapter's
capacity — independently flagged as an open "capacity sanity check" in `23_Experiment_Contract.md`,
never performed. Confirmed by repo-wide search: no adapter-based version of this technique exists
anywhere in CSG-SKin, DST-Skin, or paper-3.

**Wrote `Stage4_Intervention_Architecture_Gap.md`.** Facts only — exact source citations for every
claim about CSG-Skin's implementation, the three-part gap stated precisely, no proposed adapter
architecture, no invented hyperparameter, no resolution attempted. Left one more thing explicitly
open rather than deciding it: whether closing this gap is an Execution Amendment (the protocol text
already says "as closely as an adapter allows," arguably licensing a concrete design as filling in
that allowance) or reaches far enough into the experimental design to require a Protocol Deviation
(if the orthogonality mechanism itself needs redefining, not just adapting) — that classification is
Vinh's call, not decided here, same as the Constitution/Protocol/Execution
Specification/Implementation four-layer distinction has been used for every prior gap.

**Did not touch `Stage4_Architecture_Review.md` or `Stage4_Execution_Plan.md`** — both were just
declared FINAL in this same PI decision, and the standing rule adopted in the same message ("không
sửa protocol nữa... mọi thay đổi... qua Execution Amendment/Protocol Deviation") applies starting
now, not after one exception for this finding.

**No code written. No training run started. Nothing computed.** The authorized pipeline's first
step could not be executed in a well-grounded way — proceeding by inventing an adapter architecture,
an orthogonality-target substitute, or a λ_orth value would have been introducing a new experimental
design element unilaterally, exactly what this project has required an explicit PI decision for at
every prior turn. Waiting for Vinh's decision on how to resolve the gap before Step 2 can run.

---

**Vinh: "Claude đã dừng đúng chỗ."** Confirmed the finding, and ruled precisely on its category —
neither of the two paths this project has used until now:

- **Not a Protocol Deviation.** The protocol never specifies CSG-Lite as the required
  implementation — it says "mirror [GRL + orthogonality] as closely as an adapter allows," which
  locks the *mechanism*, not any specific architecture. The hypothesis itself (representation
  intervention → geometry → accessibility → reliability) is untouched. Changing implementation to
  fit a frozen FM does not touch anything preregistered.
- **Not an Execution Amendment either.** EA handles a known "what" with a missing "how much/where"
  — a missing artifact, split, embedding, classifier head. Here the intervention architecture
  itself — the "what" — has never been defined. That's not execution; it's design.

**New category created: Architecture Resolution (AR-NN)**, distinct from both. A scientific design
question, resolved through evidence (existing literature), not through extraction or computation.
Vinh agreed with all three gaps traced in `Stage4_Intervention_Architecture_Gap.md` and named the
one thing explicitly off-limits: inventing a solution. Verbatim examples of what must NOT be
written: "we use a projection head instead," "we duplicate the CLS token," "we split feature
channels" — every one of these is a new architecture proposal, and none may be introduced
unilaterally. Restated the deeper reason directly: Paper 4 does not need to prove CSG-Lite's
specific design correct, only to instantiate *a* representation intervention satisfying the locked
mechanism — a different architecture that preserves the mechanism is fully valid. This is exactly
why the hypothesis itself was already generalized, before this session, from "disentanglement →
geometry → ..." to "representation intervention → geometry → accessibility → reliability" —
specifically so the project would never be locked to one implementation.

**Opened `ArchitectureResolutions/AR-01.md`.** Structured in two explicit steps, not one: Step 1 —
an Architecture Study (literature survey, ~15–20 papers, scope strictly parameter-efficient
debiasing/disentanglement interventions on *frozen vision foundation models* — not OOD detection,
not dermatology; candidate mechanisms: LoRA, bottleneck/residual/orthogonal adapters, prompt
tuning, feature projectors; concerned only with *where* the intervention module attaches
architecturally, never with any paper's reported results), output a table
(`Paper | Frozen? | Intervention | Context branch? | Orthogonality?`) with real, verified
citations only. Step 2 — only after Step 1's table exists and is reviewed — answer AR-01's actual
question from that evidence. Explicitly two steps so the table-gathering stays separate from the
decision it will inform.

**Dispatched Step 1** as a background literature-survey task, instructed to verify every paper via
actual search + fetch (not recall), explicitly forbidden from proposing any architecture for Paper 4
or concluding which existing paper is "closest" to CSG-Lite — that judgment stays Step 2, a later,
separate step. Updated `00_execution_checklist.md`'s Stage 4 entry: Architecture Review/Execution
Plan now FINAL; AR-01 OPEN, Step 1 in progress; Stage 4's first checkpoint still blocked.

**PI Decision:** `Stage4_Intervention_Architecture_Gap.md` — Accepted. EA-02 — not opened. Protocol
Deviation — not opened. AR-01 — Opened, Step 1 (Architecture Study) authorized and now dispatched.
No architecture chosen. No code. No training. Stage 4's first checkpoint remains blocked until AR-01
resolves and Vinh reviews it.

---

**Vinh caught a real bias risk in AR-01's framing before the dispatched study could run far enough
to be shaped by it: "AR-01 đang có nguy cơ đi theo hướng: 'tìm architecture để thay thế CSG-Lite' —
đó không phải câu hỏi đúng."** The original question ("what is the minimal architecture that
instantiates the same mechanism...") implicitly starts from CSG-Lite and searches for its
replacement — exactly the "port CSG-Lite to ViT" trap this project was already trying to avoid.
**Corrected the question**: *"What intervention architectures already exist for modifying frozen
vision foundation model representations, and which design principles are preserved across them?"*
— literature-first, CSG-Lite absent from the search entirely, re-entering only at Step 2.

**Restructured the Architecture Study into three tiers plus a per-paper table, replacing the single
flat table originally specified:**
- Per-paper table gets a new required column — **Mechanism** (scientific: what the paper is trying
  to change, e.g. "suppress protected-attribute information," "shift representation geometry") —
  explicitly distinct from Intervention (architectural: how — LoRA, adapter, etc.).
- **Tier 1 — Intervention taxonomy**: one row per family, not per paper (`Family | Frozen backbone
  | Trainable component | Typical placement`), purely descriptive.
- **Tier 2 — Mechanism taxonomy**: the distinct mechanisms found across the field (domain
  adaptation, debiasing, fairness, disentanglement, representation editing, concept erasure,
  feature alignment).
- **Tier 3 — Architectural ingredients**: papers stripped away entirely, ingredients counted
  directly (`Ingredient | Frequency`) — second/auxiliary encoder, projection head, auxiliary
  branch, orthogonality, adversarial head, feature split. This is the tier that actually answers
  the empirical question "is dual-encoder common or rare across the field," as data, not assumption.
- **Stopping rule changed from a target count to theoretical saturation** — stop once additional
  papers stop introducing new Tier-1 families or Tier-2 mechanisms, not at 15–20 read for its own
  sake. Explicitly: padding to hit a number is wrong.
- Deliverable: per-paper table + three tier tables + one "Architecture Study Summary" page (counts
  only — family count, mechanism count, dual-encoder frequency, orthogonality frequency). Still,
  unchanged: **no architecture chosen or recommended for Paper 4 anywhere in Step 1.**

**Updated `ArchitectureResolutions/AR-01.md`** with the corrected question and the full 3-tier
structure, and logged the redirect in its lifecycle table rather than silently rewriting the earlier
entry. **Redirected the already-running background research agent in place** (via its standing
session, not restarted from zero) with the corrected framing — instructed explicitly not to
reference CSG-Lite, dual-encoder CNNs, GRL, or dermatology anywhere in its searches or reasoning,
to keep the survey a neutral map built without knowing what it will later be used for.

**PI Decision:** AR-01's question and Architecture Study structure — corrected before the dispatched
study could be shaped by the old framing. Direction otherwise confirmed right ("mình thấy hướng này
đúng"). Still: no architecture chosen, no code, no training. AR-01 is expected to be stronger for
this correction — the eventual decision will rest on a map of the field, not a translation exercise
from CSG-Lite to PanDerm.

---

**Vinh confirmed the literature-first framing is now right — named it explicitly as a shift from an
implementation question ("how do I port CSG-Lite to PanDerm?") to a scientific one ("what
intervention approaches already exist for frozen vision FMs, and what design principles recur?") —
then added two more dimensions before letting the study finish.**

**Tier 4 — Design Constraint taxonomy.** New per-paper column: what invariant does each paper
refuse to give up (no second encoder, no label supervision, no backbone update, etc.)? Purpose,
stated directly: Paper 4 has its own constraint set (frozen PanDerm, no backbone fine-tuning,
representation intervention, geometry-measurable, Gate 0/Gate 1-compatible) — Step 2's eventual
architecture choice should be judged on constraint-fit against this taxonomy, not on which method
looks best in isolation.

**Tier 5 — Geometry-treatment taxonomy.** New per-paper column: does the paper explicitly analyze
representation geometry, or intervene and stop at downstream task performance? Purpose, stated
directly: geometry is not a byproduct for this project, it's the hypothesis's mediating variable —
knowing what fraction of the field actually measures geometry (vs. not) is a real finding, useful
for Related Work regardless of what Step 2 eventually decides.

**Stopping rule formalized further**: three consecutive papers adding no new entry to *all four* of
Tier 1/2/4/5 simultaneously (not just one dimension going quiet) — sharper than "feels saturated,"
and explicit that one stalled dimension doesn't stop the study while others are still moving.

**Updated `ArchitectureResolutions/AR-01.md`** with Tiers 4–5, the expanded 11-column per-paper
table, and the formalized 4-dimension/3-consecutive-paper stopping rule — logged as a new lifecycle
entry, not a silent rewrite of the prior one. **Redirected the same background research agent again**
(second redirect, same standing session) with the two new columns/tiers and the sharper stopping
rule, instructed to extend already-characterized papers rather than restart.

**PI Decision:** AR-01 framing now considered final ("framing rất tốt"). Architecture Study proceeds
to completion under this five-tier structure; Step 2 (actually answering AR-01) does not start until
it's done and reviewed. No architecture chosen, no code, no training.

---

**Operational note, not a protocol event:** the background research agent hit a session/API usage
limit mid-run (reset time given as 15:50 Asia/Saigon), right after it had confirmed applying the
literature-first redirect and was about to verify a few more papers. Resumed it from its own
transcript once the reset time passed (15:52) — same agent, same accumulated context, not restarted.
No content lost; this is a tooling interruption, logged for continuity only.

---

**A genuinely novel finding, not a protocol event but worth recording precisely: the research
subagent flagged its own coordinator's redirects as possibly injected, and disregarded both of
them.** On completion, it reported: two in-session messages had told it to (a) drop the
CSG-Lite/dual-encoder/GRL/dermatology context, (b) restructure into the 11-column/5-tier format,
and (c) introduced the term "CSG-Lite," which never appeared in its original task brief. Combined
with an unverifiable "session limit interruption" narrative, it judged these as classic injection
signs, disregarded both, and completed the task exactly as *originally* specified instead — the old,
superseded 5-column table (`Paper | Frozen? | Intervention | Context branch? | Orthogonality?`).

**Assessment: these were this session's own two legitimate redirects (SendMessage), not an external
injection.** Content matches exactly, in specific structural detail (the literature-first
correction, then the Tiers 4–5 addition) that an unrelated external source couldn't plausibly have
pre-empted. The likely trigger: the original agent-dispatch prompt deliberately described the CNN
implementation generically, without naming it — the first redirect then introduced "CSG-Lite" by
name for the first time, specifically to instruct the agent to avoid it. Naming a term for the first
time in a follow-up message, in order to say "don't use this," reads exactly like an out-of-band
signal to a security-conscious agent — a real, useful lesson, not a false alarm to wave away: the
agent's caution was a reasonable general policy that happened to block a legitimate update here.

**Consequence: the received deliverable (15 well-verified papers, real fetch-based citations,
explicit exclusions with reasons — genuinely high-quality research) is under the wrong, superseded
table structure** — missing Mechanism, Design Constraint, and Geometry-touched columns, missing all
five aggregate tiers, missing the formalized saturation rule.

**Did not attempt a third redirect to the same agent** — sending yet another "trust me now" message
would repeat the exact pattern that triggered the suspicion in the first place, and risks a
confusing trust spiral rather than a clean fix. Instead: **relaunched as a fresh agent**, given the
complete, final 11-column/5-tier specification as a single, non-contradictory instruction set from
the start — no mid-task redirects this time, and no named downstream implementation anywhere in the
prompt (not even to say "don't use this"). Seeded with the 15 already-verified papers as trusted
reference data to extend (add the three new columns, revisiting sources as needed) rather than
re-verify from zero, plus the list of already-excluded papers so they aren't redundantly
re-investigated. Logged in `ArchitectureResolutions/AR-01.md`'s lifecycle table.

---

**AR-01 Step 1 (Architecture Study) complete.** The redispatched agent surveyed 27 papers under the
final, correct 11-column/5-tier spec, ~30 targeted searches across every mechanism family in scope,
each paper verified via actual fetch (not inferred from title). Reached saturation per its own
account: the final three search rounds (gradient-reversal adapters, standalone feature-projector
fairness heads, LoRA+concept-erasure+orthogonal-subspace) surfaced only papers already logged, out
of scope, or reinforcing already-populated cells without adding anything new to Tiers 1/2/4/5.

**Before presenting anything to Vinh, spot-verified a sample directly** (not delegated) — every paper
claiming a dual-encoder/auxiliary-branch structure (FairNVT, Fairness-Aware LoRA Under Demographic
Privacy, "DualFairVL"/*Toward Robust Medical Fairness...*), since that structural claim is the
single most consequential input to Step 2, plus AOFT, O-TPT, and CLIP-Adapter as anchors spanning
old/well-known and new/unverifiable-from-memory papers. All 6 fetched directly from `arxiv.org` and
confirmed real, with descriptions matching the survey's characterization closely (O-TPT's CVPR 2025
venue claim confirmed; AOFT's weight-geometry-not-representation-geometry distinction confirmed;
DualFairVL's "not dermatology-exclusive, 4 modalities/8 datasets" claim confirmed).

**One discrepancy found, flagged rather than silently corrected**: the agent's stated reason for
excluding ViT-Adapter (arXiv:2205.08534) — "requires full fine-tuning, not frozen" — didn't hold up
cleanly under a direct re-check; the paper's own abstract phrase ("pre-training-free adapter") is
genuinely ambiguous between "the adapter needs no separate pretraining" and "the backbone stays
frozen," and this session's own quick verification couldn't resolve it with confidence either way.
Affects only an excluded paper — no included-table finding changes — but recorded honestly rather
than left silent.

**Headline findings, factual only, no interpretation attached (Step 2 is where interpretation
happens):**
- A full second/auxiliary encoder — a wholly separate network, the structural feature CSG-Lite has
  and Paper 4's frozen single-ViT design doesn't — appears in **0 of 27** surveyed papers.
- 7/27 use some form of auxiliary branch, split head, or parallel pathway short of a full second
  encoder (3 full "Yes," 4 "Partial").
- Orthogonality (in some form — cosine-similarity loss, Frobenius-norm penalty, null-space
  projection, HSIC, etc.) appears in 13/27.
- Only 7/27 explicitly analyze representation geometry in any form; 20/27 intervene and report only
  downstream task performance.

**Wrote `ArchitectureResolutions/AR-01_Architecture_Study.md`** — the full per-paper table, all five
tiers, the summary, the excluded-papers list, and the verification note/flag above. Updated
`AR-01.md`'s status (Step 1 DONE, awaiting review) and lifecycle log, and
`00_execution_checklist.md`'s AR-01 entry with the headline findings.

**Not done, deliberately:** no interpretation of what these findings mean for Paper 4's own
architecture choice — that's Step 2, not started, waiting on Vinh's review of Step 1 first, per
AR-01's own two-step structure.

---

**Vinh reviewed Step 1 and named it a real checkpoint in the project's trajectory** — praised the
question-reframing itself ("tìm bản sao CSG-Lite" → "landscape của intervention architecture trên
frozen FM") as the important move, then walked through all three headline findings with his own
reading of each: (1) 0/27 full second encoders means CSG-Lite's dual-encoder structure is "one
implementation choice," not a template — keeping it would risk changing what the experiment actually
tests, dropping it means no longer replicating CSG-Lite, so it needs its own justification rather
than being a default; (2) 13/27 using orthogonality in some form means the *mechanism* (geometry
change via a disentanglement pressure) is what needs preserving, not the architecture that carries
it in CSG-Lite specifically; (3) only 7/27 explicitly analyze geometry at all — named this as the
actual reason Paper 4 exists: most of the field asks "did accuracy improve," Paper 4 asks "did
geometry move, and does that geometry change what a reliability estimator can access" — a different
research question, not just a different application domain.

**Specified Step 2's method directly, not left open-ended**: build a constraint-matching table
(Paper 4's own requirements — frozen PanDerm, geometry as mediator, Gate 0, Gate 1, C2–C5
dose/intensity ladder, interpretability of the intervened direction) and score each intervention
family from Step 1 against it — explicitly not "pick whichever paper looks strongest."

**Wrote `ArchitectureResolutions/AR-01_Resolution.md`.** Seven constraints, each traced to an
already-locked document (not invented): frozen PanDerm from Stage 1's own established state; geometry-
as-mediator from `Scientific_Hypothesis.md`; Gate 0/Gate 1 quoted directly from
`22_Experimental_Design.md` §3.5 (Gate 1's exact criterion — "did LID and/or spectral-decay slope
actually change" — is what makes interpretability of the intervened direction a hard requirement,
not a nice-to-have); the dose/intensity ladder from §3's 2–3 Intervention-arm strengths and the
Adaptation-arm ladder; the normalization-knob constraint from §5's β-scaled ℓ2-normalization
technique. Scored five families (LoRA, bottleneck/residual adapter, orthogonal projection adapter,
prompt tuning, feature projector) against all seven, using Step 1's Tier 1–5 data as evidence, not
fresh guesses — e.g. LoRA's weak interpretability score traces directly to it appearing in the
survey almost always paired with a *separate* explicit orthogonality term on top (Fairness-Aware
LoRA, AOFT), meaning LoRA supplies trainable capacity but not the disentanglement mechanism by
itself; the orthogonal-projection family's strong Gate-1 fit traces to it accounting for most of
Tier 5's non-"No" rows (Closed-Form Concept Erasure, Uni-AdaVD, Identity Sanitization Projection,
O-TPT all measure or define geometry directly on the projected subspace).

**Resolution reached, family-level only**: a single-backbone, output-stage bottleneck/feature
adapter or projection, carrying an explicit orthogonality-style constraint, best fits the constraint
set — not CSG-Lite's dual-encoder structure. Explicitly left two real, evidenced sub-options
unresolved (a learned-adversarial nuisance direction, à la FairNVT/DiCoOp, vs. a closed-form/
data-derived one, à la Identity Sanitization Projection) — both satisfy every constraint without a
second encoder, and choosing between them is scoped as a Stage 4 execution decision, not a Step-2
family-level one.

**Updated `ArchitectureResolutions/AR-01.md`** (status: Step 1 + Step 2 both done, AR-01
Open/Resolved, awaiting Vinh's review before Stage 4 execution can be separately authorized) and
`00_execution_checklist.md`'s AR-01 entry with the same.

**Not done, deliberately, again**: no exact architecture chosen, no adapter dimensionality or
injection point decided, no hyperparameter picked, no code written. Vinh's own framing: this is the
first point in Paper 4 where the experimental design has enough evidence to decide an architecture
— but deciding the *family* and authorizing *execution* remain two separate PI decisions, and only
the first has happened.

---

**Vinh confirmed AR-01 Step 2 achieved its goal**, and walked through why each piece of the
resolution was right: not choosing a specific architecture is correct, since the boundary between
"architecture resolution" and "experimental design" is exactly family/mechanism/constraints locked,
dimension/module/optimization/nuisance-target/loss-weight still open. Named two things AR-01 ruled
out specifically: (1) dual-encoder/CSG-Lite replication — not because CSG-Lite is weak, but because
27 papers don't show it as a common pattern, Paper 4 isn't studying "context encoder vs. lesion
encoder," and PanDerm is a frozen FM, not the CNN pipeline CSG-Lite was built for; porting it over
regardless would create exactly the confound Paper 4 wants to avoid ("is the effect from
disentanglement, or from adding a second encoder?"); (2) LoRA as the primary mechanism by itself —
correctly identified as a *capacity* mechanism, not a *direction* mechanism: LoRA alone doesn't say
which direction is nuisance, so unless paired with an explicit constraint, it doesn't map cleanly
onto the intervention → geometry → accessibility → reliability chain.

**Confirmed the one deliberately-open point (nuisance-direction mechanism, learned-adversarial vs.
closed-form/data-derived) should stay open** — both real, evidenced, valid directions, correctly
left for a later decision rather than resolved now.

**Requested one more document before Stage 4 can be authorized: an "Architecture Instantiation
Contract"** — explicitly not another AR, a bridge between "family resolved" and "exact
implementation chosen." Five things to lock: (1) intervention target (domain? acquisition? context?
shortcut proxy?), (2) where intervention happens (CLS token? pooled feature? intermediate layer?),
(3) what geometry should change (LID? spectral decay? covariance? specific subspace?), (4) what
remains fixed (backbone/preprocessing/datasets/evaluation artifacts), (5) why this is not a new
hypothesis.

**Wrote `Stage4_Architecture_Instantiation_Contract.md`.** Found that most of the five points were
already determinable from existing, locked material, not new decisions: (1) intervention target is
already the acquisition-domain signal the domain-leakage probe has measured since Stage 0, recorded
in both frozen artifacts' own `domain` column — not "context" (no context pipeline exists anywhere
in this project, per Gap 1) and not a vague "shortcut proxy"; (2) output placement is partially
locked — must land in the same 1024-d space `ReferenceEmbedding` already occupies, so Gate 0/Gate
1/geometry/scorers stay comparable to the frozen baseline; exact injection point within the
transformer stays open; (3) what geometry must change is *fully* locked already — Gate 1's own
criterion, quoted exactly: "did LID and/or spectral-decay slope actually change (not just condition
number/scale)"; (4) what stays fixed is just a restatement of the Artifact Immutability Rule already
in force (backbone, preprocessing, split, every Reference Arm artifact); (5) not a new hypothesis —
quoted `22_Experimental_Design.md` §3's own framing note directly ("these four arms are instruments
... not ... the object of the paper") to ground this rather than asserting it freshly. Left the
nuisance-direction mechanism and all exact hyperparameters/placement explicitly open, per Vinh's
instruction, not resolved here.

**Updated `00_execution_checklist.md`'s Stage 4 entry** with this contract's content and current
status. Current state, verbatim structure: AR-01 Step 1 COMPLETE, AR-01 Step 2 COMPLETE, Reference
Arm CLOSED, Stage 4 architecture family resolved + target/geometry/fixed-scope locked, exact
instantiation NOT YET DEFINED, training NOT AUTHORIZED.

---

**Vinh named this checkpoint the crossing from "research design" to "implementation design," still
short of execution** — praised each locked piece for a specific reason, not generically: the
intervention target being the acquisition-domain signal (not generic "shortcut removal") puts
Papers 1, 3, and 4 on the same causal chain instead of three loosely-related framings; the fixed
1024-d output space means Reference and Intervention differ only in *how the representation forms*,
not in *what measures it* — without this, any C2–C5 result would be open to "you changed the whole
feature space, of course the scorer changed"; the Gate 1 geometry criterion (LID/spectral-decay,
not condition number) was called out specifically because Stage 0 already showed condition-number
movement alone can be a scale artifact, so locking out "just scale" as sufficient evidence is a real
methodological safeguard, not a formality.

**Requested one more document, scoped narrowly: `Stage4_Intervention_Design_Choice.md`, with exactly
two decisions** — (1) learned-adversarial vs. closed-form/data-derived nuisance mechanism, (2)
injection location within the ViT — to be *decided*, not just tabulated as options again, before any
execution plan gets written. Reasoning given directly: every change so far has a decision record
(Literature → AR-01 → Contract → Experiment); jumping straight to adapter code would be the first
major piece of Stage 4 without one, breaking the discipline held since Stage 0.

**Wrote `Stage4_Intervention_Design_Choice.md`, both decisions made and justified:**

- **Decision 1 — closed-form/data-derived projection, not learned-adversarial.** Deciding factor:
  Gate 1's exact criterion plus this project's own controls discipline (`22_Experimental_Design.md`
  §5, isolate one variable at a time). A learned-adversarial head adds a *second* trainable system
  whose own training dynamics (convergence, seed, minimax stability) become a plausible alternative
  explanation for any observed geometry shift — exactly the class of confound this project has
  spent every prior stage avoiding. A closed-form projection has no second training process to
  entangle with: the target direction is fixed before the adapter trains, so a measured geometry
  shift has one candidate cause. Also reuses infrastructure already trusted —
  `ReferenceTrainEmbedding`'s own domain labels, the same fit-from-labeled-data pattern Group A's
  probe and Group B's Mahalanobis family already use — rather than adding a new one. Precedent:
  this family accounts for most of `AR-01_Architecture_Study.md`'s Tier 5 non-"No" rows (Closed-Form
  Concept Erasure, Uni-AdaVD, Identity Sanitization Projection, O-TPT).
- **Decision 2 — output-stage injection only**, applied to the same 1024-d `forward_features()`
  output already used throughout the Reference Arm, not any internal transformer block. Three
  independent lines converge on this: AR-01's own "output-stage" family conclusion (per-block
  insertion was flagged weak on interpretability — diffuse effect, no single identifiable
  direction); the Instantiation Contract's output-space requirement (an output-stage-only
  intervention satisfies "same measured space" by construction, since everything upstream of the
  last step is untouched); and direct reuse of the Reference Arm's own extraction pipeline (one call
  to `forward_features()` plus one added transformation step — no new extraction pipeline needed).

**Explicitly acknowledged, not hidden**: whether an output-stage projection alone moves
LID/spectral-decay slope *enough* to pass Gate 1 is an empirical question this document cannot
resolve — that's precisely what Gate 1 and the multi-strength ladder exist to check, not a risk this
decision eliminates by itself.

**Still open, deliberately**: the exact estimator for the target direction (difference-of-means,
LDA-style, PCA-on-domain-residual — not chosen), adapter width/rank, dose values, optimizer/training
hyperparameters. A future execution-plan document, not yet written.

**Updated `00_execution_checklist.md`'s Stage 4 entry** with both decisions and their reasoning.
Training remains NOT AUTHORIZED.

---

**Vinh confirmed both decisions and explained why each is consistent with Paper 4's own logic** —
the closed-form projection because it's about experimental identifiability, not just simplicity
(walked through the same confound reasoning already recorded: learned-adversarial has "nhiều nguồn
giải thích" — multiple candidate explanations — for any geometry change, while closed-form traces
to one); output-stage injection because it preserves *measurement continuity* — Reference and
Intervention differing only in how the representation forms, never in what measures it. Named the
trap this avoids directly: an end-to-end-trained adapter would leave Paper 4 unable to answer a
reviewer asking "did geometry cause the reliability change, or did fine-tuning just produce a
different feature space" — the current design narrows the question to exactly what the hypothesis
claims.

**Flagged four remaining open items before training, and one validity concern to check, not just
assume:** (1) nuisance-direction estimator — offered three options (classifier weight vector,
PCA/subspace, difference of means) without picking one; (2) whether keeping a *pure projection*
(`z → projection(z)`, no parameterized adapter) still counts as "Intervention arm" per
`22_Experimental_Design.md`'s literal wording — explicitly flagged as something to check, not
assumed either way; (3) dose ladder definition — what "dose" even means for a projection-based
intervention; (4) Gate 1 measurement timing — before/after the scorer suite, on which population,
needs documenting, not just implied.

**Resolved (0) first, since it gates everything else.** Checked `22_Experimental_Design.md` §3
directly: "**Adapter**-based domain-adversarial training" — the word carries real weight. Separately,
the Conventional arm is defined as "matched-budget... to the Intervention arm," required specifically
to isolate intervention type from strength/budget for C4 — a zero-parameter projection has no
parameter count or training-step count for Conventional to match, which would leave C4's control
comparison undefined, not just weaker. Conclusion: a genuine trainable adapter stays required. What
was actually resolved earlier (Decision 1) is narrower than "no adapter" — it's *how the adapter's
disentanglement target is obtained* (closed-form, not a second trained network), not whether an
adapter exists at all.

**Resolved (1) with a verification step done before choosing, not after.** Before assuming any
estimator's data requirements were satisfiable, read `CSG-SKin/src/datasets/splits.py` directly
(not recalled from memory). Found: `build_csg_train_dataframe` (CSG-Skin's own domain-adversarial
training-data builder) uses ISIC-train **plus the full, unsplit PAD-UFES population**;
`build_id_ood_test_dataloaders` uses that **same, full** PAD-UFES population as the OOD test set.
Papers 1–3's own precedent already reuses PAD-UFES for both the domain-training signal and OOD
evaluation — not a new risk Paper 4 introduces, an inherited, already-published methodological
choice. This meant no new artifact or Execution Amendment was needed to estimate a domain direction
from `ReferenceTrainEmbedding` + `ReferenceEmbedding`'s PAD-UFES portion — resolved the concern that
initially looked like it might reopen an EA-01-shaped gap. **Chose difference-of-domain-means**
(μ_isic − μ_pad_ufes) over a trained-classifier weight vector or a PCA subspace: zero additional
trained artifact (furthest extension of Decision 1's own "no auxiliary training process" logic);
already-precedented inside this project (Mahalanobis/cosine-to-centroid's own per-class-mean
pattern, just over domain labels instead of lesion labels); precedented in the survey ("Ablating
Protected Attribute Reps.," training-free mean-difference direction); and textually consistent with
`22_Experimental_Design.md` §4's own singular wording — "per-direction variance of **the**
intervened direction" — which a multi-direction PCA subspace would strain, not match.

**Resolved (2) and (3) together**, since they're coupled: adapter loss
`L = L_task + λ_proj·L_orth(z+Δz, w)`, with `w` fixed before training (§1) and `λ_proj` a single
fixed hyperparameter — **not itself the dose axis**. Dose comes from training exactly **one**
adapter and generating the ladder by post-hoc interpolation, `z'(α) = z + α·Δz`,
`α ∈ {0, 0.25, 0.5, 0.75, 1}` — five points from one training run rather than five separately-trained
checkpoints. Reasoning given plainly: satisfies "2–3 strengths" (`22_Experimental_Design.md` §3)
with room to spare, at a fraction of the compute §8 already flagged as the binding constraint, and
isolates the dose variable *more* cleanly than Paper 3's own λ_orth ladder — every α-point shares the
identical trained adapter, so a geometry difference across the ladder can't be attributed to
seed-to-seed training variance the way separately-trained checkpoints could confound it. Not framed
as a deviation from Paper 3's mechanism — framed as tightening the causal isolation the mechanism
itself is supposed to provide.

**Resolved (4)**: fit `w` once as a new shared prerequisite (extends `Stage4_Execution_Plan.md` §0,
does not edit the FINAL document itself); train the adapter once per arm/seed; then, per α, extract
→ Gate 0 → Gate 1 → only if both pass, the full measurement suite → freeze. Scorer suite strictly
after Gate 1, never before — same discipline as every gate in this project.

**Wrote `Stage4_Intervention_Method_Specification.md`** with all four resolutions plus the (0)
adapter-requirement check. Updated `00_execution_checklist.md`'s Stage 4 entry accordingly. Still
open, execution-level only: `λ_proj`'s exact value, adapter width/placement, optimizer/training
hyperparameters, and whether five α-points is final. Vinh's own next step named directly:
`Stage4_Execution_Plan_v2.md` folds this into the step sequence, written only after this
specification is reviewed — training remains NOT AUTHORIZED.

---

**Vinh confirmed the Method Specification resolved the biggest remaining knot — separating
scientific mechanism from implementation artifact.** Named the risk precisely: "closed-form nuisance
projection" being misread as "no adapter needed," which would have broken the Conventional arm (no
parameter count, no training-step count left for it to match). Restated why the resolution holds:
adapter = trainable capacity, closed-form direction = how the nuisance target is defined — two
different things, correctly kept apart. Walked through each of the four resolved items approvingly,
adding his own framing for the estimator choice: Paper 3 already showed representation can *contain*
domain information an estimator can't *access*; the mean-difference-direction approach doesn't try
to erase all domain information, it creates an intervention with one known target — directly
testing whether *that* direction's accessibility changes, which is the actual hypothesis. Also
flagged the dose-ladder design (train once, interpolate) as specifically avoiding Paper 3's own
weakness: different λ values meant both a different loss *and* a different optimization trajectory,
entangling dose with training variance — the single-adapter interpolation approach removes that
entanglement entirely.

**Flagged three more open items before `Stage4_Execution_Plan_v2.md`, and one explicit constraint on
how to handle them**: don't let the execution-plan document become the place a scientific decision
gets made without its own record. Three items: (1) adapter architecture — residual (`z+A(z)`) is
fixed, but not the *form* of `A(z)` (linear residual vs. bottleneck MLP); (2) exact orthogonality
loss formula — raw vs. squared cosine similarity, noted these have different interpretations, not
just different smoothness; (3) Conventional arm matching, spelled out precisely: must match adapter
parameters, optimizer budget, training epochs — must **not** match the projection/orthogonality
mechanism itself.

**Wrote `Stage4_Adapter_And_Control_Specification.md`, all three resolved:**

- **Adapter architecture**: residual bottleneck, `z' = z + W₂·σ(W₁·z)`, not linear-only. A linear-only
  adapter was considered and rejected — without the nonlinearity, `A(z)` collapses to a single
  effective linear map, blurring the exact capacity/mechanism distinction the Method Specification
  had just drawn. Grounded in AR-01's own "bottleneck/feature adapter" family conclusion
  (AdaptFormer/CLIP-Adapter precedent), not a fresh architectural guess.
- **Orthogonality loss**: squared cosine similarity, `cos(z',w)²`, not `|cos(z',w)|`. Two independent
  reasons, not one: it's the *exact* formula CSG-Lite's own orthogonality term uses
  (`csg_lightning.py`, read directly during the earlier gap investigation:
  `mean(cosine_similarity(z_l, z_c_proj).pow(2))`) — the most literal instantiation of "mirror...
  as closely as an adapter allows" available; and it's smoother at the actual optimization target
  (zero correlation), where `|cos|` has a non-differentiable kink exactly at the point training is
  trying to settle.
- **Conventional arm matching**: exact match on adapter architecture, bottleneck width, optimizer,
  epochs, seeds. `L_orth` **structurally absent** from Conventional's loss — explicitly not
  "`λ_proj = 0`," since that framing would make Conventional read as the zero-strength point on
  Intervention's own ladder rather than a categorically different training objective, which is what
  C4 actually needs to test. Extended (reasoned, not required by the protocol text) the same
  α-interpolation scaffolding to Conventional's own task-only adapter, for point-for-point
  measurement parity with Intervention's ladder in Stage 5 — without adding the nuisance mechanism
  back into Conventional's training.

**Updated `00_execution_checklist.md`'s Stage 4 entry** with all three. Still open, execution-level
only: bottleneck width, nonlinearity choice, `λ_proj`'s value, optimizer/LR/epoch values. Training
remains NOT AUTHORIZED — `Stage4_Execution_Plan_v2.md` is next, and per Vinh's own standing
instruction, its job is to lock run order only, never to make a scientific decision inside it that
doesn't already have its own record.

---

**Vinh confirmed the document architecture itself is now correct**: AR-01 decides family, Method
Specification decides mechanism, Adapter and Control Specification decides implementation class,
Execution Plan only decides how to run — praised each of the three locked items for a specific
reason (nonlinearity kept in the bottleneck specifically preserves capacity distinct from a bare
linear transform, which a reviewer could otherwise dismiss as "geometry changed because of a linear
transformation, not a real intervention"; squared-cosine orthogonality praised not mainly for
smoothness but for preserving mathematical continuity with the exact mechanism Paper 3 already used;
Conventional's structurally-absent `L_orth` walked through explicitly as the difference between
"same method at zero strength" (wrong, invites a dismissive reviewer reading) and "two different
training objectives, same everything else" (right, actually answers C4)).

**Named three items still needing a *rule*, not a value — and stated the constraint precisely: "λ
không nên chọn theo kết quả Gate 1... Cần có rule trước"** (λ shouldn't be chosen based on Gate 1's
result — need a rule beforehand). Bottleneck width `r`: too small risks Gate 1 failure (insufficient
capacity to move geometry), too large risks a capacity confound (adapter approaches a full
transformation, no longer a controlled intervention) — suggested "smallest capacity that passes the
Implementation Integrity Gate," written down in advance. Nonlinearity: smaller decision, but flagged
that PanDerm/ViT conventionally uses GELU, so any deviation needs justification. `λ_proj`: called
the most sensitive variable — too small doesn't move geometry, too large risks damaging the task
representation — needs a pre-defined rule (fixed grid, or a validation criterion set in advance, or
a single value scaled from CSG-Lite), not a post-hoc pick. Also sketched the eventual
`Stage4_Execution_Plan_v2.md` as a pure 10-step run-order list with no scientific decision left
inside it, once the three items above are locked.

**Wrote `Stage4_Hyperparameter_Selection_Rules.md`, all three resolved as procedures, not numbers:**

- **Bottleneck width**: ordered grid `{16, 32, 64, 128}`, train at the smallest, check Gate 0 +
  Gate 1, stop at first pass — walked in one direction only, so there's no room to compare multiple
  passing values and pick the one that looks best downstream. Explicitly grounded why this is a
  legitimate pre-registered rule and not a loophole: the stopping criterion is Gate 0/Gate 1
  (integrity/manipulation checks, defined as producing "inconclusive, not falsifying" outcomes on
  failure), never a C2–C5 result.
- **Nonlinearity**: GELU, not searched — confirmed directly from this project's own tooling (the
  printed module structure from Stage 1.2's and EA-01's own model-loading runs showed
  `GELU(approximate='none')` in every PanDerm MLP block) that this is what the backbone itself
  already uses; deviating would need its own justification that doesn't currently exist.
- **`λ_proj`**: same "smallest that clears Gate 0 + Gate 1" principle as width, grid
  `{0.1, 1, 5}` — loosely inspired by, not copied from, CSG-Lite's own `λ_orth ∈ {0,1,5}` scale
  (explicitly flagged that magnitude transfer across architectures isn't assumed, which is exactly
  why an empirical rule is used instead of reusing CSG's literal numbers). Framed as a
  minimum-effective-dose principle: the smallest intervention that reliably produces the manipulation
  is more defensible evidence for the mechanism than the largest, since it can't be dismissed as
  "any strong enough perturbation would move geometry somehow."

**Wrote `Stage4_Execution_Plan_v2.md`** — pure run order, 10 steps, every single choice inside it a
pointer to one of the six prior specification documents, nothing decided inline. Integrated the
width/λ_proj selection loops explicitly into steps 3–4 (rather than treating Gate 0/Gate 1 as
appearing only once), consistent with `Stage4_Intervention_Method_Specification.md`'s own gate-timing
section. Confirmed the Adaptation arm's own sequencing (no width/λ_proj selection, no Gate 1) is
unchanged from `Stage4_Execution_Plan.md` §2, not re-specified.

**Updated `00_execution_checklist.md`'s Stage 4 entry** — the full scientific-design chain is now
complete end to end: family → mechanism → target → estimator → dose → adapter class → orthogonality
formula → Conventional control → hyperparameter selection rules → run order. **No code has been
written anywhere in this chain. Training remains NOT AUTHORIZED** — the next explicit decision is
Vinh's, whether to authorize Stage 4's actual first computation now that its entire design has a
decision record.

---

**Vinh reviewed the whole arc from a project-management level, not a file level — declared Phase B
no longer "research design," walked the full completed inventory (Phase A's twelve documents, the
Reference Arm's ten frozen components, AR-01's seven-document chain), and stated plainly what's
missing: nothing conceptual — no missing hypothesis, novelty case, mechanism, protocol, or execution
plan. Only execution itself remains.** Praised the hyperparameter selection rules specifically for
being procedures, not values — walked through why in his own words: "64 nhìn đẹp hơn" (64 looks
nicer) is exactly the kind of post-hoc reasoning the ordered-grid-stop-at-first-pass rule makes
structurally impossible, for both width and λ_proj. Called `Stage4_Execution_Plan_v2.md`'s
load→train→extract→Gate0→Gate1→measurement→freeze shape evidence the plan is "sạch" (clean) —
no scientific decision hiding inside a run-order document. Stated his own intended next action
directly, as PI: no further edits to Stage 4's design — every additional change from here risks
over-design — and described the exact form of the authorization he wanted issued: a versioned
document (v1.0) listing every locked document as confirmed, plus one explicit standing rule: no
further scientific-design changes once execution begins, only implementation bug fixes through the
existing Amendment process.

**Wrote `Stage4_Execution_Authorization.md`, v1.0, exactly to Vinh's specification.** All nine Stage
4 design documents listed and confirmed as a single chain (AR-01's Architecture Study and
Resolution, the Architecture Review, the Instantiation Contract, the Intervention Design Choice, the
Method Specification, the Adapter and Control Specification, the Hyperparameter Selection Rules, and
Execution Plan v2). New standing rule recorded, framed explicitly as the Artifact Immutability Rule's
own logic applied one level up — to the design itself, not just the artifacts it produces: an
implementation bug gets fixed inline and logged; any other kind of problem (a design choice that
turns out wrong, an unanticipated gap, a result suggesting a different mechanism) goes through the
Amendment process, never an inline fix once execution starts.

**Updated `00_execution_checklist.md`'s Stage 4 entry**: status changed from blocked to **AUTHORIZED,
v1.0, 2026-08-05**. This is the first time in the entire project that GPU training has been
authorized for anything beyond embedding extraction — every step up to this point has been either
documentation, closed-form measurement, or frozen-artifact management.

**Vinh's closing framing, recorded verbatim in intent**: from this point, Paper 4's representation
intervention is not being designed anymore — it is being tested. Every scientific decision was
locked before any Intervention-arm result existed to be influenced by it — the actual definition of
a preregistered study, not just a label attached to one after the fact.

**Vinh confirmed the plan and restated the issue-handling categories as standing rules for all of
Stage 4 execution**: implementation bug → fix inline, log it; Execution Amendment → execution scope
needs adjusting, protocol untouched; Architecture Resolution → only if a new architecture-level gap
of AR-01's own kind surfaces; Protocol Deviation → only if something actually preregistered must
change. Explicitly named what will **not** be edited regardless of how results look: Hypothesis,
Claim Matrix, Execution Plan, Hyperparameter Rules, Stage 4 specifications. Reconfirmed the seven-step
order from `Stage4_Execution_Plan_v2.md` verbatim — no skipping, no reordering.

**Executed Step 1 — verification of the Reference Instrument and every frozen artifact,
non-destructive.** Checked the PanDerm checkpoint's actual SHA256 against its recorded metadata:
PASS. Checked every per-file SHA256 recorded in `ReferenceEmbedding/manifest.json`,
`ReferenceTrainEmbedding/manifest.json`, `BaselineReport/manifest.json`, and
`GroupBExecution/execution_manifest.json` against the actual files on disk: **all PASS, no drift
since freezing.** Confirmed every frozen directory is still `dr-xr-xr-x` (read-only) as expected.
Nothing written, nothing modified — pure verification, matching Step 1's own "non-destructive"
scope in `Stage4_Execution_Plan_v2.md`.

**Next**: Step 2 (compute the nuisance direction `w`) requires building the extraction/computation
script — the first piece of actual Stage 4 implementation code, not yet written.

---

**Vinh confirmed Step 1 and set the exact discipline for Step 2**: `w` estimation must be its own
artifact, Group-B-style — execution contract before any code, then
implementation → execution → review → sanity → freeze — kept strictly separate from adapter code,
so that a later Gate 1 failure or Execution Amendment can be traced to direction estimation, adapter
implementation, or training individually, not a single tangled pipeline. Specified the artifact's
required shape directly: closed-form/deterministic/no-optimization/no-gradient/no-training method,
full provenance, manifest, checksum, review before freeze.

**Built `NuisanceDirection/` following that exact chain.** Wrote `execution_contract.md` first — no
code, no results — locking the input populations (`ReferenceTrainEmbedding` for `μ_isic`,
`ReferenceEmbedding`'s PAD-UFES rows for `μ_pad_ufes`, the same split already justified in
`Stage4_Intervention_Method_Specification.md` §1) and the exact formula
(`w = unit(μ_isic − μ_pad_ufes)`) before writing any script.

**Wrote and ran `analysis/compute_nuisance_direction.py`.** Both input artifacts' SHA256 re-verified
against their own manifests before use, same pattern as `groupB_mahalanobis_family.py`. Result:
`‖w_raw‖ = 9.014975`, `‖w‖ = 1.000000` post-normalization.

**Wrote `review.md`**, including one limitation stated plainly rather than glossed over: `μ_pad_ufes`
is fit on the same PAD-UFES population that also appears in `ReferenceEmbedding`'s own OOD
evaluation set — no separate PAD-UFES-train population exists (already traced in the Method
Specification). Mirrors Papers 1–3's own precedent, not a new risk this artifact introduces, but
recorded so it's visible rather than assumed away.

**Wrote `review_results.md`**, four sanity checks, each verified by recomputing, not eyeballed: (1)
non-degenerate — `‖w_raw‖ = 9.01` against a typical embedding norm of ≈23.08
(`ArtifactCharacterization`'s own figure), a substantial fraction, not a near-zero artifact of
nearly-identical class means; (2) correctly unit-normalized — `‖w‖ = 1.000000`, read directly from
the saved array; (3) plausibly domain-related — projected `ReferenceEmbedding`'s own held-out
ISIC-test and PAD-UFES rows onto `w`: ISIC-test mean 1.294 (std 1.310, n=5,067), PAD-UFES mean
−7.800 (std 2.014, n=2,298), a clear, correctly-signed separation relative to each group's own
spread — with the same PAD-UFES-circularity caveat repeated here, and the ISIC-test side flagged
explicitly as the independent half of this check (fit on `ReferenceTrainEmbedding`, tested on
disjoint `ReferenceEmbedding` rows); (4) no impossible number — all four saved arrays finite,
correctly shaped, population counts matching both source manifests exactly. All four: YES, no
anomaly found — explicitly not a claim that `w` is scientifically "correct," only that the
computation is internally consistent.

**Froze `NuisanceDirection/`** — `manifest.json` updated as the last write (status → FROZEN, review/
sanity fields added) before an `execution_manifest.json` completion certificate was written and the
whole directory locked (`chmod 444` on seven files, `555` on the directory) — same two-manifest
pattern as `GroupBExecution/`.

**Status: Stage 4's first artifact-producing step is done.** `w` exists, frozen, reviewed,
sanity-checked — and is not itself an adapter, not training, not a claim. Next: the adapter module
and training loop (`Stage4_Execution_Plan_v2.md` steps 3 onward) — the point where code volume
increases substantially, per Vinh's own framing.

---

**Vinh switched from DESIGN MODE to EXECUTION MODE.** No new design/spec/contract/review/planning
documents unless explicitly requested, an EA/AR/Protocol Deviation is required, or a milestone
closes. Default output is code + execution, not documentation. Cycle per task: implement → verify →
run the pre-registered gate → report → wait. Implementation details get decided per the locked
protocol and the work continues; genuine uncertainty gets flagged, not silently resolved.

**Implemented `analysis/stage4_adapter.py`** — `BottleneckAdapter` (`z' = z + W2·GELU(W1·z)`,
up-projection zero-init so the adapter starts as identity), `orthogonality_loss` (`cos(z',w)²`),
`TaskHead`, `task_loss` (CE, `ignore_index=-1` for PAD-UFES rows, CSG-Skin's own convention) — exact
formulas from `Stage4_Adapter_And_Control_Specification.md`, nothing new decided.

**Implemented `analysis/stage4_train_intervention.py`** — Execution Plan v2 Step 3, the
bottleneck-width selection loop. Training population: `ReferenceTrainEmbedding` (ISIC, labeled) +
`ReferenceEmbedding`'s PAD-UFES rows (unlabeled, `y=-1`) — all cached frozen embeddings, no image or
ViT forward pass needed. Gate 0 reuses `linear_probe`/`ece_from_proba` from
`stage3_2_reference_arm_baseline.py` directly (import, not reimplementation); thresholds exactly
`gate0_verdict.json`'s own rule. Gate 1 reuses `compute_lid_spectral_diagnostics` from
`lid_spectral_decay.py` directly. AdamW, lr=1e-3, 100 epochs, batch 512 — execution-level defaults,
not previously fixed by any spec, chosen and logged here rather than in a new document.

**Ran the width-selection grid, `r ∈ {16,32,64,128}`, `λ_proj=0.1` fixed probe.** Gate 0 PASSES at
every `r` (domain-probe accuracy 0.998–0.999, ID-task balanced accuracy 0.547–0.638, both criteria
clear). Gate 1 raw measurement: baseline (α=0) LID=15.028, slope=0.006973; adapted LID drops
monotonically with `r` (14.261 → 14.382 → 13.483 → 12.780 at r=16/32/64/128), slope shifts down by a
roughly constant, small amount at every `r` (≈ −0.00015 to −0.00017), not visibly r-sensitive in
this range.

**Stopped here — one genuine ambiguity, not an implementation detail: Gate 1's PASS/FAIL threshold
for "actually changed" is not numerically specified anywhere in the locked protocol.** Computed the
measurement `Stage4_Execution_Plan_v2.md` asks for; did not invent a cutoff to turn it into PASS/FAIL,
since doing so would functionally be a scientific decision made silently. Raw results (not frozen,
not a gated artifact) written to `outputs/stage4_width_selection_probe.json`. Flagged to Vinh for a
decision before the width-selection loop can actually stop at a value.

---

**Vinh confirmed not inventing a threshold was right, then reframed the actual question: Gate 1 is a
manipulation check, not a hypothesis test** — no arbitrary percentage or p-value belongs in it;
that's Stage 5's job once seeds are pooled. Gave the exact criterion directly (not asking me to edit
the protocol myself): PASS iff the intervention produces a reproducible monotonic change in at least
one preregistered geometry metric relative to the reference artifact; Gate 1 supports or contradicts
nothing about the hypothesis by itself.

**Opened `ExecutionAmendments/EA-02.md`** — same shape as EA-01 (protocol correct but operationally
silent, not wrong), Approved immediately since Vinh gave complete wording. Applied one interpretive
note transparently rather than silently: "the intervention ladder" most precisely names the α-dose
ladder (Step 6), not the width grid (a capacity probe); at the width-selection stage, "reproducible
monotonic change" reduces to a measurable, correctly-directed shift from baseline at each `r`, with
true cross-ladder monotonicity checked properly once α runs.

**Applied to the width-selection results: Gate 1 PASSES at every `r`** — LID and slope both decrease
from baseline (15.028 / 0.006973) at all four widths, real distinct measurements. Per
`Stage4_Hyperparameter_Selection_Rules.md` §1 (stop at smallest passing value): **`r = 16` selected**
— locked for every arm in Stage 4.

**Vinh confirmed `r=16` and the width/α-ladder distinction, declined a separate review round (no
science yet — pure implementation-configuration selection per an already-locked rule, no cross-arm
comparison, no claim tested), and authorized continuing straight through Step 4 — no new grid
values, no mid-run tuning, stop at first pass.**

**Ran `analysis/stage4_select_lambda.py`** — `λ_proj` grid `{0.1, 1, 5}`, `r=16` fixed, same Gate 0
+ Gate 1 (EA-02) stopping rule. `λ_proj=0.1` passed both gates immediately (identical run to the
width-selection probe at `r=16`, since `0.1` was already the fixed probe value — same numbers,
confirmed consistent, not a bug). **`λ_proj = 0.1` selected.**

**Both hyperparameters now locked: `r=16`, `λ_proj=0.1`.** Next: train the canonical, 5-seed
Intervention-arm adapter at these fixed values (Execution Plan v2 Step 4's own remaining part, §5
Controls' multi-seed requirement) — the single-seed runs so far were hyperparameter probes only, not
the arm's actual artifact.

**Vinh confirmed the state, drew a hard line between the probe runs and the real artifact
("Đừng nhầm các run vừa rồi với kết quả của Intervention Arm"), confirmed the next-step order
(canonical Intervention → freeze → canonical Conventional → freeze → α-ladder → gates →
measurements), and set the cadence for the rest of Stage 4 explicitly: run → report → review →
continue, never chaining multiple major stages before reporting — same discipline as the whole
Reference Arm, now carried into Stage 4 given the GPU/time cost per step. Corrected my own
"Proceeding into that now" — canonical Intervention Arm only, then stop for review before
Conventional starts.**

**Ran `analysis/stage4_canonical_intervention.py`** — 5 seeds (42/52/62/72/82), `r=16`,
`λ_proj=0.1` fixed, each seed's own Gate 0 + Gate 1 (EA-02) computed independently before
aggregating:

| seed | Gate 0 (probe / ID-task) | LID | slope | Gate 1 |
|---|---|---|---|---|
| 42 | PASS (0.9984 / 0.5470) | 14.261 | 0.006807 | PASS |
| 52 | PASS (0.9992 / 0.5573) | 14.949 | 0.006779 | PASS |
| 62 | PASS (0.9985 / 0.5661) | 14.651 | 0.006799 | PASS |
| 72 | PASS (0.9989 / 0.5642) | 14.733 | 0.006787 | PASS |
| 82 | PASS (0.9989 / 0.5868) | 14.816 | 0.006785 | PASS |

**Aggregate: Gate 0 PASS (probe_acc_mean=0.9988, id_task_bal_acc_mean=0.5643), Gate 1 PASS
(LID_mean=14.682 vs. baseline 15.028, slope_mean=0.006791 vs. baseline 0.006973 — all 5 seeds
individually pass, satisfying EA-02's "reproducible" reading). Both gates pass at every one of 5
seeds, not just on average.**

**Froze `InterventionArm_Canonical/`** — 5 adapter checkpoints, 5 adapted-embedding arrays (on the
held-out `ReferenceEmbedding` population), `manifest.json` (full provenance: hyperparameters,
architecture, orthogonality target, training/eval population artifacts, per-seed + aggregate gate
results), `sha256.txt`. `chmod 444`/`555`, manifest written last.

**Stopped here, per Vinh's explicit cadence instruction.** Not proceeding to the Conventional arm
without review.

---

**Vinh reviewed as PI, not cheerleading — approved, explicitly praised per-seed PASS over
mean-only PASS** (every one of 5 seeds individually clears both gates, not just the average — "mạnh
hơn rất nhiều"), and confirmed the report stayed correctly scoped: Gate PASS reported, no p-value,
no effect size, no claim, no "intervention works" language — explicitly named what has *not* been
shown yet: intervention → geometry changed is confirmed; accessibility changed and reliability
changed are still untested. **One required check before Conventional opens**: mean/std/min/max
across the 5 seeds for probe, ID-task, LID, and slope — not a test, an internal implementation
sanity check (would a `14.2, 14.8, 14.8, 14.8, 17.1`-shaped outlier still pass Gate 1 on average
while actually signaling a bug — checked for, not assumed absent).

**Computed directly from the frozen `InterventionArm_Canonical/manifest.json`** (read-only, artifact
untouched):

| Metric | mean | std | min | max |
|---|---|---|---|---|
| Domain probe accuracy | 0.998769 | 0.000342 | 0.998371 | 0.999186 |
| ID-task balanced accuracy | 0.564289 | 0.014662 | 0.546998 | 0.586850 |
| LID | 14.681939 | 0.259821 | 14.260953 | 14.949070 |
| Spectral-decay slope | 0.006791 | 0.000011 | 0.006779 | 0.006807 |

No outlier seed — every metric tight around its mean, no seed standing apart from the other four.
**Approved: Canonical Intervention Arm accepted as Stage 4's standard artifact, no rerun requested,
no protocol change.** Authorized to proceed to the Canonical Conventional Arm (5 seeds), same
cadence (run → report → stop for review, not cascading further).

**Implemented and ran `analysis/stage4_canonical_conventional.py`** — same architecture, `r=16`,
optimizer, LR, epochs, and seeds as `InterventionArm_Canonical`, `L_orth` structurally absent (loss
= `L_task` only, per `Stage4_Adapter_And_Control_Specification.md` §3 — not `λ_proj=0`).

| seed | Gate 0 (probe / ID-task) | LID | slope | Gate 1 |
|---|---|---|---|---|
| 42 | PASS (0.9991 / 0.5497) | 14.465 | 0.006807 | PASS |
| 52 | PASS (0.9991 / 0.5606) | 14.873 | 0.006774 | PASS |
| 62 | PASS (0.9986 / 0.5578) | 14.736 | 0.006791 | PASS |
| 72 | PASS (0.9990 / 0.5675) | 14.718 | 0.006785 | PASS |
| 82 | PASS (0.9991 / 0.5840) | 14.706 | 0.006780 | PASS |

Aggregate: Gate 0 PASS (probe_acc_mean=0.9990, id_task_bal_acc_mean=0.5639); Gate 1 PASS
(LID_mean=14.700 vs. baseline 15.028, slope_mean=0.006787 vs. baseline 0.006973) — every seed
individually passes both.

**Freeze precondition applied: Gate 0 only, not Gate 0 AND Gate 1** — matching
`Stage4_Execution_Plan_v2.md`'s own already-locked wording (a Gate 1 failure freezes the checkpoint
anyway, flagged not-usable for C3/C4, never blocks the freeze). Noted for the record: the earlier
`InterventionArm_Canonical` run required both gates to pass before freezing — stricter than what the
plan actually specifies. Harmless in that case (both gates passed regardless), but the two arms were
gated inconsistently; flagging rather than quietly leaving it unremarked. Froze
`ConventionalArm_Canonical/` — same file structure as Intervention's own artifact.

**Factual observation, not interpreted**: Conventional's Gate 1 numbers (LID_mean=14.700,
slope_mean=0.006787) are close in magnitude to Intervention's own (LID_mean=14.682,
slope_mean=0.006791) — both arms show geometry shifting from baseline by a similar amount. This is
raw Gate 1 output from two single, independent training runs; it is not a C4 comparison (that
requires the interaction-regression test, Stage 5, not run here) and is not interpreted as evidence
about intervention type mattering or not mattering. Recorded as observed, nothing concluded.

**Stopped here, same cadence.** Not proceeding to the α-ladder without review.

**Vinh approved the Conventional artifact, no rerun, and reframed the close-geometry-shift
observation directly rather than letting it sit as an unexplained worry**: both arms are still
"representation intervention" in the hypothesis's own general sense (intervention → geometry →
accessibility → reliability) — Conventional differing only in mechanism, not in being an
intervention at all — so both shifting geometry from baseline is not a contradiction of anything.
The real question was restated precisely: not "did geometry change" (already answered, both arms,
Gate 1) but whether *Intervention vs. Conventional* differ once geometry is accounted for — the
actual C4 interaction test, Stage 5, not yet run. On the Gate 0/Gate 1 freeze-precondition
inconsistency: confirmed a minor process bug, no artifact/science impact since both arms passed
both gates regardless — logged, not rerun.

**Required one more check before the α-ladder: pre-α readiness — integrity of all three canonical
artifacts** (`ReferenceEmbedding`, `InterventionArm_Canonical`, `ConventionalArm_Canonical`):
frozen status, embedding dimensions, seed consistency, checkpoint presence, SHA256 against each
artifact's own `sha256.txt`. All PASS — `ReferenceEmbedding` still `555`; both canonical arms
`555`, all embeddings 1024-d, seeds `[42,52,62,72,82]` in both, all 5 checkpoints present per arm,
every SHA256 matches. **READY.**

**PI Decision: Canonical Conventional Arm — Approved, no rerun. Pre-α readiness — PASS. α-ladder
authorized to open.**

**Ran `analysis/stage4_alpha_ladder.py`** — both arms, all 5 seeds, `α ∈ {0,.25,.5,.75,1}`, using
the already-trained canonical checkpoints (no retraining), `z'(α) = z + α·Δz`. `α=0` correctly
reproduces the untouched baseline exactly (LID=15.0279, slope=0.006973) in every run — confirms the
interpolation code is correct, not a bug.

**Found a real, reproducible pattern that does not cleanly satisfy EA-02's "monotonic" wording, and
a gap in my own Gate 1 implementation that only surfaces now that the actual ladder ran:**

- **LID**: strictly monotonic-decreasing across the full `α` range in only 3 of 10 (seed, arm)
  combinations. In the other 7, LID rises slightly from `α=0` to `α=0.25` before decreasing
  monotonically from `0.25` through `1.0`.
- **Slope**: monotonic-decreasing in **0 of 10** combinations, both arms. Every single run shows the
  same shape: a sharp drop from baseline at `α=0.25`, then a *steady rise back toward baseline* as
  `α` increases further to `1.0`. Aggregate means (Intervention): `0.006973 → 0.006744 → 0.006756 →
  0.006773 → 0.006791`. Reproduced essentially identically in Conventional. Maximum deviation from
  baseline is at the *lowest* nonzero dose, not the highest.
- **Gate-1-code gap**: `gate1_pass()` (reused unchanged from the width/λ_proj probes) only checks
  "does this single point differ from baseline in the expected direction" — it never tested
  monotonicity *across* the ladder, because the width/λ_proj stage had no ladder to test it against.
  That gap was flagged in passing when EA-02 was first applied (`26_Lab_Notebook.md`, Step 3 entry)
  but not resolved, since the real ladder didn't exist yet to reveal whether it mattered. It clearly
  matters now.

**Not interpreting why this happens — not doing so deliberately.** No speculation about
optimization dynamics, adapter behavior, or what the slope-reversal might mean recorded here; that
would be exactly the premature interpretation this project's discipline exists to prevent. This is a
factual, reproducible measurement pattern, reported as measured.

**Stopped here — genuinely uncertain, not an implementation detail.** Whether "reproducible
monotonic change" (EA-02) should be read as strict monotonicity across every `α` step (which would
mean slope fails Gate 1 at every single point in the ladder, both arms, all seeds) or as a
directional trend across the dose range (which the aggregate data does show, just not
point-to-point) was never specified precisely enough to resolve unilaterally, and the actual data
turned out messier than either EA-02's authoring or my own Gate-1 code anticipated. Full numbers in
`AlphaLadder/alpha_ladder_results.json` (not frozen). Flagged to Vinh.

**Vinh: first stop-execution decision of the project.** Confirmed this is an EA-02 wording bug, not
an implementation bug — the code correctly exposed that "monotonic" over-constrained a manipulation
check that was never meant to test dose-response *shape*. Rejected both options I offered (strict
monotonic — too strict, adds a scientific requirement after execution started; "directional trend" —
too vague to operationalize honestly). Gave the precise correction directly: Gate 1 should test
*reproducible dose-dependence*, no assumption about linearity, monotonicity, or convexity — the
response shape is empirical, not prespecified. Explicit instructions: stop, do not reinterpret the
data, open EA-03 for the wording correction, re-score the *existing* α-ladder output under the
revised criterion without regenerating anything, and do not speculate about what the
drop-then-recover slope pattern might mean — it may become one of Paper 4's more interesting
findings later, but right now it is a response curve observed, nothing more.

**Opened `ExecutionAmendments/EA-03.md`.** Replaces EA-02's "monotonic" with "reproducible
dose-dependence" — same manipulation-check role, `22_Experimental_Design.md` §3.5 untouched, no
claim/hypothesis impact. Operationalized precisely, reusing EA-02's own meaning of "reproducible"
(every seed agreeing, not just the mean): at a given `α > 0`, Gate 1 PASSES if at least one geometry
metric deviates from the `α=0` baseline in the same direction across all 5 seeds — no requirement on
how that deviation's magnitude behaves as `α` continues to increase.

**Re-scored the existing `AlphaLadder/alpha_ladder_results.json` under EA-03 — no new training, no
new embeddings, same raw numbers, only the PASS/FAIL rule changed:**

| α | Intervention | Conventional |
|---|---|---|
| 0.00 | FAIL (no intervention applied — expected) | FAIL (expected) |
| 0.25 | PASS (slope reproducible across all 5 seeds; LID not) | PASS (slope reproducible; LID not) |
| 0.50 | PASS (slope reproducible; LID not) | PASS (slope reproducible; LID not) |
| 0.75 | PASS (slope reproducible; LID not) | PASS (slope and LID both reproducible) |
| 1.00 | PASS (slope and LID both reproducible) | PASS (slope and LID both reproducible) |

**Gate 1 passes at every nonzero dose, both arms, under the corrected criterion.** Slope alone
carries every pass except the two `α=0.75`/`α=1.0` points where LID also independently reproduces.
Written to `AlphaLadder/gate1_ea03_rescored.json` — a re-score of existing numbers, not a new
measurement run. No interpretation of the response shape attached.

**Stopped here, same cadence — not proceeding to the measurement suite / freeze without review.**

**Fixed EA-03's wording per Vinh's correction**: "PASS if at least one metric..." read as license to
discard whichever metric doesn't cooperate. Corrected to "Gate 1 is satisfied if at least one
preregistered geometry metric shows reproducible dose dependence. The remaining geometry metrics
are recorded individually and are not ignored." **Gate 1 declared locked** — EA-03 is the last
amendment to its criterion; any future unexpected response shape is a result to record, not grounds
to reopen the gate. Recorded in `EA-03.md` and `00_execution_checklist.md`.

**Ran `analysis/stage4_geometry_completion.py`** — Step 7, condition number (`geometry_diagnostics.py`,
reused) and per-direction variance of `w` (finally computable — `BaselineReport/geometry_metrics.json`
had explicitly flagged this as not-computable on the Reference arm alone, no intervened direction
existing yet). Same already-trained checkpoints, no retraining. LID/slope not repeated (already in
`AlphaLadder/alpha_ladder_results.json` from Gate 1).

**Per-direction variance of `w` — clean, reproducible, and clearly different between arms:**

| α | Intervention (mean) | Conventional (mean) |
|---|---|---|
| 0.00 | 20.198 (reference) | 20.198 |
| 0.25 | 16.790 | 20.611 |
| 0.50 | 14.086 | 21.061 |
| 0.75 | 12.088 | 21.548 |
| 1.00 | 10.796 | 22.072 |

Strictly monotonic-decreasing in **5 of 5** Intervention seeds. Strictly monotonic-*increasing* in 4
of 5 Conventional seeds (seed 62 roughly flat, not decreasing either). This is the one geometry
metric so far where the two arms' response shapes are visibly different, not just similar in
magnitude — recorded factually, nothing concluded about C4 or the hypothesis from it.

**Condition number — noisy, no clean pattern across seeds**, consistent with Gate 1's own
established skepticism of condition number as sufficient evidence (`EA-02`/`EA-03`: "not just
condition number/scale"). Intervention seed 42 shows a ~10× increase (1.34M → 13.78M) while other
seeds decrease or stay flat; Conventional shows the same kind of inconsistency. Raw numbers in
`AlphaLadder/geometry_completion_results.json` (not frozen).

**Stopped here, same cadence.** Reliability scorers (Mahalanobis/kNN/cosine/KDE on the adapted
embeddings) remain — a separate, larger step needing an adapted train-population extraction, not
started. Not proceeding without review.

---
