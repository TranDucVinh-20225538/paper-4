# Phase 3 — locked protocol for same-modality external evaluation

**Status: LOCKED.** Committed before any external-domain embedding was extracted
or any external score computed. The commit that adds this file is the timestamp;
everything downstream is evaluated against what is written here, and any
departure is reported as a departure rather than folded in silently.

## 1. Why this phase exists

PAD-UFES-20 confounds three things at once. It is a different clinical site, a
different modality (smartphone clinical photography against dermoscopy), and it
participated in operator calibration during development. It is therefore not a
locked unseen OOD domain, and it yields near-ceiling AUROC for every scorer, so
"the scorer did not respond" and "the design could not have seen a response"
produce the same flat curve.

The feasibility analysis in `phase3_headroom_feasibility.py` establishes what
headroom would buy, and the answer is not uniform: cosine, k-NN and KDE already
carry 15-21x dynamic range at the present operating point and would gain 2-3x
more, while Mahalanobis sits at S/MDE = 0.1 and does not reach 1 under either
model of a harder domain. **Mahalanobis's flat response is therefore predicted
to remain flat here.** That prediction is recorded now so that confirming it
later is not mistaken for a finding.

## 2. Datasets

| | Derm7pt | PH2 |
|---|---|---|
| source | Interactive Atlas of Dermoscopy (Argenziano), SFU release_v0 | Hospital Pedro Hispano, Matosinhos |
| cases | 1011, each with a paired dermoscopic and clinical image | 200 dermoscopic |
| modality | both | dermoscopic |
| in ISIC 2019? | no — ISIC 2019 is HAM10000 + BCN20000 + MSK | no |

Derm7pt's paired images are the reason it is the primary external set. The pair
holds the lesion population **exactly** fixed and varies only modality, which
separates the modality effect from the site effect. PAD-UFES-20 cannot do this.

## 3. Dataset pairs — reported separately, never pooled

The ID population is unchanged and frozen: the 5,067 held-out ISIC 2019 images
already in the reference artifact.

| pair | OOD side | role |
|---|---|---|
| P1 | Derm7pt dermoscopic | **primary** — same modality, external site |
| P2 | Derm7pt clinical | modality control on the identical lesions |
| P3 | PH2 dermoscopic | second same-modality site, small |
| P0 | PAD-UFES-20 | the existing comparison, unchanged |

P2 minus P1 is a within-lesion modality contrast. P1 minus P0 mixes site and
modality and is reported as such.

## 4. Label mapping — frozen

Target taxonomy is the pipeline's own: `MEL NV BCC AK BKL DF VASC SCC` = 0..7.

**Derm7pt** (1011 → 983 included, 28 excluded):

| source diagnosis | n | → |
|---|---|---|
| melanoma, melanoma (in situ), (<0.76mm), (0.76-1.5mm), (>1.5mm) | 248 | MEL |
| clark / reed or spitz / dermal / blue / congenital / combined / recurrent nevus | 575 | NV |
| basal cell carcinoma | 42 | BCC |
| seborrheic keratosis, lentigo | 69 | BKL |
| dermatofibroma | 20 | DF |
| vascular lesion | 29 | VASC |
| melanoma metastasis | 4 | *excluded* — a metastatic deposit, not a primary lesion |
| melanosis | 16 | *excluded* — ISIC's BKL is seborrheic keratosis / solar lentigo / lichen-planus-like keratosis; melanosis is not one of them |
| miscellaneous | 8 | *excluded* — no determinate class |

**PH2** (200, all included): clinical diagnosis 0 = common nevus (80) and
1 = atypical nevus (80) → NV; 2 = melanoma (40) → MEL.

Neither external set contains AK or SCC. This is recorded, not corrected: the
scorers are unsupervised with respect to these labels and the label mapping
exists for reporting composition, not for scoring.

## 5. Overlap screen

The advisor requires a patient/image overlap check. The ISIC 2019 source images
are no longer on disk, so a pixel or perceptual hash against them is not
possible. Stating that plainly matters more than substituting something and
calling it equivalent.

What is possible is a **near-duplicate screen in embedding space**, against both
frozen ISIC populations — the 16,211 training embeddings and the 5,067 held-out
embeddings. Under a fixed encoder, a duplicate or near-duplicate image produces a
near-identical embedding.

Pre-specified:

- similarity = cosine between L2-normalised PanDerm embeddings
- for each external image, retain its maximum similarity to any ISIC image
- the threshold is **calibrated, not assumed**: compute the within-ISIC
  nearest-neighbour similarity distribution among the 5,067 held-out images and
  take its 99.9th percentile as the flag level
- any external image at or above that level is **excluded** and listed
- if more than 2% of an external set is flagged, that set is reported as
  potentially contaminated and its results are demoted to a sensitivity analysis

This is weaker than a pixel hash and is labelled as such wherever it is
reported. It cannot prove identity; it can only fail to find evidence of it.

## 6. What is frozen and may not be touched

Nothing about the intervention or the scorers may be adjusted on these domains.
Specifically frozen, carried over unchanged:

- Mahalanobis ridge `1e-5`; k-NN `k=10`; KDE Gaussian, Scott's rule, floor `1e-6`
- the fitting population: the **16,211 adapted training embeddings**, not the
  evaluation set (fitting on the evaluation set yields AUROC 0.99997, flat in
  alpha, and fails silently)
- the alpha ladder, the gamma ladder, and the 5 seeds 42/52/62/72/82
- PanDerm ViT-L/16, checkpoint-499, preprocessing per
  `assets/reference_preprocessing.json` (Resize 256 bilinear, CenterCrop 224,
  ImageNet normalisation with the repo's verbatim std 0.228)

Gamma is **not** re-selected on the external domains.

## 7. Analyses — identical to the primary protocol

Run per pair, in this order, with raw per-image scores retained:

1. four scorers on the frozen configurations, raw scores written to disk
2. RQ1 susceptibility `S(delta)` at the observed distortion, reported at median
   and max delta, with the max-delta row labelled as the only valid bound
3. RQ3 geometry matching: G = (LID, spectral slope), standardisation on the
   conventional arm, caliper 0.25 primary with 0.15 and 0.35 as sensitivity,
   matching blind to AUROC, each observation used once, uncertainty clustered on
   seed, negative control by permuting identity within matched sets
4. hierarchical bootstrap CIs, seeds then examples

## 8. Pre-registered predictions

Recorded so that a confirmation is not later reported as a discovery.

1. Mahalanobis stays flat on every pair. S/MDE < 1 throughout.
2. P1 (same modality) yields lower AUROC than P0 (cross modality) — that is the
   headroom this phase is for. If P1 is also near-ceiling, the phase has failed
   to deliver headroom and that is reported as a null result, not rescued by
   switching to another dataset.
3. P2 minus P1 is positive: the clinical arm of the same lesions is easier to
   separate from dermoscopic ISIC than the dermoscopic arm is.

## 9. Stopping rule

If the overlap screen demotes both Derm7pt and PH2, Phase 3 reports the screen
and stops. No third dataset is sought, because searching for a dataset that
passes is selection on the outcome.
