# Claim Matrix — Paper 4

Status: updated after Tier-1 reads (2026-08-04). Card format (one entry per paper) used instead of a flat table because the
required field set is wide (20 fields) and a table would truncate unreadably. Fields left blank
where not yet determinable from the available full text or article record.

---

## Claim Hierarchy (added 2026-08-04, session 6 — post-Gallo, mechanism-focused)

Per-paper cards above answer "what does each paper claim?" This table re-cuts the same evidence
by **what is now known vs. still open**, so the experimental design in `22_Experimental_Design.md`
can be built directly against it. Every row below traces to a specific paper/evidence-register
entry; nothing here is asserted without a Level A (or, where marked, qualitative-only) source.

| Claim | Status | Source (Level) |
|---|---|---|
| Foundation-model pretraining (CLIP especially) improves ID accuracy over random/ImageNet init in dermatology | Known | Gallo et al., §3.3, Tables 3–4 (Level A) |
| Increasing fine-tuning intensity improves ID accuracy but hurts near-OOD separability | Known | Gallo et al., §3.4, Fig. 2 (Level A) |
| Mahalanobis/MDS beats logit-space post-hoc scorers (MSP, TS, EBO) for dermatology OOD detection, including on FM backbones | Known | Gallo et al., §3.2, Table 2 (Level A); Paper 2 (CNN case, our own) |
| A supervised linear probe can recover acquisition-domain identity from a CNN dermatology representation at near-ceiling accuracy | Known | Paper 1 (our own, Level A — our data) |
| Domain-adversarial disentanglement (GRL+orthogonality) reduces domain-probe leakage while *not* improving — in fact flattening — Mahalanobis-AUROC, on a CNN | Known | Paper 3 (our own, Level A — our data) |
| Representation geometry (condition number) shifts substantially under disentanglement, on a CNN | Known | Paper 3 (our own, Level A — our data) |
| Geometry changes qualitatively under ordinary fine-tuning intensity too (not just disentanglement) | Suggested (qualitative only) | Gallo et al., Fig. 4 t-SNE (Level A for the figure, but the paper itself only interprets it qualitatively — no quantitative geometry metric computed) |
| Which specific geometric quantities (LID, spectral-decay slope, condition number, variance allocation) explain the trade-off Gallo observes, or the flat-Mahalanobis result Paper 3 observes | **Unknown** | No source — Mechanism Map (`08_Mechanism_Map.md`) Mechanism 1 (arXiv:2510.15202) proposes LID×spectral-decay as the general-vision answer, untested here |
| Whether distance-estimator degradation tracks geometry specifically (not just intervention-strength label or probe accuracy) | **Unknown** | Prediction 1, untested — this is the core question Step 6 must answer |
| Whether holding geometry constant across differing intervention strengths holds reliability-estimator degradation constant (causal, not correlational) | **Unknown** | Prediction 2, untested |
| Whether an ordinary-fine-tuning intervention (Gallo-style) and a disentanglement intervention (GRL-style) produce the *same* geometry→accessibility relationship, or different ones | **Unknown** | Not tested by any source — this is the control-arm question the Gallo verification specifically surfaced |
| Whether domain information stays linearly decodable on an FM representation after a disentanglement-style intervention, the way it does on the CNN | **Unknown** | Paper 3's central finding, never tested on an FM by anyone found in this project's literature search |

**Reading the table:** the top six rows are the load-bearing "known" facts Paper 4 is allowed to
cite without re-proving. The bottom five rows are the actual content of Paper 4 — every one of
them is a mechanism question, not a "does X exist" question, which is the intended effect of the
Gallo verification: it converted the project from novelty-defense mode into mechanism-testing
mode. `22_Experimental_Design.md` maps one experimental component to each of these five unknown
rows.

---

## OUR PROJECT (Papers 1–3)

### Paper 1 — "Beyond Accuracy: Quantifying and Reducing Domain Shortcut Reliance in Multi-Source Skin Lesion Classification"
- **Venue / Year:** (submission in progress), 2026
- **Research Question:** Does high in-distribution accuracy mean a classifier has learned to ignore acquisition-source shortcuts, and can adversarial disentanglement reduce domain-shortcut reliance without an accuracy cost?
- **Scientific Claim:** A ResNet-50 baseline heavily encodes acquisition source in its lesion representation (probe accuracy 0.979). Adversarial disentanglement (GRL) reduces this domain leakage by ~27pp beyond backbone-capacity effects alone, while *improving* balanced accuracy and calibration (no utility-leakage trade-off). Mahalanobis-based OOD detection degrades substantially under disentanglement (0.868→~0.40 AUROC) — a secondary, hypothesis-generating finding.
- **Evidence:** 5-seed multi-seed linear probing on frozen z_l, Welch's t-test p=4.1e-7, Cohen's d=35.3, matched-backbone control isolating capacity vs. objective effects.
- **Model Family:** ResNet-50 (baseline), EfficientNet-B3 dual-encoder (CSG-Lite).
- **Pretraining Paradigm:** Fully supervised, ImageNet-initialized, task fine-tuned.
- **Clinical Modality:** Dermoscopy (ISIC 2019) + smartphone macro photography (PAD-UFES-20).
- **Dataset:** 27,629 images (25,331 ISIC 2019 + 2,298 PAD-UFES-20).
- **Representation:** 16-d disentangled lesion code z_l + 64-d context code z_c.
- **Shortcut:** Domain/acquisition-source shortcut, quantified via linear probe.
- **OOD:** Mahalanobis distance, secondary finding only.
- **Calibration:** Measured, improves under disentanglement.
- **Reliability Estimation:** Not the focus; touched via OOD side-finding.
- **Geometry:** Not analyzed directly (that's Paper 3).
- **Limitations:** Supervised CNN/EfficientNet only; two-domain dermoscopy-vs-smartphone testbed; OOD finding not mechanistically resolved.
- **Relation to Paper 4:** Establishes the CNN-paradigm baseline claim ("domain shortcuts are real, disentanglement removes them, but breaks OOD detection") that Paper 4 would test for generalization to FMs.
- **Novelty overlap w/ found lit:** Shortkit-ML and arXiv:2602.18502 both frame disentanglement/multi-method shortcut auditing as benchmarks — need deep read to check backbone coverage.
- **Reading priority:** n/a (our own paper).

### Paper 2 — "DST-Skin: Reliability-Aware Evaluation of Skin Lesion Classifiers under Acquisition Shift"
- **Venue / Year:** (submission in progress), 2026
- **Research Question:** Which post-hoc reliability scores (logit-space vs. feature-space) remain informative under a realistic dermoscopy→smartphone acquisition shift, and does OOD-separability predict safe triage behavior?
- **Scientific Claim:** Logit-space confidence scores (MSP, energy, logit-norm) collapse under acquisition shift (AUROC as low as 0.62); feature-space scores on the same penultimate embeddings remain far more separable (AUROC up to 0.97) — explained mechanistically via "logit dispersion." But OOD-AUROC superiority does not guarantee the lowest triage risk under selective prediction, and in-distribution calibration does not transfer to shifted inputs.
- **Evidence:** 3 CNN backbones (ResNet-18/50, EfficientNet-B3) × 7 reliability scores, risk-coverage curves, ECE under shift.
- **Model Family:** ResNet-18, ResNet-50, EfficientNet-B3.
- **Pretraining Paradigm:** Fully supervised, ImageNet-initialized.
- **Clinical Modality:** Dermoscopy (ISIC 2018) → smartphone (PAD-UFES-20).
- **Dataset:** ISIC 2018 train, PAD-UFES-20 as external shifted eval.
- **Representation:** Standard penultimate-layer CNN features.
- **Shortcut:** Not directly probed (that's Paper 1/3's angle); acquisition shift treated as covariate shift.
- **OOD:** Core focus — 7 scorers benchmarked (MSP, energy, logit-norm, Mahalanobis, kNN, ViM, ReAct).
- **Calibration:** Core focus — ECE in-distribution vs. under shift.
- **Reliability Estimation:** Core focus — selective prediction / risk-coverage.
- **Geometry:** "Logit dispersion" mechanism proposed but not a full geometric analysis (that's Paper 3).
- **Limitations:** Binary benign/malignant only; one acquisition-shift pair; CNNs only; research-prototype, not clinically validated.
- **Relation to Paper 4:** Establishes the CNN-paradigm reliability-scoring hierarchy (feature-space > logit-space) that Gallo et al. (below) now confirms at Level A on FM backbones.
- **Novelty overlap w/ found lit:** **High, Level-A-confirmed overlap with Gallo et al. (CVPR 2026W)** — same PAD-UFES-20 shift target and feature-space-over-logit reliability finding, confirmed demonstrated for CLIP/ViT/Swin adaptation regimes (full text read 2026-08-04). See `03_Novelty_Threats.md`.
- **Reading priority:** n/a (our own paper).

### Paper 3 — "Decodable but Not Accessible: Auditing Distance-Based Reliability Estimation on Disentangled Skin Lesion Representations"
- **Venue / Year:** (submission in progress), 2026
- **Research Question:** Does representation geometry reshaped by domain-adversarial disentanglement training explain (or predict) distance-based reliability-estimator performance?
- **Scientific Claim:** Disentanglement strength (λ_orth = 0, 1, 5) reshapes representation geometry substantially (condition number shifts two orders of magnitude, Kendall's τ=0.84). This geometric change is *not* accompanied by any improvement in Mahalanobis-AUROC (flat, ~0.40, below chance at every level) — nor in 7 other scorers spanning distinct principles (cosine-to-centroid, kNN, energy, ViM, KDE), 6 of which converge on the same null result. Yet a supervised probe recovers domain membership at 0.72–0.81 AUROC from the *same* embeddings — so the information is present but "inaccessible" to non-probing reliability estimators.
- **Evidence:** Dose-response ladder across 3 disentanglement strengths, 8 scorers, 5 geometry metrics, Kendall's τ correlation, exact p-values.
- **Model Family:** EfficientNet-B3 dual-encoder (CSG-Lite family), 16-d representation.
- **Pretraining Paradigm:** Fully supervised + adversarial (GRL) fine-tuning.
- **Clinical Modality:** Dermoscopy (ISIC) vs. smartphone (PAD-UFES).
- **Dataset:** Same ISIC/PAD-UFES-20 testbed as Papers 1–2.
- **Representation:** 16-d z_l across 3 disentanglement checkpoints.
- **Shortcut:** Domain shortcut is the object being disentangled away; recoverable via probe regardless.
- **OOD:** Core focus, 8 distinct scorers.
- **Calibration:** Not the focus.
- **Reliability Estimation:** Core focus — this *is* the paper's contribution: geometry ≠ reliability-estimator usability.
- **Geometry:** Core focus — condition number, 5 metrics total.
- **Limitations:** Single architecture family, single disentanglement mechanism (GRL + orthogonality), CNN/EfficientNet only.
- **Relation to Paper 4:** This is the paper whose central finding (geometry-reshaping ≠ reliability-estimator improvement) is most specifically untested on FM embeddings. No literature found yet that runs a disentanglement dose-response on a *frozen or fine-tuned medical FM* and checks Mahalanobis/probe dissociation the way Paper 3 does.
- **Novelty overlap w/ found lit:** "Dissecting Mahalanobis" (arXiv:2510.15202) and "One Model, Many Behaviors" (arXiv:2601.10836) are the closest general-domain mechanistic analyses of training-induced effects on Mahalanobis/OOD performance — worth a deep read to check if they already generalize this specific claim outside medical imaging.
- **Reading priority:** n/a (our own paper).

---

## LITERATURE (Tier-1/2 candidates found this session)

### Gallo et al., "Pre-Training and Fine-Tuning Effects on Out-of-Distribution Detection in Dermatology with Vision Foundation Models"
- **✅ EVIDENCE LEVEL: A — full text read (2026-08-04, session 5), user-provided CVF PDF, all 10
  pages including references.** Supersedes the prior B/C card entirely. See
  `07_Evidence_Register.md` for the per-claim ✓/△/✗ verification table with quotes, and Threat 1
  in `03_Novelty_Threats.md` for the updated novelty assessment.
- **Venue / Year:** CVPR 2026 Workshops (Med-Reasoner), pages 6932–6941.
- **Research Question:** How do backbone architecture, pre-training strategy, and fine-tuning intensity affect post-hoc OOD-detector performance in dermatology, and can a cascaded two-stage design recover a better ID/OOD operating point?
- **Scientific Claim:** MDS (Mahalanobis Distance Scoring) substantially outperforms all other post-hoc detectors tested (confirmed, §3.2 — quoted). CLIP pretraining produces the most separable near-OOD embeddings among the pretraining strategies tested. Increasing fine-tuning intensity trades ID accuracy for near-OOD separability (a consistent, monotonic trend across 6 adaptation regimes). A cascaded OOD-filter + ID-specialized-classifier design recovers a better joint operating point (98.95% OOD detection, 96.41% ID accuracy) than either component alone.
- **Evidence:** Derm7pt-clinic + PAD-UFES-20 (ID/near-OOD) + HAM10000/Fitzpatrick17k/Bug-Bites-Craqq (near-OOD) + 6 far-OOD natural-image sets; 4 backbone architectures × 5 pretraining strategies (Tables 3–4); 6 fine-tuning regimes (R1–R6, linear probe/partial/full FT × cosine/step-decay); post-hoc detectors MSP, TS, EBO, MDS, RMDS, kNN.
- **Model Family:** Swin-B, Swin-L, ViT-B/16, ViT-L/14 — confirmed vision foundation models and ImageNet-scale transformers, no CNN comparator.
- **Pretraining Paradigm:** Random init, ImageNet21k, IN21k+Adam(+Aug), CLIP — varied deliberately (§3.3) to isolate pretraining effects, separate from the fine-tuning-intensity experiment (§3.4) which fixes backbone/pretraining and varies only adaptation regime.
- **Clinical Modality:** Clinical photography (PAD-UFES-20, Derm7pt-clinic) + dermoscopy (HAM10000) — same modality family as our project, but combined differently: Gallo treats clinical-vs-dermoscopic as the near-OOD split itself, not as a within-ID acquisition shortcut.
- **Dataset:** PAD-UFES-20 (shared with us) + Derm7pt-clinic + HAM10000 + Fitzpatrick17k + Bug-Bites-Craqq + BCN20000/DERM12345/MCR-SL/MRA-MIDAS (Table 1). No ISIC.
- **Representation:** ViT/Swin feature embeddings; MDS models class-conditional Gaussian structure. Geometry evaluated only qualitatively via PCA(50)+t-SNE (Fig. 4) — no condition number, LID, or spectral-decay metric anywhere in the paper.
- **Shortcut:** **Confirmed absent (Level A).** No domain/acquisition-source linear probe, no adversarial classifier, no GRL, no orthogonality regularization, no disentanglement or debiasing objective anywhere in the text — confirmed by exhaustive term search across the full read paper.
- **OOD:** Core focus — 6 post-hoc detectors × 3 shift scenarios × fixed threshold deployment evaluation.
- **Calibration:** Not evaluated (no ECE, no calibration curves anywhere in the paper).
- **Reliability Estimation:** Core focus, in the fine-tuning-intensity/deployment-threshold framing — confirmed Level A.
- **Geometry:** Qualitative only (t-SNE); no quantitative intervention-strength geometry analysis. Confirmed Level A absence.
- **Limitations:** Authors' own stated limitations (§4): "limited set of dermatology datasets," reliance on post-hoc detectors only, "do not include extensive multi-seed analysis or longitudinal drift scenarios," two-stage design evaluated as a deployment strategy rather than a new detection method.
- **Relation to Paper 4:** **Confirmed (not just risked) to close the naive Paper-2-style replication** — feature-space beats logit-space is now demonstrated at Level A on FM backbones (CLIP, ViT, Swin) in dermatology. **Confirmed (not just risked) to leave the Paper-1/3-style mechanism completely open** — no shortcut probe, no disentanglement intervention, no quantitative geometry, no probe-decodability test anywhere in the paper. Their fine-tuning-intensity trade-off finding is a real, Level-A-confirmed competing explanation that Paper 4's experimental design must control for via a matched ordinary-fine-tuning arm (already planned in `16_Minimal_Sufficient_Experiment.md`).
- **Novelty overlap:** High and now confirmed with generic FM reliability scoring / fine-tuning trade-offs (Paper-2-style claim). Low and now confirmed (not just inferred) with the locked RQ's specific mechanism (Paper-1/3-style intervention × geometry × probe-decodability question).
- **Reading priority:** Read in full — threat resolved at Level A.

### "Self-Supervision on Images and Text Reduces Reliance on Visual Shortcut Features" (arXiv:2206.07155)
- **Venue / Year:** arXiv, 2022 (earlier than our other finds; foundational)
- **Research Question:** Does self-supervised (image+text) pretraining reduce shortcut reliance compared to supervised pretraining?
- **Scientific Claim:** Yes, in general-domain vision benchmarks — SSL/CLIP-style pretraining measurably reduces shortcut reliance relative to supervised baselines.
- **Evidence:** General vision benchmarks with known spurious correlations (not medical).
- **Model Family:** CLIP-style SSL vs. supervised CNN/ViT.
- **Pretraining Paradigm:** Contrastive image-text SSL vs. supervised.
- **Clinical Modality:** None — general vision.
- **Dataset:** Natural-image shortcut benchmarks.
- **Shortcut:** Core focus.
- **OOD / Calibration / Reliability Estimation / Geometry:** Not addressed.
- **Limitations:** Not medical; not dermatology; not acquisition-source shortcuts specifically; predates modern medical FMs (PanDerm, MONET, etc.).
- **Relation to Paper 4:** This is the strongest *general-domain* evidence for the "generalizes" side of our hypothesis (FM-style pretraining reduces shortcuts). It sets up the open question precisely: does this general-vision result hold for *domain/acquisition* shortcuts specifically, in a *medical* FM, using the same rigorous probing methodology as Paper 1?
- **Novelty overlap:** Low direct overlap (different domain, different shortcut type, different method), but high conceptual relevance — should anchor our Introduction's "prior evidence on one side" argument.
- **Reading priority:** Tier 2 (already have the claim from abstract; read in full if used as a key citation).

### arXiv:2602.18502 — "Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging: A Benchmark Study"
- **Status:** Read via arXiv HTML (v2).
- **Research Question / evidence:** Compares rebalancing, GRL adversarial training, dCor, MINE and MMD on Morpho-MNIST, CheXpert (pleural effusion with sex confounder), and OCT (drusen with a synthetic frequency-notch confounder).
- **Model family:** Task-trained shared encoder with explicit, small split latent spaces; no foundation model, CLIP/DINO, or frozen-embedding experiment.
- **OOD / reliability / geometry:** Assesses task performance, latent-space disentanglement, correlation-strength sensitivity and cost; no Mahalanobis/kNN/energy or post-mitigation OOD reliability study.
- **Relation to Paper 4:** Strong methodological prior but no direct test of the medical-FM intervention × geometry × reliability-accessibility question.
- **Reading priority:** Read via arXiv HTML through Methods §3.5-start only (~56% of paper);
  Results §4 and Discussion §5 not retrieved (fetch truncated twice, two renderers) — one more
  retrieval attempt recommended.

### Shortkit-ML (medRxiv 2026.04.29.26352053)
- **Venue / Year:** medRxiv, 2026 (preprint)
- **Research Question:** Can a unified, multi-method framework detect shortcut learning in clinical-AI embedding spaces more reliably than any single method?
- **Scientific Claim:** Calibrated agreement across complementary detectors identifies shortcut signals more reliably than any single detector; pattern holds across synthetic benchmarks, chest X-ray cohorts, multiple backbone families, and out-of-domain CelebA validation.
- **Evidence:** 20+ detection methods, 6 mitigation strategies, CheXpert, MIMIC-CXR, synthetic ground-truth benchmarks, CelebA.
- **Model Family:** CheXpert: ResNet-50, DenseNet-121, ViT-B/16, ViT-B/32, DINOv2, RAD-DINO and MedSigLIP. MIMIC-CXR: RAD-DINO, ViT-B/16, ViT-B/32 and MedSigLIP.
- **Clinical Modality:** Chest X-ray (+ CelebA as out-of-domain natural-image check). No dermatology.
- **Relation to Paper 4:** Confirms (Level A, full text read) that FM embeddings can encode demographic attributes (RAD-DINO sex-probe F1 = 0.990 on MIMIC-CXR, quoted directly), so Paper 4 must not make first-leakage claims. It *does* have a mitigation module (§3.3: adversarial debiasing, last-layer retraining, contrastive debiasing, explanation regularization, with pre/post comparison) — correcting an earlier overstatement that it lacked one — but the reported Experiments/Results (§4) do not exercise or report mitigation outcomes, and it never evaluates Mahalanobis or any distance-based OOD/reliability score.
- **Novelty overlap:** High, Level-A-confirmed for generic FM shortcut *detection*; low for intervention × geometry × reliability accessibility in dermatology (confirmed absent, not just inferred).
- **Reading priority:** Full text read — threat resolved on primary-source evidence.

### PanDerm (arXiv:2410.15038 / Nat. Med.)
- **Relation to Paper 4:** Primary candidate FM backbone for our experiments (SSL ViT-L, strong benchmark performance, outperforms MONET/CLIP/BiomedCLIP). Not itself a novelty threat — it's an artifact we would use, not a competing scientific claim.
- **Reading priority:** Tier 1 — need architecture/pretraining details to plan Step 6 experimental design (linear probing access, checkpoint availability, license).

### "Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection" (arXiv:2510.15202)
- **Relation to Paper 4:** General-domain mechanistic account of *why* Mahalanobis-AUROC depends on feature geometry/normalization — directly useful for explaining (or challenging) Paper 3's flat/below-chance result, and for predicting whether the same mechanism would appear in FM embeddings (which have very different geometry: higher-dim, contrastively-trained, often normalized).
- **Reading priority:** Tier 1 — likely a key mechanistic citation for Paper 4's motivation.

### "One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution Detection" (arXiv:2601.10836)
- **Relation to Paper 4:** General-domain study of how training interventions reshape OOD-detection behavior — same high-level theme as Paper 3. Needs a full read to check if it already runs something like our disentanglement dose-response design outside medical imaging (which would still leave the medical-FM instantiation open, but would narrow novelty on the general mechanism).
- **Reading priority:** Tier 1.

---

## Deferred (Tier 3, logged for completeness, not read further this session)
StaRFM (2507.09222); "On the OOD Robustness of Foundation Models in Medical Image Segmentation" (2311.11096); "Delving into OOD Detection with Medical VLMs" (2503.01020); CANDOR (2607.18451); "Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography" (2606.11606); "Toward Clinically Ready Foundation Models in Medical Image Analysis" (2603.14271); "Visual concept ranking uncovers medical shortcuts..." (2602.05096); "Robustness and sex differences in skin cancer detection" (2504.11415); "Effect of Demographic Bias on Skin Lesion Classification" (2606.03214).
