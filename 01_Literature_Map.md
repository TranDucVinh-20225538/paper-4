# Literature Map — Paper 4

Status: updated after Tier-1 reads (2026-08-04). The map remains non-exhaustive, but the three
initial novelty threats have been resolved at methods/results level.

Candidate RQ under test: *Do findings on shortcut learning and reliability estimation
generalize from supervised CNNs to medical foundation models?*

---

## Cluster A — Medical / Dermatology Foundation Models

| Paper | Tier | Note |
|---|---|---|
| PanDerm (Yan et al., *Nat. Med.* 2025/2026; arXiv:2410.15038) | 1 | Multimodal derm FM, ViT-L, SSL (masked latent modeling + CLIP-teacher alignment), 2M+ images / 11 institutions / 4 modalities, 28 benchmarks. Likely candidate FM backbone for our experiments. Beats MONET, CLIP, BiomedCLIP on their benchmarks. |
| MONET (derm-specific CLIP) | 2 | Referenced as a baseline inside PanDerm's evaluation. Candidate second FM backbone. |
| "Transformer-Based Foundation Learning for Robust and Data-Efficient Skin Disease Imaging" (*Diagnostics* 2026; PMC12897459) | 2 | Derm SSL ViT; cross-dataset transfer gains of +3.5–5.8% over supervised counterparts — a generalization claim worth checking against our disentanglement/shortcut findings. |
| "Towards Scalable Foundation Models for Digital Dermatology" (arXiv:2411.05514) | 3 | Scaling/data-curation paper for derm FMs. |
| "Toward Clinically Ready Foundation Models in Medical Image Analysis: Adaptation Mechanisms and Deployment Trade-offs" (arXiv:2603.14271) | 2 | General medical-FM review; frames adaptation strategies against robustness/calibration/auditability. Useful for framing, not a direct threat. |

## Cluster C — Shortcut Learning

| Paper | Tier | Note |
|---|---|---|
| "Self-Supervision on Images and Text Reduces Reliance on Visual Shortcut Features" (arXiv:2206.07155) | 1 | **Foundational prior claim**: SSL/CLIP-style pretraining reduces shortcut reliance vs. supervised training — but in *general* vision, not medical/dermatology, and not tested against acquisition/domain shortcuts specifically. This is the closest existing evidence for the "generalizes" side of our hypothesis. |
| "Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging: A Benchmark Study" (arXiv:2602.18502) | Read | CheXpert/OCT/Morpho-MNIST benchmark of GRL, rebalancing and dependence-minimization methods. No FM backbone, no OOD/reliability estimators, and no residual linear-probe analysis. Strong method prior, not a direct threat to locked RQ. |
| "Visual concept ranking uncovers medical shortcuts used by large multimodal models" (arXiv:2602.05096) | 2 | Commercial LMMs on skin lesions; shortcut behavior varies by skin-type subgroup and in-context examples. Different mechanism (prompting/ICL) from our probing/disentanglement approach, but same clinical shortcut concern. |
| Shortkit-ML (medRxiv 2026.04.29.26352053) | Read | CXR embedding audit across ResNet/DenseNet, ViT, DINOv2, RAD-DINO and MedSigLIP. Establishes that sensitive attributes can remain probe-decodable in FM embeddings; no training intervention, OOD estimator, or dermatology setting. |
| "Robustness and sex differences in skin cancer detection: logistic regression vs CNNs" (arXiv:2504.11415) | 3 | Derm-specific shortcut/bias, but compares LR vs. CNN, not FM. |
| "Effect of Demographic Bias on Skin Lesion Classification" (arXiv:2606.03214) | 3 | Demographic bias in derm classifiers; tangential. |
| npj Digital Medicine, "Shortcut learning in medical AI hinders generalization..." (2024/2025) | 3 | Method for estimating generalization gap from shortcuts without external data; general medical AI. |

## Cluster D — Reliability / Calibration / OOD

| Paper | Tier | Note |
|---|---|---|
| **Gallo et al., "Pre-Training and Fine-Tuning Effects on Out-of-Distribution Detection in Dermatology" (CVPR 2026 Workshops, Med-Reasoner)** | **Read — direct Paper-2 overlap** | PAD-UFES-20 + Derm7pt-clinic/HAM10000 scenarios; MSP, TS, energy, MDS, relative MDS and kNN. MDS dominates in their ViT-B/16 CLIP comparison; frozen/partial/full ordinary fine-tuning shifts the ID-accuracy vs near-OOD trade-off. No domain probe or debiasing/disentanglement intervention. |
| StaRFM (arXiv:2507.09222) | 2 | CLIP/SAM calibration + distribution-shift robustness via Fisher-info + confidence-misalignment penalties. General + medical (BraTS/ATLAS), not dermatology, not shortcut-focused. |
| "On the OOD Robustness of Foundation Models in Medical Image Segmentation" (arXiv:2311.11096) | 2 | Claims FM-based models are more robust to domain shift than other architectures (segmentation task); relevant as a generic "FM claims transfer" prior, but different task/modality and no shortcut-mechanism analysis. |
| "Delving into OOD Detection with Medical Vision-Language Models" (arXiv:2503.01020) | 2 | First systematic OOD-detection study across medical VLMs; general medical imaging, not derm-specific, not shortcut/disentanglement-linked. |
| CANDOR (arXiv:2607.18451) | 2 | Chance-calibrated nearest-neighbor discordance metric for frozen FM encoders; 22 encoders, 20 datasets, 7 domains (dermatology unclear if included — check). Relevant methodologically (frozen-embedding reliability geometry) but not shortcut/disentanglement-linked. |
| "Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography" (arXiv:2606.11606) | 2 | Echoes Paper 3's "decodable but not accessible" theme (CLS-token pooling at chance, patch-level probes near-perfect) but chest X-ray, not dermatology, and about pooling/localization rather than domain shortcuts or disentanglement. |
| "Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection" (arXiv:2510.15202) | 2 | General-domain mechanistic study of *why* Mahalanobis AUROC depends on feature geometry/normalization — directly relevant to explaining Paper 3's flat/below-chance Mahalanobis result; not medical, not FM-specific. |
| "One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution Detection" (arXiv:2601.10836) | 2 | General-domain study of how training interventions reshape OOD-detection behavior — same theme as Paper 3 (representation-shaping intervention decoupled from reliability-estimator performance), worth checking for overlap with our dose-response design. |

## Cluster E — Representation Geometry

Session 3 scope note: this pass deliberately searched for *mechanism* literature (why would
information be probe-decodable but distance-inaccessible?) rather than more FM or dermatology
papers. Full mechanism analysis, evidence levels, and a synthesis table are in
`08_Mechanism_Map.md` — this section only indexes the papers found.

| Paper | Tier | Note |
|---|---|---|
| "Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection" (arXiv:2510.15202) | **1 — read at Level A** | **Primary mechanism candidate.** Large-scale study across foundation-model/SSL backbones: Mahalanobis OOD performance is predicted by local intrinsic dimensionality (LID) × within-class spectral-decay slope, not by information content per se. Directly supports a falsifiable explanation for Paper 3's dissociation. See Mechanism Map §1. |
| "Beyond the Class Subspace: Teacher-Guided Training for Reliable OOD Detection" (arXiv:2603.11269) | 2 (Level B/C, unread) | Neural-collapse framing: supervised training concentrates variance in a class-discriminative subspace, suppressing orthogonal directions. Candidate second mechanism (variance suppression without erasure). See Mechanism Map §2. |
| "MahaVar: OOD Detection via Class-wise Mahalanobis Distance Variance under Neural Collapse" (arXiv:2605.14413) | 1 (not yet read) | Formalizes the neural-collapse ↔ Mahalanobis-variance connection directly. Priority read for next session. |
| "Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality" (arXiv:1801.02613) | 3 (Level B/C, analogy only) | Adversarial-*input*-perturbation literature (not adversarial-*training*): LID rises around adversarially perturbed points. Weak-evidence analogy for how GRL/adversarial representation training might raise local dimensionality. |
| "Interpreting the Curse of Dimensionality from Distance Concentration and Manifold Effect" (arXiv:2401.00422) | 2 (Level B/C, general theory) | Standard high-dimensional-statistics result: distances concentrate as dimensionality rises, degrading distance-based (not linear) methods. Background confound to control for, not an intervention-specific mechanism. |
| "On Linear Separation Capacity of Self-Supervised Representation Learning" (arXiv:2310.19041) | 2 (Level B/C, unread) | Theoretical conditions for SSL-representation linear separability — relevant as a baseline/boundary-condition check (are FM embeddings more linearly separable *before* any intervention?), not a dissociation mechanism itself. |

## Cluster B — Representation Learning

No additional Cluster-B-specific papers found beyond those already logged in Cluster E above and
in Cluster C (disentanglement methods, arXiv:2602.18502). The mechanism search this session
effectively merged B and E, since the relevant literature (neural collapse, LID, disentanglement
geometry) spans both clusters. Domain-invariant representation learning specifically applied to
*foundation models* (as opposed to general representation-geometry theory) remains unsearched —
logged below as a coverage gap.

---

## Coverage gaps to close next session
1. Read MahaVar (arXiv:2605.14413) and "Beyond the Class Subspace" (arXiv:2603.11269) at
   primary-source level — both are Level B/C mechanism candidates currently.
2. Domain-invariant representation learning applied specifically to foundation models (as
   opposed to general geometry/neural-collapse theory) — not yet searched.
3. Citation-graph search around Gallo, Müller & Berens, and ShortKit-ML.
4. PanDerm checkpoint/licensing read before committing to the experimental backbone (still
   deferred — see session-3 decision not to touch checkpoints before the Research Gate opens).
