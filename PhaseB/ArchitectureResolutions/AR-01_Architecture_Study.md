# AR-01 — Step 1: Architecture Study

**Literature survey only. No architecture chosen or recommended for Paper 4 anywhere in this
document.** Produced per `AR-01.md`'s Step 1 spec: a neutral map of how existing literature
intervenes on frozen vision foundation model representations, gathered without reference to
CSG-Lite, GRL, dual-encoder CNNs, or dermatology. Method: background research agent, ~30 targeted
searches across every mechanism family in scope, each paper verified via actual fetch of its
abstract/text (not inferred from title). A first pass under a superseded framing was discarded and
redone from a clean, single-shot spec — see `26_Lab_Notebook.md`, 2026-08-05, for that episode.

**Verification note (this session, before presenting to Vinh):** spot-checked 6 of the 27 included
papers directly via `arxiv.org` abstract fetch — the highest-stakes ones (every paper claiming a
dual-encoder/auxiliary-branch structure: FairNVT, Fairness-Aware LoRA Under Demographic Privacy,
"DualFairVL"/Toward Robust Medical Fairness) plus AOFT, O-TPT, and CLIP-Adapter as anchors. All 6
confirmed real and accurately characterized. One discrepancy found in the *excluded* list (below,
flagged, does not affect any included-table finding). The remaining ~21 included papers and 9
excluded papers were not independently re-verified beyond the agent's own fetch-based process — full
independent verification of all 36 citations was judged disproportionate to a Step 1 evidence-gathering
pass; flag any specific paper for a closer check before Step 2 leans on it heavily.

**Saturation:** reached per the agent's account — the final three search rounds (gradient-reversal
adapters, standalone feature-projector fairness heads, LoRA+concept-erasure+orthogonal-subspace)
surfaced only papers already logged, out of scope, or reinforcing already-populated cells without
adding new ones. 27 papers surveyed.

---

## 1. Per-paper table (27 papers, 11 columns)

| Paper | FM | Frozen | Intervention | Mechanism | Design Constraint | Dual encoder | Orthogonality | Projection | Geometry touched? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Visual Prompt Tuning (Jia et al., arXiv:2203.12119, ECCV 2022) | ViT | Yes | Learnable prompt tokens in input space (Shallow: layer 1; Deep: every layer) | General task adaptation of frozen backbone via input-space perturbation | Minimal-parameter PEFT (<1% params), input-space only | No | No | No | No | — |
| AdaptFormer (Chen et al., arXiv:2205.13535, NeurIPS 2022) | ViT | Yes | AdaptMLP bottleneck adapter parallel to MLP/FFN in every block | General task adaptation of frozen backbone | Minimal-parameter PEFT (<2% params) | No | No | No | No | — |
| CLIP-Adapter (Gao et al., arXiv:2110.04544) | CLIP | Yes | Bottleneck feature adapter after final layer, residual-blended | Few-shot task adaptation via residual feature blending | Few-shot supervision only | No (existing CLIP dual branches) | No | No | No | Verified this session |
| Tip-Adapter (arXiv:2111.03930, ECCV 2022) | CLIP | Yes | Training-free key-value cache adapter from few-shot features | Few-shot task adaptation, non-parametric | Training-free (no backprop) | No | No | No | No | — |
| FairNVT (Tang et al., arXiv:2604.16780) | ViT-Base | Yes | Per-block bottleneck adapters (task + sensitive); Gaussian noise on sensitive embedding | Suppress protected-attribute info from task representation via noise injection | Single-stage, lightweight adapters, no adversarial objective | Yes — separate adapter+head, task vs. sensitive | Yes — cosine-similarity loss | No | No — attacker is a nonlinear MLP probe, not geometric analysis | Verified this session |
| Fairness-Aware LoRA Under Demographic Privacy (Kamalaruban et al., arXiv:2503.05684) | ViT-Base | Yes | LoRA in Q/V of every self-attention layer (task-LoRA + sensitive-LoRA) | Suppress spurious correlation between representation and sensitive attribute | Demographic privacy — attributes/predictors inaccessible to developer | Yes — task-LoRA vs. sensitive-attribute-LoRA | Yes — Frobenius-norm penalty | No | No | Verified this session — abstract confirms orthogonality loss as one of three evaluated methods, ViT-Base backbone confirmed |
| AdapterTune (Khazem, arXiv:2603.14706) | ViT | Yes | Zero-init residual low-rank bottleneck adapter, in series after every block | Preserve pretrained-function stability / avoid early-epoch representation drift | Zero-init — exact match to pretrained function at init | No | No | No | No (theoretical framing only) | — |
| LoRA for Frozen VLM BIQA (Adam & Li, arXiv:2606.02002) | SigLIP (+CLIP-H) | Yes | LoRA in Q/V of frozen SigLIP | Inject task-specific perceptual signal into frozen features | Minimal-parameter PEFT (0.23%) | Partial — 3-stream (NSS+SigLIP+CLIP-H) concat | No | No | No | — |
| DiCoOp (arXiv:2603.28555) | CLIP | Yes | CoOp-style prompt tuning + gradient-reversal layer on domain-specific context | Suppress domain-specific information (domain-invariance) | No additional encoder; context-vector-only optimization | No | No | No | No | — |
| Debiased Prompt Tuning w/o Annotations (arXiv:2503.08368) | CLIP | Yes | CoOp-style prompt tuning + group reweighting from zero-shot pseudo-labels | Reduce reliance on spurious correlation / worst-group robustness | No manual annotation of spurious attributes | No | No | No | No | — |
| FairLLaVA (arXiv:2603.26008) | LLaVA (CLIP ViT-L + LLM) | Yes | LoRA in LM blocks + projector + demographic-attribute-classifier (DAC) head | Regularize representations to be demographic-invariant (MI minimization) | Lightweight LoRA plug-in, no backbone update | Partial — DAC head for MI estimate | No (MI minimization instead) | No | No | — |
| FairerCLIP (arXiv:2403.15593) | CLIP | Yes | RKHS debiasing functions on image/text features, closed-form | Debias joint image/text representations via statistical-independence (HSIC) minimization | Closed-form; works with or without labels | No | No (HSIC instead) | Yes — RKHS closed-form function | No (spectral by construction, not by separate analysis) | — |
| Closed-Form Concept Erasure via Double Projections (arXiv:2604.10032) | Stable Diffusion (variants) + FLUX | Yes (training-free) | Closed-form projection in left null space of concept directions | Remove target-concept-correlated directions | Training-free, closed-form | No | Yes — null-space/orthogonal-complement | Yes | Null-space/subspace analysis | — |
| Uni-AdaVD (arXiv:2607.14521) | U-Net/DiT/AR image gen./text-to-video diffusion | Yes (training-free) | Orthogonal value decomposition in multimodal-attention value space, inference-time | Suppress target semantic/concept directions | Training-free, weight-preserving, architecture-universal | No | Yes — orthogonal value decomposition | Yes | No | — |
| Identity Sanitization Projection (arXiv:2604.05296) | CLIP, DINOv2/v3, SSCD | Yes (training-free) | Post-hoc orthogonal projection removing top-k PCA identity directions | Remove identity-correlated directions, preserve utility | Training-free, single-shot, no second encoder, no label supervision at mitigation | No | Yes — orthogonal complement of identity subspace | Yes | Linear probe (verification attack) + subspace analysis | — |
| AOFT (Yang et al., arXiv:2507.13260) | ViT | Yes | Down/up-projection matrices of LoRA/Adapter replaced by an approximately-orthogonal matrix from a single learnable vector | Improve generalization by structurally enforcing orthogonality among adaptation directions | Low-rank/parameter-efficient structure retained | No | Yes — approximately-orthogonal adapter/LoRA matrices | No | No — orthogonality on **weight matrices**, not representations | Verified this session — weight-geometry vs. representation-geometry distinction confirmed |
| A Prompt Array Keeps the Bias Away (Berg et al., arXiv:2203.11933, AACL 2022) | CLIP | Yes | Learnable prompt array prepended to text queries; adversarial debiasing + contrastive loss | Suppress protected-attribute-correlated signal from text-query representation | Prompt-only training; minimal degradation to image-text representation | Partial — adversarial discriminator, training only | No | No | No | — |
| Ablating Protected Attribute Reps. (Ratzlaff et al., arXiv:2410.13976) | LLaVA | Yes | Mean-difference direction u between biased/standard activations; projected out of residual stream at inference | Suppress protected-attribute direction from internal representation during generation | Training-free, ~1000-sample budget | No | Yes — rank-1 orthogonal projection removing direction u | Yes | No | — |
| Contrastive Adapters for FM Group Robustness (Zhang & Ré, arXiv:2207.07180) | CLIP | Yes | Lightweight adapter on frozen FM embeddings, contrastive objective | Reduce intra-class/intra-group dispersion to improve worst-group robustness | Minimal-parameter PEFT (≤1%), no FM finetuning | No | No | No | No (conceptual discussion, no formal measurement) | — |
| BendVLM (Gerych et al., arXiv:2411.04420) | CLIP | Yes | Orthogonal projection removing attribute-direction subspace; per-input nearest-neighbor equalization | Remove protected-attribute subspace, input-adaptive/nonlinear | Fine-tuning-free; no a priori input knowledge; no global correction | No | Yes — V = I − A(AᵀA)⁻¹Aᵀ | Yes | No | — |
| DualFairVL / "Toward Robust Medical Fairness..." (Xia et al., arXiv:2508.18886) | Medical VLM, CLIP-style dual encoder | Yes | Ridge-regularized projection → orthogonalized sensitive/target text anchors; hypernetwork instance-aware visual prompts; cross-attention fusion | Disentangle and align sensitive vs. target-attribute info across modalities while debiasing | Frozen dual encoders; minimal trainable params (3.6M) | Yes — parallel dual-branch, sensitive vs. target | Yes — orthogonalized text anchors + prototype separation loss | Yes | No explicit report beyond the mechanism itself | Verified this session. 4 medical imaging modalities, 8 datasets — not dermatology-exclusive |
| D2TPT (Song et al., arXiv:2511.11690) | CLIP | Yes | Test-time image + text prompts; retrieval-augmented modulation + cross-modal consistency distillation | Correct model-side overconfidence and data-side cross-modal misalignment at test time | Zero-shot/unlabeled test data only | Partial — modality-specific prompt branches | No | No | Partial — L2 cross-modal feature-distance | — |
| O-TPT (Sharifdeen et al., arXiv:2503.12096, CVPR 2025) | CLIP | Yes | Test-time learnable prompts; orthogonality constraint on textual class features | Improve calibration via feature-dispersion/orthogonality of textual class features | Prompt-only tuning, no fine-tuning | No | Yes — orthogonality of textual features | No | Feature dispersion/orthogonality analysis (core contribution) | Verified this session — CVPR 2025 confirmed |
| A-TPT (Ahamed et al., arXiv:2510.26441) | CLIP | Yes | Test-time learnable prompts; angular-diversity regularizer on hypersphere | Improve calibration via angular-diversity-driven uniformity | Prompt-only tuning | Partial — angular diversity generalizes orthogonality | No | Angular/hyperspherical distance analysis | Frozen status inferred from TPT paradigm, not directly quoted |
| D-TPT (Han & Hwang, arXiv:2510.09473) | CLIP | Yes | Test-time learnable prompts; dimensional-entropy maximization regularizer | Improve calibration by reducing reliance on one dominant feature dimension | Prompt-only tuning | No | No | No | Dimensional-entropy/dominant-dimension analysis | Frozen status inferred from TPT paradigm, not directly quoted |
| BiPrompt (Gupta et al., arXiv:2601.02147) | CLIP | Yes | Learnable gating/normalization/temperature; Grad-CAM-guided fg/bg masking + cosine-similarity penalty | Minimize spurious-cue mutual information; steer toward isotropic, causal representation | No retraining/domain supervision; only scalar params updated | No | Yes — cosine-similarity penalty (bg vs. full-image) | No | No (isotropy stated as goal, not measured) | Only gating/norm/temp scalars learned, not prompt tokens despite the name |
| Erasing CLIP Memories (Mishra et al., arXiv:2512.14137) | CLIP | Yes (training-free) | Closed-form nullspace projection from orthonormal basis of target-class text-embedding subspace | Remove class-specific knowledge/alignment (targeted unlearning) | Training-free, data-free, non-destructive | No | Yes — orthonormal-basis nullspace projection | Yes | Null-space/subspace analysis | — |

## 2. Tier 1 — Intervention taxonomy

| Family | Frozen backbone | Trainable component | Typical placement |
|---|---|---|---|
| Visual/input-space prompt tuning | Yes | Learnable input-space prompt tokens | Prepended to patch-embedding sequence |
| Text/context prompt tuning (CoOp-style) | Yes | Learnable context vectors / prompt array | Prepended to text-encoder token input |
| Instance-conditional / hypernetwork-generated prompt | Yes | Hypernetwork + generated visual prompt | Injected into visual encoder, conditioned on instance |
| Bottleneck adapter (in-block) | Yes | Down→up projection bottleneck | Parallel/serial to attention/FFN, every block |
| Feature/output adapter | Yes | Bottleneck adapter, residual-blended | After final encoder output |
| Orthogonal adapter/LoRA | Yes | Structurally-orthogonal low-rank matrices | Replaces standard down/up-projection |
| LoRA (standard) | Yes | Low-rank ΔW on attention weights | Q/V projections of self-attention |
| Training-free key-value cache adapter | Yes | Non-parametric cache from few-shot features | Appended after encoder output |
| Closed-form kernel/RKHS debiasing function | Yes | RKHS operator (closed-form solve) | Applied to image/text features before similarity |
| Closed-form linear/null-space projection (static) | Yes | None (precomputed projector) | Applied post-hoc to frozen embeddings |
| Closed-form input-adaptive projection | Yes | None (computed per-query at inference) | Applied post-hoc, locally |
| Orthogonal value decomposition | Yes | None (training-free decomposition) | Value space of multimodal attention |
| Learnable re-centering/gating | Yes | Scalar gating/normalization/temperature | Applied to text-side class embeddings |

## 3. Tier 2 — Mechanism taxonomy

| Mechanism | Count |
|---|---|
| Suppress/remove protected-attribute (sensitive) information | 6 |
| General task adaptation (non-fairness) | 2 |
| Few-shot task adaptation via feature blending | 2 |
| Concept erasure / remove target-concept-correlated directions | 3 |
| Calibration via geometric feature-dispersion regularization | 3 |
| Reduce reliance on spurious correlation / worst-group robustness | 2 |
| Adaptation stability / avoid representation drift | 1 |
| Perceptual/task-specific signal injection | 1 |
| Orthogonality-structured adaptation for generalization | 1 |
| Debias joint representations via statistical independence | 1 |
| Domain-invariance / suppress domain-specific information | 1 |
| Disentangle + align sensitive vs. target attributes cross-modally | 1 |
| Input-adaptive/local debiasing of embeddings | 1 |
| Test-time cross-modal alignment + confidence calibration | 1 |
| Isotropy-driven causal/domain-invariant representation shift | 1 |

## 4. Tier 3 — Architectural ingredients

| Ingredient | Frequency |
|---|---|
| Orthogonality constraint (explicit) | 12 |
| Projection head (dedicated closed-form/learned projection) | 8 |
| Auxiliary branch (parallel task/sensitive pathway) | 5 |
| Feature split (explicit task vs. sensitive/nuisance decomposition) | 4 |
| Adversarial head (discriminator predicting sensitive attribute) | 2 |
| **Second/auxiliary encoder (full separate encoder network)** | **0** |

## 5. Tier 4 — Design Constraint taxonomy

| Constraint | Count |
|---|---|
| Minimal-parameter-budget PEFT (explicit small % trainable) | 6 |
| Training-free (no gradient-based training at all) | 7 |
| Prompt-only optimization | 6 |
| No label/annotation supervision for spurious/sensitive attributes | 2 |
| Zero-shot/unlabeled test-time adaptation | 1 |
| Demographic/attribute privacy (attributes inaccessible to developer) | 1 |
| Few-shot supervision only | 1 |
| Preserve pretrained initialization / avoid representation drift | 1 |
| Single-stage, no adversarial objective | 1 |
| Structural low-rank/orthogonal parameter efficiency | 1 |

## 6. Tier 5 — Geometry-treatment taxonomy

| Geometry treatment | Count |
|---|---|
| No (downstream performance only) | 20 |
| Null-space/subspace analysis | 2 |
| Linear probe + subspace/null-space analysis | 1 |
| Feature dispersion / orthogonality analysis | 1 |
| Angular/hyperspherical distance analysis | 1 |
| Dimensional-entropy / dominant-dimension analysis | 1 |
| Cross-modal embedding-distance (alignment) analysis | 1 |

## Architecture Study Summary

- Total papers surveyed: **27**
- Intervention families found (Tier 1): **13**
- Mechanisms found (Tier 2): **15**
- Design constraints found (Tier 4): **10**
- Geometry-treatment categories found (Tier 5, including "No"): **7**
- Papers with dual-encoder/auxiliary-branch structure (Yes + Partial): **7** (Yes: 3 — FairNVT, Fairness-Aware LoRA, DualFairVL; Partial: 4 — LoRA-BIQA, FairLLaVA, A Prompt Array, D2TPT)
- Papers using orthogonality (Yes + Partial): **13** (Yes: 12; Partial: 1 — A-TPT)
- Papers touching representation geometry in any form: **7 / 27**
- Papers with no geometric analysis: **20 / 27**
- **Full second/auxiliary encoder (a wholly separate network, CSG-Lite's own structure): 0 / 27**

## Papers found but excluded (with reason)

| Paper | arXiv ID | Reason excluded |
|---|---|---|
| Individually fair representation learning for DINOv2 | (no arXiv preprint found) | Paywalled — could not fetch/verify text |
| Orthogonal Concept Erasure for Diffusion Models | 2605.28902 | Edits backbone weights directly via closed-form transform — fails "frozen backbone + separate module" scope |
| Where Vision Becomes Text (OCR Routing Bottleneck) | 2602.22918 | Pure mechanistic-interpretability analysis, no intervention mechanism |
| Toward Localizing and Repairing Bias in Attention Heads (ROBIN) | 2607.12863 | Pure NLP (BERT/GPT-2/DistilGPT-2) |
| CUBIC | 2505.11060 | Diagnostic/bias-identification only, no intervention |
| Configurable Fairness (VLM parity optimization) | 2403.10624 | VLM used only as an external oracle for a separate ResNet18 classifier — doesn't intervene on the FM's own representations |
| Lightweight Fairness for LLM-Based Recommendations | 2603.23780 | Pure NLP/recommendation-system domain |
| FairViT-GAN | 2509.23859 | Insufficient evidence of a frozen backbone (described as fully-trained hybrid) |
| Better Hit the Nail on the Head... | 2212.04273 | Pure NLP (EMNLP 2022) |

**Verification flag (this session):** the agent's stated reason for excluding **ViT-Adapter
(arXiv:2205.08534)** — "requires full fine-tuning, not a frozen backbone" — did not hold up under a
direct re-check this session. The paper's own abstract describes a "pre-training-free adapter," which
plausibly means the adapter needs no separate pretraining stage, not that the backbone is frozen
during downstream training — genuinely ambiguous from the abstract alone, and not resolved with
confidence either way here. This affects only an *excluded* paper (does not change any Tier 1–5
count or the headline "0 full second encoders" finding), but is flagged rather than silently
corrected, since I could not confirm which reading is right.
