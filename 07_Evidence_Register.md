# Evidence Register — Paper 4

Purpose: track, per claim used anywhere in the Paper 4 documents, what level of evidence backs
it, and whether it has actually been verified against a primary source. Rule going forward:

**Evidence hierarchy**
- **Level A — Primary source** (methods/results/discussion, actually read by us). Usable to lock an RQ, accept/reject a novelty threat, or write Introduction claims.
- **Level B — Abstract / figure / conclusion only.** Usable for screening and reading-priority triage. NOT usable to lock an RQ or close a novelty threat.
- **Level C — Secondary summary** (search-engine synthesis, LLM-generated summary, aggregator snippet, not the paper itself). Never usable to conclude novelty either way.

A claim only counts as "Verified" if it was checked at Level A with a locatable quote/section.

---

## Gallo et al., "Pre-Training and Fine-Tuning Effects on Out-of-Distribution Detection in Dermatology with Vision Foundation Models" (CVPR 2026W)

**Update (2026-08-04, session 5) — FULL TEXT NOW READ, Level A. The user directly uploaded the
CVF Open Access PDF (the CVPR-workshop version, "identical to the accepted version" per its own
watermark, pages 6932–6941). Read in full via the Read tool (all 10 pages, including References).**
This supersedes every prior Level B/C entry below. Per-claim verification against the actual text,
with quotes:

| Claim | Evidence level obtained | Verified? |
|---|---|---|
| Uses PAD-UFES-20 and Derm7pt-clinic | A — Table 1; §3.1 Scenario 1: "The in-distribution (ID) set consists of clinical datasets (PAD-UFES-20 and Derm7pt-clinic)" | ✅ |
| Feature-space (Mahalanobis/MDS) beats logit-based scores | A — §3.2, quoted directly: "A consistent and pronounced trend emerges: MDS substantially outperforms all other post-hoc methods considered." Table 2 confirms numerically (MDS 99.98/90.18 AUROC vs. MSP/TS/EBO in the 41–69 range). | ✅ |
| Independent variable is pretraining/fine-tuning intensity (linear probe → partial FT → full FT, two LR schedules) | A — §3.4, quoted almost verbatim: "Three adaptation regimes with increasing specialization are considered: (i) linear probing, where the encoder remains frozen; (ii) partial fine-tuning, where only the final transformer layers are updated; and (iii) full fine-tuning, where all backbone parameters are optimized... Each regime is trained with both cosine and step-decay schedules, yielding six configurations (R1–R6)." | ✅ |
| No domain-shortcut linear probe, no disentanglement/debiasing intervention tested | A — confirmed by exhaustive absence across the full read text: zero occurrences of "adversarial," "GRL," "gradient reversal," "disentangl-," or "orthogonal" anywhere in the paper. The only "linear probing" in the paper is adaptation regime (i) above (a frozen-encoder classifier head) — not a domain/acquisition-source leakage audit in Paper 1's sense. No shortcut-probe methodology of any kind appears. | ✅ |
| Backbone = fixed ViT-B/16 CLIP model selected via MDS | A, but **partial/corrected** — true for the detector-comparison experiment (§3.2, where ViT-B/16 CLIP is used to select MDS as the reference detector) and for the fine-tuning-intensity experiment (§3.4, quoted: "The backbone (ViT-B/16 CLIP), MDS detector, and preprocessing pipeline are kept fixed so that only the effect of fine-tuning intensity changes"). **Not true globally**: §3.3 deliberately varies backbone architecture (Swin-B, Swin-L, ViT-B/16, ViT-L/14) crossed with pretraining strategy (random init, ImageNet21k, IN21k+Adam, IN21k+Adam+Aug, CLIP) — Tables 3–4. | ⚠️ Partial (corrected) |

**New Level-A findings not previously anticipated (bear directly on novelty/risk):**
- Central finding is a quantified **ID-accuracy vs. near-OOD-separability trade-off** as fine-tuning intensity increases: "minimal adaptation preserves strong near-OOD separability (AUROC ≈ 96%) but yields lower diagnostic accuracy... increased fine-tuning substantially improves ID performance (up to >62%) at the cost of reduced near-OOD separability (AUROC ≈ 83–85%)" (§3.4, Fig. 2).
- Their own proposed mechanism, stated qualitatively, is representation-compression: "As the representation becomes increasingly optimized for the target task... the margin between in-distribution samples and semantically adjacent but unseen conditions is compressed, making near-OOD inputs harder to distinguish" — but this is **never tested quantitatively**. The only geometry evidence in the paper is a qualitative t-SNE plot (Fig. 4, PCA-50 → t-SNE on a 50-image-per-dataset sample). No condition number, no LID, no spectral-decay slope, no covariance-structure metric of any kind appears anywhere in the paper.
- No probe-decodability test of any kind is run — the paper never asks whether OOD/domain information remains linearly recoverable when a distance-based score fails. Paper 3's central "decodable but not accessible" dissociation is completely untested here.
- Discussion's future-work pointer is generic, not our mechanism: "investigate training procedures that preserve representation structure during adaptation, incorporate uncertainty-aware objectives, and improve near-OOD separability without sacrificing diagnostic performance" — no mention of adversarial disentanglement, domain-shortcut removal, or acquisition-source leakage.
- Their "Cross-Domain Shift" (Scenario 1) treats clinical-vs-dermoscopic modality itself as the near-OOD split (PAD-UFES-20 + Derm7pt-clinic as ID, HAM10000 as near-OOD) — structurally different from our acquisition-shortcut framing, where ISIC (dermoscopic) and PAD-UFES-20 (smartphone) are combined as ID and acquisition source is the shortcut predicted from the shared representation.

**Access status (superseded):** Could not obtain primary-source *text* automatically this project
(CVF fetch attempts and browser automation all failed for structural reasons, documented in prior
entries below for the record). **Resolved 2026-08-04 (session 5) by direct user upload of the PDF.**

**Prior update (2026-08-04, session 4) — proceedings *location* Level A verified** (kept for
record): official CVF index confirmed the paper is real, correctly titled, correctly attributed,
published in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) Workshops*, June 2026, pages 6932–6941, BibTeX key `Gallo_2026_CVPR`. That entry resolved
existence/citation only; **content is now also Level A per this update.**

---

## arXiv:2602.18502, "Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging"

Primary source obtained via `arxiv.org/html/2602.18502` — **Abstract, Section 1 (Introduction), Section 2 (Related Work), and Section 3 (Methods) through the start of 3.5 (Experimental design) were read directly.** Sections 4 (Results) and 5 (Discussion) were NOT retrieved — the fetch tool truncated the page before reaching them, and a second attempt via the ar5iv mirror hit the same truncation point. This is disclosed, not papered over.

| Claim | Evidence level obtained | Verified? |
|---|---|---|
| Authors: Sarah Müller, Philipp Berens (Hertie Institute / Tübingen AI Center) | A — title block read directly | ✅ |
| Datasets: Morpho-MNIST, CheXpert, OCT | A — confirmed via section headers + body text ("Radiology task (CheXpert)", "Ophthalmology task (OCT)", "Artificial dataset (Morpho-MNIST)") | ✅ |
| Methods benchmarked: rebalancing (data-centric baseline), adversarial classifier (AdvCl, GRL-style, citing Ganin & Lempitsky's "Unsupervised domain adaptation by backpropagation"), distance correlation, MINE, MMD | A — AdvCl description and citation read directly (§3.3); MINE and MMD equations read directly (§3.4.2–3.4.3) | ✅ |
| CheXpert confounder = sex; OCT confounder = synthetic frequency-notch | Not reached — this level of dataset-construction detail sits in §3.5.1–3.5.2, which the fetch did not reach | ❌ (plausible but unconfirmed) |
| No foundation-model backbone anywhere in the paper | A for the ~56% of the paper actually read (Abstract/Intro/RelatedWork/Methods 3.1–3.5-start): zero mentions of "foundation model," "ViT," "CLIP," "ResNet," or "pretrain[ed]" in that span. Results/Discussion/Training-details (§3.5.4, §4, §5) not read. | ⚠️ Partial — true for ~56% of the paper, not confirmed for the rest |
| No Mahalanobis/kNN/energy/OOD-reliability score evaluated anywhere in the paper | Same partial basis as above — zero hits in the ~56% read, remainder unread | ⚠️ Partial |
| Discussion explicitly names foundation models as future work | Not reached at all (§5 never retrieved) | ❌ Not confirmed |

---

## Shortkit-ML (medRxiv 2026.04.29.26352053)

Primary source obtained via `medrxiv.org/.../v1.full-text` — **full text read**, including Abstract, Introduction, Related Work, Framework Architecture, Experimental Design, Experimental Results (4.2.1–4.2.4), Discussion (5.1–5.7), Limitations and Future Work (§6), and Conclusion (§7).

| Claim | Evidence level obtained | Verified? |
|---|---|---|
| CheXpert backbones: ResNet-50, DenseNet-121, ViT-B/16, ViT-B/32, DINOv2, RAD-DINO, MedSigLIP (7 total, n=2,000) | A — §4.1.2, quoted directly | ✅ |
| MIMIC-CXR backbones: RAD-DINO, ViT-B/16, ViT-B/32, MedSigLIP (4 total, n≈1,491) | A — §4.1.2, quoted directly | ✅ |
| MIMIC-CXR RAD-DINO sex-probe F1 = 0.990 | A — Figure 2 caption, §4.2.2, quoted directly: "probe F1 = 0.990" | ✅ |
| No Mahalanobis / distance-based OOD-reliability estimator anywhere in the framework | A — zero hits for "Mahalanobis" across full text; "Geometric Analysis" module (§ "Geometric Analysis") uses centroid distance/effect size, a different mechanism, for shortcut *detection* not distribution-shift *reliability estimation* | ✅ |
| No dermatology, no ISIC/PAD-UFES-20 | A — only CheXpert, MIMIC-CXR, CelebA evaluated | ✅ |
| Framework "does not train a debiasing/disentanglement intervention" | **✗ CORRECTION NEEDED.** §3.3 ("Mitigation and Evaluation Loop") explicitly describes model-level mitigation modules — "adversarial debiasing, explanation regularization, last-layer retraining, and contrastive debiasing" — with pre/post comparison via "rerunning the same detection pipeline on modified embeddings." **However**, the reported Experiments (§4) and Results (§4.2.1–4.2.4) only report *detection* results (synthetic, CheXpert/MIMIC-CXR, CelebA) — no mitigation pre/post numbers appear anywhere in Results, Limitations, or Conclusion. So: the framework *has* a mitigation capability; this *paper* does not report mitigation experiment results. | ⚠️ Partial — original claim was too strong; corrected below |
| Does not study an intervention dose-response | A (consistent with the above — no dose-response results reported in this paper) | ✅ as corrected |
| Uses pre-extracted embeddings for auditing | A — confirmed, embeddings are model-agnostic inputs to the framework | ✅ |

---

## Net effect of this verification pass on the locked RQ/Gap

- **Shortkit-ML:** verification mostly confirms the prior analysis, with one correction (mitigation is a *described framework capability*, not an *exercised, reported experiment* in this paper — the distinction matters for how strongly we can cite it, but does not change the novelty conclusion).
- **arXiv:2602.18502:** verification confirms the core structural claims (datasets, methods, GRL/MINE/MMD) at Level A for ~56% of the paper; the "no FM, no OOD-reliability metric" claim is Level A for that same 56% but not confirmed for Results/Discussion, which were not retrieved. Worth a second retrieval attempt (try smaller chunked fetches or a different mirror) before treating it as fully closed.
- **Gallo et al.: now Level A, full text (2026-08-04, session 5).** The single most consequential paper for the RQ (see `03_Novelty_Threats.md` Threat 1) is fully read. Net result: the Paper-2-style claim (feature-space beats logit-space on FM backbones in dermatology, across fine-tuning intensities) is **now confirmed scooped at Level A**, not just suspected. The Paper-1/3-style mechanism (domain-shortcut linear probe, disentanglement/adversarial intervention, quantitative geometry metrics, probe-vs-distance decodability dissociation) is **confirmed absent** from this paper — none of it appears anywhere in the text. This is a *stronger* de-risking than the provisional Gallo-ablation analysis assumed, because that analysis could only argue "we don't know," whereas this reads "we checked, and it isn't there." **All three Tier-1 novelty threats are now Level A (or, for Threat 2, Level A for the majority of the paper).** See `03_Novelty_Threats.md` for the updated Net Assessment and Research Gate status.
