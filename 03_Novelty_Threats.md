# Novelty Threat Analysis — Paper 4

Per project rules: whenever a paper close to our idea is discovered, stop and analyze before
continuing. This document records that stop and its resolution.

**Status after full-text verification pass (2026-08-04, session 5): ALL THREE TIER-1 THREATS NOW
LEVEL A (or majority-Level-A).** Project rule, per `07_Evidence_Register.md`: no RQ may be locked
or novelty threat closed on abstract/secondary-summary evidence alone (Level B/C) — only on
primary-source reading (Level A). Under that standard:

- **THREAT 1 (Gallo et al., CVPR 2026W) — NOW LEVEL A, full text read.** The user directly
  uploaded the CVF Open Access PDF (2026-08-04); all 10 pages read in full, including References.
  This is the single most consequential paper for the RQ, and it is now fully verified rather
  than a secondary-source claim. See the rewritten THREAT 1 section below and
  `07_Evidence_Register.md` for the per-claim ✓/△/✗ breakdown with quotes.
- **THREAT 2 (arXiv:2602.18502) — Level A for ~56% of the paper** (Abstract, Introduction,
  Related Work, Methods through the start of §3.5). Results (§4) and Discussion (§5) were not
  retrieved (fetch truncated twice, via two different renderers). The core structural claims
  (datasets, methods used) are verified; the "no FM, no OOD-reliability metric anywhere in the
  paper" claim is verified only for the part actually read. Not re-attempted this session.
- **THREAT 3 (Shortkit-ML) — Level A, full text read.** One correction made below: the earlier
  claim that the framework "does not train a debiasing/disentanglement intervention" was too
  strong. See the correction in the THREAT 3 section.

**This satisfies the Research Gate rule as literally stated** (see below) — every threat rated
HIGH risk is now Level A. Threat 2 remains partial (56%) but was rated MODERATE, not HIGH, so it
does not block under the rule as written. **Per Vinh's earlier explicit instruction not to proceed to Step 6, this document
reports the Gate status but does not itself initiate Step 6.**
See the Net Assessment and Gate Status sections below.

**Project rule (added 2026-08-04, session 3) — the Research Gate:**

```
Novelty Threat → Verified (Level A) → RQ Lock → Experimental Design (Step 6)
```

Step 6 may not begin while any novelty threat rated HIGH or "highest risk" is still Level B/C
("Status: UNVERIFIED"). This is a hard gate, not a soft preference.

**Access plan for Gallo et al. — resolved.** Automated access (CVF fetch, browser automation) had
failed for structural reasons and was not going to be re-attempted per project decision. The user
resolved this directly by uploading the PDF (2026-08-04, session 5), which is now the Level-A
source for everything in the THREAT 1 section below.

---

## THREAT 1 (highest risk, now Level A): Gallo et al., "Pre-Training and Fine-Tuning Effects on Out-of-Distribution Detection in Dermatology with Vision Foundation Models" (CVPR 2026 Workshops, Med-Reasoner)

**✅ VERIFICATION STATUS (2026-08-04, session 5): VERIFIED — Level A, full text read.** The user
uploaded the CVF Open Access PDF directly; all 10 pages read in full, including Tables 1–6,
Figures 1–4, and References. Every claim below is checked against the actual text, with quotes
recorded in `07_Evidence_Register.md`. This section supersedes every prior version of THREAT 1,
which was written from unverifiable search-engine summaries.

**Overlap (confirmed):**
- Same clinical domain (dermatology) and shares PAD-UFES-20 as a dataset (also uses Derm7pt-clinic, HAM10000, Fitzpatrick17k, Bug-Bites-Craqq, and others — not ISIC).
- Directly tests reliability/OOD-detector behavior (MSP, TS, EBO, MDS, RMDS, kNN) on foundation-model backbones (CLIP-pretrained ViT, ImageNet21k-pretrained ViT/Swin) — this runs the "generalize from CNN findings to FMs" question for the *reliability-scoring* half of our candidate RQ, and reports a result.
- **Confirmed finding, quoted (§3.2):** "A consistent and pronounced trend emerges: MDS substantially outperforms all other post-hoc methods considered." This is the same qualitative ranking Paper 2 established for supervised CNNs, now demonstrated at Level A for FM backbones in dermatology (Table 2: MDS 99.98%/90.18% AUROC on Derm7pt-clinic; 100.00%/97.59% on PAD-UFES-20, vs. 41–89% for MSP/TS/EBO/kNN/RMDS). **This closes the naive Paper-2-style version of the RQ — confirmed, not suspected.**

**Overlap — what is NOT there (confirmed by full read, not inferred):**
- **No domain-shortcut linear probe of any kind.** Zero occurrences of "adversarial," "GRL," "gradient reversal," "disentangl-," or "orthogonal" anywhere in the 10-page text. The paper's own "linear probing" (§3.4) is an *adaptation regime* — training a linear classification head on a frozen encoder — not a domain/acquisition-source leakage audit. This is a fundamentally different use of the term "probe" from Paper 1's methodology.
- **No adversarial/domain-disentanglement intervention is trained anywhere.** The only training axes manipulated are (a) backbone architecture × pretraining strategy (§3.3, Tables 3–4) and (b) ordinary fine-tuning intensity × LR schedule (§3.4, R1–R6). Neither is a debiasing or shortcut-mitigation objective.
- **No quantitative geometry analysis.** The only geometric evidence is a qualitative PCA(50)→t-SNE plot on a 50-image-per-dataset sample (Fig. 4). No condition number, no local intrinsic dimensionality, no spectral-decay slope, no covariance-structure metric — the exact quantities Paper 3 and the Paper 4 hypothesis are built around.
- **No probe-decodability test.** The paper never asks whether OOD/domain information remains linearly recoverable from an embedding when a distance-based score (like MDS) fails to separate it. Paper 3's central "decodable but not accessible" dissociation — the actual object of Paper 4's hypothesis — is untested here in every respect: no probe run, no distance-score/probe comparison, no dose-response ladder.
- **No calibration analysis** (no ECE, no calibration curves).

**Difference (confirmed):**
- Independent variable is *pretraining source × ordinary fine-tuning intensity* (frozen → partial → full FT, two schedules), not *domain-shortcut disentanglement strength*. These are mechanistically different interventions on the representation — one is task specialization, the other is an explicit adversarial suppression objective.
- "Cross-Domain Shift" (their Scenario 1) treats clinical-vs-dermoscopic modality itself as the ID/near-OOD split (PAD-UFES-20 + Derm7pt-clinic as ID, HAM10000 as near-OOD) — structurally different from our setup, where ISIC (dermoscopic) and PAD-UFES-20 (smartphone) are *combined as ID* and acquisition source is the shortcut predicted from a shared representation.
- Uses Derm7pt-clinic, not ISIC — the exact acquisition-shift pair differs from ours.

**New information this read adds (not previously anticipated from any B/C source):**
- Gallo's own qualitative mechanism account — "the margin between in-distribution samples and semantically adjacent but unseen conditions is compressed" under increasing fine-tuning intensity — is a coarser, non-adversarial analog of the phenomenon Paper 3 studies quantitatively. This is worth citing as convergent motivation (a second, independent line of evidence that training-induced representation reshaping affects near-OOD separability), but it is not a competing result: it is untested at the geometry level and uses a different intervention type.
- Their fine-tuning-intensity trade-off is now a **Level-A-confirmed empirical pattern** (not a hypothesized confound) that Paper 4's experimental design must control for via a matched ordinary-fine-tuning arm — this was already planned in `16_Minimal_Sufficient_Experiment.md`'s control-arm design, and is now better justified, not newly threatened.
- Discussion's future-work statement is generic ("training procedures that preserve representation structure... without sacrificing diagnostic performance") and does not name adversarial disentanglement or shortcut removal — confirming the authors themselves did not consider this specific mechanism a natural next step from their own results.

**Remaining novelty (confirmed, not just risk-assessed):**
- Paper 2's claim ("feature-space reliability scores > logit-space, generalizes to FMs") — **confirmed scooped.** A Paper 4 built around re-confirming this would be redundant. This was already the working assumption; it is now settled.
- Paper 1's mechanism (domain-shortcut linear-probe leakage on a medical-FM representation, and whether adversarial disentanglement removes it) — **confirmed untested** by this paper.
- Paper 3's mechanism (disentanglement-driven geometric reshaping dissociated from Mahalanobis-AUROC while remaining probe-decodable, on FM embeddings) — **confirmed untested** by this paper, and remains the most novelty-protected of the three prior results.

**Risk level: CONFIRMED HIGH for a Paper-2-style framing (settled, not just likely). CONFIRMED LOW for the actual selected Paper-1/Paper-3-style framing** (downgraded from the prior provisional "LOW-MODERATE," because absence is now demonstrated by a full read rather than inferred from missing search results).

**Resolved verdict:** Gallo et al. answers the reliability-scoring-hierarchy half of the original broad RQ ("do CNN findings generalize to FMs?") for the *scoring-method* dimension, on FM backbones, in dermatology. It does not touch, at any level, the intervention × geometry × probe-decodability question that is Paper 4's actual selected hypothesis. **The Research Gate condition for this threat is met.**

---

## THREAT 2: arXiv:2602.18502, "Mitigating Shortcut Learning via Feature Disentanglement in Medical Imaging: A Benchmark Study"

**Overlap:** Feature disentanglement to mitigate shortcuts in medical imaging; it benchmarks
rebalancing, GRL adversarial classification, and dCor/MINE/MMD latent-space splitting.

**Full-text resolution (2026-08-04) — Level A for Abstract/Intro/Related Work/Methods
through the start of §3.5; Results §4 and Discussion §5 NOT retrieved (confirmed by direct
read via `arxiv.org/html/2602.18502`, cross-checked with the ar5iv mirror, both truncating at
the same point):**
- ✅ Verified directly: authors Sarah Müller & Philipp Berens (Hertie Institute / Tübingen AI
  Center); datasets Morpho-MNIST, CheXpert, OCT; methods benchmarked include a rebalancing
  baseline, an adversarial classifier ("AdvCl") explicitly citing the Ganin & Lempitsky
  gradient-reversal domain-adaptation paper, distance correlation, MINE, and MMD latent-space
  splitting (equations for MINE and MMD read directly in §3.4.2–3.4.3).
- ⚠️ Partially verified: "CheXpert confounder = sex, OCT confounder = synthetic frequency-notch"
  — plausible and consistent with how these datasets are normally used, but the specific
  confounder-construction text sits in §3.5.1–3.5.2, which the fetch did not reach. Not
  confirmed at Level A.
- ⚠️ Partially verified: "no foundation-model backbone; no Mahalanobis/kNN/energy/OOD-reliability
  score anywhere in the paper" — true for the ~56% of the paper actually read (zero hits on
  "foundation model," "ViT," "CLIP," "ResNet," "pretrain," "Mahalanobis," "OOD" in that span),
  but Results, Discussion, and Training Details (§3.5.4) were not read, so this cannot be
  claimed for the full paper yet.
- ❌ Not verified: "Discussion names larger datasets and foundation models as future work" — §5
  was never retrieved. This specific claim should be treated as unconfirmed, not true, until
  read.

**Remaining novelty:** It is strong, now Level-A-confirmed prior art for selecting and
implementing disentanglement controls (rebalancing + AdvCl/GRL + dCor/MINE/MMD), but whether it
tests the medical-FM setting or anything resembling Paper 3's geometry/reliability dissociation
remains open pending a second retrieval attempt at §4/§5.

**Resolved verdict: MODERATE methodological overlap (confirmed); LOW direct overlap with the
locked RQ for the ~56% read, UNCONFIRMED for the remainder.**

---

## THREAT 3: Shortkit-ML (medRxiv 2026.04.29.26352053)

**Overlap:** A general framework explicitly designed to detect shortcut learning in clinical-AI
embedding spaces across CNN, transformer, self-supervised, and medical vision-language
backbones.

**Full-text resolution (2026-08-04) — Level A, full text read directly via medRxiv
`.full-text`, including Results §4.2.1–4.2.4, Discussion §5.1–5.7, Limitations §6, Conclusion §7:**
- ✅ Verified directly (§4.1.2, quoted): CheXpert uses 7 backbones — ResNet-50, DenseNet-121,
  ViT-B/16, ViT-B/32, DINOv2, RAD-DINO, MedSigLIP (n=2,000); MIMIC-CXR uses 4 — RAD-DINO,
  ViT-B/16, ViT-B/32, MedSigLIP (n≈1,491).
- ✅ Verified directly (Fig. 2 caption / §4.2.2, quoted verbatim): "probe F1 = 0.990" for MIMIC-CXR
  RAD-DINO sex encoding.
- ✅ Verified directly: zero mentions of "Mahalanobis" anywhere in the full text; the framework's
  "Geometric Analysis" module uses centroid distance / effect size for shortcut *detection*, a
  different mechanism from distance-based OOD/reliability *estimation*. No dermatology, no
  ISIC/PAD-UFES-20 — only CheXpert, MIMIC-CXR, CelebA.
- ❌ **CORRECTION to the original claim.** §3.3 ("Mitigation and Evaluation Loop") explicitly
  describes model-level mitigation modules — "adversarial debiasing, explanation regularization,
  last-layer retraining, and contrastive debiasing" — assessed by "rerunning the same detection
  pipeline on modified embeddings" for pre/post comparison. So it is **false** that the framework
  "does not train a debiasing/disentanglement intervention" — it has that capability built in.
  **What remains true and is now Level-A-confirmed:** the reported Experiments (§4) and Results
  (§4.2.1–4.2.4) only report *detection* results (synthetic, CheXpert/MIMIC-CXR, CelebA); no
  mitigation pre/post numbers, and no intervention dose-response, appear anywhere in Results,
  Limitations, or Conclusion. The corrected, accurate statement is: *ShortKit-ML ships a
  mitigation capability but this paper does not exercise or report it* — not "it doesn't have
  one."

**Remaining novelty:** Even accounting for the correction above, this paper does not run its
mitigation modules, does not touch dermatology or ISIC/PAD-UFES-20, and does not use
distance-based OOD/reliability scoring. Our dermatology-specific, hypothesis-driven angle — and
specifically the Paper 3 mechanism — survives this paper intact, now on Level-A evidence rather
than an abstract-level inference.

**Resolved verdict: HIGH, Level-A-confirmed overlap with a claim of "first FM-embedding
demographic-leakage detection across backbone families"; LOW overlap with the locked
intervention-and-accessibility RQ (confirmed, not just inferred).**

---

## THREAT 4 (low risk, foundational context): "Self-Supervision on Images and Text Reduces Reliance on Visual Shortcut Features" (arXiv:2206.07155)

**Overlap:** Establishes, in general-domain vision (not medical), that SSL/CLIP-style pretraining reduces shortcut reliance relative to supervised pretraining — i.e., prior evidence exists for the "generalizes, and even improves" side of our hypothesis, in a non-medical setting.

**Difference:** Not medical, not dermatology, not about acquisition/domain-source shortcuts specifically, not about reliability estimation, predates current medical FMs (PanDerm etc.), uses different benchmarks and different shortcut definitions entirely.

**Remaining novelty:** High. This paper is best used as a citation supporting the motivation ("there's reason to expect FMs behave differently, but medical acquisition-shortcuts + reliability-estimator geometry hasn't been tested") rather than as a threat.

**Risk level: LOW.** Treat as supporting citation, not competing claim.

---

## Net assessment (2026-08-04, session 5 — all three Tier-1 threats now Level A or majority-Level-A)

All three Tier-1 threats are now resolved on primary-source evidence:
ShortKit-ML confirms our dermatology/Mahalanobis/intervention-dose-response angle survives
intact. arXiv:2602.18502 confirms strong methodological overlap on disentanglement technique
choice but, for the ~56% of the paper actually read, no FM and no OOD-reliability evaluation —
the remaining 44% (Results, Discussion) is still unread and could in principle change this, though
it was rated MODERATE (not HIGH) risk and does not gate Step 6 under the stated rule. **Gallo et
al. is now fully read (Level A, all 10 pages).**

**What the Gallo read actually changes, concretely:**
- The naive Paper-2-style version of the RQ ("does feature-space beat logit-space OOD scoring on
  FMs?") is now **confirmed** answered by prior work, not just suspected. Paper 4 was already not
  designed around this claim as its main contribution, so this does not require a redesign — it
  removes a possible fallback framing, not the selected one.
- The selected Paper-1/3-style mechanism (domain-disentanglement intervention × quantitative
  geometry × probe-decodability dissociation) is **confirmed untested** by Gallo — this is a
  stronger result than the prior "no evidence found" status, because it comes from having read
  the entire paper rather than from failed searches. The novelty case for the selected RQ is now
  on firmer ground than the Gallo-ablation robustness check (below) assumed, since that check
  could only reason about *absence of information*, not *confirmed absence in the actual text*.
- Gallo's fine-tuning-intensity trade-off finding is now a Level-A-confirmed empirical pattern
  that strengthens (does not threaten) the case for including a matched ordinary-fine-tuning
  control arm in the experimental design — this was already planned, and is now better justified.

The narrowed direction — testing whether *domain-disentanglement adaptation* of a dermatology FM
produces the Paper-3-style geometry/reliability-estimator dissociation while remaining
probe-decodable — remains the best-supported candidate, and is now supported by a **complete**
read of the single most consequential competing paper, not a partial or absent one.

---

## Research Gate status (2026-08-04, session 5)

```
Novelty Threat → Verified (Level A) → RQ Lock → Experimental Design (Step 6)
```

- Threat 1 (Gallo et al., rated HIGH for the Paper-2 framing / previously LOW-MODERATE for the
  selected framing): **Level A — verified.**
- Threat 2 (arXiv:2602.18502, rated MODERATE): **Level A for ~56%** — not HIGH-rated, does not
  block per the rule as written, though the unread 44% (Results/Discussion) remains a soft loose
  end worth closing before submission.
- Threat 3 (Shortkit-ML, rated HIGH for generic FM-leakage detection / LOW for the locked RQ):
  **Level A — verified.**

**Per the rule as stated, the Gate condition is met — no HIGH-risk threat remains Level B/C.**

**GATE OPENED (2026-08-04, session 6, by Vinh's explicit decision).** Vinh's assessment: the Gate
opens not because Gallo "supports" the hypothesis, but because Gallo now clearly establishes the
boundary between what has been done and what hasn't. Step 6 (Experimental Design) has begun — see
`22_Experimental_Design.md`. Per Vinh's own framing: "Không phải vì Gallo 'ủng hộ' hypothesis. Mà
vì Gallo đã xác định rõ ranh giới cái gì đã được làm và cái gì chưa." The RQ is now locked (see
`05_RQ_Candidates.md`) and relabeled "Geometry-mediated accessibility after representation
intervention" (see `Scientific_Hypothesis.md`) — disentanglement is the tested instantiation, not
the definition of the hypothesis.

---

## Robustness check: RQ re-evaluated with Threat 1 (Gallo et al.) fully removed

Purpose: an RQ that only survives because of one unverified paper's specific abstract-level
claim is a fragile RQ. This section re-derives the novelty assessment using **only Level-A
(primary-source-verified) evidence, plus Level-B evidence from papers other than Gallo** —
i.e., as if Gallo et al. did not exist. The comparison at the end reports only what changes.

**1. Does the proposed RQ still make scientific sense without Gallo?**
Yes, and its case is not weaker for the absence — the RQ was never *supported* by Gallo, it was
only *risk-assessed* against it. With Gallo removed, there is now **zero verified evidence,
at any evidence level, that anyone has run a distance-based reliability/OOD estimator
(Mahalanobis or otherwise) on a dermatology foundation model at all**, let alone under a
disentanglement-style intervention. Shortkit-ML (Level A) never touches Mahalanobis or
dermatology. Müller & Berens (Level A, partial) never touches foundation models or
OOD/reliability scoring. The self-supervision paper (Level B, general vision) never touches
reliability estimation. So the core mechanism the RQ is built on — Paper 3's geometry-vs-
estimator dissociation, tested on an FM under an explicit debiasing intervention — remains
completely untouched by anything we've verified. The RQ stands on its own evidence, not on
Gallo's absence of interference.

**2. Which claims remain unsupported (open, no evidence either way)?**
- Whether feature-space (Mahalanobis-style) reliability scores outperform logit-space scores
  *for dermatology foundation models specifically, under any adaptation regime* — this was
  previously treated as "probably already shown" (via Gallo). Without Gallo, it reverts to
  **completely unsupported** — we do not know, from any verified source, whether Paper 2's
  core CNN-era finding holds, partially holds, or fails for FMs in dermatology.
- Whether ordinary fine-tuning intensity (frozen → full fine-tune) is a *competing explanation*
  that could produce a geometry/reliability dissociation similar to what a disentanglement
  intervention would produce — previously cited as "Gallo shows this." Without Gallo, this is
  an **untested assumption**, not an established fact. It still needs to be built into the
  experimental design as a control condition, but can no longer be justified by citation.

**3. Which claims become weaker?**
- The justification for narrowing *away* from a Paper-2-style replication ("that territory is
  probably already answered, don't waste a paper on it") weakens to unsupported. This was a
  risk-avoidance argument, and the risk it was avoiding is no longer established.
- The boundary condition in `05_RQ_Candidates.md` ("otherwise adaptation alone is a competing
  explanation, as Gallo shows") loses its citation and must be reframed as our own hypothesis
  to test via a matched ordinary-fine-tuning control arm, not a cited fact.

**4. Which claims remain completely open (unaffected either way)?**
- Paper 1's mechanism (domain-shortcut linear-probe leakage magnitude on a medical-FM
  representation, frozen vs. lightly adapted) — open regardless of Gallo; nothing found, with or
  without Gallo, tests this in dermatology.
- Paper 3's mechanism (disentanglement-driven geometry reshaping decoupled from distance-based
  reliability-estimator performance, while remaining probe-decodable) on FM embeddings — open
  regardless of Gallo; this was never claimed to be touched by Gallo's abstract either.
- Whether FM-style pretraining changes domain-shortcut *magnitude* relative to the supervised
  CNN baseline established in Paper 1 — open regardless of Gallo (Gallo, even per its own
  abstract, never runs a shortcut probe).
- Dermatology-specific shortcut auditing generally — Shortkit-ML establishes the analogous
  chest-X-ray result, dermatology remains untested regardless of Gallo.

**Differences from the previous (Gallo-inclusive) assessment — this is the actual answer to the
ablation:**

| Aspect | With Gallo (Level B/C) | Without Gallo (ablation) |
|---|---|---|
| Core RQ (intervention × geometry × estimator-accessibility on a dermatology FM) | Provisionally selected | **Unchanged — provisionally selected** |
| "Paper-2-style FM replication is redundant" | Asserted as likely true (risk: HIGH for that framing) | **Downgraded to unsupported — reopens as a legitimate secondary/supporting result, not required as the main claim** |
| "Adaptation intensity alone is a competing explanation" | Cited as an established fact | **Reframed as our own hypothesis to test via a control arm** |
| Overall confidence in novelty | Narrower, but grounded in one unverified paper's specific findings | **Direction identical; confidence interval wider in both directions until Gallo is actually read — could turn out more novel (if Gallo tests less than feared) or reveal a real overlap (if Gallo tests more than its abstract suggests)** |

**Conclusion of the robustness check:** the RQ does not depend on Gallo et al. for its validity.
Gallo affects confidence and scope calibration, not direction. This is the result we wanted from
this check — an RQ that only survived because of one unverified paper would have been a bad RQ.
This one didn't move.

**Status note (2026-08-04, session 5): this ablation is now superseded by the actual Level-A
read, kept here for the record.** The real result is better than the ablation predicted in one
respect — the ablation could only conclude "the RQ doesn't depend on an unknown," whereas the
full read concludes "the RQ's specific mechanism is confirmed absent from the paper that most
threatened it." The ablation's directional conclusion (RQ survives, confidence/scope calibration
is what moves) held up exactly as predicted.
