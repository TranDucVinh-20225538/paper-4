# RQ Candidates — Paper 4

**Status: LOCKED (2026-08-04, session 6, by Vinh's explicit decision).** Per project rule (see
`07_Evidence_Register.md`): an RQ may not be locked on abstract/secondary-summary evidence. All
three Tier-1 novelty threats are now verified at primary-source level (Level A, or Level A for
the majority of the paper for Threat 2). Gallo et al. (CVPR 2026W) was directly uploaded by the
user and read in full (2026-08-04) — see `03_Novelty_Threats.md` THREAT 1 and
`07_Evidence_Register.md` for the complete verification. Research Gate opened and Step 6
(Experimental Design) begun — see `22_Experimental_Design.md`.

**Gallo et al. — confirmed result (not the provisional ablation anymore):** the paper confirms,
at Level A, that feature-space (MDS) OOD scoring beats logit-space scoring on FM backbones in
dermatology — closing the naive Paper-2-style framing for good. It also confirms, by full read,
that it contains **no** domain-shortcut linear probe, **no** adversarial/disentanglement
intervention, **no** quantitative geometry analysis, and **no** probe-decodability test — meaning
the RQ below is untouched by this paper's actual content, not just by its absence from search
results. This is a stronger result than the session-3 ablation predicted (see
`03_Novelty_Threats.md`'s ablation status note): the ablation could only argue "the RQ doesn't
depend on an unverified paper," whereas the full read now shows "the RQ's specific mechanism is
confirmed absent from the paper that most threatened it."

**Per the Research Gate rule as literally stated, the condition for RQ lock is now met.** Per
Vinh's explicit earlier instruction, this document reports that status but does not treat it as
authorization to proceed to Step 6 (Experimental Design) — that decision is Vinh's to make.

## Original candidate RQ (as given)
"Do findings on shortcut learning and reliability estimation generalize from supervised CNNs to
medical foundation models?" — **too broad / partially pre-answered** for the reliability-scoring
half (see Threat 1, Gallo et al.).

## Locked Research Question (2026-08-04, session 6)

**When a dermatology foundation model's representation is reshaped by an explicit intervention —
domain-adversarial disentanglement as the primary instantiation tested, with a matched
ordinary-fine-tuning arm as a control for intervention type — do the resulting changes in
embedding geometry mediate the accessibility of acquisition-domain shift to post-hoc distance-based
reliability estimators, independent of whether that information remains linearly decodable?**

Chain under test: `Representation intervention → Geometry changes → Accessibility changes →
Reliability changes` (see `Scientific_Hypothesis.md`, revised 2026-08-04). Domain-adversarial
disentanglement is the tested instance of "representation intervention," not its definition — see
the relabeling rationale in `Scientific_Hypothesis.md`.

### Operational subquestions

1. Relative to matched ordinary adaptation, does domain-disentanglement change domain-probe
   performance, diagnostic utility, calibration, and external-domain performance?
2. Across a pre-registered intervention-strength ladder, how do FM embedding geometry measures
   co-vary with Mahalanobis, kNN, ViM, energy/logit scores, and selective-risk behavior?
3. Does a residual linear domain probe recover acquisition source where unsupervised
   distance-based estimators are flat or uninformative — the Paper-3 dissociation?

### Falsifiable outcomes

- **Supports the Paper-3 generalization:** a reproducible geometry shift and residual probe
  decodability occur without a commensurate reliability-score gain.
- **Refutes it:** debiasing removes probe decodability, or geometry/reliability move together.
- **Also valuable:** FM pretraining prevents the CNN-era dissociation; that establishes a
  meaningful boundary condition rather than a failed replication.

### Resolved by the Gallo full-text read (session 5)

"Does feature-space (Mahalanobis-style) reliability scoring beat logit-space scoring for
dermatology FMs" is now **confirmed answered by Gallo et al. (Level A)** — yes, consistently,
across backbones and adaptation regimes. This should **not** be designed as a main or secondary
claim of Paper 4 — it would be a direct, confirmed redundancy, not just a risk. It can still
appear briefly as background/motivation ("prior work has established X; we build on it to ask Y"),
but not as a result Paper 4 presents as its own.

The boundary condition below (fine-tuning intensity as a competing explanation) is now a
**Level-A-confirmed empirical pattern from Gallo**, not an untested assumption — this strengthens
the requirement for a matched ordinary-fine-tuning control arm rather than weakening it.
