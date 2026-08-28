# One-Sentence Paper — Paper 4

**Purpose**: the coherence test before Introduction. If this sentence isn't smooth, the Introduction
won't be. Per Vinh's roadmap step 4.5 (2026-08-06), written after the JBHI narrative survey and
before the extended abstract.

**Locked wording constraints** (from the JBHI survey + Vinh's review of it):
1. Contribution verb is *evaluate / investigate / test / present*, never *propose* attached to a
   method claimed to outperform. "Framework **for evaluation**," not "framework replacing prior
   methods."
2. No "mechanism" in the opening — reviewer will ask *causal mechanism or explanatory hypothesis?*
   Use "previously reported **account**" / "previously proposed **explanation**"; define mediation
   only after Methods.
3. Non-positive findings use the JBHI hedge register (survey §4): epistemic-strength ("limited
   evidence"), quantified-description ("remarkably dropped," "largely absent"), scope-caveat ("under
   the evaluated..."). Never "failed" / "does not generalize" — zero occurrences in JBHI 2023–2025.

---

## Selected (v2, relocked 2026-08-06 after primary-source verification)

> Using a preregistered causal evaluation within a single frozen dermatology foundation model, we
> tested whether the previously reported association between representation geometry and
> distance-based detector performance accounts for the probe–reliability dissociation reported on
> convolutional networks, and found limited empirical support under the evaluated backbone, layer,
> and task, where the dissociation was approximately two orders of magnitude smaller than
> originally reported.

**Why v1 was withdrawn.** v1 said the account "transfers to a frozen dermatology foundation model."
Primary-source verification (arXiv:2510.15202v3, PDF full text) established that the geometric
account was itself developed **across foundation-model backbones** — verbatim: "We conduct a
large-scale study across diverse foundation-model backbones." A transfer-to-FM framing therefore
rested on a false premise. Verification further established that the same paper contains **zero**
uses of "probe" and never discusses the probe–reliability dissociation: no prior source advanced a
geometric account *of the dissociation*.

**What the evidence actually supports** — three distinct positions, kept distinct everywhere:

1. The **dissociation** was reported on convolutional networks (probe 0.72–0.81 vs Mahalanobis
   ≈0.40, i.e. below chance).
2. The **geometry–performance association** was established separately, correlationally, across
   frozen foundation-model backbones ("track", not "cause"). That paper does run interventions, but
   they target its own internal mechanism term, not detector AUROC.
3. **This paper formulates the hypothesis linking (1) and (2), and tests it causally within one
   fixed frozen backbone.** Formulating the link is part of the contribution, not prior work.

No priority claim is made. The word "transfer" does not appear in the manuscript's framing.

**Why this one.** It carries the design *and* the observation that explains why support is limited.
The weaker framing ("we tested transfer and found limited support") reads as *we ran the test and it
came out negative*, which invites the reviewer question *were you underpowered?* — a question we
would have to partly concede, since n=5 seeds under 8-way Holm is thin. The selected sentence
answers it pre-emptively with an observation, not an inference.

**Two rejected phrasings, and why** (both proposed and corrected during PI review, 2026-08-06):

1. *"the failure it was proposed to explain is largely absent at baseline"* — rejected: "largely
   absent" is interpretation, not result. Reviewer asks *how absent? what threshold defines
   largely?* and there is no principled answer.
2. *"the probe–reliability dissociation was not reproduced"* — rejected: **not true to C5**. The
   dissociation *was* reproduced and was statistically significant in 9/10 configs (paired
   probe-vs-Mahalanobis, p = 2.5e-05 to 8.4e-08). What collapsed is its magnitude, not its
   existence. A reviewer holding our own C5 table would catch this immediately.
3. *"the probe–reliability dissociation **central to that account**"* and its replacement
   *"the **characteristic** probe–reliability dissociation"* — both rejected: each asserts the
   dissociation is a defining/core feature, one interpretive step beyond describing it. Neither is
   needed once "than in the original CNN setting" identifies which dissociation is meant, so the
   qualifier was dropped rather than replaced.

The retained wording states the magnitude collapse directly: mean probe−Mahalanobis gap ≈0.0016
here (probe ≈0.99999, Mahalanobis ≈0.998) against ~0.32–0.41 in Paper 3 (probe 0.72–0.81 vs
Mahalanobis ≈0.40) — a ratio of ≈200×, so "two orders of magnitude" is accurate and conservative.
This is the significance-≠-effect-size caveat pointed in the less obvious direction: the
significance is real, the effect size is what did not transfer.

*where* rather than *because*: `because` asserts that limited support *follows from* the small gap —
defensible, but one more claim to defend. `where` states both facts and leaves the connection to
Discussion, matching the Result/Discussion separation applied to condition_number.

Hedges used, mapped to the survey's inventory: *two orders of magnitude smaller*
(quantified-description, cf. "performance remarkably dropped"), *limited empirical support*
(epistemic-strength), *under the evaluated backbone, layer, and task* (scope-caveat).

## Alternates (kept, not discarded)

**A — transfer-only framing.** Closer to the original draft shape; drops the baseline-absence
explanation.

> Using a preregistered causal evaluation, we tested whether a previously reported
> representation-geometry account of distance-based OOD-detection degradation transfers from CNNs to
> a frozen dermatology foundation model, and found limited empirical support under the evaluated
> backbone, layer, and task.

**B — finding-first framing.** Leads with the observation rather than the design; strongest claim,
but foregrounds the negative before establishing the rigor that makes it credible. (Revised to drop
the rejected "largely absent" phrasing — see above.)

> Using a preregistered causal evaluation on a frozen dermatology foundation model, we found the
> probe–reliability dissociation previously reported on CNNs to be two orders of magnitude smaller
> than reported, leaving limited empirical support for the representation-geometry account proposed
> to explain it.

## Governance rule (locked 2026-08-06, per Vinh)

**This sentence is the manuscript's constitution.** Every sentence in Abstract, Introduction,
Results, and Discussion must be a longer version of it — never a stronger or a weaker one.

- **Ceiling.** Anything above this claim level is cut, not softened: "the mechanism does not
  transfer," "geometry stops explaining reliability," "foundation models invalidate the account,"
  "the phenomenon is absent," "fails," "does not generalize."
- **Floor.** Anything so weak it loses the message is also wrong: "we evaluated X" with no finding
  attached.
- **Vocabulary.** "account" / "explanation," never "mechanism," until mediation is formally defined
  in Methods.

**Structural consequence — the RQ changed shape.** Paper 4 began as *does the mediation mechanism
transfer?* After C2–C5 it is *how much of the previously reported phenomenon is still observable in
a frozen FM, and does its proposed explanation hold there?* The order matters and must be preserved
everywhere:

1. the phenomenon (probe–reliability dissociation, reported on CNNs) — **first**;
2. the representation-geometry account proposed to explain it — **second**;
3. Paper 4 evaluates both under transfer to a frozen FM.

This makes C5 the foundation of the paper rather than a secondary result, and pre-empts the reviewer
question *why test an explanation for a phenomenon you never established?* — because the paper
states from the opening that it is evaluating both.

## Introduction flow this implies (JBHI gap → evaluate → hedged implication)

1. Prior work reports that representation geometry explains the accessibility of acquisition-domain
   information to distance-based reliability estimators (CNNs).
2. Whether that account holds for frozen vision foundation models is untested.
3. We conducted a preregistered causal evaluation (Gate 0 implementation checks, Gate 1 manipulation
   checks, preregistered claims C1–C5, Holm correction).
4. The probe–reliability dissociation is approximately two orders of magnitude smaller on this
   backbone than in the original CNN setting; the preregistered geometry metrics show no
   corrected-significant association with scorer performance.
5. These findings motivate caution when extrapolating CNN-derived representation-level explanations
   to frozen foundation models.
