# Reviewer Pre-Mortem — Paper 4

Assume the paper has been submitted in its current hypothesis form. Reviewer #2 rejects it. This
is that review, restricted to novelty, hypothesis, mechanism, and internal validity — not writing
style. Followed by a proposed RQ redesign intended to make the rejection go away.

---

## The Review

**Summary of contribution.** The authors extend a previously reported finding (Paper 3: under a
domain-adversarial disentanglement intervention on a supervised CNN, distance-based reliability
estimators fail while a linear probe still recovers domain information) to a foundation-model
setting, and propose that representation geometry — not architecture — mediates this dissociation.

**On novelty.** The central empirical claim — that Mahalanobis-style OOD detection depends on
representation geometry (local intrinsic dimensionality, spectral decay) rather than on raw
information content — is not new; it is imported from arXiv:2510.15202, a paper about natural-
image OOD detection under ordinary pretraining variation, with no disentanglement or debiasing
intervention involved. What this submission adds is applying that account, post hoc, to a
different intervention (domain-adversarial training) in a different domain (dermatology). That is
a reasonable extension, but the framing — "geometry mediates accessibility, architecture is
incidental" — is stated as if it were a general law, when the evidence base is at most a
two-architecture, one-dataset-pair comparison. A sample of one CNN setup and one (or two)
foundation-model setups cannot support an architecture-agnostic claim; it can at most show the
pattern was "not contradicted" in this specific instance. The manuscript should not claim
generality it has not earned.

**On the hypothesis.** Prediction 3 (conditional independence of architecture given geometry) is,
in effect, a null-result prediction: "once you control for geometry, architecture doesn't matter."
Null results are notoriously ambiguous. Failing to find a residual architecture effect is
consistent with two very different states of the world: (a) architecture truly doesn't matter, or
(b) the study did not sample enough geometric diversity in the foundation-model arm to detect an
effect if one exists (this is explicitly flagged as a confound in the authors' own confound map,
item #3 there — credit for identifying it, but it is not resolved). Without a power analysis or an
explicit minimum-detectable-effect argument, "no residual architecture effect found" is not
interpretable evidence for the hypothesis.

**On mechanism.** No causal pathway is offered connecting the actual training objective (a
gradient-reversal domain-adversarial loss) to the specific geometric quantities the paper measures
(local intrinsic dimensionality, spectral-decay slope). The two candidate mechanisms are stated as
descriptive correlates borrowed from other settings (general-vision OOD detection; supervised
neural collapse), not derived from first principles for *this* intervention. Mechanism B, in
particular, is cited from a paper the authors themselves have not read at primary-source level
(their own Evidence Register marks it Level B/C) — building part of the paper's explanatory
apparatus on an unverified secondary source is a real liability if it reaches submission unresolved.
A toy or synthetic demonstration — even outside the medical domain, with a known ground-truth
generative process — that a suppression-style objective provably moves LID/spectral-decay in the
predicted direction would go a long way toward converting this from a correlational story into an
explanatory one. As submitted, it is descriptive.

**On internal validity.** The retrospective test on Paper 3's own existing CNN checkpoints is not
independent evidence for the hypothesis — it is the same data (or the same underlying phenomenon)
that motivated the hypothesis in the first place. Presenting a retrospective "confirmation" on
that data as validating evidence would be circular; it can only be treated as a sanity check, and
the manuscript needs to say so explicitly rather than let a casual reader conflate it with an
independent test. Separately, the confound map the authors provide (dimensionality mismatch,
normalization-convention differences, potential FM pretraining-data leakage, pooling/layer-choice
sensitivity, sample-size imbalance) lists at least four confounds independently capable of
producing the entire observed CNN-vs-FM pattern with no "geometry mediates accessibility"
mechanism required at all. Until these are controlled — not just listed — any cross-architecture
comparison in this paper is uninterpretable. This is the review's central objection: the paper's
own appendix materials show the authors know what could go wrong, but the main submission does
not yet demonstrate that these were actually ruled out.

**Recommendation:** Reject in current framing. The underlying empirical question is worth
answering, and the falsifiable structure of the hypothesis is a genuine strength, but the
manuscript claims more generality than a two-architecture comparison can support, and has not yet
ruled out the confounds it itself identifies.

---

## Proposed RQ redesign to defuse this review

1. **Narrow the claim from a law to an audit.** Replace "geometry mediates accessibility,
   architecture is incidental" (a general claim) with: *"We test whether a specific, pre-specified
   geometric account (LID × spectral-decay slope), developed for OOD detection under ordinary
   pretraining variation in general vision, also accounts for a previously reported dissociation
   between probe-decodability and distance-based reliability estimation under a domain-
   disentanglement intervention — in both a supervised CNN and a foundation-model setting."* This
   is an audit/replication-extension framing, matching the fallback already anticipated in
   `09_Risk_Register.md` item #6, and it does not require the paper to establish a general law from
   two data points — only to report whether the account holds, partially holds, or fails in each
   setting, which is informative either way.

2. **Explicitly demote the retrospective CNN test to "hypothesis-generating," not confirmatory.**
   State this in the paper itself, not just internally. The prospective foundation-model test
   becomes the paper's actual confirmatory evidence.

3. **Promote the confound controls from appendix to mandatory robustness arms.** Dimensionality-
   matching, multiple normalization schemes, multiple pooling choices, and LID-estimator
   sensitivity (all already named in `11_Confound_Map.md`) should be built into the core Step 6
   design, not added reactively after a reviewer asks. A reviewer cannot ask "did you check X" if
   X is already reported as a robustness table.

4. **Soften causal language to correlational/diagnostic language** ("consistent with," "predicted
   by") unless a synthetic toy demonstration of the causal pathway (training objective → geometry
   shift → estimator failure, under known ground truth) is added. Such a toy demonstration is a
   cheap, medium-priority addition that would most directly answer the mechanism objection above.

5. **Add an explicit power/range-diversity argument for Prediction 3.** Either report the achieved
   range of geometry values sampled in the FM arm and argue it is wide enough to detect a
   plausible architecture effect, or reframe the "no residual architecture effect" result as
   suggestive rather than conclusive.

These five changes do not require new literature or new mechanisms — they require disciplining the
claim to match what the confound map and prediction audit already show is actually testable.
