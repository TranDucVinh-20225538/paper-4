# Prediction Audit — Paper 4

Research Gate remains CLOSED. No methods proposed here. Source material: `Scientific_Hypothesis.md`,
`08_Mechanism_Map.md`, `02_Claim_Matrix.md` only. For each of the three predictions, this audits
what would support it, what would falsify it, what else could produce the same observation, and
what would distinguish the two.

---

## Prediction 1

*The magnitude of distance-based-estimator degradation under a suppression-style intervention
should co-vary with measurable shifts in LID and/or within-class spectral-decay slope of the
intervened direction — not simply with a scalar "intervention strength" label, and not with
probe accuracy.*

**1. Supporting observation:** Across the disentanglement ladder, degradation in Mahalanobis-AUROC
tracks LID × spectral-decay-slope more tightly than it tracks the raw intervention-strength
parameter (λ_orth) itself, while probe accuracy stays comparatively decoupled from both.

**2. Falsifying observation:** Mahalanobis-AUROC degrades monotonically with λ_orth even when
LID × spectral-decay stays roughly constant across the ladder — i.e., degradation tracks the
intervention label better than it tracks the measured geometry. This would mean intervention
strength affects distance estimation through some pathway the chosen geometry metrics don't
capture.

**3. Alternative explanation for the same observation:** λ_orth is a single knob that plausibly
moves *many* properties of the representation simultaneously — LID, spectral decay, class-mean
separation, feature norm, condition number (Paper 3's own metric). A correlation between
LID×spectral-decay and Mahalanobis degradation could arise purely because both are downstream of
the same common cause (λ_orth), with no mediating causal role for geometry at all. This is a
classic confounding-by-common-cause problem, not evidence of mediation.

**4. Distinguishing control:** A design that manipulates geometry *independently* of nominal
intervention strength — e.g., applying the β-scaled ℓ2-normalization knob from the primary
mechanism source (arXiv:2510.15202) post hoc, at a *fixed* λ_orth, to shift LID/spectral-decay
without retraining — would let us check whether Mahalanobis-AUROC moves with geometry even when
intervention strength is held constant. Statistically, a formal mediation analysis (does λ_orth's
effect on Mahalanobis-AUROC vanish after conditioning on the geometry metrics?) is the
non-experimental version of the same check.

---

## Prediction 2

*If a geometry metric is held approximately constant across interventions of different nominal
strength, distance-based-estimator degradation should also stay approximately constant. An
intervention engineered to suppress salience without shifting LID/spectral-decay should not
reproduce the collapse.*

**1. Supporting observation:** Two conditions with different λ_orth but matched LID×spectral-decay
(via the normalization knob, or naturally occurring in the ladder) show similar Mahalanobis-AUROC.

**2. Falsifying observation:** Two conditions with matched geometry metrics still show materially
different Mahalanobis-AUROC — meaning the chosen geometric summary is incomplete or wrong, or
intervention strength has a direct effect the summary doesn't capture.

**3. Alternative explanation for the same observation:** Distance-estimator degradation could be
driven by **between-domain mean separation** (how far apart the two domain centroids sit in
representation space) rather than by within-class local geometry (LID, spectral decay) at all.
Aggressive disentanglement plausibly pulls domain-conditional means closer together *and* changes
local manifold shape at the same time — these are conceptually distinct geometric quantities
(between-class vs. within-class structure) that would be confounded in the natural ladder. A
"clean" salience-suppressing-without-geometry-shift intervention may also be difficult to
construct at all: any real training change to a subspace's loss weighting is likely to alter its
local density/dimensionality as a side effect, making this prediction hard to test cleanly rather
than simply false — a feasibility risk worth flagging on its own.

**4. Distinguishing control:** Measure and statistically separate between-domain mean separation
from LID × spectral-decay across every condition (ladder + normalization-knob variants), and
enter both as covariates in predicting Mahalanobis-AUROC to see which one actually carries the
explanatory weight.

---

## Prediction 3

*Probe-decodability and distance-estimator accessibility should be conditionally independent
given the geometry metrics — once geometry is known, whether the representation came from a CNN
or an FM should add no further predictive power for distance-estimator failure.*

**1. Supporting observation:** A joint model across CNN (Paper 3, retrospective) and FM
(prospective) data points shows that, after regressing Mahalanobis-AUROC on the geometry metrics,
adding an architecture indicator variable does not significantly improve the fit.

**2. Falsifying observation:** Architecture remains a significant predictor of Mahalanobis-AUROC
even after conditioning on measured geometry — CNN and FM representations with matched geometry
values still show systematically different distance-estimator behavior.

**3. Alternative explanation for apparent independence (i.e., a way to look confirmed while being
wrong):** If the chosen FM checkpoint happens to occupy a narrow, CNN-like range of geometry
values — e.g., because of shared normalization conventions or similar effective dimensionality —
"no residual architecture effect" could reflect **insufficient geometric diversity sampled**, not
a genuine architecture-independence finding. A null result from a narrow-range comparison is not
distinguishable, from the data alone, from a null result from a true absence of effect.

**4. Distinguishing control:** Deliberately sample or engineer a wide range of geometry values
within the FM arm — via the normalization knob, multiple candidate layers/pooling choices, and
if feasible more than one FM backbone with a different pretraining objective — so there is enough
geometric variance to actually detect a residual architecture effect if one exists, rather than
relying on whatever geometry the intervention ladder happens to produce by default.

---

## Cross-cutting note

All three predictions share the same structural vulnerability: **λ_orth (or "architecture") is a
confound for any geometry metric that is itself downstream of the same training/pretraining
process.** None of the three predictions can be cleanly tested by observing the natural ladder
alone — each requires some way of moving geometry *independently* of its usual cause (intervention
strength, or architecture) to distinguish mediation from mere co-occurrence. This is a design
requirement to carry into Step 6, not a reason to abandon the hypothesis — it is exactly what
`09_Risk_Register.md` and `11_Confound_Map.md` (next) are for.
