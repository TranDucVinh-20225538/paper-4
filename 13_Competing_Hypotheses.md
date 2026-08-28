# Competing Hypotheses — Paper 4

Forget the geometry hypothesis for this document. Three alternative, scientifically plausible
mechanisms for the same observation (Paper 3: probe succeeds, 8 distance/density-based scorers
fail, under a disentanglement dose-response ladder). Each must explain Paper 3, be consistent with
current literature, and make different predictions than the geometry hypothesis. No winner is
chosen here.

---

## Alternative 1 — Distributional-family mismatch ("wrong parametric assumption," not geometry per se)

**Claim:** Mahalanobis assumes (near-)Gaussian, typically single- or per-class, structure. If
disentanglement training produces a non-Gaussian distribution along the domain axis — multimodal,
skewed, or heavy-tailed, rather than merely shifted in local intrinsic dimensionality or spectral
decay — then any Gaussian-parametric estimator fails because the *distributional family*
assumption is violated, independent of the two specific scalar geometry summaries (LID,
spectral-decay slope) our hypothesis picks out. Under this account, the right diagnostic would be
multimodality/skewness/kurtosis metrics, not LID or spectral decay.

**Does it explain Paper 3?** Partially, but with a real tension worth flagging: Paper 3 already
tested a kernel density estimator (KDE) — a non-parametric method that does *not* assume
Gaussianity — as one of its 8 scorers, and KDE also failed, converging with the other 6 non-probe
methods on the same null result. If the failure were purely a Gaussian-parametric-assumption
problem, a non-parametric method like KDE should have been comparatively robust to it. That it
wasn't is evidence *against* a pure distributional-family account, though a proponent could argue
KDE's own bandwidth/density estimates are themselves geometry-sensitive (high-LID regions make
density estimation unreliable too) — which would partly collapse this alternative back into the
geometry hypothesis rather than keeping it fully distinct.

**Consistent with literature?** Yes — Gaussian-assumption violation is a standard, well-established
critique of Mahalanobis-family methods generally.

**Distinguishing prediction:** Multimodality/skewness metrics of the domain-relevant subspace
should predict Mahalanobis degradation better than LID × spectral-decay does. If both track
degradation about equally well, the two accounts are hard to separate with this data; if
skewness/multimodality metrics predict *nothing* (flat across the ladder) while LID/spectral-decay
does, that would favor the geometry hypothesis over this alternative.

---

## Alternative 2 — Entangled/distributed encoding ("linear-combination accessibility," not geometry)

**Claim:** Domain information may not be "suppressed but present along one direction" at all.
Disentanglement training could instead push the domain signal into a distributed, entangled
encoding — spread across many dimensions in a way that requires a full-rank *supervised* linear
map (which can learn an arbitrary weighted combination using labels) to recover, but is invisible
to any *unsupervised* method that looks for marginal or low-dimensional structure (density peaks,
class-conditional clusters, a dominant covariance direction). Under this account, the
distinguishing factor is not local density/dimensionality geometry at all, but the *encoding
scheme* — concentrated-and-suppressed (our hypothesis) vs. distributed-and-entangled (this
alternative).

**Does it explain Paper 3?** Yes, cleanly — it directly explains why a supervised probe (0.72–0.81
AUROC) succeeds while 8 distinct unsupervised/marginal methods (spanning distance, density, and
energy-based principles) all fail: none of them are designed to find an entangled, multi-dimension
combination, while the probe is built precisely to do so.

**Consistent with literature?** Yes — distributed/entangled representations are a well-documented
phenomenon in representation learning generally (e.g., superposition in neural network features),
independent of any adversarial-training-specific mechanism.

**Distinguishing prediction:** The probe's learned weight vector should be diffuse (mass spread
across many of the 16 dimensions) rather than concentrated on one or two directions, under this
alternative. Under the geometry hypothesis, no particular claim is made about probe-weight
concentration — the two accounts diverge sharply on this observable, making it a clean
distinguishing test that is directly computable from data Paper 3 likely already has (the probe
weights themselves).

---

## Alternative 3 — Finite-sample estimation artifact ("statistical power," not structural geometry)

**Claim:** As disentanglement strength increases, the domain-relevant direction's variance may
shrink (per Mechanism B in `08_Mechanism_Map.md`). A shrunk-variance direction is intrinsically
harder to estimate reliably from a finite sample when fitting a covariance matrix — Mahalanobis
could fail simply because the *effective signal-to-noise ratio for estimating that direction's
contribution* drops with disentanglement strength, not because the true population-level geometry
structurally defeats distance-based methods. Under this account, the failure is a curable
estimation problem, not a structural one.

**Does it explain Paper 3?** Yes, in outline — it predicts the same qualitative pattern (distance
methods fail, probe succeeds) because probes are comparatively sample-efficient relative to
full covariance estimation, especially in the smaller PAD-UFES-20 domain (~2.3k images).

**Consistent with literature?** Yes — the sensitivity of covariance-based methods to sample size
and shrinkage is well established (this is exactly why Paper 2 already uses Ledoit–Wolf shrinkage).

**Distinguishing prediction:** This is the account's sharpest divergence from the geometry
hypothesis: it predicts Mahalanobis performance should **recover with more in-distribution samples
used to fit the covariance, at a fixed checkpoint/geometry**. The geometry hypothesis predicts no
such recovery, because LID and spectral decay are treated as population-level properties, not
estimation noise (though both are, in practice, estimated from finite samples too — a
complication, not a clean escape, for either side). A subsampling-vs-full-sample comparison at
fixed disentanglement strength is the direct test.

---

## Comparison to the geometry hypothesis (no winner declared)

| | Explains Paper 3? | Consistent with literature? | Distinguishing prediction vs. geometry hypothesis |
|---|---|---|---|
| **Our hypothesis (geometry mediates accessibility)** | Yes | Yes (arXiv:2510.15202, Level A) | LID × spectral-decay slope tracks degradation; architecture adds no residual predictive power once geometry is known |
| **Alt 1 — distributional-family mismatch** | Partially — in tension with KDE's failure in Paper 3's own existing data | Yes | Multimodality/skewness should predict degradation better than LID/spectral-decay |
| **Alt 2 — entangled/distributed encoding** | Yes, cleanly | Yes | Probe weights should be diffuse across dimensions, not concentrated |
| **Alt 3 — finite-sample estimation artifact** | Yes, in outline | Yes | Mahalanobis-AUROC should recover with more ID samples at fixed geometry; the geometry hypothesis predicts no such recovery |

All three alternatives are testable with observables already implied by the existing experimental
setup (probe weight vectors, distributional-shape metrics, sample-size sweeps) — none requires a
new architecture or dataset to distinguish from the geometry hypothesis. That is worth carrying
into Step 6: several of these distinguishing tests are cheap enough to run alongside, rather than
instead of, the main geometry analysis.
