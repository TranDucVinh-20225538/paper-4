# Open Questions — Paper 4

**Status: GAP CONFIRMED at Level A (revised 2026-08-04, session 5).** All three source claims
below are now primary-source verified. Point (i) (ShortKit-ML) and point (iii) (Müller & Berens,
partial) were already Level A. Point (ii) (Gallo et al.) is now also **Level A** — the user
uploaded the CVF PDF and it was read in full; "MDS outperforms logit-space scores for dermatology
OOD detection across FM backbones and fine-tuning regimes" is now a confirmed fact, not an
unverified secondary-source claim. The gap statement below is now a closed, verified conclusion
rather than a provisional best-guess.

## Confirmed research gap (2026-08-04, session 5 — Gallo et al. verified)

Medical-FM studies show (i) that frozen/pretrained embeddings can encode demographic shortcuts
(ShortKit-ML — **Level A verified**) and (ii) that Mahalanobis/MDS outperforms logit scores for
dermatology OOD detection across pretraining strategies and fine-tuning regimes (Gallo et al. —
**Level A verified, full text read 2026-08-04**). Medical disentanglement benchmarks show that
representation-level mitigation can change latent structure (Müller & Berens — **Level A verified
for ~56% of the paper; the paper does not evaluate any OOD/reliability score in the portion
read**). **No identified study joins these three elements, and this is now a stronger claim than
before:** Gallo et al. — the paper closest to our setting on dataset, modality, and reliability
metric — was read in full and confirmed to contain no domain-disentanglement intervention, no
quantitative geometry analysis, and no linear-probe decodability test. The "no identified study"
claim is no longer just "our search didn't find one" — it is "we read the single most likely
candidate cover-to-cover and it isn't there."

The scientific gap is therefore not "do FMs have shortcuts?" or "does Mahalanobis beat logits on
FMs?" (both now answered by prior work, confirmed). It is whether an explicit shortcut-mitigation
intervention makes the same information *less accessible to unsupervised distance-based
estimators without making it linearly undecodable* — and whether this remains true in
high-dimensional medical-FM embeddings. That question remains open and is now on firmer novelty
footing than at any prior point in this project.

## Candidate directions assessed

1. **Domain-shortcut leakage in medical-FM representations.** Does a frozen or lightly-adapted
   medical foundation-model embedding (e.g., PanDerm, MONET, DINOv2-style derm SSL) leak
   acquisition-source identity the way the supervised CNN baseline does in Paper 1 (probe accuracy
   ~0.98)? Is the magnitude comparable, larger, or smaller? This is useful as a baseline, but it
   is not itself novel enough as a main claim: ShortKit-ML already establishes shortcut/
   demographic decodability across FM families in chest X-ray.

2. **Does disentanglement/debiasing training on an FM reproduce Paper 3's dissociation?** If a
   medical FM is fine-tuned with an adversarial domain-disentanglement objective (or similar
   debiasing intervention), does representation geometry shift substantially while
   distance-based reliability estimators (Mahalanobis etc.) stay flat/uninformative, even though
   the shortcut remains linearly probe-decodable — i.e., does the "decodable but not accessible"
   phenomenon replicate outside the CNN/GRL setting Paper 3 used? This is the most novelty-protected
   candidate and is now the selected direction.

3. **Does the CNN-paradigm reliability-scoring hierarchy (feature-space > logit-space) hold for
   FMs specifically under a disentanglement/debiasing intervention**, as opposed to just under
   fine-tuning-intensity variation (which Gallo et al. **confirms**, Level A)? This reframes
   away from the part of the RQ that is now settled by prior work.

## Resolved by the Gallo full-text read (session 5 — see `03_Novelty_Threats.md`, THREAT 1)
- "Does feature-space OOD detection beat logit-space OOD detection on foundation models in
  dermatology?" — **confirmed yes, Level A, by Gallo et al.** This is now closed, not open. It
  must not be presented as a Paper 4 finding, main or secondary — it would be a confirmed
  redundancy, not a risk. Background citation only.

## Boundary conditions for the claim

- Claim an intervention-specific dermatology-FM mechanism; do not claim first FM shortcut
  leakage, first FM OOD detection (both now confirmed pre-existing, Level A), or universal
  behavior across FMs.
- Compare frozen/lightly adapted and debiased checkpoints, with matched ordinary fine-tuning
  controls; adaptation intensity alone is a **Level-A-confirmed competing explanation** (Gallo et
  al.'s central finding), so this control arm is now a required design element, not a hedge.
- Treat the predicted dissociation as falsifiable, not assumed: all of probe accuracy, geometry,
  and reliability metrics can move independently.
