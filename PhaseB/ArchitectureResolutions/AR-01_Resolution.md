# AR-01 — Step 2: Resolution

**Answers AR-01's question from `AR-01_Architecture_Study.md`'s evidence. Chooses an intervention
*family* (or minimal combination of ingredients), not an exact architecture spec.** Dimensionality,
injection point, exact λ/scale parameterization, and adapter code remain Stage 4 *execution*
decisions — out of scope here, same boundary `Stage4_Architecture_Review.md` already drew. CSG-Lite
re-enters here only as a reference point (per `AR-01.md`'s own two-step design), not as a target to
reproduce.

## 1. Paper 4's constraints, grounded

Not invented here — each traces to an already-locked document.

| Constraint | Source | Requirement |
|---|---|---|
| Frozen PanDerm | Established since Stage 1; `02_stage1_setup.md` records the current state as "no adapter, no GRL, no orthogonality loss" | Backbone weights untouched; intervention adds parameters only, never modifies or requires access to backbone weights |
| Geometry is the mediator, not a byproduct | `Scientific_Hypothesis.md`; `22_Experimental_Design.md` §0 (the chain under test) | The adapter must act on (or produce) the *same* embedding space the geometry metrics and reliability scorers measure — not some internal signal invisible to downstream measurement |
| Gate 0 compatibility | `22_Experimental_Design.md` §3.5 | ID task balanced accuracy/ECE and domain-leakage probe must remain computable and meaningfully above trivial baseline after intervention — representation must not collapse |
| Gate 1 compatibility | `22_Experimental_Design.md` §3.5, quoted: "Did LID and/or spectral-decay slope actually change (not just condition number/scale)?" | Needs an identifiable, dial-able intervention target — a checkable "did the geometry actually move" signal, not just an opaque global transform |
| C2–C5 dose/intensity ladder | `22_Experimental_Design.md` §3 (2–3 Intervention-arm strengths mirroring λ_orth = 0,1,5) and §3 (Adaptation arm's progressive ladder) | The family must support smooth intensity variation via a scalar or small parameter set — not just an on/off toggle |
| Normalization-knob compatibility (C3) | `22_Experimental_Design.md` §5, the β-scaled ℓ2-normalization knob (arXiv:2510.15202) | Must apply cleanly to a well-defined, fixed-dimensional embedding vector — not a diffuse weight-space edit |
| Interpretability of the intervened direction | `22_Experimental_Design.md` §4 measurement 3: "per-direction variance of the intervened direction" | Needs an identifiable direction/subspace the intervention acts on, or per-direction variance has nothing to be computed *of* |

## 2. Family fit, against `AR-01_Architecture_Study.md`'s evidence

| Family | Fit |
|---|---|
| **LoRA** (Q/V projection updates, every attention layer) | Frozen-compatible; natural dose knob (rank *r* or scaling α); Gate 0 well-supported (established PEFT literature, e.g. Fairness-Aware LoRA, LoRA-BIQA). **Weak on interpretability**: effect is spread diffusely across many layers/heads — no single identifiable "intervened direction," which works against the per-direction-variance measurement Gate 1/C2 need. Present in the survey almost always *paired with* an explicit orthogonality term on top (Fairness-Aware LoRA: Frobenius-norm penalty between task-LoRA/sensitive-LoRA; AOFT: structural orthogonality on the LoRA matrices themselves) — LoRA supplies the *trainable capacity*, not the disentanglement mechanism by itself. |
| **Bottleneck / residual adapter** (in-block or single output-stage) | Frozen-compatible; dose knob = bottleneck width or an output-scaling coefficient; Gate 0 well-supported (AdaptFormer, CLIP-Adapter). **Interpretability depends on placement**: per-block (AdaptFormer-style) is as diffuse as LoRA; a *single, late-stage* bottleneck producing one low-dimensional output embedding is far more interpretable — structurally the closest single-branch analogue to CSG-Lite's own 16-d `z_lesion` bottleneck, without needing a second encoder to produce it. |
| **Orthogonal projection adapter** (closed-form null-space projection, orthogonal value decomposition, or an explicit orthogonality loss on a projected subspace) | Frozen-compatible, several variants training-free. **Best fit for Gate 1 and interpretability by construction** — the "intervened direction(s)" *are* the explicit target of the projection (Closed-Form Concept Erasure, Uni-AdaVD, Identity Sanitization Projection, O-TPT all measure or define geometry directly on the projected subspace — this family accounts for most of Tier 5's non-"No" rows). Dose knob = number of directions removed / projection strength — less standard than a scalar loss weight, but well-precedented (Identity Sanitization Projection's top-*k*). **Requires a way to identify the target subspace** — needs domain-labeled data to fit against, which Paper 4 already has (`ReferenceTrainEmbedding`'s `domain` column) — a genuine practical advantage, not a new dependency. |
| **Prompt tuning** (visual/input-space) | Frozen-compatible; well-supported Gate 0 (VPT). **Weakest fit on interpretability and geometry-mediator directness** — prompts modify the *input*, and the resulting geometry shift is real but emergent, mediated through the entire frozen transformer stack; hard to attribute to one identifiable direction. Dose knob (prompt depth/magnitude) is plausible but conceptually murkier as an intensity ladder than a loss weight. |
| **Feature projector** (post-hoc learned or closed-form transform on frozen output features, e.g. CLIP-Adapter/FairerCLIP-style) | Frozen-compatible; **good fit for the geometry-mediator constraint** — transforms the final embedding directly, so geometry shift is measured on exactly the space being probed. Interpretability good if the transform is simple/low-rank with a controllable regularization strength; weaker if implemented as an opaque deep MLP. Overlaps substantially with "orthogonal projection adapter" when the transform includes an explicit orthogonality/decorrelation term (FairerCLIP: RKHS closed-form; several others). |

## 3. What the evidence rules out, and what it doesn't decide

**Ruled out by Tier 3's own count**: a full second/auxiliary encoder (0/27) is not required to
instantiate an orthogonality-style mechanism — 13/27 papers achieve it through a single-backbone
architecture with an internal split, a projected subspace, or a paired low-rank adapter, never a
wholly separate network. Continuing to treat CSG-Lite's dual-encoder structure as necessary would
mean designing against evidence, not from it.

**Not decided by this evidence alone**: whether the disentanglement target should be (a) a learned,
adversarially-trained direction (GRL + classifier head reading an adapted embedding — no second
encoder needed, evidenced by FairNVT and DiCoOp) or (b) a closed-form/PCA-derived direction fit
directly from labeled domain data (evidenced by Identity Sanitization Projection, Closed-Form
Concept Erasure) — both are real, evidenced patterns in the survey that satisfy every constraint in
§1 without a second encoder, and choosing between them is an execution-level decision (what to fit,
on what data, with what stability), not a family-level one this document is scoped to settle.

## 4. Resolution

**A single-backbone, output-stage intervention — a low-dimensional bottleneck/feature adapter (or
equivalently, a projected subspace) carrying an explicit orthogonality-style constraint — best fits
Paper 4's constraint set, without reproducing CSG-Lite's dual-encoder structure.** This is not one
specific architecture; it names the family-level shape the evidence supports:

- **Trainable capacity**: a bottleneck/feature adapter or low-rank (LoRA-style) update, placed to
  produce one identifiable, fixed-dimensional output embedding — not scattered uniformly across
  every block, so that a single "intervened direction" exists for Gate 1 and per-direction variance
  to act on.
- **Disentanglement mechanism**: an explicit orthogonality-style constraint (cosine-similarity loss,
  Frobenius-norm penalty, or closed-form/null-space projection — all evidenced, all
  single-backbone-compatible) between the adapted embedding and an identified nuisance direction —
  not a second encoder's output.
- **Dose/intensity ladder**: a scalar (loss weight, projection rank, or adapter scaling) — every
  evidenced variant in this shape supports one.

**Not resolved here, deliberately**: which specific mechanism produces the nuisance direction
(learned-adversarial vs. closed-form/data-derived), exact adapter dimensionality and injection
point, exact λ/scale values, and how this maps onto Intervention-arm strengths vs. the Conventional/
Adaptation arms' own ladders. These are Stage 4 execution decisions, to be worked out (and logged)
only once this resolution is reviewed and Stage 4 execution is separately authorized —
`Stage4_Execution_Plan.md`'s pipeline (Train → Embedding extraction → Gate 0 → Gate 1 → Execution
Review → Sanity → Freeze) applies unchanged once that happens.

## What this resolution does not do

No architecture spec ready for code. No adapter dimensionality, injection point, or hyperparameter
chosen. No training. AR-01 itself is not closed by this document — it moves from Open/Step-1-done to
**Open/Step-2-resolved, awaiting Vinh's review** before Stage 4 execution can be authorized.
