# Stage 4 — Intervention Architecture Gap

**Investigation note. No code written. No training executed. No architecture decision made.**
Produced before attempting Step 2 ("Train adapter checkpoint") of `Stage4_Execution_Plan.md` §1,
under Vinh's authorization (2026-08-05) of the first Intervention-arm checkpoint (lowest strength).
This project's standing rule, established at EA-01: trace and report a gap precisely before writing
any code that would silently resolve it. This note does the same thing for a gap found one step
earlier than EA-01's — before training can start at all, not before a scorer can run.

## What was authorized

Vinh, verbatim: *"Checkpoint rung đầu tiên (Intervention, strength thấp nhất), kèm đúng pipeline đã
khóa: Train → Embedding extraction → Gate 0 → Gate 1 → Execution Review → Sanity → Freeze."*
`Stage4_Architecture_Review.md` and `Stage4_Execution_Plan.md` were declared FINAL in the same
decision — this note does not touch either file.

## What "Train adapter checkpoint" requires, per protocol

`22_Experimental_Design.md` §3, the only description of the Intervention arm's mechanism anywhere
in the locked protocol: *"Adapter-based domain-adversarial training (GRL + orthogonality), 2–3
strengths mirroring Paper 3's λ_orth = 0, 1, 5 as closely as an adapter allows."* This is a
conceptual sketch, not an implementation spec — "as closely as an adapter allows" already
anticipates some adaptation will be needed. `Stage4_Architecture_Review.md`'s own closing note
already flagged "exact adapter architecture" as an unresolved execution-level decision, out of
scope for that review.

## What CSG-Skin's actual GRL + orthogonality implementation is (investigated, not modified)

Read in full, not from memory: `CSG-SKin/src/models/csg_lite.py`, `csg_lightning.py`,
`src/losses/csg_losses.py`, `scripts/train_csg.py`, `scripts/run_full_benchmark.sh`.

- **`CSGLite` is a dual-encoder architecture** — two *separate* EfficientNet backbones with no
  shared trunk: a `context_backbone` (RGB, surrounding-skin crop) and a `lesion_backbone`
  (grayscale-3ch, lesion crop). Not a single encoder with a branch — two independently-trained CNNs.
- **GRL**: standard Ganin-Lempitsky autograd trick (identity forward, negated/scaled gradient
  backward), `lambd` scheduled via a sigmoid ramp (`γ=20`, floor `0.2`) — separate from λ_orth.
- **Orthogonality term is a latent-space cosine-alignment penalty, not weight orthogonality**:
  `z_context` (64-d, from the *context* backbone) is linearly projected to 16-d
  (`project_context_for_orth`, no bias) and penalized via
  `mean(cosine_similarity(z_lesion, z_context_proj)^2)` — i.e. it decorrelates the lesion embedding
  from a linear image of a *second, independently-encoded* context embedding.
- **Domain-adversarial head** branches off `z_lesion` (16-d, BN-normalized) through the GRL, an MLP
  (`Linear→BN→ReLU→Dropout→Linear→BN→ReLU→Dropout→Linear→2`).
- **Combined loss**: `L = L_cls + λ_ctx·L_ctx + λ_adv·L_adv + λ_orth·L_orth + λ_supcon·L_supcon`.
  Confirmed λ_orth values matching Paper 4's cited ladder: `runA_grl`→0.0, `runB_orth1`→1.0,
  `runB`→5.0 (`run_full_benchmark.sh`, `run_one_shot_cbm.sh`) — `train_csg.py`'s own hardcoded
  default (`LAMBDA_ORTH=10.0`) is overridden in the actual benchmark runs, not what was used.
  `z_lesion` is fixed at 16-d, `z_context` at 64-d, throughout.
- **Optimizer**: AdamW, three param groups with differential LR multipliers (adversarial head at
  30× base LR, lesion classifier at 0.2×). Base LR 2e-4, batch size 32, 20 epochs (`train_csg.py`
  active defaults).

## The gap

Confirmed by a repo-wide search (CSG-SKin, DST-Skin, paper-3, Paper4): **no adapter-based
implementation of this technique exists anywhere**, and applying it to Paper 4's Reference
Instrument (a single frozen PanDerm ViT-L/16 with no dual-encoder structure, 1024-d, adapter not
yet designed) is not a direct port — three specific things have no defined equivalent yet:

1. **The orthogonality target.** CSG-Skin's term needs a *second, independently-encoded* context
   embedding, produced by a wholly separate CNN branch trained on a separate image crop. Paper 4's
   design has one frozen backbone and one (not-yet-designed) adapter — no second encoder, no
   context-image-crop pipeline established anywhere in this project's dataset handling. What plays
   the role of "context embedding" here has never been specified.
2. **Adapter architecture and dimensionality.** CSG's 16-d `z_lesion` is the output of a fully
   fine-tuned, dedicated CNN branch. Nothing in the protocol specifies what dimensionality,
   placement, or injection point an adapter on a frozen 1024-d PanDerm backbone should use to
   produce an analogous embedding.
3. **λ_orth / GRL calibration for a different capacity regime.** CSG's λ=0,1,5 were calibrated for
   full end-to-end CNN training with its own parameter budget. Nothing establishes these values
   transfer meaningfully to a small adapter's much smaller parameter budget — `23_Experiment_Contract.md`
   independently flags an adapter-capacity "sanity check" as a requirement, not yet performed.

This is not an implementation detail to fill in while writing code — it is a design decision with
direct scientific consequences for what the Intervention arm actually measures (what, precisely, is
being made orthogonal to what). Choosing an answer without an explicit PI decision would mean
introducing a new experimental design element unilaterally — exactly what the Constitution and
every prior step in this project have required routing through an explicit decision instead of
silent resolution.

## What has NOT been done

No code written. No adapter architecture chosen. No context-embedding substitute chosen. No λ_orth
or GRL hyperparameter chosen for the adapter setting. No training run started. Nothing computed.

## Open question this note does not answer

Whether resolving this is an **Execution Amendment** (the protocol's own text already says "as
closely as an adapter allows," so a concrete adapter design could be read as filling in that
allowance) or reaches far enough into the experimental design to require a **Protocol Deviation**
(if the orthogonality mechanism itself needs to be redefined, not just adapted) is itself unresolved
here — that classification is Vinh's call, per the Constitution/Protocol/Execution
Specification/Implementation four-layer distinction this project has used since EA-01, not
something this note decides.
