# Group B Protocol Reconciliation Note

**Status: AUDIT ONLY. No embeddings generated. No artifact modified — `ReferenceEmbedding/` was not
touched; `BaselineReport/` was not touched beyond the Tier-2 wording correction logged separately.**
Produced per Vinh's PI instruction (2026-08-05): "Prepare a protocol reconciliation note that traces
every Group B scorer back to its required fitting population (train / reference / held-out), compare
that requirement against the currently frozen ReferenceEmbedding artifact, and determine whether the
missing ISIC-train representation is a protocol omission or an intentional exclusion."

---

## 1. What triggered this note

While executing Stage 3.2 (`stage3_2_reference_arm_baseline.py`), Group A (domain-leakage probe,
geometry metrics) ran cleanly on `ReferenceEmbedding/`'s existing held-out ID(ISIC-test,
n=5,067)+OOD(PAD-UFES, n=2,298) population — no separate training population needed, since both
measurements fit/evaluate via an internal split of that same held-out data. Attempting to plan Group
B (distance-based reliability scorers) surfaced that this is **not** true of every scorer. This note
traces exactly what each one needs, rather than assuming they all fail the same way.

## 2. Scorer-by-scorer trace (every claim below cites the exact source read, not inferred)

| Scorer | Needs a separate **fitting/training population**? | Needs a **trained classifier head** (logits)? | Evidence (file, function) |
|---|---|---|---|
| **Mahalanobis** (primary) | **Yes** — `compute_mahalanobis_params_from_arrays(z_train, y_train, ...)` fits per-class means + shared covariance on a population distinct from the scored `z_id`/`z_ood` | No | `CSG-SKin/src/utils/ood_metrics.py`; called with `z_train` in `paper-3/scripts/extract_auroc_e2.py::compute_mahalanobis_scores` |
| **kNN** | **Yes** — `NearestNeighbors(n_neighbors=k_max).fit(z_train)`; the reference pool must be the full training population for k-th-neighbor distances to be meaningful | No | `paper-3/scripts/extract_auroc_e2.py::compute_knn_scores` |
| **Cosine-to-centroid** | **Yes** — uses `means`, the *same* per-class means array already fit for Mahalanobis (i.e. fit on `z_train`, not on the query set, to avoid circularity) | No | `paper-3/scripts/extract_auroc_e2.py::cosine_centroid_scores` |
| **Energy** | No | **Yes** — score is computed from reconstructed logits: `z @ W.T + b`, where `W`/`b` are `model.lesion_classifier.weight`/`.bias` read directly out of a *trained* checkpoint's state_dict | `paper-3/scripts/extract_e2_8_extra_scorers.py::energy_scores`, `reconstruct_logits` |
| **ViM** | **Yes** — `fit_vim(train_feats, train_logits, fc_weight, fc_bias, ...)` fits a null-space projection on `train_feats`/`train_logits` | **Yes** — needs `fc_weight`/`fc_bias` (a trained head) both to fit and to score (`vim_score` computes `logits = feats @ w.T + b` at query time too) | `DST-Skin/src/utils/ood_vim_react.py::fit_vim`, `vim_score` |
| **KDE** (`density_kde`) | **Yes** — per-class `KernelDensity(...).fit(z_c)` where `z_c` is drawn from `z_train`, not the query set | No | `paper-3/scripts/extract_e2_8_extra_scorers.py::density_kde_scores` |
| **Relative Mahalanobis** | **No implementation found anywhere in Papers 1–3's codebase** (`paper-3/`, `CSG-SKin/`, `DST-Skin/` all searched — zero matches for "relative mahalanobis," "RMDS," or similar). It is named in `22_Experimental_Design.md` §4 measurement 2 as a secondary scorer, but "reused from Paper 3's implementation" is **not accurate** for this one specifically — there is nothing to reuse. The standard formulation (Ren et al. 2021) would also need a training-fit background model, by construction, if implemented. | Would depend on implementation | (absence confirmed by repo-wide search, not by reading a specific file) |

**Summary of what's actually missing, precisely:**

- **Six of seven** Group B scorers need a **training/reference population** distinct from the
  held-out ID/OOD split (Mahalanobis, kNN, cosine-to-centroid, ViM, KDE, and — if it existed —
  relative Mahalanobis).
- **Two** of seven need a **trained classifier head** for logits (Energy, ViM) — a *different* kind
  of missing artifact, unrelated to which images were extracted.
- **One** (relative Mahalanobis) has no implementation to reuse at all, independent of data
  availability.

ViM is blocked by two independent causes simultaneously, not one.

## 3. What `ReferenceEmbedding/` actually contains

`assets/reference_embeddings/ReferenceEmbedding/` (frozen, Stage 2.1): embeddings for the **held-out
evaluation split only** — ID = ISIC-test (n=5,067), OOD = PAD-UFES (n=2,298). No ISIC-train
embeddings. No trained classification head exists for the Reference Instrument at all — PanDerm is
used purely via `forward_features()` (Stage 1.2/1.3's established convention); the `head` layer from
`VisionTransformer`'s `num_classes=1000` default is untouched, uninitialized-for-this-task, and
deliberately never evaluated (this is also why MSP was excluded from Group B in the execution plan).

**Consequence:** the training-population gap (§2, six scorers) *could* in principle be closed by
extracting ISIC-train embeddings (a new, not-yet-authorized extraction). The classifier-head gap
(Energy, ViM) **cannot** be closed by any embedding extraction alone — it requires deciding whether
the Reference arm ever gets a trained linear head for the ID task, and if so, training one (a
different kind of step than "extract more embeddings").

## 4. Was the missing ISIC-train population a protocol omission or an intentional exclusion?

**Corrected 2026-08-05, per Vinh:** the analysis below originally concluded "protocol omission."
That conflated two different things. **The protocol document itself is not wrong and is not being
amended** — re-read carefully, `22_Experimental_Design.md` §2 simply doesn't state one way or the
other whether the Reference arm specifically needs the train split; it names a train split for "the
FM arm" in general without resolving this case. That's not an error in the protocol, it's a
underspecified case. **What actually happened is that Stage 2.1's *execution* scope — the concrete
decision of what to extract — resolved that silence one way (held-out only), and that resolution
turned out to be incomplete once Group B's requirements were traced.** So: the protocol stays
exactly as written, untouched, not reopened. The gap lives in the **execution specification**
(Stage 2.1's scope), not in the preregistered design. See `ExecutionAmendments/EA-01.md` for the
correction at the right level — an execution amendment, not a protocol deviation.

**The preregistered protocol text itself already specifies a train split for the FM arms.**
`22_Experimental_Design.md` §2 (Datasets), quoted exactly:

> "FM arm (Stages 1–3): ISIC 2019 (**train split**) + PAD-UFES-20 (train split) for
> representation/adaptation; held-out splits of both for evaluation, matching Papers 1–3's train/eval
> protocol exactly."

This is not ambiguous about *whether* a train split belongs in scope — it explicitly names one. What
is ambiguous is *purpose*: the text says the train split is "for representation/adaptation," which
reads most naturally as serving the **Intervention arm's actual training process** (GRL/orthogonality
fine-tuning needs training data to train on). The Reference arm, by definition, does no
adaptation — so Stage 2.1's scope (as authorized: "the complete preregistered ID and OOD datasets,"
resolved at the time to mean the held-out evaluation split) did not extract a train split, and that
reading is defensible **for the "representation/adaptation" purpose specifically**.

**But Group A already shows there's a second, distinct purpose the train split serves that has
nothing to do with adaptation: fitting a reliability estimator.** Mahalanobis, kNN, cosine-to-centroid,
ViM, and KDE all need *some* population to define "what normal looks like" against which a query
point's distance is measured — this is true even for a **frozen, unadapted** Reference arm, precisely
because these are the measurement instruments Paper 3's own Reference-arm-equivalent baseline
(λ_orth=0, i.e. `runA_grl`) was scored against, using its own ISIC-train population.

**Reading the evidence together:** the protocol document already names the train split as in-scope
for "the FM arm" generally (not scoped only to the Intervention arm), and Paper 3's own precedent
(which this project is explicitly matching) fits every Group B scorer on a training population even
for its least-intervened checkpoint. The narrower interpretation that produced Stage 2.1's actual
scope (train split needed only for "adaptation," therefore not needed by an arm that doesn't adapt)
does not hold once Group B's specific measurement requirements are traced individually, as this note
does.

**This is an execution-specification omission, not a protocol omission and not an intentional
exclusion.** Precisely:

- **Protocol** (`22_Experimental_Design.md`, `24_PreRegistration.md`, `Scientific_Hypothesis.md`,
  `02_Claim_Matrix.md`): **unchanged, not at fault.** Nothing here contradicts or requires editing
  the preregistered design — it was silent on this specific case, not wrong.
- **Execution specification** (Stage 2.1's actual extraction scope): **incomplete** — it should have
  included an ISIC-train extraction for the Reference arm, alongside the held-out split, so that
  Group B's fitting requirement could be satisfied without a later gap.
- **Resolution path:** an **Execution Amendment** (`ExecutionAmendments/EA-01.md`) — a category
  distinct from `24_PreRegistration.md` §13's Deviations Log, since nothing preregistered is being
  changed. No protocol deviation entry is needed for this.

## 5. What this note does not do

Does not extract any embeddings. Does not fit any scorer. Does not modify `ReferenceEmbedding/`,
`BaselineReport/`, or any other frozen artifact. Does not decide the omission-vs-exclusion question —
lays out the traced evidence for Vinh's decision, per instruction.
