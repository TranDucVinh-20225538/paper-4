# Stage 3.1 — Artifact Characterization

PI instruction, verbatim (2026-08-05): "Characterize the frozen Reference Embedding Artifact only.
Compute descriptive properties (shape, dtype, norm distribution, variance statistics, basic
integrity summaries) and document them. Do not compute any quantities that test Claim C1–C5,
including Mahalanobis, probe accuracy, LID, spectral decay, Energy, ViM, KDE, AUROC, or
hypothesis-related statistics."

## Result (executed 2026-08-05)

| Property | Value |
|---|---|
| Shape | (7365, 1024) |
| dtype | float32 |
| Size | 28.77 MB |
| Norm (pooled) | mean=23.077, std=0.858, range=[20.698, 26.018] |
| Norm (ISIC, descriptive) | mean=22.619 |
| Norm (PAD-UFES, descriptive) | mean=24.086 |
| Per-dim variance | mean=0.0879, total (sum of per-dim)=90.05 |
| Class counts | AK=903, BCC=1510, BKL=760, DF=48, MEL=956, NV=2819, SCC=318, VASC=51 |

Norm histogram (pooled + per-domain, purely descriptive): `assets/ArtifactCharacterization/norm_histogram.png`.

**Observation, explicitly not interpreted here:** the ISIC and PAD-UFES norm distributions are
visibly offset in the histogram. This is reported as a descriptive fact only — no test statistic,
separability score, or significance claim was computed, and no interpretation of what it means for
the hypothesis is made at this stage. Any such interpretation belongs to a properly authorized
Claim C1+ analysis (Stage 3.2 onward), not to Artifact Characterization.

## Known-issues note

Per Stage 3.0's Issue #1 (`assets/ArtifactIssuesRegister/`): 34/7365 rows (~0.46%) are pixel-identical
duplicate pairs, included as-is (no action taken). Their contribution to the statistics above is a
small, symmetric double-count — noted for completeness, not asserted as negligible without this note.

## What was NOT computed

No Mahalanobis, no probe accuracy, no LID, no spectral-decay slope, no Energy, no ViM, no KDE, no
AUROC, no hypothesis-testing statistic of any kind. Per-domain breakdowns are descriptive
counts/means/percentiles only.

## Artifacts

- `assets/ArtifactCharacterization/characterization_report.json` — full descriptive statistics.
- `assets/ArtifactCharacterization/norm_histogram.png` — the figure above.
- `assets/ArtifactCharacterization/characterization_manifest.json` — scope-boundary manifest (added
  2026-08-05, per Vinh, after the directory's first freeze — a logged one-time unfreeze/amend/
  re-freeze, not a silent edit; see `26_Lab_Notebook.md`): references `ReferenceEmbedding/`'s exact
  file hashes, states `contains_only_descriptive_statistics: true`, `claims_tested: []`, and an
  explicit boundary statement so this artifact is never mistaken for Stage 3.2 analysis.
- `assets/ArtifactCharacterization/sha256.txt` — checksums (recomputed to cover all three files
  above after the manifest was added).
- Frozen: files `chmod 444`, directory `chmod 555`.
- Script: `analysis/stage3_1_artifact_characterization.py`. Log: `logs/stage3_1_run_2026-08-05.log`.

## Not yet authorized

Stage 3.2 (Claim C1) and beyond — each requires its own PI decision, per the pattern held since
Stage 1. Per the claim-discipline rule in `00_execution_checklist.md`, any future Claim C1+
write-up must explicitly reference `assets/ArtifactIssuesRegister/` (Issue #1) and state whether it
is expected to materially affect that claim.
