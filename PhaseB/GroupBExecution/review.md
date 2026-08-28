# Group B Execution Review

Not Discussion, not Result — a structural record of what fed into what, so a future paper draft
never needs to reopen the run log or the source repos to answer "where did Mahalanobis come from."

## Inputs

**Fit population — `ReferenceTrainEmbedding`** (EA-01)
n = 16,211 (ISIC 2019 train split)
`embeddings.npy` SHA256 = `419ba2c73e95f3b97063f3d6c7fac5fd2faa0393b1a1cc267096b69d730a62df`
Status: FROZEN · path: `assets/reference_embeddings/ReferenceTrainEmbedding/`

**Evaluation population — `ReferenceEmbedding`** (Stage 2.1)
n_id = 5,067 (ISIC-test) · n_ood = 2,298 (PAD-UFES)
`embeddings.npy` SHA256 = `3a12b0e08440e63f4a6a2821705d1947a3063a4543c80d1ceafa323902eac9ff`
Status: FROZEN · path: `assets/reference_embeddings/ReferenceEmbedding/`

Both SHA256 values were re-verified against each artifact's own `manifest.json` immediately before
use (not assumed from freeze-time) — confirmed in `logs/groupB_mahalanobis_family_run_2026-08-05.log`.

## Implementation

| Scorer | Source | How reused |
|---|---|---|
| Mahalanobis | `CSG-SKin/src/utils/ood_metrics.py::compute_mahalanobis_params_from_arrays` + `::mahalanobis_min_squared_distances` | **Imported directly, unmodified.** `num_classes=8`, `reg_eps=1e-5`. |
| Cosine-to-centroid | `paper-3/scripts/extract_auroc_e2.py::cosine_centroid_scores` | **Reproduced verbatim** (formula + logic identical). Not imported: that file's module-level imports pull in paper-3-local `_repo_paths` and Lightning checkpoint classes unrelated to this pure-numpy formula. Reuses the *same* per-class means already fit for Mahalanobis — not refit. |
| kNN | `paper-3/scripts/extract_auroc_e2.py::compute_knn_scores` | **Reproduced verbatim.** `K_VALUES=(1,10,50)`, primary `k=10`. Pooled reference pool = full `z_train`, all classes. |
| KDE | `paper-3/scripts/extract_e2_8_extra_scorers.py::density_kde_scores` | **Reproduced verbatim.** Per-class Gaussian kernel, Scott's-rule bandwidth (floor 1e-6). |
| AUROC / FPR@95 | `CSG-SKin/src/utils/ood_metrics.py::fpr_at_95_tpr` (+ `sklearn.roc_auc_score`) | **Imported directly, unmodified**, for all four scorers. `y=1` for OOD, higher score = more OOD-like. |

Script: `analysis/groupB_mahalanobis_family.py`. Log: `logs/groupB_mahalanobis_family_run_2026-08-05.log`.
Specification: `GroupBExecution/execution_contract.md` (written and reviewed before this script existed).

## Outputs

4 JSON files in `GroupBExecution/`, none frozen:

- `maha_results.json`
- `cosine_results.json`
- `knn_results.json` (all three k values)
- `kde_results.json`

Each carries its own provenance block (fit/eval artifact paths + status, script path, CSG-SKin
commit, hyperparameters, timestamp) — not just the number.

## Protocol deviations

**None.** Every fit population, evaluation population, formula, and hyperparameter matches
`execution_contract.md` exactly as specified before any code was written.

## Known limitations

- **Energy** — BLOCKED. No trained classifier head exists for the Reference Instrument.
- **ViM** — PARTIALLY BLOCKED. Fitting population (`ReferenceTrainEmbedding`) now exists; classifier
  head still missing — same open question as Energy.
- **Relative Mahalanobis** — NOT IMPLEMENTED. No code found anywhere in the reused codebases
  (`paper-3/`, `CSG-SKin/`, `DST-Skin/`); a separate open decision, independent of data availability.
- **Per-direction variance** — not computed. No resolved definition yet.
- None of the above were worked around. All three remain open, separate decisions.
