# Group B Execution Contract

**Tier 0 (Validation/Planning). No code executed here. No score, distance, AUROC, FPR95, or output
artifact is produced by this document.** Required by Vinh (2026-08-05) as the precondition for
Mahalanobis-family execution, the same PI decision that authorized it: *"Mahalanobis-family
execution: AUTHORIZED (sau khi tạo Group B Execution Contract)."* This locks fit population,
evaluation population, method, and output artifact for each authorized scorer **before** any of
them runs, so results can be checked against what was promised here, not just against each other.

## Scope

Covers only the four scorers Vinh explicitly authorized: **Mahalanobis, kNN, Cosine-to-centroid,
KDE.** Energy, ViM, and Relative Mahalanobis are explicitly out of scope — not contracted here, not
run here — per `03c_groupB_readiness.md` (Energy/ViM: classifier-head dependency, unresolved;
Relative Mahalanobis: no implementation exists anywhere in the reused codebases).

## Contract table

| Scorer | Fit population | Evaluation population | Output artifact |
|---|---|---|---|
| Mahalanobis | `ReferenceTrainEmbedding` — z_train, y_train (n=16,211) | `ReferenceEmbedding` — z_id = ISIC-test (n=5,067), z_ood = PAD-UFES (n=2,298) | `maha_results.json` |
| kNN | `ReferenceTrainEmbedding` — z_train as an unlabeled reference pool (n=16,211) | `ReferenceEmbedding` — z_id, z_ood | `knn_results.json` |
| Cosine-to-centroid | `ReferenceTrainEmbedding` — reuses the *same* per-class means already fit for Mahalanobis; no separate fit | `ReferenceEmbedding` — z_id, z_ood | `cosine_results.json` |
| KDE | `ReferenceTrainEmbedding` — z_train, y_train, per-class (n=16,211) | `ReferenceEmbedding` — z_id, z_ood | `kde_results.json` |

ID vs. OOD split within `ReferenceEmbedding` is read from its own `metadata.csv`'s `domain` column
(`isic` → ID, `pad_ufes` → OOD) — not re-derived. True labels of `z_id`/`z_ood` are not used by any
of the four scorers; only `y_train` (all valid ISIC 0–7 labels, since `ReferenceTrainEmbedding` is
ISIC-train only) is consumed as ground truth.

## Method + hyperparameters (reused, not invented)

Every formula and constant below is read directly from the exact source functions Paper 4's reuse
discipline has followed since Stage 3.2 Group A — not re-derived or guessed:

- **Mahalanobis** — `compute_mahalanobis_params_from_arrays(z_train, y_train, num_classes=8,
  reg_eps=1e-5)` + `mahalanobis_min_squared_distances(z, means, precision)`, both imported from
  `CSG-SKin/src/utils/ood_metrics.py`. `reg_eps=1e-5` matches `paper-3/scripts/extract_auroc_e2.py`'s
  own deliberate choice (documented there as intentionally departing from CSG-SKin's
  `eval_ood_benchmarks.py` default of `1e-3`). `num_classes=8` matches Paper 4's own ISIC label
  space (`MEL, NV, BCC, AK, BKL, DF, VASC, SCC`).
- **kNN** — `NearestNeighbors(n_neighbors=k_max).fit(z_train)`, `K_VALUES = (1, 10, 50)`, primary
  `k=10` — exact grid from `paper-3/scripts/extract_auroc_e2.py::compute_knn_scores`; all three
  always computed and reported, none selected after seeing results.
- **Cosine-to-centroid** — `score(z) = min_c [1 − cosine_similarity(z, mean_c)]`, reusing the same
  per-class `means` array already fit for Mahalanobis (drops the covariance/precision step, isolates
  one variable) — `paper-3/scripts/extract_auroc_e2.py::cosine_centroid_scores`.
- **KDE** — per-class `KernelDensity(kernel="gaussian", bandwidth=Scott's-rule)`, bandwidth
  `= max(σ̄ · n_c^(−1/(d+4)), 1e-6)`, `score(z) = −max_c[log p_c(z)]` — `paper-3/scripts/
  extract_e2_8_extra_scorers.py::density_kde_scores`.
- **AUROC / FPR@95** — `y=1` for OOD, higher score = more OOD-like, both metrics via
  `roc_auc_score` + `CSG-SKin/src/utils/ood_metrics.py::fpr_at_95_tpr` — same convention for all
  four scorers, matching every prior AUROC computation in Papers 1–3 and in this project's own Gate 0.

## What each output artifact will (and will not) contain

Tier 1 (Measurement) only: raw AUROC, FPR@95, sample counts, and — for kNN — all three k-grid
results. No cross-arm comparison (there is only one arm so far — Intervention doesn't exist yet), no
Tier-3 language, no claim-testing framing. Full provenance (which artifact, which script, which
commit, timestamp) recorded the same way every prior Stage 3.2 output has been.

## What this contract does not do

Does not run any scorer. Does not compute any score, distance, AUROC, or FPR95. Does not create
`maha_results.json`, `knn_results.json`, `cosine_results.json`, or `kde_results.json`. Purely a
specification, written and available for review before execution, so nothing above can quietly
drift once code exists.
