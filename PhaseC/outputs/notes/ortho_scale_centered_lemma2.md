# ortho_scale (centred) — the Lemma 2 configuration

These rows in `generalisation_audit.csv` / `bounds_evaluation.csv` with
`operator=ortho_scale` use

    T(z) = μ + c·Q·(z − μ),   μ = train mean.

That is the setting Lemma 2 is stated for. Cosine is allowed to move (and does).
**`ortho_scale` is not a failed gate configuration** — it is the operator family
the lemma is about. Keep it in the audit for that reason.

## What the bounds writer emits per affine configuration

Checked against completed `lemma2_cosine` rows in `bounds_evaluation.csv`
(38 rows as of the check; all four brief quantities present, non-null):

| Brief quantity | CSV columns |
|---|---|
| Operator spectrum | `sigma_max`, `sigma_min`, `cond` (also `I_minus_A_2`, `I_minus_AAT_2`) |
| r(v) = ‖(A−I)(v−μ)+b‖/‖v‖ | `r_v_max` (max over evaluation vectors; zeros counted in `n_zero_norm_vectors`) |
| max_j r(μ_j) over centroids | `r_mu_max` |
| Predicted bound and observed \|s̃(Tz) − s(z)\| | `predicted_max`, `observed_max` (plus median / tightness / `bound_holds`) |

Lemma 2 is also *evaluated* on other affine operators (`ortho_linear`,
`spectral_temper`, …) because the numerical form applies whenever
T(z) = A(z−μ)+μ+b. Interpretation differs:

- **`ortho_scale` (centred):** the lemma's intended regime; cosine moves; bound is informative.
- **`ortho_linear` (μ = 0, T(z)=cQz):** outside the centred setting that makes cosine
  non-invariant; cosine stays flat — that is the code-correctness *gate*, not the
  lemma demonstration.

## Pure-linear gate (separate)

`operator=ortho_linear` is T(z) = c·Q·z with μ = 0. See
`kde_bandwidth_mechanism.md` for the kde / kde_det reading of that gate.
