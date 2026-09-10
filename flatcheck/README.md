# flatcheck

Three checks on whether a flat evaluation result carries information.

You intervene on a representation, rerun a distance-based out-of-distribution
scorer, and the AUROC does not move. That has at least three explanations and an
aggregate summary cannot tell them apart:

- the scorer is **invariant** to the change,
- the scorer **cannot respond** to any change, because almost no
  in-distribution / out-of-distribution pair sits within reach of the
  perturbation,
- the **analysis cannot reject** whatever it observes, because its correction
  threshold lies below the smallest p-value its resampling scheme can attain.

A fourth applies when the curve is *not* flat, and is easier to miss: the
apparent response can be an artifact of evaluating an operator on the domain the
operator was built from.

Each check is one function. None needs more data than you already have.

```bash
pip install flatcheck
```

## Can the scorer respond?

`S(delta)` is the fraction of ID–OOD pairs whose score gap the perturbation can
close — an upper bound on the AUROC change it can produce.

```python
from flatcheck import susceptibility

r = susceptibility(scores_id, scores_ood, after - before)
print(r)      # delta[median]=7.44  S=0.000065
```

Read it comparatively. Two scorers evaluated at their own perturbation
magnitudes have commensurable `S`, and an order-of-magnitude gap separates a
scorer that declined to move from one that could not. In the study this came
from, Mahalanobis and cosine differ by a factor of 735 — and the *less*
responsive one is the one whose raw scores move further.

The bound is only valid at `quantile="max"`, and there it is usually vacuous
(`r.vacuous` says so). That is not a defect of the bound; it is why the median
row exists.

For a scale-covariant score such as a k-nearest-neighbour distance the
perturbation is multiplicative, so pass the ratio:

```python
susceptibility(id_, ood, np.maximum(after/before, before/after), kind="ratio")
```

## Can the design reject at all?

A permutation over `k` exchangeable units attains no p below `1/k!`. A step-down
correction over `m` tests has a first threshold of `alpha/m`.

```python
from flatcheck import reachable, max_family_size, min_units

reachable(n_units=5, n_tests=16).ok     # False
max_family_size(5)                      # 6  -- the most 5 units can support
min_units(16)                           # 6  -- the fewest 16 tests need
```

At five seeds and alpha = .05 no family larger than six tests can clear the
first step, whatever the data show. Reporting such a null describes the design.
Check it when you write the analysis plan; it is one division, and it is
uncomfortable to discover afterwards.

If the permutation is sampled rather than enumerated, pass `n_arrangements`
instead — the floor becomes `1/n_arrangements`.

## Was the operator built from the domain it is evaluated on?

Nuisance removal and shortcut suppression usually define their target from a
contrast between the in-distribution population and a reference shifted one.
Where that reference is also an evaluation domain, the intervention was trained
to suppress the direction along which that domain separates.

```python
from flatcheck import alignment, nuisance_direction

w = nuisance_direction(emb_id, emb_reference)
for a in alignment(w, emb_id, {"held-out": emb_a, "external": emb_b}):
    print(a)
# held-out    |cos| = 1.0000   construction target -- effects here are not mechanism evidence
# external    |cos| = 0.0016   near-orthogonal -- the operator does not point at this domain
```

No sample size enters this one and the answer does not change with more seeds.

## Command line

```bash
flatcheck scores.csv --score maha --after maha_after --seeds 5 --tests 16
```

One row per evaluation example, a 0/1 column marking the shifted population, and
the score before and after the intervention.

## Where this came from

These are the diagnostics from a study of when representation geometry predicts
distance-based OOD ranking. Two of the three failures they detect were present
in that study's own earlier analysis and went unnoticed until the corresponding
check was applied, which is the reason they are packaged separately from it.

`tests/` reproduces the published numbers where the study's artifacts are
available and skips cleanly where they are not.

## License

MIT.
