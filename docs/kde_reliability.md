# sklearn's KernelDensity is not reliable at d = 1024

The KDE scores this study reports were produced by `sklearn.neighbors.KernelDensity`
on 1024-dimensional PanDerm embeddings. That estimator does not compute correctly
in this regime, and the reported KDE numbers are wrong by about 0.10 AUROC.

## What was observed

Three checks, in increasing strength.

**Two sklearn algorithms disagree with each other.** For the same data, bandwidth
and queries, `algorithm="kd_tree"` and `algorithm="ball_tree"` return different
log-densities — for one query, 95.83 against 441.61; for another, 441.08 against
145.30. At least one is wrong, and nothing in the API says which.

**The wrong values sit at the ceiling.** With 300 class points and h = 0.25, the
largest attainable log-density, reached only when a query coincides exactly with
a training point, is 472.87. sklearn returns 441.08, 437.29 and 439.24 for three
queries whose true nearest squared distance is 31.8 or more. Values that near the
ceiling mean the tree believed those points were essentially on top of a training
point.

**A direct computation agrees with sklearn wherever sklearn is checkable.**
Summing the kernel in log space, `logsumexp(-d^2/2h^2) - log n - (d/2)log(2*pi) - d*log(h)`,
matches sklearn to floating-point in one dimension, on synthetic 1024-dimensional
data, on the queries where the two tree algorithms happen to agree, and on queries
that coincide with a training point, where both give the ceiling 475.98 exactly.
It is the same estimator, not an approximation of it: same Gaussian kernel, same
Scott's-rule bandwidth per class, same 1e-6 floor.

7,334 of 7,365 evaluation points have a log-density that differs by more than 1.

## Consequence for the reported numbers

At the reference configuration, recomputing all four scorers from the retained
embeddings reproduces three of them exactly and moves the fourth:

| scorer | stored | recomputed | difference |
|---|---|---|---|
| Mahalanobis | 0.9984 | 0.9984 | 0.0000 |
| cosine | 0.9805 | 0.9805 | 0.0000 |
| k-NN (k=10) | 0.9979 | 0.9979 | 0.0000 |
| **KDE** | **0.8942** | **0.9988** | **+0.1046** |

That the other three reproduce exactly is what rules out an error in the
reconstruction and localises the problem to the KDE path.

The effect on the manuscript is not confined to one cell. Table 5 classifies KDE
as the one scorer with **material** headroom, H = 0.1058, against near-ceiling or
limited for the rest. Corrected, KDE's headroom is 0.0012 and it is near ceiling
like the others. The claim that one estimator retained room to respond does not
survive; every evaluated scorer is near ceiling.

Table 4's KDE column, the C−A and C−B contrasts of +0.0046 and −0.1056, and the
statement that cosine and KDE are descriptively closer to history-matched than to
slope-matched all rest on the same scores and need recomputing.

The direction of the correction is worth noting: it strengthens the paper's own
near-ceiling conclusion while removing the specific evidence the manuscript used
to argue that one scorer escaped it.

## What was changed

`PhaseB/analysis/generate_per_example_scores.py` computes KDE directly. Squared
distances use the BLAS expansion rather than `cdist`, which is 26 times faster
and agrees to 2e-11; this matters because the exact path through `cdist` takes
sixteen minutes per configuration against about one second.

Paper 3 also reports a `density_kde` scorer, on 16-dimensional representations.
Whether the same failure occurs at that dimension has not been checked here.
