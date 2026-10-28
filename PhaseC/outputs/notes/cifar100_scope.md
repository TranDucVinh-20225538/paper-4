# CIFAR-100 round two — scope (before numbers exist)

ImageNet-1k is absent on this cluster and ImageNet-200 is a 200-class subset of
it, so round two as originally specified is blocked at the ID set. Round two is
therefore OpenOOD's **CIFAR-100** benchmark, available now. ImageNet-200 becomes
round three, contingent on an ID path (see `imagenet200_scope.md`,
`imagenet200_feasibility.md`, `imagenet200_imglist_hashes.md` — kept).

This is not a downgrade standing in for ImageNet. For this study it is the
better second benchmark: every scorer here estimates parameters per class.
CIFAR-10 gives 5 000 images per class; CIFAR-100 gives 500. A tenfold drop in
n_c degrades Σ̂_c and σ̄_c directly — the quantities whose behaviour under
transformation the audit measures. Predicted directions under `ortho_linear`:

| Scorer | Prediction under smaller n_c |
|---|---|
| `kde` (σ̄ bandwidth) | KDE bandwidth anomaly **grows** relative to CIFAR-10 (σ̄_c estimated from 1/10 the data) |
| `kde_det` | stays flat (det(Σ_c)^(1/(2d)) is rotation-invariant however noisily Σ_c is estimated) |
| Mahalanobis | degrades a third way, through **conditioning**, not through equivariance |

Three mechanisms, three predicted directions, on data already present.

## Splits (OpenOOD CIFAR-100 taxonomy)

| Role | Set | Source |
|---|---|---|
| ID train | CIFAR-100 train | torchvision / OpenOOD (50 000; 500 per class) |
| ID test | CIFAR-100 test | torchvision / OpenOOD (10 000) |
| Near-OOD | CIFAR-10 test | same images as round-one ID test |
| Near-OOD | TIN | OpenOOD `benchmark_imglist/cifar100/test_tin.txt` (filtered; never raw `tin/`) |
| Near-OOD | CIFAR-10 test | seed `ood_cifar10` = 4109 (full 10 000; no subsample) |
| Far-OOD | MNIST, SVHN, Textures, Places365 | same sources / seeds as CIFAR-10 round (`SEEDS` 4108 / 4102 / 4103 / 4104; Places365 seed/count as `places365_subsample.md`) |
| `ood_ref` | CIFAR-10 train | adapter construction only, never evaluated |

Difficulty axis in FINDINGS remains measured baseline AUROC per
(backbone, OOD, scorer), not the near/far labels.

## Same pair, both directions

Round one: CIFAR-10 ID vs CIFAR-100 near-OOD.
Round two: CIFAR-100 ID vs CIFAR-10 near-OOD.
Identical images, different margin geometry — a clean control for anything
claimed about margin distributions.

## ResNet-18 — reverse overlap (mirror of the ImageNet discussion)

`resnet18_cifar_openood` is the OpenOOD **CIFAR-10** checkpoint. In round two the
ID set is CIFAR-100 and CIFAR-10 is near-OOD, so this backbone was trained on
what is now the near-OOD set.

Record it the same way the ImageNet-200 note records ResNet-50 / ViT on
ImageNet-200 ⊂ ImageNet-1k:

- Within-backbone logic is unchanged: every comparison is against that
  backbone's own dose-0 baseline; ΔAUROC does not cross backbones.
- A reader will notice the reverse overlap, so it is written here **before**
  the numbers exist — not as a defence invented after the fact.
- Keep the backbone in the sweep; label the overlap explicitly in FINDINGS.

Resolution and normalisation remain the pinned CIFAR-10 preprocess
(`Resize(32)`, CIFAR-10 mean/std) and must pass the fail-closed
`assert_transform` (see `transform_guard_failclosed.md`).

## ImageNet-1k admin query (do not wait)

Ask cluster administrators whether ImageNet-1k exists anywhere on the system
**outside** `/data2` (shared dataset mounts are common). Keep downloading the
OpenOOD OOD zips already in flight (NINCO, SSB-hard, iNaturalist, OpenImage-O).
Do not block round two on the answer.

## Order

1. CIFAR-10 sweep + bounds close (in progress; do not interrupt).
2. CIFAR-100 extract + audit (this round).
3. ImageNet-200 if/when an ID path exists (round three).
