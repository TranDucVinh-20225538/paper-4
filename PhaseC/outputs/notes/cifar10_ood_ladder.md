# CIFAR-10 OOD ladder — six sets, not five

The brief listed five far/near rungs against CIFAR-10 ID:
CIFAR-100, SVHN, Textures/DTD, Places365, LSUN.

Phase C evaluates **six** OOD sets (plus `ood_ref` = CIFAR-100 train for
adapter construction only, never scored):

| # | name | OpenOOD near/far label | Role vs the brief |
|---|---|---|---|
| 1 | `cifar100` | near | brief |
| 2 | `tin` | near | **added** — OpenOOD/SCOOD *filtered* TinyImageNet imglist (CIFAR-10-class overlaps removed); not in the five-name brief list |
| 3 | `mnist` | far | **sixth / substitute** — OpenOOD v1.5 far-OOD taxonomy (replaces FashionMNIST on the v1 leaderboard pairing; see manifest `benchmark_pairing_note`). Source: `torchvision.MNIST(train=False)` (10 000 test images, no subsample) |
| 4 | `svhn` | far | brief |
| 5 | `texture` | far | brief (DTD) |
| 6 | `places365` | far | brief |
| — | `lsun` | far | **not used** on this CIFAR-10 launch (loader seed `ood_lsun=4106` is reserved; ImageNet-200 round is separate) |

So the sixth set relative to the brief’s five is **`tin`** if counting
additions alongside a swapped far rung, or **`mnist`** if comparing to the
brief’s far list that named LSUN. Either way the evaluated tuple is exactly

    (cifar100, tin, mnist, svhn, texture, places365)

Difficulty in FINDINGS is measured baseline AUROC, not the near/far labels.

## MNIST grayscale → RGB

MNIST is single-channel 28×28. Channel replication is **not** a loader fork:
every split (including MNIST) goes through `WithTransform`, which applies
`PIL.Image.convert("RGB")` before the backbone’s pinned torchvision pipeline
(Resize / CenterCrop / Normalize with that backbone’s mean/std). The contract
is pinned as `dataset_level` in `assets/embeddings/<backbone>/preprocess.json`
and enforced by `datasets_phasec.assert_transform` (same guard as size /
interpolation / Normalize). Embeddings already extracted for this launch used
that path.
