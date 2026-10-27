# ImageNet-200 round — scope (before numbers exist)

Round two is in scope. Extraction starts only after the CIFAR-10 sweep and its
bounds evaluation close. This note fixes splits, seeds, overlap, and the
ResNet-18 role so FINDINGS does not have to invent a defence later.

## Splits (OpenOOD taxonomy)

| Role | Set | Source |
|---|---|---|
| ID | ImageNet-200 | OpenOOD 200-class list **verbatim** (`benchmark_imglist/imagenet200/train_*.txt`, `val_*.txt`, `test_*.txt`); record file SHA-256 in the manifest when fetched |
| Near-OOD | SSB-hard | OpenOOD `benchmark_imglist/imagenet/test_ssb_hard.txt` — **requires ImageNet-21k**; drop if 21k absent (see `imagenet200_feasibility.md`) |
| Near-OOD | NINCO | OpenOOD `test_ninco.txt` + NINCO images |
| Far-OOD | iNaturalist (OOD subset) | OpenOOD `test_inaturalist.txt` — the OOD subset, not full iNat |
| Far-OOD | Textures/DTD | **Reuse CIFAR-10 round files** under `images_classic/texture` (5 640 images); path-list SHA-256 `c7161032d3e42e9ebef03649f2474ca7bdfcdd9c91caafde46660a2499155b55` |
| Far-OOD | OpenImage-O | OpenOOD `test_openimage_o.txt` + images |

Difficulty axis in FINDINGS remains measured baseline AUROC per
(backbone, OOD, scorer), not the near/far labels.

## Subsample seeds (to be applied at extract; reserved now)

Same registry style as CIFAR-10 (`datasets_phasec.SEEDS`). Caps keep rungs
comparable to the ID test size. Counts filled when lists are on disk:

| Split key | Seed | Cap / rule | Source split |
|---|---:|---|---|
| `imagenet200/id_train` | 4121 | 260 per class (52 000) | OpenOOD train imglist |
| `id_test` (IN-200) | — | full OpenOOD test list (no subsample unless over cap) | OpenOOD test imglist |
| `ood_ssb_hard` | 4111 | `MAX_OOD=10000` if larger | OpenOOD SSB-hard list → 21k images |
| `ood_ninco` | 4112 | `MAX_OOD=10000` if larger | OpenOOD NINCO list |
| `ood_inaturalist` | 4113 | `MAX_OOD=10000` if larger | OpenOOD iNat OOD list |
| `ood_texture` | 4103 | none (5 640; reuse) | same DTD tree as CIFAR-10 |
| `ood_openimage_o` | 4115 | `MAX_OOD=10000` if larger | OpenOOD OpenImage-O list |
| `ood_ref` | 4107 | `MAX_REF=10000` | OpenOOD `val_openimage_o` (adapter construction only) |

Every extract write records `subsample_seed`, `n_source`, `n_used`, and the
imglist SHA-256 in the npz sidecar / manifest — same contract as Places365
(`places365_subsample.md`).

## Pretraining overlap — state it; do not remove it

The ID/OOD boundary is contaminated differently per backbone:

- **ResNet-50 / ViT-B/16** (ImageNet-1k supervised): ImageNet-200 ⊂ ImageNet-1k,
  so they saw the ID classes and the other 800 during pretraining. NINCO is built
  to exclude ImageNet-1k objects; SSB-hard draws from ImageNet-21k, which overlaps
  ImageNet-1k pretraining.
- **CLIP** (web-scale image–text) and **DINOv2** (LVD-142M) have different exposure;
  neither is “clean” in the ImageNet-1k sense either.

**Why this does not threaten the audit.** Every comparison is within a backbone,
against that backbone’s own dose-0 baseline. ΔAUROC and ranking-reach quantities
are computed on that backbone’s own score distribution. Contamination moves the
baseline; it does not change what a transformation does to the ordering at that
baseline.

**What it requires.** Never compare absolute AUROC across backbones as one
quantity. Report each backbone’s baseline separately (deliverable 4: baseline
separability is the difficulty axis). Say the overlap explicitly in FINDINGS.md
so a reader who raises it finds the answer on the page.

## ResNet-18 — keep, label out-of-domain

`resnet18_cifar_openood` is the OpenOOD CIFAR-10 checkpoint: 32×32 input, ten
classes, trained from scratch. On ImageNet-200 it is a badly mismatched feature
extractor; baseline AUROC will be poor.

**Run it anyway**, labelled deliberately out-of-domain — not as a failed
backbone. Deliverable (4) asks whether ranking reach tracks baseline
separability; that needs the low end of the axis, which this backbone supplies.

Resolution and normalisation still come from that checkpoint’s pinned
preprocessing JSON (`Resize(32)`, CIFAR-10 mean/std) and must pass
`assert_transform`. No loader-side upscale to 224.

## Order

1. CIFAR-10 sweep + bounds close.
2. ImageNet-1k path available; imglists hashed in manifest.
3. Drop or keep SSB-hard based on 21k presence.
4. Extract → adapters → audit (same operator grid).
