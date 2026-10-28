# Transform pin guard — fail closed (Follow-up 04)

## Defect

An ad-hoc re-check of the live transform against the pins raised
`ImportError: cannot import name 'BACKBONES'` and continued after a manual
verify. That is fail-open. A guard whose failure is logged and skipped, or
satisfied by a later hand check, gives confidence without protection.

Separately, `extract_embeddings.py` used to auto-write a missing pin from the
registry and continue. A missing pin is a failure, not a skip.

## Fix (structure, not the import name)

1. `datasets_phasec.assert_transform` — any exception (missing file, unreadable
   JSON, schema error, pipeline drift) becomes `SystemExit`. Callers must not
   catch it.
2. `extract_embeddings.py` — missing pin aborts before any write; no auto-write.
3. On pass, write `assets/embeddings/<backbone>/<id_set>/transform_guard.json`
   **before** the first embedding write (`passed`, `pin_sha256`, `checked_at`).
   Silent absence of that receipt is detectable afterwards.

## Hand re-verify of embeddings already written under the broken era

Live `build_transform()` matches every current pin pipeline (ops, sizes,
interpolation, antialias, mean/std, `dataset_level`) for all five backbones —
asserted 2026-10-27 after the fail-closed rewrite.

| Backbone | Splits | `preprocess_sha256` in npz | Verdict |
|---|---|---|---|
| `resnet50_imagenet` | all 9 | matches current pin `a3e6a7e0…` | correct transform |
| `vitb16_imagenet` | all 9 | matches current pin `9c050c62…` | correct transform |
| `clip_vitb16` | all 9 | matches current pin `631ac8fe…` | correct transform |
| `dinov2_vitb14` | all 9 | matches current pin `9e5379a8…` | correct transform |
| `resnet18_cifar_openood` | `id_train`, `id_test`, `ood_ref`, `ood_cifar100`, `ood_svhn`, `ood_texture`, `ood_places365` | `306896dd…` (pin file later rewritten) | correct transform — see below |
| `resnet18_cifar_openood` | `ood_mnist`, `ood_tin` | matches current pin `c2f9efee…` | correct transform |

### ResNet-18 hash split

Extract log `extract_resnet18.log` (2026-10-27 15:07) records
`preprocessing matches the pinned record … ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']`
**before** every write of the seven `306896dd…` splits. The pin JSON on disk was
rewritten at 15:34 (adding / expanding provenance fields including
`dataset_level`); that changed the file SHA to `c2f9efee…` without changing the
pipeline ops, sizes, or CIFAR-10 mean/std. MNIST and TIN were extracted after
that rewrite under `c2f9efee…`, with the same live pipeline.

So every ResNet-18 embedding on disk was asserted against the live
`Resize(32) → CenterCrop(32) → ToTensor → Normalize(CIFAR-10)` path. The hash
mismatch is pin-*file* provenance churn, not a transform fork. No re-extract
required for correctness; future extracts will refuse to treat a pin mismatch as
current (`is_current` compares `preprocess_sha256`) and will rewrite only under
`--force` after a new fail-closed guard receipt.

## Content check — ResNet-18 `306896dd` vs current pin (Follow-up 05)

Old pin bytes hashing to `306896dd…` are **not recoverable** from disk or git
(first commit already stores `c2f9efee…`). A hash mismatch explained by churn and
one explained by a changed transform look the same until the dictionaries are opened.

**Field-by-field compare of the applied transform** (live `build_transform()` /
current pin `pipeline` — the transform `assert_transform` checked at extract):

| field | live / current pin | match |
|---|---|---|
| Resize.antialias | True | True |
| Resize.interpolation | bilinear | True |
| Resize.size | 32 | True |
| CenterCrop.size | [32, 32] | True |
| Normalize.mean | [0.4914, 0.4822, 0.4465] | True |
| Normalize.std | [0.247, 0.2435, 0.2616] | True |

**Verdict:** pipeline_semantically_identical = `True`. No differing
transform field (resize, crop, interpolation, antialias, mean, std). Non-transform
pin keys present now: ['backbone', 'description', 'frozen_from', 'dataset_level']. Recorded permanently
under `assets/embeddings/resnet18_cifar_openood/cifar10/transform_guard.json` →
`pin_lineage`.

