# ImageNet-200 round — cluster feasibility (before any large download)

Checked 2026-10-27 on `/data2` (node used by Phase C). Negative answers are
deliberate: the design must fit what is here, and the paper must say which sets
it covers.

## Storage

| Path | Size | Used | Avail |
|---|---:|---:|---:|
| `/data2` (`/dev/sdb1`) | 175 T | 144 T (83%) | **31 T** |

No separate home quota tool answered (`quota` → “ask your administrator”); the
binding constraint is the shared `/data2` volume. Phase C CIFAR-10 data + assets
currently ≈ 8.4 G under this home. ImageNet-1k alone is ~140 G (train+val);
ImageNet-21k is multiple terabytes — staging 21k here is not automatic.

## ImageNet-1k (hard prerequisite for ImageNet-200 ID)

**Not found** under:

- `/data2/shared` (depth-3 scan for `ILSVRC*`, `imagenet-1k`, `imagenet1k`, …)
- `/cm/shared`, `/mnt`
- `PhaseC/data/openood/images_largescale/` (directory absent)
- this home’s `OpenMIBOOD/data` (only `benchmark_imglist` configs; no image trees)

Without ImageNet-1k train/val on disk, ImageNet-200 extraction cannot start.
Acquisition must provide a path (or a download source the cluster can reach) and
be recorded in the manifest before round-two extract.

## ImageNet-21k (SSB-hard)

**Unavailable.** Only HuggingFace model caches mentioning `in21k` were found
(weights, not the dataset). Per the brief: **do not substitute** another set and
call it SSB-hard. If 21k remains absent after acquisition planning, **drop
SSB-hard** and state that near-OOD on this cluster is NINCO-only (plus whatever
else is actually present).

## OpenOOD install / version

| Item | Status |
|---|---|
| Full OpenOOD Python package | Not installed as a pinned env package for Phase C; Phase C uses folder + imglist loaders |
| `PhaseC/data/openood` | Partial: `images_classic/{places365,svhn,texture,tin}` + `benchmark_imglist/cifar10/test_tin.txt` only |
| `benchmark_imglist/imagenet200/` | **Absent** on this tree (must fetch) |
| `images_largescale/` | **Absent** |

CIFAR-10 far-OOD taxonomy in Phase C follows **OpenOOD v1.5** naming (MNIST,
SVHN, Textures, Places365) while the ResNet-18 checkpoint is the **v1**
`cifar10_res18` file when v1.5 is unavailable — already recorded in
`manifest.json` / `benchmark_pairing_note`. Round two must name the imglist
revision (file SHA-256) explicitly once `benchmark_imglist/imagenet200` is
fetched; do not assume v1 ≡ v1.5 split definitions.

## What can start now (I/O only, parallel with CIFAR-10 sweep)

1. Fetch OpenOOD **benchmark_imglist** for `imagenet200/` and `imagenet/` (small).
2. Reuse round-one **Textures/DTD** files; path-list SHA-256 already measured:
   `c7161032d3e42e9ebef03649f2474ca7bdfcdd9c91caafde46660a2499155b55` (5 640 images).
3. **Cannot** start ImageNet-1k / 21k / NINCO / iNaturalist / OpenImage-O image
   pulls until a source URL or shared path is known — none is on this cluster today.

## Gap summary (design impact)

| Set | Runnable now? | If missing |
|---|---|---|
| ImageNet-200 ID (needs 1k) | **No** | Block round-two extract |
| SSB-hard (needs 21k) | **No** | Drop SSB-hard; say so in FINDINGS |
| NINCO / iNaturalist-OOD / OpenImage-O | Lists absent; images absent | Fetch with OpenOOD script once 1k path exists |
| Textures/DTD | **Yes** (reuse CIFAR-10 copy) | — |
| ResNet-18 on IN-200 | Code path exists; needs 1k+lists | Keep deliberately OOD (see `imagenet200_scope.md`) |
