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

## ImageNet-21k (full) vs OpenOOD-packaged SSB-hard

**Full ImageNet-21k: unavailable** on this cluster (only HF model caches
mentioning `in21k`). We will **not** invent a substitute and call it SSB-hard.

**However:** OpenOOD v1.5 ships a curated `ssb_hard` image archive under
`images_largescale/` (gdown id `1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE` in
`OpenOOD/scripts/download/download.py`). That archive *is* the OpenOOD
benchmark artifact for SSB-hard (drawn from 21k upstream), not a homemade
replacement. **Plan:** download OpenOOD’s `ssb_hard` package; do **not** stage
full 21k. If that gdown fails, drop SSB-hard and say near-OOD is NINCO-only.

Same pattern for NINCO / iNaturalist-OOD / OpenImage-O / (optional) OpenOOD’s
`imagenet_1k` packaging — each has a gdown id in that script.

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

1. Fetch OpenOOD **benchmark_imglist** (gdown id `1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1`) — in flight.
2. Reuse round-one **Textures/DTD**; path-list SHA-256:
   `c7161032d3e42e9ebef03649f2474ca7bdfcdd9c91caafde46660a2499155b55` (5 640 images).
3. Queue OpenOOD image zips (`ssb_hard`, `ninco`, `inaturalist`, `openimage_o`) via
   the same download script — I/O only; no GPU.
4. **ImageNet-1k** remains the hard gap: either OpenOOD’s `imagenet_1k` gdown
   (`1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj`, large) or a shared ILSVRC path. Not on
   disk today; do not start round-two extract until it is.

OpenOOD version for round two: **v1.5 download script / imglist zip** (NeurIPS
2023 DistShift short version lineage). Exact imglist file SHA-256 recorded in
the manifest once the zip lands.

## Gap summary (design impact)

| Set | Runnable now? | If missing |
|---|---|---|
| ImageNet-200 ID (needs 1k) | **No** (1k absent) | Block extract |
| SSB-hard | Lists via imglist zip; images via OpenOOD `ssb_hard` gdown (**not** full 21k) | Drop only if that package fails |
| NINCO / iNaturalist-OOD / OpenImage-O | Same — OpenOOD packages | Fetch in parallel |
| Textures/DTD | **Yes** (reuse) | — |
| ResNet-18 on IN-200 | Code path exists; needs 1k+lists | Keep deliberately OOD (see `imagenet200_scope.md`) |
