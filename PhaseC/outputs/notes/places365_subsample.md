# Places365 subsample (CIFAR-10 far-OOD)

| Field | Value |
|---|---|
| Source | OpenOOD classic folder `images_classic/places365` under `PAPER4_OPENOOD` (not torchvision `Places365` for this launch — OpenOOD folder wins in `_raw_split`) |
| Split | Places365 **validation** images as packaged by OpenOOD (`Places365_val_*.jpg` under 368 scene categories) |
| n_source | 182 532 |
| Cap | `MAX_OOD = 10_000` |
| n_used | 10 000 |
| Subsample seed | `SEEDS["ood_places365"] = 4104` (seeded without-replacement draw in `datasets_phasec.subsample`) |
| Path-list SHA-256 | `7bfbed3425bddbbb03bbfbc10c5218269f4e919539e6310f37e54b8ab2ff5674` (sorted relative paths under the OpenOOD folder; fingerprints the curated tree we used) |

OpenOOD’s list was used **verbatim as the on-disk folder** (recursive image glob), not a separate imglist file. Torchvision Places365 `(split=val, small=True)` remains a fallback in code if the OpenOOD folder is absent; embeddings on disk for all five backbones record `source: openood folder images_classic/places365` and `subsample_seed: 4104`.
