# Checkpoint Identity — PanDerm ViT-L/16

Frozen identity record, per Vinh (2026-08-05): "Đừng chỉ ghi PanDerm" — reviewers ask *which*
checkpoint. This file is the answer. Do not edit the values below after the fact; if the checkpoint
is ever re-downloaded or replaced, add a new dated record, don't overwrite this one.

| Field | Value |
|---|---|
| Model | PanDerm |
| Architecture | ViT-L/16 (24 transformer blocks, hidden dim 1024, patch 16×16, 197 tokens ⇒ 224×224 input) |
| Pretraining scheme (inferred from checkpoint structure) | Context-autoencoder-style masked image modeling — encoder + cross-attention "regresser" + pixel-reconstruction "decoder" head, BEiT-style layer-scale (`gamma_1`/`gamma_2`) and separate q/v bias, relative position embedding (`rd_pos_embed`) |
| Paper | arXiv:2410.15038 / Nature Medicine 2025, "A Multimodal Vision Foundation Model for Clinical Dermatology" |
| Source repo | https://github.com/SiyuanYan1/PanDerm |
| Repo commit (HEAD at time of download, via `git ls-remote`) | `fd7a80748ba7fc3e203fed88f909f4689d0d6f24` |
| Checkpoint file (as named in repo README) | `panderm_ll_data6_checkpoint-499.pth` |
| Download URL | https://drive.google.com/file/d/1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE/view |
| Local filename | `PhaseB/assets/checkpoints/panderm_vitl16_checkpoint-499.pth` |
| Download date | 2026-08-05 (2026-08-04T17:11:38Z) |
| Download tool | `gdown` 6.1.0 |
| File size | 1,418,820,654 bytes (1.32 GiB) |
| SHA256 | `23405e92f853c6f7ac9da80ccf4cfde84d4feb22cd9d7b98c7e850e3875554b2` |
| Source file mtime (preserved by gdown from Google Drive) | 2025-04-29 (predates this project — consistent with an untampered original artifact) |
| License | CC-BY-NC-ND 4.0 — "may only be used for non-commercial academic research purposes with proper attribution" (verbatim, repo README) |
| Modifications made to the file | **None.** Downloaded, hashed, loaded read-only for structural verification. Not converted, not re-serialized, not re-saved in any format. |

## Structural verification (2026-08-05)

`torch.load(path, map_location="cpu", weights_only=False)` succeeds. The file is a raw
`collections.OrderedDict` (the state_dict itself, no wrapper dict/epoch/optimizer state).

- **458 tensors, 354,660,064 total parameters.**
- **`encoder.*`** subtree (the part relevant to downstream feature extraction — Stage 1's actual
  target): 366 tensors, **303,326,208 parameters** — cls_token, pos_embed, patch_embed (16×16 conv,
  1024 channels), 24 transformer blocks, final norm. Matches standard ViT-L/16 parameter count.
- **Pretraining-only components, NOT part of the downstream encoder** (present in the checkpoint
  because it was saved directly from the self-supervised pretraining run, not re-exported as an
  encoder-only checkpoint): `mask_token`, `rd_pos_embed` (relative-distance position embedding used
  by the pretraining objective), `regresser.*` (4 cross-attention regressor blocks), `decoder.*`
  (norm + linear head projecting to 768 = 16×16×3 pixel-reconstruction target). These will not be
  loaded when the encoder is wired up for feature extraction — noted here so their absence from a
  future `model.load_state_dict(..., strict=True)` call on an encoder-only module is expected, not
  a loading failure.

This is the "checkpoint loads" verification only, per the gated Stage 1 sequence
(`../02_stage1_setup.md`). Instantiating the actual PanDerm encoder class and running a forward
pass (single-image inference) is the next, separately-gated step — not done here, and not
conflated with this structural check.

## What was NOT done (per Vinh's constraint: no modification)

- Not converted to another format (no `.safetensors`, no re-pickling, no state_dict re-keying).
- Not merged, pruned, or quantized.
- Not re-uploaded or re-hosted anywhere.
- The `encoder.*` subset was inspected read-only above; no separate encoder-only file was written.
