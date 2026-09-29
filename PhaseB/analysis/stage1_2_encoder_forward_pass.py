"""
Stage 1.2 -- instantiate the PanDerm encoder using the OFFICIAL repository
implementation, validate exactly one forward pass, record embedding
dimensionality and preprocessing, then stop.

Per Vinh (2026-08-05): "Proceed to Stage 1.2 only ... Do not perform batch
inference, embedding extraction, probing, or any scientific analysis."

This script does not define a model class or a preprocessing pipeline of its
own. Both come directly from the cloned official repo
(PhaseB/external/PanDerm, commit fd7a80748ba7fc3e203fed88f909f4689d0d6f24):

  - `models.get_encoder(args, 'PanDerm_Large_LP')` -- the exact function the
    repo's own `linear_eval.py` calls, for the exact model name whose
    reproduce script (`classification/script/lp_reproduce.sh`) references
    THIS project's checkpoint file by name
    (panderm_ll_data6_checkpoint-499.pth). This is the repo's own recipe for
    this specific checkpoint, not a recipe chosen by inference.
  - Internally (models/builder.py, PanDerm_Large_LP branch) this constructs
    `panderm_large_patch16_224()` (their registered VisionTransformer
    factory: embed_dim=1024, depth=24, patch=16, class defaults
    lin_probe=True, linear_type='standard', use_mean_pooling=False) and
    loads the checkpoint with a bare `k.replace("encoder.", "")` prefix
    strip + `strict=False` -- their own choice, for this exact checkpoint,
    not ours. Reused verbatim via `models.get_encoder`; not reimplemented.
  - `models.get_encoder` also returns the official eval transform
    (Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(imagenet
    mean/std)) -- used as-is, not re-derived.

The forward pass itself calls `model.forward_features(x, is_train=False)`,
matching exactly how the repo's own
`panderm_model/downstream/extract_features.py::extract_features_from_dataloader`
extracts embeddings (NOT `model(x)`, which would route through the
classification head irrelevantly left over from `num_classes=1000` class
default -- unused here, per the repo's own extraction code).

strict=False explanation (independently verified here, not just assumed):
see the "key diff" section below -- computed by set difference against the
already-loaded model's own state_dict keys, not by a second load.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

import data_roots

REPO_ROOT = Path(__file__).resolve().parents[1] / "external" / "PanDerm" / "classification"
CHECKPOINT = Path(__file__).resolve().parents[1] / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
TEST_IMAGE = data_roots.root("dst_skin_images") / "ISIC_0034336.jpg"

sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from models import get_encoder  # official repo import, classification/models/__init__.py

    class _Args:
        pretrained_checkpoint = str(CHECKPOINT)

    print(f"[stage1.2] Repo: {REPO_ROOT}")
    print(f"[stage1.2] Checkpoint: {CHECKPOINT}")
    print(f"[stage1.2] Test image: {TEST_IMAGE}")

    # --- Official model + transform construction (models/builder.py::get_encoder) ---
    model, eval_transform = get_encoder(_Args(), "PanDerm_Large_LP")
    model.eval()

    print("\n[stage1.2] Preprocessing pipeline (official, from get_encoder's return value):")
    print(f"  {eval_transform}")

    # --- Independent key-diff check (explains the repo's own strict=False; no second load) ---
    raw_state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    stripped_keys = {k.replace("encoder.", "") for k in raw_state_dict.keys()}
    model_keys = set(model.state_dict().keys())

    unexpected = sorted(stripped_keys - model_keys)  # in checkpoint, not in this encoder module
    missing = sorted(model_keys - stripped_keys)  # in this encoder module, not in checkpoint

    print(f"\n[stage1.2] strict=False key diff (independently computed, not from a second load):")
    print(f"  Unexpected (checkpoint keys with no home in VisionTransformer): {len(unexpected)}")
    unexpected_top = sorted({k.split('.')[0] for k in unexpected})
    print(f"    top-level groups: {unexpected_top}")
    print(f"  Missing (VisionTransformer params not covered by checkpoint): {len(missing)}")
    if missing:
        print(f"    {missing}")

    # --- Single forward pass (matches extract_features.py: forward_features, not forward) ---
    image = Image.open(TEST_IMAGE).convert("RGB")
    print(f"\n[stage1.2] Loaded test image: {TEST_IMAGE.name}, original size={image.size}")

    x = eval_transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    print(f"[stage1.2] Preprocessed tensor shape: {tuple(x.shape)}, dtype={x.dtype}")

    with torch.no_grad():
        embedding = model.forward_features(x, is_train=False)

    print(f"\n[stage1.2] Forward pass succeeded.")
    print(f"[stage1.2] Embedding shape: {tuple(embedding.shape)}")
    print(f"[stage1.2] Embedding dtype: {embedding.dtype}")
    print(f"[stage1.2] Embedding stats: mean={embedding.mean().item():.6f}, "
          f"std={embedding.std().item():.6f}, "
          f"any_nan={torch.isnan(embedding).any().item()}, "
          f"any_inf={torch.isinf(embedding).any().item()}")

    print("\n[stage1.2] STOP per Vinh's scope: no batch inference, no embedding cache, "
          "no probing, no scientific analysis performed.")


if __name__ == "__main__":
    main()
