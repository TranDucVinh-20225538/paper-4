"""The five frozen backbones Phase C audits, with their published preprocessing pinned.

Why: the audit's claim is about scorers operating on a *given* embedding space.
That claim is only about the space if the space is reproducible, so each backbone
here is defined by (a) the checkpoint it loads, hashed after loading, (b) the
preprocessing the checkpoint's authors published, written down as data
(`spec["preprocess"]`) and compared against the live torchvision transform before
any embedding is written, (c) the dimension of the vector it returns.

    resnet18_cifar_openood  OpenOOD v1.5 ResNet-18 (32x32 variant) trained on CIFAR-10,
                            seed s0; 512-d post-avgpool vector.
    resnet50_imagenet       torchvision ResNet-50, IMAGENET1K_V1; 2048-d post-avgpool.
    vitb16_imagenet         timm vit_base_patch16_224 (augreg2_in21k_ft_in1k, timm's
                            default tag); 768-d CLS token after the final norm.
    dinov2_vitb14           facebookresearch/dinov2 ViT-B/14 (hub weights); 768-d CLS
                            token after the final norm.
    clip_vitb16             OpenAI CLIP ViT-B/16 image tower via timm's
                            `vit_base_patch16_clip_quickgelu_224.openai` (the QuickGELU
                            variant, which is what the OpenAI weights were trained
                            with); 512-d projected image embedding.

`load_model(device)` returns an nn.Module in eval mode whose forward maps a
normalised image batch to (B, dim) float features, and records the checkpoint
provenance in `bb.provenance` (file path, file SHA-256 where a single file
exists, and the SHA-256 of the state dict bytes, which is what is pinned because
it is independent of the container format). Nothing here touches datasets.

Command line:
    python3 PhaseC/analysis/backbones.py --list
    python3 PhaseC/analysis/backbones.py --write-pins          # (re)write assets/preprocessing/*.json
    python3 PhaseC/analysis/backbones.py --prefetch [names...] # load on CPU, print sha/dim
    python3 PhaseC/analysis/backbones.py --fetch-resnet18      # download the OpenOOD checkpoint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
PHASEC = HERE.parent
REPO = PHASEC.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "PhaseB" / "analysis"))
import data_roots  # noqa: E402

PREPROC_DIR = PHASEC / "assets" / "preprocessing"
INTERP = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}

# OpenOOD publishes both v1 (`cifar10_res18`) and v1.5 (`cifar10_res18_v1.5`). Drive often
# quotas the v1.5 asset; this audit pins whichever file is actually on disk and records the
# version in provenance. Far-OOD of v1 used FashionMNIST; v1.5 replaced it with SVHN — if the
# checkpoint is v1 while the OOD ladder uses SVHN (+ MNIST), the manifest must say so.
OPENOOD_RES18_V15_GDRIVE_ID = "1byGeYxM_PlLjT72wZsMQvP6popJeWBgt"
OPENOOD_RES18_V1_GDRIVE_ID = "1rPEScK7TFjBn_W_frO-8RSPwIG6_x0fJ"
OPENOOD_RES18_GDRIVE_ID = OPENOOD_RES18_V15_GDRIVE_ID   # preferred fetch target
OPENOOD_RES18_NAME = "cifar10_res18_v1.5"
OPENOOD_RES18_SEED_DIR = "s0"    # OpenOOD v1.5 ships three training seeds; s0 is pinned
OPENOOD_RES18_V1_NAME = "cifar10_res18"

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------- hashing
def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def state_dict_sha256(sd) -> str:
    """SHA-256 over (name, dtype, shape, raw bytes) of every tensor, in sorted-name order."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        a = t.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes() if a.numel() else b"")
    return h.hexdigest()


# ---------------------------------------------------------------- preprocessing as data
def _pipe(resize, crop, interp, mean, std):
    return [
        {"op": "Resize", "size": resize, "interpolation": interp, "antialias": True},
        {"op": "CenterCrop", "size": [crop, crop]},
        {"op": "ToTensor"},
        {"op": "Normalize", "mean": list(mean), "std": list(std)},
    ]


def build_transform_from_spec(pipeline):
    """torchvision Compose from a pinned pipeline list (the inverse of `describe_transform`)."""
    ops = []
    for s in pipeline:
        if s["op"] == "Resize":
            ops.append(T.Resize(s["size"], interpolation=INTERP[s["interpolation"]], antialias=s.get("antialias", True)))
        elif s["op"] == "CenterCrop":
            ops.append(T.CenterCrop(tuple(s["size"])))
        elif s["op"] == "ToTensor":
            ops.append(T.ToTensor())
        elif s["op"] == "Normalize":
            ops.append(T.Normalize(mean=s["mean"], std=s["std"]))
        else:
            raise ValueError(f"unknown op {s['op']!r}")
    return T.Compose(ops)


def describe_transform(tf):
    """The live transform as the same list-of-dicts schema the pins use."""
    out = []
    for t in tf.transforms:
        n = t.__class__.__name__
        if n == "Resize":
            out.append({"op": n, "size": t.size if isinstance(t.size, int) else list(t.size),
                        "interpolation": t.interpolation.value, "antialias": bool(t.antialias)})
        elif n == "CenterCrop":
            out.append({"op": n, "size": list(t.size)})
        elif n == "Normalize":
            out.append({"op": n, "mean": [float(x) for x in t.mean], "std": [float(x) for x in t.std]})
        else:
            out.append({"op": n})
    return out


# ---------------------------------------------------------------- the registry
@dataclass
class Backbone:
    name: str
    kind: str
    dim: int
    source_url: str
    pipeline: list
    id_sets: tuple
    description: str
    frozen_from: dict
    provenance: dict = field(default_factory=dict)
    _loader: object = None

    def preprocess_spec(self) -> dict:
        # dataset_level must match datasets_phasec.DATASET_LEVEL_RGB (assert_transform enforces it).
        return {"backbone": self.name, "description": self.description, "frozen_from": self.frozen_from,
                "dataset_level": "PIL image .convert('RGB') before the pipeline",
                "pipeline": self.pipeline}

    def build_transform(self):
        return build_transform_from_spec(self.pipeline)

    def load_model(self, device="cpu") -> nn.Module:
        model, prov = self._loader(self)
        self.provenance = prov
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @property
    def checkpoint_sha256(self) -> str:
        if "state_dict_sha256" not in self.provenance:
            raise RuntimeError("load_model() first: the checkpoint hash is computed after loading")
        return self.provenance["state_dict_sha256"]


class _Wrap(nn.Module):
    """Uniform interface: forward(x) -> (B, dim) float32 features."""

    def __init__(self, net, fn):
        super().__init__()
        self.net, self._fn = net, fn

    def forward(self, x):
        return self._fn(self.net, x).float()


def _portable(p) -> str:
    """A path without the machine's home directory: `<hf repo dir>@<revision>/<file>` or just the file name."""
    p = Path(p)
    if "snapshots" in p.parts:
        i = p.parts.index("snapshots")
        return f"{p.parts[i - 1]}@{p.parts[i + 1]}/{p.name}"
    return p.name


def _hf_file_sha(repo_id):
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None, None
    for fn in ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin"):
        try:
            p = hf_hub_download(repo_id, fn)
            return _portable(p), sha256_file(p)
        except Exception:
            continue
    return None, None      # the state-dict hash still pins the weights


def _torch_hub_file(url):
    p = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    return (_portable(p), sha256_file(p)) if p.is_file() else (None, None)


# -- resnet18 (OpenOOD)
def _classify_res18_ckpt(path: Path) -> dict:
    """Tag a checkpoint file as OpenOOD v1 or v1.5 from its path / archive name."""
    name = path.name.lower()
    parts = "/".join(path.parts).lower()
    if "v1.5" in name or "v1.5" in parts or OPENOOD_RES18_NAME in parts:
        return {"openood_checkpoint_version": "v1.5",
                "openood_checkpoint_id": OPENOOD_RES18_NAME,
                "source_url": f"https://drive.google.com/uc?id={OPENOOD_RES18_V15_GDRIVE_ID}",
                "openood_far_ood_as_published": ["mnist", "svhn", "texture", "places365"],
                "benchmark_pairing_note": "checkpoint and far-OOD ladder both follow OpenOOD v1.5 (SVHN, not FashionMNIST)"}
    return {"openood_checkpoint_version": "v1",
            "openood_checkpoint_id": OPENOOD_RES18_V1_NAME,
            "source_url": f"https://drive.google.com/uc?id={OPENOOD_RES18_V1_GDRIVE_ID}",
            "openood_far_ood_as_published": ["mnist", "fashionmnist", "texture", "places365"],
            "benchmark_pairing_note": (
                "checkpoint is OpenOOD v1 (cifar10_res18); PhaseC far-OOD ladder follows the v1.5 "
                "taxonomy (MNIST, SVHN, Textures, Places365) rather than v1's FashionMNIST. "
                "Numbers are not drop-in comparable to the published OpenOOD v1 leaderboard row.")}


def _find_res18_ckpt():
    """Prefer OpenOOD v1.5 (<ckpt>/cifar10_res18_v1.5.ckpt or s0/best.ckpt). Fall back to the v1
    torch-serialized zip (cifar10_res18.zip), which is a bare state_dict and is what Drive still
    serves when the v1.5 asset hits the view quota. Provenance records which file was used."""
    d = Path(data_roots.root("phasec_checkpoints"))
    d.mkdir(parents=True, exist_ok=True)
    direct = d / f"{OPENOOD_RES18_NAME}.ckpt"
    if direct.is_file():
        return direct
    for z in sorted(d.glob(f"{OPENOOD_RES18_NAME}*.zip")):
        out = d / z.stem
        if not out.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
    cands = sorted(p for p in d.rglob("best.ckpt") if OPENOOD_RES18_SEED_DIR in p.parts)
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise RuntimeError(f"ambiguous OpenOOD checkpoints, keep one: {cands}")
    # v1 fallback: bare state_dict zip, loadable by torch.load directly
    for alt in (d / f"{OPENOOD_RES18_V1_NAME}.zip", d / f"{OPENOOD_RES18_V1_NAME}.ckpt",
                d / f"{OPENOOD_RES18_V1_NAME}.pth"):
        if alt.is_file() and alt.stat().st_size > 10_000:
            return alt
    raise FileNotFoundError(
        f"OpenOOD CIFAR-10 ResNet-18 checkpoint not found under {d}.\n"
        f"  Fetch: python3 PhaseC/analysis/backbones.py --fetch-resnet18\n"
        f"  (Google Drive id {OPENOOD_RES18_V15_GDRIVE_ID} for v1.5; "
        f"{OPENOOD_RES18_V1_GDRIVE_ID} for v1. If Drive reports 'quota exceeded', download "
        f"{OPENOOD_RES18_NAME}.zip elsewhere and drop it, or its {OPENOOD_RES18_SEED_DIR}/best.ckpt, into that directory. "
        f"{OPENOOD_RES18_V1_NAME}.zip is also accepted; the manifest will flag the v1↔v1.5 pairing.)")


def fetch_resnet18():
    import gdown
    d = Path(data_roots.root("phasec_checkpoints"))
    d.mkdir(parents=True, exist_ok=True)
    # Prefer v1.5; fall back to v1 when Drive quotas the newer asset.
    for name, fid in ((OPENOOD_RES18_NAME, OPENOOD_RES18_V15_GDRIVE_ID),
                      (OPENOOD_RES18_V1_NAME, OPENOOD_RES18_V1_GDRIVE_ID)):
        out = d / f"{name}.zip"
        if out.is_file() and out.stat().st_size > 10_000:
            return out
        try:
            gdown.download(id=fid, output=str(out), quiet=False)
            if out.is_file() and out.stat().st_size > 10_000:
                return out
        except Exception as e:
            print(f"[backbones] fetch {name} failed: {e}")
    raise FileNotFoundError("could not fetch OpenOOD ResNet-18 v1.5 or v1 checkpoint")


def _load_resnet18(bb):
    from resnet18_32x32 import ResNet18_32x32
    path = _find_res18_ckpt()
    obj = torch.load(path, map_location="cpu")
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    sd = {k.replace("module.", "", 1).replace("backbone.", "", 1) if k.startswith(("module.", "backbone.")) else k: v
          for k, v in sd.items()}
    net = ResNet18_32x32(num_classes=10)
    net.load_state_dict(sd, strict=True)
    ver = _classify_res18_ckpt(path)
    # Prefer the version-accurate Drive URL over the registry default (which names v1.5).
    prov = {"file": str(path.relative_to(REPO)) if REPO in path.parents else path.name,
            "file_sha256": sha256_file(path), "state_dict_sha256": state_dict_sha256(net.state_dict()),
            "openood_seed_dir": OPENOOD_RES18_SEED_DIR if ver["openood_checkpoint_version"] == "v1.5" else None,
            **ver}
    return _Wrap(net, lambda n, x: n.forward_features(x)), prov


# -- resnet50 (torchvision)
def _load_resnet50(bb):
    import torchvision
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    net = torchvision.models.resnet50(weights=w)
    prov = {"state_dict_sha256": state_dict_sha256(net.state_dict()), "source_url": w.url}
    prov["file"], prov["file_sha256"] = _torch_hub_file(w.url)
    net.fc = nn.Identity()
    return _Wrap(net, lambda n, x: n(x)), prov


# -- timm
def _load_timm(bb, arch, kwargs, tag=None, expect_head=None):
    import timm
    name = f"{arch}.{tag}" if tag else arch
    net = timm.create_model(name, pretrained=True, **kwargs)
    cfg = net.pretrained_cfg
    prov = {"state_dict_sha256": state_dict_sha256(net.state_dict()), "source_url": bb.source_url,
            "timm": timm.__version__, "pretrained_cfg_hf_hub_id": cfg.get("hf_hub_id"),
            "pretrained_cfg_mean": list(cfg["mean"]), "pretrained_cfg_std": list(cfg["std"]),
            "pretrained_cfg_interpolation": cfg["interpolation"], "pretrained_cfg_crop_pct": cfg["crop_pct"]}
    prov["file"], prov["file_sha256"] = _hf_file_sha(cfg.get("hf_hub_id"))
    if expect_head is not None:
        assert net.num_classes == expect_head, f"{name}: head is {net.num_classes}-d, expected {expect_head}"
    return _Wrap(net, lambda n, x: n(x)), prov


def _load_vit(bb):
    return _load_timm(bb, "vit_base_patch16_224", {"num_classes": 0})


def _load_clip(bb):
    """CLIP's visual projection is the timm head (num_classes stays 512), so net(x) is the image embedding.
    The quickgelu variant is the architecture the OpenAI weights were trained with."""
    return _load_timm(bb, "vit_base_patch16_clip_quickgelu_224", {}, tag="openai", expect_head=512)


# -- dinov2 (torch.hub, local repo if cached)
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"


def _load_dinov2(bb):
    hub = Path(torch.hub.get_dir())
    repo = hub / "facebookresearch_dinov2_main"
    if repo.is_dir():
        net = torch.hub.load(str(repo), "dinov2_vitb14", source="local", pretrained=False)
    else:
        net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False)
    sd = torch.hub.load_state_dict_from_url(DINOV2_URL, map_location="cpu", progress=False)
    net.load_state_dict(sd, strict=True)
    prov = {"state_dict_sha256": state_dict_sha256(net.state_dict()), "source_url": DINOV2_URL,
            "hub_repo": "facebookresearch/dinov2 (hubconf dinov2_vitb14)"}
    prov["file"], prov["file_sha256"] = _torch_hub_file(DINOV2_URL)
    return _Wrap(net, lambda n, x: n(x)), prov


def _registry():
    r = {}

    def add(**kw):
        b = Backbone(**kw)
        r[b.name] = b

    add(name="resnet18_cifar_openood", kind="cnn_cifar_supervised", dim=512,
        source_url=f"https://drive.google.com/uc?id={OPENOOD_RES18_V15_GDRIVE_ID}",
        pipeline=_pipe(32, 32, "bilinear", CIFAR10_MEAN, CIFAR10_STD),
        id_sets=("cifar10", "cifar100"),
        description=("OpenOOD ResNet18_32x32 on CIFAR-10, post-avgpool 512-d. "
                     "Prefers cifar10_res18_v1.5; accepts cifar10_res18 (v1) when v1.5 is unavailable — "
                     "see provenance.openood_checkpoint_version and benchmark_pairing_note. "
                     "On CIFAR-100 ID this is the reverse-overlap case (trained on what is now near-OOD)."),
        frozen_from={"repo": "https://github.com/Jingkang50/OpenOOD", "network": "resnet18_32x32",
                     "preprocessor": "base (Resize(32, bilinear), CenterCrop(32), ToTensor, Normalize cifar10)",
                     "v1_gdrive_id": OPENOOD_RES18_V1_GDRIVE_ID,
                     "v15_gdrive_id": OPENOOD_RES18_V15_GDRIVE_ID},
        _loader=_load_resnet18)
    add(name="resnet50_imagenet", kind="cnn_imagenet_supervised", dim=2048,
        source_url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        pipeline=_pipe(256, 224, "bilinear", IMAGENET_MEAN, IMAGENET_STD),
        id_sets=("cifar10", "cifar100", "imagenet200"),
        description="torchvision resnet50 IMAGENET1K_V1, fc stripped, post-avgpool 2048-d",
        frozen_from={"repo": "https://github.com/pytorch/vision", "weights": "ResNet50_Weights.IMAGENET1K_V1",
                     "transforms": "Resize(256, bilinear), CenterCrop(224), Normalize ImageNet"},
        _loader=_load_resnet50)
    add(name="vitb16_imagenet", kind="vit_imagenet_supervised", dim=768,
        source_url="https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pipeline=_pipe(248, 224, "bicubic", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        id_sets=("cifar10", "cifar100", "imagenet200"),
        description="timm vit_base_patch16_224 (augreg2_in21k_ft_in1k), CLS after final norm, 768-d",
        frozen_from={"repo": "https://github.com/huggingface/pytorch-image-models",
                     "pretrained_cfg": "crop_pct 0.9 (224/0.9 = 248 resize), bicubic, mean=std=0.5"},
        _loader=_load_vit)
    add(name="dinov2_vitb14", kind="vit_selfsupervised", dim=768,
        source_url=DINOV2_URL,
        pipeline=_pipe(256, 224, "bicubic", IMAGENET_MEAN, IMAGENET_STD),
        id_sets=("cifar10", "cifar100", "imagenet200"),
        description="DINOv2 ViT-B/14 (LVD-142M), CLS token after final norm, 768-d",
        frozen_from={"repo": "https://github.com/facebookresearch/dinov2",
                     "transforms": "Resize(256, bicubic), CenterCrop(224), Normalize ImageNet "
                                   "(dinov2/data/transforms.py make_classification_eval_transform)"},
        _loader=_load_dinov2)
    add(name="clip_vitb16", kind="vlm_contrastive", dim=512,
        source_url="https://huggingface.co/timm/vit_base_patch16_clip_224.openai",
        pipeline=_pipe(224, 224, "bicubic", CLIP_MEAN, CLIP_STD),
        id_sets=("cifar10", "cifar100", "imagenet200"),
        description="OpenAI CLIP ViT-B/16 image tower (timm quickgelu variant), projected 512-d",
        frozen_from={"repo": "https://github.com/openai/CLIP",
                     "transforms": "Resize(224, bicubic), CenterCrop(224), Normalize CLIP",
                     "timm_crop_pct_override": "OpenAI's preprocess has no crop_pct (resize == crop == 224); "
                                              "timm's pretrained_cfg lists 0.9, which is timm's own convention"},
        _loader=_load_clip)
    return r


REGISTRY = _registry()
BACKBONE_NAMES = tuple(REGISTRY)


def get(name) -> Backbone:
    if name not in REGISTRY:
        raise KeyError(f"unknown backbone {name!r}; choose from {BACKBONE_NAMES}")
    return REGISTRY[name]


# ---------------------------------------------------------------- pins
def pin_path(name) -> Path:
    return PREPROC_DIR / f"{name}.json"


def write_pin(bb: Backbone, overwrite=False) -> Path:
    p = pin_path(bb.name)
    if p.exists() and not overwrite:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bb.preprocess_spec(), indent=2) + "\n")
    return p


def check_pretrained_cfg(bb: Backbone):
    """Independent check for the timm backbones: the pin must agree with the library's own pretrained_cfg
    (mean, std, interpolation, and resize = crop / crop_pct)."""
    pr = bb.provenance
    if "pretrained_cfg_mean" not in pr:
        return
    norm, resize, crop = bb.pipeline[-1], bb.pipeline[0], bb.pipeline[1]["size"][0]
    want_resize = int(np.floor(crop / pr["pretrained_cfg_crop_pct"]))   # timm: math.floor(size / crop_pct)
    resize_ok = int(resize["size"]) == want_resize or "timm_crop_pct_override" in bb.frozen_from
    ok = (np.allclose(norm["mean"], pr["pretrained_cfg_mean"]) and np.allclose(norm["std"], pr["pretrained_cfg_std"])
          and resize["interpolation"] == pr["pretrained_cfg_interpolation"] and resize_ok)
    if not ok:
        raise SystemExit(f"[backbones] {bb.name}: pin disagrees with timm pretrained_cfg {pr}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--write-pins", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prefetch", nargs="*", metavar="NAME", help="load these (default all) on CPU and print hashes")
    ap.add_argument("--fetch-resnet18", action="store_true")
    a = ap.parse_args()
    if a.list:
        for b in REGISTRY.values():
            print(f"{b.name:<24} {b.kind:<26} dim={b.dim:<5} id_sets={b.id_sets}  {b.source_url}")
    if a.write_pins:
        for b in REGISTRY.values():
            print("pin", write_pin(b, a.overwrite))
    if a.fetch_resnet18:
        print(fetch_resnet18())
    if a.prefetch is not None:
        for n in (a.prefetch or BACKBONE_NAMES):
            b = get(n)
            try:
                m = b.load_model("cpu")
            except FileNotFoundError as e:
                print(f"[backbones] {n}: MISSING -- {e}")
                continue
            res = 32 if n == "resnet18_cifar_openood" else 224
            with torch.no_grad():
                out = m(torch.randn(2, 3, res, res))
            assert tuple(out.shape) == (2, b.dim), (n, out.shape)
            check_pretrained_cfg(b)
            print(f"[backbones] {n}: dim={out.shape[1]} state_dict_sha256={b.checkpoint_sha256}\n"
                  f"            file={b.provenance.get('file')} file_sha256={b.provenance.get('file_sha256')}")


if __name__ == "__main__":
    main()
