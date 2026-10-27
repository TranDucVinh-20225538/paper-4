"""Phase C datasets: the OOD difficulty ladder, with every subsample seed written down.

Why: the audit asks, per backbone, how far each OOD set sits from the ID set in
embedding space because reach S depends on how many ID-OOD score pairs are close.
A rung that is silently different between machines makes the ladder irreproducible,
so each rung is a named loader that reports where its images came from, how many
there were, and the seed of any subsample.

The OpenOOD near/far labels are recorded as metadata only. The difficulty axis used
in FINDINGS is the measured baseline AUROC per (backbone, OOD, scorer), not those
labels: near/far is a coarse prior, not the reported ordering.

ID sets
    cifar10      train (50,000) fits the scorers; test (10,000) is the ID evaluation set.
                 Near-OOD (OpenOOD label): cifar100 (test), tin (OpenOOD *filtered* imglist —
                 1,207 CIFAR-10-class overlaps removed; never the raw TinyImageNet dump).
                 Far-OOD (OpenOOD label): mnist, svhn, texture (DTD), places365.
                 `ood_ref` is CIFAR-100 *train*, adapter construction only, never evaluated.
    imagenet200  OpenOOD ImageNet-200 from benchmark_imglist (round two; not this launch).

TIN is loaded only via an OpenOOD/SCOOD imglist. Loading the raw `images_classic/tin`
tree would reintroduce the 1,207 class-overlapping images OpenOOD removed and would
contaminate baseline AUROC — the paper's own overlap-screening argument forbids that.

Sources: torchvision under data_roots `phasec_data`/torchvision, then OpenOOD folder /
imglist layout. A missing rung raises DatasetUnavailable; callers log and continue
except for TIN, which refuses the raw-folder fallback.

Command line:
    python3 PhaseC/analysis/datasets_phasec.py --status [--download]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PHASEC = HERE.parent
REPO = PHASEC.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "PhaseB" / "analysis"))
import data_roots  # noqa: E402
from backbones import describe_transform  # noqa: E402

__all__ = ["SEEDS", "MAX_OOD", "ID_SETS", "OOD_NAMES", "DatasetUnavailable", "subsample", "build_split",
           "split_names", "write_preprocess_json", "assert_transform", "tv_root", "openood_root",
           "WithTransform"]

#: Every seed used while building splits. Keys are split names (or "<id_set>/id_train_per_class").
#: The values are arbitrary distinct integers, fixed once; changing one changes which images are used.
SEEDS = {
    "ood_cifar100": 4101, "ood_svhn": 4102, "ood_texture": 4103, "ood_places365": 4104,
    "ood_tin": 4105, "ood_lsun": 4106, "ood_ref": 4107, "ood_mnist": 4108,
    "ood_ssb_hard": 4111, "ood_ninco": 4112, "ood_inaturalist": 4113, "ood_openimage_o": 4115,
    "imagenet200/id_train": 4121,      # the per-class draw of the 260-per-class training subset
}
#: OOD sets larger than this are subsampled to it (seeded); the cap keeps every rung the same order
#: of magnitude as the ID test set so AUROC standard errors are comparable across rungs.
MAX_OOD = 10_000
#: Reference OOD (adapter construction only): the same cap.
MAX_REF = 10_000
IMAGENET200_TRAIN_PER_CLASS = 260     # 200 x 260 = 52,000, the size of CIFAR-10 train

# OpenOOD CIFAR-10 taxonomy (informational). Difficulty reporting uses measured baseline AUROC.
CIFAR10_NEAR_OOD = ("cifar100", "tin")
CIFAR10_FAR_OOD = ("mnist", "svhn", "texture", "places365")

ID_SETS = {
    "cifar10": {
        "ood": ("cifar100", "tin", "mnist", "svhn", "texture", "places365"),
        "near_ood": CIFAR10_NEAR_OOD,
        "far_ood": CIFAR10_FAR_OOD,
    },
    "imagenet200": {"ood": ("ssb_hard", "ninco", "inaturalist", "texture", "openimage_o")},
}
OOD_NAMES = {k: v["ood"] for k, v in ID_SETS.items()}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}

#: OpenOOD / SCOOD imglists for CIFAR-10 TIN (filtered). Tried in order; raw-folder fallback is forbidden.
CIFAR10_TIN_IMGLISTS = (
    "benchmark_imglist/cifar10/test_tin.txt",          # OpenOOD layout
    "benchmark_imglist/cifar10/test_tin200.txt",
    "imglist/cifar10/test_tin.txt",                    # after installing from SCOOD/OpenOOD zip
    "imglist/tin_cifar10_filtered.txt",
)

CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


class DatasetUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- roots
def tv_root() -> Path:
    return Path(data_roots.root("phasec_data")) / "torchvision"


def openood_root() -> Path | None:
    try:
        r = Path(data_roots.root("openood"))
        if r.exists():
            return r
    except RuntimeError:
        pass
    r = Path(data_roots.root("phasec_data")) / "openood"
    return r if r.exists() else None


# ---------------------------------------------------------------- subsample + wrappers
def subsample(dataset, n, seed):
    """Seeded subsample without replacement: (Subset, sorted indices). n >= len keeps everything."""
    N = len(dataset)
    if n is None or n >= N:
        idx = np.arange(N)
        return dataset, idx
    idx = np.sort(np.random.default_rng(seed).choice(N, size=n, replace=False))
    return torch.utils.data.Subset(dataset, idx.tolist()), idx


class ImageList(torch.utils.data.Dataset):
    """Images from explicit paths; yields (PIL RGB, label)."""

    def __init__(self, paths, labels=None):
        self.paths = [str(p) for p in paths]
        self.labels = np.full(len(self.paths), -1, np.int64) if labels is None else np.asarray(labels, np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return Image.open(self.paths[i]).convert("RGB"), int(self.labels[i])


class WithTransform(torch.utils.data.Dataset):
    """(PIL, label) -> (tensor, label). `ood=True` forces label -1 (OOD images carry no ID class)."""

    def __init__(self, base, tf, ood=False):
        self.base, self.tf, self.ood = base, tf, ood

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        if not hasattr(img, "convert"):
            raise TypeError("base dataset must yield PIL images")
        return self.tf(img.convert("RGB")), (-1 if self.ood else int(y))


# ---------------------------------------------------------------- torchvision providers
def _extract_if_complete(root: Path, tar_name: str, md5: str, extracted_dir: str):
    tar = root / tar_name
    if (root / extracted_dir).is_dir() or not tar.is_file():
        return
    h = hashlib.md5()
    with open(tar, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != md5:
        raise DatasetUnavailable(f"{tar} present but md5 {h.hexdigest()} != {md5}: download incomplete "
                                 "(still running?)")
    with tarfile.open(tar) as t:
        t.extractall(root)


def _tv_cifar(which, train, download):
    import torchvision
    root = tv_root()
    root.mkdir(parents=True, exist_ok=True)
    cls, tar, md5, d = (torchvision.datasets.CIFAR10, "cifar-10-python.tar.gz", CIFAR10_MD5, "cifar-10-batches-py") \
        if which == 10 else (torchvision.datasets.CIFAR100, "cifar-100-python.tar.gz", CIFAR100_MD5, "cifar-100-python")
    _extract_if_complete(root, tar, md5, d)
    try:
        ds = cls(str(root), train=train, download=download)
    except RuntimeError as e:
        raise DatasetUnavailable(f"CIFAR-{which} not in {root} (run with --download): {e}")
    return ds, {"source": f"torchvision.CIFAR{which}", "root": str(root.relative_to(REPO)) if REPO in root.parents else str(root),
                "archive_md5": md5}


def _tv_svhn(download):
    import torchvision
    root = tv_root()
    try:
        ds = torchvision.datasets.SVHN(str(root), split="test", download=download)
    except RuntimeError as e:
        raise DatasetUnavailable(f"SVHN test not in {root} (run with --download): {e}")
    return ds, {"source": "torchvision.SVHN(split=test)", "archive_md5": "eb5a983be6a315427106f1b164d9cef3"}


def _tv_dtd(download):
    import torchvision
    root = tv_root()
    try:
        parts = [torchvision.datasets.DTD(str(root), split=s, partition=1, download=download)
                 for s in ("train", "val", "test")]
    except (RuntimeError, FileNotFoundError) as e:
        raise DatasetUnavailable(f"DTD not in {root} (run with --download): {e}")
    return torch.utils.data.ConcatDataset(parts), {"source": "torchvision.DTD(all splits, partition 1)"}


def _tv_places(download):
    import torchvision
    root = tv_root()
    try:
        ds = torchvision.datasets.Places365(str(root), split="val", small=True, download=download)
    except (RuntimeError, FileNotFoundError, OSError) as e:
        raise DatasetUnavailable(f"Places365 val (small) not in {root} (run with --download): {e}")
    return ds, {"source": "torchvision.Places365(split=val, small=True)"}


def _tv_mnist(download):
    """MNIST test set as far-OOD for CIFAR-10 (OpenOOD taxonomy). Softest far-OOD rung."""
    import torchvision
    root = tv_root()
    root.mkdir(parents=True, exist_ok=True)
    try:
        ds = torchvision.datasets.MNIST(str(root), train=False, download=download)
    except RuntimeError as e:
        raise DatasetUnavailable(f"MNIST test not in {root} (run with --download): {e}")
    # MNIST yields grayscale; ImageList/WithTransform convert to RGB. Wrap so PIL L -> RGB works.
    return ds, {"source": "torchvision.MNIST(train=False)", "openood_group": "far_ood"}


# ---------------------------------------------------------------- OpenOOD providers
_CLASSIC_DIRS = {
    "texture": ("texture", "textures", "dtd"),
    "places365": ("places365", "places"),
    "tin": ("tin", "tiny_imagenet", "tinyimagenet"),
    "lsun": ("lsun_c", "lsun_r", "lsun"),
    "svhn": ("svhn",),
    "mnist": ("mnist",),
}


def _glob_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in IMG_EXT)


def _openood_folder(name):
    root = openood_root()
    if root is None:
        raise DatasetUnavailable(f"{name}: no OpenOOD root (set PAPER4_OPENOOD or place data under PhaseC/data/openood)")
    for alias in _CLASSIC_DIRS.get(name, (name,)):
        for base in (root / "images_classic", root / "images_largescale", root):
            d = base / alias
            if d.is_dir():
                paths = _glob_images(d)
                if paths:
                    return ImageList(paths), {"source": f"openood folder {d.relative_to(root)}", "root": "openood",
                                              "n_found": len(paths)}
    raise DatasetUnavailable(f"{name}: no images under {root}/images_classic/{_CLASSIC_DIRS.get(name, (name,))}")


def _read_imglist(rel_candidates):
    root = openood_root()
    if root is None:
        raise DatasetUnavailable("no OpenOOD root")
    for rel in rel_candidates:
        for p in (root / "benchmark_imglist" / rel, root / rel):
            if p.is_file():
                paths, labels = [], []
                for ln in p.read_text().splitlines():
                    if not ln.strip():
                        continue
                    f = ln.split()
                    paths.append(f[0])
                    labels.append(int(f[1]) if len(f) > 1 else -1)
                return root, p, paths, np.asarray(labels, np.int64)
    raise DatasetUnavailable(f"imglist {rel_candidates[0]} not found under {root}")


def _openood_list(rel_candidates):
    """ImageList from an OpenOOD imglist ('relpath label' per line). The image base directory is
    decided once, from the first entry, so a missing image tree fails here and not hours into extraction."""
    root, lst, rels, labels = _read_imglist(rel_candidates)
    base = next((b for b in (root / "images_largescale", root / "images_classic", root)
                 if (b / rels[0]).is_file()), None)
    if base is None:
        # OpenOOD imglists often use paths like `tin/test/...` relative to images_classic
        base = next((b for b in (root / "images_largescale", root / "images_classic", root)
                     if any((b / r).is_file() for r in rels[:20])), None)
    if base is None:
        raise DatasetUnavailable(f"{lst.name}: first image {rels[0]} not found under {root}")
    missing = sum(1 for r in rels if not (base / r).is_file())
    if missing:
        raise DatasetUnavailable(f"{lst.name}: {missing}/{len(rels)} listed images missing under {base}")
    return ImageList([base / r for r in rels], labels), {
        "source": f"openood imglist {lst.relative_to(root)}", "root": "openood", "n_listed": len(rels),
        "imglist_path": str(lst.relative_to(root)),
        "imglist_sha256": hashlib.sha256(lst.read_bytes()).hexdigest()}


def _resolve_imglist_file(rel_candidates):
    """Return (root, path) for the first existing imglist among candidates, searching openood root
    and the phasec_data/scood install layout."""
    roots = []
    oo = openood_root()
    if oo is not None:
        roots.append(oo)
    roots.append(Path(data_roots.root("phasec_data")) / "scood")
    roots.append(Path(data_roots.root("phasec_data")) / "openood")
    tried = []
    for root in roots:
        if not root.is_dir():
            continue
        for rel in rel_candidates:
            for p in (root / rel, root / "data" / rel, root / Path(rel).name):
                tried.append(str(p))
                if p.is_file() and p.stat().st_size > 100:
                    return root, p
    raise DatasetUnavailable(
        f"filtered TIN imglist not found (tried {len(tried)} paths under openood/scood). "
        f"Install OpenOOD benchmark_imglist (Drive id 1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1) or "
        f"SCOOD data/imglist so one of {list(rel_candidates)} exists. "
        f"Raw TinyImageNet folder loading is forbidden (would reintroduce ~1,207 CIFAR-10-class overlaps).")


def _cifar10_tin_filtered():
    """OpenOOD/SCOOD filtered TinyImageNet OOD for CIFAR-10 — never the raw tin/ tree.

    OpenOOD removes ~1,207 TIN images that share CIFAR-10 class semantics (SCOOD re-split);
    v1.5 additionally holds out 1,000 images / 20 categories as OOD val. The imglist path and
    SHA-256 are recorded so the manifest can show exactly which filter was used.
    """
    root, lst = _resolve_imglist_file(CIFAR10_TIN_IMGLISTS)
    # Also accept SCOOD-style lists that mark ID-overlap rows with a non-negative label: keep only OOD (label < 0).
    paths, labels = [], []
    for ln in lst.read_text().splitlines():
        if not ln.strip():
            continue
        f = ln.split()
        lab = int(f[1]) if len(f) > 1 else -1
        # SCOOD imglists mark the 1,207 redirected-ID images with CIFAR-10 class ids (>=0);
        # OpenOOD test_tin.txt lists only the retained OOD rows (label -1). Keep OOD only.
        if lab >= 0:
            continue
        paths.append(f[0])
        labels.append(-1)
    if not paths:
        # OpenOOD lists are OOD-only with optional missing label — re-read without the filter.
        for ln in lst.read_text().splitlines():
            if not ln.strip():
                continue
            f = ln.split()
            paths.append(f[0])
            labels.append(int(f[1]) if len(f) > 1 else -1)
    # Resolve image base: try path as written, then under images_classic/, then tin/ only.
    def _exists(base, rel):
        return (base / rel).is_file()

    bases = [root, root / "images", root / "images_classic", root / "data" / "images",
             Path(data_roots.root("phasec_data")) / "openood" / "images_classic",
             Path(data_roots.root("phasec_data")) / "openood"]
    # de-dup while preserving order
    seen, bases_u = set(), []
    for b in bases:
        s = str(b.resolve()) if b.exists() else str(b)
        if s not in seen:
            seen.add(s); bases_u.append(b)
    base = next((b for b in bases_u if _exists(b, paths[0])), None)
    if base is None:
        # Some lists use bare filenames under tin/test/images/
        tin_test = Path(data_roots.root("phasec_data")) / "openood" / "images_classic" / "tin" / "test" / "images"
        if tin_test.is_dir() and (tin_test / Path(paths[0]).name).is_file():
            abs_paths = [tin_test / Path(p).name for p in paths]
            missing = sum(1 for p in abs_paths if not p.is_file())
            if missing:
                raise DatasetUnavailable(f"TIN imglist {lst}: {missing}/{len(abs_paths)} files missing under {tin_test}")
            return ImageList(abs_paths, labels), {
                "source": f"filtered TIN imglist {lst} (resolved by basename under tin/test/images)",
                "imglist_path": str(lst), "imglist_sha256": hashlib.sha256(lst.read_bytes()).hexdigest(),
                "n_listed": len(abs_paths), "openood_group": "near_ood",
                "filter_note": "OpenOOD/SCOOD CIFAR-10-class overlap filter (~1,207 removed); raw tin/ forbidden"}
        raise DatasetUnavailable(f"TIN imglist {lst}: first image {paths[0]} not found under {bases_u}")
    abs_paths = [base / p for p in paths]
    missing = sum(1 for p in abs_paths if not p.is_file())
    if missing:
        raise DatasetUnavailable(f"TIN imglist {lst}: {missing}/{len(abs_paths)} files missing under {base}")
    try:
        rel_lst = str(lst.relative_to(root))
    except ValueError:
        rel_lst = str(lst)
    return ImageList(abs_paths, labels), {
        "source": f"filtered TIN imglist {rel_lst}",
        "imglist_path": rel_lst, "imglist_sha256": hashlib.sha256(lst.read_bytes()).hexdigest(),
        "n_listed": len(abs_paths), "openood_group": "near_ood",
        "filter_note": "OpenOOD/SCOOD CIFAR-10-class overlap filter (~1,207 removed); raw tin/ forbidden"}


_IMAGENET200_LISTS = {
    "id_train": ("imagenet200/train_imagenet200.txt", "train_imagenet200.txt"),
    "id_test": ("imagenet200/test_imagenet200.txt", "test_imagenet200.txt"),
    "ood_ref": ("imagenet200/val_openimage_o.txt", "val_openimage_o.txt"),
    "ood_ssb_hard": ("imagenet200/test_ssb_hard.txt", "test_ssb_hard.txt"),
    "ood_ninco": ("imagenet200/test_ninco.txt", "test_ninco.txt"),
    "ood_inaturalist": ("imagenet200/test_inaturalist.txt", "test_inaturalist.txt"),
    "ood_openimage_o": ("imagenet200/test_openimage_o.txt", "test_openimage_o.txt"),
    "ood_texture": ("imagenet200/test_textures.txt", "test_textures.txt"),
}


# ---------------------------------------------------------------- split registry
def split_names(id_set):
    return ["id_train", "id_test", "ood_ref"] + [f"ood_{n}" for n in OOD_NAMES[id_set]]


def _raw_split(id_set, split, download):
    """(base dataset yielding PIL/label, info). Un-subsampled."""
    if id_set == "cifar10":
        if split in ("id_train", "id_test"):
            return _tv_cifar(10, split == "id_train", download)
        name = "ref" if split == "ood_ref" else split[4:]
        if name == "ref":
            ds, info = _tv_cifar(100, True, download)
            return ds, {**info, "role": "adapter-construction reference (CIFAR-100 train), never evaluated"}
        if name == "cifar100":
            ds, info = _tv_cifar(100, False, download)
            return ds, {**info, "openood_group": "near_ood"}
        tv = {"svhn": _tv_svhn, "texture": _tv_dtd, "places365": _tv_places, "mnist": _tv_mnist}
        if name in tv:
            try:
                ds, info = tv[name](download)
                group = "near_ood" if name in CIFAR10_NEAR_OOD else ("far_ood" if name in CIFAR10_FAR_OOD else None)
                if group:
                    info = {**info, "openood_group": group}
                return ds, info
            except DatasetUnavailable as e_tv:
                try:
                    ds, info = _openood_folder(name)
                    group = "near_ood" if name in CIFAR10_NEAR_OOD else ("far_ood" if name in CIFAR10_FAR_OOD else None)
                    return ds, {**info, **({"openood_group": group} if group else {})}
                except DatasetUnavailable as e_oo:
                    raise DatasetUnavailable(f"{name}: torchvision: {e_tv}; openood: {e_oo}")
        if name == "tin":
            # Hard requirement: filtered imglist only. Never fall back to raw images_classic/tin.
            return _cifar10_tin_filtered()
        if name == "lsun":
            return _openood_folder(name)
    elif id_set == "imagenet200":
        if split in _IMAGENET200_LISTS:
            ds, info = _openood_list(_IMAGENET200_LISTS[split])
            if split == "ood_ref":
                info["role"] = "adapter-construction reference (OpenOOD val_openimage_o), never evaluated"
            return ds, info
    raise DatasetUnavailable(f"unknown split {id_set}/{split}")


def build_split(id_set, split, transform, download=False):
    """(WithTransform dataset, info dict) for one split. Applies the documented subsampling.

    info: source, n_source, n_used, indices (np.int64 into the source dataset), subsample_seed
    (None when everything is used), plus provider fields. OOD labels are -1.
    """
    if id_set not in ID_SETS:
        raise KeyError(id_set)
    base, info = _raw_split(id_set, split, download)
    n_source = len(base)
    seed, cap = None, None
    if split.startswith("ood_"):
        cap = MAX_REF if split == "ood_ref" else MAX_OOD
        seed = SEEDS[split]
        base, idx = subsample(base, cap, seed)
        if len(idx) == n_source:
            seed = None
    elif id_set == "imagenet200" and split in ("id_train", "id_test"):
        if split == "id_train":
            labels = np.asarray(base.labels)
            rng = np.random.default_rng(SEEDS["imagenet200/id_train"])
            keep = []
            for c in np.unique(labels):
                ci = np.flatnonzero(labels == c)
                keep.append(ci if len(ci) <= IMAGENET200_TRAIN_PER_CLASS
                            else rng.choice(ci, IMAGENET200_TRAIN_PER_CLASS, replace=False))
            idx = np.sort(np.concatenate(keep))
            base = torch.utils.data.Subset(base, idx.tolist())
            seed, cap = SEEDS["imagenet200/id_train"], IMAGENET200_TRAIN_PER_CLASS
        else:
            idx = np.arange(n_source)
    else:
        idx = np.arange(n_source)
    info = {**info, "n_source": n_source, "n_used": len(idx), "indices": np.asarray(idx, np.int64),
            "subsample_seed": seed, "cap": cap}
    return WithTransform(base, transform, ood=split.startswith("ood_")), info


# ---------------------------------------------------------------- preprocessing guard
def write_preprocess_json(path, spec):
    """Write the pinned preprocessing spec (a dict with a "pipeline" list) as stable JSON; returns its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


#: Exact dataset-level contract pinned in every backbone preprocess JSON and
#: enforced by WithTransform before the torchvision pipeline runs. Grayscale
#: OOD (MNIST) must become 3-channel here — never by an ad-hoc loader fork.
DATASET_LEVEL_RGB = "PIL image .convert('RGB') before the pipeline"


def assert_transform(tf, json_path):
    """Raise SystemExit if the live transform differs from the pinned record in any op, size,
    interpolation, antialias flag, mean or std, or if the dataset-level RGB convert is missing.

    The RGB convert is not a torchvision op (PIL mode change before the pipeline). It is
    pinned under `dataset_level` and applied by `WithTransform` / `ImageList` so single-
    channel sets (MNIST) take the same channel-replication + resize + Normalize path as
    every other rung, under the same guard.
    """
    pin = json.load(open(json_path))
    pinned_level = pin.get("dataset_level")
    if pinned_level != DATASET_LEVEL_RGB:
        raise SystemExit(
            f"preprocessing drift: dataset_level {pinned_level!r} vs pinned {DATASET_LEVEL_RGB!r} "
            f"(grayscale→RGB must be the pinned contract, not a loader-side fork)"
        )
    spec = pin["pipeline"]
    live = describe_transform(tf)
    got, want = [s["op"] for s in live], [s["op"] for s in spec]
    if got != want:
        raise SystemExit(f"preprocessing drift: live {got} vs pinned {want}")
    for l, p in zip(live, spec):
        for k in ("size", "interpolation", "antialias"):
            if k in p and l.get(k) != p[k]:
                raise SystemExit(f"preprocessing drift in {p['op']}.{k}: live {l.get(k)} vs pinned {p[k]}")
        if p["op"] == "Normalize":
            if not (np.allclose(l["mean"], p["mean"]) and np.allclose(l["std"], p["std"])):
                raise SystemExit(f"normalisation drift: live {l['mean']}/{l['std']} vs pinned {p['mean']}/{p['std']}")
    print(f"[datasets] preprocessing matches the pinned record {Path(json_path).name}: "
          f"dataset_level=RGB + {got}")
    return True


# ---------------------------------------------------------------- status CLI
def main():
    ap = argparse.ArgumentParser(description="report which Phase C splits are available")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--download", action="store_true", help="let torchvision download missing archives")
    ap.add_argument("--id-set", default=None, choices=sorted(ID_SETS))
    a = ap.parse_args()
    from torchvision import transforms as T
    tf = T.ToTensor()
    for ids in ([a.id_set] if a.id_set else sorted(ID_SETS)):
        print(f"== {ids}")
        for sp in split_names(ids):
            try:
                ds, info = build_split(ids, sp, tf, download=a.download)
                print(f"  ok    {sp:<18} n_used={info['n_used']:>6} / {info['n_source']:<6} seed={info['subsample_seed']}  {info['source']}")
            except DatasetUnavailable as e:
                print(f"  --    {sp:<18} {str(e)[:160]}")


if __name__ == "__main__":
    main()
