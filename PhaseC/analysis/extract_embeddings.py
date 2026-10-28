"""Extract frozen-backbone embeddings for one (backbone, ID set) and write them with their provenance.

Why: the audit needs nothing from the images except the embedding each frozen backbone
assigns them, so extraction is done once, pinned, and every later script reads arrays.
Before any embedding is written the live torchvision transform is compared with the pinned
preprocessing record (assets/preprocessing/<backbone>.json). The pin must already exist and be
readable — a missing pin aborts; the job never auto-writes a pin and continues. A passing check
is recorded to transform_guard.json before the first embedding write so a silent absence is
detectable afterwards.

    python3 PhaseC/analysis/extract_embeddings.py --backbone resnet50_imagenet --id-set cifar10 --device cuda

Layout, under PhaseC/assets/embeddings/:
    <backbone>/preprocess.json          the pin the live transform was asserted against
    <backbone>/checkpoint_sha256.txt    state-dict SHA-256 (+ file SHA-256 and URL where a file exists)
    <backbone>/<id_set>/<split>.npz     embeddings (float32, n x d), labels (int64; ID class, -1 for OOD),
                                        indices (int64, row in the source dataset), and the provenance
                                        scalars (ckpt_sha256, preprocess_sha256, subsample_seed, n_source)
    <backbone>/<id_set>/<split>.npz.sha256   file and array-content SHA-256 (the zip embeds timestamps,
                                        so the content hash is the reproducible one)
    <backbone>/<id_set>/metadata.csv    one row per split: source, counts, seed, hashes, timings
Splits: id_train, id_test, ood_ref (adapter construction only), ood_<name> for each rung of the ladder.
An existing split is skipped when its sidecar hash matches the array contents and the stored checkpoint and
preprocessing hashes match the current ones; --force re-extracts. A split whose data is unavailable is logged
and skipped; the exit code is still 0 so a partial ladder can be audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backbones  # noqa: E402
import datasets_phasec as dsp  # noqa: E402

PHASEC = HERE.parent
EMB_ROOT = PHASEC / "assets" / "embeddings"


def array_sha256(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(p):
    return backbones.sha256_file(p)


def _read_sidecar(p: Path):
    side = Path(str(p) + ".sha256")
    if not (p.is_file() and side.is_file()):
        return None
    try:
        return json.loads(side.read_text())
    except json.JSONDecodeError:
        return None


def is_current(p: Path, ckpt_sha, pre_sha) -> bool:
    side = _read_sidecar(p)
    if side is None:
        return False
    try:
        z = np.load(p, allow_pickle=False)
        ok = (array_sha256(z["embeddings"], z["labels"], z["indices"]) == side["content_sha256"]
              and str(z["ckpt_sha256"]) == ckpt_sha and str(z["preprocess_sha256"]) == pre_sha)
    except Exception:
        return False
    return ok


@torch.no_grad()
def embed(model, ds, device, batch_size, num_workers, name, dim):
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=(device.type == "cuda"))
    feats, labels, t0, seen = [], [], time.perf_counter(), 0
    for x, y in dl:
        feats.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
        seen += len(y)
        if (seen // batch_size) % 20 == 0 or seen == len(ds):
            print(f"[extract] {name}: {seen}/{len(ds)}  {seen / (time.perf_counter() - t0):.0f} img/s", flush=True)
    E = torch.cat(feats).numpy().astype(np.float32)
    y = torch.cat(labels).numpy().astype(np.int64)
    if E.shape != (len(ds), dim):
        raise SystemExit(f"bad embedding shape {E.shape}, expected ({len(ds)}, {dim})")
    if not np.isfinite(E).all():
        raise SystemExit(f"{name}: non-finite embeddings")
    return E, y


def update_metadata(path: Path, row: dict):
    df = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    if len(df) and "split" in df:
        df = df[df.split != row["split"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    order = [s for s in ("id_train", "id_test", "ood_ref") if s in set(df.split)] + sorted(
        s for s in set(df.split) if s.startswith("ood_") and s != "ood_ref")
    df.set_index("split").loc[order].reset_index().to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True, choices=backbones.BACKBONE_NAMES)
    ap.add_argument("--id-set", default="cifar10", choices=sorted(dsp.ID_SETS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--splits", nargs="*", default=None, help="subset of splits (default: all for the ID set)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--download", action="store_true", help="allow torchvision to download missing archives")
    ap.add_argument("--out", type=Path, default=EMB_ROOT)
    a = ap.parse_args(argv)

    bb = backbones.get(a.backbone)
    if a.id_set not in bb.id_sets:
        print(f"[extract] {a.backbone} is not defined for ID set {a.id_set} (supports {bb.id_sets}); skipping")
        return 0
    device = torch.device(a.device)
    if device.type == "cuda":
        # strict fp32: no TF32 in matmul or conv, so the embeddings do not depend on the GPU generation
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    pin = backbones.pin_path(a.backbone)
    if not pin.is_file():
        raise SystemExit(
            f"[extract] TRANSFORM GUARD FAILED: pinned preprocessing missing for {a.backbone}: {pin}. "
            f"Write the pin deliberately (backbones.py --write-pins) and commit it before extract; "
            f"auto-writing a pin and continuing is forbidden.")
    tf = bb.build_transform()
    # Fail closed: assert_transform converts any exception into SystemExit. Do not wrap this.
    guard = dsp.assert_transform(tf, pin)
    pre_sha = guard["pin_sha256"]

    try:
        model = bb.load_model(device)
    except FileNotFoundError as e:
        print(f"[extract] {a.backbone}: checkpoint unavailable -- {e}")
        return 0
    backbones.check_pretrained_cfg(bb)
    ckpt_sha = bb.checkpoint_sha256
    print(f"[extract] {a.backbone} dim={bb.dim} checkpoint state_dict sha256={ckpt_sha[:16]}...")

    bdir = a.out / a.backbone
    odir = bdir / a.id_set
    odir.mkdir(parents=True, exist_ok=True)
    # Receipt before the first embedding write: silent absence of this file means the guard did not run.
    receipt = {
        **guard,
        "backbone": a.backbone,
        "id_set": a.id_set,
        "ckpt_sha256": ckpt_sha,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "before_first_write": True,
    }
    (odir / "transform_guard.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[extract] transform guard PASSED; receipt {odir / 'transform_guard.json'}")
    shutil.copyfile(pin, bdir / "preprocess.json")
    def _prov_line(k, v):
        if isinstance(v, (list, dict, tuple)):
            return f"{k} {json.dumps(v)}\n"
        return f"{k} {v}\n"
    (bdir / "checkpoint_sha256.txt").write_text(
        f"state_dict_sha256 {ckpt_sha}\n" + "".join(
            _prov_line(k, v) for k, v in bb.provenance.items()
            if k != "state_dict_sha256" and v is not None))

    splits = a.splits or dsp.split_names(a.id_set)
    for sp in splits:
        out = odir / f"{sp}.npz"
        if not a.force and is_current(out, ckpt_sha, pre_sha):
            print(f"[extract] {sp}: up to date (sidecar hash matches), skipping")
            continue
        try:
            ds, info = dsp.build_split(a.id_set, sp, tf, download=a.download)
        except dsp.DatasetUnavailable as e:
            print(f"[extract] {sp}: UNAVAILABLE, skipped -- {str(e)[:300]}")
            continue
        print(f"[extract] {sp}: {info['n_used']} of {info['n_source']} images, source={info['source']}, "
              f"subsample_seed={info['subsample_seed']}")
        t0 = time.perf_counter()
        E, y = embed(model, ds, device, a.batch_size, a.num_workers, f"{a.backbone}/{a.id_set}/{sp}", bb.dim)
        np.savez(out, embeddings=E, labels=y, indices=info["indices"],
                 ckpt_sha256=np.array(ckpt_sha), preprocess_sha256=np.array(pre_sha),
                 subsample_seed=np.array(-1 if info["subsample_seed"] is None else info["subsample_seed"], np.int64),
                 n_source=np.array(info["n_source"], np.int64))
        content = array_sha256(E, y, info["indices"])
        fsha = file_sha256(out)
        Path(str(out) + ".sha256").write_text(json.dumps(
            {"file": out.name, "file_sha256": fsha, "content_sha256": content, "n": int(len(E)), "dim": int(E.shape[1])},
            indent=1) + "\n")
        update_metadata(odir / "metadata.csv", {
            "split": sp, "n": len(E), "dim": E.shape[1], "source": info["source"], "n_source": info["n_source"],
            "subsample_seed": info["subsample_seed"], "cap": info.get("cap"),
            "openood_group": info.get("openood_group"),
            "imglist_path": info.get("imglist_path"), "imglist_sha256": info.get("imglist_sha256"),
            "mean_norm": float(np.linalg.norm(E, axis=1).mean()), "n_classes": int(len(np.unique(y[y >= 0]))),
            "content_sha256": content, "file_sha256": fsha, "ckpt_sha256": ckpt_sha, "preprocess_sha256": pre_sha,
            "torch": torch.__version__, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[extract] wrote {out}  {E.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
