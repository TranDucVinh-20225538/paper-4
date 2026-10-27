"""Write PhaseC/outputs/manifest.json: everything needed to say what produced every number.

Why: the audit's numbers are only as reproducible as the inputs they were computed from. The manifest pins, in one
file that can be diffed between runs, the frozen checkpoints (state-dict SHA-256, file SHA-256 and URL), the dataset
sources, versions, split sizes and subsample seeds that went into every embedding file, the preprocessing records,
every random seed any script draws from, the library versions, the GPU the embeddings were computed on, and the
SHA-256 of every embedding file, adapter file and output CSV (hashes only; the arrays are not copied here), plus
the hash of each analysis script and the git commit it ran at.

    python3 PhaseC/analysis/write_manifest.py
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PHASEC = HERE.parent
REPO = PHASEC.parent


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return p.name


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=REPO)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:
        return None


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("torch", "torchvision", "timm", "numpy", "sklearn", "scipy", "pandas", "PIL", "huggingface_hub", "gdown"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception as e:
            out[m] = f"unavailable ({type(e).__name__})"
    try:
        import torch
        out["cuda_runtime"] = torch.version.cuda
        out["cudnn"] = str(torch.backends.cudnn.version())
    except Exception:
        pass
    return out


def gpu():
    q = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return {"nvidia_smi_here": q.splitlines() if q else None,
            "note": "the GPU the embeddings were computed on is in embeddings.<backbone>.<id_set>.<split>.device_name; "
                    "nvidia-smi here describes the machine that wrote this manifest, which may be a login node"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=PHASEC / "outputs" / "manifest.json")
    a = ap.parse_args(argv)

    import backbones
    import datasets_phasec as dsp
    import run_audit as ra
    import train_adapters as ta

    emb_root, ad_root, pre_dir = PHASEC / "assets" / "embeddings", PHASEC / "assets" / "adapters", PHASEC / "assets" / "preprocessing"
    man = {"written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "git": {"commit": run(["git", "rev-parse", "HEAD"]), "dirty_files": (run(["git", "status", "--porcelain", "PhaseC"]) or "").splitlines()[:50]},
           "packages": versions(), "gpu": gpu()}

    man["backbones"] = {}
    for n, bb in backbones.REGISTRY.items():
        entry = {"kind": bb.kind, "dim": bb.dim, "source_url": bb.source_url, "id_sets": list(bb.id_sets),
                 "description": bb.description, "frozen_from": bb.frozen_from, "preprocess_pipeline": bb.pipeline}
        pin = backbones.pin_path(n)
        if pin.is_file():
            entry["preprocess_pin"] = {"file": rel(pin), "sha256": sha(pin)}
        ck = emb_root / n / "checkpoint_sha256.txt"
        if ck.is_file():
            for ln in ck.read_text().splitlines():
                k, _, v = ln.partition(" ")
                if not k:
                    continue
                if v[:1] in "[{":
                    try:
                        v = json.loads(v)
                    except json.JSONDecodeError:
                        pass
                entry.setdefault("checkpoint", {})[k] = v
        else:
            entry["checkpoint"] = None
        man["backbones"][n] = entry

    man["embeddings"] = {}
    for md in sorted(emb_root.glob("*/*/metadata.csv")):
        bbn, ids = md.parts[-3], md.parts[-2]
        df = pd.read_csv(md)
        for _, r in df.iterrows():
            f = md.parent / f"{r.split}.npz"
            side = md.parent / f"{r.split}.npz.sha256"
            emb = {
                "file": rel(f), "n": int(r.n), "dim": int(r.dim), "source": r.source, "n_source": int(r.n_source),
                "subsample_seed": None if pd.isna(r.subsample_seed) else int(r.subsample_seed),
                "file_sha256": r.file_sha256, "content_sha256": r.content_sha256,
                "ckpt_sha256": r.ckpt_sha256, "preprocess_sha256": r.preprocess_sha256,
                "device_name": r.get("device_name", None) if "device_name" in r else None,
                "torch": r.torch, "sidecar": json.loads(side.read_text()) if side.is_file() else None}
            for col in ("openood_group", "imglist_path", "imglist_sha256"):
                if col in r and not (isinstance(r[col], float) and pd.isna(r[col])):
                    emb[col] = r[col]
            man["embeddings"].setdefault(bbn, {}).setdefault(ids, {})[r.split] = emb
    man["datasets"] = {"id_sets": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                   for k, v in dsp.ID_SETS.items()},
                       "cifar10_near_ood": list(dsp.CIFAR10_NEAR_OOD),
                       "cifar10_far_ood": list(dsp.CIFAR10_FAR_OOD),
                       "cifar10_tin_imglists": list(dsp.CIFAR10_TIN_IMGLISTS),
                       "max_ood": dsp.MAX_OOD, "max_ref": dsp.MAX_REF,
                       "imagenet200_train_per_class": dsp.IMAGENET200_TRAIN_PER_CLASS,
                       "archive_md5": {"cifar10": dsp.CIFAR10_MD5, "cifar100": dsp.CIFAR100_MD5},
                       "difficulty_axis": (
                           "OpenOOD near/far labels are informational only. Reported difficulty ordering "
                           "and reach-vs-difficulty plots use measured identity-baseline AUROC per "
                           "(backbone, ood_set, scorer) from generalisation_audit.csv."),
                       "tin_filter_note": (
                           "CIFAR-10 TIN must come from an OpenOOD/SCOOD filtered imglist "
                           "(~1,207 CIFAR-10-class overlaps removed). Raw images_classic/tin is refused."),
                       "note": "per-split source, counts, imglist_sha256 and seeds are under `embeddings`"}

    # Surface ResNet-18 OpenOOD version pairing explicitly (v1 ckpt + v1.5 far-OOD is easy to miss).
    r18 = man["backbones"].get("resnet18_cifar_openood", {})
    ck = r18.get("checkpoint") or {}
    far_pub = ck.get("openood_far_ood_as_published")
    if isinstance(far_pub, str):
        # checkpoint_sha256.txt stores the list via str(list); recover tokens that look like dataset names
        far_pub = [t.strip("[]',\"") for t in far_pub.replace(",", " ").split() if t.strip("[]',\"")]
    man["openood_resnet18_pairing"] = {
        "checkpoint_version": ck.get("openood_checkpoint_version"),
        "checkpoint_id": ck.get("openood_checkpoint_id"),
        "file_sha256": ck.get("file_sha256"),
        "state_dict_sha256": ck.get("state_dict_sha256"),
        "source_url": ck.get("source_url") or r18.get("source_url"),
        "benchmark_pairing_note": ck.get("benchmark_pairing_note"),
        "far_ood_ladder_used": list(dsp.CIFAR10_FAR_OOD),
        "far_ood_as_published_for_this_ckpt": far_pub,
    }

    man["seeds"] = {
        "dataset_subsampling": dsp.SEEDS,
        "adapter_training": list(ta.ADAPTER_SEEDS),
        "operator_seeds": ra.OPERATOR_SEEDS, "operator_rng_stream_ids": ra.OPERATOR_STREAM,
        "pair_sampling": ra.PAIR_SEED, "bounds_subsample": ra.BOUNDS_SEED, "audit_smoke_subsample": ra.AUDIT_SUBSAMPLE_SEED}
    man["protocol"] = {
        "temper_gammas": ra.TEMPER_GAMMAS, "zca_alphas": ra.ZCA_ALPHAS, "residual_t": ra.RESIDUAL_T,
        "adapter_alphas": ra.ADAPTER_ALPHAS, "general_affine_cond": ra.GENERAL_AFFINE_COND,
        "check1_thresholds": ra.THRESHOLDS, "check1_grid": {"tau": ra.SENS_TAU, "S_high": ra.SENS_SHIGH},
        "n_pairs_default": ra.reach.N_PAIRS, "bounds_n_per_population": ra.BOUNDS_N_PER_POP,
        "ratio_valid_min": ra.RATIO_VALID_MIN, "ratio_preferred_spearman": ra.RATIO_PREFERRED_SPEARMAN,
        "scorers": list(ra.scorers.SCORER_NAMES), "reg_eps": ra.scorers.REG_EPS, "adapter_hp": ta.HP}

    man["adapters"] = {}
    for p in sorted(ad_root.glob("*/*/adapter_seed*.pt")):
        man["adapters"].setdefault(p.parts[-3], {}).setdefault(p.parts[-2], {})[p.stem] = {"file": rel(p), "sha256": sha(p)}

    man["outputs"] = {}
    for p in sorted((PHASEC / "outputs").glob("*")):
        if p.is_file() and p.name != "manifest.json":
            man["outputs"][p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
    man["scripts"] = {rel(p): sha(p) for p in sorted(HERE.glob("*.py"))}
    man["scripts"][rel(REPO / "PhaseB/analysis/data_roots.py")] = sha(REPO / "PhaseB/analysis/data_roots.py")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(man, indent=1, default=str) + "\n")
    print(f"[manifest] wrote {rel(a.out)}: {len(man['embeddings'])} backbones with embeddings, "
          f"{sum(len(v) for v in man['adapters'].values())} adapter groups, {len(man['outputs'])} output files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
