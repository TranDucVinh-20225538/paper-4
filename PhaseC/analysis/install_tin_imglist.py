"""Install the OpenOOD/SCOOD filtered TIN imglist used by PhaseC CIFAR-10.

Why: TIN as OOD for CIFAR-10 must exclude the ~1,207 images that share CIFAR-10
class semantics (SCOOD re-split). Loading the raw TinyImageNet dump contaminates
baseline AUROC. This script installs a filtered imglist under
`PhaseC/data/openood/benchmark_imglist/cifar10/test_tin.txt` from either:

  --from-scood-zip PATH   SCOOD scood_benchmark.zip (extracts only imglist entries)
  --from-openood-zip PATH OpenOOD benchmark_imglist.zip
  --from-file PATH        a ready test_tin.txt (copied into place)

After install, `datasets_phasec.build_split('cifar10','ood_tin',...)` records the
imglist path and SHA-256 in the embedding metadata / manifest.

    python3 PhaseC/analysis/install_tin_imglist.py --from-scood-zip PhaseC/data/scood/scood_benchmark.zip
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASEC = HERE.parent
OUT_DIR = PHASEC / "data" / "openood" / "benchmark_imglist" / "cifar10"
OUT = OUT_DIR / "test_tin.txt"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write(lines: list[str], note: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = "".join(ln if ln.endswith("\n") else ln + "\n" for ln in lines)
    OUT.write_text(text)
    print(f"[tin-imglist] wrote {OUT} ({len(lines)} lines) sha256={sha(OUT)}")
    print(f"[tin-imglist] source: {note}")
    return OUT


def from_file(src: Path) -> Path:
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    if len(lines) < 1000:
        raise SystemExit(f"{src} looks too short ({len(lines)} lines) for a TIN OOD list")
    return _write(lines, f"copied from {src}")


def from_openood_zip(zpath: Path) -> Path:
    with zipfile.ZipFile(zpath) as zf:
        cands = [n for n in zf.namelist() if n.endswith("cifar10/test_tin.txt") or n.endswith("/test_tin.txt")]
        if not cands:
            raise SystemExit(f"no cifar10/test_tin.txt inside {zpath}; names sample: {zf.namelist()[:20]}")
        name = sorted(cands, key=len)[0]
        data = zf.read(name).decode()
    lines = [ln for ln in data.splitlines() if ln.strip()]
    return _write(lines, f"OpenOOD zip {zpath.name}::{name}")


def _scood_sc_label(meta: str) -> int:
    """SCOOD imglist meta is either an int label or a JSON blob with sc_label (semantic CIFAR label).

    sc_label >= 0  → redirected ID (CIFAR-10-class overlap; must be dropped from OOD)
    sc_label == -1 → true OOD
    """
    meta = meta.strip()
    if not meta:
        return -1
    if meta[:1] == "{":
        import json
        d = json.loads(meta)
        return int(d.get("sc_label", d.get("label", -1)))
    return int(meta.split()[0])


def from_scood_zip(zpath: Path) -> Path:
    """SCOOD ships data/imglist/... ; keep only OOD rows (sc_label < 0) for TinyImageNet.

    SCOOD stores TinyImageNet *val* images under the path prefix `tin/test/images/val_*.JPEG`
    inside its zip; the same bytes live at `tin/val/images/val_*.JPEG` in the OpenOOD layout
    we already extracted. Remap the prefix so the imglist resolves against images_classic/.
    """
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        prefer = [n for n in names if "imglist" in n.lower() and "tin" in n.lower()
                  and ("cifar10" in n.lower() or "cifar_10" in n.lower() or "benchmark_cifar10" in n.lower())]
        if not prefer:
            prefer = [n for n in names if "imglist" in n.lower() and n.endswith(".txt") and "tin" in n.lower()]
        if not prefer:
            sample = [n for n in names if "imglist" in n.lower()][:40]
            raise SystemExit(f"no TIN imglist inside {zpath}; imglist sample: {sample}")
        prefer = sorted(prefer, key=lambda n: (0 if "test" in n.lower() else 1, len(n)))
        name = prefer[0]
        raw = zf.read(name).decode()
        print(f"[tin-imglist] using SCOOD entry {name}")
    kept, skipped_id = [], 0
    for ln in raw.splitlines():
        if not ln.strip():
            continue
        path, _, meta = ln.partition(" ")
        sc = _scood_sc_label(meta)
        if sc >= 0:
            skipped_id += 1
            continue
        # Remap SCOOD's tin/test/images/val_* → OpenOOD tin/val/images/val_*
        if path.startswith("tin/test/images/val_"):
            path = "tin/val/images/" + path.split("/")[-1]
        elif path.startswith("tin/test/images/") and path.split("/")[-1].startswith("val_"):
            path = "tin/val/images/" + path.split("/")[-1]
        kept.append(f"{path} -1")
    print(f"[tin-imglist] kept {len(kept)} OOD rows, skipped {skipped_id} redirected-ID rows (sc_label>=0)")
    if skipped_id < 1000 or not (8500 <= len(kept) <= 9000):
        print(f"[tin-imglist] WARNING: expected ~1207 removed / ~8793 kept; got skipped={skipped_id} kept={len(kept)}")
    return _write(kept, f"SCOOD zip {zpath.name}::{name} (sc_label<0; skipped_id={skipped_id}; paths remapped to tin/val/images)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-scood-zip", type=Path)
    g.add_argument("--from-openood-zip", type=Path)
    g.add_argument("--from-file", type=Path)
    a = ap.parse_args(argv)
    if a.from_scood_zip:
        from_scood_zip(a.from_scood_zip)
    elif a.from_openood_zip:
        from_openood_zip(a.from_openood_zip)
    else:
        from_file(a.from_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
