"""
Phase 3 -- perceptual-hash overlap check against ISIC 2019 source images.

The embedding screen in phase3_overlap_screen.py is the primary instrument and
passed cleanly. This is the conventional check a reviewer expects, run in
addition rather than instead: it detects near-identical pixels, where the
embedding screen detects near-identical content. The two fail in different
directions, which is the reason to run both -- a crop or a rescale that the
encoder maps to the same place is caught by neither alone.

Requires the ISIC 2019 training images on disk. Point --isic at the directory;
the script walks it for ISIC_*.jpg and hashes every file, so a partial download
degrades to a partial check and says so rather than silently passing.

    python3 PhaseB/analysis/phase3_phash_overlap.py --isic /path/to/ISIC_2019_Training_Input
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase3"
HAMMING_FLAG = 8          # of 64 bits; <=8 is the conventional near-duplicate line


def phash(path, size=32, keep=8):
    """DCT perceptual hash, 64 bits. Standard construction, no library needed."""
    from scipy.fftpack import dct
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(im, float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:keep, :keep]
    med = np.median(d[1:].ravel() if d.size > 1 else d.ravel())
    return np.packbits((d > med).ravel())


def hamming(A, b):
    return np.unpackbits(A ^ b[None, :], axis=1).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.isic.rglob("ISIC_*.jpg"))
    if not files:
        raise SystemExit(f"no ISIC_*.jpg under {args.isic}")
    print(f"[phash] hashing {len(files)} ISIC images")
    H = np.vstack([phash(f) for f in files])
    ids = np.array([f.stem for f in files])

    # how close do two genuinely distinct ISIC images get? calibrates the line
    sample = np.random.default_rng(0).choice(len(H), min(2000, len(H)), replace=False)
    d_within = []
    for i in sample:
        d = np.unpackbits(H ^ H[i][None, :], axis=1).sum(1)
        d[i] = 65
        d_within.append(d.min())
    d_within = np.array(d_within)
    print(f"[phash] within-ISIC nearest hamming: median {np.median(d_within):.0f}, "
          f"p1 {np.percentile(d_within,1):.0f}, min {d_within.min()} "
          f"({int((d_within<=HAMMING_FLAG).sum())} of {len(sample)} are "
          f"near-duplicates of another ISIC image)")

    rows, summary = [], []
    for f in sorted(OUT.glob("external_*.npz")):
        name = f.stem.replace("external_", "")
        d = np.load(f, allow_pickle=True)
        paths = [Path(p) for p in d["paths"]]
        hits = 0
        for j, p in enumerate(paths):
            dd = hamming(H, phash(p))
            k = int(dd.argmin())
            if dd[k] <= HAMMING_FLAG:
                hits += 1
                rows.append(dict(dataset=name, case=str(d["case"][j]), path=str(p),
                                 isic_id=ids[k], hamming=int(dd[k])))
        summary.append(dict(dataset=name, n=len(paths), flagged=hits,
                            frac=hits / len(paths), threshold_bits=HAMMING_FLAG,
                            n_isic_hashed=len(files)))
        print(f"[phash] {name}: n={len(paths)}  flagged={hits} "
              f"({hits/len(paths)*100:.2f}%)")

    pd.DataFrame(summary).to_csv(OUT / "phash_overlap_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "phash_overlap_flagged.csv", index=False)
    print(f"\n[phash] wrote phash_overlap_summary.csv ({len(rows)} flagged)")


if __name__ == "__main__":
    main()
