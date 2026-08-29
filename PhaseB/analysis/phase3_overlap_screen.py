"""
Phase 3 -- near-duplicate screen against the frozen ISIC populations.

The advisor requires a patient/image overlap check before an external domain is
admitted.  The ISIC source images have been deleted, so a pixel or perceptual
hash against them cannot be computed.  This screen is the weaker instrument that
remains: under a fixed encoder a duplicate image yields a near-identical
embedding, so cosine similarity in PanDerm space detects duplicates without
needing the pixels back.  It cannot prove identity, only fail to find evidence
of it, and it is labelled that way wherever it is reported.

Threshold is calibrated rather than assumed, per LOCKED_PROTOCOL.md section 5:
the flag level is the 99.9th percentile of the within-ISIC nearest-neighbour
similarity among the 5,067 held-out images -- i.e. how similar two genuinely
distinct ISIC images get.  Anything an external image beats that is suspicious.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase3"
REF = ROOT / "assets" / "reference_embeddings"
FLAG_PCT = 99.9
DEMOTE_FRAC = 0.02


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def max_sim(a, b, block=512):
    """Max cosine similarity of each row of a against all of b."""
    out = np.empty(len(a), np.float32)
    idx = np.empty(len(a), np.int64)
    for i in range(0, len(a), block):
        s = a[i:i + block] @ b.T
        out[i:i + block] = s.max(1)
        idx[i:i + block] = s.argmax(1)
    return out, idx


def main():
    art = np.load(REF / "panderm_reference_embedding_artifact.npz", allow_pickle=True)
    dom = art["domain"].astype(str)
    isic_eval = unit(art["embeddings"][dom == "isic"].astype(np.float64))
    isic_train = unit(np.load(REF / "ReferenceTrainEmbedding" / "embeddings.npy").astype(np.float64))
    isic_all = np.vstack([isic_train, isic_eval])
    print(f"[screen] ISIC reference: {len(isic_train)} train + {len(isic_eval)} held-out "
          f"= {len(isic_all)}")

    # calibrate: how similar do two genuinely distinct ISIC held-out images get?
    s = isic_eval @ isic_eval.T
    np.fill_diagonal(s, -np.inf)
    within = s.max(1)
    thr = float(np.percentile(within, FLAG_PCT))
    print(f"[screen] within-ISIC NN similarity: median {np.median(within):.4f}, "
          f"p99 {np.percentile(within,99):.4f}, p{FLAG_PCT} {thr:.4f}, max {within.max():.4f}")

    rows, summary = [], []
    for f in sorted(OUT.glob("external_*.npz")):
        name = f.stem.replace("external_", "")
        d = np.load(f, allow_pickle=True)
        e = unit(d["embeddings"].astype(np.float64))
        ms, mi = max_sim(e, isic_all)
        flagged = ms >= thr
        frac = float(flagged.mean())
        summary.append(dict(dataset=name, n=len(e), flagged=int(flagged.sum()),
                            frac_flagged=frac, threshold=thr,
                            max_sim=float(ms.max()), median_sim=float(np.median(ms)),
                            demoted=frac > DEMOTE_FRAC))
        for j in np.where(flagged)[0]:
            rows.append(dict(dataset=name, case=str(d["case"][j]),
                             path=str(d["paths"][j]), sim=float(ms[j]),
                             isic_index=int(mi[j])))
        print(f"[screen] {name}: n={len(e)}  max_sim={ms.max():.4f}  "
              f"median={np.median(ms):.4f}  flagged={flagged.sum()} ({frac*100:.2f}%)"
              f"{'  DEMOTED' if frac > DEMOTE_FRAC else ''}")

    pd.DataFrame(summary).to_csv(OUT / "overlap_screen_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "overlap_screen_flagged.csv", index=False)
    print(f"\n[screen] wrote overlap_screen_summary.csv "
          f"({len(rows)} flagged images listed)")


if __name__ == "__main__":
    main()
