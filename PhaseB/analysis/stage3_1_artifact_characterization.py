"""
Stage 3.1 -- Artifact Characterization.

PI instruction, verbatim (2026-08-05): "Characterize the frozen Reference
Embedding Artifact only. Compute descriptive properties (shape, dtype, norm
distribution, variance statistics, basic integrity summaries) and document
them. Do not compute any quantities that test Claim C1-C5, including
Mahalanobis, probe accuracy, LID, spectral decay, Energy, ViM, KDE, AUROC,
or hypothesis-related statistics."

Everything here is a description of the artifact as it is, not a comparison
that bears on any claim. Per-domain (ID/OOD) breakdowns are reported purely
descriptively (count, mean, std, percentiles) -- no test statistic, no
p-value, no distance/separability score, no significance claim of any kind.
A reader could eyeball the numbers, but nothing here computes or asserts
whether domains differ in a way that matters for the hypothesis.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceEmbedding"
ISSUES_REGISTER = Path(__file__).resolve().parents[1] / "assets" / "ArtifactIssuesRegister" / "issues_register.json"
OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "ArtifactCharacterization"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe(x: np.ndarray) -> dict:
    return {
        "n": int(x.shape[0]),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p5": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }


def main() -> None:
    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
    metadata = pd.read_csv(ARTIFACT_DIR / "metadata.csv")

    with open(ISSUES_REGISTER) as f:
        issues = json.load(f)

    # --- Shape / dtype / memory ---
    basic = {
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "nbytes": int(embeddings.nbytes),
        "human_size": f"{embeddings.nbytes / (1024**2):.2f} MB",
    }

    # --- Norm distribution (L2 norm per row) ---
    norms = np.linalg.norm(embeddings, axis=1)
    norm_pooled = _describe(norms)
    is_isic = (metadata["domain"] == "isic").to_numpy()
    norm_by_domain = {
        "isic": _describe(norms[is_isic]),
        "pad_ufes": _describe(norms[~is_isic]),
    }

    # --- Variance statistics ---
    per_dim_variance = np.var(embeddings, axis=0)  # (1024,)
    variance_summary = {
        "total_variance_sum_of_per_dim": float(np.sum(per_dim_variance)),
        "per_dim_variance_mean": float(np.mean(per_dim_variance)),
        "per_dim_variance_std": float(np.std(per_dim_variance)),
        "per_dim_variance_min": float(np.min(per_dim_variance)),
        "per_dim_variance_max": float(np.max(per_dim_variance)),
        "per_dim_variance_median": float(np.median(per_dim_variance)),
    }

    # --- Per-class row counts (descriptive only -- not a class-separability claim) ---
    class_counts = metadata["label_name"].value_counts().sort_index().to_dict()
    domain_counts = metadata["domain"].value_counts().to_dict()

    # --- Basic integrity summary (BY REFERENCE to Stage 3.0 -- not recomputed here) ---
    integrity_summary = {
        "source": "Stage 3.0 Artifact Audit (not recomputed in this stage)",
        "no_nan": True,
        "no_inf": True,
        "row_count": int(embeddings.shape[0]),
        "known_issues_registered": len(issues["issues"]),
        "known_issues_note": (
            f"{issues['issues'][0]['description']} (Issue #1, {issues['issues'][0]['status']}). "
            "34/7365 rows (~0.46%) are pixel-identical duplicate pairs, included as-is per Stage "
            "3.0's 'no action taken' decision. Their contribution to the descriptive statistics "
            "below is a small, symmetric double-count (each pair counted twice), not expected to "
            "shift any pooled statistic meaningfully at this sample size -- noted for completeness, "
            "not asserted as negligible without this note."
        ),
    }

    # --- Norm histogram (descriptive figure only) ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(norms[is_isic], bins=40, alpha=0.6, label=f"ISIC (n={int(is_isic.sum())})", color="tab:blue")
    ax.hist(norms[~is_isic], bins=40, alpha=0.6, label=f"PAD-UFES (n={int((~is_isic).sum())})", color="tab:orange")
    ax.set_xlabel("Embedding L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("Stage 3.1 -- Reference Embedding Artifact: norm distribution\n"
                  "Descriptive only -- not a separability claim, no test statistic computed", fontsize=10)
    ax.legend()
    fig.tight_layout()
    hist_path = OUT_DIR / "norm_histogram.png"
    fig.savefig(hist_path, dpi=200)
    plt.close(fig)

    report = {
        "protocol_step": "Stage 3.1 -- Artifact Characterization",
        "date": "2026-08-05",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifact_dir": str(ARTIFACT_DIR),
        "basic": basic,
        "norm_distribution_pooled": norm_pooled,
        "norm_distribution_by_domain_DESCRIPTIVE_ONLY": norm_by_domain,
        "variance_statistics": variance_summary,
        "class_counts": class_counts,
        "domain_counts": domain_counts,
        "integrity_summary_by_reference": integrity_summary,
        "figures": {"norm_histogram": str(hist_path)},
        "not_computed_here": (
            "No Mahalanobis, no probe accuracy, no LID, no spectral-decay slope, no Energy, no "
            "ViM, no KDE, no AUROC, no hypothesis-testing statistic of any kind. Per-domain "
            "breakdowns above are descriptive counts/means/percentiles only -- not a "
            "separability, distance, or significance claim; no comparison test was run."
        ),
    }

    report_path = OUT_DIR / "characterization_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[stage3.1] Shape: {basic['shape']}, dtype: {basic['dtype']}, size: {basic['human_size']}")
    print(f"[stage3.1] Norm (pooled): mean={norm_pooled['mean']:.3f}, std={norm_pooled['std']:.3f}, "
          f"range=[{norm_pooled['min']:.3f}, {norm_pooled['max']:.3f}]")
    print(f"[stage3.1] Norm (ISIC): mean={norm_by_domain['isic']['mean']:.3f}")
    print(f"[stage3.1] Norm (PAD-UFES): mean={norm_by_domain['pad_ufes']['mean']:.3f}")
    print(f"[stage3.1] Per-dim variance: mean={variance_summary['per_dim_variance_mean']:.4f}, "
          f"total={variance_summary['total_variance_sum_of_per_dim']:.2f}")
    print(f"[stage3.1] Class counts: {class_counts}")
    print(f"[stage3.1] Report written: {report_path}")
    print(f"[stage3.1] Figure written: {hist_path}")

    # --- Freeze ---
    manifest_sha = _sha256_file(report_path)
    with open(OUT_DIR / "sha256.txt", "w") as f:
        f.write(f"{manifest_sha}  characterization_report.json\n")
        f.write(f"{_sha256_file(hist_path)}  norm_histogram.png\n")

    for f in OUT_DIR.iterdir():
        f.chmod(0o444)
    OUT_DIR.chmod(0o555)
    print(f"[stage3.1] Frozen (chmod read-only): {OUT_DIR}")

    print("\n[stage3.1] STOP: no claim-bearing quantity computed. Descriptive characterization only.")


if __name__ == "__main__":
    main()
