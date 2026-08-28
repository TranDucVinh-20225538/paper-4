"""
Stage 0 exploratory scatter plot -- composite geometry metric vs. Mahalanobis-AUROC.

Per Vinh (2026-08-05): added to LOOK at the restricted-range pattern the C1
Kendall's-tau gate already reported numerically, not to argue anything new.
Exploratory visualization only -- no inferential claim, no fitted line, no
p-value on the plot. The confirmatory numbers live in
outputs/stage0_c1_kendall_tau.csv; this figure does not add to or substitute
for them.

Reads outputs/stage0_geometry_auroc_merged.csv (written by
stage0_retrospective.py). Writes outputs/figure_stage0_composite_vs_auroc.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

RUNG_STYLE = {
    "runA_grl": {"color": "tab:blue", "marker": "o", "label": r"runA_grl ($\lambda_{orth}=0$)"},
    "runB_orth1": {"color": "tab:orange", "marker": "s", "label": r"runB_orth1 ($\lambda_{orth}=1$)"},
    "runB": {"color": "tab:red", "marker": "^", "label": r"runB ($\lambda_{orth}=5$)"},
}


def main() -> None:
    df = pd.read_csv(OUTPUT_DIR / "stage0_geometry_auroc_merged.csv")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for rung, style in RUNG_STYLE.items():
        sub = df[df["rung"] == rung]
        ax.scatter(
            sub["composite_lid_x_slope"], sub["auroc"],
            color=style["color"], marker=style["marker"], s=70,
            label=style["label"], zorder=3, edgecolor="black", linewidth=0.5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"s{int(row['seed'])}",
                (row["composite_lid_x_slope"], row["auroc"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=style["color"],
            )

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=1, label="chance (AUROC=0.5)")

    ax.set_xlabel(r"Composite geometry metric  $P = \mathrm{LID} \times |\mathrm{slope}|$")
    ax.set_ylabel("Mahalanobis AUROC (ID=ISIC vs. OOD=PAD-UFES-20)")
    ax.set_title(
        "Stage 0 (C1) -- composite geometry vs. accessibility, per checkpoint\n"
        "Exploratory visualization only -- see stage0_c1_kendall_tau.csv for the inferential test",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, zorder=0)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_stage0_composite_vs_auroc.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "figure_stage0_composite_vs_auroc.pdf")
    plt.close(fig)
    print(f"[stage0_figure] Written to {OUTPUT_DIR}/figure_stage0_composite_vs_auroc.{{png,pdf}}")


if __name__ == "__main__":
    main()
