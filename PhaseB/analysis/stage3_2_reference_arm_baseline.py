"""
Stage 3.2 -- Reference Arm Baseline Evaluation (PARTIAL execution).

Executes what's fully resolved per
ReferenceArmBaseline/reference_arm_baseline_execution_plan.md:
  - Group A, Tier 1->2: domain-leakage probe (5 seeds + shuffle-label sanity
    check), methodology grounded in CSG-SKin/scripts/check_leakage.py.
  - Group A, Tier 1->2: geometry metrics -- condition number (reused from
    paper-3's geometry_diagnostics.py), LID + within-class spectral-decay
    slope (reused from PhaseB/analysis/lid_spectral_decay.py) -- computed on
    the COMBINED ID+OOD population, per 22_Experimental_Design.md SS4
    measurement 3's explicit instruction ("computed on the same held-out
    split as the scorers above").
  - ID task performance (Tier 1->2): SAME probe methodology as the
    domain-leakage probe, applied to the 8-class diagnostic label instead of
    domain -- a natural extension of an already-resolved method, not a new
    invention.
  - Tier 0: Gate 0 verdict, from the two measurements above.
  - Group A auxiliary: probe-weight diffuseness (participation ratio of the
    domain probe's fitted coefficients, seed=42).

Explicitly NOT executed here (flagged, not silently worked around):
  - Per-direction variance of "the not-yet-intervened candidate direction"
    (part of measurement 3) -- no resolved operationalization exists yet for
    what that direction IS on the Reference arm alone, before any
    intervention has been designed. Needs a PI decision.
  - Group B (Mahalanobis, kNN, energy, ViM, KDE, cosine-to-centroid,
    relative-Mahalanobis): Paper 3's own implementation of these scorers
    requires a FITTED reference population (ISIC-TRAIN embeddings, ~16k
    images) distinct from the held-out ID/OOD population used to evaluate
    them. ReferenceEmbedding/ (Stage 2.1) contains only the held-out
    ID(ISIC-test)+OOD(PAD-UFES) split -- no ISIC-train embeddings were
    extracted, because Stage 2.1's authorized scope was explicitly "the
    complete preregistered ID and OOD datasets" (evaluation split), not a
    training split. This is a genuine data-availability gap, not a
    methodology choice -- needs a PI decision (e.g. a Stage 2.2 extraction)
    before Group B can be computed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSG_ROOT = Path("/Users/cubo/Research/CSG-SKin")
PAPER3_SCRIPTS = Path("/Users/cubo/Research/paper-3/scripts")
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "assets" / "reference_embeddings" / "ReferenceEmbedding"
ISSUES_REGISTER = Path(__file__).resolve().parents[1] / "assets" / "ArtifactIssuesRegister" / "issues_register.json"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "ReferenceArmBaseline" / "BaselineReport"

PROBE_SEEDS = [42, 52, 62, 72, 82]
SHUFFLE_SEED = 123
TEST_SIZE = 0.3
LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]  # CSG-Skin src/datasets/constants.py
PAPER1_CNN_DOMAIN_PROBE_ACC = 0.979  # context reference, not a hard pass/fail line -- see Gate 0 below

sys.path.insert(0, str(CSG_ROOT))
sys.path.insert(0, str(PAPER3_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for lid_spectral_decay


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def linear_probe(features: np.ndarray, targets: np.ndarray, seed: int, test_size: float = TEST_SIZE):
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=test_size, random_state=seed, stratify=targets
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    proba = clf.predict_proba(x_test)
    return {
        "seed": seed,
        "accuracy": float(accuracy_score(y_test, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "n_test": int(len(y_test)),
    }, proba, y_test, clf


def ece_from_proba(proba: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(labels), 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / n) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def participation_ratio(w: np.ndarray) -> float:
    """Effective number of dimensions the probe's weight vector concentrates on. 1=one dim, d=uniform."""
    w2 = w.astype(np.float64) ** 2
    return float((w2.sum() ** 2) / (w2 ** 2).sum())


def main() -> None:
    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
    metadata = pd.read_csv(ARTIFACT_DIR / "metadata.csv")
    with open(ISSUES_REGISTER) as f:
        issues = json.load(f)

    domain_binary = (metadata["domain"] == "pad_ufes").to_numpy().astype(np.int64)  # 0=isic, 1=pad_ufes
    label_idx = metadata["label_idx"].to_numpy().astype(np.int64)

    print(f"[stage3.2] Loaded ReferenceEmbedding/: {embeddings.shape}, "
          f"n_isic={(domain_binary == 0).sum()}, n_pad={(domain_binary == 1).sum()}")
    print(f"[stage3.2] Known issues consulted: Issue #1 ({issues['issues'][0]['status']}) -- "
          f"34 duplicate rows, small symmetric contribution, not expected to materially affect "
          f"probe/geometry baseline measurements at n=7365.")

    # =================== Group A, Measurement 1: Domain-leakage probe ===================
    print("\n[stage3.2] === Domain-leakage probe (5 seeds) ===")
    probe_raw = []
    aurocs = []
    first_seed_proba = first_seed_ytest = first_seed_clf = None
    for seed in PROBE_SEEDS:
        result, proba, y_test, clf = linear_probe(embeddings, domain_binary, seed)
        auroc = float(roc_auc_score(y_test, proba[:, 1]))
        result["auroc"] = auroc
        probe_raw.append(result)
        aurocs.append(auroc)
        print(f"  seed={seed}: acc={result['accuracy']:.4f} bal_acc={result['balanced_accuracy']:.4f} "
              f"auroc={auroc:.4f}")
        if seed == PROBE_SEEDS[0]:
            first_seed_proba, first_seed_ytest, first_seed_clf = proba, y_test, clf

    probe_accs = [r["accuracy"] for r in probe_raw]
    probe_summary = {
        "seeds": PROBE_SEEDS,
        "accuracy_mean": float(np.mean(probe_accs)),
        "accuracy_std": float(np.std(probe_accs)),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "majority_class_baseline": float(max((domain_binary == 0).mean(), (domain_binary == 1).mean())),
        "paper1_cnn_reference_accuracy": PAPER1_CNN_DOMAIN_PROBE_ACC,
    }
    print(f"  mean acc={probe_summary['accuracy_mean']:.4f}+/-{probe_summary['accuracy_std']:.4f}  "
          f"mean auroc={probe_summary['auroc_mean']:.4f}  majority_baseline={probe_summary['majority_class_baseline']:.4f}")

    # Shuffle-label sanity check
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled_domain = domain_binary.copy()
    rng.shuffle(shuffled_domain)
    shuffle_result, _, _, _ = linear_probe(embeddings, shuffled_domain, seed=PROBE_SEEDS[0])
    probe_summary["shuffle_label_sanity_check"] = {
        "shuffle_seed": SHUFFLE_SEED, "probe_seed": PROBE_SEEDS[0],
        "accuracy": shuffle_result["accuracy"],
    }
    print(f"  shuffle-label sanity check (seed={SHUFFLE_SEED}): acc={shuffle_result['accuracy']:.4f} "
          f"(should be near majority baseline)")

    # Probe-weight diffuseness (auxiliary, measurement 5) -- from the first-seed fit
    logreg = first_seed_clf.named_steps["logisticregression"]
    diffuseness = participation_ratio(logreg.coef_[0])
    probe_summary["weight_diffuseness_participation_ratio"] = diffuseness
    probe_summary["weight_diffuseness_note"] = f"seed={PROBE_SEEDS[0]} fit only; range [1, {embeddings.shape[1]}]"
    print(f"  probe-weight diffuseness (participation ratio, seed={PROBE_SEEDS[0]}): {diffuseness:.2f} "
          f"of {embeddings.shape[1]} dims")

    # =================== Group A, Measurement 3: Geometry metrics (excl. per-direction variance) ===================
    print("\n[stage3.2] === Geometry metrics (combined ID+OOD population, per SS4's instruction) ===")
    from geometry_diagnostics import condition_number  # paper-3, reused not reimplemented
    from src.utils.ood_metrics import compute_mahalanobis_params_from_arrays  # CSG-Skin, reused

    _class_means, precision = compute_mahalanobis_params_from_arrays(embeddings, label_idx, num_classes=8, reg_eps=1e-5)
    kappa = condition_number(precision)

    from lid_spectral_decay import compute_lid_spectral_diagnostics  # PhaseB Stage 0's own module

    lid_diag = compute_lid_spectral_diagnostics(embeddings, label_idx, num_classes=8, k=50)

    geometry_metrics = {
        "population": "combined ID(ISIC-test)+OOD(PAD-UFES), n=7365, 8-class label_idx grouping",
        "condition_number": kappa,
        "lid_mean": lid_diag.lid_mean,
        "lid_k": lid_diag.lid_k,
        "spectral_decay_slope": lid_diag.spectral_slope,
        "composite_lid_x_slope": lid_diag.composite,
        "per_direction_variance": None,
        "per_direction_variance_note": (
            "NOT COMPUTED. 'Per-direction variance of the intervened direction' "
            "(22_Experimental_Design.md SS4 measurement 3, Mechanism 2) has no resolved "
            "operationalization on the Reference arm alone -- there is no 'intervened direction' "
            "yet, since no intervention has been designed or run (Stage 4 doesn't exist). "
            "Flagged for a PI decision, not invented here."
        ),
    }
    print(f"  condition_number={kappa:.4f}  LID={lid_diag.lid_mean:.4f}  "
          f"spectral_decay_slope={lid_diag.spectral_slope:.6f}  composite={lid_diag.composite:.4f}")
    print("  per_direction_variance: NOT COMPUTED (flagged, see geometry_metrics.json note)")

    # =================== Measurement 4: ID task performance ===================
    print("\n[stage3.2] === ID task performance (8-class diagnostic probe, 5 seeds) ===")
    id_task_raw = []
    id_bal_accs = []
    id_eces = []
    for seed in PROBE_SEEDS:
        result, proba, y_test, _clf = linear_probe(embeddings, label_idx, seed)
        ece = ece_from_proba(proba, y_test)
        result["ece"] = ece
        id_task_raw.append(result)
        id_bal_accs.append(result["balanced_accuracy"])
        id_eces.append(ece)
        print(f"  seed={seed}: acc={result['accuracy']:.4f} bal_acc={result['balanced_accuracy']:.4f} ece={ece:.4f}")

    class_counts = pd.Series(label_idx).value_counts()
    majority_class_frac = float(class_counts.max() / len(label_idx))
    id_task_summary = {
        "seeds": PROBE_SEEDS,
        "balanced_accuracy_mean": float(np.mean(id_bal_accs)),
        "balanced_accuracy_std": float(np.std(id_bal_accs)),
        "ece_mean": float(np.mean(id_eces)),
        "ece_std": float(np.std(id_eces)),
        "majority_class_baseline_accuracy": majority_class_frac,
        "chance_balanced_accuracy": 1.0 / 8,
    }
    print(f"  mean bal_acc={id_task_summary['balanced_accuracy_mean']:.4f}+/-{id_task_summary['balanced_accuracy_std']:.4f}  "
          f"mean ece={id_task_summary['ece_mean']:.4f}  chance={id_task_summary['chance_balanced_accuracy']:.4f}")

    # =================== Tier 0: Gate 0 verdict ===================
    print("\n[stage3.2] === Gate 0 (Implementation Integrity Gate) ===")
    domain_leakage_detected = bool(
        probe_summary["accuracy_mean"] > probe_summary["majority_class_baseline"] + 0.05
        and probe_summary["accuracy_mean"] > shuffle_result["accuracy"] + 0.05
    )
    id_task_above_trivial = bool(
        id_task_summary["balanced_accuracy_mean"] > id_task_summary["chance_balanced_accuracy"] + 0.10
    )
    gate0_pass = domain_leakage_detected and id_task_above_trivial

    gate0_verdict = {
        "protocol_step": "Stage 3.2 -- Gate 0 (Tier 0, Implementation Integrity Gate)",
        "criterion_1_domain_leakage_detected": {
            "pass": domain_leakage_detected,
            "probe_accuracy_mean": probe_summary["accuracy_mean"],
            "majority_class_baseline": probe_summary["majority_class_baseline"],
            "shuffle_control_accuracy": shuffle_result["accuracy"],
            "paper1_cnn_reference_accuracy": PAPER1_CNN_DOMAIN_PROBE_ACC,
            "rule": "probe_accuracy_mean > majority_baseline + 0.05 AND probe_accuracy_mean > shuffle_control + 0.05 "
                    "(operationalizes 'domain leakage present, not spurious' -- Risk Register item #3's concern; "
                    "Paper 1's 0.979 reported as context, not a hard pass/fail threshold, since 'consistent with' "
                    "was not specified as an exact-match criterion in 22_Experimental_Design.md SS3.5)",
        },
        "criterion_2_id_task_above_trivial": {
            "pass": id_task_above_trivial,
            "balanced_accuracy_mean": id_task_summary["balanced_accuracy_mean"],
            "chance_balanced_accuracy": id_task_summary["chance_balanced_accuracy"],
            "rule": "balanced_accuracy_mean > chance + 0.10",
        },
        "gate_result": "PASS" if gate0_pass else "FAIL",
    }
    print(f"  criterion 1 (domain leakage detected): {domain_leakage_detected}")
    print(f"  criterion 2 (ID task above trivial): {id_task_above_trivial}")
    print(f"  === GATE 0: {gate0_verdict['gate_result']} ===")

    # =================== Freeze what's computed ===================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with open(OUTPUT_DIR / "probe_results_raw.json", "w") as f:
        json.dump({"tier": 1, "group": "A", "measurement": "domain_leakage_probe", "per_seed": probe_raw}, f, indent=2)
    with open(OUTPUT_DIR / "probe_results_summary.json", "w") as f:
        json.dump({"tier": 2, "group": "A", "measurement": "domain_leakage_probe", **probe_summary}, f, indent=2)
    with open(OUTPUT_DIR / "geometry_metrics.json", "w") as f:
        json.dump({"tier": "1+2", "group": "A", **geometry_metrics}, f, indent=2)
    with open(OUTPUT_DIR / "id_task_performance.json", "w") as f:
        json.dump({"tier": "1+2", "group": None, "per_seed": id_task_raw, "summary": id_task_summary}, f, indent=2)
    with open(OUTPUT_DIR / "gate0_verdict.json", "w") as f:
        json.dump({"tier": 0, **gate0_verdict, "timestamp": timestamp}, f, indent=2)

    not_computed = {
        "per_direction_variance": "No resolved operationalization for the Reference arm alone -- flagged, PI decision needed.",
        "group_b_reliability_scorers": (
            "Mahalanobis/kNN/energy/ViM/KDE/cosine-to-centroid/relative-Mahalanobis NOT computed -- "
            "Paper 3's implementation needs a fitted ISIC-TRAIN reference population not present in "
            "ReferenceEmbedding/ (Stage 2.1 extracted only the held-out ID+OOD evaluation split). "
            "Genuine data-availability gap, PI decision needed (e.g. a Stage 2.2 extraction)."
        ),
        "tier_3_claims_tested": [],
    }
    with open(OUTPUT_DIR / "not_yet_computed.json", "w") as f:
        json.dump(not_computed, f, indent=2)

    manifest = {
        "artifact": "Reference Arm Baseline Report (PARTIAL)",
        "protocol_step": "Stage 3.2",
        "status": "PARTIAL -- Group B and per-direction variance not computed, see not_yet_computed.json",
        "created_at": timestamp,
        "derived_from": {
            "reference_embedding_artifact": str(ARTIFACT_DIR),
            "issues_register_consulted": "Issue #1 -- not expected to materially affect these baseline "
                                          "measurements at n=7365 (34/7365 rows, ~0.46%)",
        },
        "files": {},
        "tier_3_claims_tested": [],
    }
    for fname in ["probe_results_raw.json", "probe_results_summary.json", "geometry_metrics.json",
                  "id_task_performance.json", "gate0_verdict.json", "not_yet_computed.json"]:
        manifest["files"][fname] = {"sha256": _sha256_file(OUTPUT_DIR / fname)}
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[stage3.2] Outputs written to {OUTPUT_DIR}/ (NOT yet frozen -- pending PI review of the two open items)")
    print("[stage3.2] STOP: Group B and per-direction variance not computed. No Tier 3 / claim evaluated.")


if __name__ == "__main__":
    main()
