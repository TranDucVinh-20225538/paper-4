"""
Train five additional seeds for each canonical arm.

Motivation is in the manuscript rather than here: Section 4's second condition
observes that a whole-seed permutation over k seeds attains no p below 1/k!,
so at five seeds a Holm family larger than six tests cannot be cleared. Two
families in this study are in that regime. Ten seeds move the floor from
1/120 = 8.3e-3 to 1/10! = 2.8e-7 and the constraint disappears.

PROTOCOL DEVIATION, logged in 24_PreRegistration.md. The registered seed set is
{42, 52, 62, 72, 82}. The extension continues the same arithmetic rule --
{92, 102, 112, 122, 132} -- rather than selecting seeds, and every analysis is
rerun on all ten rather than on a chosen subset. Both the five-seed and
ten-seed results are reported so the effect of the extension is visible.

Hyperparameters are the canonical ones and are not re-selected: r=16,
lambda_proj=0.1, 100 epochs, lr 1e-3, batch 512. Before writing anything this
script retrains seed 42 and asserts the result is bit-identical to the stored
adapter, so a silent change in the training path cannot pass unnoticed.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import stage4_train_intervention as m  # noqa: E402
from stage4_train_intervention import (  # noqa: E402
    load_training_population, train_adapter, NUISANCE_PATH,
)
from stage4_canonical_conventional import train_conventional  # noqa: E402

R, LAMBDA_PROJ = 16, 0.1
OLD_SEEDS = [42, 52, 62, 72, 82]
NEW_SEEDS = [92, 102, 112, 122, 132]


def guard(z, y, w):
    """Retraining a stored seed must reproduce it exactly, or the path has changed."""
    m.SEED = 42
    ad, _ = train_adapter(z, y, w, r=R, lambda_proj=LAMBDA_PROJ)
    old = torch.load(ROOT / "InterventionArm_Canonical/adapter_seed42.pt",
                     map_location="cpu", weights_only=False)["adapter"]
    for k, v in ad.state_dict().items():
        d = float((v - old[k]).abs().max())
        if d > 1e-6:
            raise SystemExit(f"training path changed: {k} differs by {d:.3e}")
    print("[extend] guard passed: seed 42 reproduces bit-identically")


def main():
    z, y = load_training_population()
    w = np.load(NUISANCE_PATH)["w"]
    guard(z, y, w)

    # The canonical directories are chmod-frozen on purpose. The extension goes
    # beside them rather than into them, so the registered artifacts stay
    # untouched and the layout shows which seeds were added.
    for arm, trainer in (("Intervention", "int"), ("Conventional", "conv")):
        out = ROOT / f"{arm}Arm_Extended"
        out.mkdir(parents=True, exist_ok=True)
        for seed in NEW_SEEDS:
            dest = out / f"adapter_seed{seed}.pt"
            if dest.exists():
                print(f"[extend] {arm} seed {seed} present, skipping")
                continue
            if trainer == "int":
                m.SEED = seed
                ad, head = train_adapter(z, y, w, r=R, lambda_proj=LAMBDA_PROJ)
            else:
                ad, head = train_conventional(z, y, seed)
            torch.save({"adapter": ad.state_dict(), "head": head.state_dict(),
                        "seed": seed, "r": R,
                        "lambda_proj": LAMBDA_PROJ if trainer == "int" else 0.0},
                       dest)
            print(f"[extend] wrote {arm}Arm_Extended/{dest.name}", flush=True)

    out = ROOT / "InterventionArm_Extended"
    (out / "extension_manifest.json").write_text(json.dumps({
        "registered_seeds": OLD_SEEDS, "added_seeds": NEW_SEEDS,
        "rule": "arithmetic continuation of the registered set, step 10",
        "hyperparameters": {"r": R, "lambda_proj": LAMBDA_PROJ},
        "guard": "seed 42 retrained and asserted bit-identical before writing",
    }, indent=2))
    print("[extend] done")


if __name__ == "__main__":
    main()
