"""Compute Spearman rank preservation for every audit configuration.

Why: Check 1's registered invariant label requires high reach, which is
unreachable for an exactly rank-invariant scorer with wide margins. Rank
preservation is a statement about ordering, so test ordering:

    rho = spearman(s(z), s_tilde(T z))  over ID test + each OOD population

Writes PhaseC/outputs/rank_preservation.csv (one row per
backbone × id_set × ood × operator × dose × seed × scorer). Resume-safe;
backbones can run as parallel processes. Skips reach / bounds / Check 1 grid —
those already live in generalisation_audit.csv; this only adds rho.

    python3 PhaseC/analysis/compute_rank_preservation.py --backbone resnet18_cifar_openood --id-set cifar10
    python3 PhaseC/analysis/compute_rank_preservation.py --backbone resnet18_cifar_openood \\
        --scorers maha rmd cosine knn10 --operators ortho_scale general_affine
"""
from __future__ import annotations

import argparse
import fcntl
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reach  # noqa: E402
import run_audit as audit  # noqa: E402
import scorers  # noqa: E402
from train_adapters import ADAPTER_ROOT, EMB_ROOT  # noqa: E402

PHASEC = HERE.parent
OUT = PHASEC / "outputs"
RANK_CSV = "rank_preservation.csv"
COLS = ["backbone", "id_set", "ood_set", "operator", "dose", "scorer", "seed",
        "rank_spearman", "rank_one_minus_rho", "n_discordant_pairs",
        "n_pairs_untied_before", "n_eval", "config_seconds"]


def _append(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, header=False, index=False, float_format="%.10g")
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                f.write(",".join(df.columns) + "\n")
            f.write(buf.getvalue())
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def completed(path: Path, backbone, id_set, scorers_wanted):
    """Configs that already have every requested scorer for every ood_set seen so far.

    A config is done when at least one row exists for each requested scorer (any
    ood). Partial scorer runs are re-done for that config so a later --scorers
    expansion does not leave holes; callers that want additive scorers should
    filter the todo list themselves.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    ch = pd.read_csv(path, usecols=["backbone", "id_set", "operator", "dose", "seed", "scorer"])
    ch = ch[(ch.backbone == backbone) & (ch.id_set == id_set)]
    if len(ch) == 0:
        return set()
    want = set(scorers_wanted)
    done = set()
    for key, g in ch.groupby(["operator", "dose", "seed"]):
        if want.issubset(set(g.scorer.unique())):
            done.add((key[0], round(float(key[1]), 8), int(key[2])))
    return done


def rank_stats(b, a, n_pairs, pair_seed):
    b = np.asarray(b, dtype=np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()
    if np.ptp(b) == 0 and np.ptp(a) == 0:
        rho = 1.0
    elif np.ptp(b) == 0 or np.ptp(a) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(b, a).statistic)
    rng = np.random.default_rng(pair_seed)
    ii = rng.integers(0, len(b), n_pairs)
    jj = rng.integers(0, len(b), n_pairs)
    db = np.sign(b[ii] - b[jj])
    da = np.sign(a[ii] - a[jj])
    untied = db != 0
    return {
        "rank_spearman": rho,
        "rank_one_minus_rho": float(1.0 - rho) if np.isfinite(rho) else float("nan"),
        "n_discordant_pairs": int(np.sum(untied & (db != da))),
        "n_pairs_untied_before": int(untied.sum()),
        "n_eval": int(len(b)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--id-set", default="cifar10")
    ap.add_argument("--scorers", nargs="*", default=list(scorers.SCORER_NAMES))
    ap.add_argument("--operators", nargs="*", default=None)
    ap.add_argument("--n-pairs", type=int, default=reach.N_PAIRS)
    ap.add_argument("--emb-root", type=Path, default=EMB_ROOT)
    ap.add_argument("--adapter-root", type=Path, default=ADAPTER_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)

    names = tuple(a.scorers)
    for n in names:
        if n not in scorers.SCORER_NAMES:
            raise SystemExit(f"unknown scorer {n}")

    ids = {"backbone": a.backbone, "id_set": a.id_set}
    Ztr, ytr, Zte, oods = audit.load_embeddings(a.backbone, a.id_set, a.emb_root)
    if not oods:
        print(f"[rank] {a.backbone}/{a.id_set}: no OOD embeddings; nothing to do")
        return 0
    n_id = len(Zte)
    E = np.concatenate([Zte] + list(oods.values()))
    slices, o = {}, n_id
    for name, Z in oods.items():
        slices[name] = slice(o, o + len(Z))
        o += len(Z)
    print(f"[rank] {a.backbone}/{a.id_set}: train {Ztr.shape}, E {E.shape}, scorers={names}", flush=True)

    adir = a.adapter_root / a.backbone / a.id_set
    adapters = {int(p.stem.replace("adapter_seed", "")): p for p in sorted(adir.glob("adapter_seed*.pt"))} if adir.is_dir() else {}
    cfgs = audit.make_config_list(False, a.operators, adapter_available=bool(adapters))
    cfgs = [c for c in cfgs if c[0] != "adapter" or c[2] in adapters]
    # identity is the baseline; skip (rho trivial). dose-0 temper/zca/adapter are identity maps — still run
    # so the table covers them; they should give rho=1 for every scorer.
    path = a.out / RANK_CSV
    done = set() if a.redo else completed(path, a.backbone, a.id_set, names)
    todo = [c for c in cfgs if c[0] != "identity" and (c[0], round(c[1], 8), c[2]) not in done]
    print(f"[rank] {len(todo)} configurations to run (done={len(done)})", flush=True)

    t_base = time.perf_counter()
    base = scorers.score_all(Ztr, ytr, E, names=names)
    print(f"[rank] identity scores in {time.perf_counter() - t_base:.1f}s", flush=True)

    for i, (op, dose, seed) in enumerate(todo):
        t0 = time.perf_counter()
        state = torch.load(adapters[seed], map_location="cpu") if op == "adapter" else None
        Xtr, XE, meta = audit.transform(op, dose, seed, Ztr, E, state)
        after = scorers.score_all(Xtr, ytr, XE, names=names)
        rows = []
        for ood, sl in slices.items():
            idx = np.r_[0:n_id, sl.start:sl.stop]
            for sc in names:
                st = rank_stats(base["scores"][sc][idx], after["scores"][sc][idx], a.n_pairs, audit.PAIR_SEED)
                rows.append({
                    "backbone": a.backbone, "id_set": a.id_set, "ood_set": ood,
                    "operator": op, "dose": dose, "scorer": sc, "seed": seed,
                    **st, "config_seconds": float("nan"),
                })
        sec = time.perf_counter() - t0
        for r in rows:
            r["config_seconds"] = sec
        _append(path, pd.DataFrame(rows, columns=COLS))
        # Affine prediction check: print worst 1-rho for maha/rmd
        sub = [r for r in rows if r["scorer"] in ("maha", "rmd")]
        if sub and op in ("ortho_linear", "ortho_scale", "general_affine", "zca", "spectral_temper") and dose != 0:
            worst = max(r["rank_one_minus_rho"] for r in sub if np.isfinite(r["rank_one_minus_rho"]))
            print(f"[rank] {i + 1}/{len(todo)} {op} dose={dose} seed={seed}  {sec:.0f}s  "
                  f"worst 1-rho maha/rmd={worst:.3e}", flush=True)
        else:
            print(f"[rank] {i + 1}/{len(todo)} {op} dose={dose} seed={seed}  {sec:.0f}s", flush=True)
    print(f"[rank] done {a.backbone}/{a.id_set}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
