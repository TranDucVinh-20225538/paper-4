"""Train residual-bottleneck adapters with a domain-adversarial objective on frozen embeddings.

Why: Phase B's finding was that an adapter built to hide an ID-vs-OOD axis left some scorers'
AUROC flat. Phase C asks whether that holds across backbones and benchmarks, so it needs the same kind
of intervention built the same way on each backbone: a residual bottleneck that is *constructed* to
make ID and a reference OOD set indistinguishable while keeping the ID task intact. Because the backbone is
frozen and the embeddings are pre-extracted, training touches arrays only (seconds per adapter on a CPU).

Model, in the feature space z of the backbone (standardised by ID-train mean/std during training only):
    delta(z) = up(GELU(down(z))),   down: Linear(d, 16), up: Linear(16, d), up initialised to zero
    z_a      = z + delta(z)                                  (dose alpha = 1; the audit applies z + alpha delta)
    task:    CE of a linear probe on z_a against the ID class labels
    domain:  BCE of a small MLP discriminator on GRL(z_a), ID train (0) vs the reference OOD set (1)
The gradient reversal layer (Ganin & Lempitsky) makes the adapter maximise the discriminator's loss, with the
usual ramp lambda_p = 2 / (1 + exp(-10 p)) - 1 over training progress p. The reference OOD set is `ood_ref`
(CIFAR-100 *train* for CIFAR-10, OpenOOD val_openimage_o for ImageNet-200): disjoint from every evaluated OOD set.

The trained weights are folded back to raw feature units, so that
`operators.apply_adapter(Z, state, alpha, gelu_approximate=False)` reproduces z + alpha * delta(z) on raw
embeddings (checked numerically before saving). GELU is the exact (erf) form the module trains with.

Saved to PhaseC/assets/adapters/<backbone>/<id_set>/adapter_seed<seed>.pt as
    {"adapter": {down.weight, down.bias, up.weight, up.bias}, "w_construction": unit(mu_id - mu_ref),
     "w_learned": unit(E_ref[delta] - E_id[delta]) (direction the adapter moves the reference relative to ID),
     "mu_id", "mu_ref", "meta": {seed, hyper-parameters, diagnostics, input hashes}}.

    python3 PhaseC/analysis/train_adapters.py --backbone resnet50_imagenet --id-set cifar10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PHASEC = HERE.parent
EMB_ROOT = PHASEC / "assets" / "embeddings"
ADAPTER_ROOT = PHASEC / "assets" / "adapters"

#: 10 seeds, the same values Phase B used, listed here explicitly for Phase C (a different domain).
ADAPTER_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
HP = {"bottleneck": 16, "epochs": 30, "batch_half": 128, "lr": 1e-3, "weight_decay": 1e-4,
      "lambda_adv": 1.0, "disc_hidden": 128, "gelu": "exact"}


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Adapter(nn.Module):
    def __init__(self, d, r=16):
        super().__init__()
        self.down, self.up = nn.Linear(d, r), nn.Linear(r, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z):
        return self.up(F.gelu(self.down(z)))


def load_split(backbone, id_set, split, root=EMB_ROOT):
    p = Path(root) / backbone / id_set / f"{split}.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    z = np.load(p)
    return z["embeddings"], z["labels"]


def train_one(Ztr, ytr, Zref, seed, device="cpu", hp=HP):
    d, K = Ztr.shape[1], int(ytr.max()) + 1
    m, s = Ztr.mean(0), np.maximum(Ztr.std(0), 1e-6)
    X = torch.tensor((Ztr - m) / s, dtype=torch.float32, device=device)
    R = torch.tensor((Zref - m) / s, dtype=torch.float32, device=device)
    y = torch.tensor(ytr, dtype=torch.long, device=device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    torch.manual_seed(int(seed))
    ad, probe = Adapter(d, hp["bottleneck"]).to(device), nn.Linear(d, K).to(device)
    disc = nn.Sequential(nn.Linear(d, hp["disc_hidden"]), nn.ReLU(), nn.Linear(hp["disc_hidden"], 1)).to(device)
    opt = torch.optim.Adam(list(ad.parameters()) + list(probe.parameters()) + list(disc.parameters()),
                           lr=hp["lr"], weight_decay=hp["weight_decay"])
    B = hp["batch_half"]
    steps_per_epoch = max(1, len(X) // B)
    total, step = hp["epochs"] * steps_per_epoch, 0
    log = []
    for ep in range(hp["epochs"]):
        perm_i = torch.randperm(len(X), generator=g)
        for k in range(steps_per_epoch):
            bi = perm_i[k * B:(k + 1) * B].to(device)
            bj = torch.randint(0, len(R), (B,), generator=g).to(device)
            xa = X[bi] + ad(X[bi])
            ra = R[bj] + ad(R[bj])
            lam = hp["lambda_adv"] * (2.0 / (1.0 + np.exp(-10.0 * step / total)) - 1.0)
            dom_in = _GRL.apply(torch.cat([xa, ra]), lam)
            dom_y = torch.cat([torch.zeros(B), torch.ones(B)]).to(device)
            dlog = disc(dom_in).squeeze(1)
            l_task = F.cross_entropy(probe(xa), y[bi])
            l_dom = F.binary_cross_entropy_with_logits(dlog, dom_y)
            opt.zero_grad()
            (l_task + l_dom).backward()
            opt.step()
            step += 1
        if ep == hp["epochs"] - 1 or ep % 5 == 0:
            log.append({"epoch": ep, "task_ce": float(l_task), "domain_bce": float(l_dom), "grl_lambda": float(lam),
                        "domain_acc": float(((dlog > 0).float() == dom_y).float().mean())})
    with torch.no_grad():
        xa = X + ad(X)
        task_acc = float((probe(xa).argmax(1) == y).float().mean())
        dn = ad(X).norm(dim=1) / X.norm(dim=1).clamp_min(1e-12)
    return ad, m, s, {"task_acc_train": task_acc, "delta_over_z_median": float(dn.median()), "log": log}


def fold_to_raw(ad, m, s):
    """Fold the standardisation into the weights: apply_adapter(raw z) == z + s * delta((z - m) / s)."""
    W1, b1 = ad.down.weight.detach().double().numpy(), ad.down.bias.detach().double().numpy()
    W2, b2 = ad.up.weight.detach().double().numpy(), ad.up.bias.detach().double().numpy()
    m, s = m.astype(np.float64), s.astype(np.float64)
    return {"down.weight": torch.tensor(W1 / s[None, :], dtype=torch.float32),
            "down.bias": torch.tensor(b1 - W1 @ (m / s), dtype=torch.float32),
            "up.weight": torch.tensor(s[:, None] * W2, dtype=torch.float32),
            "up.bias": torch.tensor(s * b2, dtype=torch.float32)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--id-set", default="cifar10")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(ADAPTER_SEEDS))
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--emb-root", type=Path, default=EMB_ROOT)
    ap.add_argument("--out", type=Path, default=ADAPTER_ROOT)
    a = ap.parse_args(argv)
    from operators import apply_adapter

    hp = {**HP, "epochs": a.epochs}
    try:
        Ztr, ytr = load_split(a.backbone, a.id_set, "id_train", a.emb_root)
        Zref, _ = load_split(a.backbone, a.id_set, "ood_ref", a.emb_root)
    except FileNotFoundError as e:
        print(f"[adapters] missing embeddings ({e}); skipping {a.backbone}/{a.id_set}")
        return 0
    Zte, _ = load_split(a.backbone, a.id_set, "id_test", a.emb_root)
    mu_id, mu_ref = Ztr.astype(np.float64).mean(0), Zref.astype(np.float64).mean(0)
    w_con = (mu_id - mu_ref) / np.linalg.norm(mu_id - mu_ref)
    odir = a.out / a.backbone / a.id_set
    odir.mkdir(parents=True, exist_ok=True)
    in_sha = hashlib.sha256(Ztr.tobytes()[:1 << 24] + Zref.tobytes()[:1 << 24]).hexdigest()
    for seed in a.seeds:
        path = odir / f"adapter_seed{seed}.pt"
        if path.is_file() and not a.force:
            print(f"[adapters] {path.name}: exists, skipping")
            continue
        t0 = time.perf_counter()
        ad, m, s, diag = train_one(Ztr, ytr, Zref, seed, a.device, hp)
        state = fold_to_raw(ad, m, s)
        # check: folded raw-unit adapter == the module trained in standardised units
        chk = Zte[:512].astype(np.float64)
        ours, meta = apply_adapter(chk, state, 1.0, gelu_approximate=False)
        with torch.no_grad():
            ref = chk + s * ad(torch.tensor((chk - m) / s, dtype=torch.float32, device=a.device)).cpu().numpy()
        err = float(np.abs(ours - ref).max() / (np.abs(ref - chk).max() + 1e-12))
        if err > 1e-3:
            raise SystemExit(f"folded adapter disagrees with trained module (relative {err:.2e})")
        dz = lambda Z: apply_adapter(Z, state, 1.0, gelu_approximate=False)[0].astype(np.float64) - Z
        w_learned = dz(Zref).mean(0) - dz(Ztr).mean(0)
        nl = np.linalg.norm(w_learned)
        w_learned = w_learned / nl if nl > 0 else w_learned
        torch.save({"adapter": state, "w_construction": torch.tensor(w_con, dtype=torch.float32),
                    "w_learned": torch.tensor(w_learned, dtype=torch.float32),
                    "mu_id": torch.tensor(mu_id), "mu_ref": torch.tensor(mu_ref),
                    "meta": {"seed": int(seed), "backbone": a.backbone, "id_set": a.id_set, "hp": hp,
                             "n_train": len(Ztr), "n_ref": len(Zref), "fold_check_rel_err": err,
                             "diagnostics": diag, "inputs_sha256_prefix": in_sha,
                             "seconds": round(time.perf_counter() - t0, 1)}}, path)
        last = diag["log"][-1]
        print(f"[adapters] {a.backbone}/{a.id_set} seed {seed}: task_acc={diag['task_acc_train']:.4f} "
              f"domain_acc={last['domain_acc']:.3f} |delta|/|z|={diag['delta_over_z_median']:.3f} "
              f"fold_err={err:.1e} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
