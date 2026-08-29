"""
Phase 3 -- extract PanDerm embeddings for the locked external domains.

Runs the same instrument, the same checkpoint and the same preprocessing that
produced the reference artifact, on Derm7pt (dermoscopic and clinical arms) and
PH2.  The transform is taken from the PanDerm repo's own get_encoder rather than
reimplemented, so fidelity does not depend on my reading of the pinned spec --
the spec in assets/reference_preprocessing.json is the audit record to check it
against, and this script asserts against it.

Label mapping and exclusions are read from the locked protocol, not decided
here.  See PhaseB/phase3/LOCKED_PROTOCOL.md section 4.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PANDERM = ROOT / "external" / "PanDerm" / "classification"
CKPT = ROOT / "assets" / "checkpoints" / "panderm_vitl16_checkpoint-499.pth"
PREPROC = ROOT / "assets" / "reference_preprocessing.json"
OUT = ROOT / "outputs" / "phase3"
DERM7PT = Path("/Users/cubo/Downloads/release_v0")
PH2 = Path("/Users/cubo/Downloads/PH2Dataset")

LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
L2I = {l: i for i, l in enumerate(LABELS)}

# section 4 of the locked protocol, verbatim
DERM7PT_MAP = {
    "melanoma": "MEL", "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL", "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "clark nevus": "NV", "reed or spitz nevus": "NV", "dermal nevus": "NV",
    "blue nevus": "NV", "congenital nevus": "NV", "combined nevus": "NV",
    "recurrent nevus": "NV",
    "basal cell carcinoma": "BCC",
    "seborrheic keratosis": "BKL", "lentigo": "BKL",
    "dermatofibroma": "DF",
    "vascular lesion": "VASC",
    "melanoma metastasis": None, "melanosis": None, "miscellaneous": None,
}
PH2_MAP = {0: "NV", 1: "NV", 2: "MEL"}


def build_derm7pt(arm: str) -> pd.DataFrame:
    m = pd.read_csv(DERM7PT / "meta" / "meta.csv")
    unknown = set(m.diagnosis) - set(DERM7PT_MAP)
    if unknown:
        raise SystemExit(f"diagnosis not in the locked mapping: {sorted(unknown)}")
    m["cls"] = m.diagnosis.map(DERM7PT_MAP)
    keep = m[m.cls.notna()].copy()
    keep["path"] = [str(DERM7PT / "images" / p) for p in keep[arm]]
    keep["label_idx"] = keep.cls.map(L2I)
    return keep[["case_num", "path", "cls", "label_idx", "diagnosis"]]


def build_ph2() -> pd.DataFrame:
    rows = []
    for ln in open(PH2 / "PH2_dataset.txt", encoding="latin-1"):
        if not ln.strip().startswith("||"):
            continue
        f = [c.strip() for c in ln.strip().strip("|").split("||")]
        if len(f) >= 3 and f[0] != "Name":
            rows.append((f[0], f[2]))
    out = []
    for name, dx in rows:
        p = PH2 / "PH2 Dataset images" / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp"
        if not p.is_file():
            raise SystemExit(f"missing PH2 image {p}")
        cls = PH2_MAP[int(dx)]
        out.append(dict(case_num=name, path=str(p), cls=cls,
                        label_idx=L2I[cls], diagnosis=f"ph2_class_{dx}"))
    return pd.DataFrame(out)


class Images(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def check_transform(tf):
    """Assert the live transform still matches the pinned audit record."""
    spec = json.load(open(PREPROC))["pipeline"]
    got = [t.__class__.__name__ for t in tf.transforms]
    want = [s["op"] for s in spec]
    if got != want:
        raise SystemExit(f"preprocessing drift: live {got} vs pinned {want}")
    norm = tf.transforms[-1]
    pin = spec[-1]
    if not (np.allclose(norm.mean, pin["mean"]) and np.allclose(norm.std, pin["std"])):
        raise SystemExit(f"normalisation drift: live {norm.mean}/{norm.std} vs pinned")
    print(f"[phase3] preprocessing matches the pinned record: {got}")


@torch.no_grad()
def embed(model, df, name, dev, bs=32):
    tf = model._tf
    ds = Images(df.path, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    chunks, t0, seen = [], time.perf_counter(), 0
    for x in dl:
        chunks.append(model.forward_features(x.to(dev), is_train=False).float().cpu())
        seen += x.shape[0]
        if seen % (bs * 10) == 0 or seen == len(ds):
            r = seen / (time.perf_counter() - t0)
            print(f"[phase3] {name}: {seen}/{len(ds)}  {r:.1f} img/s", flush=True)
    return torch.cat(chunks).numpy().astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PANDERM))
    from models import get_encoder

    class A:
        pretrained_checkpoint = str(CKPT)

    model, tf = get_encoder(A(), "PanDerm_Large_LP")
    check_transform(tf)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(dev).eval()
    model._tf = tf

    sets = {
        "derm7pt_derm": build_derm7pt("derm"),
        "derm7pt_clinic": build_derm7pt("clinic"),
        "ph2": build_ph2(),
    }
    for name, df in sets.items():
        if (OUT / f"external_{name}.npz").is_file():
            print(f"[phase3] {name}: already extracted, skipping")
            continue
        print(f"\n[phase3] {name}: {len(df)} images")
        print("   " + df.cls.value_counts().to_string().replace("\n", "\n   "))
        e = embed(model, df, name, dev)
        if e.shape != (len(df), 1024):
            raise SystemExit(f"bad shape {e.shape}")
        np.savez_compressed(OUT / f"external_{name}.npz", embeddings=e,
                            labels=df.label_idx.to_numpy(),
                            paths=df.path.to_numpy(),
                            case=df.case_num.astype(str).to_numpy(),
                            diagnosis=df.diagnosis.to_numpy())
        print(f"[phase3] wrote external_{name}.npz  {e.shape}")


if __name__ == "__main__":
    main()
