"""Where the data this study reads lives, in one place instead of nineteen.

Nineteen scripts used to open with a literal path under one author's home
directory. That is invisible while it works and total while it does not: on any
other machine every one of them fails on its first line, for the same reason,
and the reason is never the one being debugged.

Each root resolves in three steps: an environment variable, then `data_roots.json`
beside this package, then nothing. The committed JSON records the layout the
study was run on, so the original machine keeps working untouched; anywhere else,
set the variables or edit the file.

Resolution does not check that a root exists, because the literals it replaces
did not either. A script fails where it always failed, at the point of use, with
the path it was actually given. To check a configuration before running anything:

    python3 PhaseB/analysis/data_roots.py

Two roots are gone from the machine this study ran on and are recorded for
provenance rather than for reading: `archive`, the working copy of the ISIC
crops, and the image directories under `csg`. The embedding artifacts in
`assets/reference_embeddings/` are what survives of them, hash-pinned. Stage 3.0's
audit and the duplicate registry read images and therefore cannot be re-run
without restoring those corpora from their distributors.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PHASEB = Path(__file__).resolve().parents[1]
REPO = PHASEB.parent
CONFIG = PHASEB / "data_roots.json"

#: root name -> environment variable that overrides it
ENV_VAR = {
    "archive": "PAPER4_ARCHIVE",
    "csg": "PAPER4_CSG_ROOT",
    "paper3": "PAPER4_PAPER3",
    "paper3_scripts": "PAPER4_PAPER3_SCRIPTS",
    "dst_skin_images": "PAPER4_DST_SKIN_IMAGES",
    "derm7pt": "PAPER4_DERM7PT",
    "ph2": "PAPER4_PH2",
    # Phase C (flat-response audit generalisation). Optional: not needed for Phase B.
    "openood": "PAPER4_OPENOOD",
    "phasec_data": "PAPER4_PHASEC_DATA",
    "phasec_checkpoints": "PAPER4_PHASEC_CKPT",
}

#: roots only Phase C reads. They fall back to a repo-relative default, and they
#: do not make `python3 PhaseB/analysis/data_roots.py` fail when absent, so the
#: Phase B check keeps its meaning.
OPTIONAL = {
    "openood": None,
    "phasec_data": "PhaseC/data",
    "phasec_checkpoints": "PhaseC/assets/checkpoints",
}

_WHAT = {
    "archive": "working copy of the ISIC crops (deleted; provenance only)",
    "csg": "Paper 1/2 repository, imported for its dataset and model code",
    "paper3": "Paper 3 repository, read for its checkpoint results matrix",
    "paper3_scripts": "Paper 3 scripts directory, imported for shared helpers",
    "dst_skin_images": "DST-Skin reader images, used by the forward-pass checks",
    "derm7pt": "Derm7pt release_v0, external evaluation domain",
    "ph2": "PH2 dataset, external evaluation domain",
    "openood": "OpenOOD benchmark data (ImageNet/CIFAR OOD splits), Phase C",
    "phasec_data": "Phase C dataset root (default PhaseC/data in this repository)",
    "phasec_checkpoints": "Phase C model checkpoints (default PhaseC/assets/checkpoints)",
}


def _configured() -> dict[str, str]:
    if not CONFIG.exists():
        return {}
    return json.loads(CONFIG.read_text()).get("roots", {})


def root(name: str) -> Path:
    """The configured location of `name`, unchecked."""
    if name not in ENV_VAR:
        raise KeyError(f"unknown data root {name!r}; known: {', '.join(sorted(ENV_VAR))}")
    from_env = os.environ.get(ENV_VAR[name])
    if from_env:
        return Path(from_env).expanduser()
    configured = _configured().get(name)
    if configured:
        return Path(configured).expanduser()
    default = OPTIONAL.get(name)
    if default:
        return REPO / default
    raise RuntimeError(
        f"data root {name!r} ({_WHAT[name]}) is not configured. "
        f"Set {ENV_VAR[name]}, or add it under \"roots\" in {CONFIG}."
    )


def require(name: str) -> Path:
    """The configured location of `name`, which must exist."""
    p = root(name)
    if not p.exists():
        raise FileNotFoundError(
            f"data root {name!r} ({_WHAT[name]}) is configured as {p}, which does not exist. "
            f"Set {ENV_VAR[name]} to a local copy."
        )
    return p


def resolve(recorded: str) -> Path:
    """Expand a recorded path written with a {root} placeholder.

    Paths that outlive the machine they were recorded on are written
    `{archive}/NV/ISIC_0055906.jpg` rather than as an absolute path, so a reader
    with their own copy of a corpus can find the file and a reader without one
    still learns which file a row came from. `{repo}` means this repository.
    """
    if not recorded.startswith("{"):
        return Path(recorded).expanduser()
    name, _, rest = recorded[1:].partition("}")
    base = REPO if name == "repo" else root(name)
    return base / rest.lstrip("/")


def main() -> int:
    configured = _configured()
    missing = 0
    required = [n for n in sorted(ENV_VAR) if n not in OPTIONAL]
    for name in sorted(ENV_VAR):
        optional = name in OPTIONAL
        if os.environ.get(ENV_VAR[name]):
            where, src = Path(os.environ[ENV_VAR[name]]).expanduser(), ENV_VAR[name]
        elif name in configured:
            where, src = Path(configured[name]).expanduser(), CONFIG.name
        elif optional and OPTIONAL[name]:
            where, src = REPO / OPTIONAL[name], "default"
        else:
            tag = "unset*" if optional else "unset"
            print(f"  {tag:<8} {name:<16} {_WHAT[name]}\n           set {ENV_VAR[name]}")
            missing += not optional
            continue
        mark = "ok" if where.exists() else ("MISSING*" if optional else "MISSING")
        print(f"  {mark:<8} {name:<16} {where}   [{src}]")
        missing += (not where.exists()) and not optional
    print(f"\n{len(required) - missing} of {len(required)} roots resolve."
          "  (* = Phase C only, not counted)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
