# Phase 3 external embeddings

`external_derm7pt_derm.npz`, `external_derm7pt_clinic.npz` and
`external_ph2.npz` are not in the repository. They are extracted output, not
source: `PhaseB/analysis/phase3_extract_external.py` regenerates them from the
PanDerm checkpoint and the three external corpora, asserting the live transform
against the pinned spec in `assets/reference_preprocessing.json` before it
writes anything. `.gitignore` excludes `*.npz` for the same reason it excludes
the checkpoint.

`sha256.txt` in this directory is the record to check a copy against, whether
regenerated or downloaded.

## Regenerating them

The script needs three things the repository does not carry: the PanDerm
checkpoint, the PanDerm classification code (pinned in
`PhaseB/external/PanDerm.pin`), and the Derm7pt and PH2 corpora from their
distributors. It also currently reads those corpora from absolute paths under
one author's home directory, so on any other machine the paths at the top of
the script have to be pointed at local copies first. Making them configurable
is open work, tracked in the repository's issue list rather than here.
