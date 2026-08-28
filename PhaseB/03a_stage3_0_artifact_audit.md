# Stage 3.0 — Artifact Audit

Introduced 2026-08-05, per Vinh's PI decision immediately after Stage 2.1's completion. Not part of
`22_Experimental_Design.md`'s claim list — this is a data-integrity check on the frozen Reference
Embedding Artifact, deliberately placed before any claim-bearing computation touches it.

**PI framing (Vinh, verbatim intent):** "Stage 3 không nên bắt đầu bằng Mahalanobis... Stage 3 nên
là 3.0 Artifact Audit... Không phân tích. Chỉ audit."

## Scope

```
ReferenceEmbedding/ → random sample → visual inspection → spot check → (already frozen)
```

Goal: confirm the artifact is not corrupted, has no duplicates, no missing samples, and that nothing
new was generated in the process of auditing it. Not a hypothesis test — a data-integrity check.

### Checks

1. **File-integrity re-verification** — recompute SHA256 of the (now read-only)
   `embeddings.npy`/`labels.csv`/`metadata.csv` and confirm they match `manifest.json`'s recorded
   hashes exactly, i.e. nothing has changed since freeze.
2. **No duplicate rows** — duplicate `path` values (same image counted twice) and duplicate
   embedding vectors (bitwise-identical rows, which would indicate a batching/accumulation bug).
3. **No missing/corrupted samples** — re-check row count, NaN, Inf, zero-norm rows on the final,
   on-disk artifact (not just trusting Stage 2.1's in-memory checks — this re-reads the frozen
   files from scratch).
4. **Random-sample visual inspection** — randomly sample a handful of rows, load the corresponding
   *source images* (from `metadata.csv`'s `path` column), and visually confirm they are genuine
   dermatology images consistent with their recorded label/domain. Sent to Vinh for direct viewing,
   not just described.
5. **Reproducibility spot-check** — for the same sampled rows, re-run the source image through the
   Reference Instrument **independently** (fresh forward pass, not reading the stored embedding) and
   compare to the artifact's stored embedding for that row. Confirms the artifact reflects what the
   instrument actually produces, not a corrupted or mismatched write.

## Explicitly out of scope for 3.0

Mahalanobis, probe, LID, spectral-decay slope, energy, ViM, KDE, or any hypothesis-testing
statistic. No interpretation of embedding *content* — only whether the artifact is intact,
complete, and correctly traceable to its source images.

## Result (executed 2026-08-05)

| # | Check | Result |
|---|---|---|
| 1 | File-integrity re-verification | **PASS** — SHA256 of all 3 frozen files matches `manifest.json` exactly |
| 2 | No duplicate rows | **17 duplicate-embedding groups found (34 rows)** — see below, root-caused, not silently resolved |
| 3 | No missing/corrupted samples | **PASS** — 7,365 rows, no NaN, no Inf, no zero-norm, re-checked from the on-disk frozen files |
| 4 | Random-sample visual inspection | **PASS** — 12 rows sampled (seed=42), montage sent to Vinh: `outputs/stage3_0_sample_montage.png` |
| 5 | Reproducibility spot-check | **PASS** — same 12 images re-run through the Reference Instrument fresh; max abs diff ≤ 9.5e-6, cosine similarity ≥ 0.9999999 against stored embeddings, for all 12 |

**On check 2 — investigated, not just flagged.** 0 duplicate *paths* (every row has a distinct
source file), but 17 groups of bitwise-identical *embedding vectors* across different paths.
Independently verified (pixel array comparison, not file-hash — one pair had different file bytes
but identical pixel content, i.e. a re-encoding of the same image data) that **all 17 groups are
pixel-identical images** already present in the underlying public ISIC 2019 / PAD-UFES-20 datasets
— not an extraction pipeline defect. Wording note (corrected 2026-08-05, per Vinh): "pixel-identical
images" is what pixel comparison actually proves; this does **not** claim they are the same
photograph, lesion, or patient capture — that would be a real-world provenance claim pixel identity
alone does not establish, even though it's consistent with it. **13 of the 17 duplicate-image
groups carry conflicting diagnostic labels** for the same pixel-identical image in the source
metadata (e.g. the same PAD-UFES-20 image labeled `SCC` under one image ID and `BCC` under another).
This is a label-reliability property of the source data, not something this audit resolves,
corrects, or silently drops rows for — the frozen artifact is unmodified and faithfully reflects the
preregistered dataset, duplicates and label conflicts included. Full detail:
`outputs/stage3_0_artifact_audit_report.json`'s `check_2_no_duplicates.root_cause_diagnosis`, and
the dedicated `assets/DuplicateRegistry/` (Stage 3.0b).

**Overall: 4/5 checks clean PASS; check 2 surfaced and fully root-caused a genuine source-data
property (not a corruption/pipeline defect).** No modification made to the frozen artifact in
response.

## Artifacts

- `outputs/stage3_0_artifact_audit_report.json` — all check results, including the duplicate
  root-cause diagnosis.
- `outputs/stage3_0_duplicate_diagnosis.json` — the duplicate-group investigation on its own.
- `outputs/stage3_0_sample_montage.png` — the visually-inspected sample, sent to Vinh.
- Run log: `logs/stage3_0_run_2026-08-05.log`.

## Not yet authorized

Stage 3.1 (Artifact Characterization — rows/dim/memory/dtype/norm distribution, still structural,
not a claim) and Stage 3.2+ (Claim C1 onward) each require their own PI decision. In particular, a
PI decision on the duplicate/label-conflict finding above (keep as-is, flag for a future
sensitivity check, or something else) is open — not resolved by this audit.

---

# Stage 3.0b — Duplicate Registry + Artifact Issues Register (2026-08-05)

PI decision, verbatim intent: "Artifact → Audit" and "Artifact → Cleaning" are two different
things — record findings, don't act on them. Two new frozen artifacts, evidence only, no dataset
modification:

- **`assets/DuplicateRegistry/`** — the raw evidence for the duplicate finding specifically:
  `duplicate_groups.csv` (one row per group member: path, domain, label, embedding SHA256,
  pixel-identity confirmation, label-conflict flag), `montage.png` (all 17 pairs, side by side,
  row index + label overlaid), `notes.md` (methodology, findings, explicit "NO ACTION TAKEN").
- **`assets/ArtifactIssuesRegister/`** — the higher-level register every future claim should
  reference: `issues_register.json` / `.md`, one entry so far (Issue #1: duplicate images with
  label conflict, discovered at Stage 3.0, status "logged, no action taken," resolution "open").
  Policy: new issues found by later audits are **appended** as new entries, never edited into or
  removed from existing ones.

Both directories frozen the same way as `ReferenceEmbedding/`: files `chmod 444`, directory
`chmod 555`, verified non-writable.

**This is an observation for the eventual paper's limitations/supplementary section, not a
contribution — e.g.:** "During artifact auditing, 17 groups of pixel-identical images were
identified in the preregistered datasets, including 13 groups with conflicting labels." Framed as a
property of the inherited source data, not a finding this project claims credit for.

## Artifacts (Stage 3.0b)

- `assets/DuplicateRegistry/{duplicate_groups.csv, montage.png, notes.md}`
- `assets/ArtifactIssuesRegister/{issues_register.json, issues_register.md}`
- Script: `analysis/stage3_0b_duplicate_registry_and_issues.py`. Log: `logs/stage3_0b_run_2026-08-05.log`.
