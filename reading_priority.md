# Reading Priority — Paper 4

## Tier 0 — BLOCKING, unresolved (verification pass, 2026-08-04 session 2)
1. **Gallo et al. (CVPR 2026W) — STILL NOT READ.** All prior claims about this paper are
   Level B/C (search-engine summaries) only. Attempted this session: direct fetch of the
   `openaccess.thecvf.com` HTML page (empty), direct fetch of the corresponding PDF URL (empty),
   Claude in Chrome (extension not connected). Next attempts: retry Claude in Chrome once
   reconnected; try a Semantic Scholar or ResearchGate mirror; try emailing the authors
   (Ignazio Gallo, Leonardo Sgroi, Mattia Gatti, Silvia Corchs) directly; or check institutional
   library access to the CVPR 2026 Workshops proceedings. **This is the single highest-priority
   open item in the whole project** — the RQ lock depends on it.

## Tier 1 — completed novelty reads (Level A, primary source verified)
2. arXiv:2602.18502 — read via arXiv HTML through Methods §3.5-start. Results §4/Discussion §5
   NOT read (fetch truncated twice, two different renderers). No FM or OOD/reliability
   evaluation *in the portion read*; unconfirmed for the rest. Worth one more retrieval attempt
   (smaller chunked fetch) before treating as fully closed.
3. Shortkit-ML — full text read. Includes FM families (DINOv2/RAD-DINO/MedSigLIP) for shortcut
   *detection* only; has a mitigation module but does not exercise/report it in this paper; no
   Mahalanobis/OOD-reliability scoring; no dermatology.

## Tier 1 — needed before experimental design is locked
1. PanDerm (arXiv:2410.15038 / Nat. Med.) — architecture, checkpoint access, feature-extraction interface and license.
5. "Dissecting Mahalanobis: How Feature Geometry and Normalization Shape OOD Detection" (arXiv:2510.15202) — mechanistic grounding for Paper 3's null result; check applicability to FM embedding geometry.
6. "One Model, Many Behaviors: Training-Induced Effects on Out-of-Distribution Detection" (arXiv:2601.10836) — check for overlap with Paper 3's dose-response design outside medical imaging.

## Tier 2 — medium read (abstract + figures + results + conclusion)
- "Self-Supervision on Images and Text Reduces Reliance on Visual Shortcut Features" (arXiv:2206.07155)
- "Transformer-Based Foundation Learning for Robust and Data-Efficient Skin Disease Imaging" (Diagnostics 2026, PMC12897459)
- "Toward Clinically Ready Foundation Models in Medical Image Analysis" (arXiv:2603.14271)
- StaRFM (arXiv:2507.09222)
- "On the OOD Robustness of Foundation Models in Medical Image Segmentation" (arXiv:2311.11096)
- "Delving into OOD Detection with Medical Vision-Language Models" (arXiv:2503.01020)
- CANDOR (arXiv:2607.18451)
- "Frozen Foundation-Model Embeddings Discard Small-Lesion Signal in Chest Radiography" (arXiv:2606.11606)

## Tier 3 — quick mapping (abstract + main claim only)
- "Visual concept ranking uncovers medical shortcuts used by large multimodal models" (arXiv:2602.05096)
- "Robustness and sex differences in skin cancer detection: logistic regression vs CNNs" (arXiv:2504.11415)
- "Effect of Demographic Bias on Skin Lesion Classification" (arXiv:2606.03214)
- npj Digital Medicine, "Shortcut learning in medical AI hinders generalization..."
- "Towards Scalable Foundation Models for Digital Dermatology" (arXiv:2411.05514)
- MONET (derm-specific CLIP)

## Not yet searched (next session)
- Cluster B (representation learning / domain-invariant learning applied to foundation models)
- Cluster E (representation geometry of foundation-model embeddings specifically — anisotropy, condition number, linear separability studies on ViT/CLIP-style medical encoders)
- Systematic Google Scholar / Semantic Scholar citation-graph search (this session used web search only)
