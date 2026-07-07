---
name: behavior-lab-analysis
description: Use for behavior-lab tasks involving CALMS21, CEBRA, behavior time-series, skeleton behavior discovery, notebooks, reports, visualization, clustering metrics, or ethology analysis.
---

# Behavior Lab Analysis

## Workflow

1. Read `README.md` and the relevant document/notebook before editing.
2. Identify the pose source, feature block, discovery method, labels if any, and output path.
3. Separate data availability, code behavior, and interpretation. Do not treat missing data as a code result.
4. Prefer targeted changes in `src/behavior_lab/`, `docs/`, or the relevant notebook/report path.
5. Run the smallest relevant check and report what could not be verified.

## Analysis Rules

- Do not invent quantitative results, plots, labels, or dataset contents.
- For CEBRA, treat embeddings as feature representations, not behavior categories by themselves.
- For unsupervised discovery, inspect silhouette with temporal bout structure and transition behavior; use ARI/NMI only when labels exist.
- For cross-dataset comparison, report pose source, geometry, temporal scale, and method family separately.
- If no code change is needed, say so and explain the evidence.

## Verification

- Loader/preprocessing change: run the targeted `pytest tests/test_data/...` test when available.
- Core tensor/skeleton/graph change: run the relevant `pytest tests/test_core/...` test.
- Notebook/report change: verify output files and summarize limitations.
- Important changes require a verifier pass that checks edge cases, overinterpretation, and no-change possibility.
