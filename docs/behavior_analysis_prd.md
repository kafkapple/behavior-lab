# Behavior Analysis PRD

> SSOT for the project scope, operating principles, and comparison contract.
> Keep this short. Details live in [behavior_analysis_workbench.md](behavior_analysis_workbench.md)
> and [behavior_analysis_principles.md](behavior_analysis_principles.md).

## 1. Objective

Build a reusable, notebook-first behavior-analysis workbench that can:

1. ingest pose data from DLC, SLEAP, triangulated 3D, and canonical local datasets,
2. convert them into a shared `(T, K, D)` contract,
3. compare discovery methods on the same feature/metric axes, and
4. produce reproducible visualizations for motifs, syllables, bouts, and transitions.

The project is not a single model. It is a comparison system.

## 2. Users And Questions

- Researchers comparing pose-derived behavior discovery methods across datasets.
- Questions:
  - Which pose source is robust enough for the dataset?
  - Which feature block matches the behavioral question?
  - Which discovery method yields stable syllables or motifs?
  - Which result is comparable across datasets, and which is not?

## 3. Scope

### In scope

- pose loaders and canonical sequence normalization
- DLC / SLEAP / triangulation handoff
- feature blocks: raw, kinematic, dyadic, spectral, learned embeddings
- discovery families: PCA/KMeans, B-SOiD, keypoint-MoSeq, SUBTLE, hBehaveMAE
- comparison metrics: silhouette, CH/DB, ARI/NMI when labels exist, bout duration, transition entropy
- notebook-driven exploration, batch comparison, and report generation

### Out of scope

- one-off notebook logic that cannot be reused as a module
- method-specific fallbacks that hide missing dependencies or mismatched input shapes
- claims of biological truth from a single metric
- mixing 2D and 3D results without reporting geometry and missingness

## 4. Operating Principles

1. **SSOT first**: one canonical data contract, one method catalog, one metrics definition.
2. **Minimal implementation**: add abstractions only when they reduce duplication or mismatch.
3. **Module boundaries**:
   - `behavior-tools` stays on acquisition, splitting, curation, and export.
   - `behavior-lab` owns pose loading, feature extraction, discovery, metrics, and notebooks.
4. **Comparable layers**: compare methods only when pose source, temporal scale, and output type are explicit.
5. **No silent fallback**: failures should be visible, typed, and documented.

## 5. Evaluation Contract

### Pose layer

- keypoint coverage / missingness
- triangulation quality or confidence distribution
- coordinate frame and units

### Representation layer

- raw geometry preservation
- kinematic stability
- temporal consistency
- cross-dataset transferability

### Discovery layer

- silhouette / CH / DB for internal clustering quality
- ARI / NMI only when an annotation vocabulary is comparable
- bout duration, transition matrix entropy, and syllable usage for temporal structure

### Visualization standard

- one embedding plot, one bout-duration plot, one transition matrix, one ethogram per run
- same color palette per method family across datasets
- same temporal resolution reported in every figure caption

## 6. Recent Method Baseline

Use these families as the current comparison anchor:

- pose source: SuperAnimal / DLC / SLEAP / triangulation
- representation: CEBRA-style time embedding, hand-crafted kinematics, spectral windows
- discovery: B-SOiD, keypoint-MoSeq, SUBTLE, hBehaveMAE

Interpretation rule:

- better silhouette means better geometry separation, not necessarily better biology
- better transition structure means the sequence grammar is more coherent, not necessarily more correct
- if metrics disagree, keep the disagreement visible instead of averaging it away

### Selected recent references

- [SuperAnimal](https://arxiv.org/abs/2203.07436) - foundation-style pose estimation for animal behavior.
- [CEBRA](https://arxiv.org/abs/2204.00673) - contrastive latent embedding that also applies to behavior time series.
- [MoSeq protocol](https://arxiv.org/abs/2211.08497) - syllable discovery and transition visualization for spontaneous mouse behavior.
- [B-KinD](https://arxiv.org/abs/2112.05121) - self-supervised keypoint discovery from behavioral video.
- [Animal pose estimation survey](https://arxiv.org/abs/2410.09312) - broader 2024 review of multimodal animal pose estimation.
- [CNN-based animal behavior survey](https://arxiv.org/abs/2301.06187) - supervised / SSL / unsupervised behavior analysis landscape.

## 7. Milestone Slices

1. stable loaders and canonical pose normalization
2. feature catalog and method registry
3. notebook-based single-dataset smoke run
4. batch comparison across datasets
5. consistent visual report and retrospective notes

## 8. Acceptance Criteria

- the same dataset can run through multiple methods without code edits
- notebook results are reproducible from the documented environment
- comparison tables expose pose source, feature block, method, and metric side by side
- every method family has a known failure mode documented in SSOT

## Backlinks

- [Workbench](behavior_analysis_workbench.md)
- [Theory Appendix](behavior_analysis_principles.md)
- [Notebook MoC](../notebooks/behavior_analysis_workbench/README.md)
- [Boundary](../../behavior-tools/docs/behavior_lab_boundary.md)
