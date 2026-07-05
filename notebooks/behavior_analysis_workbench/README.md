# Behavior Analysis Workbench Notebooks

Notebook-first interface for pose-keypoint behavior analysis.

| Notebook | Purpose |
|---|---|
| `00_end_to_end_overview.ipynb` | Local dataset -> features -> discovery labels -> bouts/transitions/embedding smoke run. |
| `01_sleap_import_and_triangulation.ipynb` | SLEAP `.slp`/analysis `.h5` import path and 2D-to-3D triangulation handoff. |
| `02_method_comparison_matrix.ipynb` | Dataset x feature x discovery method comparison template. |
| `03_batch_results_all_methods.ipynb` | Loads the executed batch comparison across all available local datasets/methods. |

Default cells run with lightweight dependencies. Optional visual/SLEAP outputs
activate when `matplotlib` and `behavior-lab[sleap]` are installed.

Main manual: `../../docs/behavior_analysis_workbench.md`.
Project PRD: `../../docs/behavior_analysis_prd.md`.
Theory appendix: `../../docs/behavior_analysis_principles.md`.
