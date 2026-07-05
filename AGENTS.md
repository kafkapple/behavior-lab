# AGENTS.md

## Scope

Use these instructions for all work in `behavior-lab`.

## Project Shape

- This repo is a modular skeleton-based behavior recognition platform.
- Core code lives under `src/behavior_lab/`; tests live under `tests/`.
- Data belongs in `data/{dataset_name}/` and must not be invented or committed.
- Reports and generated artifacts belong under `outputs/` or established `reports/`/notebook output folders.

## Commands

- Install for development: `pip install -e ".[dev]"`
- Unit tests: `pytest`
- E2E smoke: `python scripts/test_e2e.py`
- Lint: `ruff check .`

## Working Rules

- Before changing behavior-analysis notebooks or docs, read the relevant README/doc first.
- Do not fabricate metrics, plots, tables, or dataset availability. If data is missing, state that explicitly.
- Treat no code change as a valid outcome when the issue is documentation, missing data, config, or already fixed behavior.
- For CEBRA, clustering, and behavior discovery results, do not rely on one metric alone. Check silhouette together with ARI/NMI when labels exist and bout/transition structure when relevant.
- For long-running training, external downloads, or large dataset processing, ask before running.

## Verification

- Run the smallest relevant test for the touched area.
- For loader/preprocessing/model changes, prefer targeted `pytest tests/...` first, then broader tests if risk warrants.
- For notebook/report work, verify the generated output path and summarize limitations.
- After important changes, perform a critical verifier pass: assume the change may be wrong, check edge cases and whether no change would have been better.
