# SUBTLE viewer: where files live (plan, 261023)

Status: S1 to S3 done 261024 (data root `data/subtle/cort28d/`, results `outputs/subtle/`, retention tool). Applies to `tools/subtle_viewer/`.

## 1. Problem
- Results were written to `<root>/.subtle_results/` of a scratch folder. Scratch is wiped on reboot, so a deliverable can vanish.
- One-off scripts, logs and screenshots piled up next to the data, and some paths were written into handoffs.

## 2. Three classes of file
- Source: skeleton `.json.gz` and raw videos. Authoritative copy stays on the server; a local copy is never the master.
- Working store: proxies (`*.mp4`), sidecars (`*.subtle.json`), `.subtle_projects/`, `.subtle_models/`. Regenerable from source plus the project file.
- Deliverables: one result folder per export (tables, figure, clips). Small, citable, kept.

## 3. Proposed locations (repo convention: top-level `data/` and `outputs/` are gitignored)
- Data root: `data/subtle/<dataset>/` holds sessions, proxies, sidecars, `.subtle_projects/`, `.subtle_models/`.
- Results: `outputs/subtle/<project>/<YYMMDD_HHMMSS>_n<k>_<criterion>/`.
- Every result folder carries `summary.json` with project, `fit_id`, seed, criterion, group sizes and the code commit, so it can be rebuilt.
- Vault notes link to the result folder and quote the headline numbers only; no video or csv is copied into the vault.

## 4. Decision (261024)
- The data root is `data/subtle/cort28d/` inside `behavior-lab`; the rig calibration sits in `data/avatar/calibration/`.
- Result bundles for sharing (the clips and `sweep.png`) can also be copied to the lab Drive folder by hand; nothing syncs automatically.

## 5. Rules
- One-off scripts and logs go to the job temp directory, never to the repo, the vault or a data root.
- Docs name only durable paths. A temporary path is never written into a manual, note or handoff.
- A check worth keeping becomes part of `selftest.py` or `browser_check.mjs`.
- Retention: keep the latest 3 result folders per project and any folder containing a file named `KEEP`; list the rest and ask before deleting.

## 6. Steps (each one has a check)
- S1 done: data moved; the browser check passed 30 of 30 on the new root.
- S2 done: `--results-dir` (default `outputs/subtle`); `summary.json` carries `commit` and `project`; selftest checks the path rule.
- S3 done: `clean.py` lists result folders beyond the latest 3 per project (a `KEEP` file protects one); it deletes only with `--apply`.
- S4 end-of-session sweep in 3 lines: temp files gone, docs name no temp path, newest result folder linked in the status note.
