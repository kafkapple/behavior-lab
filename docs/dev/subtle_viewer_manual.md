# SUBTLE Lite Viewer manual (261023)

## Start
- Start with `subtle_workthrough.md` in Help for data → project → Train/Map. Help opens a module list; `?` opens it and Esc closes it.
- Terminal: `subtle [folder] [--port N] [--no-open]` (default folder is local `data/subtle/cort28d/`, port 8777). The Sessions panel shows scan progress and the active root.
- Deep link: `http://127.0.0.1:8777/#open=<path>&cam=<0-4>&frame=<n>` opens a session at a frame and camera.
- **Saved results** in the left Results & reference section lists prior result bundles for projects in the active data root. It shows each README and PNG, with CSV/JSON downloads. It does not recompute statistics. Use `subtle` for the CORT L8 figures, `subtle g3` for the G3 quality figure, or `subtle trainmap` for the saved model and held-out mapping record.

## Files the viewer reads
- Result file: `*.json.gz` with `skeleton3d` (3D, 11 joints) and `keypoint` (2D per camera). No unzip needed.
- Video: per-camera `*.mp4` whose name has the animal token and `cam_N` (for example `..._cort1_cam_2.mp4`). Subfolders do not matter.
- Large originals: make a small copy for display with `tools/subtle_viewer/make_proxy.sh raw.mp4 out.mp4`. The script checks that the frame count is equal.
- Sidecar `<name>.subtle.json` next to the result file: cluster labels, embedding, `fit_id`, `seed`. The original file is never changed.

## Screen
- One screen, no page scroll. Left: five collapsible workflow sections and sessions. Center: video with 2D keypoints, 3D skeleton, cluster map. Right: the selected workflow module, including Help. Bottom: play, slider, timeline, key list.
- Top toolbar: cluster level. The Discover & map module holds fitting controls.

## Views
- 3D skeleton (follows the animal): wheel = zoom, drag = rotate, shift or right-drag = pan, double-click or Enter = back to the selected camera's view.
- The default 3D pose is the selected camera's view: the server fits a weak-perspective camera from the 2D keypoints to the 3D points (`core.cam_view`) and rotates the skeleton to it. The footer prints the camera; "free view" means no fit was possible.
- Cluster map: wheel = zoom (around the cursor), drag = pan, click = jump to the nearest frame, double-click = reset.
- Video: use **All cameras** or press 0 for every available camera; select a tile or press 1-5 for one camera. Only cameras with video appear.
- Keypoints: the overlay scales down with smaller camera tiles. **View settings → Color scheme** defaults to the BS AVATAR body-part palette in both 2D video and 3D skeleton. Each joint has a fixed anatomical color; an edge takes its destination joint's color, matching the BS overlay. Thus an edge color change along the skeleton describes body-part identity, not a transition, motion magnitude, or confidence. **Legacy cluster edges** restores the old overlay: green 2D points and current-cluster-colored edges/3D skeleton. The separate Cluster ID badge retains the cluster color in either mode. Low-confidence points are dim and edges dashed. **Conf %** shows the per-point detector confidence. Key `k` toggles all keypoints.
- Why an edge can disappear: the skeleton topology contains only predefined joint pairs, and the viewer draws a pair only when both endpoints have usable 2D detections in that camera (`confidence > 0.05`). No line between two otherwise visible points can simply mean they are not a defined anatomical edge. Missing lines do not establish that the body part is absent. The 3D view may still show a filled point from earlier frames.
- Timeline: one color per frame; click or drag to scrub, also while playing. A hatched region means no label exists there: labels cover only the frames that were fitted (frames/session). Map the session with a saved model to label all frames.

## Clusters
- One id has one color in every view (Tableau palette). Ids mean something only inside one fit; the legend shows the `fit_id`.
- For a sourced explanation of transition probabilities, DIB supercluster merging, and TPI, see [the focused survey](subtle_transition_dib_tpi_survey.md).
- `cluster` select: `sub` or `super`. For `super` the second select lists the levels (`L0: n clusters`); the default is the level with the most clusters. Why levels: `subtle_theory.md`.
- Clusters tab (right): one row per id with color, name field and share. Click = filter, double-click = solo, `all`, `none`. With a filter, playback skips other clusters and `next bout` jumps to the next run.
- Names: type in the name field; it is saved per fit and level in `<root>/.subtle_projects/cluster_names.json` and shown in the Stats tab.

## Project metadata
- Project module: a description box, a note per group, and animal, cond.(ition), date per session. `Save project` writes them into `<name>.project.json` under `meta`; the server keeps only these fields, 200 characters each. The save timestamp and data root are also recorded. If a comparison needs a saved project, the app opens this module and highlights the Save button.
- Deep links for checks: `#open=<path>&cam=1&frame=600&all=1&project=<name>&tab=proj` (all = every camera tile; tab = clu, proj, freq, stat, log).

## Running SUBTLE
- Add sessions to groups, then choose **Explore clusters** in Discover & map. One fit covers all sessions, so IDs are comparable. **Train & save** creates a reusable SUBTLE model; **Map new sessions** applies it to project sessions.
- Clustering: `UMAP + Phenograph` (original SUBTLE, default) or `PCA consensus` (5 Leiden runs in PCA space, merged, cluster count matched to the UMAP path; UMAP only for the map). A third entry `B-SOiD (UMAP + HDBSCAN)` labels 10 fps bins mapped back to frames, -1 = noise (grey); it serves Analyze only, Train and Seed stability run SUBTLE (the server says so). The list comes from `/api/methods`; the sidecar records `method` and `clustering`. PCA consensus is slower: all 95,991 demo frames took 2,884 s. Evidence and the hold rule: vault `_Notes/261024_SUBTLE_개선_연구계획`.
- Set a seed (default 0) for a repeatable fit; empty = random. What the seed controls: `subtle_theory.md` section 4. Max frames limits the time (3,000 frames for 8 animals took about 100 s).
- Model: `Train & save` fits the project sessions once and saves `<root>/.subtle_models/<name>/model.pkl` and `model.json` with source paths, method, seed and creation time. `Map new sessions` loads a saved model and labels every frame of the project sessions with it; mapped sidecars record model name and mapping time.
- **Transitions** computes one-frame exit probabilities and self-retention per cluster. Group heatmaps are descriptive animal/session-level means with per-row support; see [survey and definitions](subtle_transition_dib_tpi_survey.md).
- **Stop server** is under the left Reference section and closes the local HTTP server after responding to the button click.
- Frequency table: the sum must be 100 and all sessions must come from one fit; otherwise the badge turns red.

## Statistics
- Stats tab, `Compare groups`: for each pair of groups, the share of each cluster per animal, exact permutation p of the mean difference, Hedges g, Benjamini-Hochberg adjusted p, and stars (* below .05, ** below .01, *** below .001, on the adjusted p). One dot = one animal; bars = mean ± sample SD, clipped to the 0–100% share axis. These error bars describe animal variation, not a confidence interval.
- The page prints the smallest possible p for the group sizes; with 3 vs 5 animals it is 0.0357, so ** and *** cannot appear.
- Demo (8 animals, cort 3 vs veh 5, level L4, first 3,000 frames): adjusted p 0.77 to 1.0, no star. Groups are provisional (folder names).
- DEK tab, `Compute DEK`: load a saved project with two groups and labels from one fit, select the desired `super` level (or `sub`), and press the button. The progress bar shows which session is being converted to kinematic features; the page remains usable. The operation has the same steps as the Justin app, with our 11-node layout by node name.
  - Frames are centered like `avatar_preprocess`, then the public `subtle.Kinematics` gives 84 features: speed, edge and angle terms, each as value (n=0), first (n=1) and second (n=2) time derivative.
  - Per animal and cluster the features are averaged; an animal needs at least 30 frames in a cluster. Welch t per cell (scipy `ttest_ind`, same call as the app), Benjamini-Hochberg over all cells of a pair, Hedges g.
  - Heatmap: rows = clusters, cell = signed t (orange = first group higher, blue = second), `*` on adjusted p; hover shows p, adjusted p, g. Pick type (vertex, edge, angle), derivative and pair above the table.
  - Demo (cort 3 vs veh 5, level L4, 420 cells): smallest adjusted p 0.38, no star. Shuffling the 8 animals over all 56 splits gave 4.6 percent raw p below 0.05, as expected for no effect.
  - Not excluded: frames whose 3D points were copied from earlier frames (the viewer's loader reads no per-frame flag). Speed terms of such frames are zero, so read speed differences with care.
  - `tools/subtle_viewer/sample_report.py <data-root> --project cort_vs_veh --level 8` writes a descriptive CSV, PNG and provenance bundle under `outputs/subtle/`. This computes on existing mapped sidecars; it does not refit SUBTLE.
  - Full eight-session CORT example, fit `9fc3c01d`, super L8, 95,991 frames: nine share tests had no BH-adjusted p below .05 (minimum .0804). DEK tested 756 cells and four had adjusted p below .05 (minimum .0191): cluster 0 `E0:neck-earR`, `V1:earR`; cluster 2 `E0:neck-earR`, `E0:tailstart-neck`. These are exploratory signals in folder-defined provisional groups, with possible copied-3D-frame artifacts. They are not a treatment conclusion. The saved bundle's `summary.json` records the exact data and code state.

## Level sweep and saved results
- Question it answers: which number of superclusters (level) shows a group difference, without fooling ourselves by trying many levels.
- "Most clusters" is not a valid rule: it ignores the data. The sweep compares every level on one criterion chosen before looking (select next to `Sweep n`).
  - `omnibus` (default): PERMANOVA pseudo-F on Hellinger-transformed shares of all clusters together. One number per level; handles the sum-to-100 constraint; no cluster multiplicity.
  - `maxt`: the largest |Welch t| of any cluster.
  - `nsig`: number of clusters whose raw permutation p is below .05.
  - `meanp`: mean raw permutation p over the clusters (lower is better).
- Null and selection correction: every criterion is a statistic with its own null over all splits of the animals into groups of the observed sizes (exact up to 20,000 splits). Levels are compared by the z-score against that null; the best level's z is referred to the null of the best level over all splits, which gives p_fw (Westfall-Young max-T logic applied across levels). Cluster flags use max-T over all levels and clusters.
- Why this is valid here: the clustering never sees the group labels, so permuting labels does not require refitting. Conditions: units exchangeable under the null, criterion chosen beforehand, levels from one fit. `/fact --deep` (Gemini, GPT, web) agreed on these conditions; PERMANOVA also reacts to unequal group spread, and zero shares need Hellinger rather than Aitchison.
- Reading it: if p_fw is not below .05, the page says in red that no difference should be reported. With 3 vs 5 animals the smallest possible p is 0.0179 (1 of 56 splits).
- `Use n=k` switches the level; the legend marks significant clusters with ★ and exploratory ones (raw p below .05, not significant) with ☆. `Show flagged only` filters the map, timeline and replay to them.
- `Export results` writes `outputs/subtle/<project>/<time>_n<k>_<criterion>/` (server option `--results-dir`; `summary.json` records the code commit): `sweep_levels.csv` (all criteria, plus the recommended n per criterion), `clusters.csv`, `summary.json`, `sweep.png`, and `cluster_<id>_<tag>.mp4` (up to 3 bouts per group from the proxy video, 2D keypoints and a caption). If nothing is significant, the 2 clusters with the largest |t| are clipped and tagged `exploratory_not_significant`. A group without per-camera video is skipped and the log says so.
- Checks: planted effect found, hand-known p (2 of 70), and all 56 assignments of the real 8 animals gave 2 false positives (3.6 percent) for each of the four criteria.
- Demo (cort 3 vs veh 5, first 3,000 frames): omnibus n = 3 (p_fw 0.55), maxt n = 3 (0.77), nsig n = 2 (1.0), meanp n = 3 (0.61); joint fit 63f43998 after the CWT fix. No criterion is significant.

## Proxy videos for all sessions
- `python tools/subtle_viewer/fetch_cams.py <root> --project <name>` makes five camera proxies per session next to the session file. The 5-camera composite (about 10 GB) is copied from the server one session at a time, cut into the five cells in one decoding pass (cells from `core.CELLS`), written half size (H.264 CRF 28, keyframe every 20 frames), checked against the session's frame count, and then deleted.
- Safe to re-run: sessions with all five proxies are skipped. Needs 14 GB free. Overlay viewing only; inference must use the raw video.
- Defaults point at the CORT 28d composites on the lab server; change `--remote-dir` and `PREFIX` in the module for another dataset. Time: about 25 to 30 minutes per session.

## Result retention
- `python tools/subtle_viewer/clean.py` lists result folders beyond the latest 3 per project; a file named `KEEP` inside a folder protects it. `--apply` deletes the listed folders.

## Seed stability
- Stats tab, `Seed stability`: refits the project sessions with seeds 0, 1, 2 (isolated, sidecars untouched; about 80 s per seed for 8 animals at 3,000 frames) and saves `<name>.stability.json`.
- Shows the adjusted Rand index (ARI) between seeds for subclusters and for the default supercluster level, and the group test at every level (min adjusted p, number of clusters below .05), so no level is chosen after seeing the stars.
- `stable` means every seed agrees on whether any cluster passes .05 at its default level; it can also mean a stable null. `seed-dependent` means they disagree.
- Also reports AMI (does not favour coarse partitions) and label run lengths (median, mean, share of 1-frame runs), cut at session borders.
- Demo refit 261024 (3,000 frames, seeds 0-2): UMAP + Phenograph ARI 0.44, AMI 0.72, 39% 1-frame runs; PCA consensus ARI 0.60-0.62, AMI 0.79-0.80, 4% 1-frame runs. Both gave no cluster below .05.
- Demo (8 animals, first 3,000 frames): ARI 0.41 to 0.45 for subclusters (61 to 66 clusters) and 0.40 to 0.60 for the default supercluster level. No seed gave a cluster below .05. The partition itself is only partly reproducible across seeds, so keep the seed fixed and say which seed a figure used.

## Keys
- Same keys as the BS viewer: Space play, Left/Right 1 frame, Up/Down 10 frames, `,` `.` 1 frame, `+` `-` zoom skeleton, Enter reset view, `h` hide side panels, `?` manuals, `Esc` clear filter.
- SUBTLE viewer additions: `1`-`5` camera, `0` all cameras, `k` keypoints, `l` Clusters tab.

## Limits
- The unit of a group comparison is the animal, not the frame. With 3 vs 5 animals the smallest possible exact permutation p is 0.0357.
- Group names come from folder names; there is no dosing record for the AVATAR `CORT28d` animals.
- 3D points can be copied from the previous frame when fewer than 2 cameras saw them, and no NaN is shown (see `subtle_pipeline_io.md`).

## More
- Theory (DIB, levels, randomness): `subtle_theory.md`. Stage input and output: `subtle_pipeline_io.md`. Planned model, DEK and statistics: `subtle_stats_plan.md`. Plan and design basis: `subtle_viewer_prd.md`.
