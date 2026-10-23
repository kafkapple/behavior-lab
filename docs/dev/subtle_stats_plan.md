# SUBTLE viewer: model, kinematics and statistics plan (261023)

Stages P9 to P12 of `subtle_viewer_prd.md`. Each stage is one commit with its own check; the next starts after the check passes.
Nothing here is built yet. Facts come from the read-only copies in `~/dev/external/subtle_ref/` and from our own fits.

## 0. Findings that change the design
- Node layout mismatch (must be solved by node name):
  - Our `.json.gz` has 11 nodes: nose, neck, earL, earR, forelegL, forelegR, tailstart, hindlegL, hindlegR, tail, tailend.
  - The Justin `avatar_configs` (`kinematics.py`) assumes 9 nodes addressed by index: nose, neck, anus, chest, rfoot, lfoot, rhand, lhand, tip.
  - Its indices on our files would measure wrong joints (index 2 is earL here and "anus" there).
- The app's DEK takes the mean of each kinematic feature per cluster per animal, then compares groups per cluster and feature with a Welch t-test.
  Its p-value comes from a self-written t CDF ("simplified" in the code), with no multiple-comparison correction and no star display found.
- The app trains a Mapper, stores it as `.pkl`, and later runs it on other files. The viewer refits every time (P9 fixes this).
- 3D points can be copied from the previous frame when fewer than 2 cameras saw them. Kinematics of such frames (speed 0, jumps) are not behavior.

## 1. P9 Train, save, load, map
- Input: sessions with `skeleton3d` only. Video is not needed.
- `train`: fit on the chosen sessions with a seed; write `<root>/.subtle_models/<name>/model.pkl` and `model.json`.
  `model.json` holds model id, config, seed, training sessions, frames per session, cluster counts per level, creation time, code commit.
- `map`: load a model, run `predict_raw` on new sessions, write sidecars whose `fit_id` is the model id.
- Check 1: map the training sessions again; report label agreement with the training labels (expected below 100 percent because mapping classifies by nearest neighbor).
- Check 2: the same model, mapped twice, gives identical labels.
- Check 3: frequency refuses sessions from different model ids (already enforced through `fit_id`).
- Risk: `pickle` of the Mapper depends on package versions; store the versions in `model.json` and refuse a load on mismatch.
- Risk: the isolated subprocess must write and read the model file, not return objects.

## 2. P10 Kinematics and DEK
- Config by node name for the 11-node skeleton: nodes, edges (from `node.line`), angles (body, head, forelimbs, hindlimbs, tail; names and triplets listed in the doc before coding).
- Features: for n = 0, 1, 2 (the value, its first and its second time derivative, as in `Kinematics`): node norms, edge terms, angle terms. One vector per frame; fps 20.
- Reuse the public `subtle.Kinematics` class (it takes any `nodes`, `edges`, `angles`); no copy of Justin's file.
- Per animal and cluster: mean of every feature over that cluster's frames; clusters with fewer than a minimum number of frames per animal (default 30) are left out and reported.
- Frames whose 3D points were copied (P5.5 flag) are excluded from the means; the excluded share is shown.
- Group test per cluster and feature: Welch t-test from `scipy.stats.ttest_ind(equal_var=False)`; permutation test as the option for 3 vs 5 animals.
- Correction: Benjamini-Hochberg across all cluster x feature tests of one level, with an off switch. Both raw and corrected p are shown.
- Effect size: standardized difference (Hedges g) next to every p.
- Output: heatmap of t (rows cluster, columns feature), cell mark by corrected p, n animals per group, level and `fit_id` in the title.
- Check: reproduce the app's t-values on the same inputs (within 1e-6) before we trust ours; the p-values differ by design (exact CDF).
- Check: a feature built from shuffled group labels gives about 5 percent raw p below 0.05 over many cells.

## 3. P11 Cluster statistics and plots
- Unit: animal. For each cluster, the share of the animal's valid frames in that cluster.
- Test: exact permutation (all splits) by default. Smallest possible p for 3 vs 5 is 0.0357; the page prints this bound beside the result.
- Plot: bar or box per group with one dot per animal, group mean, error bar = standard error, star line above.
- Stars: * p<0.05, ** p<0.01, *** p<0.001, ns otherwise; computed from the corrected p of the chosen level; the correction family = clusters of that level.
- Always shown: n animals per group, level, `fit_id`, seed, the test name. Folder-name groups carry the label "provisional".
- Export: PNG and a CSV with every number in the plot.
- Check: a hand-computed example (4 animals, known shares) matches the table and the stars.
- Check: stars disappear when the groups are swapped for a random split.

## 4. P12 Stability
- Run the fit with seeds 0, 1, 2; show the adjusted Rand index between seeds for sub and super labels at the chosen level.
- A group result is shown as "stable" only if the star pattern repeats across the seeds. Otherwise the page says "seed-dependent".
- Level choice: show the result for every level in a small table, so no one level is picked after seeing the stars.

## 5. Residual risks (named, not hidden)
- Only the first 3,000 frames (5 minutes) per animal were fitted; full length (about 12,000) is untested for time and memory.
- Group names come from folder names; there is no dosing record. Every group figure carries "provisional".
- 3D quality: copied points and left-right swaps of hind legs are known (see the vault AVATAR 3D defects note).
- The viewer's kinematics are our own node mapping; the app's DEK on AVATAR may use different joints, so numbers will not match the app.
- Multiple testing grows with levels x clusters x features; the plan fixes one level before looking and reports the correction family.
