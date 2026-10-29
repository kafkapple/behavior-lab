# SUBTLE walkthrough: video → keypoints → project → train → map

The viewer opens an AVATAR `*.json.gz` with `skeleton3d` and optional `keypoint`/video. Its **Video tutorial** module makes a real 300-frame example from the CORT source video, using the SLEAP 1423 checkpoint on eren. The Mac hosts the viewer and saves outputs; SSH access to eren is required for this inference step. Other raw videos need the AVATAR upstream pipeline. The viewer never changes the source JSON; labels go into `*.subtle.json` sidecars. For transitions, DIB and TPI, see [the focused survey](subtle_transition_dib_tpi_survey.md).

## 1. Pick a data root

Run `subtle` for the local CORT28d demo (`data/subtle/cort28d/`, 8 sessions) or `subtle /absolute/path/to/dataset` for another root. The full root path appears beside “SUBTLE Lite”; the Sessions panel shows folders, sessions, videos and elapsed scan time. A source folder must contain `*.json.gz` below it. If the list is empty, check the displayed root and the file extension.

| Sample | Where | Ready for this viewer? |
|---|---|---|
| CORT28d, cort1–3 and veh1–5 | local `data/subtle/cort28d/{cort,veh}/*_full_sleap_1423.json.gz` | Yes; 8 session files and existing camera proxies |
| Shin Juhee, G3, 8 recordings | eren `/home/data/kafkapple/data/mouse-avatar-2605g3/derived/subtle_by_juhee/*.json.gz` | Source files verified on eren; one checked JSON has `skeleton3d` and `keypoint`. No local copy was found in this repo. |
| Other AVATAR recordings | prepare `skeleton3d` JSON with the AVATAR keypoint and triangulation pipeline | Only after checking JSON format and camera alignment |

The G3 raw composite videos are in the sibling `raw/` folder; seven recordings have split camera files under `processed/<recording>/`. The viewer pairs camera videos anywhere below the selected root when their recording stem and `cam_1`–`cam_5` match. Select the G3 dataset root to include both `derived/` and `processed/`; selecting `derived/` alone shows the skeleton without video. The damaged `260529-baseline-G3-M3` recording has a decision pending on usable frames. Do not treat its final frames as aligned without that decision. The G3 group names and dosing are not verified by folder names.

## 2. Inspect the input

For a video-only start, open **2 · Keypoints → Video tutorial** and click **Predict keypoints from video**. It cuts a short real CORT clip on eren, runs the verified SLEAP checkpoint, downloads `tutorial/cort1_toy.json.gz` and the composite video, and splits five camera views. The step progress shows clip, inference, download and validation. Press **Reload session list**, open the toy session, turn on **Conf %**, and compare points, edges, video and 3D motion. `tutorial/provenance.json` records source, model and time. This one animal cannot support group statistics; use cohort examples for that stage.

Wait for the Sessions scan to finish. Click one session. Check that the 3D skeleton moves, frame count is plausible, and available camera overlays align with the video. The `Quality` tab runs label-free consistency checks; it is not ground-truth 3D accuracy. For G3, first compare the provided keypoints with the five camera views and note the shorter keypoint streams reported in the G3 data plan.

Upstream stages for a new recording are: composite video → five camera views → 2D keypoints (SLEAP or Lightning Pose) → calibrated 3D skeleton → viewer input. The tutorial runs this path for its fixed toy clip. The AVATAR rig's cell geometry lives in `configs/rig/avatar.yaml`; do not infer camera coordinates from the displayed video size.

## 3. Create a project and assign sessions

Open the **Project** tab. Type a project name, for example `cort28d_review`, then type a group name such as `cort`. Open each intended session in the left panel and click **+ add open session**. Repeat for `veh` and its sessions. Add a description and the animal, condition and date fields, then **Save**. Use **Load** to reopen the saved project. Project membership is the data selection; the fit runs jointly on all its sessions so cluster IDs are comparable. Check every group and session before running: the group labels should come from metadata, not guesses about file names.

For G3, point `subtle <local-G3-root>` at a **local copy** of the verified JSON files, make a project with the eight recordings, and set groups only after the experimental assignment is confirmed. Running against the eren directory requires that directory to be mounted locally. Raw videos alone are not a viewer input.

## 4. Explore or save a mapping model

Open **3 · Discover & map** and set `frames/session` (0 = all), `seed` (0 = repeatable), and `Method` there.

- **Explore clusters**: fit the chosen method jointly and write session sidecars; no reusable model is saved. The default is original SUBTLE's UMAP + Phenograph. The B-SOiD option has UMAP neighbors and HDBSCAN minimum-cluster-size settings; it is for exploration only.
- **Train & save**: enter a model name, then start training. It fits the project sessions and saves `.subtle_models/<name>/model.pkl` and `model.json` under the data root. Train requires a `subtle/*` method. The model record includes source paths, seed, method and creation time.
- **Map new sessions**: choose a saved model, assign target sessions to the current project, then map. It labels all frames of those sessions and records model name and mapping time in sidecars. The selected model, not a new fit, supplies the mapping. Check the fit ID and mapping log; mapped training frames can differ from original fit labels.

The expandable method table in Discover & map distinguishes the runnable SUBTLE and B-SOiD paths from Keypoint-MoSeq. Keypoint-MoSeq uses a separate uv environment in this repository and its fit is not yet connected to this viewer; the [method paper](https://doi.org/10.1038/s41592-024-02318-2) and [fitting documentation](https://keypoint-moseq.readthedocs.io/en/latest/fitting.html) describe its temporal model and parameters. B-SOiD's UMAP/HDBSCAN route is described in its [method paper](https://www.nature.com/articles/s41467-021-25420-x). Do not compare raw cluster IDs across these methods.

The Discover & map job line and bar show the stage, percent, elapsed time and error. The Log module has detailed messages. SUBTLE fits can take minutes; start with a small `frames/session` limit before deciding on a full fit. Seed stability launches three fits and is a separate, longer job.

## 5. Read and export

The timeline and map show labels for the selected level. Frequency requires complete labels and one fit ID across sessions; a red Σ badge means the comparison is invalid. Stats, DEK, level sweep, seed stability and Export results are separate actions. For any group claim, inspect animal-level shares, corrected statistics, and bouts/transitions; a cluster map alone is insufficient. The Stats tab can export CSV/PNG; Export results writes a bundle under `outputs/subtle/<project>/`.

The G3 source contains a previous `subtle` result. Opening it is for inspection; Train or Map creates this viewer's own sidecar. Keep the previous result and the new fit distinct, and record the seed, source file and model ID when comparing them.
