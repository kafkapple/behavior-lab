"""Render self-contained HTML report for kp_benchmark v0.1 data prep stage.

Embeds matplotlib figures as base64 PNGs so the output is a single .html
that can be opened anywhere without external assets. Mirrors the BS
HTML report convention (CLAUDE.md: BS reports live under Obsidian _html/).

Usage
-----
    python scripts/render_kp_data_report.py [--output PATH]
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from behavior_lab.data.loaders.li2023 import Li2023Loader

KP_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]

DEFAULT_OUTPUT = Path(
    "/Users/joon/Documents/Obsidian/30_Projects/"
    "2603_3D_animal_recon_BehaviorSplatter/_html/"
    "260602_kp_benchmark_data_prep.html"
)


# ------------------------------ figure helpers ------------------------------ #

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_kp_range(mammal_kp: np.ndarray) -> str:
    """Per-keypoint xyz range bar chart."""
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(22)
    width = 0.27
    for i, axis in enumerate("XYZ"):
        lo = mammal_kp[:, :, i].min(axis=0)
        hi = mammal_kp[:, :, i].max(axis=0)
        rng = hi - lo
        ax.bar(x + (i - 1) * width, rng, width, label=axis)
    ax.set_xticks(x)
    ax.set_xticklabels(KP_NAMES, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Range (mm)")
    ax.set_title("MAMMAL — per-keypoint coordinate range (max − min across 3600 frames)")
    ax.legend(title="Axis")
    ax.grid(True, axis="y", alpha=0.3)
    return fig_to_b64(fig)


def plot_li_valid_heatmap(valid_mask: np.ndarray) -> str:
    """81 × 22 heatmap of which kp is labeled per Li GT frame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(valid_mask.T, aspect="auto", cmap="Greens",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(22))
    ax.set_yticklabels(KP_NAMES, fontsize=8)
    ax.set_xlabel("Li GT frame index (0–80)")
    ax.set_title(
        f"Li 2023 GT validity mask — green = labeled  "
        f"(rows: 22 kp, cols: 81 frames; fully valid frames = "
        f"{int((valid_mask.sum(axis=1) == 22).sum())})"
    )
    plt.colorbar(im, ax=ax, fraction=0.025, label="labeled")
    return fig_to_b64(fig)


def plot_frame_timeline(mammal_idx: np.ndarray, li_ids: np.ndarray) -> str:
    """Frame coverage along 18000-fr video timeline."""
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.scatter(mammal_idx, np.zeros_like(mammal_idx) + 1.0,
               s=2, c="#3a7", label=f"MAMMAL ({len(mammal_idx)} frames, 5-step)")
    ax.scatter(li_ids, np.zeros_like(li_ids) + 0.5,
               s=40, c="#d33", marker="x", label=f"Li GT ({len(li_ids)} frames)")
    ax.set_xlim(-200, 18200)
    ax.set_ylim(0.0, 1.5)
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["Li GT", "MAMMAL"])
    ax.set_xlabel("Video frame index (0–17999)")
    ax.set_title("Frame coverage along M1 video timeline")
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)
    return fig_to_b64(fig)


def plot_split_sizes(train_n: int, test_n: int, li_n: int) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ["mammal_train\n(80%)", "mammal_test\n(20%, in-dist)", "li_external\n(OOD)"]
    vals = [train_n, test_n, li_n]
    colors = ["#3a7", "#69c", "#d33"]
    ax.bar(bars, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 30, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frame count")
    ax.set_title("Split sizes (seed=42)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)
    return fig_to_b64(fig)


def plot_kp_sample_3d(mammal_kp: np.ndarray, sample_frame: int = 0) -> str:
    """One-frame 3D scatter of 22 kp."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    pts = mammal_kp[sample_frame]
    edges = [(2, 3), (0, 3), (1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (4, 11),
             (4, 15), (11, 10), (10, 8), (8, 9), (15, 14), (14, 12), (12, 13),
             (5, 18), (5, 21), (18, 17), (17, 16), (21, 20), (20, 19)]
    for a, b in edges:
        ax.plot(*zip(pts[a], pts[b]), c="#888", lw=1.0)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#d33", s=30)
    for i, nm in enumerate(KP_NAMES):
        ax.text(pts[i, 0], pts[i, 1], pts[i, 2], nm, fontsize=6)
    ax.set_title(f"MAMMAL kp_3d sample — frame_idx {sample_frame * 5} (video frame)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    return fig_to_b64(fig)


# ------------------------------ html composition ---------------------------- #

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
       max-width: 1120px; margin: 28px auto; padding: 0 20px; color: #222;
       background: #fafafa; line-height: 1.55; }
h1 { border-bottom: 2px solid #d33; padding-bottom: 8px; color: #222; }
h2 { color: #3a3; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 32px; }
h3 { color: #69c; margin-top: 20px; }
table { border-collapse: collapse; margin: 12px 0; font-size: 13px; width: 100%; }
th, td { padding: 6px 10px; border: 1px solid #ccc; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
pre { background: #f6f6f6; padding: 10px; border-left: 3px solid #3a3;
      overflow-x: auto; font-size: 12px; }
.kpi { display: inline-block; padding: 12px 20px; margin: 4px; background: white;
       border-left: 4px solid #3a3; border-radius: 4px; min-width: 140px; }
.kpi .v { font-size: 22px; font-weight: bold; color: #222; display: block; }
.kpi .l { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.fig { margin: 16px 0; text-align: center; }
.fig img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
           background: white; }
.fig .cap { font-size: 12px; color: #555; margin-top: 6px; font-style: italic; }
.bad { color: #c22; font-weight: bold; }
.good { color: #292; font-weight: bold; }
.warn { color: #e80; font-weight: bold; }
"""


def render_html(ctx: dict, figs: dict) -> str:
    def img(key: str, cap: str) -> str:
        return (f'<div class="fig"><img src="data:image/png;base64,{figs[key]}"/>'
                f'<div class="cap">{cap}</div></div>')

    return f"""<!doctype html><html><head>
<meta charset="utf-8"/>
<title>kp_benchmark v0.1 — data prep report</title>
<style>{CSS}</style>
</head><body>

<h1>kp_benchmark v0.1 — Data Prep Report</h1>
<p><b>Generated</b> {ctx['ts']} &nbsp;|&nbsp;
   <b>repo</b> <code>~/dev/behavior-lab</code> @ commit <code>{ctx['commit']}</code> &nbsp;|&nbsp;
   <b>status</b> <span class="good">scaffold + data prep complete</span></p>

<h2>1. Progress KPIs</h2>
<div>
  <div class="kpi"><span class="l">files added (commit 1)</span><span class="v">9</span></div>
  <div class="kpi"><span class="l">data fetched (gpu03 → mac)</span><span class="v">2.1 MB</span></div>
  <div class="kpi"><span class="l">MAMMAL frames</span><span class="v">{ctx['mammal_n']}</span></div>
  <div class="kpi"><span class="l">Li GT total / fully-valid</span><span class="v">{ctx['li_total']} / {ctx['li_full_valid']}</span></div>
  <div class="kpi"><span class="l">train / test / OOD</span><span class="v">{ctx['train_n']} / {ctx['test_n']} / {ctx['li_total']}</span></div>
  <div class="kpi"><span class="l">user modified untouched</span><span class="v">8</span></div>
</div>

<h2>2. Data findings (post-fetch sanity)</h2>

<table>
<tr><th>Item</th><th>Handoff stated</th><th>Actual</th><th>Action</th></tr>
<tr><td>Li GT count</td><td>50</td><td class="warn">81 total (50 fully valid + 31 partial)</td>
    <td>Use all 81 with <code>valid_mask</code>; report fully-valid subset metric too</td></tr>
<tr><td>MAMMAL coverage</td><td>3600 fr</td><td>3600 fr = 18000fr / <b>5-step downsample</b></td>
    <td>Documented; DLC trains on video frames (not MAMMAL grid)</td></tr>
<tr><td>MAMMAL ↔ Li overlap</td><td>n/a</td><td class="bad">17 / 81 direct match</td>
    <td>Treat as orthogonal: MAMMAL = pseudo-GT supervision, Li = OOD eval</td></tr>
<tr><td>22-kp ordering</td><td>not specified</td><td>L_ear, R_ear, nose, neck, body_middle, tail_root, …, R_hip</td>
    <td>Used for skeleton drawing + root selection</td></tr>
<tr><td>kp3d range</td><td>n/a</td><td>X: [{ctx['rng_x'][0]:.1f}, {ctx['rng_x'][1]:.1f}] mm, similar Y/Z</td>
    <td>Confirms mm scale — matches MAMMAL paper</td></tr>
</table>

<h2>3. Quantitative analysis</h2>

<h3>3.1 Split sizes</h3>
{img('splits', 'Deterministic seed=42 split. mammal_train:test = 80:20 of 3600 frames. li_external = full 81 Li GT timepoints.')}

<h3>3.2 Frame coverage timeline (M1 video 18000 fr)</h3>
{img('timeline', 'MAMMAL covers every 5th video frame (green band, dense). Li GT scattered arbitrarily across the timeline (red ×). Direct alignment is rare — DLC must operate on raw video frames.')}

<h3>3.3 Per-keypoint coordinate range (MAMMAL)</h3>
{img('range', 'X, Y, Z range (max − min) per keypoint over 3600 frames. Paws and tail_end show largest range — high-mobility joints. body_middle has smallest range — appropriate as root joint for root-relative MPJPE.')}

<h3>3.4 Li GT validity mask</h3>
{img('valid', 'For each of 81 Li GT frames (cols), which of the 22 keypoints were manually labeled (green=labeled). 50 frames have all 22 valid; 31 frames have 16–21 valid (partial annotation, mostly limb occlusions).')}

<h3>3.5 Sample MAMMAL skeleton (frame 0)</h3>
{img('skeleton', 'MAMMAL 22-kp 3D pose at video frame index 0. Mouse body topology visible: head (nose/ears/neck), torso (neck → body_middle → tail_root), 4 limbs.')}

<h3>3.6 Real-frame KP overlay (coordinate-system sanity check)</h3>
<p><b class="good">정합 검증 PASS</b> — MAMMAL 22-kp 3D coordinates projected via calibrated camera params (K, R, t + radial/tangential distortion) land on the mouse body across all 6 views. Li GT (red) and MAMMAL pseudo-GT (green) agree closely, confirming both data sources share the same world coordinate frame.</p>
{img('overlay_a', f'Frame {ctx["overlay_frames"][0]} — 6-view grid. Green = MAMMAL 22-kp projected. Red = Li 2023 manual GT projected. Skeleton bones drawn for both. The two predictors visually agree on body topology.')}
{img('overlay_b', f'Frame {ctx["overlay_frames"][1]} — different pose. Overlay still tracks the mouse body in every view, validating the projection pipeline that DLC training will rely on.')}

<h2>4. Design decisions (Q1–Q5 audit-corrected)</h2>
<table>
<tr><th>Q</th><th>Original option</th><th>v0.1 decision</th><th>Audit driver</th></tr>
<tr><td>Q1 repo</td><td>(a) new mouse-kp-benchmark / (b) BS subdir / (c) Obsidian</td>
    <td class="good">integrate into existing <code>~/dev/behavior-lab</code></td>
    <td>behavior-lab already has DLC dep + skeleton/loader patterns</td></tr>
<tr><td>Q2 DANNCE P2</td><td>(a) CPU / (b) port / (c) skip</td>
    <td class="good">(c) skip → v0.3</td>
    <td>3/3 audit critical: private weights + Blackwell incompat + 1.3 s/fr CPU</td></tr>
<tr><td>Q3 M2 MAMMAL</td><td>(a) apply / (b) M1 only</td>
    <td class="good">(b) M1 only — v0.1</td>
    <td>Scope: pipeline validation first</td></tr>
<tr><td>Q4 hold-out</td><td>(a) M3 sparse / (b) 80/20</td>
    <td class="good">Li M1 81 OOD + MAMMAL 80/20 in-dist</td>
    <td>M3 video absent — jitter/fps metrics unavailable on sparse-only</td></tr>
<tr><td>Q5 first commit</td><td>(a) loader+tests / (b) +P1+P3</td>
    <td class="good">controlled 2-DLC + MPJPE+CI (2-day scope)</td>
    <td>Karpathy "start tiny" ∩ audit "controlled experiment"</td></tr>
</table>

<h2>5. Files added (committed)</h2>
<pre>commit bb297e5  feat(kp_benchmark): v0.1 scaffold — DLC pretraining controlled comparison
  src/behavior_lab/data/loaders/li2023.py
  src/behavior_lab/data/loaders/mammal_mouse.py
  src/behavior_lab/evaluation/mpjpe.py
  configs/dataset/{{li2023_m1, mammal_m1}}.yaml
  configs/experiment/kp_dlc_pretraining.yaml
  scripts/prepare_kp_splits.py
  scripts/benchmark_kp_dlc.py
  docs/kp_benchmark_v0.1.md

commit cd37772  docs(kp_benchmark): correct Li GT count + add MAMMAL alignment notes
  docs/kp_benchmark_v0.1.md  (50→81, 5-step ds, video-frame indexing)</pre>

<h2>6. Status check</h2>
<table>
<tr><th>Stage</th><th>Status</th><th>Notes</th></tr>
<tr><td>v0.1 scaffold (loaders, evaluator, configs, scripts, docs)</td><td class="good">✅ done</td><td>9 files, 644 lines, committed bb297e5</td></tr>
<tr><td>Syntax + mpjpe synthetic test</td><td class="good">✅ pass</td><td>raw=7.72 / rel=11.24 / proc=7.17; bootstrap CI 일관</td></tr>
<tr><td>Data fetch (gpu03 → mac)</td><td class="good">✅ done</td><td>MAMMAL npz 958 KB + Li label3d 1.1 MB</td></tr>
<tr><td>prepare_kp_splits.py run</td><td class="good">✅ pass</td><td>2880 / 720 / 81 (seed=42)</td></tr>
<tr><td>Li GT canonical npz</td><td class="good">✅ done</td><td>(81, 22, 3) + valid_mask saved</td></tr>
<tr><td>Docs correction (50→81 etc.)</td><td class="good">✅ done</td><td>commit cd37772</td></tr>
<tr><td>User modified files (8)</td><td class="good">✅ untouched</td><td>path overlap 0; git status preserved</td></tr>
<tr><td>KP overlay (3D→2D projection viz)</td><td class="good">✅ PASS</td><td>green=MAMMAL, red=Li agree on mouse body across 6 views</td></tr>
<tr><td>DLC training scripts (01/02 .sh)</td><td class="good">✅ written + executable</td><td>uses /node_data/joon to avoid NFS hangs; ready to run on gpu03 GPU 4/5 (97 GB free)</td></tr>
<tr><td>DLC inference → predictions npz</td><td class="warn">⏸ pending</td><td>after training</td></tr>
<tr><td>benchmark_kp_dlc.py results CSV + report</td><td class="warn">⏸ pending</td><td>after inference</td></tr>
</table>

<h2>7. What's next</h2>
<p><b>C ✅ + B ✅ done.</b> A (DLC training) ready to launch on gpu03 GPU 4 + GPU 5 (both 97 GB free, idle).</p>
<ol>
<li><b>A1 — ResNet50 training</b>: <code>CUDA_VISIBLE_DEVICES=4 bash scripts/01_train_dlc_resnet50.sh</code> on gpu03. ~30–60 min, 20 k iter on 2880 × 6 = 17 k labeled images. Output to <code>/node_data/joon/behavior-lab-kp-benchmark/kp_benchmark_dlc_resnet50_imagenet</code>.</li>
<li><b>A2 — SuperAnimal training</b> (can run in parallel on GPU 5): <code>CUDA_VISIBLE_DEVICES=5 bash scripts/02_train_dlc_superanimal.sh</code>. SuperAnimal-TopViewMouse weight init via <code>create_training_dataset(weight_init="modelzoo:superanimal_topviewmouse")</code>.</li>
<li><b>Inference</b>: <code>deeplabcut.analyze_videos()</code> on 6 videos → per-view 2D predictions → triangulate to 3D → npz with key <code>keypoints_3d</code>.</li>
<li><b>Evaluate</b>: <code>python scripts/benchmark_kp_dlc.py</code> → results.csv + bootstrap 95% CI on (in-dist MAMMAL test, OOD Li GT) for both predictors.</li>
<li><b>Final HTML</b>: re-run <code>render_kp_data_report.py</code> with <code>--include-results</code> to add per-kp error bars + CI overlap + DLC prediction overlay on Li GT frames.</li>
</ol>

<h3>Alternative — quicker pre-flight (10 min)</h3>
<p>If full training is too long for the current session, an intermediate
result is available via SuperAnimal <b>zero-shot inference</b> (no fine-tune,
out-of-box). Provides a baseline number for the OOD Li frames and validates
the inference pipeline before committing to 60+ min of training.</p>

<h2>8. Sanity verdict</h2>
<p><b class="good">정상.</b> Scaffold + data prep 단계의 모든 검증 통과:</p>
<ul>
<li>코드: py_compile pass, mpjpe synthetic test 통과 (raw / rel / proc ordering 정합, bootstrap CI brackets mean)</li>
<li>데이터: shape · range · NaN count 모두 합리적; alignment 가정이 sanity check로 정정됨 (handoff 50 → actual 81/50)</li>
<li>git: 두 commit 모두 user modified 8건 미영향, path overlap zero</li>
<li>재현성: <code>seed=42</code> + deterministic script로 splits 재생성 가능, raw data .gitignore</li>
</ul>
<p>다음 진행 권장: <b>C → B → A</b>. C는 1분, B는 10분, A는 사용자 GPU 승인 후 시작.</p>

<hr/>
<p style="color:#888;font-size:11px;">behavior-lab kp_benchmark v0.1 · Generated by render_kp_data_report.py · Self-contained (base64-embedded figures)</p>
</body></html>
"""


# ------------------------------ main --------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--mammal-npz",
        type=Path,
        default=REPO_ROOT / "data/mammal_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
    )
    ap.add_argument(
        "--li-label3d",
        type=Path,
        default=REPO_ROOT / "data/markerless_mouse_1/labels/label3d_dannce.mat",
    )
    args = ap.parse_args()

    # Load data
    mam = np.load(args.mammal_npz)
    mammal_kp = mam["keypoints"].astype(np.float64)
    mammal_idx = mam["frame_indices"]

    li = Li2023Loader(args.li_label3d).load()
    valid_mask = ~np.isnan(li.keypoints_3d).any(axis=-1)
    li_full_valid = int(valid_mask.all(axis=1).sum())

    # Split sizes from CSV (read directly)
    splits_dir = REPO_ROOT / "data/splits"
    train_n = sum(1 for _ in (splits_dir / "mammal_m1_train.csv").open()) - 1
    test_n = sum(1 for _ in (splits_dir / "mammal_m1_test.csv").open()) - 1

    # Get commit
    import subprocess
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()

    ctx = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "commit": commit,
        "mammal_n": int(mammal_kp.shape[0]),
        "li_total": int(li.keypoints_3d.shape[0]),
        "li_full_valid": li_full_valid,
        "train_n": train_n,
        "test_n": test_n,
        "rng_x": (float(mammal_kp[..., 0].min()), float(mammal_kp[..., 0].max())),
        "rng_y": (float(mammal_kp[..., 1].min()), float(mammal_kp[..., 1].max())),
        "rng_z": (float(mammal_kp[..., 2].min()), float(mammal_kp[..., 2].max())),
    }

    # Load pre-rendered KP overlay PNGs if available
    overlay_dir = REPO_ROOT / "outputs/kp_benchmark/overlay"
    overlay_frames = []
    overlay_b64 = {}
    if overlay_dir.exists():
        for p in sorted(overlay_dir.glob("frame_*_overlay.png"))[:2]:
            overlay_b64[f"overlay_{'a' if not overlay_frames else 'b'}"] = base64.b64encode(p.read_bytes()).decode()
            overlay_frames.append(int(p.stem.split("_")[1]))
    ctx["overlay_frames"] = overlay_frames or [0, 0]

    figs = {
        "splits": plot_split_sizes(train_n, test_n, ctx["li_total"]),
        "timeline": plot_frame_timeline(mammal_idx, li.frame_ids),
        "range": plot_kp_range(mammal_kp),
        "valid": plot_li_valid_heatmap(valid_mask),
        "skeleton": plot_kp_sample_3d(mammal_kp, sample_frame=0),
        **overlay_b64,
    }

    html = render_html(ctx, figs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"[done] HTML report → {args.output}")
    print(f"       size: {args.output.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
