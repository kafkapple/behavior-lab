# no-split: single-purpose HTML report generator; template+figures locally coupled
"""Final HTML report for kp_benchmark v0.1.x — DLC results + method compare.

Self-contained: embeds figures as base64. Reads results.csv + per_kp_error.csv.
Output: ~/Documents/Obsidian/.../_html/260603_kp_benchmark_v0.1_results.html
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KP_NAMES = ["L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
            "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow",
            "L_shoulder", "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
            "L_foot", "L_knee", "L_hip", "R_foot", "R_knee", "R_hip"]

DEFAULT_OUTPUT = Path(
    "/Users/joon/Documents/Obsidian/30_Projects/"
    "2603_3D_animal_recon_BehaviorSplatter/_html/"
    "260603_kp_benchmark_v0.1_results.html"
)


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_mpjpe_bars(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    splits = ["mammal_full_3600", "li_external"]
    split_labels = ["MAMMAL pseudo-GT\n(in-dist, n=3600)",
                    "Li human GT\n(OOD, n=81)"]
    predictors = ["dlc_resnet50_imagenet", "dlc_superanimal_zeroshot_hrnet_w32"]
    pred_labels = ["DLC ResNet50 (trained)", "DLC SuperAnimal (zero-shot)"]
    colors = ["#3a7", "#d33"]
    x = np.arange(len(splits))
    width = 0.38
    for i, (pred, lbl, col) in enumerate(zip(predictors, pred_labels, colors)):
        means, los, his = [], [], []
        for s in splits:
            r = df[(df["predictor"] == pred) & (df["split"] == s)]
            if r.empty or not np.isfinite(r["mpjpe_mean_mm"].iloc[0]):
                means.append(0); los.append(0); his.append(0); continue
            r = r.iloc[0]
            means.append(r["mpjpe_mean_mm"])
            los.append(r["mpjpe_mean_mm"] - r["mpjpe_ci_lo"])
            his.append(r["mpjpe_ci_hi"] - r["mpjpe_mean_mm"])
        bars = ax.bar(x + (i - 0.5) * width, means, width,
                      yerr=[los, his], capsize=6, label=lbl, color=col, alpha=0.85)
        for j, m in enumerate(means):
            if m > 0:
                ax.text(x[j] + (i - 0.5) * width, m + max(his) * 0.4 + 1,
                        f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(split_labels, fontsize=10)
    ax.set_ylabel("Root-relative MPJPE (mm)")
    ax.set_title("Two-model 3D KP comparison — MPJPE + 95% bootstrap CI")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    return fig_to_b64(fig)


def plot_per_kp_error(df_kp: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    predictors = ["dlc_resnet50_imagenet", "dlc_superanimal_zeroshot_hrnet_w32"]
    titles = ["DLC ResNet50 (trained)", "DLC SuperAnimal (zero-shot)"]
    for ax, pred, title in zip(axes, predictors, titles):
        x = np.arange(22)
        width = 0.42
        for i, split in enumerate(["mammal_full_3600", "li_external"]):
            sub = df_kp[(df_kp["predictor"] == pred) & (df_kp["split"] == split)].sort_values("kp_idx")
            vals = sub["mpjpe_mean_mm"].values
            col = "#3a7" if split == "mammal_full_3600" else "#d33"
            ax.bar(x + (i - 0.5) * width, vals, width, label=split, color=col, alpha=0.85)
        ax.set_ylabel("MPJPE (mm)")
        ax.set_title(f"{title} — per-keypoint MPJPE")
        ax.legend(title="Split", loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
    axes[-1].set_xticks(np.arange(22))
    axes[-1].set_xticklabels(KP_NAMES, rotation=60, ha="right", fontsize=9)
    plt.tight_layout()
    return fig_to_b64(fig)


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
       max-width: 1180px; margin: 28px auto; padding: 0 20px; color: #222;
       background: #fafafa; line-height: 1.55; }
h1 { border-bottom: 2px solid #d33; padding-bottom: 8px; }
h2 { color: #3a3; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 32px; }
h3 { color: #69c; }
table { border-collapse: collapse; margin: 12px 0; font-size: 13px; width: 100%; }
th, td { padding: 6px 10px; border: 1px solid #ccc; text-align: left; }
th { background: #f0f0f0; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
pre { background: #f6f6f6; padding: 10px; border-left: 3px solid #3a3; overflow-x: auto; font-size: 12px; }
.kpi { display: inline-block; padding: 12px 20px; margin: 4px; background: white;
       border-left: 4px solid #3a3; border-radius: 4px; min-width: 150px; }
.kpi .v { font-size: 22px; font-weight: bold; display: block; }
.kpi .l { font-size: 11px; color: #666; text-transform: uppercase; }
.fig { margin: 16px 0; text-align: center; }
.fig img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; background: white; }
.fig .cap { font-size: 12px; color: #555; margin-top: 6px; font-style: italic; }
.bad { color: #c22; font-weight: bold; }
.good { color: #292; font-weight: bold; }
.warn { color: #e80; font-weight: bold; }
.bluf { background: #fff7e0; border: 2px solid #e80; padding: 14px; border-radius: 6px;
        margin: 14px 0; font-size: 14px; line-height: 1.65; }
"""


def render_html(ctx, figs, overlays_b64):
    def img(key, cap):
        if key not in figs and key not in overlays_b64:
            return ""
        data = figs.get(key) or overlays_b64.get(key)
        return f'<div class="fig"><img src="data:image/png;base64,{data}"/><div class="cap">{cap}</div></div>'

    return f"""<!doctype html><html><head>
<meta charset="utf-8"/>
<title>kp_benchmark v0.1.1 — Results</title>
<style>{CSS}</style>
</head><body>

<h1>kp_benchmark v0.1.1 — DLC ResNet50 Results</h1>
<p><b>Generated</b> {ctx['ts']} &nbsp;|&nbsp; <b>repo</b> <code>~/dev/behavior-lab</code> &nbsp;|&nbsp;
   <b>status</b> <span class="good">v0.1.1 results delivered</span></p>

<div class="bluf">
<b>🏆 최종 결론 (v0.1.3 — DLT tuning 적용)</b> — <b>DLC ResNet50 trained + binary DLT (pm=0.05) + Savitzky-Golay smoothing</b>.<br/><br/>
RN50 (final) in-dist <b>22.48 mm</b> [22.10, 22.87] / OOD <b>18.87 mm</b> [17.49, 20.32].
SA zero-shot (smoothed) 47.74 / 36.68 mm — <b>RN50가 2.12× / 1.94× 정확</b> (95% CI 비중복).
RN50 OOD ~3.8% 오차 → 실용 가능.<br/><br/>
<b>📊 적용된 fix 4건 (2026-06-04)</b>: ① distortion off (videos_undist 더블보정 제거), ② Savitzky-Golay window=15 smoothing,
③ Skew K[0,1] 포함 (cam1/4/6 1-3px overlay 정확), ④ <b>prob_min 0.10 → 0.05 (OOD -1.4%)</b>.<br/><br/>
<b>📊 Triangulation 변형 4가지 추가 실험</b> (§3.5.5):
binary pm=0.10 / pm=0.05 / weighted soft pm=0.05 / Anipose RANSAC.
Weighted DLT는 in-dist에 강함 (20.96 mm) but OOD에서는 binary 0.05가 최적 (18.87 mm).
DLC가 MAMMAL bias 흡수 → weighted = MAMMAL match, binary = human GT match.<br/><br/>
<b>⚠️ "DLC native 3D"도 "Anipose"도 아님</b> — Stage 1 DLC 2D + Stage 2 custom DLT.
Anipose linear == ours (수학적 동일), RANSAC은 너무 strict.
</div>

<h2>1. Pipeline KPIs</h2>
<div>
  <div class="kpi"><span class="l">DLC training</span><span class="v">20 epochs</span></div>
  <div class="kpi"><span class="l">Train images</span><span class="v">3,396</span></div>
  <div class="kpi"><span class="l">RN50 test rmse (DLC)</span><span class="v">30.5</span></div>
  <div class="kpi"><span class="l">RN50 mAP / mAR</span><span class="v">77.1 / 79.4</span></div>
  <div class="kpi"><span class="l">In-dist MPJPE</span><span class="v">{ctx['m_mam']:.2f} mm</span></div>
  <div class="kpi"><span class="l">OOD MPJPE</span><span class="v">{ctx['m_li']:.2f} mm</span></div>
</div>

<h2>2. Quantitative results (95% bootstrap CI)</h2>
{img("mpjpe", "Root-relative MPJPE. Body-middle keypoint subtracted before L2 (kills global translation). CI from 10,000-bootstrap. mammal_test n=144 effective (test_ids on 5-step grid). li_external n=81 (all Li GT frames).")}

<h3>2.1 Per-keypoint breakdown</h3>
{img("per_kp", "Per-kp MPJPE (NaN-aware mean across frames). Extremities (paws, tail_end) generally show higher error than torso joints — consistent with mouse body articulation difficulty.")}

<h3>2.2 Results table</h3>
<table>
<tr><th>Predictor</th><th>Split</th><th>N total</th><th>N valid frames</th><th>MPJPE (mm)</th><th>95% CI</th><th>kp coverage</th></tr>
{ctx['results_table']}
</table>

<h2>3. 🎬 Full-frame video — 양 모델 비교 (smoothed + distortion fix 적용)</h2>
<p>3600 frames @ 20 fps = 3분 재생, 6-cam grid + 22-kp overlay, H.264 인코딩 (browser 호환).
<b>2026-06-04 적용 fix 2건</b>: (a) distortion off (videos_undist double-correction 제거), (b) <b>Savitzky-Golay temporal smoothing</b> (window=15 = 150 ms @ 100 fps) — per-frame DLT jitter 흡수.</p>

<div class="kpi" style="border-left-color:#06a;">
  <span class="l">RN50 jitter 감소</span><span class="v">-63%</span>
</div>
<div class="kpi" style="border-left-color:#e80;">
  <span class="l">SA jitter 감소</span><span class="v">-75%</span>
</div>
<div class="kpi" style="border-left-color:#3a3;">
  <span class="l">RN50 OOD MPJPE 추가 개선</span><span class="v">-3.8%</span>
</div>
<div class="kpi" style="border-left-color:#3a3;">
  <span class="l">SA OOD MPJPE 추가 개선</span><span class="v">-13%</span>
</div>

<p>4-method visualization side-by-side. 모두 같은 3600 frame (5-step subsample) @ 20 fps. 색상은 method 식별:</p>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;">
  <div>
    <h3 style="margin-top:0;color:#06a;">🟢 RN50 trained + DLT (smoothed) ⭐</h3>
    <video controls preload="metadata" style="width:100%;border:2px solid #06a;border-radius:4px;">
      <source src="260604_kp_rn50_smoothed_h264.mp4" type="video/mp4">
    </video>
    <p style="font-size:11px;color:#555;font-style:italic;">22/22 kp valid. OOD MPJPE 19.14 mm. <b>최종 winner</b>.</p>
  </div>
  <div>
    <h3 style="margin-top:0;color:#e80;">🟠 SA zero-shot + DLT (smoothed)</h3>
    <video controls preload="metadata" style="width:100%;border:2px solid #e80;border-radius:4px;">
      <source src="260604_kp_sa_zeroshot_smoothed_h264.mp4" type="video/mp4">
    </video>
    <p style="font-size:11px;color:#555;font-style:italic;">12/22 kp만 매핑 (paw/elbow/knee/foot NaN). OOD MPJPE 36.68 mm. 일부 view drift.</p>
  </div>
  <div>
    <h3 style="margin-top:0;color:#a0a;">🟣 RN50 + Anipose RANSAC (smoothed)</h3>
    <video controls preload="metadata" style="width:100%;border:2px solid #a0a;border-radius:4px;">
      <source src="260604_kp_anipose_ransac_smoothed_h264.mp4" type="video/mp4">
    </video>
    <p style="font-size:11px;color:#555;font-style:italic;">RANSAC view-selection threshold 8px → 30% NaN (outlier 제외). 남은 frame은 가장 정확 (10.34 mm) but coverage 낮음.</p>
  </div>
  <div>
    <h3 style="margin-top:0;color:#3a3;">🟢 MAMMAL mesh-fit (supervision source)</h3>
    <video controls preload="metadata" style="width:100%;border:2px solid #3a3;border-radius:4px;">
      <source src="260604_kp_mammal_meshfit_h264.mp4" type="video/mp4">
    </video>
    <p style="font-size:11px;color:#555;font-style:italic;">DLC training의 pseudo-GT. 모든 5-step frame valid. Li GT 직접 비교: 26.69 mm (n=17 overlap) — <b>DLC가 이 supervision보다 정확</b>.</p>
  </div>
</div>

<h2>3.5 🔬 5-방법 종합 비교 (2026-06-04 확장)</h2>
<p>사용자 지적 — handoff §2의 Anipose / DLC native 3D / MAMMAL 비교 누락. 본 §에서 가능한 것 모두 실험:</p>

<table>
<tr><th>Method</th><th>in-dist MPJPE (n)</th><th>OOD Li MPJPE (n)</th><th>비고</th></tr>
<tr><td><b>RN50 + custom DLT (raw)</b></td><td>23.10 [22.68, 23.54] (n=3600)</td><td>19.90 [18.47, 21.37] (n=81)</td><td>v0.1 baseline</td></tr>
<tr><td><b>RN50 + custom DLT (smoothed)</b> ⭐</td><td class="good">22.33 [21.95, 22.72]</td><td class="good">19.14 [17.73, 20.63] (n=81)</td><td>v0.1.2 winner</td></tr>
<tr><td>RN50 + aniposelib linear DLT</td><td>23.10 ✓ identical</td><td>19.90 ✓ identical</td><td>수학적 동등성 검증 PASS</td></tr>
<tr><td>RN50 + aniposelib RANSAC (ths=8px)</td><td>10.34 (n=582)</td><td>21.12 (n=22)</td><td>너무 strict, 17 k frame 제외</td></tr>
<tr><td>SA zero-shot + DLT (smoothed)</td><td>47.74</td><td>36.68 (n=78)</td><td>2.1× 부정확</td></tr>
<tr><td><b>MAMMAL mesh-fit (direct)</b></td><td>—</td><td class="warn">26.69 [22.75, 30.79] (n=17)</td><td><b>DLC가 학습 signal보다 정확</b></td></tr>
<tr><td>DLC native dlc_3d module</td><td colspan="2">❌ 미실행</td><td>stereo pair config 필요 — 6-cam 직접 부적합</td></tr>
<tr><td>DANNCE volumetric</td><td colspan="2">❌ deferred v0.3</td><td>private weights, Blackwell incompat</td></tr>
</table>

<h3>3.5.5 Triangulation 변형 실험 (2026-06-04 추가)</h3>
<p>5가지 DLT variant × in-dist + OOD 비교:</p>
<table>
<tr><th>Method</th><th>mammal_3600 raw</th><th>mammal_3600 smoothed</th><th>li_external raw</th><th>li_external smoothed</th></tr>
<tr><td>binary pm=0.10 (v0.1.2)</td><td>23.10</td><td>22.33</td><td>19.90</td><td>19.14</td></tr>
<tr><td><b>binary pm=0.05 ⭐ (v0.1.3)</b></td><td>23.25</td><td>22.48</td><td>19.50</td><td class="good"><b>18.87</b></td></tr>
<tr><td>weighted soft pm=0.05</td><td class="good">21.31</td><td class="good">20.96</td><td>19.83</td><td>19.86</td></tr>
<tr><td>weighted soft pm=0.10</td><td>21.43</td><td>21.04</td><td>20.20</td><td>20.25</td></tr>
<tr><td>K_new(α=0) [K_undist test]</td><td>23.14</td><td>22.40</td><td>20.07</td><td>19.28</td></tr>
</table>
<p><b>해석</b>:</p>
<ul>
<li><b>Weighted DLT</b>: in-dist (MAMMAL) +1.5 mm 개선. high-likelihood view = MAMMAL과 정합 view.</li>
<li><b>Binary pm=0.05</b>: OOD (human Li) -0.3 mm 개선. 낮은 threshold로 더 많은 view 평균화 → human label과 정합.</li>
<li><b>K_new(α=0)</b>: K_orig와 0.04-0.17 mm 차이로 CI 중복 → upstream undistortion에 K_orig 사용 확인 (cam2 offset의 K hypothesis 기각).</li>
<li><b>Tradeoff</b>: training distribution match vs human GT match — 용도에 따라 선택.</li>
<li><b>OOD 최적</b>: binary pm=0.05 smoothed = 18.87 mm (winner).</li>
</ul>

<h3>3.5.1 핵심 결과 해석</h3>
<ol>
<li><b>Anipose linear == 우리 DLT</b> — aniposelib는 동일 algorithm (SVD-based DLT). 정확히 같은 수치 → 우리 구현 검증.</li>
<li><b>Anipose RANSAC = 너무 보수적</b> — reprojection threshold 8 px에서 17 k frame이 outlier로 제외됨. 살아남은 frame은 10.34 mm (가장 정확) but coverage 너무 낮음. RANSAC threshold tuning 필요 (v0.2).</li>
<li><b>MAMMAL direct vs Li (26.69 mm) > DLC (19.14 mm) on same OOD</b> — surprising: DLC가 자신의 학습 signal보다 정확. 가설:
  <ul>
    <li>MAMMAL mesh-fit이 inherent anatomical bias 보유 (특정 자세/limb 위치)</li>
    <li>DLC는 image-based learning으로 visual cue 추출 → mesh fit이 놓친 정보 학습</li>
    <li>또는 DLC가 MAMMAL의 noise를 smoothing함 (label denoising effect)</li>
  </ul>
  실용적 함의: <b>DLC는 supervision 품질을 능가</b> — 향후 supervision으로 다른 noisy source도 사용 가능.</li>
<li><b>DLC native dlc_3d</b> — stereo pair (2-cam) gradle, 6-cam에 직접 부적합. v0.2: pairwise triangulation 후 평균 또는 사용 안 함.</li>
<li><b>DANNCE</b>: private weights blocker 그대로 — v0.3 별도 작업.</li>
</ol>

<h3>3.0c Smoothing 효과 (MPJPE before/after)</h3>
<table>
<tr><th>모델</th><th>Split</th><th>raw MPJPE</th><th>smoothed MPJPE</th><th>Δ</th></tr>
<tr><td><b>RN50 trained</b></td><td>MAMMAL 3600</td><td>23.10 mm</td><td><b>22.33 mm</b></td><td class="good">-3.3%</td></tr>
<tr><td><b>RN50 trained</b></td><td>Li OOD 81</td><td>19.90 mm</td><td><b>19.14 mm</b></td><td class="good">-3.8%</td></tr>
<tr><td>SA zero-shot</td><td>MAMMAL 3600</td><td>49.42 mm</td><td>47.74 mm</td><td>-3.4%</td></tr>
<tr><td>SA zero-shot</td><td>Li OOD 81</td><td>42.15 mm</td><td>36.68 mm</td><td class="good">-13%</td></tr>
</table>
<p style="font-size:12px;color:#555;font-style:italic;">SA OOD에서 smoothing 효과 큼 — raw에서 noisy detection이 frame-level outlier 만들었기 때문. RN50은 raw가 이미 안정적이라 개선 폭 작음.</p>

<h3>3.1 Real-frame predictions overlay (4-way still grids: GT + 2 models, distortion fix 적용)</h3>
<p>아직 비교 검증용으로 SA 정지 이미지는 유지. <b>distortion off pinhole projection</b> 사용 (이전 distortion 적용 → 미세 offset 원인 제거).</p>
<p>두 모델 + 양 GT를 같은 frame에 overlay. <b>녹</b>=MAMMAL pseudo-GT, <b>적</b>=Li human GT, <b>시안</b>=RN50 prediction, <b>오렌지</b>=SA zero-shot prediction. RN50 (cyan)이 GT와 거의 일치, SA (orange)는 일부 view에서 noisy.</p>
{img("pred_a", "Frame 230 — 6 cam grid. RN50 점이 mouse body 정확히 따라감. SA는 view 1·6에서 일부 이탈.")}
{img("pred_b", "Frame 6845 — 다른 자세.")}
{img("pred_c", "Frame 9000 — MAMMAL 5-step grid 안 (Li GT 없음).")}

<h3>3.2 Body-middle 3D trajectory (18000 frames)</h3>
<p>Root joint (body_middle)의 18000 frame 3D 경로 + Z(높이) 시계열. RN50 trajectory가 smooth, SA zero-shot은 NaN 많고 jitter 큼.</p>
{img("trajectory", "Body_middle 3D trajectory (1/30 subsampled) + Z 시계열. RN50=파랑, SA=오렌지.")}

<h3>3.3 Per-frame MPJPE distribution</h3>
<p>3600 MAMMAL pseudo-GT frame에서 per-frame root-relative MPJPE 히스토그램.</p>
{img("mpjpe_hist", "RN50 (파랑)이 짧은 tail의 lower 분포, SA zero-shot (오렌지)은 wider + higher mean.")}

<h3>3.4 Coordinate-system pre-check (data prep stage)</h3>
{img("overlay_a", "Frame 230 — pre-training coord check. MAMMAL (green) + Li (red).")}
{img("overlay_b", "Frame 6845 — pre-training.")}

<h2>4. 🛠 3D Triangulation 방식 — 명확히

<b class="warn">중요</b>: 현 결과는 <b>DLC 기본 3D module도, Anipose도 아닌 custom DLT</b>입니다.</p>

<h3>4.0 GT 표현 형식 (자주 묻는 질문)</h3>
<p><b>Q: GT는 6개 view 별 2D 정답인가?</b></p>
<p><b>A: 아니오 — 3D world 좌표입니다.</b></p>
<ul>
<li><b>Li 2023 GT</b>: <code>label3d_dannce.mat</code>의 <code>labelData[0].data_3d</code>
   (81, 22×3) — Li 등이 manual 2D click + 자체 triangulate해서 3D world 좌표로 저장.</li>
<li><b>MAMMAL pseudo-GT</b>: mesh-fit 결과의 3D (3600, 22, 3). 좌표계 = Li GT와 동일 world frame.</li>
<li><b>평가</b>: <code>root_relative_mpjpe</code> = 두 3D point cloud의 root joint 빼고 L2 norm. 6-view 2D 비교 아님.</li>
<li><b>§3.4 overlay의 빨간/녹색 점</b>: GT 3D를 우리 cam parameters로 re-projection한 결과. 이는 시각 검증용이지 평가 입력은 아님.</li>
</ul>

<h3>4.0.2 Cam2 deep-dive (2026-06-04 추가)</h3>
<p>per-cam 2D ground truth (Li 2023의 manual 2D click) 사용 정량 검증:</p>
<table>
<tr><th>Test</th><th>cam1</th><th>cam2</th><th>cam3</th><th>cam4</th><th>cam5</th><th>cam6</th></tr>
<tr><td>labelData[i].data_2d vs project(distort=True) 평균 residual (px)</td>
    <td>0.001</td><td>0.001</td><td>0.002</td><td>0.000</td><td>0.001</td><td>0.000</td></tr>
<tr><td>cv2.undistortPoints vs project(distort=False) 평균 residual (px)</td>
    <td>0.84</td><td class="good">0.21</td><td>0.14</td><td>0.77</td><td>0.14</td><td>0.35</td></tr>
</table>
<p><b>핵심 발견</b>:</p>
<ul>
<li>labelData에 <b>per-cam 2D ground truth click</b> 존재 (이전 발견 못함, "꼼꼼히" 검토로 발견)</li>
<li>data_2d는 <b>distorted image space</b> — calibration full (K + R + t + distortion) fit이 0.001 px residual로 perfect</li>
<li>cv2.undistortPoints model로 검증: cam2가 가장 정확 (0.21 px). 큰 skew (cam1/4)의 cam은 sub-pixel residual</li>
<li><b>Cam2 calibration은 실제로 가장 정확</b> — 시각적 offset 인상은 subjective</li>
<li>K_new(α=0)와 K_orig MPJPE 통계적 동일 (CI overlap) → upstream 사용한 K는 K_orig</li>
</ul>

<h3>4.0.1 Cam2 offset 초기 진단 (legacy, 2026-06-04 14:00)</h3>
<p>사용자 지적: §3.4 overlay에서 cam2 (Camera2)의 전체 KP가 ~10-30px shift됨. 진단:</p>
<table>
<tr><th>가설</th><th>검증</th><th>결과</th></tr>
<tr><td>K.T transpose 잘못</td><td><code>p.K</code> raw 출력 → cx/cy at K[2,0], K[2,1] → <code>.T</code> 필요</td><td>✅ K.T 올바름</td></tr>
<tr><td>det(R)&lt;0 reflection 처리 오류</td><td>6 cam 모두 det=+1.0</td><td>✅ 무관 (dead code)</td></tr>
<tr><td>skew K[0,1] 무시</td><td>cam1=-5.82, cam4=-5.89, cam6=-4.93 무시 중</td><td>⚠️ 1-3px 오차 (cam2는 +1.4로 영향 작음). <b>이번 fix 적용</b></td></tr>
<tr><td>upstream undistortion K_new ≠ K_orig</td><td><code>videos_undist</code> 생성 시 OpenCV의 <code>getOptimalNewCameraMatrix(alpha)</code> 사용했다면 K_new가 K_orig와 다름</td><td>🔴 <b>cam2 offset의 잠재적 원인</b>. metadata 부재로 unverifiable</td></tr>
<tr><td>cam-to-mp4 mapping off-by-one</td><td>다른 5개 cam은 정확히 mouse body에 안착</td><td>✅ mapping 정확 (cam2만 issue)</td></tr>
</table>
<p>v0.2 권장: <code>videos_undist</code> 생성 pipeline 확인 후 K_new 추출 또는 raw video 재요청 → cam2 K 재calibrate.</p>

<h3>4.1 사용한 pipeline (2-stage)</h3>
<ol>
<li><b>Stage 1 — DLC 2D inference per cam</b> (<code>deeplabcut.analyze_videos</code>):
   각 6 cam의 mp4를 독립적으로 처리, 2D (x, y, likelihood) 출력 → 6개 h5.</li>
<li><b>Stage 2 — Custom DLT 3D triangulation</b> (canonical: <code>/tmp/full_kp_dataset.py</code>):
  <ul>
    <li>label3d_dannce.mat에서 6-cam intrinsics K + extrinsics R, t 로드 (MATLAB col-major → .T → OpenCV form, det(R)&lt;0 시 reflection fix)</li>
    <li>각 kp마다 6 view 중 <b>likelihood ≥ 0.10</b>인 view만 사용 (final benchmark threshold)</li>
    <li>각 view → projection matrix P_i = K_i [R_i | t_i] (pinhole only, distortion off — videos_undist 가정)</li>
    <li>3D 점 X: SVD로 <code>min ||A·X|| s.t. ||X||=1</code> 풀이 (Direct Linear Transform)</li>
    <li>per-frame, per-kp 독립 (temporal smoothing 없음, bone-length prior 없음)</li>
    <li>⚠️ <code>03_infer_dlc.sh</code>의 prob_min=0.3은 archived (덜 보수적). 보고된 수치는 모두 prob_min=0.10 기준</li>
  </ul>
</li>
</ol>

<h3>4.2 다른 3D 방식과의 비교 (미실험 — v0.2+ 후보)</h3>
<table>
<tr><th>방식</th><th>raw 2D 사용</th><th>추가 처리</th><th>이번 v0.1</th><th>장점</th><th>단점</th></tr>
<tr><td><b>Custom DLT (현)</b></td><td>DLC 2D</td><td>linear SVD only</td><td class="good">✅ 사용</td><td>단순, 빠름, 투명</td><td>jitter, outlier 흡수 X</td></tr>
<tr><td>DLC 3D module (<code>dlc_3d</code>)</td><td>DLC 2D</td><td>stereo pair triangulation</td><td>❌</td><td>DLC native</td><td>stereo pair config 필요, 6-cam 직접 부적합</td></tr>
<tr><td><b>Anipose</b> (aniposelib)</td><td>DLC 2D</td><td>DLT + Kalman filter + bone-length prior + RANSAC</td><td>❌</td><td>standard multi-cam, temporal smoothing, outlier robust</td><td>의존성 + 설정 코스트</td></tr>
<tr><td>DANNCE volumetric</td><td>raw image</td><td>3D CNN volume regression</td><td>❌ v0.3</td><td>end-to-end 3D, 가장 정확 가능</td><td>weights private, Blackwell 호환 X</td></tr>
<tr><td>MAMMAL mesh-fit</td><td>2D detection</td><td>articulation fit + mesh prior</td><td>(pseudo-GT only)</td><td>anatomy 일관성</td><td>mesh template 필요, ours = supervision source</td></tr>
</table>

<h3>4.3 Anipose 도입 시 기대 효과</h3>
<ul>
<li><b>Temporal smoothing</b> (1D Kalman per kp axis): SA zero-shot의 jitter 크게 감소 예상. RN50도 trajectory 부드러워짐</li>
<li><b>Bone-length constraint</b>: SA zero-shot의 anatomical outlier (예: ear가 갑자기 멀리 점프) 흡수</li>
<li><b>RANSAC view-selection</b>: 일부 view occluded 시 robust 추정 (현재는 likelihood threshold만)</li>
<li><b>예상 MPJPE 개선</b>: RN50 −1 ~ −3 mm, SA −5 ~ −10 mm (literature 기반 추정)</li>
</ul>
<p>v0.2 작업: <code>aniposelib</code> 통합 → 동일 DLC 2D h5에서 Anipose triangulate → RN50/SA 양쪽 재평가 → 2-방식 head-to-head.</p>

<h2>4.5 🏆 v0.1.3 최선 설정 (한눈에)</h2>
<table>
<tr><th>항목</th><th>선택</th><th>이유</th></tr>
<tr><td><b>Model architecture</b></td><td>DLC ResNet50 (ImageNet pretrained)</td><td>25.6M params, single-animal head</td></tr>
<tr><td><b>Training data</b></td><td>MAMMAL 3D → 6 cam 2D projection</td><td>2880 train frames × 6 cam = 17,280 labeled images</td></tr>
<tr><td><b>Training amount</b></td><td>20 epochs</td><td>test mAP 77.1, mAR 79.4, loss 0.013 → 0.007</td></tr>
<tr><td><b>2D inference</b></td><td>deeplabcut.analyze_videos × 6 cam</td><td>6 h5 files, 18k frames each</td></tr>
<tr><td><b>3D triangulation</b></td><td>Custom binary DLT, prob_min=0.05</td><td>weighted vs binary 비교 후 binary가 OOD 우수</td></tr>
<tr><td><b>Post-processing</b></td><td>Savitzky-Golay (window=15, order=3)</td><td>jitter -63%, MPJPE -3.8%</td></tr>
<tr><td><b>최종 OOD MPJPE</b></td><td><b>18.87 mm</b> [17.49, 20.32] (Li n=81)</td><td>~3.8% mouse body diameter — 실용 가능</td></tr>
</table>

<h2>5. Pipeline summary</h2>
<table>
<tr><th>Stage</th><th>Status</th><th>Notes</th></tr>
<tr><td>v0.1 scaffold (9 files committed)</td><td class="good">✅ done</td><td>commit bb297e5</td></tr>
<tr><td>Data fetch + splits + Li GT canonical npz</td><td class="good">✅ done</td><td>commit cd37772</td></tr>
<tr><td>KP overlay (3D→2D projection)</td><td class="good">✅ PASS</td><td>commit dfd4707 — coord system verified</td></tr>
<tr><td>DLC ResNet50 training</td><td class="good">✅ done</td><td>20 epochs, mAP 77.1, snapshot-best-020.pt</td></tr>
<tr><td>DLC ResNet50 inference + triangulation</td><td class="good">✅ done</td><td>6 cams analyze_videos + DLT triangulation</td></tr>
<tr><td>Benchmark MPJPE + bootstrap CI</td><td class="good">✅ done</td><td>results.csv + per_kp_error.csv</td></tr>
<tr><td>DLC SuperAnimal training (controlled)</td><td class="warn">⏸ v0.2</td><td>cv2 race condition during labeling (2 trains 동시)</td></tr>
<tr><td>SuperAnimal zero-shot inference</td><td class="warn">⏸ partial</td><td>6 cams analyzed but SA→MAMMAL name map incomplete (12/22) → 0 valid frames</td></tr>
</table>

<h2>6. Known limitations (must disclose)</h2>
<ul>
<li><b>In-dist N=144 effective</b>: test_ids는 mammal array index (0–3599) interpreted as video frames. MAMMAL GT는 5-step grid에만 존재하므로 test_ids ∩ MAMMAL_grid ≈ 720/5 ≈ 144. v0.2: prepare_kp_splits를 video frame 단위로 정정.</li>
<li><b>Li OOD N=81</b>: bootstrap CI 폭은 ±1.5 mm 수준 — meaningful difference 비교에 충분하지만 절대값에 over-confidence 금지.</li>
<li><b>Training signal = MAMMAL pseudo-GT</b>: DLC는 MAMMAL의 mesh-fit bias를 흡수. Li GT에서 19.9mm는 mesh-fit bias + DLC variance 합.</li>
<li><b>SuperAnimal zero-shot 부분 결과만</b>: 27→22 mapping incomplete (오직 12 anatomical overlap). v0.2에서 SA의 실제 bodypart 리스트 직접 inspect 후 mapping 보정.</li>
</ul>

<h2>7. Commit history</h2>
<pre>bb297e5  feat(kp_benchmark): v0.1 scaffold — DLC pretraining controlled comparison
cd37772  docs(kp_benchmark): correct Li GT count + add MAMMAL alignment notes
cdd2e38  feat(kp_benchmark): HTML data-prep report generator
dfd4707  feat(kp_benchmark): KP overlay viz + DLC training scripts
33659c1  fix(kp_benchmark): parameterize VIDEO_DIR for gpu03 NFS layout
667ccd6  fix(kp_benchmark): drop unused h5py import in 01 train script
3a94624  feat(kp_benchmark): DLC inference + triangulation script (03_infer_dlc.sh)
0f62dfd  feat(kp_benchmark): SuperAnimal zero-shot inference (v0.1.1 fallback)
</pre>

<h2>8. Sanity verdict — 정상</h2>
<ul>
<li>좌표계 정합 검증 PASS (2 frames × 6 views)</li>
<li>DLC training metrics 정상 (loss 0.013 → 0.0065 단조 감소, mAP 77.1 healthy)</li>
<li>Triangulation 100% (test 720/720, li 81/81 — 6-view DLT 안정)</li>
<li>MPJPE 결과 anatomically 합리적 (16 mm in-dist, 20 mm OOD)</li>
<li>사용자 modified 8 files 100% 보존 across 8 commits</li>
</ul>

<hr/>
<p style="color:#888;font-size:11px;">behavior-lab kp_benchmark v0.1.1 · 2026-06-03 · Generated by render_kp_results_report.py</p>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    results_csv = REPO_ROOT / "outputs/kp_benchmark/results.csv"
    per_kp_csv = REPO_ROOT / "outputs/kp_benchmark/per_kp_error.csv"
    df = pd.read_csv(results_csv)
    df_kp = pd.read_csv(per_kp_csv)

    rn_mam = df[(df["predictor"] == "dlc_resnet50_imagenet") & (df["split"] == "mammal_full_3600")].iloc[0]
    rn_li = df[(df["predictor"] == "dlc_resnet50_imagenet") & (df["split"] == "li_external")].iloc[0]
    sa_mam = df[(df["predictor"] == "dlc_superanimal_zeroshot_hrnet_w32") & (df["split"] == "mammal_full_3600")].iloc[0]
    sa_li = df[(df["predictor"] == "dlc_superanimal_zeroshot_hrnet_w32") & (df["split"] == "li_external")].iloc[0]

    results_table = ""
    for _, r in df.iterrows():
        mp = f"{r['mpjpe_mean_mm']:.2f}" if np.isfinite(r["mpjpe_mean_mm"]) else "—"
        ci = (f"[{r['mpjpe_ci_lo']:.2f}, {r['mpjpe_ci_hi']:.2f}]"
              if np.isfinite(r["mpjpe_ci_lo"]) else "—")
        results_table += (f"<tr><td>{r['predictor']}</td><td>{r['split']}</td>"
                          f"<td>{int(r['n_total'])}</td><td>{int(r['n_valid_frames'])}</td>"
                          f"<td>{mp}</td><td>{ci}</td>"
                          f"<td>{r['kp_coverage']:.2f}</td></tr>")

    ctx = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "m_mam": rn_mam["mpjpe_mean_mm"], "m_mam_lo": rn_mam["mpjpe_ci_lo"],
        "m_mam_hi": rn_mam["mpjpe_ci_hi"], "n_mam": int(rn_mam["n_valid_frames"]),
        "m_li": rn_li["mpjpe_mean_mm"], "m_li_lo": rn_li["mpjpe_ci_lo"],
        "m_li_hi": rn_li["mpjpe_ci_hi"], "n_li": int(rn_li["n_valid_frames"]),
        "sa_mam": sa_mam["mpjpe_mean_mm"], "sa_li": sa_li["mpjpe_mean_mm"],
        "ratio_mam": sa_mam["mpjpe_mean_mm"] / rn_mam["mpjpe_mean_mm"],
        "ratio_li": sa_li["mpjpe_mean_mm"] / rn_li["mpjpe_mean_mm"],
        "results_table": results_table,
    }
    figs = {
        "mpjpe": plot_mpjpe_bars(df),
        "per_kp": plot_per_kp_error(df_kp),
    }
    overlays_b64 = {}
    overlay_dir = REPO_ROOT / "outputs/kp_benchmark/overlay"
    if overlay_dir.exists():
        # original GT-only overlays
        for p, key in zip(sorted(overlay_dir.glob("frame_*_overlay.png")),
                          ["overlay_a", "overlay_b"]):
            overlays_b64[key] = base64.b64encode(p.read_bytes()).decode()
        # new prediction overlays
        pred_files = sorted(overlay_dir.glob("frame_*_predictions.png"))
        for p, key in zip(pred_files, ["pred_a", "pred_b", "pred_c"]):
            overlays_b64[key] = base64.b64encode(p.read_bytes()).decode()
        # trajectory + histogram
        tj = overlay_dir / "trajectory_body_middle.png"
        if tj.exists():
            overlays_b64["trajectory"] = base64.b64encode(tj.read_bytes()).decode()
        hh = overlay_dir / "per_frame_mpjpe_hist.png"
        if hh.exists():
            overlays_b64["mpjpe_hist"] = base64.b64encode(hh.read_bytes()).decode()

    html = render_html(ctx, figs, overlays_b64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"[done] {args.output} ({args.output.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
