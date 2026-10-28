"""Figures + one HTML page for m2_superanimal_2d_score.py output (any rows, any point set).

Every method is drawn the same way: one figure per picked frame, rows = methods, columns = cameras, the same crop
around the hand label in every panel, white ring = hand label, coloured cross = that method. Source pixels are not
changed; markers are drawn on top. Fig. M2 per-joint median px, Fig. M3 cumulative error on the common set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavior_lab.pose.html_common import page, zoom_img
from behavior_lab.pose.predictors.styles import LABEL_RING, legend_html, style

POINTS = {"ext8": ["nose", "L_ear", "R_ear", "tail_end", "L_paw_end", "R_paw_end", "L_foot", "R_foot"],
          "head4": ["nose", "L_ear", "R_ear", "tail_end"]}


def overlay(d: dict, rows: list[str], pts: list[str], frames_dir: Path, fr: int, out: Path, fig_id: str) -> None:
    names = list(d["names"])
    ix = [names.index(n) for n in pts]
    n_cam = d["gt2"].shape[0]
    fi = int(np.where(d["frames"] == fr)[0][0])
    fig, axes = plt.subplots(len(rows), n_cam, figsize=(2.4 * n_cam, 2.4 * len(rows)), squeeze=False)
    for c in range(n_cam):
        img = plt.imread(frames_dir / f"cam{c + 1}_f{fr:06d}.png")
        g = d["gt2"][c, fi, ix]
        ok = np.isfinite(g).all(1)
        x0, y0 = np.nanmin(g[ok], 0) - 60
        x1, y1 = np.nanmax(g[ok], 0) + 60
        for r, key in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xlim(max(x0, 0), min(x1, img.shape[1]))
            ax.set_ylim(min(y1, img.shape[0]), max(y0, 0))
            ax.scatter(*g.T, s=34, facecolors="none", edgecolors=LABEL_RING.color, linewidths=1.6)
            ax.scatter(*g.T, s=50, facecolors="none", edgecolors="black", linewidths=0.6)
            st = style(key)
            ax.scatter(*d[key][c, fi, ix].T, s=46, marker=st.marker, color=st.color, edgecolors="black", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"cam{c + 1}", fontsize=9)
            if c == 0:
                ax.set_ylabel(style(key).label, fontsize=7)
    fig.suptitle(f"Fig. {fig_id} · frame {fr} · white ○ hand label, × method ({', '.join(pts)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def per_joint(m: dict, rows: list[str], out: Path) -> None:
    joints = list(m["per_joint_median_px"]["base"])
    x = np.arange(len(joints))
    w = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, key in enumerate(rows):
        v = [m["per_joint_median_px"][key][j] or np.nan for j in joints]
        ax.bar(x + (i - (len(rows) - 1) / 2) * w, v, w, label=style(key).label, color=style(key).color)
    ax.set_xticks(x, joints, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("median error (px)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Fig. M2 · per-joint median 2D error, common frame·cam set", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cdf(d: dict, rows: list[str], pts: list[str], out: Path) -> None:
    names = list(d["names"])
    ix = [names.index(n) for n in pts]
    errs = {k: np.linalg.norm(d[k] - d["gt2"], axis=-1)[:, :, ix] for k in rows}
    common = np.logical_and.reduce([np.isfinite(e) for e in errs.values()])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k in rows:
        v = np.sort(errs[k][common])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=style(k).label, color=style(k).color)
    ax.set_xscale("log")
    ax.set_xlabel("2D error (px, log)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Fig. M3 · error distribution, common set", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", required=True, type=Path, help="score json (npz next to it)")
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--picks", type=int, nargs="+", required=True, help="label frames for the overlay figures")
    ap.add_argument("--title", default="SuperAnimal zero-shot 2D accuracy, M2 hand labels")
    ap.add_argument("--goal", default="손 라벨 없는 새 rig 에 쓸 학습 0 검출기의 정확도를 손 라벨로 잰다.")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    s = json.loads(a.score.read_text())
    d = dict(np.load(a.score.with_suffix(".npz")))
    rows = [str(r) for r in d["rows"]]
    m = s["main"]
    pset = next(k[len("failure_rate_"):] for k in m if k.startswith("failure_rate_"))
    pts = POINTS[pset]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    for i, fr in enumerate(a.picks):
        name = f"fig_m1_{i + 1}_f{fr}.png"
        overlay(d, rows, pts, a.frames_dir, fr, a.out_dir / name, f"M1.{i + 1}")
        figs.append(name)
    per_joint(m, rows, a.out_dir / "fig_m2_per_joint.png")
    cdf(d, rows, pts, a.out_dir / "fig_m3_cdf.png")

    def table(res: dict) -> str:
        head = "<tr><th>row</th><th class=c>실패율</th><th class=c>중앙값 px</th><th class=c>중앙값 mm 상당</th></tr>"
        body = "".join(f"<tr><td><span style='color:{style(k).color}'>■</span> {style(k).label}</td><td class=c>{res[f'failure_rate_{pset}'][k]:.3f}</td>"
                       f"<td class=c>{res[f'median_{pset}_px'][k]:.1f}</td><td class=c>{res[f'median_{pset}_mm_eq'][k]:.2f}</td></tr>" for k in rows)
        diffs = "".join(f"<tr><td>{k}</td><td class=c>{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]</td></tr>" for k, v in res["paired_diff_mm_eq"].items())
        return f"<table>{head}{body}</table><table><tr><th>짝 차이 (mm 상당, 중앙값 [95 % frame bootstrap])</th><th class=c>값</th></tr>{diffs}</table>"

    p = next(v for k, v in s.items() if k.startswith("pcut_"))
    best = min((k for k in rows if k != "base"), key=lambda k: m[f"median_{pset}_px"][k])
    legend = legend_html(rows)
    body = f"""<h1>{a.title}</h1>
<div class=card><b>목표</b> {a.goal}<br>
<b>결정 요약</b> 학습 0 행 중 중앙값이 가장 작은 것은 {style(best).label} ({m[f'median_{pset}_px'][best]:.1f} px), A′ 는 {m[f'median_{pset}_px']['base']:.1f} px. 판정선 없음, 결정은 사람.<br>
<b>정의</b> 점 = {', '.join(pts)} · 기준 = 손 라벨 3D 를 Label3D 보정으로 투영 · mm 상당 = px × 깊이 Z / fx · 공통 집합 = 모든 행이 있는 frame · cam · 관절 ({m[f'n_common_{pset}']} 점, {m['n_common_frames']} frame)<br>
<b>색</b> {legend}</div>
<h2>주 지표 (likelihood 문턱 없음)</h2><p class=lead>모든 행을 같은 공통 집합에서 잰다.</p>{table(m)}
<h2>부가: SA 행에 likelihood ≥ 0.5</h2><p class=lead>문턱을 두면 오차와 실패율이 함께 바뀐다.</p>{table(p)}
<h2>방법별 overlay</h2><p class=lead>모든 칸이 같은 crop 이다. 흰 원과 × 의 거리가 오차다.</p>
{''.join(zoom_img(f) for f in figs)}
<h2>관절별 · 분포</h2><p class=lead>Fig. M2 는 관절별 중앙값, Fig. M3 는 공통 집합 전체의 누적 분포다.</p>
{zoom_img("fig_m2_per_joint.png")}{zoom_img("fig_m3_cdf.png", "max-width:560px;width:100%")}
<h2>한계</h2><p class=mut>M2 1 마리 · 1 rig. A′ 는 같은 rig 로 학습해 학습 0 행과 여러 요인이 다르다. 기준 2D 는 3D 재투영이라 가림을 구분하지 못한다.</p>"""
    (a.out_dir / "index.html").write_text(page(a.title, body, "", {}))
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
