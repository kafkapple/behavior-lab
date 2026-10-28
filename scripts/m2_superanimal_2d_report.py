"""Figures + one HTML page for the M2 SuperAnimal 2D accuracy scores (m2_superanimal_2d_score.py output).

Fig. 1 overlay sheet (hand label / A' / SuperAnimal 2D on 3 frames x 6 cameras, crops around the animal),
Fig. 2 per-joint median px, Fig. 3 cumulative extremity-8 error (mm-equivalent). Source pixels are not
changed; markers are drawn on top.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavior_lab.pose.html_common import page

ROWS = {"base": ("A′ (DANNCE, same rig)", "#2f6fdb"), "sa_quad_2d": ("SuperAnimal 2D", "#e07b00"),
        "sa_quad_dlt2d": ("SuperAnimal DLT→2D", "#8a3ffc")}
EXT = ["nose", "L_ear", "R_ear", "tail_end", "L_paw_end", "R_paw_end", "L_foot", "R_foot"]


def overlay(d: dict, frames_dir: Path, picks: list[int], out: Path) -> None:
    names = list(d["names"])
    ext = [names.index(n) for n in EXT]
    n_cam = d["gt2"].shape[0]
    fig, axes = plt.subplots(len(picks), n_cam, figsize=(2.6 * n_cam, 2.8 * len(picks)))
    for r, fr in enumerate(picks):
        fi = int(np.where(d["frames"] == fr)[0][0])
        for c in range(n_cam):
            ax = axes[r, c]
            img = plt.imread(frames_dir / f"cam{c + 1}_f{fr:06d}.png")
            g = d["gt2"][c, fi, ext]
            ok = np.isfinite(g).all(1)
            x0, y0 = np.nanmin(g[ok], 0) - 60
            x1, y1 = np.nanmax(g[ok], 0) + 60
            ax.imshow(img)
            ax.set_xlim(max(x0, 0), min(x1, img.shape[1]))
            ax.set_ylim(min(y1, img.shape[0]), max(y0, 0))
            ax.scatter(*g.T, s=28, facecolors="none", edgecolors="white", linewidths=1.4)
            for key, mk in (("base", "x"), ("sa_quad_2d", "+")):
                p = d[key][c, fi, ext]
                ax.scatter(*p.T, s=30, marker=mk, color=ROWS[key][1], linewidths=1.4)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"cam{c + 1}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"frame {fr}", fontsize=9)
    fig.suptitle("Fig. M1 · white ○ hand label, blue × A′, orange + SuperAnimal 2D (extremity 8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def per_joint(m: dict, out: Path) -> None:
    joints = list(m["per_joint_median_px"]["base"])
    x = np.arange(len(joints))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, (key, (lab, col)) in enumerate(ROWS.items()):
        v = [m["per_joint_median_px"][key][j] or np.nan for j in joints]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, label=lab, color=col)
    ax.set_xticks(x, joints, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("median error (px)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Fig. M2 · per-joint median 2D error, common frame·cam set", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def cdf(d: dict, out: Path) -> None:
    names = list(d["names"])
    ext = [names.index(n) for n in EXT]
    errs = {k: np.linalg.norm(d[k] - d["gt2"], axis=-1)[:, :, ext] for k in ROWS}
    common = np.logical_and.reduce([np.isfinite(e) for e in errs.values()])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, (lab, col) in ROWS.items():
        v = np.sort(errs[k][common])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=lab, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("extremity-8 2D error (px, log)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Fig. M3 · error distribution, common set", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", required=True, type=Path, help="score_stage0.json (npz next to it)")
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--picks", type=int, nargs="+", required=True, help="label frames for Fig. M1")
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    s = json.loads(a.score.read_text())
    d = dict(np.load(a.score.with_suffix(".npz")))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    overlay(d, a.frames_dir, a.picks, a.out_dir / "fig_m1_overlay.png")
    per_joint(s["main"], a.out_dir / "fig_m2_per_joint.png")
    cdf(d, a.out_dir / "fig_m3_cdf.png")

    def row(res: dict, key: str) -> str:
        return (f"<tr><td>{ROWS[key][0]}</td><td class=c>{res['failure_rate_ext8'][key]:.3f}</td>"
                f"<td class=c>{res['median_ext8_px'][key]:.1f}</td><td class=c>{res['median_ext8_mm_eq'][key]:.2f}</td></tr>")

    def diffs(res: dict) -> str:
        return "".join(f"<tr><td>{k}</td><td class=c>{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]</td></tr>" for k, v in res["paired_diff_mm_eq"].items())

    m, p = s["main"], s["pcut_0.5"]
    sw = m["lr_swap_sa_quad_2d"]
    med, pj = m["median_ext8_px"], m["per_joint_median_px"]
    d_sa = m["paired_diff_mm_eq"]["sa_quad_2d-base"][0]
    sw_lo, sw_hi = min(v["swap_fraction"] for v in sw.values()), max(v["swap_fraction"] for v in sw.values())
    body = f"""<h1>SuperAnimal zero-shot 2D accuracy, M2 hand labels</h1>
<div class=card><b>목표</b> 손 라벨 없는 새 rig 에 쓸 학습 0 검출기의 정확도를 손 라벨로 잰다.<br>
<b>결정 요약</b> SuperAnimal-Quadruped 2D 는 같은 rig 로 학습한 A′ 보다 끝점 8 중앙값이 멀다 ({med['sa_quad_2d']:.1f} 대 {med['base']:.1f} px). 판정선 없음, 결정은 사람.<br>
<b>정의</b> 기준 = 손 라벨 3D 를 Label3D 보정으로 투영 · mm 상당 = px × 깊이 Z / fx · 공통 집합 = 세 행이 모두 있는 frame · cam · 관절 ({m['n_common_ext8']} 점, {m['n_common_frames']} frame) · 색: 파랑 A′ · 주황 SuperAnimal 2D · 보라 SuperAnimal DLT→2D</div>
<h2>주 지표 (likelihood 문턱 없음)</h2><p class=lead>SuperAnimal 2D 는 A′ 보다 끝점 8 에서 mm 상당 {d_sa:.2f} mm 멀다 (짝 차이 중앙값).</p>
<table><tr><th>row</th><th class=c>실패율</th><th class=c>중앙값 px</th><th class=c>중앙값 mm 상당</th></tr>{''.join(row(m, k) for k in ROWS)}</table>
<table><tr><th>짝 차이 (mm 상당, 중앙값 [95 % frame bootstrap])</th><th class=c>값</th></tr>{diffs(m)}</table>
<h2>부가: 두 SuperAnimal 행에 likelihood ≥ 0.5</h2><p class=lead>문턱을 두면 오차는 줄지만 2D 실패율이 {p['failure_rate_ext8']['sa_quad_2d']:.2f} 로 오른다.</p>
<table><tr><th>row</th><th class=c>실패율</th><th class=c>중앙값 px</th><th class=c>중앙값 mm 상당</th></tr>{''.join(row(p, k) for k in ROWS)}</table>
<table><tr><th>짝 차이</th><th class=c>값</th></tr>{diffs(p)}</table>
<h2>그림</h2><p class=lead>Fig. M1: 발 끝 (paw_end) 에서 주황이 흰 원에서 가장 멀다. tail_end 에서는 파랑이 더 멀다.</p>
<img src="fig_m1_overlay.png" style="width:100%">
<p class=lead>Fig. M2: SuperAnimal 은 tail_end 에서 A′ 보다 가깝고 ({pj['sa_quad_2d']['tail_end']:.1f} 대 {pj['base']['tail_end']:.1f} px), 발 끝 · 몸통 관절에서 멀다.</p>
<img src="fig_m2_per_joint.png" style="width:100%">
<p class=lead>Fig. M3: 문턱 없는 DLT→2D 는 꼬리 쪽 분포가 가장 길다 (낮은 likelihood 검출이 삼각측량을 끈다).</p>
<img src="fig_m3_cdf.png" style="max-width:560px;width:100%">
<h2>좌우 바뀜 (SuperAnimal 2D)</h2><p class=lead>발 · 뒷발 점의 {sw_lo:.0%}–{sw_hi:.0%} 는 반대쪽 손 라벨에 더 가깝다.</p>
<table><tr><th>관절</th><th class=c>바뀜 비율</th><th class=c>min(L,R) 중앙값 px</th></tr>{''.join(f"<tr><td>{k}</td><td class=c>{v['swap_fraction']:.2f}</td><td class=c>{v['median_min_px']:.1f}</td></tr>" for k, v in sw.items())}</table>
<h2>한계</h2><p class=mut>M2 1 마리 · 1 rig. A′ 는 같은 rig 로 학습해 SuperAnimal 과 3 요인이 다르다. 기준 2D 는 3D 재투영이라 가림을 구분하지 못한다. SuperAnimal 입력은 mp4v 로 다시 압축한 영상이다.</p>"""
    (a.out_dir / "index.html").write_text(page("M2 SuperAnimal 2D accuracy", body, "", {}))
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
