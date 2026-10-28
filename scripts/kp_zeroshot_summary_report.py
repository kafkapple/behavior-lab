"""One summary page for the zero-shot keypoint study: M2 stage 0 (8 extremities), M2 stage 1 (head + tail 4) and
AVATAR (SUBTLE hand labels). Numbers are read from the three score JSONs; colours come from predictors.styles.

Layout follows html-report-pages: tldr -> goal -> hypothesis -> data -> method -> metrics -> result -> decision,
then a definitions box, then tables T1-T3 and figures with captions.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from behavior_lab.pose.html_common import page, zoom_img
from behavior_lab.pose.predictors.styles import legend_html, style


def cell(row: str) -> str:
    s = style(row)
    return f"<span style='color:{s.color};font-weight:700'>■ {s.marker}</span> {s.label}"


def m2_table(res: dict, pset: str, rows: list[str]) -> str:
    body = "".join(f"<tr><td>{cell(r)}</td><td class=c>{res[f'failure_rate_{pset}'][r]:.2f}</td><td class=c>{res[f'median_{pset}_px'][r]:.1f}</td>"
                   f"<td class=c>{res[f'median_{pset}_mm_eq'][r]:.2f}</td></tr>" for r in rows)
    return f"<table><tr><th>방법</th><th class=c>실패율</th><th class=c>중앙값 px</th><th class=c>중앙값 mm 상당</th></tr>{body}</table>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage0", required=True, type=Path)
    ap.add_argument("--stage1", required=True, type=Path)
    ap.add_argument("--avatar", required=True, type=Path)
    ap.add_argument("--fig-m2", required=True, type=Path, help="stage 1 overlay png (all methods)")
    ap.add_argument("--fig-avatar", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    s0, s1, av = (json.loads(p.read_text())["main"] for p in (a.stage0, a.stage1, a.avatar))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for f in (a.fig_m2, a.fig_avatar):
        shutil.copy(f, a.out_dir / f.name)

    q0, b0 = s0["median_ext8_px"]["sa_quad_2d"], s0["median_ext8_px"]["base"]
    d0 = s0["paired_diff_mm_eq"]["sa_quad_2d-base"]
    t1 = s1["failure_rate_head4"]["sa_tvm_png"]
    inp = s1["paired_diff_mm_eq"]["sa_quad_2d-sa_quad_png"]
    sv, sq = av["sleap1423_val"]["clean"]["all_parts"], av["sa_quad_on_val"]["clean"]["all_parts"]
    sq_all = av["sa_quad"]["clean"]["all_parts"]
    cams = av["sa_quad"]["clean"]["by_camera"]
    side = [v["median_px"] for c, v in cams.items() if c != "bot"]

    av_rows = [("sleap1423_val", sv, "같은 val 32 장"), ("sa_quad", sq, "같은 val 32 장"), ("sa_quad", sq_all, "450 장 전체"),
               ("sa_tvm_bot", av["sa_tvm_bot"]["clean"]["all_parts"], "bot cam 128 장")]
    t3 = "".join(f"<tr><td>{cell(r)}</td><td>{where}</td><td class=c>{v['n']}</td><td class=c>{v['fail']:.2f}</td><td class=c>{v['median_px']:.1f}</td></tr>"
                 for r, v, where in av_rows)

    body = f"""<h1>학습 0 keypoint 검출기 정확도 요약</h1>
<div class=card>
<b>tldr</b> SuperAnimal-Quadruped (학습 0) 는 손 라벨 기준으로 같은 rig 학습 모델보다 약 2 배 멀고, AVATAR 에서는 절반 이상 검출하지 못한다.<br>
<b>목표</b> 손 라벨 없는 새 rig 에서 학습 0 검출기를 첫 keypoint source 로 쓸 수 있는지 정확도로 판단한다.<br>
<b>가설 (탐색)</b> 학습 0 검출기는 같은 rig 로 학습한 모델보다 멀다. 판정선은 두지 않는다 (문헌 margin 없음).<br>
<b>데이터</b> M2 손 라벨 91 frame × 6 cam (DANNCE Label3D) · AVATAR SUBTLE SLEAP 라벨 450 frame (단일 시점, 알려진 귀 오류 점 제외).<br>
<b>방법</b> DLC 3.0.1 SuperAnimal zero-shot (상자 문턱 0.9). 비교 = M2 는 A′, AVATAR 는 SLEAP 1423 의 val 예측.<br>
<b>지표</b> 2D 오차 = ‖예측 − 기준‖ px. mm 상당 = px × Z / fx (M2). 실패율 = 그 점의 출력이 없는 비율. 좌우 쌍은 1:1 배정 (AVATAR).<br>
<b>결과</b> M2 끝점 8: {q0:.1f} 대 A′ {b0:.1f} px (짝 차이 +{d0[0]:.2f} mm 상당 [{d0[1]:+.2f}, {d0[2]:+.2f}]). TopViewMouse 실패율 {t1:.2f}. AVATAR 같은 32 장: SA {sq['median_px']:.1f} px · 실패 {sq['fail']:.2f} 대 SLEAP {sv['median_px']:.1f} px · 실패 {sv['fail']:.2f}.<br>
<b>결정</b> 옆 · 비스듬한 cam 의 첫 학습 0 source = SA-Quadruped (사용자 261028). AVATAR 첫 source 는 사람 결정 대기.</div>
<div class=card><b>정의 · 색</b><br>{legend_html(['base', 'sleap1423_val', 'sa_quad_2d', 'sa_quad_png', 'sa_quad_dlt2d', 'sa_quad_base', 'sa_tvm_png', 'sa_tvm_eartip_png'])}<br>
A′ = DANNCE 를 M1 라벨로만 학습해 M2 를 예측한 모델 · SA = SuperAnimal · DLT = 여러 cam 삼각측량 · val = 학습에 안 쓰인 frame</div>
<h2>M2 끝점 8</h2><p class=lead>학습 0 검출은 A′ 보다 약 2 배 멀고, 삼각측량해도 줄지 않는다.</p>
<p><b>Table T1 · M2 extremity-8 error</b></p>{m2_table(s0, 'ext8', ['base', 'sa_quad_2d', 'sa_quad_dlt2d'])}
<h2>M2 머리 · 꼬리 4 점</h2><p class=lead>입력 압축은 영향이 없고 (짝 차이 {inp[0]:+.3f} mm 상당), TopViewMouse 는 옆 cam 에서 거의 실패한다.</p>
<p><b>Table T2 · M2 head + tail 4 error</b></p>{m2_table(s1, 'head4', ['base', 'sa_quad_2d', 'sa_quad_png', 'sa_tvm_png'])}
{zoom_img(a.fig_m2.name)}<p><b>Fig. K1 · M2 overlay, one row per method</b><br>같은 crop 에서 흰 원과 기호의 거리가 오차다. 그림을 누르면 원본 크기로 열린다.</p>
<h2>AVATAR 손 라벨</h2><p class=lead>SA-Quadruped 는 SLEAP 1423 보다 멀고 실패가 많다. side cam 은 {min(side):.1f}–{max(side):.1f} px, bot cam 은 {cams['bot']['median_px']:.1f} px.</p>
<p><b>Table T3 · AVATAR error vs SUBTLE hand labels</b></p><table><tr><th>방법</th><th>범위</th><th class=c>n (이미지 · 부위)</th><th class=c>실패율</th><th class=c>중앙값 px</th></tr>{t3}</table>
{zoom_img(a.fig_avatar.name)}<p><b>Fig. K2 · AVATAR overlay, one row per method</b><br>노란 원은 몸 밖에 찍힌 라벨 오류라 채점에서 뺐다.</p>
<h2>한계</h2><p class=mut>M2 는 1 마리 · 1 rig. AVATAR SLEAP 비교는 32 장이고, SLEAP 은 같은 주석자의 라벨로 학습했다. 몇 px 라벨 오차는 재라벨 없이 모른다.</p>"""
    (a.out_dir / "index.html").write_text(page("Zero-shot keypoint summary", body, "", {}))
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
