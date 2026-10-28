"""One style schema for every keypoint-predictor report: colour = detector + input, marker = point-mapping variant.

Okabe-Ito colours (colour-blind safe). Yellow is kept for flagged hand labels only, white rings for hand labels.
Score JSON row keys from the M2 and AVATAR scorers map onto these entries through ROW_ALIAS.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    label: str
    color: str
    marker: str
    meaning: str


LABEL_RING = Style("hand label", "#ffffff", "o", "사람 라벨 (흰 원)")
FLAGGED_RING = Style("flagged label error", "#F0E442", "o", "알려진 라벨 오류, 몸 밖 (노란 원)")

STYLES: dict[str, Style] = {
    "dannce_a1": Style("A′ (DANNCE, same rig)", "#0072B2", "D", "같은 rig 로 학습한 3D 모델, 상한 참고"),
    "sleap1423": Style("SLEAP 1423 (val)", "#CC79A7", "s", "SUBTLE 학습 모델, 학습에 안 쓰인 val frame"),
    "sa_quad_video": Style("SA-Quadruped · video", "#E69F00", "^", "학습 0, mp4v 영상 입력"),
    "sa_quad_image": Style("SA-Quadruped · image", "#D55E00", "v", "학습 0, png 이미지 입력"),
    "sa_quad_image_base": Style("SA-Quadruped · image (ear base)", "#D55E00", "<", "같은 추론, 귀 = earbase · 목 = neck_end 대응"),
    "sa_quad_dlt": Style("SA-Quadruped · DLT→2D", "#56B4E9", "P", "학습 0 검출을 여러 cam 삼각측량 뒤 재투영"),
    "sa_tvm_image": Style("SA-TopViewMouse · image", "#009E73", "X", "학습 0, 위에서 찍은 영상으로만 학습"),
    "sa_tvm_image_eartip": Style("SA-TopViewMouse · image (ear tip)", "#009E73", "*", "같은 추론, 귀 = ear_tip 대응"),
}

ROW_ALIAS = {"base": "dannce_a1", "sa_quad_2d": "sa_quad_video", "sa_quad_png": "sa_quad_image", "sa_quad_dlt2d": "sa_quad_dlt",
             "sa_tvm_png": "sa_tvm_image", "sa_tvm_eartip_png": "sa_tvm_image_eartip", "sa_quad": "sa_quad_image",
             "sa_quad_base": "sa_quad_image_base", "sleap1423_val": "sleap1423", "sa_tvm_bot": "sa_tvm_image"}

assert all(v in STYLES for v in ROW_ALIAS.values())
assert len({(s.color, s.marker) for s in STYLES.values()}) == len(STYLES), "two entries share colour and marker"


def style(row: str) -> Style:
    return STYLES[ROW_ALIAS.get(row, row)]


def legend_html(rows: list[str]) -> str:
    """Definitions-box lines: one per colour/marker actually drawn, plus the two label rings."""
    seen, out = set(), []
    for r in rows:
        s = style(r)
        if s.label not in seen:
            seen.add(s.label)
            out.append(f"<span style='color:{s.color};font-weight:700'>■ {s.marker}</span> {s.label} — {s.meaning}")
    out += [f"<span style='color:#888'>○</span> {LABEL_RING.meaning}", f"<span style='color:#c9b800'>○</span> {FLAGGED_RING.meaning}"]
    return "<br>".join(out)
