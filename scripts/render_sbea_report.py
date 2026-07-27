"""Self-contained HTML report for the SBeA DLC → DLT → SAM2 run.

Inputs are the artefacts the pipeline already writes, so this adds no state:
  --npz   behavior-lab scripts/sbea_dlc_triangulate.py output
  --masks sdannce-poc segmentation/kp_sam2.py output dir (optional)

Usage:
    python scripts/render_sbea_report.py \\
        --npz /node_data/joon/data/sbea_kp3d/rec10-M2-20221108-kp3d.npz \\
        --video-root /node_data_2/joon/data/external/SBeA/individual \\
        --masks /node_data/joon/data/sbea_masks/rec10-M2 \\
        --out outputs/sbea_report.html
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Anatomical wiring: forelimbs hang off the neck, hindlimbs off the tail root —
# same correction applied to KP22 (260702), not a single mid-body hub.
EDGES = [(0, 1), (0, 2), (0, 3), (3, 12), (12, 13), (13, 14), (14, 15),
         (3, 4), (4, 8), (3, 5), (5, 9), (13, 6), (6, 10), (13, 7), (7, 11)]


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img_b64(img: np.ndarray, width: int = 460) -> str:
    s = width / img.shape[1]
    small = cv2.resize(img, (width, max(1, int(img.shape[0] * s))))
    return base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 86])[1]).decode()


def overlay_2d(video: Path, frame: int, kp: np.ndarray, thr: float) -> str | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    for a, b in EDGES:
        if kp[a, 2] >= thr and kp[b, 2] >= thr:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)), (255, 190, 0), 2)
    for x, y, q in kp:
        if np.isfinite([x, y]).all():
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0) if q >= thr else (0, 140, 255), -1)
    return img_b64(img)


def overlay_mask(video: Path, frame: int, mask_npz: Path) -> str | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok or not mask_npz.exists():
        return None
    with np.load(mask_npz) as z:
        keys = [k for k in z.files if k.startswith("animal_")]
        m = np.logical_or.reduce([z[k].astype(bool) for k in keys]) if keys else None
    if m is None:
        return None
    img[m] = (0.5 * np.array([255, 120, 0]) + 0.5 * img[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (255, 120, 0), 2)
    return img_b64(img)


def fig_residual(resid: np.ndarray, names: list[str]) -> tuple[str, str]:
    per_kp = np.nanmedian(resid, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.bar(range(len(names)), per_kp, color="#4c8bf5")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("median reproj. error (px)")
    ax.grid(axis="y", alpha=.25)
    bar = fig_b64(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    v = resid[np.isfinite(resid)]
    ax.hist(v, bins=50, color="#4c8bf5", alpha=.85)
    ax.axvline(np.median(v), color="#e2574c", ls="--", label=f"median {np.median(v):.1f} px")
    ax.set_xlabel("reproj. error (px)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=.25)
    return bar, fig_b64(fig)


def fig_skeleton(kp3d: np.ndarray, frames: list[int]) -> str:
    fig = plt.figure(figsize=(9, 3.2))
    for i, t in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
        P = kp3d[t]
        for a, b in EDGES:
            if np.isfinite(P[[a, b]]).all():
                ax.plot(*zip(P[a], P[b]), c="#4c8bf5", lw=1.4)
        ok = np.isfinite(P).all(axis=1)
        ax.scatter(*P[ok].T, c="#e2574c", s=9)
        ax.set_title(f"t={t}", fontsize=9)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    return fig_b64(fig)


def fig_mask_area(masks: Path, cams: list[int]) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for c in cams:
        files = sorted((masks / f"Camera{c}").glob("mask_*.npz"))
        xs, ys = [], []
        for f in files:
            with np.load(f) as z:
                keys = [k for k in z.files if k.startswith("animal_")]
                if keys:
                    xs.append(int(f.stem.split("_")[1]))
                    ys.append(int(np.logical_or.reduce([z[k].astype(bool) for k in keys]).sum()))
        if xs:
            ax.plot(xs, ys, lw=1.3, label=f"Camera{c}")
    ax.set_xlabel("frame"); ax.set_ylabel("mask area (px)")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    return fig_b64(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--masks", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    z = np.load(a.npz, allow_pickle=True)
    kp3d, kp2d = z["keypoints_3d"], z["keypoints_2d"]
    resid, frames = z["reproj_residual_px"], z["frame_indices"]
    names = [str(n) for n in z["keypoint_names"]]
    order, thr = [int(i) for i in z["camera_order"]], float(z["prob_min"])
    session = a.npz.stem.replace("-kp3d", "")
    n_cam = kp2d.shape[0]

    cards2d = "".join(
        f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>camera-{c} '
        f'(P[{order[c]}]) · mean likelihood {np.nanmean(kp2d[c, ..., 2]):.3f}</figcaption></figure>'
        for c in range(n_cam)
        if (b := overlay_2d(a.video_root / f"{session}-camera-{c}.mp4", frames[0], kp2d[c, 0], thr)))

    bar, hist = fig_residual(resid, names)
    sk = fig_skeleton(kp3d, [0, len(kp3d) // 3, 2 * len(kp3d) // 3, len(kp3d) - 1])

    mask_html = "<p class='sub'>마스크 산출물 없음 — <code>--masks</code> 미지정</p>"
    if a.masks and a.masks.exists():
        cards = "".join(
            f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>camera-{c}</figcaption></figure>'
            for c in range(n_cam)
            if (b := overlay_mask(a.video_root / f"{session}-camera-{c}.mp4", frames[0],
                                  a.masks / f"Camera{c + 1}" / f"mask_{frames[0]:06d}.npz")))
        mask_html = (f"<div class='grid'>{cards}</div>"
                     f"<img class='w' src='data:image/png;base64,{fig_mask_area(a.masks, list(range(1, n_cam + 1)))}'>")

    valid = np.isfinite(kp3d).all(axis=2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(TEMPLATE.format(
        session=session, T=len(frames), K=len(names), n_cam=n_cam,
        lik=np.nanmean(kp2d[..., 2]), valid=valid.mean(),
        med=np.nanmedian(resid), ident=float(z["residual_identity_px"]),
        best=float(z["residual_best_px"]), order=order, thr=thr,
        cards2d=cards2d, bar=bar, hist=hist, sk=sk, mask_html=mask_html,
        joints=", ".join(names)), encoding="utf-8")
    print(f"wrote {a.out}")


TEMPLATE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>SBeA — DLC to SAM2 pipeline report</title>
<style>
:root{{color-scheme:light dark}}
body{{font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:2rem;
 background:#fbfbfd;color:#1a1a1a;max-width:1180px;margin-inline:auto}}
@media(prefers-color-scheme:dark){{body{{background:#131316;color:#e8e8ea}}
 td,th{{border-color:#2a2a2e}} th{{background:#1e1e22}} figcaption,.sub{{color:#a8a8ad}}
 .box{{background:#1a1a1f;border-color:#2a2a2e}}}}
h1{{font-size:1.55rem;margin:0 0 .2rem}} h2{{font-size:1.1rem;margin:2.2rem 0 .7rem}}
.sub{{color:#666;margin:0 0 1.4rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.4rem 0}}
td,th{{border:1px solid #e4e4e8;padding:.42rem .65rem;text-align:right}}
td:first-child,th:first-child{{text-align:left}} th{{background:#f2f2f5;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:1rem}}
figure{{margin:0}} img{{width:100%;border-radius:6px;display:block}}
img.w{{margin-top:1rem}} figcaption{{font-size:.82rem;color:#666;margin-top:.35rem}}
.box{{background:#f6f7f9;border:1px solid #e4e4e8;border-radius:8px;padding:.8rem 1rem;margin:.8rem 0}}
code{{font-size:.88em}} .k{{font-variant-numeric:tabular-nums}}
.ssot{{font-size:.85rem;color:#666;border-left:3px solid #4c8bf5;padding:.5rem .8rem;margin:1.2rem 0;background:#f6f7f9;border-radius:0 6px 6px 0}}
</style>
<h1>SBeA — 저자 DLC 모델 → DLT 삼각측량 → SAM2 마스크</h1>
<p class="sub">세션 <b>{session}</b> · {T} 프레임 × {n_cam} 카메라 · {K} 관절 · likelihood 임계 {thr}</p>

<p class="ssot">좌표·규약 단일 진입점 → <code>behavior-lab/docs/dev/README.md §2</code> (포즈 텐서 <code>(T,K,D)</code>·카메라 규약·스켈레톤 정의). 이 리포트의 수치는 그 정본을 따른다.</p>
<h2>1. 요약</h2>
<table>
<tr><th>단계</th><th>결과</th></tr>
<tr><td>2D 검출 (저자 DLC snapshot-1030000)</td><td class="k">평균 likelihood {lik:.3f}</td></tr>
<tr><td>3D 삼각측량 (선형 DLT)</td><td class="k">valid {valid:.1%}</td></tr>
<tr><td>재투영 잔차</td><td class="k">median {med:.1f} px</td></tr>
<tr><td>카메라↔P 순서 (자동 선택)</td><td class="k">{order} — {best:.1f} px vs identity {ident:.1f} px</td></tr>
</table>
<div class="box"><b>관절 순서</b>(저자 CSV 헤더 그대로): {joints}</div>

<h2>2. 2D 검출 — 카메라별</h2>
<div class="grid">{cards2d}</div>

<h2>3. 재투영 잔차</h2>
<div class="grid">
<figure><img src="data:image/png;base64,{bar}"><figcaption>관절별 median</figcaption></figure>
<figure><img src="data:image/png;base64,{hist}"><figcaption>전체 분포</figcaption></figure>
</div>
<div class="box">잔차 ~{med:.0f} px 는 SAM2 프롬프트 용도로는 충분하나(마우스 폭 ~200 px) 정량 3D 분석에는 부족하다.
가장 유력한 원인은 <b>렌즈 왜곡 미모델링</b> — SBeA 는 왜곡 계수를 배포하지 않는다. 특정 카메라 불량은 아님(4대 모두 유사).</div>

<h2>4. 3D 스켈레톤</h2>
<img class="w" src="data:image/png;base64,{sk}">

<h2>5. SAM2 마스크 (키포인트 프롬프트)</h2>
{mask_html}
</html>"""


if __name__ == "__main__":
    main()
