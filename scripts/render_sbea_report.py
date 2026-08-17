"""Self-contained HTML report for the SBeA DLC → DLT → SAM2 run.

Inputs are the artefacts the pipeline already writes, so this adds no state:
  --npz   behavior-lab scripts/sbea_dlc_triangulate.py output
  --masks sdannce-poc segmentation/kp_sam2.py output dir (optional)

Usage:
    python scripts/render_sbea_report.py \\
        --npz /mnt/d/data/derived/mac_backups_260813/gpu03_nonM5_260813/sbea_kp3d/rec10-M2-20221108-kp3d.npz \\
        --video-root /mnt/d/data/raw/SBeA/individual \\
        --masks /mnt/d/data/derived/mac_backups_260813/gpu03_nonM5_260813/sbea_masks/rec10-M2 \\
        --out outputs/sbea_report.html
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
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


def read_mask(mask_npz: Path) -> np.ndarray | None:
    """Union of the animal masks in one npz.

    Three key conventions are in circulation and a consumer must read all three
    (sdannce-poc/docs/repo_boundary.md, 260727): `animal_N` from the current kp-SAM2
    producer, bare `mask` from BS `auto_sam3`, and legacy `ratN`. Reading only
    `animal_*` silently returns nothing for the sam3 tree.
    """
    if not mask_npz.exists():
        return None
    with np.load(mask_npz) as z:
        keys = [k for k in z.files if k.startswith(("animal_", "rat"))] or \
               [k for k in z.files if k == "mask"]
        if not keys:
            return None
        return np.logical_or.reduce([z[k].astype(bool) for k in keys])


def cam_dir(root: Path, c: int) -> Path | None:
    """Per-camera mask dir. kp-SAM2 writes `Camera1..4`, auto_sam3 writes `cam0..3`."""
    for name in (f"Camera{c + 1}", f"cam{c}"):
        if (root / name).is_dir():
            return root / name
    return None


def overlay_mask(video: Path, frame: int, mask_npz: Path) -> str | None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    m = read_mask(mask_npz) if ok else None
    if not ok or m is None:
        return None
    img[m] = (0.5 * np.array([255, 120, 0]) + 0.5 * img[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (255, 120, 0), 2)
    return img_b64(img)


def overlay_reproj(video: Path, frame: int, det: np.ndarray, proj: np.ndarray,
                   thr: float) -> str | None:
    """Detected 2D (green) vs the 3D solution projected back (red), joined by a line.

    This is the check the residual number stands for: if the red dots sit on the green
    ones the triangulation agrees with the detector, and the line length IS the residual.
    Overlaying only the detections (as this report did before) cannot show that.
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    for (x, y, q), (u, v) in zip(det, proj):
        if not (np.isfinite([x, y, u, v]).all() and q >= thr):
            continue
        cv2.line(img, (int(x), int(y)), (int(u), int(v)), (200, 200, 200), 1)
        cv2.circle(img, (int(x), int(y)), 4, (0, 220, 0), -1)
        cv2.circle(img, (int(u), int(v)), 4, (0, 60, 235), -1)
    return img_b64(img)


def _tri(xy: np.ndarray, mask: np.ndarray, Ps: np.ndarray) -> np.ndarray:
    """Linear DLT for one frame — same solver the pipeline uses, imported to avoid a second copy."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sbea_bundle_adjust as ba
    return ba.triangulate_batch(xy, mask, np.asarray(Ps))


def project(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = np.concatenate([X, np.ones((len(X), 1))], 1) @ P.T
    w = h[:, 2:]
    return np.where(np.abs(w) > 1e-9, h[:, :2] / np.where(w == 0, 1, w), np.nan)


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


def fig_mask_area(masks: Path, cams: list[int], max_pts: int = 400) -> str:
    """Mask area over time — a flat-ish line means the segmenter kept tracking the animal."""
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for c in cams:
        d = cam_dir(masks, c)
        files = sorted(d.glob("mask_*.npz")) if d else []
        files = files[:: max(1, len(files) // max_pts)]      # sam3 tree has ~8k/cam
        xs, ys = [], []
        for f in files:
            m = read_mask(f)
            if m is not None:
                xs.append(int(f.stem.split("_")[1]))
                ys.append(int(m.sum()))
        if xs:
            ax.plot(xs, ys, lw=1.3, label=f"camera-{c}")
    ax.set_xlabel("frame"); ax.set_ylabel("mask area (px)")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    return fig_b64(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--masks", type=Path, help="kp-SAM2 tree (Camera1..4)")
    ap.add_argument("--masks-sam3", type=Path, help="auto_sam3 tree (cam0..3)")
    ap.add_argument("--ba-npz-dir", type=Path,
                    help="kp3d dir; fits the extrinsic BA correction so the report can "
                         "show reprojection before and after it")
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

    # Reprojection check, before and (optionally) after the extrinsic BA correction.
    Ps = z["P_matrices"]
    variants = [("보정 전 (배포 캘리브 그대로)", Ps)]
    if a.ba_npz_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sbea_bundle_adjust as ba
        # Fit on the OTHER sessions of the same calibration group — never on the one shown,
        # so the improvement in this report is held-out, matching the 260816 protocol.
        same = [f for f in sorted(a.ba_npz_dir.glob("*-kp3d.npz"))
                if ba.group_key(np.load(f)["P_matrices"]) == ba.group_key(Ps)
                and f.name != a.npz.name]
        base = [ba.decompose(P) for P in Ps]
        variants.append(("보정 후 (외부 파라미터 BA · 이 세션은 적합에서 제외 = held-out)",
                         ba.apply_offsets(base, ba.fit([ba.load(f, 400) for f in same[:5]], base))))

    reproj_rows = ""
    for label, PP in variants:
        # Re-triangulate with THIS camera set. Reprojecting the original 3D solution through
        # corrected cameras would measure the camera change, not the corrected reconstruction —
        # and would show almost no improvement even when the correction is real.
        xy0 = kp2d[:, 0, :, :2].transpose(1, 0, 2)
        m0 = (kp2d[:, 0, :, 2].T >= thr) & np.isfinite(xy0).all(2)
        X = _tri(xy0, m0, PP)
        cards = "".join(
            f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>camera-{c} · '
            f'median {np.nanmedian(np.linalg.norm(project(X, PP[c]) - kp2d[c, 0, :, :2], axis=1)):.1f} px'
            f'</figcaption></figure>'
            for c in range(n_cam)
            if (b := overlay_reproj(a.video_root / f"{session}-camera-{c}.mp4", frames[0],
                                    kp2d[c, 0], project(X, PP[c]), thr)))
        reproj_rows += f"<h3>{label}</h3><div class='grid'>{cards}</div>"

    mask_html = "<p class='sub'>마스크 산출물 없음 — <code>--masks</code> 미지정</p>"
    sets = [(n, p) for n, p in (("kp-SAM2 (키포인트 프롬프트)", a.masks),
                                ("auto_sam3 (자동, 키포인트 불요)", a.masks_sam3))
            if p and p.exists()]
    if sets:
        mask_html = ""
        for label, root in sets:
            cards = "".join(
                f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>camera-{c} · '
                f'{int(m.sum()):,} px</figcaption></figure>'
                for c in range(n_cam)
                if (d := cam_dir(root, c)) and (m := read_mask(d / f"mask_{frames[0]:06d}.npz")) is not None
                and (b := overlay_mask(a.video_root / f"{session}-camera-{c}.mp4", frames[0],
                                       d / f"mask_{frames[0]:06d}.npz")))
            n_files = sum(len(list(d.glob("mask_*.npz"))) for c in range(n_cam)
                          if (d := cam_dir(root, c)))
            mask_html += (f"<h3>{label}</h3><p class='sub'><code>{root}</code> · "
                          f"{n_files:,} 파일</p><div class='grid'>{cards}</div>"
                          f"<img class='w' src='data:image/png;base64,{fig_mask_area(root, list(range(n_cam)))}'>")

    valid = np.isfinite(kp3d).all(axis=2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(TEMPLATE.format(
        session=session, T=len(frames), K=len(names), n_cam=n_cam,
        lik=np.nanmean(kp2d[..., 2]), valid=valid.mean(),
        med=np.nanmedian(resid), ident=float(z["residual_identity_px"]),
        best=float(z["residual_best_px"]), order=order, thr=thr,
        cards2d=cards2d, bar=bar, hist=hist, sk=sk, mask_html=mask_html,
        reproj_rows=reproj_rows,
        joints=", ".join(names)), encoding="utf-8")
    print(f"wrote {a.out}")


TEMPLATE = """<!doctype html><html lang="ko"><meta charset="utf-8">
<title>SBeA — 키포인트·마스크 시각 검증</title>
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
<h1>SBeA — 키포인트·마스크 시각 검증</h1>
<p class="sub">세션 <b>{session}</b> · {T} 프레임 × {n_cam} 카메라 · {K} 관절 · likelihood 임계 {thr}</p>

<p class="ssot">좌표·규약 단일 진입점 → <code>behavior-lab/docs/conventions.md</code> — 포즈 텐서 <code>(T,K,D)</code> · 스켈레톤(KP22/rat23/SBeA16) · 카메라·투영 규약. 그 문서의 값은 <code>tests/test_conventions_doc.py</code> 가 소스와 대조하므로 이 리포트와 드리프트할 수 없다.</p>
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
<p class="sub">저자 DLC 스냅샷의 출력. 3D 삼각측량의 <b>입력</b>이다 — 점이 쥐 위에 있으면 정상.</p>
<div class="grid">{cards2d}</div>

<h2>3. 재투영 검증 — 눈으로 보는 잔차</h2>
<p class="sub"><span style="color:#0a0">●</span> 검출 2D · <span style="color:#eb3c00">●</span> 3D 해를 되쏜 것 ·
회색 선 길이 = 그 관절의 잔차. <b>두 점이 겹치면 삼각측량이 검출과 일치</b>한다.
사지 끝(발톱)에서 선이 길면 좌/우 라벨 불일치이지 캘리브 문제가 아니다.</p>
{reproj_rows}

<h2>4. 재투영 잔차 — 수치</h2>
<div class="grid">
<figure><img src="data:image/png;base64,{bar}"><figcaption>관절별 median</figcaption></figure>
<figure><img src="data:image/png;base64,{hist}"><figcaption>전체 분포</figcaption></figure>
</div>
<div class="box">잔차 ~{med:.0f} px 는 SAM2 프롬프트 용도로는 충분하나(마우스 폭 ~200 px) 정량 사지 kinematics 에는 부족하다.
<b>원인은 렌즈 왜곡이 아니다</b> — 260728 에 20세션 전량으로 기각했다(잔차 방향 |cos| 0.632 = 등방성 0.64).
실제 성분은 둘: ① <b>캘리브 외부 파라미터 드리프트</b> — BA 로 제거되며 중앙선 관절이 19~21 → 10.7 px 로 반토막
② <b>카메라별 좌/우 라벨 불일치</b> — 대칭 관절에만 남고 후처리 재라벨로는 못 고친다(260817 검정).
근거 = vault <code>260816_behaviorlab_sbea_bundle_adjustment</code> · <code>260817_behaviorlab_sbea_lr_after_ba</code>.</div>

<h2>5. 3D 스켈레톤</h2>
<img class="w" src="data:image/png;base64,{sk}">

<h2>6. 마스크 — 두 세트 비교</h2>
<div class="box"><b>같은 세션에 마스크가 두 벌 있다.</b> 대체재가 아니라 <b>프롬프트 소스가 다른 것</b>
(<code>sdannce-poc/docs/repo_boundary.md</code>): <b>kp-SAM2</b> 는 우리 DLT 3D 키포인트로 프롬프트하고,
<b>auto_sam3</b> 는 키포인트 없이 자동으로 뽑는다. 어느 쪽을 정본으로 삼을지는 아래 비교를 보고 판단한다.</div>
{mask_html}
</html>"""


if __name__ == "__main__":
    main()
