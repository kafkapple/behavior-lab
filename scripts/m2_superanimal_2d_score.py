"""2D accuracy of SuperAnimal-Quadruped zero-shot on the M2 (markerless_mouse_2) hand labels.

Pre-registered in vault 2603_BehaviorSplatter/_Notes/261028_BS_superanimal_zeroshot_kp_pilot_stub.md.
Rows: base = DANNCE A' 3D reprojected (upper reference), then one row per --sa source (DLC video h5 dir or
run_dlc.py CSV dir), and for each --dlt-for label the same detections triangulated by unweighted DLT
(rig.multiview.triangulate) and reprojected. Stage 0 = --sa sa_quad_2d=superanimal_quadruped:<h5 dir>
--dlt-for sa_quad_2d --points ext8 (stage 1 note 261028_BS_superanimal_m2_stage1_png_stub). Reference = hand-label 3D projected with the Label3D calibration (distortion included).

Self-checks (asserts): hand-label projection vs Label3D data_2d <= 5 px median in every camera;
A' extremity-8 3D mean reproduces gate_A_scratch.json ext8_mean_mm; h5 rows match the label frames.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

from behavior_lab.rig.multiview import triangulate

# SuperAnimal-Quadruped name -> mouse_22 name (= BehaviorSplatter scripts/analysis/superanimal_m2_gate.py MAP)
MAP = {"nose": "nose", "left_earend": "L_ear", "right_earend": "R_ear", "tail_end": "tail_end",
       "front_left_paw": "L_paw_end", "front_right_paw": "R_paw_end", "back_left_paw": "L_foot", "back_right_paw": "R_foot",
       "neck_base": "neck", "back_middle": "body_middle", "tail_base": "tail_root",
       "front_left_knee": "L_elbow", "front_right_knee": "R_elbow", "back_left_knee": "L_knee", "back_right_knee": "R_knee"}
# TopViewMouse names = behavior-lab pose/predictors/joints.py _SA_TOPVIEWMOUSE; the _eartip variant swaps the ear point
MAP_TVM = {"nose": "nose", "left_ear": "L_ear", "right_ear": "R_ear", "tail_end": "tail_end", "neck": "neck", "tail_base": "tail_root"}
MAPS = {"superanimal_quadruped": MAP, "superanimal_topviewmouse": MAP_TVM,
        "superanimal_topviewmouse_eartip": {**{k: v for k, v in MAP_TVM.items() if "ear" not in k}, "left_ear_tip": "L_ear", "right_ear_tip": "R_ear"}}
POINTS = {"ext8": ["nose", "L_ear", "R_ear", "tail_end", "L_paw_end", "R_paw_end", "L_foot", "R_foot"],
          "head4": ["nose", "L_ear", "R_ear", "tail_end"]}
SWAP = {"L_paw_end": "R_paw_end", "R_paw_end": "L_paw_end", "L_foot": "R_foot", "R_foot": "L_foot"}


def label3d_cams(params: list) -> list[dict]:
    """Label3D stores K and r transposed (row-vector convention): K = K.T, R = r.T."""
    cams = []
    for p in params:
        rd, td = np.ravel(p["RDistort"]), np.ravel(p["TDistort"])
        R, t = np.asarray(p["r"], float).T, np.asarray(p["t"], float).reshape(3, 1)
        cams.append(dict(K=np.asarray(p["K"], float).T, D=np.array([rd[0], rd[1], td[0], td[1], rd[2]]), R=R, t=t, P=np.hstack([R, t])))
    return cams


def project(X: np.ndarray, cam: dict) -> np.ndarray:
    """(..., 3) world mm -> (..., 2) distorted px; NaN stays NaN."""
    flat = X.reshape(-1, 3)
    out = np.full((len(flat), 2), np.nan)
    ok = np.isfinite(flat).all(1)
    if ok.any():
        uv, _ = cv2.projectPoints(flat[ok].reshape(-1, 1, 3), cv2.Rodrigues(cam["R"])[0], cam["t"], cam["K"], cam["D"])
        out[ok] = uv.reshape(-1, 2)
    return out.reshape(X.shape[:-1] + (2,))


def depth(X: np.ndarray, cam: dict) -> np.ndarray:
    return (X @ cam["R"].T + cam["t"].ravel())[..., 2]


def cluster_ci(diff: np.ndarray, frame_of: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    """Median of paired differences; 95 % CI by resampling frames (all cameras and joints of a frame together)."""
    frames = np.unique(frame_of)
    groups = [diff[frame_of == f] for f in frames]
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return [float(np.median(diff)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label3d", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path, help="DANNCE A' save_data_AVG*.mat")
    ap.add_argument("--gate-json", required=True, type=Path, help="gate_A_scratch.json (reproduction check)")
    ap.add_argument("--sa", action="append", required=True, help="label=model:dir (DLC video h5 or run_dlc.py CSV)")
    ap.add_argument("--dlt-for", action="append", default=[], help="SA label that also gets a DLT->2D row")
    ap.add_argument("--points", choices=list(POINTS), default="ext8")
    ap.add_argument("--tag", default="kp-err2d[m2·hand91·ext8·superanimal_quadruped]")
    ap.add_argument("--skeleton", required=True, type=Path, help="mouse_22.yaml")
    ap.add_argument("--pcut", type=float, default=0.5, help="secondary report only")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    names = yaml.safe_load(a.skeleton.read_text())["keypoint_names"]
    assert len(names) == 22
    L = sio.loadmat(str(a.label3d), simplify_cells=True)
    lab = L["labelData"]
    frames = np.asarray(lab[0]["data_frame"]).astype(int).ravel()
    order = np.argsort(frames)
    frames = frames[order]
    sid = np.asarray(lab[0]["data_sampleID"]).astype(int).ravel()[order]
    gt3 = np.asarray(lab[0]["data_3d"], float)[order].reshape(len(frames), 22, 3)
    cams = label3d_cams(L["params"])
    n_cam, n_f = len(cams), len(frames)

    gt2 = np.stack([project(gt3, c) for c in cams])  # (cam, frame, joint, 2)
    for c in range(n_cam):
        lab2 = np.asarray(lab[c]["data_2d"], float)[order].reshape(n_f, 22, 2)
        med = float(np.nanmedian(np.linalg.norm(gt2[c] - lab2, axis=-1)))
        assert med <= 5.0, f"cam{c + 1} calibration self-check {med:.2f} px"
        print(f"[check] cam{c + 1} hand-label projection vs Label3D 2D median {med:.2f} px")

    m = sio.loadmat(str(a.pred))
    pr = np.transpose(np.asarray(m["pred"], float), (0, 2, 1))
    pidx = {s: i for i, s in enumerate(np.asarray(m["sampleID"]).astype(int).ravel())}
    base3 = np.full_like(gt3, np.nan)
    for k, s in enumerate(sid):
        if s in pidx:
            base3[k] = pr[pidx[s]]
    ext_i = [names.index(n) for n in POINTS["ext8"]]
    e3 = np.linalg.norm(base3 - gt3, axis=-1)[:, ext_i]
    want = json.loads(a.gate_json.read_text())["ext8_mean_mm"]
    assert abs(np.nanmean(e3) - want) < 1e-6, (np.nanmean(e3), want)
    print(f"[check] A' ext8 3D mean {np.nanmean(e3):.4f} mm = gate json {want:.4f}; predicted frames {int(np.isfinite(base3[:, 0, 0]).sum())}/{n_f}")

    # SuperAnimal detections, rows in the sorted label-frame order (video packing order = sorted PNG names)
    src = {}
    for spec in a.sa:
        label, rest = spec.split("=", 1)
        model, path = rest.split(":", 1)
        mp, d = MAPS[model], Path(path)
        sa = np.full((n_cam, n_f, 22, 2), np.nan)
        lik = np.full((n_cam, n_f, 22), np.nan)
        nobox = np.zeros((n_cam, n_f), bool)
        for c in range(n_cam):
            h5 = glob.glob(str(d / f"cam{c + 1}_superanimal_*.h5"))
            if h5:
                (h5,) = h5
                df = pd.read_hdf(h5)
                df = df.xs(df.columns.get_level_values("individuals")[0], axis=1, level="individuals")
                df.columns = df.columns.droplevel(0)
                col = lambda kp, co, df=df: df[(kp, co)].to_numpy()
            else:
                df = pd.read_csv(d / f"{model.replace('_eartip', '')}_cam{c + 1}.csv", index_col=0)
                col = lambda kp, co, df=df: df[f"{kp}_{co}"].to_numpy()
            assert len(df) == n_f, (d, c, len(df), n_f)
            nobox[c] = (df.to_numpy() == -1).all(1)
            for j_sa, j_m in mp.items():
                jm = names.index(j_m)
                sa[c, :, jm] = np.stack([col(j_sa, "x"), col(j_sa, "y")], -1)
                lik[c, :, jm] = col(j_sa, "likelihood")
            sa[c, nobox[c]] = np.nan
            lik[c, nobox[c]] = np.nan
        src[label] = dict(sa=sa, lik=lik, nobox=nobox, map=mp)
        print(f"[check] {label}: no-box frames per cam {nobox.sum(1).tolist()}")
    dlt_label = {lb: lb[:-3] + "_dlt2d" if lb.endswith("_2d") else lb + "_dlt2d" for lb in a.dlt_for}
    rows = ["base"]
    for lb in src:
        rows.append(lb)
        if lb in dlt_label:
            rows.append(dlt_label[lb])
    sa_rows = [r for r in rows[1:] if r in src]
    pairs = [(r, "base") for r in rows[1:]] + list(zip(rows[1:-1], rows[2:]))
    pts = POINTS[a.points]
    ext_i = [names.index(n) for n in pts]
    joints = list(dict.fromkeys(v for s_ in src.values() for v in s_["map"].values()))

    def dlt_rows(sa: np.ndarray, lik: np.ndarray, mp: dict, pcut: float | None) -> np.ndarray:
        out = np.full((n_cam, n_f, 22, 2), np.nan)
        cam_list = [dict(K=c["K"], D=c["D"], P=c["P"]) for c in cams]
        for f in range(n_f):
            for j_m in mp.values():
                jm = names.index(j_m)
                obs = [(c, sa[c, f, jm]) for c in range(n_cam)
                       if np.isfinite(sa[c, f, jm]).all() and (pcut is None or lik[c, f, jm] >= pcut)]
                if len(obs) >= 2:
                    X = triangulate(obs, cam_list)
                    for c in range(n_cam):
                        out[c, f, jm] = project(X, cams[c])
        return out

    def score(pcut: float | None) -> dict:
        pred = {"base": np.stack([project(base3, c) for c in cams])}
        for lb, s_ in src.items():
            sa2 = s_["sa"].copy()
            if pcut is not None:
                sa2[s_["lik"] < pcut] = np.nan
            pred[lb] = sa2
            if lb in dlt_label:
                pred[dlt_label[lb]] = dlt_rows(s_["sa"], s_["lik"], s_["map"], pcut)
        ROWS = rows
        P_ = a.points
        err = {r: np.linalg.norm(pred[r] - gt2, axis=-1) for r in ROWS}  # (cam, frame, joint) px
        Z = np.stack([depth(gt3, c) for c in cams])
        fx = np.array([c["K"][0, 0] for c in cams])[:, None, None]
        gt_ok = np.isfinite(gt2).all(-1)
        res = {f"failure_rate_{P_}": {r: float(np.mean(~np.isfinite(err[r][:, :, ext_i])[gt_ok[:, :, ext_i]])) for r in ROWS}}
        common = gt_ok & np.logical_and.reduce([np.isfinite(err[r]) for r in ROWS])
        cmask = common[:, :, ext_i]
        cam_of, frame_of, _ = np.nonzero(cmask)
        res[f"n_common_{P_}"] = int(cmask.sum())
        res["n_common_frames"] = int(len(np.unique(frame_of)))
        for unit, scale in (("px", 1.0), ("mm_eq", Z / fx)):
            e = {r: (err[r] * scale)[:, :, ext_i][cmask] for r in ROWS}
            res[f"median_{P_}_{unit}"] = {r: float(np.median(e[r])) for r in ROWS}
            res[f"per_cam_median_{P_}_{unit}"] = {r: [float(np.median(e[r][cam_of == c])) for c in range(n_cam)] for r in ROWS}
            rng = np.random.default_rng(0)
            res[f"paired_diff_{unit}"] = {f"{x}-{y}": cluster_ci(e[x] - e[y], frame_of, rng, a.n_boot)
                                          for x, y in pairs}
        res["per_joint_median_px"] = {r: {names[j]: float(np.median(err[r][:, :, j][common[:, :, j]])) if common[:, :, j].any() else None
                                          for j in [names.index(n) for n in joints]} for r in ROWS}
        for lb in sa_rows:
            sw = {}
            for n, o in SWAP.items():
                if n not in pts:
                    continue
                j, jo = names.index(n), names.index(o)
                own = err[lb][:, :, j]
                other = np.linalg.norm(pred[lb][:, :, j] - gt2[:, :, jo], axis=-1)
                ok = np.isfinite(own) & np.isfinite(other)
                sw[n] = {"swap_fraction": float(np.mean(other[ok] < own[ok])), "median_min_px": float(np.median(np.minimum(own, other)[ok]))}
            if sw:
                res[f"lr_swap_{lb}"] = sw
        return res, pred

    a.out.parent.mkdir(parents=True, exist_ok=True)
    first = next(iter(src.values()))
    out = {"tag": a.tag, "n_frames": n_f, "n_cams": n_cam, "nobox_per_cam": first["nobox"].sum(1).tolist()}
    if len(src) > 1:
        out["nobox_per_cam_by_row"] = {lb: s_["nobox"].sum(1).tolist() for lb, s_ in src.items()}
    out["main"], pred = score(None)
    out[f"pcut_{a.pcut}"], _ = score(a.pcut)
    np.savez_compressed(a.out.with_suffix(".npz"), frames=frames, names=np.array(names), gt2=gt2, lik=first["lik"],
                        rows=np.array(rows), **pred)
    a.out.write_text(json.dumps(out, indent=2))
    p_ = a.points
    print(json.dumps({k: out["main"][k] for k in (f"n_common_{p_}", "n_common_frames", f"failure_rate_{p_}", f"median_{p_}_px", f"median_{p_}_mm_eq", "paired_diff_mm_eq")}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
