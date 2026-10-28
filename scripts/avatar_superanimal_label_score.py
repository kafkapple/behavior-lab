"""AVATAR hand labels (SUBTLE SLEAP pkg, 450 frames) vs zero-shot SuperAnimal and SLEAP 1423 val predictions, 2D.

Pre-registered in vault Behavior-Lab/_Notes/261028_AVATAR_label_qc_clean_subset_stub.md. Labels: visible points only.
Known label errors (ears only, 261013): no dark fur within 12 px (on_body.csv near_dark False) or ear-neck1 distance /
per-video median body length > 1.0. Sets: all, clean (flagged ear points removed), flagged (only those points).
Paired parts use compare._distance (1:1, min(straight, crossed) / 2); a part with one label point left is scored
as the nearest predicted point of that part and reported apart ("single"). No likelihood cut in the main table.

Self-checks (asserts): 450 images, prediction CSV rows = image list, every flagged point is an ear point.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavior_lab.pose.html_common import page
from behavior_lab.pose.predictors.compare import _distance
from behavior_lab.pose.predictors.joints import PAIRED, PART_MAP, PARTS
from behavior_lab.pose.predictors.styles import FLAGGED_RING, LABEL_RING, legend_html, style
from behavior_lab.pose.slp_labels import load_predictions, load_slp

SA_Q = PART_MAP["superanimal_quadruped"]
SA_Q_BASE = {**{k: v for k, v in SA_Q.items() if v not in ("ear", "neck")},
             "left_earbase": "ear", "right_earbase": "ear", "neck_end": "neck"}
LABEL = PART_MAP["sleap_1423"]


def session(src: str) -> str:
    return "Ptd005" if "Ptd005" in src else "Sci1"


def load_labels(path: Path) -> dict:
    """image -> {src, camera, frame, split, kp: {name: (x, y)}} with visible points only."""
    out: dict = {}
    for r in csv.DictReader(open(path)):
        if "Ptd005" not in r["source_video"] and "Sci 1" not in r["source_video"]:
            continue
        d = out.setdefault(r["image"], {"src": r["source_video"].split("/")[-1], "camera": r["camera"], "frame": int(r["frame"]),
                                        "split": r["split"], "kp": {}})
        if r["visible"] == "1" and r["x_px"]:
            d["kp"][r["keypoint"]] = (float(r["x_px"]), float(r["y_px"]))
    return out


def ear_flags(lab: dict, on_body: Path) -> set[tuple[str, str]]:
    """(image, keypoint) of ear points flagged by either 261013 measure."""
    flags = {(r["image"], r["keypoint"]) for r in csv.DictReader(open(on_body))
             if r["image"] in lab and r["keypoint"].startswith("ear") and r["near_dark"] == "False"}
    body = collections.defaultdict(list)
    for im, d in lab.items():
        if {"nose1", "tailstart1"} <= set(d["kp"]):
            body[d["src"]].append(np.hypot(*np.subtract(d["kp"]["nose1"], d["kp"]["tailstart1"])))
    med = {s: float(np.median(v)) for s, v in body.items()}
    for im, d in lab.items():
        for ear in ("earL1", "earR1"):
            if ear in d["kp"] and "neck1" in d["kp"] and np.hypot(*np.subtract(d["kp"][ear], d["kp"]["neck1"])) / med[d["src"]] > 1.0:
                flags.add((im, ear))
    assert all(k.startswith("ear") for _, k in flags)
    return flags, med


def parts_of(points: dict[str, tuple], mapping: dict[str, str]) -> dict[str, np.ndarray]:
    out = collections.defaultdict(list)
    for name, xy in points.items():
        if name in mapping:
            out[mapping[name]].append(xy)
    return {p: np.array(v, float) for p, v in out.items()}


def sa_points(csv_path: Path, images: list[str], pcut: float | None) -> dict[str, dict[str, tuple]]:
    df = pd.read_csv(csv_path, index_col=0)
    assert len(df) == len(images), (csv_path, len(df), len(images))
    names = [c[:-2] for c in df.columns if c.endswith("_x")]
    out = {}
    for im, (_, row) in zip(images, df.iterrows()):
        out[im] = {n: (row[f"{n}_x"], row[f"{n}_y"]) for n in names
                   if row[f"{n}_x"] >= 0 and (pcut is None or row[f"{n}_likelihood"] >= pcut)}
    return out


def score(lab: dict, pred: dict, mapping: dict, flags: set, med: dict, which: str) -> list[dict]:
    """One row per (image, part) with label points of the requested set."""
    rows = []
    for im, d in lab.items():
        if im not in pred:
            continue
        kp = {k: v for k, v in d["kp"].items() if ((im, k) in flags) == (which == "flagged") or which == "all"}
        if which == "flagged" and not kp:
            continue
        L, P = parts_of(kp, LABEL), parts_of(pred[im], mapping)
        for part, a in L.items():
            if part not in mapping.values():
                continue  # the model has no such point: not a failure
            need = PAIRED.get(part, 1)
            b = P.get(part, np.zeros((0, 2)))
            if len(a) == need:
                err = _distance(a, b, part) if len(b) == need else None
                kind = "pair" if need == 2 else "single_part"
            else:
                err = float(np.min(np.hypot(*(b - a[0]).T))) if len(b) else None
                kind = "half_pair"
            rows.append({"image": im, "camera": d["camera"], "session": session(d["src"]), "part": part, "kind": kind,
                         "err_px": err, "err_body": None if err is None else err / med[d["src"]]})
    return rows


def summary(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    ok = df.err_px.notna()

    def agg(g: pd.DataFrame) -> dict:
        e = g.err_px.dropna()
        return {"n": int(len(g)), "fail": round(float(g.err_px.isna().mean()), 3),
                "median_px": round(float(e.median()), 2) if len(e) else None,
                "median_body": round(float(g.err_body.dropna().median()), 3) if len(e) else None}

    main = df[df.kind != "half_pair"]
    return {"all_parts": agg(main), "by_part": {p: agg(g) for p, g in main.groupby("part")},
            "by_camera": {c: agg(g) for c, g in main.groupby("camera")}, "by_session": {s: agg(g) for s, g in main.groupby("session")},
            "half_pair": agg(df[df.kind == "half_pair"]) if (df.kind == "half_pair").any() else None, "n_scored": int(ok.sum())}


def overlay(lab: dict, preds: dict, maps: dict, picks: list[str], img_dir: Path, out: Path, flags: set) -> None:
    rows = list(preds)
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(2.6 * len(picks), 2.6 * len(rows)), squeeze=False)
    for c, im in enumerate(picks):
        img = plt.imread(img_dir / im)
        d = lab[im]
        xy = np.array(list(d["kp"].values()))
        x0, y0 = xy.min(0) - 60
        x1, y1 = xy.max(0) + 60
        for r, key in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xlim(max(x0, 0), min(x1, img.shape[1]))
            ax.set_ylim(min(y1, img.shape[0]), max(y0, 0))
            for k, (x, y) in d["kp"].items():
                if k in LABEL:
                    ring = FLAGGED_RING if (im, k) in flags else LABEL_RING
                    ax.scatter(x, y, s=34, facecolors="none", edgecolors=ring.color, linewidths=1.6)
                    ax.scatter(x, y, s=50, facecolors="none", edgecolors="black", linewidths=0.6)
            p = preds[key].get(im)
            if p is None:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", color="white", fontsize=12)
            else:
                pts = np.array([v for n, v in p.items() if n in maps[key]]).reshape(-1, 2)
                st = style(key)
                ax.scatter(*pts.T, s=46, marker=st.marker, color=st.color, edgecolors="black", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{d['camera']} · {session(d['src'])}", fontsize=8)
            if c == 0:
                ax.set_ylabel(style(key).label, fontsize=7)
    fig.suptitle("Fig. A1 · white ring = hand label, yellow ring = flagged label error, marker = method (definitions box)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="holds pkg/{labels_long.csv,on_body.csv,images/}, sa/, slp/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sa-dir", default="sa", help="folder under --root with the SuperAnimal CSVs (post-hoc bbox runs live elsewhere)")
    a = ap.parse_args()
    R = a.root
    lab = load_labels(R / "pkg/labels_long.csv")
    assert len(lab) == 450, len(lab)
    flags, med = ear_flags(lab, R / "pkg/on_body.csv")
    all_img = [str(Path(p).relative_to(R / "pkg")) for p in (R / a.sa_dir / "images_all.txt").read_text().split()]
    bot_img = [str(Path(p).relative_to(R / "pkg")) for p in (R / a.sa_dir / "images_bot.txt").read_text().split()]
    assert sorted(all_img) == sorted(lab), "image list differs from labels"

    # labels_long `frame` is the pkg index, not the video frame: match val label frames to images by content (261013 rule)
    # the split files' video table is shifted (261013), so the video name is ignored: nose1 coordinates only
    by_xy = {(round(d["kp"]["nose1"][0], 2), round(d["kp"]["nose1"][1], 2)): im for im, d in lab.items() if "nose1" in d["kp"]}
    by_key = {}
    gt_val = load_slp(R / "slp/labels_gt.val.0.slp")
    nodes = gt_val["nodes"]
    assert set(nodes) == set(LABEL), nodes
    for f in gt_val["frames"]:
        if f["pts"][0, 2] == 1:
            im = by_xy.get((round(f["pts"][0, 0], 2), round(f["pts"][0, 1], 2)))
            if im is not None:
                by_key[(f["src"], f["frame_idx"])] = im
    print(f"[check] SLEAP val AVATAR frames matched to images by nose1 coordinates: {len(by_key)}")
    assert by_key, "no val frame matched"
    sleap = {}
    for p in load_predictions(R / "slp/labels_pr.val.0.slp"):
        im = by_key.get((p["src"], p["frame_idx"]))
        if im is not None:
            sleap[im] = {n: (x, y) for n, (x, y, _) in zip(nodes, p["pts"]) if np.isfinite(x)}  # file node order
    maps = {"sa_quad": SA_Q, "sa_quad_base": SA_Q_BASE, "sleap1423_val": LABEL, "sa_tvm_bot": PART_MAP["superanimal_topviewmouse"]}
    out: dict = {"n_images": len(lab), "n_flagged_points": len(flags), "n_sleap_val_images": len(sleap),
                 "n_table": {f"{c}|{s}": n for (c, s), n in collections.Counter((d["camera"], session(d["src"])) for d in lab.values()).items()},
                 "flags_table": {f"{c}|{s}": n for (c, s), n in collections.Counter((lab[i]["camera"], session(lab[i]["src"])) for i, _ in flags).items()}}
    for tag, pcut in (("main", None), ("pcut_0.5", 0.5)):
        q = sa_points(R / a.sa_dir / "superanimal_quadruped_all.csv", all_img, pcut)
        t = sa_points(R / a.sa_dir / "superanimal_topviewmouse_bot.csv", bot_img, pcut)
        preds = {"sa_quad": q, "sa_quad_base": q, "sleap1423_val": sleap, "sa_tvm_bot": t}
        out[tag] = {k: {w: summary(score(lab, v, maps[k], flags, med, w)) for w in ("all", "clean", "flagged")} for k, v in preds.items()}
        # same-frame comparison: SA-Quadruped restricted to the SLEAP val images
        out[tag]["sa_quad_on_val"] = {w: summary(score({i: lab[i] for i in sleap}, q, SA_Q, flags, med, w)) for w in ("all", "clean")}
        if tag == "main":
            main_preds = preds
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "score_avatar.json").write_text(json.dumps(out, indent=2))
    rng = np.random.default_rng(0)
    clean_pool = [i for i in sleap if not any((i, k) in flags for k in lab[i]["kp"])]
    flag_pool = sorted({i for i, _ in flags})
    bots = [i for i in clean_pool if lab[i]["camera"] == "bot"][:2]
    sides = [i for i in clean_pool if lab[i]["camera"] != "bot"][:2]
    picks = bots + sides + list(rng.choice(flag_pool, 2, replace=False))
    overlay(lab, main_preds, maps, picks, R / "pkg", a.out / "fig_a1_overlay.png", flags)

    def tr(key: str, w: str) -> str:
        s = out["main"][key][w].get("all_parts") if out["main"][key][w] else None
        return "" if not s else f"<tr><td><span style='color:{style(key).color}'>■</span> {style(key).label}</td><td>{w}</td><td class=c>{s['n']}</td><td class=c>{s['fail']:.3f}</td><td class=c>{s['median_px']}</td><td class=c>{s['median_body']}</td></tr>"

    head = "<tr><th>method</th><th>set</th><th class=c>n (image·part)</th><th class=c>실패율</th><th class=c>중앙값 px</th><th class=c>몸 길이 비</th></tr>"
    rows_html = "".join(tr(k, w) for k in ("sa_quad", "sa_quad_base", "sa_tvm_bot", "sleap1423_val") for w in ("all", "clean", "flagged"))
    val_html = "".join(f"<tr><td>{w}</td><td class=c>{out['main']['sa_quad_on_val'][w]['all_parts']['median_px']}</td><td class=c>{out['main']['sleap1423_val'][w]['all_parts']['median_px']}</td></tr>" for w in ("all", "clean"))
    parts_html = "".join(f"<tr><td>{p}</td>" + "".join(f"<td class=c>{(out['main'][k]['clean']['by_part'].get(p) or {}).get('median_px', '')}</td>" for k in ("sa_quad", "sa_quad_base", "sleap1423_val")) + "</tr>" for p in PARTS)
    cam_html = "".join(f"<tr><td>{c}</td><td class=c>{v['n']}</td><td class=c>{v['fail']:.3f}</td><td class=c>{v['median_px']}</td></tr>" for c, v in out["main"]["sa_quad"]["clean"]["by_camera"].items())
    body = f"""<h1>AVATAR 손 라벨 대 SuperAnimal zero-shot</h1>
<div class=card><b>목표</b> 옆 · 비스듬한 rig 의 첫 학습 0 검출기 (SA-Quadruped, 결정 a) 를 AVATAR 손 라벨로 잰다.<br>
<b>정의</b> 라벨 = SUBTLE SLEAP pkg AVATAR {out['n_images']} 장, visible 점만 · 귀 오류 표시 {out['n_flagged_points']} 점 (261013 측정) · clean = 표시 점 제외 · flagged = 표시 점만 · 쌍 부위 = 1:1 배정 · 몸 길이 비 = 오차 ÷ 영상별 nose1–tailstart1 중앙값 · SLEAP val = 학습에 안 쓰인 {out['n_sleap_val_images']} 장<br>
<b>색</b> {legend_html(["sa_quad", "sa_quad_base", "sleap1423_val", "sa_tvm_bot"])}</div>
<h2>전체 · clean · flagged</h2><p class=lead>모든 부위를 합친 중앙값이다. flagged 는 라벨 쪽 오류라 큰 값이 정상이다.</p><table>{head}{rows_html}</table>
<h2>같은 frame 비교 (SLEAP val 장)</h2><p class=lead>학습에 안 쓰인 val 장에서 SA-Quadruped 와 SLEAP 1423 의 중앙값 px.</p><table><tr><th>set</th><th class=c>SA-Quadruped</th><th class=c>SLEAP 1423</th></tr>{val_html}</table>
<h2>부위별 (clean, 중앙값 px)</h2><p class=lead>귀 · 목은 대응 정의에 따라 값이 바뀐다.</p><table><tr><th>part</th><th class=c>SA-Q (end)</th><th class=c>SA-Q (base)</th><th class=c>SLEAP val</th></tr>{parts_html}</table>
<h2>cam 별 (SA-Quadruped, clean)</h2><p class=lead>bot 과 side 를 나눠 본다.</p><table><tr><th>cam</th><th class=c>n</th><th class=c>실패율</th><th class=c>중앙값 px</th></tr>{cam_html}</table>
<h2>방법별 overlay</h2><p class=lead>열 = 이미지 (clean 4 장 + 라벨 오류 2 장), 행 = 방법, 모든 칸이 같은 crop 이다.</p><img src="fig_a1_overlay.png" style="width:100%">
<h2>한계</h2><p class=mut>라벨은 학습용으로 고른 frame 이고 몸 밖 오류만 걸렀다. 몇 px 어긋난 라벨은 남는다. 단일 시점이라 라벨 오차 크기는 재라벨 없이 모른다.</p>"""
    (a.out / "index.html").write_text(page("AVATAR SuperAnimal vs hand labels", body, "", {}))
    print(json.dumps({k: out["main"][k]["clean"].get("all_parts") for k in ("sa_quad", "sa_quad_base", "sleap1423_val", "sa_tvm_bot")}, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
