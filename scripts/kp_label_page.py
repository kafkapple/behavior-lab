"""Single-file keypoint labelling page for the AVATAR GT images. A human labels; predictions are never shown.

    python scripts/kp_label_page.py --template <gt_label dir>/labels_template.csv --out <gt_label dir>/label_page.html

Open the page from the folder that contains `images/` (relative image paths). Keys: click = place the current keypoint
(visible 1), Shift+click = place as occluded (visible 0), S = skip (blank, not labelled), Backspace = undo, arrows = image,
Z = zoom toggle. Click a name in the list to select that keypoint: the next click re-places it, X or Delete clears it,
Esc cancels. - and = change the marker size. A note per image goes into the `note` column of its first row. The side
panel shows the labeller's own last finished image of the same camera as a consistency reference (never a model output).
Work is autosaved in the browser (localStorage); "Export CSV" writes the template schema that
`behavior_lab.pose.labels.load_label_table` validates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from behavior_lab.pose.labels import COLUMNS

JS = r"""
const SPEC = __SPEC__;
const KEY = "kp_label_" + SPEC.id;
let state = {};                       // image -> {keypoint: [x, y, visible] | null (skipped)}
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
let notes = {}, rad = 3;
try { notes = JSON.parse(localStorage.getItem(KEY + "_notes") || "{}"); rad = +(localStorage.getItem(KEY + "_rad") || 3); } catch (e) {}
let cur = 0, zoom = false, sel = null;   // sel = keypoint picked in the list; the next click or S applies to it
let draw = () => {};                  // replaced by the page below; a no-op under node (tests)

function nextKeypoint(img) {          // first keypoint of the image without an entry
  const s = state[img] || {};
  return SPEC.keypoints.find(k => !(k in s));
}
function csv() {
  const rows = [SPEC.columns.join(",")], noted = new Set();
  for (const r of SPEC.rows) {
    const e = (state[r.image] || {})[r.keypoint];
    const lab = e ? [e[0].toFixed(1), e[1].toFixed(1), e[2]] : ["", "", ""];
    const note = noted.has(r.image) ? "" : (notes[r.image] || "").replace(/[,\n\r"]+/g, " ").trim();   // once per image, CSV-safe
    noted.add(r.image);
    rows.push([r.frame, r.camera, r.image, r.image_w, r.image_h, r.keypoint, ...lab, note].join(","));
  }
  return rows.join("\n") + "\n";
}
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); localStorage.setItem(KEY + "_notes", JSON.stringify(notes)); localStorage.setItem(KEY + "_rad", rad); } catch (e) {}
}
function select(k) { sel = k; draw(); }
function clearKp(img, k) { const s = state[img] || {}; if (k in s) { delete s[k]; } sel = null; save(); draw(); }
function setNote(img, t) { if (t) notes[img] = t; else delete notes[img]; save(); }
function place(img, x, y, vis) {
  const k = sel || nextKeypoint(img); if (!k) return;
  (state[img] = state[img] || {})[k] = [x, y, vis]; sel = null; save(); draw();
}
function skip(img) { const k = sel || nextKeypoint(img); if (!k) return; (state[img] = state[img] || {})[k] = null; sel = null; save(); draw(); }
function undo(img) {
  const s = state[img] || {}; const done = SPEC.keypoints.filter(k => k in s);
  if (done.length) { delete s[done[done.length - 1]]; save(); draw(); }
}
if (typeof document !== "undefined") {
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d"), im = new Image();
  const COL = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabebe","#469990","#9a6324"];
  draw = function () {
    const img = SPEC.images[cur];
    document.getElementById("title").textContent = `${cur + 1} / ${SPEC.images.length}  ${img.image}`;
    const s = state[img.image] || {}, k = sel || nextKeypoint(img.image);
    document.getElementById("next").textContent = sel ? `selected: ${sel} (click = re-place, X = clear, Esc = cancel)` : k ? `next: ${k}` : "all keypoints done for this image";
    const n = Object.values(state).reduce((a, v) => a + Object.keys(v).length, 0);
    document.getElementById("count").textContent = `${n} / ${SPEC.rows.length} keypoints answered`;
    document.getElementById("legend").innerHTML = SPEC.keypoints.map((kp, i) => {
      const e = s[kp]; const st = e === undefined ? "" : e === null ? " (skipped)" : e[2] === 0 ? " (occluded)" : " ✓";
      return `<div data-kp="${kp}" style="cursor:pointer;color:${COL[i % COL.length]};${kp === k ? "font-weight:700;text-decoration:underline" : ""}">${i + 1}. ${kp}${st}</div>`;
    }).join("");
    ctx.drawImage(im, 0, 0, cv.width, cv.height);
    points(ctx, s, cv.width / img.image_w, cv.height / img.image_h, rad, sel);
    reference();
  };
  function points(c, s, sx, sy, r, hot) {              // small, half-transparent markers so the pixel under a point stays visible
    SPEC.keypoints.forEach((kp, i) => {
      const e = s[kp]; if (!e) return;
      const x = e[0] * sx, y = e[1] * sy;
      c.strokeStyle = COL[i % COL.length]; c.fillStyle = COL[i % COL.length]; c.lineWidth = 1.5;
      c.globalAlpha = 0.55; c.beginPath(); c.arc(x, y, kp === hot ? r + 3 : r, 0, 7); e[2] === 1 && kp !== hot ? c.fill() : c.stroke();
      c.globalAlpha = 0.9; c.font = "10px system-ui"; c.fillText(String(i + 1), x + r + 3, y - r - 2); c.globalAlpha = 1;
    });
  }
  const rcv = document.getElementById("ref"), rctx = rcv.getContext("2d"), rim = new Image();
  function reference() {                               // the labeller's own nearest finished image of the same camera
    const img = SPEC.images[cur], done = i => SPEC.keypoints.every(k => k in (state[SPEC.images[i].image] || {}));
    const c = SPEC.images.map((x, i) => i).filter(i => i !== cur && SPEC.images[i].camera === img.camera && done(i))
      .sort((a, b) => Math.abs(a - cur) - Math.abs(b - cur))[0];
    const cap = document.getElementById("refcap");
    if (c === undefined) { rcv.style.display = "none"; cap.textContent = "기준 이미지 없음: 이 카메라에서 끝낸 이미지가 아직 없습니다."; return; }
    const o = SPEC.images[c]; cap.textContent = `내가 끝낸 같은 카메라 이미지: ${o.image}`; rcv.style.display = "";
    rim.onload = () => { rcv.width = 300; rcv.height = Math.round(300 * o.image_h / o.image_w); rctx.drawImage(rim, 0, 0, rcv.width, rcv.height);
      points(rctx, state[o.image], rcv.width / o.image_w, rcv.height / o.image_h, 2, null); };
    if (rim.dataset.src !== o.image) { rim.dataset.src = o.image; rim.src = o.image; } else rim.onload();
  }
  document.getElementById("legend").addEventListener("click", ev => { const k = ev.target.dataset.kp; if (k) select(k === sel ? null : k); });
  const note = document.getElementById("note");
  note.addEventListener("input", () => setNote(SPEC.images[cur].image, note.value));
  function load() {
    const img = SPEC.images[cur];
    im.onload = () => {
      const w = zoom ? img.image_w : Math.min(img.image_w, 1100);
      cv.width = w; cv.height = Math.round(w * img.image_h / img.image_w); sel = null; note.value = notes[img.image] || ""; draw();
    };
    im.src = img.image;
  }
  cv.addEventListener("click", ev => {
    const r = cv.getBoundingClientRect(), img = SPEC.images[cur];
    place(img.image, (ev.clientX - r.left) * img.image_w / r.width, (ev.clientY - r.top) * img.image_h / r.height, ev.shiftKey ? 0 : 1);
  });
  document.addEventListener("keydown", ev => {
    const img = SPEC.images[cur].image;
    if (ev.target === note) return;                    // typing a note must not trigger shortcuts
    if (ev.key === "Escape") select(null);
    else if ((ev.key === "x" || ev.key === "X" || ev.key === "Delete") && sel) clearKp(img, sel);
    else if (ev.key === "-" || ev.key === "=") { rad = Math.max(1, Math.min(8, rad + (ev.key === "=" ? 1 : -1))); save(); draw(); }
    else if (ev.key === "ArrowRight" && cur < SPEC.images.length - 1) { cur++; load(); }
    else if (ev.key === "ArrowLeft" && cur > 0) { cur--; load(); }
    else if (ev.key === "s" || ev.key === "S") skip(img);
    else if (ev.key === "Backspace") { ev.preventDefault(); undo(img); }
    else if (ev.key === "z" || ev.key === "Z") { zoom = !zoom; load(); }
  });
  document.getElementById("export").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv()], {type: "text/csv"})); a.download = "labels_" + SPEC.id + ".csv"; a.click();
  };
  document.getElementById("import").onchange = async ev => {
    const lines = (await ev.target.files[0].text()).trim().split("\n"), h = lines[0].split(",");
    const ix = n => h.indexOf(n); state = {}; notes = {};
    for (const l of lines.slice(1)) {
      const c = l.split(","), img = c[ix("image")], k = c[ix("keypoint")];
      if (c[ix("visible")] !== "") (state[img] = state[img] || {})[k] = [+c[ix("x_px")], +c[ix("y_px")], +c[ix("visible")]];
      if ((c[ix("note")] || "").trim()) notes[img] = c[ix("note")].trim();
    }
    save(); load();
  };
  load();
}
if (typeof module !== "undefined") module.exports = {csv, place, skip, undo, select, clearKp, setNote, nextKeypoint, get state() { return state; }, set state(v) { state = v; }};
"""

HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AVATAR keypoint labelling</title><style>
:root{--bg:#fff;--fg:#1c1f23;--mut:#5b6570;--line:#d6dbe0}
@media (prefers-color-scheme:dark){:root{--bg:#16191d;--fg:#e6e9ec;--mut:#9aa4ae;--line:#343a41}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
canvas{border:1px solid var(--line);max-width:100%;max-height:92vh;cursor:crosshair}
aside{width:320px;position:sticky;top:16px;align-self:flex-start;max-height:96vh;overflow:auto}details{margin:8px 0}summary{cursor:pointer;font-weight:700}
li{margin:2px 0}textarea{width:100%;box-sizing:border-box;font:inherit}#ref{border:1px solid var(--line);max-width:100%}.mut{color:var(--mut)}button,input{font:inherit}
</style></head><body><main><div><h1 style="font-size:16px;margin:0 0 8px" id="title"></h1><canvas id="cv"></canvas></div>
<aside><p id="next" style="font-weight:700"></p><p class="mut" id="count"></p><div id="legend"></div>
<p><textarea id="note" rows="2" placeholder="이 이미지의 메모 (예: 앞발 좌우 불확실)"></textarea></p>
<p><button id="export">Export CSV</button> <span class="mut">중간 저장용으로도 쓴다</span></p><p>Import CSV <input id="import" type="file" accept=".csv"></p>
<details open><summary>조작</summary><ul>
<li>클릭 = 보임 (visible 1)</li><li>Shift+클릭 = 가려졌지만 위치 추정 (visible 0)</li><li>S = 건너뜀 (빈칸)</li>
<li>Backspace = 마지막 점 되돌리기</li><li>목록의 이름 클릭 = 그 keypoint 선택. 다음 클릭이 그 점을 다시 찍는다</li>
<li>선택한 뒤 X 또는 Delete = 그 keypoint 만 지움. Esc = 선택 취소</li>
<li>방향키 = 이전, 다음 이미지. Z = 확대. - 와 = 는 점 크기</li></ul></details>
<details open><summary>라벨 규칙</summary><ul>
<li>한 사람이 모든 이미지를 찍는다</li><li>왼쪽, 오른쪽은 동물 자신의 기준이다</li>
<li>바닥 카메라는 배 쪽 시점이다. 머리 방향 기준 화면 오른쪽이 동물의 왼쪽이다</li>
<li>nose1 = 코끝. neck1 = 두 귀 사이 뒤쪽</li><li>earL1, earR1 = 귀 끝 (임시 규칙)</li>
<li>foreleg, hindleg = 발 (바닥에 닿는 끝)</li><li>tailstart1 = 꼬리 시작. tailend1 = 꼬리 끝</li>
<li>tail1 = 꼬리 시작과 꼬리 끝의 가운데 (임시 규칙)</li>
<li>첫 이미지 메모에 ear=tip tail1=mid 를 적는다</li><li>모델 출력과 selection.csv 는 보지 않는다</li></ul></details>
<details open><summary>잘 모르겠을 때</summary><ul>
<li>가려졌지만 몸의 모양으로 위치를 짐작할 수 있다: Shift+클릭</li>
<li>화면 밖이거나 어디인지 짐작할 수 없다: S 로 건너뜀. 억지로 찍지 않는다</li>
<li>좌우를 구분할 수 없다: 두 점 모두 S 로 건너뛰고 메모에 적는다</li>
<li>규칙 자체가 애매하다: 아래 기준 이미지와 같은 방식으로 찍고 메모에 적는다</li></ul></details>
<details open><summary>기준 이미지 (내 라벨)</summary><p class="mut" id="refcap"></p><canvas id="ref"></canvas></details>
<p class="mut">모델 출력은 일부러 보여 주지 않는다. 같은 keypoint 는 모든 이미지에서 같은 방식으로 찍는다.</p></aside></main>
<script>__JS__</script></body></html>
"""


def build(template: Path, out: Path) -> Path:
    df = pd.read_csv(template)
    missing = [c for c in COLUMNS[:-1] if c not in df.columns]
    if missing:
        raise ValueError(f"{template} lacks columns {missing}")
    filled = df[["x_px", "y_px", "visible"]].notna().any(axis=1).sum()
    if filled:
        raise ValueError(f"{template} already has {filled} labelled rows; the page starts from an empty template")
    keypoints = list(dict.fromkeys(df["keypoint"]))
    images = df.drop_duplicates("image")[["frame", "camera", "image", "image_w", "image_h"]]
    spec = {"id": template.parent.name, "columns": list(COLUMNS), "keypoints": keypoints,
            "images": images.to_dict("records"),
            "rows": df[["frame", "camera", "image", "image_w", "image_h", "keypoint"]].to_dict("records")}
    js = JS.replace("__SPEC__", json.dumps(spec))
    out.write_text(HTML.replace("__JS__", js))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(build(args.template, args.out))


if __name__ == "__main__":
    main()
