"""Single-file keypoint labelling page for the AVATAR GT images. A human labels; predictions are never shown.

    python scripts/kp_label_page.py --template <gt_label dir>/labels_template.csv --out <gt_label dir>/label_page.html

Open the page from the folder that contains `images/` (relative image paths). Keys: click = place the current keypoint
(visible 1), Shift+click = place as occluded (visible 0), S = skip (blank, not labelled), Backspace = undo, arrows = image,
Z = zoom toggle. Work is autosaved in the browser (localStorage); "Export CSV" writes the template schema that
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
let cur = 0, zoom = false;
let draw = () => {};                  // replaced by the page below; a no-op under node (tests)

function nextKeypoint(img) {          // first keypoint of the image without an entry
  const s = state[img] || {};
  return SPEC.keypoints.find(k => !(k in s));
}
function csv() {
  const rows = [SPEC.columns.join(",")];
  for (const r of SPEC.rows) {
    const e = (state[r.image] || {})[r.keypoint];
    const lab = e ? [e[0].toFixed(1), e[1].toFixed(1), e[2]] : ["", "", ""];
    rows.push([r.frame, r.camera, r.image, r.image_w, r.image_h, r.keypoint, ...lab, ""].join(","));
  }
  return rows.join("\n") + "\n";
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function place(img, x, y, vis) {
  const k = nextKeypoint(img); if (!k) return;
  (state[img] = state[img] || {})[k] = [x, y, vis]; save(); draw();
}
function skip(img) { const k = nextKeypoint(img); if (!k) return; (state[img] = state[img] || {})[k] = null; save(); draw(); }
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
    const s = state[img.image] || {}, k = nextKeypoint(img.image);
    document.getElementById("next").textContent = k ? `next: ${k}` : "all keypoints done for this image";
    const n = Object.values(state).reduce((a, v) => a + Object.keys(v).length, 0);
    document.getElementById("count").textContent = `${n} / ${SPEC.rows.length} keypoints answered`;
    document.getElementById("legend").innerHTML = SPEC.keypoints.map((kp, i) => {
      const e = s[kp]; const st = e === undefined ? "" : e === null ? " (skipped)" : e[2] === 0 ? " (occluded)" : " ✓";
      return `<div style="color:${COL[i % COL.length]};${kp === k ? "font-weight:700" : ""}">${i + 1}. ${kp}${st}</div>`;
    }).join("");
    ctx.drawImage(im, 0, 0, cv.width, cv.height);
    SPEC.keypoints.forEach((kp, i) => {
      const e = s[kp]; if (!e) return;
      const x = e[0] * cv.width / img.image_w, y = e[1] * cv.height / img.image_h;
      ctx.strokeStyle = COL[i % COL.length]; ctx.fillStyle = COL[i % COL.length]; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); e[2] === 1 ? ctx.fill() : ctx.stroke();
      ctx.fillText(String(i + 1), x + 8, y - 8);
    });
  };
  function load() {
    const img = SPEC.images[cur];
    im.onload = () => {
      const w = zoom ? img.image_w : Math.min(img.image_w, 1100);
      cv.width = w; cv.height = Math.round(w * img.image_h / img.image_w); draw();
    };
    im.src = img.image;
  }
  cv.addEventListener("click", ev => {
    const r = cv.getBoundingClientRect(), img = SPEC.images[cur];
    place(img.image, (ev.clientX - r.left) * img.image_w / r.width, (ev.clientY - r.top) * img.image_h / r.height, ev.shiftKey ? 0 : 1);
  });
  document.addEventListener("keydown", ev => {
    const img = SPEC.images[cur].image;
    if (ev.key === "ArrowRight" && cur < SPEC.images.length - 1) { cur++; load(); }
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
    const ix = n => h.indexOf(n); state = {};
    for (const l of lines.slice(1)) {
      const c = l.split(","), img = c[ix("image")], k = c[ix("keypoint")];
      if (c[ix("visible")] !== "") (state[img] = state[img] || {})[k] = [+c[ix("x_px")], +c[ix("y_px")], +c[ix("visible")]];
    }
    save(); draw();
  };
  load();
}
if (typeof module !== "undefined") module.exports = {csv, place, skip, undo, nextKeypoint, get state() { return state; }, set state(v) { state = v; }};
"""

HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AVATAR keypoint labelling</title><style>
:root{--bg:#fff;--fg:#1c1f23;--mut:#5b6570;--line:#d6dbe0}
@media (prefers-color-scheme:dark){:root{--bg:#16191d;--fg:#e6e9ec;--mut:#9aa4ae;--line:#343a41}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{display:flex;gap:16px;padding:16px;flex-wrap:wrap}
canvas{border:1px solid var(--line);max-width:100%;max-height:92vh;cursor:crosshair}
aside{min-width:240px;position:sticky;top:16px;align-self:flex-start}.mut{color:var(--mut)}button,input{font:inherit}
</style></head><body><main><div><h1 style="font-size:16px;margin:0 0 8px" id="title"></h1><canvas id="cv"></canvas></div>
<aside><p id="next" style="font-weight:700"></p><p class="mut" id="count"></p><div id="legend"></div>
<p class="mut">click = visible, Shift+click = occluded (located), S = skip (blank), Backspace = undo, arrows = image, Z = zoom</p>
<p class="mut">No predictions are shown on purpose. Label the same keypoint the same way on every image.</p>
<p><button id="export">Export CSV</button></p><p>Import CSV <input id="import" type="file" accept=".csv"></p></aside></main>
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
