"""Dashboard page for a SLEAP label set shipped with a model: provenance, definition, checks, label-only overlay."""
from __future__ import annotations

from pathlib import Path

from .html_common import page

EXTRA_CSS = "canvas{max-width:100%;border:1px solid var(--line);background:var(--bg)}#leg{font-size:12px}"
JS = """const D=__DATA__;const COL=["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabebe","#469990","#9a6324"];
const srcs=[...new Set(D.frames.map(f=>f.src))];const ssel=document.getElementById('ssel'),fsel=document.getElementById('fsel'),all=document.getElementById('all'),cv=document.getElementById('cv'),x=cv.getContext('2d');
srcs.forEach(s=>ssel.add(new Option(s+' ('+D.frames.filter(f=>f.src===s).length+')',s)));
document.getElementById('leg').innerHTML=D.nodes.map((n,i)=>`<div style="color:${COL[i%COL.length]}">${i} ${n}</div>`).join('');
function list(){return D.frames.filter(f=>f.src===ssel.value)}
function fill(){fsel.replaceChildren(...list().map((f,i)=>new Option(f.split+' frame '+f.frame,i)));draw()}
function drawOne(f,a,sc){D.edges.forEach(([i,j])=>{x.strokeStyle='rgba(120,120,120,'+a+')';x.lineWidth=1.5;x.beginPath();x.moveTo(f.pts[i][0]*sc,f.pts[i][1]*sc);x.lineTo(f.pts[j][0]*sc,f.pts[j][1]*sc);x.stroke()});
 f.pts.forEach((p,i)=>{x.globalAlpha=a;x.strokeStyle=x.fillStyle=COL[i%COL.length];x.beginPath();x.arc(p[0]*sc,p[1]*sc,4,0,7);p[2]?x.fill():x.stroke();x.globalAlpha=1})}
function draw(){const L=list(),f=L[+fsel.value||0];const sc=Math.min(1,700/f.wh[0]);cv.width=f.wh[0]*sc;cv.height=f.wh[1]*sc;x.strokeStyle='#888';x.strokeRect(0,0,cv.width,cv.height);
 if(all.checked)L.forEach(g=>drawOne(g,0.25,sc));drawOne(f,1,sc);x.font='bold 11px sans-serif';f.pts.forEach((p,i)=>{x.fillStyle=COL[i%COL.length];x.fillText(i,p[0]*sc+6,p[1]*sc-5)});
 document.getElementById('info').textContent=`image size ${f.wh[0]} x ${f.wh[1]} px, filled dot = visible, ring = occluded but located`}
ssel.onchange=fill;fsel.onchange=draw;all.onchange=draw;fill();"""


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def build(analysis: dict, out: str | Path, provenance: list[str], image_routes: list[tuple[str, str]], metrics: dict | None = None,
          title: str = "SLEAP training labels") -> Path:
    """`analysis` = slp_labels.analyze(...); `provenance` = bullet lines; `image_routes` = (route, state) rows; `metrics` optional."""
    a = analysis
    N, r, c = a["nodes"], a["label_rates"], a["counts"]
    n_av = a["avatar_counts"]["train"] + a["avatar_counts"]["val"]
    err = a.get("val_err_px_median", {})
    rate_rows = "".join(
        f"<tr><td>{i}</td><td>{n}</td>" + (f"<td class=c>{_pct(r['avatar']['visible'][i])}</td><td class=c>{_pct(r['avatar']['occluded'][i])}</td>" if r["avatar"] else "<td></td><td></td>")
        + f"<td class=c>{_pct(r['other']['visible'][i])}</td><td class=c>{err['avatar'][i] if err.get('avatar') else ''}</td><td class=c>{err['all'][i] if err.get('all') else ''}</td></tr>"
        for i, n in enumerate(N))
    oob_rows = "".join(f"<tr><td>{k}</td><td class=c>{v[1]}</td><td class=c>{v[0]}</td><td class=c>{v[0] / max(v[1], 1) * 100:.1f} %</td></tr>" for k, v in a["oob_by_size"].items())
    lr_rows = "".join(
        f"<tr><td>{src}</td>" + "".join(f"<td class=c>{_pct(d['all']['left_positive'])} / {_pct(d['visible_only']['left_positive']) if d['visible_only']['left_positive'] is not None else '-'}</td>" for d in v.values())
        + f"<td class=c>{next(iter(v.values()))['all']['n']}</td></tr>" for src, v in a["lr_bottom_view"].items())
    lr_head = "".join(f"<th>{p}</th>" for p in next(iter(a["lr_bottom_view"].values()))) if a["lr_bottom_view"] else ""
    leak = a["val_with_train_within"]
    mt = ""
    if metrics:
        mt = "<h2>Reported model metrics</h2><ul>" + "".join(f"<li>{k}: {v}</li>" for k, v in metrics.items()) + "</ul>"
    body = f"""<h1>{title}</h1>
<p class="lead">{c['train']} train + {c['val']} val frames parsed ({a['skipped']['val']} val frames reference a missing video); no images, so every check uses coordinates only.</p>
<h2>Provenance</h2><ul>{"".join(f"<li>{x}</li>" for x in provenance)}</ul>
<h2>Composition</h2><p class="lead">{n_av} of {c['train'] + c['val']} labelled frames are on AVATAR cells.</p>
<h2>Label definition</h2><p class="lead">{len(N)} keypoints: {", ".join(f"{i} {n}" for i, n in enumerate(N))}</p><ul><li>Edges (index pairs): {a['edges']}</li></ul>
<p class="lead">Left/right on the bottom view: share of frames where the L point is on one side of the tail-to-head axis (all frames / frames with all three points visible).</p>
<table><thead><tr><th>Source video</th>{lr_head}<th>Frames</th></tr></thead><tbody>{lr_rows}</tbody></table>
<p class="mut">A consistent convention gives a share near 100 %. Side views are ambiguous and are not tested.</p>
<h2>Label quality checks</h2>
<table><thead><tr><th>Idx</th><th>Keypoint</th><th>AVATAR visible</th><th>AVATAR occluded</th><th>Other visible</th><th>Val err AVATAR px</th><th>Val err all px</th></tr></thead><tbody>{rate_rows}</tbody></table>
<table><thead><tr><th>Embedded image size</th><th>Points</th><th>Outside the image</th><th>Share</th></tr></thead><tbody>{oob_rows}</tbody></table>
{mt}<h2>Leakage between train and val</h2><ul><li>Same video and frame: {leak['0']} of {a['val_n']}</li><li>Within 5 frames: {leak['5']}; within 30: {leak['30']}; within 100: {leak['100']}</li></ul>
<h2>Overlay of the labels</h2><p class="lead">Labels are drawn on an empty frame of the right size.</p>
<div class="card"><div style="margin-bottom:8px"><select id="ssel"></select> <select id="fsel"></select> <label><input type="checkbox" id="all"> all frames of this video</label></div>
<div class="vw"><canvas id="cv"></canvas><div id="leg"></div></div><div class="mut" id="info"></div></div>
<h2>Image problem</h2><table><thead><tr><th>Route</th><th>State</th></tr></thead><tbody>{"".join(f"<tr><td>{x}</td><td>{y}</td></tr>" for x, y in image_routes)}</tbody></table>"""
    html = page(title, body, JS, {"nodes": N, "edges": a["edges"], "frames": a["avatar_frames"]}, EXTRA_CSS)
    out = Path(out)
    out.write_text(html)
    return out
