"""Self-contained comparison page of every keypoint predictor on one image set: a sortable, groupable table and a
multi-view overlay (all cameras at once, or one panel per model) with keypoint index and name labels."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pandas as pd

from ..html_common import page
from . import registry

EXTRA_CSS = """th{cursor:pointer;user-select:none;white-space:nowrap}th.s::after{content:" \\25B2"}th.s.d::after{content:" \\25BC"}
tr.g td{background:var(--card);font-weight:700}#grid{flex:1;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px}
.pn canvas{width:100%;height:auto;border:1px solid var(--line)}#leg{width:220px;font-size:12px;position:sticky;top:8px;max-height:92vh;overflow:auto}
#leg div{white-space:nowrap}.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}"""


def build(avatar_dir: str | Path, out: str | Path, embed: dict[str, str | Path] | None = None) -> Path:
    """avatar_dir holds gt_label_*/images/ and kp_gt_*/{all.csv, results.json} (see scripts/kp_compare.py).

    `embed` maps a tab title to another self-contained report page (e.g. the SLEAP training-label dashboard); each is shown in an iframe,
    so the pages keep their own scripts and one file carries the whole dashboard.
    """
    from PIL import Image

    DIR, OUT = Path(avatar_dir), Path(out)
    P = DIR / "kp_gt_261003"
    df = pd.read_csv(P / "all.csv"); res = json.load(open(P / "results.json"))
    NAME = {"sleap_1423": "SLEAP 1423", "sleap_tailless_1501": "SLEAP tailless 1501", "yolo_avatar3d_train": "YOLO AVATAR3D",
            "yolo_avatar3d_balbc": "YOLO balbc", "yolo_khu_527": "YOLO KHU-527", "rtdetr": "RT-DETRv2",
            "superanimal_quadruped": "SuperAnimal-Quadruped", "superanimal_topviewmouse": "SuperAnimal-TopViewMouse",
            "vitpose_plus_ap10k": "ViTPose++ AP-10K", "lightning_pose": "Lightning Pose", "dlc_resnet": "DLC own training",
            "sleap_retrain": "SLEAP retrain", "dannce": "DANNCE", "rtmpose_mmpose": "RTMPose / MMPose", "yolo_pose_human": "YOLO-pose human"}
    KIND = {"sleap_1423": "Project-trained (SUBTLE)", "sleap_tailless_1501": "Project-trained (SUBTLE)",
            "yolo_avatar3d_train": "Box detector", "yolo_avatar3d_balbc": "Box detector", "yolo_khu_527": "Box detector", "rtdetr": "Box detector",
            "superanimal_quadruped": "Zero-shot foundation", "superanimal_topviewmouse": "Zero-shot foundation", "vitpose_plus_ap10k": "Zero-shot foundation",
            "lightning_pose": "Needs hand labels", "dlc_resnet": "Needs hand labels", "sleap_retrain": "Needs hand labels", "dannce": "Needs hand labels",
            "rtmpose_mmpose": "Skipped", "yolo_pose_human": "Skipped"}
    order = list(NAME)
    COLOR = ["#1f77b4", "#17becf", "#e377c2", "#bcbd22", "#8c564b", "#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c"]
    ran = [m for m in order if m in set(df["model"])]
    col = {m: COLOR[i] for i, m in enumerate(ran)}
    kn = {m: list(dict.fromkeys(df[df.model == m]["keypoint"])) for m in ran}   # native keypoint order = index shown in the viewer

    imgs, pts = {}, {}
    for (f, c), g in df.groupby(["frame", "camera"]):
        key = f"{f}|{c}"
        im = Image.open(DIR / "gt_label_261002" / f"images/f{f:03d}_cam{c.split('_')[1]}.jpg").convert("RGB")
        s = 800 / im.width; im = im.resize((800, round(im.height * s)))
        b = io.BytesIO(); im.save(b, "JPEG", quality=70)
        imgs[key] = {"src": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), "scale": s, "w": im.width, "h": im.height}
        g = g[g.x_px.notna()]
        pts[key] = {m: [[kn[m].index(r.keypoint), round(r.x_px, 1), round(r.y_px, 1), round(r.conf, 3)] for r in gm.itertuples()] for m, gm in g.groupby("model")}

    det = res["detection"]; rp = {r["model"]: r for r in res["reprojection"]}
    parts = ["nose", "neck", "ear", "forepaw", "hindpaw", "tailbase", "tailmid", "tailtip"]
    ag = {}
    for a in res["agreement"]:
        if "sleap_1423" in (a["a"], a["b"]):
            o = a["b"] if a["a"] == "sleap_1423" else a["a"]; ag.setdefault(o, {})[a["part"]] = round(a["median_px"], 1)
    rows = []
    for m in order:
        spec = registry.get(m)
        r = {"model": NAME[m], "kind": KIND[m], "status": spec.status, "kp": spec.n_keypoints, "note": spec.note}
        if m in det:
            r.update(detected=round(det[m]["detected"] * 100), conf=round(det[m]["confident"] * 100),
                     reproj=None if rp[m]["median_px"] is None else round(rp[m]["median_px"], 1),
                     tri=f"{rp[m]['n_triangulated']}/{rp[m]['n_candidates']}")
            r.update({p: ag.get(m, {}).get(p) for p in parts} if m != "sleap_1423" else {})
        rows.append(r)
    data = {"imgs": imgs, "pts": pts, "names": NAME, "colors": col, "order": ran, "kn": kn, "rows": rows, "parts": parts}

    js = """const D=__DATA__;
    // ---- table: group by + click-to-sort
    const cols=[['model','Model'],['status','Status'],['kp','Keypoints'],['detected','Detected %'],['conf','Conf ≥ 0.5 %'],['reproj','Reproj median px'],['tri','Triangulated']].concat(D.parts.map(p=>[p,p]));
    const tb=document.getElementById('tb'),gb=document.getElementById('gb');let sk=null,sd=1;
    function val(r,k){return r[k]===undefined?null:r[k]}
    function heat(v){if(v==null)return '';const a=Math.min(1,v/60);return `style="background:rgba(214,39,40,${(a*0.45).toFixed(2)})"`}
    function table(){const g=gb.value;let rs=D.rows.slice();
     if(sk)rs.sort((a,b)=>{const x=val(a,sk),y=val(b,sk);if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*sd});
     let h='<thead><tr>'+cols.map(([k,l])=>`<th data-k="${k}" class="${sk===k?'s'+(sd<0?' d':''):''}">${l}</th>`).join('')+'</tr></thead><tbody>';
     const groups=g==='none'?[['',rs]]:[...new Set(D.rows.map(r=>r[g]))].map(v=>[v,rs.filter(r=>r[g]===v)]);
     for(const [name,list] of groups){if(name)h+=`<tr class=g><td colspan=${cols.length}>${name} <span class=mut>(${list.length})</span></td></tr>`;
      for(const r of list)h+='<tr>'+cols.map(([k])=>{const v=val(r,k);const ag=D.parts.includes(k);return `<td class="${k==='model'||k==='status'?'':'c'}" ${ag?heat(v):''} ${k==='model'?`title="${r.note.replace(/"/g,'')}"`:''}>${v==null?'':v}</td>`}).join('')+'</tr>'}
     tb.innerHTML=h+'</tbody>';tb.querySelectorAll('th').forEach(t=>t.onclick=()=>{const k=t.dataset.k;sd=sk===k?-sd:1;sk=k;table()})}
    gb.onchange=table;table();
    // ---- viewer
    const cams=['cam_1','cam_2','cam_3','cam_4','cam_5'];
    const frames=[...new Set(Object.keys(D.imgs).map(k=>+k.split('|')[0]))].sort((a,b)=>a-b);
    const fsel=document.getElementById('fsel'),csel=document.getElementById('csel'),mode=document.getElementById('mode'),th=document.getElementById('th'),grid=document.getElementById('grid'),box=document.getElementById('models'),leg=document.getElementById('leg'),lab=document.getElementById('lab');
    frames.forEach(f=>fsel.add(new Option('frame '+f,f)));cams.forEach(c=>csel.add(new Option(c,c)));
    D.order.forEach(m=>{const l=document.createElement('label');l.innerHTML=`<input type=checkbox data-m="${m}" ${['sleap_1423','superanimal_quadruped'].includes(m)?'checked':''}><span class=sw style="background:${D.colors[m]}"></span>${D.names[m]}`;box.appendChild(l)});
    const cache={};function img(k){return cache[k]||(cache[k]=new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=D.imgs[k].src}))}
    async function panel(k,models,title){const I=D.imgs[k],t=+th.value,d=document.createElement('div');d.className='pn';
     const c=document.createElement('canvas');c.width=I.w;c.height=I.h;const x=c.getContext('2d');x.drawImage(await img(k),0,0);x.font='bold 12px sans-serif';
     models.forEach(m=>(D.pts[k][m]||[]).forEach(p=>{if(p[3]<t)return;const px=p[1]*I.scale,py=p[2]*I.scale;x.strokeStyle=x.fillStyle=D.colors[m];x.lineWidth=2;x.beginPath();x.arc(px,py,4,0,7);x.stroke();
      x.fillText(lab.value==='name'?D.kn[m][p[0]]:String(p[0]),px+6,py-5)}));
     d.innerHTML=`<div class=mut>${title}</div>`;d.appendChild(c);return d}
    function legend(models){leg.innerHTML=models.map(m=>`<div style="color:${D.colors[m]};font-weight:700;margin-top:6px">${D.names[m]}</div>`+D.kn[m].map((n,i)=>`<div style="color:${D.colors[m]}">${i}  ${n}</div>`).join('')).join('')}
    async function draw(){document.getElementById('tv').textContent=(+th.value).toFixed(2);const f=fsel.value,on=[...box.querySelectorAll('input:checked')].map(i=>i.dataset.m);
     csel.style.display=mode.value==='model'?'':'none';legend(on);const out=[];
     if(mode.value==='cam')for(const c of cams)out.push(await panel(f+'|'+c,on,c));
     else for(const m of on)out.push(await panel(f+'|'+csel.value,[m],D.names[m]));
     grid.replaceChildren(...out)}
    [fsel,csel,mode,lab,box].forEach(e=>e.addEventListener('change',draw));th.addEventListener('input',draw);
    document.addEventListener('keydown',e=>{const i=frames.indexOf(+fsel.value);if(e.key==='ArrowRight'&&i<frames.length-1){fsel.value=frames[i+1];draw()}if(e.key==='ArrowLeft'&&i>0){fsel.value=frames[i-1];draw()}});
    draw();"""
    body = """
<h1>AVATAR keypoint predictor comparison</h1>
<p class="lead">9 predictors ran on the same 150 GT images (30 frames × 5 cameras, far_20231205); no hand labels exist yet, so this page shows agreement and stability, not accuracy.</p>
<p class="mut">Related: <a href="261003_AVATAR_training_labels.html">SLEAP 1423 training labels</a> (provenance, definition, checks).</p><h2>Models</h2><p class="lead">One table: group by type or status, click a header to sort. Agreement columns are median px distance to SLEAP 1423 (red = larger); hover a model for its note.</p>
<div>Group by <select id="gb"><option value="kind">Type</option><option value="status">Status</option><option value="none">None</option></select>
<span class="mut"> Reproj = DLT over keypoints seen by ≥ 3 cameras at conf ≥ 0.5; the calibration is 385 days newer than this clip. Counts differ per model, so medians are not on the same keypoints. Box models give box centres, not paws. ViTPose++ boxes come from SLEAP 1423, so it is not independent of SLEAP.</span></div>
<table id="tb"></table>
<h2>Overlay viewer</h2><p class="lead">All 5 cameras of a frame at once, or one camera with one panel per model. Each point shows its keypoint index; the list on the right maps index to name.</p>
<div class="card"><div style="margin-bottom:8px"><select id="fsel"></select> <select id="mode"><option value="cam">5 cameras at once</option><option value="model">models side by side</option></select> <select id="csel" style="display:none"></select>
conf ≥ <input id="th" type="range" min="0" max="0.95" step="0.05" value="0.5"> <b id="tv"></b> label <select id="lab"><option value="idx">index</option><option value="name">name</option></select> <span class="mut">← → = frame</span></div>
<div id="models" style="margin-bottom:8px"></div><div class="vw"><div id="grid"></div><div id="leg"></div></div></div>
<h2>Limits</h2><p class="lead">Nothing here ranks accuracy.</p><ul>
<li>Accuracy needs the hand labels in gt_label_261002 (labelling page: label_page.html).</li>
<li>SuperAnimal videos were encoded from the JPEG images (CRF 1), so inputs are near-identical, not byte-identical.</li>
<li>ViTPose++ AP-10K names follow mmpose ap10k.py; the checkpoint's own label list is COCO-17 order and would swap eyes, nose and neck.</li>
<li>Lightning Pose, DLC training, SLEAP retraining and DANNCE need labels (DANNCE also 3D and calibration) and are not run.</li></ul>
"""
    if embed:
        import html as _html

        tabs = ["Predictors", *embed]
        bar = "".join(f'<button class="tb" data-i="{i}">{_html.escape(t)}</button> ' for i, t in enumerate(tabs))
        frames = "".join(f'<div class="tab" data-i="{i + 1}" hidden><iframe loading="lazy" style="width:100%;height:3200px;border:0" srcdoc="{_html.escape(Path(f).read_text(), quote=True)}"></iframe></div>'
                         for i, f in enumerate(embed.values()))
        switch = "<script>document.querySelectorAll('.tb').forEach(b=>b.onclick=()=>{const i=b.dataset.i;document.querySelectorAll('.tab').forEach(t=>t.hidden=t.dataset.i!==i)})</script>"
        body = f'<div style="margin-bottom:12px">{bar}</div><div class="tab" data-i="0">{body}</div>{frames}'
        OUT.write_text(page("AVATAR keypoint dashboard", body, js, data, EXTRA_CSS).replace("</body>", switch + "</body>"))
        return OUT
    OUT.write_text(page("AVATAR keypoint predictor comparison", body, js, data, EXTRA_CSS))
    return OUT
