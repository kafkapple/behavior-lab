"""Interactive playback block for an HTML page: skeleton, cluster map and ethograms in sync.

``player_block`` returns one ``<details>`` element with its data as JSON; ``PLAYER_JS`` is the
shared script, emitted once per page. Everything is drawn on a canvas, so the page needs no
library. Keypoints, embedding and every label sequence are subsampled with ONE index vector,
so a time step always shows the pose, the map position and the labels of the same frame.
"""
# ruff: noqa: E501  (the embedded script keeps its own line lengths)
from __future__ import annotations

import html
import json

import numpy as np

from .agreement import stretch_labels


def _quantize(x: np.ndarray, lo: np.ndarray, span: float | np.ndarray) -> list[int]:
    return np.clip(np.round((x - lo) / span * 1000), 0, 1000).astype(int).ravel().tolist()


def player_data(keypoints: np.ndarray, edges: list[tuple[int, int]], embedding: np.ndarray,
                seqs: dict[str, np.ndarray], fps: float, hz: float = 10.0) -> dict:
    kp = np.asarray(keypoints, dtype=float)
    T = len(kp)
    assert len(embedding) == T, f"embedding {len(embedding)} vs keypoints {T}"
    idx = np.arange(0, T, max(1, round(fps / hz)))  # the one index vector
    xy = kp[idx][:, :, :2]
    # 1st to 99th percentile: a few tracking-loss frames must not squash the view (they clip)
    lo, hi = np.percentile(xy.reshape(-1, 2), [1, 99], axis=0)
    span = float((hi - lo).max()) or 1.0  # equal aspect
    emb = np.asarray(embedding)[idx, :2]
    elo, ehi = np.percentile(emb, [0.5, 99.5], axis=0)
    espan = ehi - elo + 1e-9
    methods = [{"name": n, "labels": stretch_labels(v, T)[idx].astype(int).tolist()}
               for n, v in seqs.items()]
    return {"hz": fps / max(1, round(fps / hz)), "n": len(idx), "k": kp.shape[1],
            "kp": _quantize(xy, lo, span), "edges": [list(map(int, e)) for e in edges],
            "emb": _quantize(emb, elo, espan), "methods": methods}


def player_block(pid: str, title: str, data: dict, *, open_: bool = False) -> str:
    opts = "".join(f'<option value="{i}">{html.escape(m["name"])}</option>'
                   for i, m in enumerate(data["methods"]))
    return (f'<details class="bl-player" id="{html.escape(pid)}"{" open" if open_ else ""}>'
            f"<summary>{html.escape(title)}</summary>"
            '<p><button type="button">play</button> '
            '<input type="range" min="0" value="0" style="width:45%"> '
            f'map colored by <select class="m">{opts}</select> speed <select class="s">'
            '<option value="1">1x</option><option value="2">2x</option>'
            '<option value="5">5x</option><option value="0.5">0.5x</option></select> '
            '<span class="info"></span></p>'
            '<canvas width="960" height="560" style="max-width:100%"></canvas>'
            f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script>'
            "</details>")


PLAYER_JS = r"""<script>
(function(){
const TAB=['#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896',
'#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7','#bcbd22',
'#dbdb8d','#17becf','#9edae5'];
const col=c=>c<0?'#bbbbbb':TAB[c%20];
function init(root){
  if(root.dataset.ready)return; root.dataset.ready='1';
  const d=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
  const cv=root.querySelector('canvas'),g=cv.getContext('2d');
  const sl=root.querySelector('input'),btn=root.querySelector('button');
  const sel=root.querySelector('select.m'),sp=root.querySelector('select.s');
  const info=root.querySelector('.info');
  const P=380,MX=440,EX=170,EW=cv.width-EX-10,EY=P+26,RH=Math.min(22,(cv.height-EY)/d.methods.length);
  sl.max=d.n-1; let t=0,timer=null,cent={},top=[];
  // ids are ranked per method so colors match the static maps (sorted id -> tab20 index)
  d.methods.forEach(m=>{const ids=[...new Set(m.labels.filter(c=>c>=0))].sort((a,b)=>a-b);
    m.rank={};ids.forEach((c,i)=>m.rank[c]=i);m.c=m.labels.map(c=>c<0?-1:m.rank[c]);});
  const eth=d.methods.map(m=>{const c=document.createElement('canvas');c.width=EW;c.height=1;
    const x=c.getContext('2d');for(let p=0;p<EW;p++){x.fillStyle=col(m.c[Math.floor(p*d.n/EW)]);
    x.fillRect(p,0,1,1);}return c;});
  const map=document.createElement('canvas');map.width=P;map.height=P;
  const ex=i=>d.emb[2*i]*(P-20)/1000+10, ey=i=>P-10-d.emb[2*i+1]*(P-20)/1000;
  function drawMap(){const m=d.methods[sel.value],x=map.getContext('2d');x.clearRect(0,0,P,P);
    x.globalAlpha=.45;const s={},n={};
    for(let i=0;i<d.n;i++){const c=m.labels[i];x.fillStyle=col(m.c[i]);x.fillRect(ex(i)-1,ey(i)-1,2,2);
      if(c>=0){(s[c]=s[c]||[0,0]);s[c][0]+=ex(i);s[c][1]+=ey(i);n[c]=(n[c]||0)+1;}}
    cent={};for(const c in s)cent[c]=[s[c][0]/n[c],s[c][1]/n[c]];
    top=Object.keys(n).sort((a,b)=>n[b]-n[a]).slice(0,12).map(Number);}
  function badge(c,m,big){const p=cent[c];if(!p)return;g.beginPath();g.arc(MX+p[0],p[1],big?11:8,0,7);
    g.fillStyle='#fff';g.fill();g.lineWidth=big?3:1.5;g.strokeStyle=col(m.rank[c]);g.stroke();
    g.fillStyle='#000';g.font=(big?'bold ':'')+'10px sans-serif';g.textAlign='center';
    g.fillText(c,MX+p[0],p[1]+3.5);}
  function draw(){const fg=getComputedStyle(root).color,m=d.methods[sel.value];
    g.clearRect(0,0,cv.width,cv.height);g.textAlign='left';g.fillStyle=fg;g.font='12px sans-serif';
    g.fillText('skeleton, top view',4,12);g.fillText('cluster map: '+m.name,MX+4,12);
    const kx=j=>d.kp[2*(t*d.k+j)]*(P-30)/1000+15, ky=j=>P-15-d.kp[2*(t*d.k+j)+1]*(P-30)/1000;
    g.strokeStyle=fg;g.lineWidth=1.5;g.globalAlpha=.7;
    d.edges.forEach(e=>{g.beginPath();g.moveTo(kx(e[0]),ky(e[0]));g.lineTo(kx(e[1]),ky(e[1]));g.stroke();});
    g.globalAlpha=1;for(let j=0;j<d.k;j++){g.beginPath();g.arc(kx(j),ky(j),3.5,0,7);g.fillStyle=TAB[j%20];g.fill();}
    g.drawImage(map,MX,0);
    // recent positions as fading dots, never joined: this plane is a projection
    for(let b=20;b>0;b--){const i=t-b;if(i<0)continue;g.globalAlpha=(21-b)/40;g.beginPath();
      g.arc(MX+ex(i),ey(i),2.5,0,7);g.fillStyle=fg;g.fill();}
    g.globalAlpha=1;
    let j=t;while(j>0&&t-j<10&&m.labels[j]===m.labels[j-1])j--;
    const a=m.labels[j-1],b=m.labels[j];
    if(j>0&&t-j<10&&a!==b&&cent[a]&&cent[b]){const p=cent[a],q=cent[b];
      const an=Math.atan2(q[1]-p[1],q[0]-p[0]);g.strokeStyle=fg;g.lineWidth=3;g.beginPath();
      g.moveTo(MX+p[0],p[1]);g.lineTo(MX+q[0],q[1]);
      g.lineTo(MX+q[0]-12*Math.cos(an-.4),q[1]-12*Math.sin(an-.4));g.moveTo(MX+q[0],q[1]);
      g.lineTo(MX+q[0]-12*Math.cos(an+.4),q[1]-12*Math.sin(an+.4));g.stroke();badge(a,m,false);}
    top.forEach(c=>badge(c,m,false));const cur=m.labels[t];if(cur>=0)badge(cur,m,true);
    g.beginPath();g.arc(MX+ex(t),ey(t),5,0,7);g.lineWidth=2;g.strokeStyle=fg;g.stroke();
    d.methods.forEach((mm,r)=>{const y=EY+r*RH;g.drawImage(eth[r],EX,y,EW,RH-4);g.fillStyle=fg;
      g.textAlign='left';g.font=(r==sel.value?'bold ':'')+'11px sans-serif';
      const c=mm.labels[t];g.fillText(mm.name.slice(0,18)+': '+(c<0?'noise':c),2,y+RH-9);});
    const cx=EX+t*EW/d.n;g.strokeStyle=fg;g.lineWidth=1.5;g.beginPath();g.moveTo(cx,EY-4);
    g.lineTo(cx,EY+d.methods.length*RH);g.stroke();
    info.textContent=(t/d.hz).toFixed(1)+' s of '+(d.n/d.hz).toFixed(0)+' s';sl.value=t;}
  function stop(){clearInterval(timer);timer=null;btn.textContent='play';}
  function play(){stop();btn.textContent='pause';timer=setInterval(()=>{t++;if(t>=d.n){t=d.n-1;stop();}draw();},1000/d.hz/sp.value);}
  btn.onclick=()=>timer?stop():(t>=d.n-1&&(t=0),play());
  sl.oninput=()=>{t=+sl.value;draw();};sel.onchange=()=>{drawMap();draw();};sp.onchange=()=>timer&&play();
  cv.onclick=e=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*cv.width/r.width,y=(e.clientY-r.top)*cv.height/r.height;
    if(y>EY-6&&x>=EX){t=Math.min(d.n-1,Math.floor((x-EX)*d.n/EW));draw();}};
  drawMap();draw();
}
document.querySelectorAll('details.bl-player').forEach(el=>{
  el.addEventListener('toggle',()=>{if(el.open)init(el);});if(el.open)init(el);});
})();
</script>"""

__all__ = ["PLAYER_JS", "player_block", "player_data"]
