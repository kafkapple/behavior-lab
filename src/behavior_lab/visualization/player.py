"""Interactive playback block for an HTML page: 3D skeleton, cluster map and ethograms in sync.

``player_block`` returns one ``<details>`` element with its data as JSON; ``PLAYER_JS`` is the
shared script, emitted once per page. Everything is drawn on a canvas, so the page needs no
library. Keypoints, embedding and every label sequence are subsampled with ONE index vector,
so a time step always shows the pose, the map position and the labels of the same frame.

Clusters of the chosen method can be switched off in the legend; playback then skips their
steps. Every skip is a cut: the trail is reset and a "cut" mark is shown, so two bouts that
were minutes apart never look like continuous motion.

A focus class narrows the view to one cluster: its points stay coloured on the map, the
others fade, arrows show where its bouts go next (share of its outgoing transitions, top 3),
and the bout list holds its runs. Picking a bout jumps to it and plays it to its end.
"""
# ruff: noqa: E501  (the embedded script keeps its own line lengths)
from __future__ import annotations

import html
import json

import numpy as np

from .agreement import stretch_labels
from .cluster_map import rank_colors
from .keypoint_schema import joint_color


def _quantize(x: np.ndarray, lo: np.ndarray, span: float | np.ndarray) -> list[int]:
    return np.clip(np.round((x - lo) / span * 1000), 0, 1000).astype(int).ravel().tolist()


def player_data(keypoints: np.ndarray, edges: list[tuple[int, int]], embedding: np.ndarray,
                seqs: dict[str, np.ndarray], fps: float, hz: float = 10.0) -> dict:
    kp = np.asarray(keypoints, dtype=float)
    T, K, D = kp.shape
    assert len(embedding) == T, f"embedding {len(embedding)} vs keypoints {T}"
    idx = np.arange(0, T, max(1, round(fps / hz)))  # the one index vector
    xyz = kp[idx] if D >= 3 else np.concatenate([kp[idx], np.zeros((len(idx), K, 1))], axis=2)
    xyz = xyz[:, :, :3]
    # 1st to 99th percentile: a few tracking-loss frames must not squash the view (they clip)
    lo, hi = np.percentile(xyz.reshape(-1, 3), [1, 99], axis=0)
    span = float((hi - lo).max()) or 1.0  # one scale for the three axes
    body = float(np.median(np.ptp(xyz, axis=1).max(axis=1))) / span  # animal size, 0..1
    emb = np.asarray(embedding)[idx, :2]
    elo, ehi = np.percentile(emb, [0.5, 99.5], axis=0)
    methods = []
    for name, v in seqs.items():
        lab = stretch_labels(v, T)[idx].astype(int)
        color = rank_colors(lab)
        order = [c for c in color if c >= 0] + ([-1] if (lab < 0).any() else [])
        methods.append({"name": name, "labels": lab.tolist(), "order": order,
                        "colors": [color[c] for c in order],
                        "share": [round(float((lab == c).mean()), 4) for c in order]})
    return {"hz": fps / max(1, round(fps / hz)), "n": len(idx), "k": K, "body": body,
            "grid": span / 10,  # floor grid spacing in the file's units
            "kp": _quantize(xyz, lo, span), "edges": [list(map(int, e)) for e in edges],
            "emb": _quantize(emb, elo, ehi - elo + 1e-9), "methods": methods,
            "joint_colors": [joint_color(k) for k in range(K)]}


def player_block(pid: str, title: str, data: dict, *, open_: bool = False) -> str:
    opts = "".join(f'<option value="{i}">{html.escape(m["name"])}</option>'
                   for i, m in enumerate(data["methods"]))
    return (f'<details class="bl-player" id="{html.escape(pid)}"{" open" if open_ else ""}>'
            f"<summary>{html.escape(title)}</summary>"
            '<p><button type="button" class="play">play</button> '
            '<input type="range" min="0" value="0" style="width:40%"> '
            f'clusters of <select class="m">{opts}</select> speed <select class="s">'
            '<option value="1">1x</option><option value="2">2x</option>'
            '<option value="5">5x</option><option value="0.5">0.5x</option></select> '
            '<label><input type="checkbox" class="follow" checked> follow animal</label> '
            '<span class="info"></span></p>'
            '<p>focus class <select class="c"><option value="">all</option></select> '
            'bout <select class="b" disabled><option value="">pick a class first</option></select> '
            '<span class="binfo"></span></p>'
            '<p class="legend" style="line-height:2"></p>'
            '<canvas width="960" height="560" style="max-width:100%;touch-action:none"></canvas>'
            f'<script type="application/json">{json.dumps(data, separators=(",", ":"))}</script>'
            "</details>")


PLAYER_JS = r"""<script>
(function(){
function init(root){
  if(root.dataset.ready)return; root.dataset.ready='1';
  const d=JSON.parse(root.querySelector('script[type="application/json"]').textContent);
  const cv=root.querySelector('canvas'),g=cv.getContext('2d');
  const sl=root.querySelector('input[type=range]'),btn=root.querySelector('button.play');
  const sel=root.querySelector('select.m'),sp=root.querySelector('select.s');
  const fol=root.querySelector('input.follow'),info=root.querySelector('.info'),leg=root.querySelector('.legend');
  const csel=root.querySelector('select.c'),bsel=root.querySelector('select.b'),binfo=root.querySelector('.binfo');
  let focus=null,bend=null,runs=[],outs=[];
  const P=380,MX=440,EX=170,EW=cv.width-EX-10,EY=P+26,RH=Math.min(22,(cv.height-EY)/d.methods.length);
  sl.max=d.n-1; let t=0,timer=null,cent={},yaw=0.6,pitch=1.0,cut=-99,trail0=0;
  d.methods.forEach(m=>{m.col={};m.on={};m.order.forEach((c,i)=>{m.col[c]=m.colors[i];m.on[c]=true;});});
  const M=()=>d.methods[sel.value];
  const eth=d.methods.map(()=>{const c=document.createElement('canvas');c.width=EW;c.height=1;return c;});
  function drawEth(r){const m=d.methods[r],x=eth[r].getContext('2d');x.clearRect(0,0,EW,1);
    for(let p=0;p<EW;p++){const c=m.labels[Math.floor(p*d.n/EW)];x.globalAlpha=m.on[c]?1:.12;x.fillStyle=m.col[c];x.fillRect(p,0,1,1);}}
  const map=document.createElement('canvas');map.width=P;map.height=P;
  const ex=i=>d.emb[2*i]*(P-20)/1000+10, ey=i=>P-10-d.emb[2*i+1]*(P-20)/1000;
  function drawMap(){const m=M(),x=map.getContext('2d');x.clearRect(0,0,P,P);const s={},n={};
    for(let pass=0;pass<2;pass++)for(let i=0;i<d.n;i++){const c=m.labels[i],on=m.on[c]&&(focus===null||c===focus);if(on!==(pass===1))continue;
      x.globalAlpha=on?.7:.08;x.fillStyle=on?m.col[c]:'#888';x.fillRect(ex(i)-1.5,ey(i)-1.5,3,3);
      if(c>=0){(s[c]=s[c]||[0,0]);s[c][0]+=ex(i);s[c][1]+=ey(i);n[c]=(n[c]||0)+1;}}
    cent={};for(const c in s)cent[c]=[s[c][0]/n[c],s[c][1]/n[c]];}
  function legend(){const m=M();leg.innerHTML='';
    const mk=(txt,fn,bg)=>{const b=document.createElement('button');b.type='button';b.textContent=txt;
      b.style.cssText='margin:1px 3px;padding:1px 6px;border-radius:9px;border:2px solid '+(bg||'currentColor')+';background:'+(bg||'transparent')+';cursor:pointer';
      b.onclick=fn;leg.appendChild(b);return b;};
    const all=v=>()=>{m.order.forEach(c=>m.on[c]=v);refresh();};
    mk('all',all(true));mk('none',all(false));
    m.order.forEach((c,i)=>{const b=mk((c<0?'noise':c)+' · '+(m.share[i]*100).toFixed(0)+'%',()=>{m.on[c]=!m.on[c];refresh();},m.col[c]);
      b.style.color='#000';b.style.opacity=m.on[c]?1:.3;});}
  function classes(){const m=M();csel.innerHTML='<option value="">all</option>'+m.order.filter(c=>c>=0)
      .map((c,i)=>'<option value="'+c+'">'+c+' · '+(m.share[m.order.indexOf(c)]*100).toFixed(0)+'%</option>').join('');
    focus=null;setFocus();}
  function setFocus(){const m=M();runs=[];outs=[];bend=null;
    if(focus!==null){let s0=-1;for(let i=0;i<=d.n;i++){const c=i<d.n?m.labels[i]:null;
        if(c===focus&&s0<0)s0=i;if(c!==focus&&s0>=0){runs.push([s0,i-1]);s0=-1;}}
      const cnt={};let tot=0;runs.forEach(r=>{const nx=m.labels[r[1]+1];if(r[1]+1<d.n&&nx>=0){cnt[nx]=(cnt[nx]||0)+1;tot++;}});
      outs=Object.keys(cnt).map(k=>[+k,cnt[k]/tot]).sort((a,b)=>b[1]-a[1]).slice(0,3);}
    bsel.disabled=focus===null;
    bsel.innerHTML=focus===null?'<option value="">pick a class first</option>':'<option value="">'+runs.length+' bouts</option>'+
      runs.map((r,i)=>'<option value="'+i+'">#'+(i+1)+'  '+(r[0]/d.hz).toFixed(1)+' s  ('+((r[1]-r[0]+1)/d.hz).toFixed(1)+' s)</option>').join('');
    binfo.textContent=focus===null?'':'next after a bout: '+(outs.map(o=>o[0]+' '+(o[1]*100).toFixed(0)+'%').join(', ')||'none');
    drawMap();draw();}
  function refresh(){legend();drawMap();drawEth(+sel.value);draw();}
  function badge(c,m,big){const p=cent[c];if(!p)return;g.beginPath();g.arc(MX+p[0],p[1],big?11:8,0,7);
    g.fillStyle='#fff';g.fill();g.lineWidth=big?3:1.5;g.strokeStyle=m.col[c];g.stroke();
    g.fillStyle='#000';g.font=(big?'bold ':'')+'10px sans-serif';g.textAlign='center';g.fillText(c,MX+p[0],p[1]+3.5);}
  function view(){ // orthographic: yaw about the third coordinate, pitch 0 = side, 90 deg = from above
    const k=d.k,o=t*k*3,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),spn=Math.sin(pitch);
    let c=[500,500,0],z=1;
    if(fol.checked){c=[0,0,0];for(let j=0;j<k;j++)for(let a=0;a<3;a++)c[a]+=d.kp[o+3*j+a]/k;z=0.45/Math.max(d.body,.02);}
    const pr=(X,Y,H)=>{const x=(X-c[0])/1000*z,y=(Y-c[1])/1000*z,h=(H-c[2])/1000*z,xr=x*cy-y*sy,yr=x*sy+y*cy;
      return [P/2+xr*(P-40),P/2-(yr*spn+h*cp)*(P-40),yr*cp-h*spn];};
    return {pr,c,o};}
  function floor(v,fg){ // grid on the floor (third coordinate = its 1st percentile) and an axis gizmo
    g.save();g.beginPath();g.rect(0,16,P,P-16);g.clip();g.strokeStyle=fg;g.lineWidth=1;g.globalAlpha=.18;
    for(let i=0;i<=10;i++){let p=v.pr(i*100,0,0),q=v.pr(i*100,1000,0);g.beginPath();g.moveTo(p[0],p[1]);g.lineTo(q[0],q[1]);g.stroke();
      p=v.pr(0,i*100,0);q=v.pr(1000,i*100,0);g.beginPath();g.moveTo(p[0],p[1]);g.lineTo(q[0],q[1]);g.stroke();}
    g.restore();g.globalAlpha=1;
    const o0=v.pr(v.c[0],v.c[1],v.c[2]),ax=[[1,0,0,'#d62728','x'],[0,1,0,'#2ca02c','y'],[0,0,1,'#1f77b4','z']];
    ax.forEach(a=>{const p=v.pr(v.c[0]+a[0]*100,v.c[1]+a[1]*100,v.c[2]+a[2]*100);let dx=p[0]-o0[0],dy=p[1]-o0[1];
      const n=Math.hypot(dx,dy)||1,L=28*Math.min(1,n/((P-40)*.1*(fol.checked?0.45/Math.max(d.body,.02):1)));dx=dx/n*L;dy=dy/n*L;
      g.strokeStyle=a[3];g.lineWidth=2;g.beginPath();g.moveTo(36,P-36);g.lineTo(36+dx,P-36+dy);g.stroke();
      g.fillStyle=a[3];g.font='11px sans-serif';g.fillText(a[4],36+dx*1.25-3,P-36+dy*1.25+4);});
    g.fillStyle=fg;g.font='10px sans-serif';g.fillText('grid '+d.grid.toPrecision(2)+' (file units)',70,P-6);}
  function pose(v){const out=[];for(let j=0;j<d.k;j++)out.push(v.pr(d.kp[v.o+3*j],d.kp[v.o+3*j+1],d.kp[v.o+3*j+2]));return out;}
  function draw(){const fg=getComputedStyle(root).color,m=M();
    g.clearRect(0,0,cv.width,cv.height);g.textAlign='left';g.fillStyle=fg;g.font='12px sans-serif';
    g.fillText('skeleton (drag to rotate)',4,12);g.fillText('cluster map: '+m.name,MX+4,12);
    const v=view();floor(v,fg);const q=pose(v);g.strokeStyle=fg;g.lineWidth=1.5;g.globalAlpha=.7;
    d.edges.forEach(e=>{g.beginPath();g.moveTo(q[e[0]][0],q[e[0]][1]);g.lineTo(q[e[1]][0],q[e[1]][1]);g.stroke();});
    g.globalAlpha=1;q.map((p,j)=>[p,j]).sort((a,b)=>b[0][2]-a[0][2]).forEach(([p,j])=>{g.beginPath();g.arc(p[0],p[1],4,0,7);g.fillStyle=d.joint_colors[j];g.fill();});
    if(t-cut<8){g.fillStyle='#d62728';g.font='bold 14px sans-serif';g.fillText('cut',P-40,14);}
    g.drawImage(map,MX,0);
    if(focus!==null&&cent[focus])outs.forEach(o=>{const p=cent[focus],r=cent[o[0]];if(!r)return;
      const an=Math.atan2(r[1]-p[1],r[0]-p[0]);g.globalAlpha=.85;g.strokeStyle=m.col[o[0]];g.lineWidth=1+6*o[1];g.beginPath();
      g.moveTo(MX+p[0],p[1]);g.lineTo(MX+r[0],r[1]);g.lineTo(MX+r[0]-14*Math.cos(an-.35),r[1]-14*Math.sin(an-.35));
      g.moveTo(MX+r[0],r[1]);g.lineTo(MX+r[0]-14*Math.cos(an+.35),r[1]-14*Math.sin(an+.35));g.stroke();g.globalAlpha=1;
      g.fillStyle=fg;g.font='10px sans-serif';g.textAlign='center';g.fillText((o[1]*100).toFixed(0)+'%',MX+(p[0]+r[0])/2,(p[1]+r[1])/2-4);badge(o[0],m,false);});
    // recent positions as fading dots, never joined (the plane is a projection); reset at a cut
    for(let b=20;b>0;b--){const i=t-b;if(i<trail0)continue;g.globalAlpha=(21-b)/40;g.beginPath();g.arc(MX+ex(i),ey(i),2.5,0,7);g.fillStyle=fg;g.fill();}
    g.globalAlpha=1;
    let j=t;while(j>trail0&&t-j<10&&m.labels[j]===m.labels[j-1])j--;
    const a=m.labels[j-1],b=m.labels[j];
    if(j>trail0&&t-j<10&&a!==b&&cent[a]&&cent[b]){const p=cent[a],r=cent[b];
      const an=Math.atan2(r[1]-p[1],r[0]-p[0]);g.strokeStyle=fg;g.lineWidth=3;g.beginPath();
      g.moveTo(MX+p[0],p[1]);g.lineTo(MX+r[0],r[1]);
      g.lineTo(MX+r[0]-12*Math.cos(an-.4),r[1]-12*Math.sin(an-.4));g.moveTo(MX+r[0],r[1]);
      g.lineTo(MX+r[0]-12*Math.cos(an+.4),r[1]-12*Math.sin(an+.4));g.stroke();badge(a,m,false);}
    m.order.slice(0,12).forEach(c=>{if(c>=0&&m.on[c])badge(c,m,false);});
    const cur=m.labels[t];if(cur>=0)badge(cur,m,true);
    g.beginPath();g.arc(MX+ex(t),ey(t),5,0,7);g.lineWidth=2;g.strokeStyle=fg;g.stroke();
    d.methods.forEach((mm,r)=>{const y=EY+r*RH;g.drawImage(eth[r],EX,y,EW,RH-4);g.fillStyle=fg;
      g.textAlign='left';g.font=(r==sel.value?'bold ':'')+'11px sans-serif';
      const c=mm.labels[t];g.fillStyle=mm.col[c];g.fillRect(2,y+3,8,RH-10);g.fillStyle=fg;
      g.fillText(mm.name.slice(0,18)+': '+(c<0?'noise':c),14,y+RH-9);});
    const cx=EX+t*EW/d.n;g.strokeStyle=fg;g.lineWidth=1.5;g.beginPath();g.moveTo(cx,EY-4);
    g.lineTo(cx,EY+d.methods.length*RH);g.stroke();
    info.textContent=(t/d.hz).toFixed(1)+' s of '+(d.n/d.hz).toFixed(0)+' s';sl.value=t;}
  function stop(){clearInterval(timer);timer=null;btn.textContent='play';}
  function step(){const m=M();let u=t+1;if(bend!==null&&u>bend){bend=null;stop();return;}while(u<d.n&&!m.on[m.labels[u]])u++;
    if(u>=d.n){stop();return;}if(u>t+1){cut=u;trail0=u;}t=u;draw();}
  function play(){stop();btn.textContent='pause';timer=setInterval(step,1000/d.hz/sp.value);}
  btn.onclick=()=>timer?stop():(t>=d.n-1&&(t=0,trail0=0),play());
  sl.oninput=()=>{t=+sl.value;trail0=t;bend=null;draw();};sel.onchange=()=>{refresh();classes();};
  csel.onchange=()=>{focus=csel.value===''?null:+csel.value;setFocus();};
  bsel.onchange=()=>{if(bsel.value==='')return;const r=runs[+bsel.value];t=r[0];trail0=t;cut=t;bend=r[1];draw();play();};sp.onchange=()=>timer&&play();fol.onchange=draw;
  const xy=e=>{const r=cv.getBoundingClientRect();return [(e.clientX-r.left)*cv.width/r.width,(e.clientY-r.top)*cv.height/r.height];};
  let drag=null;
  cv.onpointerdown=e=>{const [x,y]=xy(e);if(y>EY-6&&x>=EX){t=Math.min(d.n-1,Math.floor((x-EX)*d.n/EW));trail0=t;draw();}
    else if(x<P&&y<P){drag=[x,y];cv.setPointerCapture(e.pointerId);}};
  cv.onpointermove=e=>{if(!drag)return;const [x,y]=xy(e);yaw+=(x-drag[0])*.01;
    pitch=Math.max(0,Math.min(Math.PI/2,pitch+(y-drag[1])*.01));drag=[x,y];draw();};
  cv.onpointerup=()=>{drag=null;};
  d.methods.forEach((_,r)=>drawEth(r));refresh();classes();
}
document.querySelectorAll('details.bl-player').forEach(el=>{
  el.addEventListener('toggle',()=>{if(el.open)init(el);});if(el.open)init(el);});
})();
</script>"""

__all__ = ["PLAYER_JS", "player_block", "player_data"]
