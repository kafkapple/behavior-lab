// The three drawing views: 3D skeleton, cluster map, timeline. They redraw on bus events and never call each other.
import {$, st, on, emit, fit, lid, inF} from './state.js';
import {color} from './colors.js';

const cen = f => [0, 1, 2].map(i => f.reduce((s, j) => s + j[i], 0) / f.length);
const fg = () => getComputedStyle(document.body).color;

// ---- 3D skeleton: follows the animal; zoom, rotate, pan
let V = {yaw: .6, pitch: -.4, zoom: 1, px: 0, py: 0}, B = 1;
function drawSkel() {
  const S = st.S, c = $('cv'), x = fit(c), W = c.width, H = c.height; x.clearRect(0, 0, W, H);
  if (!S || !S.xyz.length) return;
  const f = S.xyz[st.fr], cy = Math.cos(V.yaw), sy = Math.sin(V.yaw), cp = Math.cos(V.pitch), sp = Math.sin(V.pitch), k = H * .42 / B * V.zoom, c0 = cen(f);
  const P = j => { const a = j[0] - c0[0], b = j[1] - c0[1], d = j[2] - c0[2], X = a * cy - d * sy, Z = a * sy + d * cy, Y = b * cp - Z * sp; return [W / 2 + V.px + X * k, H / 2 + V.py - Y * k]; };
  const col = color(lid(st.fr)); x.lineWidth = 3; x.strokeStyle = col; x.fillStyle = col;
  for (const [a, b] of S.edges) { const p = P(f[a]), q = P(f[b]); x.beginPath(); x.moveTo(...p); x.lineTo(...q); x.stroke(); }
  f.forEach(j => { const p = P(j); x.beginPath(); x.arc(p[0], p[1], 3.5, 0, 7); x.fill(); });
  x.fillStyle = fg(); x.font = '11px sans-serif'; x.fillText(`zoom ${V.zoom.toFixed(2)}x`, 8, H - 8);
}
function bodySize() {                                                  // median max joint distance from the centroid
  const S = st.S, r = [];
  for (let i = 0; i < S.xyz.length; i += Math.max(1, Math.floor(S.xyz.length / 500))) { const f = S.xyz[i], c = cen(f); r.push(Math.max(...f.map(j => Math.hypot(j[0] - c[0], j[1] - c[1], j[2] - c[2])))); }
  r.sort((a, b) => a - b); return Math.max(1e-6, r[r.length >> 1] * 1.6);
}
const cv = $('cv'); let drag = null;
cv.addEventListener('wheel', e => { e.preventDefault(); V.zoom = Math.min(20, Math.max(.2, V.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12))); drawSkel(); }, {passive: false});
cv.addEventListener('contextmenu', e => e.preventDefault());
cv.onmousedown = e => { drag = {x: e.clientX, y: e.clientY, v: {...V}, pan: e.shiftKey || e.button === 2}; };
cv.ondblclick = () => { V = {yaw: .6, pitch: -.4, zoom: 1, px: 0, py: 0}; drawSkel(); };

// ---- cluster map: embedding scatter, current frame ringed, non-filtered points dimmed
let M = {zoom: 1, px: 0, py: 0}, cache = null, eb = null;
function xy(p, c) { const s = Math.min(c.width, c.height) * .45 * M.zoom / Math.max(eb.rx, eb.ry); return [c.width / 2 + M.px + (p[0] - eb.cx) * s, c.height / 2 + M.py - (p[1] - eb.cy) * s]; }
function drawMap() {
  const S = st.S, c = $('mp'), x = fit(c), W = c.width, H = c.height; x.clearRect(0, 0, W, H);
  if (!S || !S.emb) { x.fillStyle = '#888'; x.font = '12px sans-serif'; x.fillText('no embedding: run SUBTLE', 10, 40); return; }
  if (!eb) { let a = 1e9, b = -1e9, d = 1e9, e = -1e9; for (const p of S.emb) { a = Math.min(a, p[0]); b = Math.max(b, p[0]); d = Math.min(d, p[1]); e = Math.max(e, p[1]); }
    eb = {cx: (a + b) / 2, cy: (d + e) / 2, rx: Math.max(1e-9, (b - a) / 2), ry: Math.max(1e-9, (e - d) / 2)}; }
  const key = [W, H, M.zoom, M.px, M.py, $('lvl').value, $('sl').value, [...st.F].join('-'), S.emb.length].join();
  if (!cache || cache.key !== key) { const o = document.createElement('canvas'); o.width = W; o.height = H; const q = o.getContext('2d');
    for (let i = 0; i < S.emb.length; i++) { const [u, v] = xy(S.emb[i], c); q.globalAlpha = inF(i) ? 1 : .08; q.fillStyle = color(lid(i)); q.fillRect(u - 1, v - 1, 2.5, 2.5); }
    cache = {key, o}; }
  x.drawImage(cache.o, 0, 0);
  const [u, v] = xy(S.emb[Math.min(st.fr, S.emb.length - 1)], c); x.strokeStyle = fg(); x.lineWidth = 2; x.beginPath(); x.arc(u, v, 8, 0, 7); x.stroke();
  x.fillStyle = fg(); x.font = '11px sans-serif'; x.fillText(`zoom ${M.zoom.toFixed(2)}x · ${S.emb.length} frames`, 8, H - 8);
}
const mp = $('mp'); let md = null, moved = false;
mp.addEventListener('wheel', e => { e.preventDefault(); const r = mp.getBoundingClientRect(), s = mp.width / r.width, mx = (e.clientX - r.left) * s - mp.width / 2, my = (e.clientY - r.top) * s - mp.height / 2;
  const z = Math.min(50, Math.max(.3, M.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15))), k = z / M.zoom; M.px = mx - (mx - M.px) * k; M.py = my - (my - M.py) * k; M.zoom = z; drawMap(); }, {passive: false});
mp.onmousedown = e => { md = {x: e.clientX, y: e.clientY, px: M.px, py: M.py}; moved = false; };
mp.onclick = e => { if (moved || !st.S || !st.S.emb) return; const r = mp.getBoundingClientRect(), s = mp.width / r.width, mx = (e.clientX - r.left) * s, my = (e.clientY - r.top) * s;
  let best = -1, bd = 1e18; for (let i = 0; i < st.S.emb.length; i++) { const [u, v] = xy(st.S.emb[i], mp), d = (u - mx) ** 2 + (v - my) ** 2; if (d < bd) { bd = d; best = i; } }
  if (best >= 0) { st.fr = best; emit('frame'); } };
mp.ondblclick = () => { M = {zoom: 1, px: 0, py: 0}; drawMap(); };
window.onmouseup = () => { drag = null; md = null; };
window.onmousemove = e => {
  if (drag) { if (drag.pan) { V.px = drag.v.px + (e.clientX - drag.x); V.py = drag.v.py + (e.clientY - drag.y); } else { V.yaw = drag.v.yaw + (e.clientX - drag.x) / 120; V.pitch = drag.v.pitch + (e.clientY - drag.y) / 120; } drawSkel(); }
  if (md) { const r = mp.getBoundingClientRect(), s = mp.width / r.width; if (Math.abs(e.clientX - md.x) + Math.abs(e.clientY - md.y) > 3) moved = true; M.px = md.px + (e.clientX - md.x) * s; M.py = md.py + (e.clientY - md.y) * s; drawMap(); }
};

// ---- timeline: one color per frame, dimmed outside the filter
function drawTl() {
  const c = $('tl'), x = fit(c); x.clearRect(0, 0, c.width, c.height); if (!st.S || !st.S.L) return;
  const n = st.S.xyz.length; for (let i = 0; i < n; i++) { x.globalAlpha = inF(i) ? 1 : .12; x.fillStyle = color(lid(i)); x.fillRect(i / n * c.width, 0, Math.max(1, c.width / n), c.height); }
  x.globalAlpha = 1; x.fillStyle = fg(); x.fillRect(st.fr / n * c.width - 1, 0, 2, c.height);
}
$('tl').onclick = e => { if (!st.S) return; st.fr = Math.floor(e.offsetX / e.target.clientWidth * st.S.xyz.length); emit('frame'); };

on('session', () => { B = bodySize(); V = {yaw: .6, pitch: -.4, zoom: 1, px: 0, py: 0}; M = {zoom: 1, px: 0, py: 0}; eb = null; cache = null; drawSkel(); drawMap(); drawTl(); });
on('frame', () => { drawSkel(); drawMap(); drawTl(); });
on('labels', () => { cache = null; drawSkel(); drawMap(); drawTl(); });
on('filter', () => { cache = null; drawMap(); drawTl(); });
window.addEventListener('resize', () => { cache = null; drawSkel(); drawMap(); drawTl(); });
