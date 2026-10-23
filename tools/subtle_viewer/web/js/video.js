// Camera tiles: one video + keypoint overlay per camera. Key 1-5 = one camera, key 0 = all cameras at once.
// The video letterboxes (object-fit: contain), so points map into the visible picture, not the element box.
// kp2d gives x, y in 0-1 of the camera cell (server: core.kp_cell). Low-confidence points are drawn dim, never dropped silently.
import {$, st, on, emit, api, toast, FPS, fit, lid} from './state.js';
import {color} from './colors.js';

const CELL_ASPECT = {0: 1.2, 1: 1.2, 2: 1.0, 3: 1.2, 4: 1.2};      // cell w/h of the rig (core.CELLS)
let tiles = [], showKp = true, all = false;
export const vids = () => tiles.map(t => t.v);
export const videoOn = () => tiles.length > 0;
export const toggleKp = () => { showKp = !showKp; draw(); return showKp; };
export const camNow = () => st.camNow;

function mk(cam, path) {
  const d = document.createElement('div'); d.className = 'tile';
  d.innerHTML = `<video muted preload="auto"></video><canvas></canvas><span class="tag">cam_${cam + 1}</span>`;
  $('vtiles').appendChild(d);
  const t = {cam, d, v: d.firstChild, c: d.children[1], KP: null};
  t.v.src = '/video?path=' + encodeURIComponent(path); t.v.onseeked = t.v.onloadeddata = draw;
  t.v.onloadedmetadata = () => { if (Math.abs(t.v.videoWidth / t.v.videoHeight - CELL_ASPECT[cam]) > .02) $('vwarn').textContent = `cam_${cam + 1}: aspect ${(t.v.videoWidth / t.v.videoHeight).toFixed(2)} ≠ rig cell`; draw(); };
  d.onclick = () => setCam(cam);                                     // click a tile in the grid = pick that camera
  api(`kp2d?path=${encodeURIComponent(st.cur.path)}&cam=${cam}`).then(r => { t.KP = r.kp; draw(); }).catch(e => toast(e.message));
  return t;
}

export async function setCam(sel = $('camsel').value, wantAll = false) {
  const cams = (st.cur && st.cur.cams) || {};
  for (const t of tiles) t.d.remove(); tiles = []; $('vwarn').textContent = ''; all = wantAll;
  const use = wantAll ? Object.keys(cams) : (cams[sel] != null ? [sel] : []);
  if (!use.length) { $('vnone').style.display = 'flex'; st.camNow = sel === '' ? null : +sel; $('camsel').value = sel; emit('cam'); return; }
  $('vnone').style.display = 'none'; st.camNow = +use[0]; if (!wantAll) $('camsel').value = sel;
  $('vtiles').dataset.n = use.length; tiles = use.map(c => mk(+c, cams[c]));
  emit('cam'); draw();
}
export const setAll = () => setCam(undefined, true);
export const isAll = () => all;

export function draw() {
  const S = st.S, d = window.devicePixelRatio || 1;
  for (const t of tiles) {
    const o = t.c, x = fit(o), v = t.v; x.clearRect(0, 0, o.width, o.height);
    if (!S || !v.videoWidth) continue;
    const tt = st.fr * S.stride;
    if (v.paused && v.readyState > 0 && Math.abs(v.currentTime - tt / FPS()) > .5 / FPS()) v.currentTime = tt / FPS();
    if (!v.paused && Math.abs(v.currentTime - tt / FPS()) > .3) v.currentTime = tt / FPS();   // scrub or filter jump while playing
    if (!showKp || !t.KP || !t.KP[tt]) continue;
    const W = o.width, H = o.height, sc = Math.min(W / v.videoWidth, H / v.videoHeight), vw = v.videoWidth * sc, vh = v.videoHeight * sc, ox = (W - vw) / 2, oy = (H - vh) / 2;
    const a = t.KP[tt], hi = p => p[2] > .25, seen = p => p[0] > 0 || p[1] > 0;      // missing points are (0,0) with conf 0
    const pt = p => [ox + p[0] * vw, oy + p[1] * vh];
    x.lineWidth = 2 * d; x.strokeStyle = color(lid(st.fr));
    for (const [i, j] of S.edges) { if (!seen(a[i]) || !seen(a[j])) continue;
      x.globalAlpha = hi(a[i]) && hi(a[j]) ? 1 : .45; x.setLineDash(hi(a[i]) && hi(a[j]) ? [] : [5 * d, 4 * d]);
      x.beginPath(); x.moveTo(...pt(a[i])); x.lineTo(...pt(a[j])); x.stroke(); }
    x.setLineDash([]);
    for (const p of a) { if (!seen(p)) continue; const [u, w] = pt(p);
      if (hi(p)) { x.globalAlpha = 1; x.fillStyle = '#2ecc71'; x.beginPath(); x.arc(u, w, 4 * d, 0, 7); x.fill(); }
      else { x.globalAlpha = .6; x.strokeStyle = '#2ecc71'; x.lineWidth = 1.5 * d; x.beginPath(); x.arc(u, w, 4 * d, 0, 7); x.stroke(); } }
    x.globalAlpha = 1;
  }
}
on('frame', draw); on('labels', draw); on('session', draw);
window.addEventListener('resize', draw);
