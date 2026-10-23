// Per-camera video with 2D keypoints drawn on it. The video letterboxes (object-fit: contain), so points map into the
// visible picture, not the element box. kp2d gives x, y in 0-1 of the camera cell (server: core.kp_cell).
import {$, st, on, api, toast, FPS, fit, lid} from './state.js';
import {color} from './colors.js';

const CELL_ASPECT = {0: 1.2, 1: 1.2, 2: 1.0, 3: 1.2, 4: 1.2};      // cell w/h of the rig (core.CELLS)
export const videoOn = () => st.camNow != null && $('vid').style.display !== 'none';

export async function setCam() {
  const sel = $('camsel').value, v = $('vid'), cams = (st.cur && st.cur.cams) || {};
  $('vwarn').textContent = ''; st.KP = null;
  if (sel === '' || !cams[sel]) { v.style.display = 'none'; $('vnone').style.display = 'flex'; v.removeAttribute('src'); st.camNow = null; draw(); return; }
  st.camNow = +sel; v.style.display = 'block'; $('vnone').style.display = 'none';
  v.src = '/video?path=' + encodeURIComponent(cams[sel]);
  v.onloadedmetadata = () => { const a = v.videoWidth / v.videoHeight;
    if (Math.abs(a - CELL_ASPECT[st.camNow]) > .02) $('vwarn').textContent = `aspect ${a.toFixed(2)} ≠ rig cell ${CELL_ASPECT[st.camNow]}: overlay may be off`; draw(); };
  v.onseeked = v.onloadeddata = draw;
  try { st.KP = (await api(`kp2d?path=${encodeURIComponent(st.cur.path)}&cam=${st.camNow}`)).kp; } catch (e) { toast(e.message); }
  draw();
}

export function draw() {
  const v = $('vid'), o = $('ov'), x = fit(o); x.clearRect(0, 0, o.width, o.height);
  const S = st.S; if (!S || !videoOn() || !v.videoWidth) return;
  const t = st.fr * S.stride;
  if (v.paused && v.readyState > 0 && Math.abs(v.currentTime - t / FPS()) > .5 / FPS()) v.currentTime = t / FPS();
  if (!st.KP || !st.KP[t]) return;
  const W = o.width, H = o.height, sc = Math.min(W / v.videoWidth, H / v.videoHeight), vw = v.videoWidth * sc, vh = v.videoHeight * sc, ox = (W - vw) / 2, oy = (H - vh) / 2;
  const a = st.KP[t], ok = p => p[2] > .25, d = window.devicePixelRatio || 1;
  x.lineWidth = 2 * d; x.strokeStyle = color(lid(st.fr));
  for (const [i, j] of S.edges) if (ok(a[i]) && ok(a[j])) { x.beginPath(); x.moveTo(ox + a[i][0] * vw, oy + a[i][1] * vh); x.lineTo(ox + a[j][0] * vw, oy + a[j][1] * vh); x.stroke(); }
  x.fillStyle = '#2ecc71'; for (const p of a) if (ok(p)) { x.beginPath(); x.arc(ox + p[0] * vw, oy + p[1] * vh, 4 * d, 0, 7); x.fill(); }
}
on('frame', draw); on('labels', draw); on('session', draw);
window.addEventListener('resize', draw);
