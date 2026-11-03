// Model comparison inside one viewer: pick a second result (B) for the open session (A), overlay B on the video, and run the agreement statistics.
// Self-contained: it injects its own left-menu entry, tab button and page at import time (index.html is not edited). Server: compare.py via /api/compare.
import {$, st, on, emit, api, toast, fit} from './state.js';
import {overlays} from './views.js';

const col = () => ($('cmpcol') && $('cmpcol').value) || '#ff9f0a';
document.querySelector('#left').insertAdjacentHTML('beforeend', '<details open><summary>Compare models</summary><button data-go="cmp">Compare two results on one video →</button></details>');
document.querySelector('#tabs').insertAdjacentHTML('beforeend', '<button data-t="cmp">Compare</button>');
document.querySelector('#right').insertAdjacentHTML('beforeend', `<div class="tabpage" id="t_cmp">
  <h3>Compare two keypoint results</h3>
  <p class="mut">A = the session open now. B = another <code>*.json.gz</code> of the same video (for example another model). Agreement only: there is no ground truth, so this does not say which is right.</p>
  <label class="mut">A (opens in the viewer)</label><select id="cmpa" style="width:100%;margin-bottom:6px"></select>
  <label class="mut">B (orange ring on the video)</label>
  <div style="display:flex;gap:6px;margin-bottom:6px"><select id="cmpb" style="flex:1;min-width:0"></select><button id="cmprefresh" title="re-read the session list">↻</button></div>
  <label class="mut"><input type="checkbox" id="cmpov" checked style="width:auto"> draw B on the video and in 3D</label> <label class="mut">B color <input type="color" id="cmpcol" value="#ff9f0a" style="width:36px;padding:0"></label>
  <div style="display:flex;gap:6px;margin:6px 0"><button id="cmpgo">Compare A and B</button><button id="cmpall" title="A against every other session in the list: one summary row each">A vs all others</button><button id="cmpsave" title="save compare.png, compare.json, node_stats.csv under the results folder">Save results</button></div>
  <div id="cmpres" class="mut">open a session, choose B, press Compare</div></div>`);
document.querySelector('#tabs button[data-t="cmp"]').addEventListener('click', () => { fillB(); setTimeout(() => $('right').classList.add('wide'), 0); });

let items = [], KB = {}, last = null;                     // KB: B's 2D keypoints per camera, keyed path|cam
async function fillB() {
  items = (await api('scan_status')).items || [];
  $('cmpa').innerHTML = items.map(s => `<option value="${s.path}">${s.path}</option>`).join('');
  if (st.cur) $('cmpa').value = st.cur.path;
  $('cmpb').innerHTML = items.filter(s => !st.cur || s.path !== st.cur.path).map(s => `<option value="${s.path}">${s.path}</option>`).join('') || '<option value="">no other session</option>';
}
$('cmprefresh').onclick = fillB;
$('cmpa').onchange = () => { document.querySelector(`.sess[data-p="${CSS.escape($('cmpa').value)}"]`)?.click(); setTimeout(fillB, 800); };   // choosing A opens it, B list drops it

const star = q => q == null ? '' : q < .001 ? '***' : q < .01 ? '**' : q < .05 ? '*' : 'ns';
async function run(save) {
  if (!st.cur) return toast('open session A first');
  const b = $('cmpb').value; if (!b) return toast('choose B');
  $('cmpres').textContent = 'computing…';
  try {
    const R = await api(`compare?a=${encodeURIComponent(st.cur.path)}&b=${encodeURIComponent(b)}&save=${save ? 1 : 0}`); last = R;
    const x = R.xyz, t = R.nodes.map(o => `<tr><td>${o.node}</td><td>${o.median_px}</td><td>${o.ci[0]}–${o.ci[1]}</td><td>${o.conf_a}</td><td>${o.conf_b}</td><td>${o.p_conf_bh == null ? '' : o.p_conf_bh < .0001 ? '&lt;.0001' : o.p_conf_bh} ${star(o.p_conf_bh)}</td></tr>`).join('');
    $('cmpres').innerHTML = `<p><b>${R.frames} frames</b> · 2D median distance <b>${R.median_px_all} px</b> (95% CI ${R.ci_all[0]}–${R.ci_all[1]}, cell px)${x ? ` · 3D <b>${x.median_pct_extent}%</b> of body extent (CI ${x.ci_pct[0]}–${x.ci_pct[1]})` : ''}</p>
      <img src="data:image/png;base64,${R.png}" style="width:100%">
      <table><tr><th>node</th><th>median px</th><th>95% CI</th><th>conf A</th><th>conf B</th><th>p (BH)</th></tr>${t}</table>
      <p class="mut">${R.caveat}</p>${R.saved ? `<p>saved: <code>${R.saved}</code></p>` : ''}`;
    draw();
  } catch (e) { $('cmpres').textContent = e.message; }
}
$('cmpgo').onclick = () => run(false);
$('cmpall').onclick = async () => {
  if (!st.cur) return toast('open session A first');
  const bs = [...$('cmpb').options].map(o => o.value).filter(Boolean); if (!bs.length) return toast('no other session');
  $('cmpres').textContent = `computing ${bs.length} pairs…`;
  try {
    const rows = await api(`compare_all?a=${encodeURIComponent(st.cur.path)}&bs=${encodeURIComponent(bs.join('|'))}`);
    $('cmpres').innerHTML = `<p><b>A = ${st.cur.path}</b> against ${rows.length} sessions. Click a row to choose it as B.</p><table><tr><th>B</th><th>nodes</th><th>2D median px (95% CI)</th><th>both confident</th><th>3D % extent</th></tr>` + rows.map(r => r.error
      ? `<tr><td>${r.b}</td><td colspan="4" class="mut">${r.error}</td></tr>`
      : `<tr class="cmprow" data-b="${r.b}" style="cursor:pointer"><td>${r.b}</td><td>${r.nodes}${r.dropped.a_only.length || r.dropped.b_only.length ? ' (shared)' : ''}</td><td>${r.median_px} (${r.ci[0]}–${r.ci[1]})</td><td>${r.both}</td><td>${r.xyz_pct == null ? '' : r.xyz_pct}</td></tr>`).join('') + '</table>';
    document.querySelectorAll('.cmprow').forEach(e => e.onclick = () => { $('cmpb').value = e.dataset.b; draw(); run(false); });
  } catch (e) { $('cmpres').textContent = e.message; }
};
$('cmpsave').onclick = () => run(true);

// ---- B on the video: one extra canvas per tile, same letterbox mapping as video.js
async function kpB(cam) {
  const b = $('cmpb').value, k = `${b}|${cam}`; if (!b) return null;
  if (!(k in KB)) { KB[k] = null; try { KB[k] = (await api(`kp2d?path=${encodeURIComponent(b)}&cam=${cam}`)).kp; draw(); } catch (e) { toast(e.message); } }
  return KB[k];
}
function draw() {
  for (const tile of document.querySelectorAll('#vtiles .tile')) {
    let o = tile.querySelector('canvas.cmp');
    if (!o) { o = document.createElement('canvas'); o.className = 'cmp'; o.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none'; tile.appendChild(o); }
    const x = fit(o), v = tile.querySelector('video'), S = st.S; x.clearRect(0, 0, o.width, o.height);
    const cam = +tile.querySelector('.tag').textContent.replace('cam_', '') - 1, on_ = $('cmpov') && $('cmpov').checked;
    if (!on_ || !S || !v.videoWidth || !$('cmpb').value) continue;
    const KP = KB[`${$('cmpb').value}|${cam}`]; if (KP === undefined) { kpB(cam); continue; } if (!KP) continue;
    const a = KP[st.fr * S.stride]; if (!a) continue;
    const d = window.devicePixelRatio || 1, W = o.width, H = o.height, sc = Math.min(W / v.videoWidth, H / v.videoHeight), vw = v.videoWidth * sc, vh = v.videoHeight * sc, ox = (W - vw) / 2, oy = (H - vh) / 2;
    const pt = p => [ox + p[0] * vw, oy + p[1] * vh], seen = p => p[2] > .05, hi = p => p[2] > .25;
    x.strokeStyle = col(); x.lineWidth = 1.5 * d;
    for (const [i, j] of S.edges) { if (!seen(a[i]) || !seen(a[j])) continue; x.globalAlpha = hi(a[i]) && hi(a[j]) ? .9 : .4; x.beginPath(); x.moveTo(...pt(a[i])); x.lineTo(...pt(a[j])); x.stroke(); }
    for (const p of a) { if (!seen(p)) continue; const [u, w] = pt(p); x.globalAlpha = 1; x.lineWidth = 2 * d; x.beginPath(); x.arc(u, w, 6 * d, 0, 7); x.stroke(); }
    x.globalAlpha = 1;
  }
}
on('frame', draw); on('cam', draw); on('session', draw);
$('cmpov').onchange = () => { draw(); on3d(); }; $('cmpb').onchange = () => { draw(); X3 = null; on3d(); }; $('cmpcol').oninput = () => { draw(); on3d(); };

// ---- B in the 3D view: B aligned to A (server: compare.aligned), drawn through the overlay hook of views.js
let X3 = null;
async function load3d() {
  const b = $('cmpb').value, key = `${st.cur && st.cur.path}|${b}`; if (!b || !st.cur || (X3 && X3.key === key)) return;
  X3 = {key, d: null}; try { X3.d = await api(`compare_xyz?a=${encodeURIComponent(st.cur.path)}&b=${encodeURIComponent(b)}&stride=${st.S.stride}`); on3d(); } catch (e) { X3 = null; toast('3D overlay: ' + e.message); }
}
const on3d = () => emit('frame');                        // redraw through the normal frame event
overlays.push((x, P, dpr) => {
  if (!$('cmpov') || !$('cmpov').checked || !$('cmpb').value) return;
  if (!X3 || !X3.d) { load3d(); return; }
  const f = X3.d.xyz[st.fr]; if (!f) return;
  x.save(); x.strokeStyle = col(); x.lineWidth = 2 * dpr; x.globalAlpha = .95;
  for (const [i, j] of X3.d.edges) { if (!f[i] || !f[j]) continue; const p = P(f[i]), q = P(f[j]); x.beginPath(); x.moveTo(...p); x.lineTo(...q); x.stroke(); }
  for (const j of f) { const p = P(j); x.beginPath(); x.arc(p[0], p[1], 5 * dpr, 0, 7); x.stroke(); }
  x.restore();
});
on('session', () => { X3 = null; });
window.addEventListener('resize', draw);
