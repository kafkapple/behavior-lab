// Entry: session list, open a session, playback, right-panel tabs, manuals.
import {$, st, on, emit, api, toast, FPS, fillLevels, relabel, inF, nextIn} from './state.js';
import {resetView, zoomSkel} from './views.js';
import './legend.js';
import {setCam, setAll, vids, videoOn, toggleKp, isAll} from './video.js';
import {refreshProjects, renderGroups} from './project.js';

async function init() {
  const L = await api('sessions'); $('root').textContent = L.length + ' sessions';
  $('list').innerHTML = L.map(s => `<div class="sess" data-p="${s.path}"><b>${s.stem}</b> <span class="mut">${s.mb} MB${Object.keys(s.cams).length ? ' · ' + Object.keys(s.cams).length + ' cam' : ''}${s.subtle ? ' · SUBTLE' : ''}</span></div>`).join('') || 'no *.json.gz under the root';
  document.querySelectorAll('.sess').forEach(e => e.onclick = () => open_(e.dataset.p, L.find(x => x.path === e.dataset.p)));
  refreshProjects();
  const h = new URLSearchParams(location.hash.slice(1));              // deep link: #open=<path>&cam=1&frame=600
  const m = h.get('open') && L.find(x => x.path === h.get('open'));
  if (m) await open_(m.path, m, h.get('cam'), +h.get('frame') || 0);
  if (m && h.get('all')) await setAll();                              // #all=1: every camera tile
  if (h.get('project')) { st.proj = await api('project?name=' + encodeURIComponent(h.get('project'))); $('pn').value = st.proj.name; renderGroups(); }   // #project=<name>
  if (h.get('run')) $({stat: 'stgo', dek: 'dkgo'}[h.get('run')]).click();                   // #run=stat|dek: compute right away (checks)
  if (h.get('tab')) document.querySelector(`#tabs button[data-t="${h.get('tab')}"]`)?.click();   // #tab=proj|freq|stat|log
}
async function open_(p, meta, cam, frame) {
  toast('loading ' + p); st.cur = {path: p, ...meta}; st.S = await api('session?path=' + encodeURIComponent(p));
  st.fr = Math.min(frame || 0, st.S.xyz.length - 1); st.F = new Set(); $('fr').max = st.S.xyz.length - 1;
  const cams = meta.cams || {}, sel = $('camsel');
  sel.innerHTML = Object.keys(cams).map(c => `<option value="${c}">cam_${+c + 1}</option>`).join('') || '<option value="">no camera</option>';
  if (cam != null && cams[cam]) sel.value = cam;
  fillLevels(); relabel();
  document.querySelectorAll('.sess').forEach(e => e.classList.toggle('on', e.dataset.p === p));
  $('vtitle').textContent = `${p.split('/').pop()} — ${st.S.n} frames`;
  await setCam(); emit('session'); emit('frame');
}
on('reload', () => st.cur && open_(st.cur.path, st.cur, $('camsel').value || null, st.fr));
$('camsel').onchange = () => setCam();
$('lvl').onchange = () => { fillLevels(); relabel(); emit('labels'); };
$('sl').onchange = () => { relabel(); emit('labels'); };

// ---- slider and per-frame readout
on('frame', () => { const S = st.S; if (!S) return; $('fr').value = st.fr; $('fi').textContent = `${st.fr * S.stride}/${S.n - 1}`;
  const id = S.L && S.L[st.fr]; $('cur').textContent = id == null ? 'no cluster labels (run SUBTLE)' : `cluster ${id}`; });
$('fr').oninput = e => { st.fr = +e.target.value; emit('frame'); };

// ---- playback: the first video is the clock when shown, else a timer; a filter makes replay skip other clusters.
// Seeking while playing (slider, timeline, map click) is picked up by video.js draw(), which re-syncs every video.
let timer = null, raf = null;
const playing = () => timer || raf;
function stop() { clearInterval(timer); cancelAnimationFrame(raf); timer = raf = null; vids().forEach(v => v.pause()); $('play').textContent = '▶'; }
function toggle() {
  if (playing()) return stop();
  if (!st.S) return; $('play').textContent = '⏸';
  if (videoOn()) { const V = vids(), v = V[0]; V.forEach(x => { x.currentTime = st.fr * st.S.stride / FPS(); x.play().catch(() => {}); });
    const tick = () => { if (!v.paused) st.fr = Math.min(st.S.xyz.length - 1, Math.round(v.currentTime * FPS() / st.S.stride));
      if (st.F.size && !inF(st.fr)) st.fr = nextIn(st.fr);
      emit('frame'); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  } else timer = setInterval(() => { st.fr = nextIn((st.fr + 1) % st.S.xyz.length); emit('frame'); }, 1000 / FPS());
}
$('play').onclick = toggle;
on('cam', () => { if (playing()) stop(); });

// ---- keyboard, same keys as the BS viewer: Space play, arrows step, +/- zoom, Enter reset view, h panels, ? manuals
const step = d => { if (!st.S) return; st.fr = Math.max(0, Math.min(st.S.xyz.length - 1, st.fr + d)); emit('frame'); };
const tab = t => document.querySelector(`#tabs button[data-t="${t}"]`).click();
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName) && e.target.type !== 'range') return;
  const k = e.key, cams = Object.keys((st.cur && st.cur.cams) || {});
  const A = {' ': toggle, ArrowLeft: () => step(-1), ArrowRight: () => step(1), ArrowUp: () => step(10), ArrowDown: () => step(-10), ',': () => step(-1), '.': () => step(1),
    '+': () => zoomSkel(1.15), '=': () => zoomSkel(1.15), '-': () => zoomSkel(1 / 1.15), Enter: resetView, h: () => document.body.classList.toggle('hide'),
    k: () => toast('keypoints ' + (toggleKp() ? 'on' : 'off')), l: () => tab('clu'), '?': () => document.querySelector('[data-d]') && document.querySelector('[data-d]').click(),
    Escape: () => { st.F = new Set(); emit('filter'); }, 0: () => cams.length ? setAll() : toast('no camera video'),
    1: () => pick(cams, 0), 2: () => pick(cams, 1), 3: () => pick(cams, 2), 4: () => pick(cams, 3), 5: () => pick(cams, 4)};
  if (A[k]) { e.preventDefault(); A[k](); }
});
function pick(cams, c) { if (cams.includes(String(c))) { $('camsel').value = c; setCam(String(c)); } else toast(`cam_${c + 1} has no video here (only ${cams.map(x => 'cam_' + (+x + 1)).join(', ') || 'none'})`); }

// ---- right panel tabs
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tabpage').forEach(x => x.classList.toggle('on', x.id === 't_' + b.dataset.t)); });

// ---- manuals: docs/dev/subtle_*.md through a small markdown reader
const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const inl = t => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
function md(src) {
  const out = [], L = src.split('\n'); let i = 0, list = false; const close = () => { if (list) { out.push('</ul>'); list = false; } };
  while (i < L.length) { const l = L[i];
    if (l.startsWith('```')) { close(); const b = []; i++; while (i < L.length && !L[i].startsWith('```')) b.push(L[i++]); out.push('<pre>' + esc(b.join('\n')) + '</pre>'); i++; continue; }
    if (/^\|.*\|$/.test(l) && /^\|[ :\-|]+\|$/.test(L[i + 1] || '')) { close(); const row = r => r.slice(1, -1).split('|').map(c => c.trim());
      let t = '<table><tr>' + row(l).map(c => '<th>' + inl(c) + '</th>').join('') + '</tr>'; i += 2;
      while (i < L.length && /^\|.*\|$/.test(L[i])) t += '<tr>' + row(L[i++]).map(c => '<td>' + inl(c) + '</td>').join('') + '</tr>'; out.push(t + '</table>'); continue; }
    const m = l.match(/^(#{1,3}) (.*)/); if (m) { close(); out.push(`<h${m[1].length}>${inl(m[2])}</h${m[1].length}>`); i++; continue; }
    const b = l.match(/^\s*[-*] (.*)/); if (b) { if (!list) { out.push('<ul>'); list = true; } out.push('<li>' + inl(b[1]) + '</li>'); i++; continue; }
    close(); if (l.trim()) out.push('<p>' + inl(l) + '</p>'); i++; }
  close(); return out.join('\n');
}
async function initDocs() {
  const N = await api('docs'); $('docs').innerHTML = N.map(n => `<a href="#" data-d="${n}">${n.replace(/^subtle_|\.md$/g, '').replace(/_/g, ' ')}</a>`).join(' · ') || 'none';
  document.querySelectorAll('[data-d]').forEach(a => a.onclick = async e => { e.preventDefault(); const d = await api('doc?name=' + a.dataset.d); $('doctext').innerHTML = md(d.text); $('docview').style.display = 'block'; });
}
$('docx').onclick = () => $('docview').style.display = 'none';
initDocs().catch(() => {});
init().catch(e => toast(e.message));
