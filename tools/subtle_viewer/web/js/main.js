// Entry: session list, open a session, playback, right-panel tabs, manuals.
import {$, st, on, emit, api, toast, FPS, fillLevels, relabel, inF, nextIn} from './state.js';
import {resetView, zoomSkel} from './views.js';
import './legend.js';
import {setCam, setAll, vids, videoOn, toggleKp, toggleConf, isAll} from './video.js';
import {refreshProjects, renderGroups} from './project.js';

async function init() {
  const R = await api('root'); $('root').textContent = R.path;
  $('dataroot').textContent = `Data root: ${R.path}`;
  await api('scan_start');
  let status;
  do {
    status = await api('scan_status');
    $('scanmsg').textContent = `Scanning ${status.path} · ${status.dirs} folders · ${status.sessions} sessions · ${status.videos} videos · ${status.elapsed}s`;
    if (!status.done) await new Promise(resolve => setTimeout(resolve, 350));
  } while (!status.done);
  if (status.error) throw new Error(`Session scan failed: ${status.error}`);
  const L = status.items; $('root').textContent = `${L.length} sessions · ${R.path}`;
  $('scanprogress').remove();
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
  $('sl').innerHTML = ''; fillLevels(); relabel();
  document.querySelectorAll('.sess').forEach(e => e.classList.toggle('on', e.dataset.p === p));
  $('vtitle').textContent = `${p.split('/').pop()} — ${st.S.n} frames`;
  await setCam(); emit('session'); emit('frame');
}
on('reload', () => st.cur && open_(st.cur.path, st.cur, $('camsel').value || null, st.fr));
$('camsel').onchange = () => setCam();
$('allcam').onclick = () => isAll() ? setCam() : setAll();
$('confbtn').onclick = () => { const on = toggleConf(); $('confbtn').classList.toggle('on', on); $('confbtn').setAttribute('aria-pressed', String(on)); };
$('shutdown').onclick = async () => { $('shutdown').disabled = true; try { await api('shutdown', {}); stop(); $('toast').textContent = 'Server stopped. You can close this tab.'; } catch (e) { $('shutdown').disabled = false; toast(e.message); } };
on('cam', () => { $('allcam').classList.toggle('on', isAll()); $('allcam').setAttribute('aria-pressed', String(isAll())); });
$('lvl').onchange = () => { fillLevels(); relabel(); emit('labels'); };
$('sl').onchange = () => { relabel(); emit('labels'); };

// ---- slider and per-frame readout
on('frame', () => { const S = st.S; if (!S) return; $('fr').value = st.fr; $('fi').textContent = `${st.fr * S.stride}/${S.n - 1}`;
  const id = S.L && S.L[st.fr]; $('cur').textContent = id == null ? 'no cluster labels (run SUBTLE)' : `cluster ${id}`; });
$('fr').oninput = e => { st.fr = +e.target.value; emit('frame'); };

// ---- playback: the first video is the clock when shown, else a timer; a filter makes replay skip other clusters.
// Seeking while playing (slider, timeline, map click) is picked up by video.js draw(), which re-syncs every video.
let timer = null, raf = null, playToken = 0;
const playing = () => timer || raf;
function stop() { playToken++; clearInterval(timer); cancelAnimationFrame(raf); timer = raf = null; vids().forEach(v => v.pause()); $('play').textContent = '▶'; }
function toggle() {
  if (playing()) return stop();
  if (!st.S) return; $('play').textContent = '⏸';
  if (videoOn()) { const V = vids(), v = V[0], token = ++playToken;
    V.forEach(x => { const start = () => { if (token !== playToken) return; x.currentTime = st.fr * st.S.stride / FPS(); x.play().catch(e => { if (e.name !== 'AbortError') toast(`Video: ${e.message}`); }); };
      if (x.readyState >= 1) start(); else x.addEventListener('loadedmetadata', start, {once: true}); });
    const tick = () => { if (!v.paused) st.fr = Math.min(st.S.xyz.length - 1, Math.round(v.currentTime * FPS() / st.S.stride));
      if (st.F.size && !inF(st.fr)) st.fr = nextIn(st.fr);
      emit('frame'); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
  } else timer = setInterval(() => { st.fr = nextIn((st.fr + 1) % st.S.xyz.length); emit('frame'); }, 1000 / FPS());
}
$('play').onclick = toggle;
on('cam', () => { if (playing()) { stop(); toggle(); } });

// ---- keyboard, same keys as the BS viewer: Space play, arrows step, +/- zoom, Enter reset view, h panels, ? manuals
const step = d => { if (!st.S) return; st.fr = Math.max(0, Math.min(st.S.xyz.length - 1, st.fr + d)); emit('frame'); };
const tab = t => document.querySelector(`#tabs button[data-t="${t}"]`).click();
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName) && e.target.type !== 'range') return;
  const k = e.key, cams = Object.keys((st.cur && st.cur.cams) || {});
  const A = {' ': toggle, ArrowLeft: () => step(-1), ArrowRight: () => step(1), ArrowUp: () => step(10), ArrowDown: () => step(-10), ',': () => step(-1), '.': () => step(1),
    '+': () => zoomSkel(1.15), '=': () => zoomSkel(1.15), '-': () => zoomSkel(1 / 1.15), Enter: resetView, h: () => document.body.classList.toggle('hide'),
    k: () => toast('keypoints ' + (toggleKp() ? 'on' : 'off')), l: () => tab('clu'), '?': () => tab('help'),
    Escape: () => { st.F = new Set(); emit('filter'); }, 0: () => cams.length ? setAll() : toast('no camera video'),
    1: () => pick(cams, 0), 2: () => pick(cams, 1), 3: () => pick(cams, 2), 4: () => pick(cams, 3), 5: () => pick(cams, 4)};
  if (A[k]) { e.preventDefault(); A[k](); }
});
function pick(cams, c) { if (cams.includes(String(c))) { $('camsel').value = c; setCam(String(c)); } else toast(`cam_${c + 1} has no video here (only ${cams.map(x => 'cam_' + (+x + 1)).join(', ') || 'none'})`); }

// ---- right panel tabs
document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tabpage').forEach(x => x.classList.toggle('on', x.id === 't_' + b.dataset.t));
  document.querySelectorAll('[data-go]').forEach(x => x.classList.toggle('on', x.dataset.go === b.dataset.t));
  $('paneltitle').textContent = b.textContent;
});
document.querySelectorAll('[data-go]').forEach(b => b.onclick = () => { document.querySelector(`#tabs button[data-t="${b.dataset.go}"]`).click(); const section = document.querySelector(`#left [data-go="${b.dataset.go}"]`)?.closest('details'); if (section) section.open = true; });

// A fixed, real AVATAR toy clip: inference runs on eren; this page stays responsive while it runs.
$('tstart').onclick = async () => {
  $('tstart').disabled = true;
  try { const {job} = await api('tutorial_start', {});
    const pollTutorial = async () => { const s = await api('tutorial_status?id=' + job);
      $('tpg').style.display = 'block'; $('tpg').value = s.pct;
      $('tutorial').textContent = `${s.stage} · ${s.pct}% · ${s.elapsed}s${s.error ? ' · ' + s.error : ''}`;
      if (!s.done) setTimeout(pollTutorial, 1000);
      else { $('tstart').disabled = false; if (s.path) $('tutorial').innerHTML += ' · <button id="treload">Reload session list</button>'; $('treload')?.addEventListener('click', () => location.reload()); }
    };
    pollTutorial();
  } catch (e) { $('tstart').disabled = false; $('tutorial').textContent = e.message; }
};

// ---- manuals: docs/dev/subtle_*.md through a small markdown reader
const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const inl = t => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
  .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
  .replace(/\[([^\]]+)\]\((subtle_[\w-]+\.md)\)/g, '<a href="#" data-dlink="$2">$1</a>');
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
  const rank = n => { const i = ['subtle_workthrough.md', 'subtle_viewer_manual.md'].indexOf(n); return i < 0 ? 2 : i; };
  const N = (await api('docs')).sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  $('docnav').innerHTML = N.map(n => `<button type="button" data-d="${n}">${n.replace(/^subtle_|\.md$/g, '').replace(/_/g, ' ')}</button>`).join('');
  document.querySelectorAll('[data-d]').forEach(a => a.onclick = async () => { const d = await api('doc?name=' + a.dataset.d); $('doctext').innerHTML = md(d.text); document.querySelectorAll('[data-d]').forEach(x => x.classList.toggle('on', x === a)); });
  $('doctext').onclick = e => { const a = e.target.closest('[data-dlink]'); if (!a) return; e.preventDefault(); $('docnav').querySelector(`[data-d="${a.dataset.dlink}"]`)?.click(); };
  $('docnav').querySelector('button')?.click();
}
initDocs().catch(() => {});
init().catch(e => toast(e.message));
