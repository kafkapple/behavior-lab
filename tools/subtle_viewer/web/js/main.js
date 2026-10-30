// Entry: session list, open a session, playback, right-panel tabs, manuals.
import {$, st, on, emit, api, toast, FPS, fillLevels, relabel, inF, nextIn} from './state.js';
import {resetView, zoomSkel} from './views.js';
import './legend.js';
import {setCam, setAll, vids, videoOn, toggleKp, toggleConf, isAll} from './video.js';
import {refreshProjects, renderGroups} from './project.js';
import {showReports} from './reports.js';
import {palette, setPalette, partOf} from './kp_palette.js';
import {color} from './colors.js';
let sessionIndex = [];

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
  const L = status.items; sessionIndex = L; $('root').textContent = `${L.length} sessions · ${R.path}`;
  if (!L.some(s => s.path === 'cort/cort1_full_sleap_1423.json.gz')) { $('tstart').disabled = true; $('tstart').title = 'The CORT toy runs only under a CORT data root (run subtle with no arguments).'; }
  $('scanprogress').remove();
  $('list').innerHTML = L.map(s => `<div class="sess" data-p="${s.path}"><b>${s.stem}</b> <span class="mut">${s.mb} MB${Object.keys(s.cams).length ? ' · ' + Object.keys(s.cams).length + ' cam' : ''}${s.subtle ? ' · SUBTLE' : ''}</span></div>`).join('') || 'no *.json.gz under the root';
  document.querySelectorAll('.sess').forEach(e => e.onclick = () => open_(e.dataset.p, L.find(x => x.path === e.dataset.p)));
  refreshProjects();
  const h = new URLSearchParams(location.hash.slice(1));              // deep link: #open=<path>&cam=1&frame=600
  const m = h.get('open') && L.find(x => x.path === h.get('open'));
  if (m) await open_(m.path, m, h.get('cam'), +h.get('frame') || 0);
  else if (L.length) await open_(L[0].path, L[0]);
  if (m && h.get('all')) await setAll();                              // #all=1: every camera tile
  if (h.get('project')) { st.proj = await api('project?name=' + encodeURIComponent(h.get('project'))); $('pn').value = st.proj.name; renderGroups(); }   // #project=<name>
  if (h.get('run')) $({stat: 'stgo', dek: 'dkgo'}[h.get('run')]).click();                   // #run=stat|dek: compute right away (checks)
  if (h.get('tab')) document.querySelector(`#tabs button[data-t="${h.get('tab')}"]`)?.click();   // #tab=proj|freq|stat|log
}
async function open_(p, meta, cam, frame) {
  toast('loading ' + p); st.cur = {path: p, ...meta}; st.S = await api('session?path=' + encodeURIComponent(p));
  st.fr = Math.min(frame || 0, st.S.xyz.length - 1); st.F = new Set(); st.A = new Set(); st.ACat = ''; $('fr').max = st.S.xyz.length - 1;
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
$('kpal').value = palette;
function showPalette() {
  const mode = $('kpal').value, nodes = st.S?.nodes || ['nose1', 'neck1', 'earL1', 'earR1', 'forelegL1', 'forelegR1', 'tailstart1', 'hindlegL1', 'hindlegR1', 'tail1', 'tailend1'];
  $('kpalmeaning').textContent = mode === 'body' ?
    'BS body-part colors are fixed by anatomy. An edge takes its destination joint’s color; a color change along the skeleton is not a behavior transition.' :
    'Legacy mode: 2D points stay green; edges and the 3D skeleton take the current cluster ID color. Color changes over time mean the selected cluster label changed.';
  $('kpallegend').innerHTML = mode === 'body' ? nodes.map((name, i) => { const p = partOf(nodes, i); return `<span><i style="background:${p?.[1] || '#7ee6b1'}"></i>${name} · ${p?.[0] || 'unmapped'}</span>`; }).join('') :
    '<span><i style="background:#7ee6b1"></i>2D point</span><span>Edges and 3D: current cluster color; see the cluster badge.</span>';
}
$('kpal').onchange = () => { setPalette($('kpal').value); showPalette(); emit('palette'); };
on('session', showPalette); showPalette();
$('shutdown').onclick = async () => { $('shutdown').disabled = true; try { await api('shutdown', {}); stop(); $('toast').textContent = 'Server stopped. You can close this tab.'; } catch (e) { $('shutdown').disabled = false; toast(e.message); } };
on('cam', () => { $('allcam').classList.toggle('on', isAll()); $('allcam').setAttribute('aria-pressed', String(isAll())); });
$('lvl').onchange = () => { fillLevels(); relabel(); emit('labels'); emit('frame'); };
$('sl').onchange = () => { relabel(); emit('labels'); emit('frame'); };

// ---- slider and per-frame readout
on('frame', () => { const S = st.S; if (!S) return; $('fr').value = st.fr; $('fi').textContent = `${st.fr * S.stride}/${S.n - 1}`;
  const id = S.L && S.L[st.fr]; $('cur').textContent = id == null ? 'no cluster labels (run SUBTLE)' : `cluster ${id}`;
  $('anncur').textContent = st.ACat && S.annotations?.[st.ACat] ? `${st.ACat}: ${S.annotations[st.ACat][st.fr]}` : '';
  $('clusterbadge').textContent = id == null ? 'No cluster label' : `Cluster ${id}`;
  $('clusterbadge').style.borderColor = id == null ? 'var(--line)' : color(id);
  $('clusterbadge').style.color = id == null ? 'var(--mut)' : color(id); });
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
    let lastMedia = v.currentTime, lastAdvance = performance.now(), lastTick = lastAdvance, fallbackFrames = 0;
    V.forEach(x => { const start = () => { if (token !== playToken) return; x.currentTime = st.fr * st.S.stride / FPS(); x.play().catch(e => { if (e.name !== 'AbortError') toast(`Video: ${e.message}`); }); };
      if (x.readyState >= 1) start(); else x.addEventListener('loadedmetadata', start, {once: true}); });
    const tick = () => { const now = performance.now(), media = v.currentTime;
      if (!v.paused && media > lastMedia + .001) { st.fr = Math.min(st.S.xyz.length - 1, Math.max(st.fr, Math.round(media * FPS() / st.S.stride))); lastAdvance = now; fallbackFrames = 0; }
      else if (now - lastAdvance > 500) { fallbackFrames += Math.max(0, (now - lastTick) * FPS() / 1000 / st.S.stride); const step = Math.floor(fallbackFrames); st.fr = Math.min(st.S.xyz.length - 1, st.fr + step); fallbackFrames -= step; }
      lastMedia = media; lastTick = now;
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
  $('right').classList.toggle('wide', ['stat', 'dek', 'qual', 'reports'].includes(b.dataset.t));
  $('panelcompact').style.display = ['stat', 'dek', 'qual', 'reports'].includes(b.dataset.t) ? '' : 'none';
  if (b.dataset.t === 'reports') showReports();
});
$('panelcompact').onclick = () => { $('right').classList.remove('wide'); $('panelcompact').style.display = 'none'; };
document.querySelectorAll('[data-go]').forEach(b => b.onclick = () => { document.querySelector(`#tabs button[data-t="${b.dataset.go}"]`).click(); const section = document.querySelector(`#left [data-go="${b.dataset.go}"]`)?.closest('details'); if (section) section.open = true; });

// A fixed, real AVATAR toy clip: inference runs on eren; this page stays responsive while it runs.
$('topen').onclick = async () => {
  const meta = sessionIndex.find(s => s.path === 'tutorial/cort1_toy.json.gz');
  if (!meta) return toast('Toy output is not in this data root. Press Predict keypoints again.');
  await open_(meta.path, meta, null, 0);
  $('tutorial').textContent = 'Completed toy opened: inspect video, Conf %, and 3D skeleton. Then create a project and follow Discover & map.';
};
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
$('tdemo').onclick = async () => {
  $('tdemo').disabled = true;
  try {
    const p = await api('project?name=cort_vs_veh');
    st.proj = p; $('pn').value = p.name; $('projsaved').textContent = `Loaded: ${p.name} · last saved ${p.updated || 'unknown'}`;
    renderGroups();
    const first = Object.values(p.groups).flat()[0];
    const meta = sessionIndex.find(s => s.path === first);
    if (meta) await open_(first, meta, null, 0);
    document.querySelector('#tabs button[data-t="stat"]').click();
    $('stgo').click();
    $('tdemostatus').textContent = 'Project loaded. Compare groups is running in Stats. Open DEK next to calculate kinematic differences.';
  } catch (e) { $('tdemostatus').textContent = e.message; }
  finally { $('tdemo').disabled = false; }
};
$('g3review').onclick = async () => {
  try {
    const p = await api('project?name=g3_juhee_source_review');
    st.proj = p; $('pn').value = p.name; $('projsaved').textContent = `Loaded: ${p.name}`; renderGroups();
    const meta = sessionIndex.find(s => s.path === p.groups.unassigned[0]);
    if (meta) await open_(meta.path, meta, null, 0);
    document.querySelector('#tabs button[data-t="qual"]').click(); $('qgo').click();
  } catch (e) { toast(e.message); }
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
