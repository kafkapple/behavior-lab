// Project: groups, save and load, the SUBTLE fit job, and the frequency table with its sum check.
import {$, st, on, emit, api, toast, nkey, cname} from './state.js';
import {color} from './colors.js';

export async function refreshProjects() { $('pl').innerHTML = (await api('projects')).map(n => `<option>${n}</option>`).join(''); }
// meta (PRD R6): st.proj.meta = {description, group_notes: {group: text}, sessions: {path: {animal, condition, date}}}; the server cleans it on save
const M = () => (st.proj.meta = Object.assign({description: '', group_notes: {}, sessions: {}}, st.proj.meta));
export function renderGroups() {
  const g = st.proj.groups, m = M(); $('pdesc').value = m.description || '';
  $('groups').innerHTML = Object.entries(g).map(([n, l]) => `<div class="grpbox"><b>${n}</b> <span class="mut">(${l.length})</span> <button data-del="${n}">×</button>` +
    `<input class="gnote" data-g="${n}" placeholder="group note" value="${(m.group_notes[n] || '').replace(/"/g, '&quot;')}">` +
    l.map(p => { const s = m.sessions[p] || {}, f = k => `<input class="sm" data-s="${p}" data-k="${k}" placeholder="${k === 'condition' ? 'cond.' : k}" title="${k}" value="${(s[k] || '').replace(/"/g, '&quot;')}">`;
      return `<div class="srow"><span class="mut">${p.split('/').pop()}</span>${f('animal')}${f('condition')}${f('date')}</div>`; }).join('') + '</div>').join('');
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { delete g[b.dataset.del]; delete M().group_notes[b.dataset.del]; renderGroups(); freq(); });
  document.querySelectorAll('.gnote').forEach(i => i.oninput = () => { M().group_notes[i.dataset.g] = i.value; });
  document.querySelectorAll('input.sm').forEach(i => i.oninput = () => { ((M().sessions[i.dataset.s] ??= {})[i.dataset.k] = i.value); });
}
$('pdesc').oninput = () => { M().description = $('pdesc').value; };
$('ga').onclick = () => { const n = $('gn').value.trim(); if (!n || !st.cur) return toast('group name + open session needed');
  const g = (st.proj.groups[n] ??= []); if (!g.includes(st.cur.path)) g.push(st.cur.path); renderGroups(); freq(); };
$('ps').onclick = async () => { st.proj.name = $('pn').value.trim(); try { await api('save_project', st.proj); toast('saved ' + st.proj.name); refreshProjects(); } catch (e) { toast(e.message); } };
$('pld').onclick = async () => { st.proj = await api('project?name=' + encodeURIComponent($('pl').value)); $('pn').value = st.proj.name; renderGroups(); freq(); };

export async function freq() {
  const p = st.proj; if (!p.name && !Object.keys(p.groups).length) return;
  if (!p.name) { $('freq').textContent = 'save the project first (needs a name)'; return; }
  await api('save_project', p);
  const sl = $('sl').style.display === '' && $('lvl').value === 'super' ? '&sl=' + $('sl').value : '';
  const F = await api(`frequency?name=${encodeURIComponent(p.name)}&level=${$('lvl').value}${sl}`);
  const ids = [...new Set(Object.values(F).flatMap(r => Object.keys(r.pct)))].sort((a, b) => a - b), R = Object.values(F);
  $('sumbadge').innerHTML = R.every(r => r.ok) ? '<span class="ok">Σ=100% ✓</span>' : '<span class="bad">Σ≠100%, labels missing, or sessions not from one fit ✗</span>';
  $('freq').innerHTML = '<table><tr><th>cluster</th>' + Object.keys(F).map(g => `<th>${g}</th>`).join('') + '</tr>' +
    ids.map(i => `<tr><td><span style="color:${color(+i)}">■</span> ${i}</td>` + R.map(r => `<td>${(r.pct[i] ?? 0).toFixed(1)}%</td>`).join('') + '</tr>').join('') +
    '<tr><td>Σ</td>' + R.map(r => `<td class="${r.ok ? 'ok' : 'bad'}">${r.sum.toFixed(2)}%${r.missing.length ? ' ⚠' + r.missing.length + ' unlabeled' : ''}</td>`).join('') + '</tr></table>';
}

$('an').onclick = async () => {
  const paths = [...new Set(Object.values(st.proj.groups).flat())];
  if (!paths.length) return toast('add sessions to groups first (the fit covers all project sessions)');
  const sd = $('seed').value; const {job} = await api('analyze', {paths, max_frames: +$('mf').value, seed: sd === '' ? null : +sd}); poll(job);
};
async function poll(id) {
  const j = await api('job?id=' + id); $('pg').style.display = 'block'; $('pg').value = j.pct;
  $('job').innerHTML = `${j.stage} — ${j.pct}% — ${j.elapsed}s${j.error ? ` <span class="bad">${j.error}</span>` : ''}`;
  if (!j.done) return setTimeout(() => poll(id), 1000);
  if (!j.error) { toast('SUBTLE done'); emit('reload'); freq(); }
}
setInterval(async () => { try { $('log').textContent = (await api('log')).slice(-60).join('\n'); } catch (e) { /* server restarting */ } }, 2000);
on('labels', freq);

// ---- models: train once on the project sessions, save, then map any session (all frames) with the saved model
const projPaths = () => [...new Set(Object.values(st.proj.groups).flat())];
async function refreshModels() { try { $('ml').innerHTML = (await api('models')).map(m => `<option value="${m.name}">${m.name} · ${m.n_sub} sub · ${m.id}</option>`).join(''); } catch (e) { /* none yet */ } }
$('mt').onclick = async () => { const name = $('mn').value.trim(), paths = projPaths(); if (!name || !paths.length) return toast('model name + project sessions needed');
  const sd = $('seed').value; const {job} = await api('train', {name, paths, max_frames: +$('mf').value, seed: sd === '' ? null : +sd}); poll(job); };
$('mm').onclick = async () => { const name = $('ml').value, paths = projPaths(); if (!name || !paths.length) return toast('pick a model + add sessions to groups');
  const {job} = await api('map', {name, paths}); poll(job); };
on('reload', refreshModels); refreshModels();

// ---- group statistics: share of each cluster per animal, exact permutation p, Hedges g, BH-adjusted stars
let lastStats = null;
$('stgo').onclick = async () => {
  if (!st.proj.name) return toast('save the project first'); await api('save_project', st.proj);
  const sl = $('sl').style.display === '' && $('lvl').value === 'super' ? '&sl=' + $('sl').value : '';
  try { lastStats = await api(`stats?name=${encodeURIComponent(st.proj.name)}&level=${$('lvl').value}${sl}&correct=${$('stbh').checked ? 1 : 0}`); } catch (e) { $('stat').innerHTML = `<span class="bad">${e.message}</span>`; return; }
  renderStats(); };
function renderStats() {
  const R = lastStats; if (!R) return;
  $('stat').innerHTML = `<div class="${R.one_fit ? 'ok' : 'bad'}">${R.one_fit ? 'all sessions from one fit' : 'sessions are NOT from one fit: ids are not comparable'}</div><div class="mut">unit: ${R.unit}. Groups come from the project (folder names unless you set them): provisional.</div>` +
    R.pairs.map(P => { const [ga, gb] = P.groups, mx = Math.max(1, ...P.tests.flatMap(t => [...t.values[ga], ...t.values[gb]]));
      return `<h3 style="margin-top:10px">${ga} (n=${P.n[ga]}) vs ${gb} (n=${P.n[gb]})</h3><div class="mut">${P.method}; smallest possible p = ${P.min_p.toFixed(4)}; ${P.correction}</div>` +
      (P.min_p >= .05 ? '<div class="bad">smallest possible p ≥ 0.05: no star is reachable with these group sizes</div>' : '') +
      '<table><tr><th>cluster</th><th>% ' + ga + ' / ' + gb + '</th><th>animals</th><th>g</th><th>p adj</th></tr>' +
      P.tests.map(t => { const nm = cname(t.cluster), dot = (v, c) => v.map(x => `<i style="left:${100 * x / mx}%;background:${c}"></i>`).join('');
        return `<tr><td>${t.cluster}${nm ? ' ' + nm : ''}</td><td>${t.mean[ga].toFixed(1)} / ${t.mean[gb].toFixed(1)}</td><td><div class="dots">${dot(t.values[ga], '#5ac8fa')}${dot(t.values[gb], '#ff9f0a')}</div></td>` +
          `<td>${t.g == null ? '–' : t.g.toFixed(2)}</td><td title="raw p ${t.p.toFixed(4)}">${t.p_adj.toFixed(3)} <b class="${t.stars === 'ns' ? 'mut' : 'ok'}">${t.stars}</b></td></tr>`; }).join('') + '</table><div class="mut">dots: ' + ga + ' blue, ' + gb + ' orange; one dot = one animal</div>'; }).join('');
}
$('stcsv').onclick = () => { if (!lastStats) return toast('run Compare groups first');
  const rows = [['group_a', 'group_b', 'cluster', 'mean_a', 'mean_b', 'diff', 'hedges_g', 'p', 'p_adj', 'stars', 'method']];
  for (const P of lastStats.pairs) for (const t of P.tests) rows.push([P.groups[0], P.groups[1], t.cluster, t.mean[P.groups[0]], t.mean[P.groups[1]], t.diff, t.g, t.p, t.p_adj, t.stars, P.method]);
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.map(r => r.join(',')).join('\n')], {type: 'text/csv'})); a.download = 'group_stats.csv'; a.click(); };
