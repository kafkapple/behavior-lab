// Project: groups, save and load, the SUBTLE fit job, and the frequency table with its sum check.
import {$, st, on, emit, api, toast, nkey, cname} from './state.js';
import {color} from './colors.js';
import {statsPng, dekPng} from './export.js';

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
  if (st.proj.name) showStability().catch(() => { $('stabres').innerHTML = ''; });      // last saved seed-stability result, if any
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
async function poll(id, done) {
  const j = await api('job?id=' + id); $('pg').style.display = 'block'; $('pg').value = j.pct;
  $('job').innerHTML = `${j.stage} — ${j.pct}% — ${j.elapsed}s${j.error ? ` <span class="bad">${j.error}</span>` : ''}`;
  if (!j.done) return setTimeout(() => poll(id, done), 1000);
  if (!j.error) { if (done) return done(); toast('SUBTLE done'); emit('reload'); freq(); }
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

// ---- DEK: per animal and cluster mean of kinematic features, Welch t per cell. Heatmap = signed t, stars on BH-adjusted p.
let lastDek = null;
$('dkgo').onclick = async () => {
  if (!st.proj.name) return toast('save the project first'); await api('save_project', st.proj);
  const sl = $('sl').style.display === '' && $('lvl').value === 'super' ? '&sl=' + $('sl').value : '';
  $('dek').textContent = 'computing…';
  try { lastDek = await api(`dek?name=${encodeURIComponent(st.proj.name)}&level=${$('lvl').value}${sl}&correct=${$('stbh').checked ? 1 : 0}`); } catch (e) { $('dek').innerHTML = `<span class="bad">${e.message}</span>`; return; }
  $('dkpair').innerHTML = lastDek.pairs.map((P, i) => `<option value="${i}">${P.groups.join(' vs ')}</option>`).join(''); renderDek(); };
['dktype', 'dkn', 'dkpair'].forEach(i => $(i).onchange = renderDek);
const sig = q => q == null ? '' : q < .001 ? '***' : q < .01 ? '**' : q < .05 ? '*' : '';
function renderDek() {
  const R = lastDek; if (!R) return; const P = R.pairs[+$('dkpair').value || 0], ty = $('dktype').value, n = $('dkn').value;
  const idx = R.columns.map((c, i) => [c, i]).filter(([c]) => c[0] === ty && c[1] === n), T = P.t.flat().filter(v => v != null), mx = Math.max(2, ...T.map(Math.abs));
  const cell = (r, [c, j]) => { const t = P.t[r][j], q = P.p_adj[r][j]; if (t == null) return '<td class="mut">–</td>';
    const a = Math.min(1, Math.abs(t) / mx), col = t > 0 ? `rgba(255,159,10,${a})` : `rgba(90,200,250,${a})`;
    return `<td style="background:${col};text-align:center" title="${c} cluster ${P.clusters[r]}: t=${t.toFixed(2)}, p=${P.p[r][j].toExponential(2)}, p_adj=${q.toExponential(2)}, g=${P.g[r][j] == null ? '–' : P.g[r][j].toFixed(2)}">${t.toFixed(1)}<b>${sig(q)}</b></td>`; };
  $('dek').innerHTML = `<div class="mut">${P.groups[0]} (orange, t>0) vs ${P.groups[1]} (blue); ${P.tests} cells; ${P.correction}; animals need ≥${R.min_frames} frames in a cluster; ${R.unit}. Groups are provisional.</div>` +
    (P.skipped.length ? `<div class="bad">clusters skipped (fewer than 2 animals in a group): ${P.skipped.map(s => s.cluster + ' (' + s.n.join('/') + ')').join(', ')}</div>` : '') +
    '<table><tr><th>cluster (nA/nB)</th>' + idx.map(([c]) => `<th style="font-size:9px;writing-mode:vertical-rl">${c.split(':')[1]}</th>`).join('') + '</tr>' +
    P.clusters.map((c, r) => `<tr><td><span style="color:${color(c)}">■</span> ${c} <span class="mut">${P.n[r].join('/')}</span></td>${idx.map(x => cell(r, x)).join('')}</tr>`).join('') + '</table>';
}
$('dkcsv').onclick = () => { if (!lastDek) return toast('run Compute DEK first');
  const rows = [['group_a', 'group_b', 'cluster', 'feature', 't', 'p', 'p_adj', 'hedges_g']];
  for (const P of lastDek.pairs) P.clusters.forEach((c, r) => lastDek.columns.forEach((f, j) => P.t[r][j] != null && rows.push([P.groups[0], P.groups[1], c, f, P.t[r][j], P.p[r][j], P.p_adj[r][j], P.g[r][j]])));
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.map(r => r.join(',')).join('\n')], {type: 'text/csv'})); a.download = 'dek.csv'; a.click(); };
$('stpng').onclick = () => lastStats ? statsPng(lastStats) : toast('run Compare groups first');
$('dkpng').onclick = () => lastDek ? dekPng(lastDek, lastDek.pairs[+$('dkpair').value || 0], $('dktype').value, $('dkn').value) : toast('run Compute DEK first');

// ---- seed stability: refit with 3 seeds, show ARI and the group result per level so no level is picked after seeing the stars
async function showStability() {
  const R = await api(`stability_result?name=${encodeURIComponent(st.proj.name)}`), s = Object.keys(R.per_seed);
  const row = (o, k) => `<tr><td>${k}</td><td>${Object.entries(o).map(([a, v]) => `${a}: ${v.toFixed(3)}`).join('<br>')}</td></tr>`;
  $('stabres').innerHTML = `<h3 style="margin-top:8px">Seed stability (${R.created})</h3><div class="${R.star_pattern === 'stable' ? 'ok' : 'bad'}">star pattern across seeds: ${R.star_pattern} (any cluster below .05 at the default level: ${Object.entries(R.star_flags).map(([a, v]) => `seed ${a} ${v ? 'yes' : 'no'}`).join(', ')})</div>` +
    `<div class="mut">first ${R.max_frames} frames per animal; groups ${R.groups.join(' vs ')} (provisional); ARI 1 = same partition, 0 = chance</div><table><tr><th>ARI</th><th>seed pairs</th></tr>${row(R.ari_sub, 'subclusters')}${row(R.ari_super_default, 'super, default level')}</table>` +
    '<table><tr><th>seed</th><th>level</th><th>clusters</th><th>min p adj</th><th>n &lt; .05</th></tr>' + s.flatMap(sd => [['sub', R.per_seed[sd].sub], ...R.per_seed[sd].levels.map(l => ['L' + l.level + (l.level === R.per_seed[sd].default_level ? ' ★' : ''), l])].map(([n, l]) => `<tr><td>${sd}</td><td>${n}</td><td>${l.k}</td><td>${l.min_p_adj.toFixed(3)}</td><td>${l.n_sig}</td></tr>`)).join('') + '</table><div class="mut">★ = default level (most clusters)</div>';
}
$('stab').onclick = async () => { if (!st.proj.name) return toast('save the project first'); await api('save_project', st.proj);
  const {job} = await api('stability', {name: st.proj.name, max_frames: +$('mf').value, seeds: [0, 1, 2]}); $('stabres').innerHTML = '<span class="mut">running 3 fits: progress is in the Project tab and the log</span>'; poll(job, showStability); };
