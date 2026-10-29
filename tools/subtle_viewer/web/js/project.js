// Project: groups, save and load, the SUBTLE fit job, and the frequency table with its sum check.
import {$, st, on, emit, api, toast, nkey, cname} from './state.js';
import {color} from './colors.js';
import {statsPng, dekPng} from './export.js';

// clustering methods come from the server (cluster.METHODS); Train and Seed stability take SUBTLE only, the server says so
const MNAME = {'subtle/umap_phenograph': 'UMAP + Phenograph (original SUBTLE, default)', 'subtle/pca_consensus': 'PCA consensus', 'bsoid/umap_hdbscan': 'B-SOiD (UMAP + HDBSCAN, 10 fps bins, -1 = noise)'};
api('methods').then(M => { $('clu').innerHTML = M.methods.map(m => `<option value="${m}">${MNAME[m] || m}</option>`).join(''); $('clu').value = M.default; $('clu').onchange(); }).catch(e => toast(e.message));
$('clu').onchange = () => { $('methodopts').innerHTML = $('clu').value.startsWith('bsoid/') ?
  '<label>UMAP neighbors <input id="bneighbors" type="number" min="3" max="500" value="60" style="width:65px"></label> <label>HDBSCAN min cluster size <input id="bmin" type="number" min="2" max="10000" value="30" style="width:68px"></label>' :
  '<div class="mut">SUBTLE: CWT → PCA → UMAP/Phenograph → optional DIB levels. Seed and frames are set below.</div>'; };
const cluArgs = () => { const m = $('clu').value; return m.startsWith('subtle/') ? {method: m, clustering: m.split('/')[1]} :
  {method: m, params: {n_neighbors: +$('bneighbors').value, min_cluster_size: +$('bmin').value}}; };
export function requireProject() {
  if (st.proj.name) return true;
  document.querySelector('#tabs button[data-t="proj"]').click();
  document.querySelector('[data-go="proj"]').closest('details').open = true;
  $('ps').classList.add('needs-save'); $('pn').focus(); $('pn').scrollIntoView({block: 'center'});
  toast('Name and Save the project here, then retry.');
  return false;
}
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
$('pn').oninput = () => { $('projsaved').textContent = 'Unsaved project name'; };
$('ps').onclick = async () => { const name = $('pn').value.trim(); try { await api('save_project', {...st.proj, name}); st.proj.name = name; $('ps').classList.remove('needs-save'); $('projsaved').textContent = `Saved: ${name}`; toast('saved ' + name); refreshProjects(); } catch (e) { toast(e.message); } };
$('pld').onclick = async () => { st.proj = await api('project?name=' + encodeURIComponent($('pl').value)); $('pn').value = st.proj.name; $('projsaved').textContent = `Loaded: ${st.proj.name} · last saved ${st.proj.updated || 'unknown'}`; renderGroups(); freq(); };

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
  const sd = $('seed').value; const {job} = await api('analyze', {paths, max_frames: +$('mf').value, seed: sd === '' ? null : +sd, ...cluArgs()}); poll(job);
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
let models = [];
async function refreshModels() { try { models = await api('models'); $('ml').innerHTML = models.map(m => `<option value="${m.name}">${m.name} · ${m.n_sub} sub · ${m.id}</option>`).join(''); showModel(); } catch (e) { $('modelinfo').textContent = e.message; } }
function showModel() { const m = models.find(x => x.name === $('ml').value); $('modelinfo').textContent = m ?
  `Fit ${m.id} · ${m.created} · seed ${m.seed} · trained on ${m.train_paths.length} sessions (${m.train_paths.join(', ')}) · ${m.frames.join(', ')} frames${m.maps?.length ? ' · last mapped ' + m.maps.at(-1).at + ': ' + m.maps.at(-1).paths.join(', ') : ''}` : 'No saved model in this data root.'; }
$('ml').onchange = showModel;
$('mt').onclick = async () => { const name = $('mn').value.trim(), paths = projPaths(); if (!name || !paths.length) return toast('model name + project sessions needed');
  if (!$('clu').value.startsWith('subtle/')) return toast('Train & save supports SUBTLE methods; use Explore clusters for B-SOiD.');
  const sd = $('seed').value; const r = await api('train', {name, paths, max_frames: +$('mf').value, seed: sd === '' ? null : +sd, ...cluArgs()}).catch(e => toast(e.message)); if (r) poll(r.job); };
$('mm').onclick = async () => { const name = $('ml').value, paths = projPaths(); if (!name || !paths.length) return toast('pick a model + add sessions to groups');
  const {job} = await api('map', {name, paths}); poll(job); };
on('reload', refreshModels); refreshModels();

// ---- group statistics: share of each cluster per animal, exact permutation p, Hedges g, BH-adjusted stars
let lastStats = null;
$('stgo').onclick = async () => {
  if (!requireProject()) return; await api('save_project', st.proj);
  const sl = $('sl').style.display === '' && $('lvl').value === 'super' ? '&sl=' + $('sl').value : '';
  try { lastStats = await api(`stats?name=${encodeURIComponent(st.proj.name)}&level=${$('lvl').value}${sl}&correct=${$('stbh').checked ? 1 : 0}`); } catch (e) { $('stat').innerHTML = `<span class="bad">${e.message}</span>`; return; }
  renderStats(); };
function renderStats() {
  const R = lastStats; if (!R) return;
  $('stat').innerHTML = `<div class="${R.one_fit ? 'ok' : 'bad'}">${R.one_fit ? 'All sessions use one fit.' : 'Fit IDs differ or are unknown: cluster IDs cannot be compared.'}</div><p class="mut">Dots = individual animals; bar = mean ± sample SD. The error bar shows spread, not a confidence interval. Folder-derived groups are provisional.</p>` +
    R.pairs.map(P => { const [ga, gb] = P.groups, nSig = P.tests.filter(t => t.p_adj < .05).length;
      const track = (t, g, c) => { const lo = Math.max(0, t.mean[g] - (t.sd[g] || 0)), hi = Math.min(100, t.mean[g] + (t.sd[g] || 0));
        return `<div class="share-track" title="${g}: each dot is one animal"><span class="share-group" style="color:${c}">${g}</span><span class="share-axis"><i class="share-sd" style="left:${lo}%;width:${hi - lo}%;background:${c}"></i><i class="share-mean" style="left:${t.mean[g]}%;background:${c}"></i>${t.values[g].map((v, i) => `<i class="share-dot" style="left:${Math.max(1, Math.min(99, v))}%;--dot-y:${5 + (i % 3) * 7}px;background:${c}"></i>`).join('')}</span><span class="share-value">${t.mean[g].toFixed(1)} ± ${t.sd[g]?.toFixed(1) ?? '–'}%</span></div>`; };
      return `<h3>${ga} (n=${P.n[ga]}) vs ${gb} (n=${P.n[gb]})</h3><p class="mut">${nSig}/${P.tests.length} clusters with selected p&lt;.05 · ${P.method} · ${P.correction} · smallest possible raw p ${P.min_p.toFixed(4)}.</p>` +
        P.tests.map(t => `<article class="stat-card ${t.p_adj < .05 ? 'sig' : ''}"><div class="stat-card-head"><span><span style="color:${color(t.cluster)}">■</span> Cluster ${t.cluster}${cname(t.cluster) ? ' · ' + cname(t.cluster) : ''}</span><span class="${t.p_adj < .05 ? 'sig-mark' : 'mut'}">${t.p_adj < .05 ? t.stars + ' ' : ''}${P.correction === 'none' ? 'p' : 'pBH'}=${t.p_adj.toFixed(3)}</span></div><div class="share-plot">${track(t, ga, '#5ac8fa')}${track(t, gb, '#ff9f0a')}<div class="share-scale"><span>0%</span><span>25</span><span>50</span><span>75</span><span>100%</span></div></div><div class="stat-meta">Hedges g ${t.g == null ? '–' : t.g.toFixed(2)} · raw p ${t.p.toFixed(4)} · ${t.p_adj < .05 ? 'highlighted under selected test' : 'not significant under selected test'}</div></article>`).join(''); }).join('');
}
$('stcsv').onclick = () => { if (!lastStats) return toast('run Compare groups first');
  const rows = [['group_a', 'group_b', 'cluster', 'mean_a', 'sd_a', 'mean_b', 'sd_b', 'diff', 'hedges_g', 'p', 'p_adj', 'stars', 'method']];
  for (const P of lastStats.pairs) for (const t of P.tests) rows.push([P.groups[0], P.groups[1], t.cluster, t.mean[P.groups[0]], t.sd[P.groups[0]], t.mean[P.groups[1]], t.sd[P.groups[1]], t.diff, t.g, t.p, t.p_adj, t.stars, P.method]);
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.map(r => r.join(',')).join('\n')], {type: 'text/csv'})); a.download = 'group_stats.csv'; a.click(); };

// ---- DEK: per animal and cluster mean of kinematic features, Welch t per cell. Heatmap = signed t, stars on BH-adjusted p.
let lastDek = null;
$('dkgo').onclick = async () => {
  if (!requireProject()) return;
  $('dkgo').disabled = true; $('dkprogress').style.display = 'block'; $('dkprogress').value = 0;
  $('dkstatus').textContent = 'Starting…'; $('dek').textContent = 'Computing kinematic features for each session…';
  try {
    await api('save_project', st.proj);
    const {job} = await api('dek_start', {name: st.proj.name, level: $('lvl').value,
      sl: $('sl').style.display === '' && $('lvl').value === 'super' ? +$('sl').value : null, correct: true});
    let s;
    do {
      s = await api('dek_job?id=' + job); $('dkprogress').value = s.pct;
      $('dkstatus').textContent = `${s.stage} · ${s.pct}% · ${s.elapsed}s`;
      if (!s.done) await new Promise(resolve => setTimeout(resolve, 1000));
    } while (!s.done);
    if (s.error) throw new Error(s.error);
    lastDek = await api('dek_result?id=' + job);
    $('dkpair').innerHTML = lastDek.pairs.map((P, i) => `<option value="${i}">${P.groups.join(' vs ')}</option>`).join('');
    renderDek();
  } catch (e) { $('dkstatus').textContent = 'Failed'; $('dek').textContent = e.message; }
  finally { $('dkgo').disabled = false; }
};
['dktype', 'dkn', 'dkpair'].forEach(i => $(i).onchange = renderDek);
const sig = q => q == null ? '' : q < .001 ? '***' : q < .01 ? '**' : q < .05 ? '*' : '';
function renderDek() {
  const R = lastDek; if (!R) return; const P = R.pairs[+$('dkpair').value || 0], ty = $('dktype').value, n = $('dkn').value;
  const idx = R.columns.map((c, i) => [c, i]).filter(([c]) => c[0] === ty && c[1] === n), T = P.t.flat().filter(v => v != null), mx = Math.max(2, ...T.map(Math.abs));
  const cell = (r, [c, j]) => { const t = P.t[r][j], q = P.p_adj[r][j]; if (t == null) return '<td class="mut">–</td>';
    const a = Math.min(.7, Math.abs(t) / mx * .7), col = t > 0 ? `rgba(255,159,10,${a})` : `rgba(90,200,250,${a})`;
    return `<td class="dek-cell ${q < .05 ? 'sig' : ''}" style="background:${col}" title="${c} cluster ${P.clusters[r]}: t=${t.toFixed(2)}, p=${P.p[r][j].toExponential(2)}, p_adj=${q.toExponential(2)}, g=${P.g[r][j] == null ? '–' : P.g[r][j].toFixed(2)}">${t.toFixed(1)} ${q < .05 ? '<span class="sig-mark">' + sig(q) + '</span>' : ''}</td>`; };
  const hits = P.clusters.flatMap((cl, r) => R.columns.flatMap((c, j) => P.p_adj[r][j] != null && P.p_adj[r][j] < .05 ? [{cl, c, q: P.p_adj[r][j]}] : []));
  $('dek').innerHTML = `<div class="mut">${P.groups[0]} (orange, t>0) vs ${P.groups[1]} (blue); ${P.tests} cells; ${P.correction}. Correction is within this selected level, not across levels. Animals need ≥${R.min_frames} frames in a cluster; ${R.unit}. Groups are provisional.</div>` +
    `<div class="dek-hits">${hits.length ? hits.map(h => `<button data-ht="${h.c[0]}" data-hn="${h.c[1]}" title="p adjusted ${h.q.toFixed(4)}">* cluster ${h.cl} · ${h.c}</button>`).join('') : '<span class="mut">No corrected p&lt;.05 cells</span>'}</div>` +
    (P.skipped.length ? `<div class="bad">clusters skipped (fewer than 2 animals in a group): ${P.skipped.map(s => s.cluster + ' (' + s.n.join('/') + ')').join(', ')}</div>` : '') +
    '<div class="dek-table-wrap"><table class="dek-table"><tr><th>cluster (nA/nB)</th>' + idx.map(([c]) => `<th title="${c}">${c.split(':')[1]}</th>`).join('') + '</tr>' +
    P.clusters.map((c, r) => `<tr><td><span style="color:${color(c)}">■</span> ${c} <span class="mut">${P.n[r].join('/')}</span></td>${idx.map(x => cell(r, x)).join('')}</tr>`).join('') + '</table></div>';
  $('dek').querySelectorAll('.dek-hits button').forEach(b => b.onclick = () => { $('dktype').value = b.dataset.ht; $('dkn').value = b.dataset.hn; renderDek(); });
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
    `<div class="mut">clustering ${R.clustering || 'umap_phenograph (before 261024)'}; first ${R.max_frames} frames per animal; groups ${R.groups.join(' vs ')} (provisional); ARI/AMI 1 = same partition, 0 = chance</div><table><tr><th>ARI</th><th>seed pairs</th></tr>${row(R.ari_sub, 'subclusters')}${row(R.ari_super_default, 'super, default level')}</table>` +
    (R.ami_sub ? `<table><tr><th>AMI</th><th>seed pairs</th></tr>${row(R.ami_sub, 'subclusters')}${row(R.ami_super_default, 'super, default level')}</table>` : '') +
    (R.per_seed[s[0]].runs_sub ? '<table><tr><th>seed</th><th>labels</th><th>median run</th><th>mean run</th><th>1-frame runs</th></tr>' + s.flatMap(sd => [['sub', R.per_seed[sd].runs_sub], ['super ★', R.per_seed[sd].runs_super_default]].map(([n, x]) => `<tr><td>${sd}</td><td>${n}</td><td>${x.median_run.toFixed(1)}</td><td>${x.mean_run.toFixed(1)}</td><td>${(100 * x.frac_runs_len1).toFixed(1)}%</td></tr>`)).join('') + '</table><div class="mut">run = consecutive frames with one label, frames (20 fps)</div>' : '') +
    '<table><tr><th>seed</th><th>level</th><th>clusters</th><th>min p adj</th><th>n &lt; .05</th></tr>' + s.flatMap(sd => [['sub', R.per_seed[sd].sub], ...R.per_seed[sd].levels.map(l => ['L' + l.level + (l.level === R.per_seed[sd].default_level ? ' ★' : ''), l])].map(([n, l]) => `<tr><td>${sd}</td><td>${n}</td><td>${l.k}</td><td>${l.min_p_adj.toFixed(3)}</td><td>${l.n_sig}</td></tr>`)).join('') + '</table><div class="mut">★ = default level (most clusters)</div>';
}
$('stab').onclick = async () => { if (!requireProject()) return; await api('save_project', st.proj);
  const r = await api('stability', {name: st.proj.name, max_frames: +$('mf').value, seeds: [0, 1, 2], ...cluArgs()}).catch(e => toast(e.message)); if (!r) return; const {job} = r; $('stabres').innerHTML = '<span class="mut">running 3 fits: progress is in the Project tab and the log</span>'; poll(job, showStability); };

// ---- level sweep: recommended n with the selection corrected (max-T), flagged clusters marked in the legend, results saved to a folder
let lastSweep = null;
const pf = v => v < .001 ? '<.001' : v.toFixed(3);
function renderSweep() {
  const R = lastSweep, r = R.recommended, ok = r.significant.length > 0;
  $('sweepres').innerHTML = `<h3 style="margin-top:8px">Level sweep: ${R.groups.join(' vs ')}</h3><div class="mut">${R.method}. Smallest possible p ${R.min_possible_p == null ? 'n/a' : R.min_possible_p.toFixed(4)}. Groups provisional.</div>` +
    `<div class="${ok ? 'ok' : 'bad'}">${ok ? `recommended n = ${r.k} (L${r.level}): clusters ${r.significant.join(', ')} significant, selection-corrected p_fw = ${pf(r.p_fw)}` : `recommended n = ${r.k} (L${r.level}) has the highest score, but no cluster is significant after correcting for the sweep (p_fw = ${pf(r.p_fw)}); do not report a difference`}</div>` +
    `<table><tr><th>n</th>${Object.keys(R.criteria).map(c => `<th title="${R.criteria[c]}">${c} z (p)</th>`).join('')}</tr>` +
    R.levels.map(o => `<tr><td>${o.k}${o.level === r.level ? ' ★' : ''}</td>${Object.keys(R.criteria).map(c => `<td>${o.criteria[c].z.toFixed(2)} (${pf(o.criteria[c].p_level)})</td>`).join('')}</tr>`).join('') + '</table>' +
    `<div class="mut">z = statistic against its own null (higher = stronger). Recommended n per criterion: ${Object.entries(R.by_criterion).map(([c, v]) => `${c} n=${v.k} (p_fw ${pf(v.p_fw)})`).join('; ')}. p in brackets = this level alone; p_fw covers the choice of level. Pick the criterion before looking; the default is omnibus.</div>` +
    `<button id="swuse">Use n=${r.k}</button> <button id="swonly">Show flagged only</button> <span id="swdir" class="mut"></span>`;
  $('swuse').onclick = () => { $('lvl').value = 'super'; $('lvl').dispatchEvent(new Event('change')); $('sl').value = r.level; $('sl').dispatchEvent(new Event('change'));
    st.flag = {level: r.level, sig: r.significant, expl: r.exploratory}; emit('labels'); toast(`level L${r.level}: ${r.k} clusters`); };
  $('swonly').onclick = () => { $('swuse').click(); const f = [...r.significant, ...r.exploratory]; if (!f.length) return toast('no flagged cluster: nothing to isolate'); st.F = new Set(f); emit('filter'); };
}
$('swgo').onclick = async () => { if (!requireProject()) return; await api('save_project', st.proj);
  try { lastSweep = await api(`sweep?name=${encodeURIComponent(st.proj.name)}&criterion=${$('swcrit').value}&excl=${$('swexcl').checked ? 1 : 0}&drop=${encodeURIComponent($('swdrop').value.replace(/\s/g, ''))}`); } catch (e) { $('sweepres').innerHTML = `<span class="bad">${e.message}</span>`; return; } renderSweep(); };
$('swexp').onclick = async () => { if (!requireProject()) return; await api('save_project', st.proj);
  const {job} = await api('export', {name: st.proj.name, criterion: $('swcrit').value}); $('sweepres').insertAdjacentHTML('beforeend', '<div class="mut">exporting: progress is in the Project tab and the log</div>');
  poll(job, async () => { const d = (await api(`export_latest?name=${encodeURIComponent(st.proj.name)}`)).dir; $('sweepres').insertAdjacentHTML('beforeend', `<div class="ok">saved: <code id="swdirtxt">${d}</code></div>`); }); };

// Descriptive exit-transition heatmaps. Each group cell is an animal-level mean; no frame-level p values.
$('trgo').onclick = async () => {
  if (!requireProject()) return;
  await api('save_project', st.proj);
  const sl = $('sl').style.display === '' && $('lvl').value === 'super' ? '&sl=' + $('sl').value : '';
  let R; try { R = await api(`transitions?name=${encodeURIComponent(st.proj.name)}&level=${$('lvl').value}${sl}`); }
  catch (e) { $('trans').textContent = e.message; return; }
  const cell = v => v == null ? '<td class="mut">—</td>' : `<td style="background:rgba(90,200,250,${(.08 + .75 * v).toFixed(2)})">${(100 * v).toFixed(0)}%</td>`;
  const table = (title, rows, support) => `<h3>${title}</h3><div class="transition-matrix"><table><tr><th>from ↓ / to →</th>${R.ids.map(id => `<th>${id}</th>`).join('')}<th>animals with exits</th></tr>` +
    rows.map((row, i) => `<tr><th>${R.ids[i]}${cname(R.ids[i]) ? ' ' + cname(R.ids[i]) : ''}</th>${row.map(cell).join('')}<td>${support?.[i] ?? 'pooled'}</td></tr>`).join('') + '</table></div>';
  const groups = Object.entries(R.groups);
  $('trans').innerHTML = `<div class="mut">fit ${R.fit_id} · ${R.definition} Diagonal is excluded; retention below is P(same next frame). A blank row had no exits. Descriptive comparison only.</div>` +
    table('All sessions, pooled transitions', R.overall.exit) + `<div class="mut">self-retention by cluster: ${R.ids.map((id, i) => `${id} ${(100 * R.overall.retention[i]).toFixed(0)}%`).join(' · ')}</div>` +
    groups.map(([name, g]) => table(`${name} (${g.n_sessions} sessions)`, g.exit_mean, g.n_animals)).join('');
};

// ---- quality: label-free consistency checks of the 3D skeleton per session (no ground truth yet). ⚠ = more than 1.5x the median of all sessions.
$('qgo').onclick = async () => { if (!requireProject()) return; await api('save_project', st.proj); $('qual').textContent = 'computing…';
  let R; try { R = await api(`quality?name=${encodeURIComponent(st.proj.name)}`); } catch (e) { $('qual').innerHTML = `<span class="bad">${e.message}</span>`; return; }
  const cols = [['copied_frames', 'copied %', v => 100 * v, 1], ['jump_frames', 'jump %', v => 100 * v, 1], ['head_bone_cv', 'head CV', v => v, 3], ['tri_support', '≥2 cams', v => 100 * v, 1], ['reproj_px', 'resid px', v => { const a = v.filter(x => x != null); return a.reduce((s, x) => s + x, 0) / a.length; }, 1]];
  const med = c => { const a = R.rows.map(r => c[2](r[c[0]])).sort((x, y) => x - y); return a[a.length >> 1]; };
  const M = cols.map(med);
  $('qual').innerHTML = '<table><tr><th>session</th>' + cols.map(c => `<th>${c[1]}</th>`).join('') + '</tr>' + R.rows.map(r => `<tr><td><span class="mut">${r.group}</span> ${r.session.split('_')[0]}</td>` +
    cols.map((c, i) => { const v = c[2](r[c[0]]), bad = c[0] === 'tri_support' ? v < M[i] * .98 : v > 1.5 * M[i]; return `<td class="${bad ? 'bad' : ''}">${v.toFixed(c[3])}${bad ? ' ⚠' : ''}</td>`; }).join('') + '</tr>').join('') + '</table>' +
    '<div class="mut">Consistency checks only. copied = a node identical to the previous frame; jump = a node moves over half a body length in one frame (heuristic); head CV = length variation of head edges; ≥2 cams = node-frames triangulation could use; resid px = weak-perspective residual per camera. A bad value flags a session; a good value does not prove accuracy. Plan: subtle_3d_quality_plan.md</div>';
};
