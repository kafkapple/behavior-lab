// Project: groups, save and load, the SUBTLE fit job, and the frequency table with its sum check.
import {$, st, on, emit, api, toast} from './state.js';
import {color} from './colors.js';

export async function refreshProjects() { $('pl').innerHTML = (await api('projects')).map(n => `<option>${n}</option>`).join(''); }
function renderGroups() {
  const g = st.proj.groups;
  $('groups').innerHTML = Object.entries(g).map(([n, l]) => `<div><b>${n}</b> <span class="mut">(${l.length})</span> <button data-del="${n}">×</button><br>` +
    l.map(p => `<span class="mut">${p.split('/').pop()}</span>`).join(', ') + '</div>').join('');
  document.querySelectorAll('[data-del]').forEach(b => b.onclick = () => { delete g[b.dataset.del]; renderGroups(); freq(); });
}
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
