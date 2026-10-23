// Cluster legend (right panel): index, color, share, and a user name saved per fit and level. Click = filter, double-click = solo.
// The same ids and colors are used by the map, timeline, skeleton, video bones and the frequency table.
import {$, st, on, emit, api, toast, lid, nextBout, nkey, cname} from './state.js';
import {color} from './colors.js';

// st.flag = result of the level sweep for one level: ★ significant after max-T over all levels, ☆ exploratory (raw p or top |t|, not significant)
const mark = id => { const f = st.flag; if (!f || $('lvl').value !== 'super' || $('sl').style.display || +$('sl').value !== f.level) return ''; return f.sig.includes(id) ? '<span title="significant, selection-corrected" style="color:#ff9f0a"> ★</span>' : f.expl.includes(id) ? '<span class="mut" title="exploratory, not significant"> ☆</span>' : ''; };
export function legend() {
  const el = $('legend'), S = st.S, L = S && S.L; if (!L) { el.innerHTML = '<span class="mut">no cluster labels: run SUBTLE (Project tab)</span>'; return; }
  const c = new Map(); for (const v of L) c.set(v, (c.get(v) || 0) + 1); const n = L.length;
  const lv = $('lvl').value === 'super' && $('sl').style.display === '' ? ` · ${$('sl').selectedOptions[0].textContent}` : '';
  el.innerHTML = `<div class="mut">${$('lvl').value}${lv} · fit ${S.fit_id || 'none (ids are not comparable)'}${S.fit_id ? '' : ''}</div>` +
    `<div style="margin:4px 0"><button id="fall">all</button> <button id="fnone">none</button> <button id="fnext" title="jump to the next run of the selected clusters">⏭ next bout</button> <span class="mut">${st.F.size ? st.F.size + ' selected: replay skips the rest' : 'click = filter · double-click = solo'}</span></div>` +
    [...c.entries()].sort((a, b) => a[0] - b[0]).map(([id, k]) =>
      `<div class="lrow${st.F.has(id) ? ' on' : ''}" data-id="${id}"><span style="color:${color(id)};font-size:14px">■</span><b>${id}${mark(id)}</b>` +
      `<input data-n="${id}" value="${cname(id).replace(/"/g, '&quot;')}" placeholder="name" ${S.fit_id ? '' : 'disabled'} title="saved per fit and level"><span class="mut">${(100 * k / n).toFixed(1)}%</span></div>`).join('');
  el.querySelectorAll('.lrow').forEach(e => { const id = +e.dataset.id;
    e.onclick = ev => { if (ev.target.tagName === 'INPUT') return; st.F.has(id) ? st.F.delete(id) : st.F.add(id); emit('filter'); };
    e.ondblclick = ev => { if (ev.target.tagName === 'INPUT') return; st.F = new Set([id]); emit('filter'); }; });
  el.querySelectorAll('input[data-n]').forEach(i => i.onchange = async () => { const k = nkey(i.dataset.n);
    try { st.names = await api('set_name', {key: k, name: i.value}); toast('name saved'); } catch (e) { toast(e.message); } });
  $('fall').onclick = () => { st.F = new Set(c.keys()); emit('filter'); };
  $('fnone').onclick = () => { st.F = new Set(); emit('filter'); };
  $('fnext').onclick = () => { st.fr = nextBout(st.fr); emit('frame'); };
}
api('names').then(n => { st.names = n; legend(); }).catch(() => {});
on('labels', legend); on('filter', legend); on('session', legend);
