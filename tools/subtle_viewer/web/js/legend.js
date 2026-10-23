// Cluster legend: one chip per id with its share of the open session; click = filter, double-click = solo.
// The same ids and colors are used by the map, timeline, skeleton, video bones and the frequency table.
import {$, st, on, emit, lid, nextBout} from './state.js';
import {color} from './colors.js';

export function legend() {
  const el = $('legend'), S = st.S, L = S && S.L; if (!L) { el.innerHTML = ''; return; }
  const c = new Map(); for (const v of L) c.set(v, (c.get(v) || 0) + 1); const n = L.length;
  const lv = $('lvl').value === 'super' && $('sl').style.display === '' ? ` · ${$('sl').selectedOptions[0].textContent}` : '';
  el.innerHTML = `<span class="mut">${$('lvl').value}${lv} · fit ${S.fit_id || 'none (ids are not comparable)'} · seed in sidecar</span> ` +
    [...c.entries()].sort((a, b) => a[0] - b[0]).map(([id, k]) =>
      `<span class="chip${st.F.has(id) ? ' on' : ''}" data-id="${id}" style="border-color:${color(id)}"><span style="color:${color(id)}">■</span> ${id} <span class="mut">${(100 * k / n).toFixed(1)}%</span></span>`).join('') +
    ` <button id="fall">all</button><button id="fnone">none</button><button id="fnext" title="jump to the next run of the selected clusters">⏭ next bout</button> ` +
    `<span class="mut">${st.F.size ? st.F.size + ' selected: replay skips the rest' : 'click = filter · double-click = solo'}</span>`;
  el.querySelectorAll('.chip').forEach(e => { const id = +e.dataset.id;
    e.onclick = () => { st.F.has(id) ? st.F.delete(id) : st.F.add(id); emit('filter'); };
    e.ondblclick = () => { st.F = new Set([id]); emit('filter'); }; });
  $('fall').onclick = () => { st.F = new Set(c.keys()); emit('filter'); };
  $('fnone').onclick = () => { st.F = new Set(); emit('filter'); };
  $('fnext').onclick = () => { st.fr = nextBout(st.fr); emit('frame'); };
}
on('labels', legend); on('filter', legend); on('session', legend);
