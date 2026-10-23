// Shared state, tiny event bus, cluster labels and filter. Every view reads from here; none talks to another view.
export const $ = id => document.getElementById(id);
export const st = {S: null, cur: null, fr: 0, F: new Set(), KP: null, camNow: null, proj: {name: '', groups: {}, meta: {}}};
const H = {};
export const on = (e, f) => (H[e] ??= []).push(f);
export const emit = e => (H[e] || []).forEach(f => f());      // events: session, frame, labels, filter, project
export const FPS = () => +$('fps').value || 20;
export const api = (p, b) => fetch('/api/' + p, b ? {method: 'POST', body: JSON.stringify(b)} : {})
  .then(r => r.json()).then(j => { if (j && j.error) throw new Error(j.error); return j; });
export const toast = m => { $('toast').textContent = m; setTimeout(() => $('toast').textContent = '', 4000); };

// ---- labels: the chosen cluster level as one id per frame (st.S.L)
export function fillLevels() {
  const raw = st.S && st.S.super, sel = $('sl');
  if ($('lvl').value !== 'super' || !raw || !raw.length || !Array.isArray(raw[0])) { sel.style.display = 'none'; return; }
  const nl = raw[0].length, cnt = [...Array(nl)].map((_, j) => new Set(raw.map(r => r[j])).size);
  sel.style.display = '';
  sel.innerHTML = cnt.map((n, j) => `<option value="${j}">L${j}: ${n} clusters</option>`).join('');
  sel.value = cnt.indexOf(Math.max(...cnt));                      // wrapper default: the level with the most clusters
}
export function relabel() {
  const S = st.S; if (!S) return;
  const raw = S[$('lvl').value];
  if (!raw || !raw.length) { S.L = null; return; }
  S.L = Array.isArray(raw[0]) ? raw.map(r => r[Math.min(r.length - 1, Math.max(0, +$('sl').value || 0))]) : raw;
}
export const lid = i => st.S && st.S.L && st.S.L[i] != null ? st.S.L[i] : null;
export const inF = i => !st.F.size || st.F.has(lid(i));
export const nextIn = i => { const n = st.S.xyz.length; for (let d = 0; d < n; d++) { const j = (i + d) % n; if (inF(j)) return j; } return i; };
export const nextBout = i => { const n = st.S.xyz.length;
  for (let d = 1; d <= n; d++) { const j = (i + d) % n; if (inF(j) && (j === 0 || !inF(j - 1) || lid(j) !== lid(j - 1))) return j; } return i; };

// ---- canvas helper: size the bitmap to its CSS box (sharp on retina) and return the 2D context
export function fit(c) {
  const r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  const w = Math.max(10, Math.round(r.width * d)), h = Math.max(10, Math.round(r.height * d));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return c.getContext('2d');
}
