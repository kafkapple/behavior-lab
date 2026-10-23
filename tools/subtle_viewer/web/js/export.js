// PNG export of the two statistics views, drawn on a canvas (the on-screen tables are HTML). Colors match the page.
import {color} from './colors.js';

const BLUE = '#5ac8fa', ORANGE = '#ff9f0a', BG = '#ffffff', FG = '#1d1d1f', MUT = '#6e6e73';
const star = q => q == null ? '' : q < .001 ? '***' : q < .01 ? '**' : q < .05 ? '*' : 'ns';
function save(c, name) { c.toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = name; a.click(); }); }
function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d'); x.fillStyle = BG; x.fillRect(0, 0, w, h); x.font = '14px sans-serif'; x.textBaseline = 'middle'; return [c, x]; }

// cluster shares: one panel per group pair, one row per cluster, one dot per animal, mean tick, stars from the adjusted p
export function statsPng(R, name = 'group_stats.png') {
  const rowH = 30, left = 70, pw = 520, gap = 70;
  const H = R.pairs.reduce((s, P) => s + 90 + P.tests.length * rowH, 20), [c, x] = canvas(left + pw + 120, H); let y = 20;
  for (const P of R.pairs) { const [ga, gb] = P.groups, mx = Math.max(1, ...P.tests.flatMap(t => [...t.values[ga], ...t.values[gb]])) * 1.05;
    x.fillStyle = FG; x.font = 'bold 15px sans-serif'; x.fillText(`${ga} (n=${P.n[ga]}) vs ${gb} (n=${P.n[gb]})  share of frames per animal (%)`, 10, y + 10);
    x.font = '12px sans-serif'; x.fillStyle = MUT; x.fillText(`${P.method}; ${P.correction}; smallest possible p ${P.min_p.toFixed(4)}; dots: ${ga} blue, ${gb} orange; groups provisional`, 10, y + 32); y += 60;
    for (const t of P.tests) { const cy = y + rowH / 2; x.fillStyle = color(t.cluster); x.fillRect(10, cy - 6, 12, 12); x.fillStyle = FG; x.font = '13px sans-serif'; x.fillText(String(t.cluster), 30, cy);
      x.strokeStyle = '#ccc'; x.beginPath(); x.moveTo(left, cy); x.lineTo(left + pw, cy); x.stroke();
      for (const [g, col, dy] of [[ga, BLUE, -4], [gb, ORANGE, 4]]) { t.values[g].forEach(v => { x.fillStyle = col; x.beginPath(); x.arc(left + v / mx * pw, cy + dy, 4, 0, 7); x.fill(); });
        x.strokeStyle = col; x.lineWidth = 2; x.beginPath(); x.moveTo(left + t.mean[g] / mx * pw, cy + dy - 8); x.lineTo(left + t.mean[g] / mx * pw, cy + dy + 8); x.stroke(); x.lineWidth = 1; }
      x.fillStyle = t.stars === 'ns' ? MUT : FG; x.fillText(`p_adj ${t.p_adj.toFixed(3)} ${t.stars}`, left + pw + 12, cy); y += rowH; }
    y += gap - 40; }
  save(c, name); return c;
}

// DEK heatmap of one pair, one feature type and derivative: rows = clusters, columns = features, cell = signed t
export function dekPng(R, P, ty, n, name = 'dek.png') {
  const idx = R.columns.map((c, i) => [c, i]).filter(([c]) => c[0] === ty && c[1] === n), cw = 64, rh = 34, left = 70, top = 130;
  const [c, x] = canvas(left + idx.length * cw + 20, top + P.clusters.length * rh + 20), T = P.t.flat().filter(v => v != null), mx = Math.max(2, ...T.map(Math.abs));
  x.fillStyle = FG; x.font = 'bold 15px sans-serif'; x.fillText(`DEK ${P.groups[0]} (orange, t>0) vs ${P.groups[1]} (blue); ${ty}${n}`, 10, 16);
  x.font = '12px sans-serif'; x.fillStyle = MUT; x.fillText(`Welch t per cell; ${P.correction}; stars on adjusted p; groups provisional`, 10, 38);
  idx.forEach(([col], j) => { x.save(); x.translate(left + j * cw + cw / 2, top - 6); x.rotate(-Math.PI / 3); x.fillStyle = FG; x.textAlign = 'left'; x.fillText(col.split(':')[1], 0, 0); x.restore(); });
  P.clusters.forEach((cl, r) => { x.fillStyle = color(cl); x.fillRect(10, top + r * rh + 10, 12, 12); x.fillStyle = FG; x.textAlign = 'left'; x.font = '13px sans-serif'; x.fillText(`${cl}  ${P.n[r].join('/')}`, 28, top + r * rh + rh / 2);
    idx.forEach(([, j], k) => { const t = P.t[r][j], q = P.p_adj[r][j]; if (t == null) return; const a = Math.min(1, Math.abs(t) / mx);
      x.fillStyle = t > 0 ? `rgba(255,159,10,${a})` : `rgba(90,200,250,${a})`; x.fillRect(left + k * cw, top + r * rh, cw - 2, rh - 2);
      x.fillStyle = FG; x.textAlign = 'center'; x.fillText(`${t.toFixed(1)}${q < .05 ? star(q) : ''}`, left + k * cw + cw / 2 - 1, top + r * rh + rh / 2 - 1); x.textAlign = 'left'; }); });
  save(c, name); return c;
}
