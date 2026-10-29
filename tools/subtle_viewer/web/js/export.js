// PNG export of the two statistics views, drawn on a canvas (the on-screen tables are HTML). Colors match the page.
import {color} from './colors.js';

const BLUE = '#5ac8fa', ORANGE = '#ff9f0a', BG = '#ffffff', FG = '#1d1d1f', MUT = '#6e6e73';
const star = q => q == null ? '' : q < .001 ? '***' : q < .01 ? '**' : q < .05 ? '*' : 'ns';
function save(c, name) { c.toBlob(b => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = name; a.click(); }); }
function canvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; const x = c.getContext('2d'); x.fillStyle = BG; x.fillRect(0, 0, w, h); x.font = '14px sans-serif'; x.textBaseline = 'middle'; return [c, x]; }

// cluster shares: one panel per group pair, one row per cluster, one dot per animal, mean tick, stars from the adjusted p
export function statsPng(R, name = 'group_stats.png') {
  const left = 125, pw = 600, rowH = 58, right = 185;
  const H = R.pairs.reduce((s, P) => s + 130 + P.tests.length * rowH, 0) + 20, [c, x] = canvas(left + pw + right, H); let y = 18;
  for (const P of R.pairs) { const [ga, gb] = P.groups;
    x.fillStyle = FG; x.font = 'bold 18px sans-serif'; x.fillText(`${ga} (n=${P.n[ga]}) vs ${gb} (n=${P.n[gb]}) · cluster share (%)`, 18, y + 12);
    x.font = '12px sans-serif'; x.fillStyle = MUT; x.fillText(`${P.method} · ${P.correction}`, 18, y + 38);
    x.fillText(`Dots: animals · thick bar: mean ± sample SD · *: ${P.correction === 'none' ? 'raw' : 'corrected'} p < .05 · provisional groups`, 18, y + 56);
    x.fillStyle = BLUE; x.fillText(`${ga} (blue)`, 18, y + 76); x.fillStyle = ORANGE; x.fillText(`${gb} (orange)`, 130, y + 76);
    for (const pct of [0, 25, 50, 75, 100]) { const xx = left + pct / 100 * pw; x.fillStyle = MUT; x.fillText(String(pct), xx - 9, y + 95); }
    y += 112;
    for (const t of P.tests) { const cy = y + rowH / 2, significant = t.p_adj < .05;
      if (significant) { x.fillStyle = '#fff3d6'; x.fillRect(4, y, c.width - 8, rowH - 2); }
      x.fillStyle = color(t.cluster); x.fillRect(14, cy - 7, 13, 13); x.fillStyle = FG; x.font = 'bold 14px sans-serif'; x.fillText(`Cluster ${t.cluster}`, 35, cy);
      for (const [g, col, dy] of [[ga, BLUE, -11], [gb, ORANGE, 11]]) {
        const yy = cy + dy, lo = Math.max(0, t.mean[g] - (t.sd[g] || 0)), hi = Math.min(100, t.mean[g] + (t.sd[g] || 0));
        x.strokeStyle = '#ddd'; x.lineWidth = 1; x.beginPath(); x.moveTo(left, yy); x.lineTo(left + pw, yy); x.stroke();
        x.strokeStyle = col; x.lineWidth = 4; x.beginPath(); x.moveTo(left + lo / 100 * pw, yy); x.lineTo(left + hi / 100 * pw, yy); x.stroke();
        t.values[g].forEach((v, i) => { x.fillStyle = col; x.beginPath(); x.arc(left + v / 100 * pw, yy + (i % 3 - 1) * 3, 3, 0, 2 * Math.PI); x.fill(); });
        x.strokeStyle = col; x.lineWidth = 2; x.beginPath(); x.moveTo(left + t.mean[g] / 100 * pw, yy - 8); x.lineTo(left + t.mean[g] / 100 * pw, yy + 8); x.stroke();
      }
      x.fillStyle = significant ? '#8c5a00' : MUT; x.font = significant ? 'bold 13px sans-serif' : '13px sans-serif';
      x.fillText(`${significant ? t.stars + ' ' : ''}${P.correction === 'none' ? 'p' : 'pBH'} ${t.p_adj.toFixed(3)}`, left + pw + 16, cy);
      y += rowH;
    }
    y += 18;
  }
  save(c, name); return c;
}

// DEK heatmap of one pair, one feature type and derivative: rows = clusters, columns = features, cell = signed t
export function dekPng(R, P, ty, n, name = 'dek.png') {
  const idx = R.columns.map((c, i) => [c, i]).filter(([c]) => c[0] === ty && c[1] === n), cw = 72, rh = 40, left = 82, top = 165;
  const [c, x] = canvas(left + idx.length * cw + 20, top + P.clusters.length * rh + 20), T = P.t.flat().filter(v => v != null), mx = Math.max(2, ...T.map(Math.abs));
  x.fillStyle = FG; x.font = 'bold 16px sans-serif'; x.fillText(`DEK ${ty}${n} · ${P.groups[0]} vs ${P.groups[1]}`, 10, 18);
  x.font = '12px sans-serif'; x.fillStyle = MUT; x.fillText(`${P.groups[0]} higher = orange · ${P.groups[1]} higher = blue · * = corrected p < .05`, 10, 40);
  x.fillText(`Welch t per cell · ${P.correction}`, 10, 60); x.fillText('Exploratory; groups are provisional.', 10, 78);
  idx.forEach(([col], j) => { x.save(); x.translate(left + j * cw + cw / 2, top - 7); x.rotate(-Math.PI / 2); x.fillStyle = FG; x.textAlign = 'left'; x.fillText(col.split(':')[1], 0, 0); x.restore(); });
  P.clusters.forEach((cl, r) => { x.fillStyle = color(cl); x.fillRect(10, top + r * rh + 10, 12, 12); x.fillStyle = FG; x.textAlign = 'left'; x.font = '13px sans-serif'; x.fillText(`${cl}  ${P.n[r].join('/')}`, 28, top + r * rh + rh / 2);
    idx.forEach(([, j], k) => { const t = P.t[r][j], q = P.p_adj[r][j]; if (t == null) return; const a = Math.min(1, Math.abs(t) / mx);
      x.fillStyle = t > 0 ? `rgba(255,159,10,${a})` : `rgba(90,200,250,${a})`; x.fillRect(left + k * cw, top + r * rh, cw - 2, rh - 2);
      if (q < .05) { x.strokeStyle = '#8c5a00'; x.lineWidth = 2; x.strokeRect(left + k * cw + 1, top + r * rh + 1, cw - 4, rh - 4); }
      x.fillStyle = FG; x.textAlign = 'center'; x.fillText(`${t.toFixed(1)}${q < .05 ? star(q) : ''}`, left + k * cw + cw / 2 - 1, top + r * rh + rh / 2 - 1); x.textAlign = 'left'; }); });
  save(c, name); return c;
}
