// One cluster id -> one color, used by every view (skeleton, map, timeline, legend, video bones, frequency table).
// First 10 = Tableau 10 (color-blind friendly), next 10 = Tableau 20 light/dark mates, beyond = golden-angle hue.
const P = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
  '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#aec7e8'];
export const color = id => id == null || id < 0 ? '#888' : id < P.length ? P[id] : `hsl(${(id * 137.508) % 360} 60% 55%)`;
