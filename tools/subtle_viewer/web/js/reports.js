import {$, st, api, dataQuery} from './state.js';

const url = (project, run, file) => '/result?' + new URLSearchParams({project, run, file}) + dataQuery;

export async function showReports() {
  const select = $('reportproject'), previous = select.value;
  try {
    const names = await api('projects');
    select.replaceChildren(...names.map(name => new Option(name, name)));
    select.value = names.includes(st.proj.name) ? st.proj.name : names.includes(previous) ? previous : names[0] || '';
    if (!select.value) { $('reportlist').textContent = 'No saved projects in this data root.'; return; }
    await loadReports();
  } catch (e) { $('reportlist').textContent = e.message; }
}

async function loadReports() {
  const project = $('reportproject').value, box = $('reportlist');
  box.textContent = 'Loading saved results…';
  try {
    const runs = await api('reports?name=' + encodeURIComponent(project));
    box.replaceChildren();
    if (!runs.length) { box.textContent = 'No saved result bundle for this project yet.'; return; }
    for (const run of runs) {
      const section = document.createElement('section'); section.className = 'report-card';
      const title = document.createElement('h3'); title.textContent = run.run; section.append(title);
      if (run.readme) { const note = document.createElement('pre'); note.className = 'report-note'; note.textContent = run.readme; section.append(note); }
      const files = document.createElement('div'); files.className = 'report-files';
      for (const file of run.files) {
        if (file === 'README.md') continue;
        const a = document.createElement('a'); a.href = url(project, run.run, file); a.textContent = file;
        a.target = '_blank'; a.rel = 'noopener'; files.append(a);
      }
      section.append(files);
      const figures = run.files.filter(f => f.endsWith('.png'));
      const priority = ['viewer_group_stats.png', 'quality_overview.png', 'group_stats_pair0.png', 'viewer_dek_E0.png'];
      figures.sort((a, b) => (priority.indexOf(a) < 0 ? 99 : priority.indexOf(a)) - (priority.indexOf(b) < 0 ? 99 : priority.indexOf(b)) || a.localeCompare(b));
      for (const [index, file] of figures.entries()) {
        const figure = document.createElement('details'); figure.className = 'report-figure'; figure.open = index === 0;
        const caption = document.createElement('summary'); caption.textContent = file; figure.append(caption);
        const img = document.createElement('img'); img.src = url(project, run.run, file);
        img.alt = `${run.run}: ${file}`; img.loading = 'lazy'; figure.append(img); section.append(figure);
      }
      box.append(section);
    }
  } catch (e) { box.textContent = e.message; }
}

$('reportproject').onchange = loadReports;
$('reportrefresh').onclick = showReports;
