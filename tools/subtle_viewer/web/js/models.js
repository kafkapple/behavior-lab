// Models tab: choose a keypoint model, predict the open session, see/open saved predictions, start training. One place instead of the five-step Pose dialog.
// Self-contained like compare.js (injects its own menu entry, tab and page). Server: pose_models.py, jobs reuse the pose_runtime venv.
import {$, st, api, toast} from './state.js';
import {watchPoseJob} from './pose.js';

document.querySelector('#left').insertAdjacentHTML('beforeend', '<details open><summary>Keypoint models</summary><button data-go="mdl">Choose model · Predict · Saved · Train →</button></details>');
document.querySelector('#tabs').insertAdjacentHTML('beforeend', '<button data-t="mdl">Models</button>');
document.querySelector('#right').insertAdjacentHTML('beforeend', `<div class="tabpage" id="t_mdl">
  <h3>1 · Model</h3><select id="mdlsel" style="width:100%"></select>
  <p class="mut" id="mdlnote">SLEAP single-instance models only. Two-stage (centroid + centered) models are listed but cannot run from one folder.</p>
  <h3>2 · Predict the open session</h3>
  <p id="mdlin" class="mut">open a session first</p>
  <button id="mdlgo">Predict and save as a session</button><pre id="mdlst" class="mut" style="white-space:pre-wrap"></pre>
  <h3>3 · Saved predictions of this video <button id="mdlref" title="re-read the folder">↻</button></h3><div id="mdlsaved" class="mut"></div>
  <details><summary>4 · Train a new model</summary>
    <p class="mut">Starts the existing training job (new model from a config, not a continuation). The finished model appears in the list above after ↻ in the Models tab.</p>
    <label>Training SLP <input id="mdltrain" style="width:90%"></label><label>Validation SLP (held out) <input id="mdlval" style="width:90%"></label>
    <label>Epochs <input id="mdlep" type="number" value="1" min="1" max="1000" style="width:70px"></label>
    <label>Device <select id="mdldev"><option>mps</option><option>cpu</option></select></label>
    <button id="mdltr">Train with the selected model's config</button></details></div>`);

let models = [];
const stem = p => p.split('/').pop().replace(/\.json\.gz$/, '').split('_sleap_')[0];
async function fill() {
  models = await api('kp_models');
  $('mdlsel').innerHTML = models.map(m => `<option value="${m.path}" ${m.usable ? '' : 'disabled'}>${m.name} · ${m.nodes.length} nodes${m.usable ? '' : ' · two-stage (not supported)'}</option>`).join('') || '<option>no model found under ~/data/subtle_models</option>';
  const ok = models.find(m => m.usable); if (ok) $('mdlsel').value = ok.path;
  $('mdlin').textContent = st.cur ? `${st.cur.path} · ${Object.keys(st.cur.cams || {}).length} camera videos` : 'open a session first';
  const all = await api('sessions'), mine = st.cur ? all.filter(s => stem(s.path) === stem(st.cur.path)) : [];
  $('mdlsaved').innerHTML = mine.map(s => `<div style="display:flex;gap:6px;align-items:center;margin:2px 0"><span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis">${s.path}</span><button data-open="${s.path}">Open</button></div>`).join('') || 'none yet';
  document.querySelectorAll('[data-open]').forEach(b => b.onclick = () => open(b.dataset.open));
}
function open(p) {
  const e = document.querySelector(`.sess[data-p="${CSS.escape(p)}"]`);
  if (e) return e.click();
  api('scan_start?force=1').then(() => { location.hash = `open=${p}`; location.reload(); });     // the session list was read before this file existed
}
document.querySelector('#tabs button[data-t="mdl"]').addEventListener('click', () => { fill().catch(e => toast(e.message)); setTimeout(() => $('right').classList.remove('wide'), 0); });
$('mdlref').onclick = () => fill().catch(e => toast(e.message));

let timer = null;
async function poll(id) {
  const r = await api('kp_predict_status?id=' + id);
  $('mdlst').textContent = `${r.model} · ${r.state}\n${(r.log || '').split('\n').filter(l => /predict cam|triangulate|error|failed/i.test(l)).slice(-3).join('\n')}`;
  if (r.state === 'running') return timer = setTimeout(() => poll(id), 1500);
  if (r.state === 'failed') return $('mdlst').textContent += '\n' + (r.result.error || '');
  $('mdlst').textContent += `\nsaved: ${r.result.session}`; await api('scan_start?force=1'); await fill(); toast('prediction saved');
}
$('mdlgo').onclick = async () => {
  if (!st.cur) return toast('open a session first');
  try { const r = await api('kp_predict', {session: st.cur.path, model: $('mdlsel').value}); clearTimeout(timer); $('mdlst').textContent = 'starting…'; poll(r.job); }
  catch (e) { $('mdlst').textContent = e.message; }
};
$('mdltr').onclick = async () => {
  const m = models.find(x => x.path === $('mdlsel').value); if (!m) return toast('choose a model');
  try { const r = await api('pose_start', {action: 'train', config: m.path + '/training_config.yaml', train: $('mdltrain').value.trim(), validation: $('mdlval').value.trim(), epochs: +$('mdlep').value, device: $('mdldev').value}); watchPoseJob(r.job); }
  catch (e) { toast(e.message); }
};
