// Drives the viewer in headless Chrome over the DevTools protocol (Node >= 22, no packages) and asserts what a user does:
// keys, timeline drag, cluster names, Train and Map buttons. usage: node browser_check.mjs <keys|train> <base url> [chrome path]
import {spawn} from 'node:child_process';
import {mkdtempSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

const [mode, base, chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'] = process.argv.slice(2);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const proc = spawn(chrome, ['--headless=new', '--disable-gpu', '--remote-debugging-port=9333', `--user-data-dir=${mkdtempSync(join(tmpdir(), 'cdp-'))}`, '--window-size=1500,900', 'about:blank'], {stdio: 'ignore'});
let ok = 0, bad = 0;
const check = (name, cond, extra = '') => { cond ? ok++ : bad++; console.log(`${cond ? 'PASS' : 'FAIL'} ${name} ${extra}`); };

async function connect() {
  for (let i = 0; i < 50; i++) { try { const l = await (await fetch('http://127.0.0.1:9333/json')).json(); const p = l.find(t => t.type === 'page'); if (p) return p.webSocketDebuggerUrl; } catch (e) { /* not up yet */ } await sleep(200); }
  throw new Error('chrome did not start');
}
const ws = new WebSocket(await connect()); await new Promise(r => ws.onopen = r);
let id = 0; const wait = new Map(); const errors = [];
ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); }
  if (d.method === 'Runtime.exceptionThrown') errors.push(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text); };
const send = (method, params = {}) => new Promise(r => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({id: i, method, params})); });
const js = async expr => { const r = await send('Runtime.evaluate', {expression: expr, awaitPromise: true, returnByValue: true}); if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails)); return r.result.result.value; };
const until = async (expr, ms = 20000) => { const t = Date.now(); while (Date.now() - t < ms) { if (await js(expr)) return true; await sleep(250); } return false; };
const VK = {ArrowRight: 39, ArrowLeft: 37, ArrowUp: 38, ArrowDown: 40, Escape: 27, ',': 188, '.': 190, Enter: 13, ' ': 32};
const key = async k => { const vk = VK[k] ?? k.toUpperCase().charCodeAt(0); const p = {key: k, code: k.length === 1 ? (/\d/.test(k) ? 'Digit' + k : 'Key' + k.toUpperCase()) : k, windowsVirtualKeyCode: vk};
  await send('Input.dispatchKeyEvent', {type: 'keyDown', ...p}); await send('Input.dispatchKeyEvent', {type: 'keyUp', ...p}); await sleep(150); };
const mouse = (type, x, y) => send('Input.dispatchMouseEvent', {type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : 1, clickCount: 1});
const frame = () => js("+document.getElementById('fi').textContent.split('/')[0]");
const $ = id => `document.getElementById('${id}')`;

await send('Runtime.enable'); await send('Page.enable');
await send('Page.navigate', {url: base});
await until(`!!${$('fi')} && ${$('fi')}.textContent.includes('/')`);

if (mode === 'keys') {
  await until(`document.querySelectorAll('.lrow').length > 0`);
  const f0 = await frame(); check('session opened at frame 600', f0 === 600, `(fi=${f0})`);
  await key('ArrowRight'); check('ArrowRight = +1', await frame() === f0 + 1);
  await key('ArrowUp'); check('ArrowUp = +10', await frame() === f0 + 11);
  await key('ArrowDown'); check('ArrowDown = -10', await frame() === f0 + 1);
  await key(','); check('"," = -1', await frame() === f0);
  await key('.'); check('"." = +1', await frame() === f0 + 1);
  await key('h'); check('h hides side panels', await js("document.body.classList.contains('hide')")); await key('h');
  await key('l'); check('l opens Clusters tab', await js(`${$('t_clu')}.classList.contains('on')`));
  await key('0'); check('0 shows 5 camera tiles', await until(`document.querySelectorAll('.tile').length === 5`), `(tiles=${await js("document.querySelectorAll('.tile').length")})`);
  await key('3'); check('3 selects cam_3 only', await until(`document.querySelectorAll('.tile').length === 1 && ${$('camsel')}.value === '2'`));
  await key('k'); check('k toggles keypoints', (await js(`${$('toast')}.textContent`)).includes('keypoints off'));
  await key('k');
  // clicking a cluster row filters; Escape clears
  await js(`document.querySelector('.lrow').click()`); check('click row = filter (row marked)', await until(`document.querySelectorAll('.lrow.on').length === 1`));
  await key('Escape'); check('Escape clears filter', await until(`document.querySelectorAll('.lrow.on').length === 0`));
  // timeline drag: press at 20 percent, drag to 60 percent
  const r = await js(`(() => { const b = ${$('tl')}.getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })()`);
  const y = r[1] + r[3] / 2, n = await js(`+${$('fr')}.max`);
  await mouse('mousePressed', r[0] + r[2] * .2, y); await mouse('mouseMoved', r[0] + r[2] * .4, y); await mouse('mouseMoved', r[0] + r[2] * .6, y);
  const fMid = await frame(); await mouse('mouseReleased', r[0] + r[2] * .6, y);
  check('timeline drag scrubs to ~60 percent', Math.abs(fMid / n - .6) < .03, `(frame ${fMid} of ${n})`);
  const fEnd = await frame(); await mouse('mouseMoved', r[0] + r[2] * .9, y); check('after release the pointer no longer scrubs', await frame() === fEnd);
  // slider drag
  const s = await js(`(() => { const b = ${$('fr')}.getBoundingClientRect(); return [b.left, b.top, b.width, b.height]; })()`);
  await mouse('mousePressed', s[0] + s[2] * .1, s[1] + s[3] / 2); await mouse('mouseMoved', s[0] + s[2] * .5, s[1] + s[3] / 2); await mouse('mouseReleased', s[0] + s[2] * .5, s[1] + s[3] / 2);
  check('slider drag to ~50 percent', Math.abs(await frame() / n - .5) < .03, `(frame ${await frame()})`);
  // play / pause moves frames
  const f1 = await frame(); await key(' '); await sleep(1500); const fp = await frame(); await key(' ');
  check('Space plays (frame moved) and pauses', fp !== f1 && await js(`${$('play')}.textContent`) === '▶', `(frame ${f1} -> ${fp})`);
  // cluster name saved to the server
  await js(`document.querySelector('.lrow input[data-n]').focus()`); await send('Input.insertText', {text: 'rearing_test'}); await key('Tab');
  const saved = await until(`fetch('/api/names').then(r => r.json()).then(d => Object.values(d).includes('rearing_test'))`);
  check('typed cluster name saved on the server', saved);
  await js(`(async () => { const d = await fetch('/api/names').then(r => r.json()); for (const k of Object.keys(d)) if (d[k] === 'rearing_test') await fetch('/api/set_name', {method: 'POST', body: JSON.stringify({key: k, name: ''})}); })()`);
  // the embedding exists for the fitted frames only (first 3,000): seek inside them before looking at the map
  await mouse('mousePressed', r[0] + r[2] * .1, y); await mouse('mouseReleased', r[0] + r[2] * .1, y);
  check('seek into the embedded frames', await frame() < 3000, `(frame ${await frame()})`);
  // trace: 10 linked dots change the map drawing; back to 1 restores it
  const hash = () => js(`(() => { const d = ${$('mp')}.getContext('2d').getImageData(0, 0, ${$('mp')}.width, ${$('mp')}.height).data; let h = 0; for (let i = 0; i < d.length; i += 1) h = (h * 31 + d[i]) | 0; return h; })()`);
  await js(`${$('trace')}.value = '1'; ${$('trace')}.dispatchEvent(new Event('change'))`); const h1 = await hash();
  await js(`${$('trace')}.value = '10'; ${$('trace')}.dispatchEvent(new Event('change'))`); const h10 = await hash();
  await js(`${$('trace')}.value = '1'; ${$('trace')}.dispatchEvent(new Event('change'))`);
  check('trace 10 changes the cluster map, trace 1 restores it', h1 !== h10 && h1 === await hash());
  // level sweep: recommendation, flagged marks, isolate, and the saved result folder
  await js(`${$('swgo')}.click()`); check('Sweep n shows the level table', await until(`document.querySelectorAll('#sweepres table tr').length >= 4`, 30000));
  const rec = await js(`(document.querySelector('#sweepres .ok, #sweepres .bad') || {}).textContent || ''`); check('recommendation text states significance honestly', /recommended n = \d/.test(rec), `(${rec.slice(0, 90)})`);
  await js(`${$('swcrit')}.value = 'meanp'; ${$('swgo')}.click()`); check('another criterion re-ranks the levels', await until(`document.querySelector('#sweepres table th') && document.querySelector('#sweepres').textContent.includes('Recommended n per criterion')`, 30000));
  await js(`${$('swcrit')}.value = 'omnibus'; ${$('swgo')}.click()`); await sleep(1500);
  await js(`${$('swuse')}.click()`); const kk = await js(`+document.getElementById('swuse').textContent.replace(/\\D/g, '')`);
  check('Use n sets the level (legend rows = n)', await until(`document.querySelectorAll('.lrow').length === ${kk}`, 5000), `(n=${kk})`);
  await js(`${$('swexp')}.click()`); check('Export results saves a folder', await until(`!!document.getElementById('swdirtxt')`, 120000), await js(`(document.getElementById('swdirtxt') || {}).textContent`));
  // PNG exports: capture the download and check the PNG signature
  await js(`window.__dl = []; HTMLAnchorElement.prototype.click = function () { fetch(this.href).then(r => r.blob()).then(async b => window.__dl.push({name: this.download, size: b.size, magic: [...new Uint8Array(await b.slice(0, 4).arrayBuffer())].join()})); }`);
  await js(`${$('stgo')}.click()`); await until(`!!document.querySelector('#stat table')`, 30000); await js(`${$('stpng')}.click()`);
  check('Stats PNG downloaded', await until(`window.__dl.some(d => d.name === 'group_stats.png' && d.magic === '137,80,78,71' && d.size > 4000)`, 8000));
  await js(`${$('dkgo')}.click()`); await until(`!!document.querySelector('#dek table')`, 60000); await js(`${$('dkpng')}.click()`);
  check('DEK PNG downloaded', await until(`window.__dl.some(d => d.name === 'dek.png' && d.magic === '137,80,78,71' && d.size > 4000)`, 8000), JSON.stringify(await js('window.__dl')));
} else if (mode === 'train') {
  check('session opened', await until(`+${$('fr')}.max > 0`, 30000));
  await js(`${$('gn')}.value = 'g1'; ${$('ga')}.click()`);
  check('group created from the open session', await until(`document.querySelectorAll('.grpbox').length === 1`));
  await js(`${$('pn')}.value = 'uitest'; ${$('ps')}.click()`); await sleep(800);
  await js(`${$('mn')}.value = 'ui1'; ${$('mt')}.click()`);
  check('Train starts a job (progress shown)', await until(`${$('pg')}.style.display === 'block'`, 8000));
  check('Train finishes without error', await until(`/done|SUBTLE done/.test(${$('job')}.textContent) || ${$('job')}.textContent.includes('100%')`, 240000) && !(await js(`${$('job')}.textContent`)).includes('Error'), `(${await js(`${$('job')}.textContent`)})`);
  await js(`${$('t_proj')}.scrollTop = 0`); await sleep(1500);
  check('model appears in the Model select', await until(`[...${$('ml')}.options].some(o => o.value === 'ui1')`, 15000));
  await js(`${$('ml')}.value = 'ui1'; ${$('mm')}.click()`);
  check('Map finishes and labels all frames', await until(`${$('log')}.textContent.includes('mapped 1 sessions with ui1')`, 240000));
} else throw new Error('mode');

check('no uncaught page exceptions', errors.length === 0, errors.join(' | ').slice(0, 300));
console.log(`\n${ok} passed, ${bad} failed`);
ws.close(); proc.kill(); process.exit(bad ? 1 : 0);
