"""Visualiseur de spectrogramme du poste d'annotation : composant Streamlit (page HTML et
JavaScript, sans outil de construction) qui renvoie à Python les intervalles tracés.

Tout ce qui ne change pas l'annotation se fait dans le navigateur, sans aller-retour avec
Python : le spectrogramme est calculé à la résolution de la vue (zoomer l'affine), une fois
la molette ou la souris lâchée (pendant le geste, l'image déjà calculée est étirée), la bande
d'écoute et le volume passent par Web Audio (on entend le réglage pendant qu'on le fait).
L'audio est servi comme fichier à côté de la page (`VIEWER_DIR/assets`) et chargé en mémoire
dans le navigateur, ce qui permet de reprendre la lecture n'importe où.

Commandes : voir `HELP` (affiché aussi dans le poste).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

VIEWER_DIR = Path(tempfile.gettempdir()) / "blanci_viewer"
KEEP_ASSETS = 40  # fichiers d'écoute gardés (les plus récents)

HELP = """\
| Geste | Effet |
|---|---|
| **Glisser** sur le spectrogramme | tracer un intervalle (type, dont faux ami : en haut) |
| **Tirer le bord** d'un intervalle | l'ajuster (le curseur devient ↔) |
| **Clic droit** sur un intervalle, ou le sélectionner puis **Suppr** | l'effacer |
| **Clic** (sans glisser) | placer la lecture à cet instant (elle reprend de là) |
| **Espace** | lecture / pause |
| **Molette** | zoom en temps autour du curseur |
| **Maj + molette** | zoom en fréquence |
| **Maj + glisser** | se déplacer (temps et fréquence) |
| **← →** | se déplacer dans le temps |
| **Double-clic**, ou « Tout » | revoir tout l'extrait |
| **Tirer un pointillé** (curseur ↕) | déplacer une limite de la bande d'écoute |

La bande d'écoute, « N'écouter que la bande », le volume, la dynamique et le micro du
spectrogramme se règlent dans la barre au-dessus du spectrogramme et s'appliquent
aussitôt ; ils sont gardés d'une séance à l'autre. Les touches agissent quand le
spectrogramme a la main : cliquer dessus d'abord.
"""

INDEX = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { margin: 0; font: 12px "Source Sans Pro", system-ui, sans-serif; color: #ddd;
         background: #0e1117; }
  .bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 2px 0 4px; }
  button, select, input { background: #262730; color: #ddd; border: 1px solid #444;
                          border-radius: 4px; padding: 2px 6px; font: inherit; }
  button { cursor: pointer; }
  button:hover { border-color: #00e676; }
  input[type=number] { width: 4.2em; }
  input[type=range] { width: 90px; vertical-align: middle; padding: 0; }
  .group { display: flex; gap: 4px; align-items: center; padding-right: 8px;
           border-right: 1px solid #333; }
  canvas { display: block; width: 100%; cursor: crosshair; outline: none; }
  #status { color: #888; margin-left: auto; }
  #players { display: flex; gap: 12px; margin-top: 4px; }
  #players div { flex: 1; }
  #players audio { width: 100%; height: 32px; }
  #players span { font-weight: 600; }
  #players .playing span { color: #00e676; }
  #list { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; min-height: 4px; }
  .chip { border: 1px solid #555; border-radius: 12px; padding: 2px 4px 2px 8px;
          display: flex; gap: 4px; align-items: center; cursor: pointer; }
  .chip.sel { border-color: #fff; }
  .chip select, .chip button { padding: 0 4px; }
</style></head><body>
<div class="bar">
  <span class="group" id="ivbar">Nouvel intervalle : <select id="newlabel"></select></span>
  <span class="group">
    <button id="zin" title="Zoom avant en temps">＋ temps</button>
    <button id="zout" title="Zoom arrière en temps">－ temps</button>
    <button id="fin" title="Zoom avant en fréquence">＋ fréq.</button>
    <button id="fout" title="Zoom arrière en fréquence">－ fréq.</button>
    <button id="win" title="Cadrer le candidat">Candidat</button>
    <button id="all" title="Tout voir (double-clic)">Tout</button>
    <label><input type="checkbox" id="follow" checked> Suivre</label>
  </span>
  <span id="status"></span>
</div>
<div class="bar">
  <span class="group">Bande
    <input type="number" id="lo" min="0" max="24" step="0.1"> –
    <input type="number" id="hi" min="0" max="24" step="0.1"> kHz
    <label><input type="checkbox" id="bandonly"> n'écouter qu'elle</label></span>
  <span class="group">Volume <input type="range" id="gain" min="0" max="30" step="1">
    <span id="gainv"></span></span>
  <span class="group">Dynamique <select id="range">
    <option value="40">40 dB</option><option value="50">50 dB</option>
    <option value="60">60 dB</option><option value="70">70 dB</option>
    <option value="80">80 dB</option></select></span>
  <span class="group">Spectrogramme <select id="chan"></select></span>
</div>
<canvas id="c" tabindex="0"></canvas>
<div id="list"></div>
<div id="players"></div>
<script>
const MAGMA = __MAGMA__;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const M = {l: 44, r: 8, t: 8, b: 24};
const PLAYHEAD = "#00e676";
const COLORS = {blanci: [80, 170, 255], blanci_chorus: [200, 120, 255],
                blanci_uncertain: [255, 170, 40], false_friend: [60, 230, 200]};
const $ = id => document.getElementById(id);
let D = null, players = [], graphs = [], buffers = [], active = 0, actx = null;
let view = null, intervals = [], selected = -1, vmax = [], loading = 0;
const spec = document.createElement("canvas"); let specKey = "", specView = null;
// Pendant un zoom ou un déplacement, l'image déjà calculée est seulement étirée ; le calcul à
// pleine résolution attend que l'on ait lâché la molette ou la souris.
let settling = false, settle = null;

// --- Réglages d'écoute, gardés d'une séance à l'autre --------------------------------------
const store = (() => { try { return window.localStorage; } catch (e) { return null; } })();
let S = (() => {
  try { return JSON.parse(store.getItem("blanci-viewer-settings")) || {}; } catch (e) { return {}; }
})();
function saveSettings() {
  try { store.setItem("blanci-viewer-settings", JSON.stringify(S)); } catch (e) {}
}
function showSettings() {
  $("lo").value = (S.band[0] / 1000).toFixed(1); $("hi").value = (S.band[1] / 1000).toFixed(1);
  $("bandonly").checked = !!S.bandOnly; $("gain").value = S.gain;
  $("gainv").textContent = `+${S.gain} dB`; $("range").value = String(S.range);
  $("chan").value = String(S.channel);
}
function applyAudio() {
  graphs.forEach(g => {
    const nyq = actx.sampleRate / 2;
    const lo = S.bandOnly ? Math.max(S.band[0], 10) : 10;
    const hi = S.bandOnly ? Math.min(S.band[1], nyq * 0.99) : nyq * 0.99;
    g.hp.forEach(f => f.frequency.value = lo);
    g.lp.forEach(f => f.frequency.value = hi);
    g.gain.gain.value = Math.pow(10, S.gain / 20);
  });
}
function setBand(lo, hi) {
  lo = Math.max(0, Math.min(lo, hi - 100)); hi = Math.max(hi, lo + 100);
  S.band = [Math.round(lo / 10) * 10, Math.round(hi / 10) * 10];
  showSettings(); applyAudio(); saveSettings(); draw();
}
$("lo").onchange = () => setBand(+$("lo").value * 1000, S.band[1]);
$("hi").onchange = () => setBand(S.band[0], +$("hi").value * 1000);
$("bandonly").onchange = () => {
  S.bandOnly = $("bandonly").checked; applyAudio(); saveSettings();
};
$("gain").oninput = () => {
  S.gain = +$("gain").value; showSettings(); applyAudio(); saveSettings();
};
$("range").onchange = () => { S.range = +$("range").value; saveSettings(); draw(); };
$("chan").onchange = () => { S.channel = +$("chan").value; saveSettings(); draw(); };

// --- Échanges avec Streamlit ----------------------------------------------------------------
function send(type, data) {
  window.parent.postMessage(Object.assign({isStreamlitMessage: true, type}, data), "*");
}
function setHeight() { send("streamlit:setFrameHeight", {height: document.body.scrollHeight}); }
const round = t => Math.round(t * 100) / 100;
function emit() {
  send("streamlit:setComponentValue", {dataType: "json", value: {
    key: D.extract, channel: active,
    intervals: intervals.map(i => [round(i.t0), round(i.t1), i.label]),
  }});
}

function render(args) {
  const fresh = !D || args.extract !== D.extract;
  D = args;
  if (S.band === undefined) {
    S = {band: D.band, bandOnly: false, gain: 0, range: 60, channel: D.channel};
  }
  $("ivbar").style.display = D.interval_mode ? "" : "none";
  if (!$("newlabel").options.length) {
    D.labels.forEach(([code, name]) => $("newlabel").add(new Option(name, code)));
    D.audios.forEach((a, i) => $("chan").add(new Option(a.name, String(i))));
  }
  showSettings();
  if (fresh) {
    view = {t0: D.t0, t1: D.t1, f0: 0, f1: D.fview};
    intervals = (D.intervals || []).map(([t0, t1, l]) => ({t0, t1, label: l}));
    selected = -1; vmax = []; specKey = ""; specView = null; settling = false; buffers = [];
    load();
    list();
  }
  resize(); setHeight();
}

// --- Audio : chargé en mémoire (lecture n'importe où), décodé pour le spectrogramme --------
async function load() {
  const token = ++loading;
  players.forEach(p => { p.pause(); URL.revokeObjectURL(p.src); });
  const box = $("players");
  box.innerHTML = ""; players = []; graphs = []; active = 0;
  $("status").textContent = "chargement de l'audio…";
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  const data = await Promise.all(D.audios.map(a => fetch(a.src).then(r => r.arrayBuffer())));
  if (token !== loading) return;
  D.audios.forEach((a, i) => {
    const div = document.createElement("div");
    div.innerHTML = `<span>${a.name}</span>`;
    const el = document.createElement("audio");
    el.controls = true; el.preload = "auto";
    el.src = URL.createObjectURL(new Blob([data[i]], {type: "audio/wav"}));
    const source = actx.createMediaElementSource(el);
    const hp = [0, 1].map(() => new BiquadFilterNode(actx, {type: "highpass", Q: 0.7071}));
    const lp = [0, 1].map(() => new BiquadFilterNode(actx, {type: "lowpass", Q: 0.7071}));
    const gain = new GainNode(actx);
    [source, ...hp, ...lp, gain].reduce((a, b) => (a.connect(b), b)).connect(actx.destination);
    graphs.push({hp, lp, gain});
    el.addEventListener("play", () => {
      if (actx.state !== "running") actx.resume();
      players.forEach((p, j) => { if (j !== i) p.pause(); });
      if (active !== i) { active = i; emit(); }
      mark(); loop();
    });
    el.addEventListener("seeked", draw);
    div.appendChild(el); box.appendChild(div); players.push(el);
  });
  applyAudio(); mark(); setHeight();
  for (let i = 0; i < data.length; i++) {
    const buf = await actx.decodeAudioData(data[i].slice(0));
    if (token !== loading) return;
    buffers[i] = buf.getChannelData(0); buffers.rate = buf.sampleRate;
    vmax[i] = reference(buffers[i]);
    draw();
  }
  $("status").textContent = "";
}
function mark() {
  [...$("players").children].forEach((d, j) => d.classList.toggle("playing", j === active));
}

// --- Spectrogramme calculé à la résolution de la vue --------------------------------------
const ffts = {};
function fftPlan(n) {
  if (ffts[n]) return ffts[n];
  const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2), win = new Float32Array(n);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = -Math.sin(2 * Math.PI * i / n);
  }
  let wsum = 0;
  for (let i = 0; i < n; i++) {
    win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); wsum += win[i];
  }
  const rev = new Uint32Array(n);
  for (let i = 0, bits = Math.log2(n); i < n; i++) {
    let r = 0; for (let b = 0; b < bits; b++) r = (r << 1) | ((i >> b) & 1); rev[i] = r;
  }
  return (ffts[n] = {cos, sin, win, rev, norm: wsum * wsum, re: new Float32Array(n),
                     im: new Float32Array(n), db: new Float32Array(n / 2 + 1)});
}
function spectrum(x, center, n) {  // dB de la trame centrée sur l'échantillon `center`
  const P = fftPlan(n), {re, im, rev, win} = P;
  const start = Math.round(center - n / 2);
  for (let i = 0; i < n; i++) {
    const s = start + i;
    re[rev[i]] = s >= 0 && s < x.length ? x[s] * win[i] : 0; im[rev[i]] = 0;
  }
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2, step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0, k = 0; j < half; j++, k += step) {
        const a = i + j, b = a + half;
        const tr = P.cos[k] * re[b] - P.sin[k] * im[b], ti = P.cos[k] * im[b] + P.sin[k] * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
  for (let k = 0; k <= n / 2; k++) {
    P.db[k] = 10 * Math.log10((re[k] * re[k] + im[k] * im[k]) / P.norm + 1e-14);
  }
  return P.db;
}
// Haut de l'échelle de couleurs, fixe pour l'extrait et propre à chaque micro : le centile
// 99,5 des niveaux entre 500 Hz et le haut de la vue par défaut, comme les figures de la
// présentation (un claquement isolé n'assombrit plus tout le reste).
function reference(x) {
  const n = 1024, cols = 600, binHz = buffers.rate / n;
  const k0 = Math.ceil(500 / binHz), k1 = Math.min(n / 2, Math.floor(D.fview / binHz));
  const levels = new Float32Array(cols * (k1 - k0 + 1));
  let m = 0;
  for (let c = 0; c < cols; c++) {
    const db = spectrum(x, (c + 0.5) / cols * x.length, n);
    for (let k = k0; k <= k1; k++) levels[m++] = db[k];
  }
  levels.sort();
  return levels[Math.floor(0.995 * (m - 1))];
}
function fftSize() {
  const rate = buffers.rate, tspan = view.t1 - view.t0, fspan = view.f1 - view.f0;
  if (fspan <= 1200) return rate > 30000 ? 8192 : 4096;
  if (fspan <= 3000) return rate > 30000 ? 4096 : 2048;
  if (tspan <= 2) return rate > 30000 ? 1024 : 512;
  return rate > 30000 ? 2048 : 1024;
}
function renderSpectrogram(cols, rows) {
  const ch = buffers[S.channel] ? S.channel : 0, x = buffers[ch];  // micro 2 pas encore décodé
  const n = fftSize(), rate = buffers.rate;
  const start = D.audios[ch] ? D.audios[ch].start : D.t0;
  const key = [view.t0, view.t1, view.f0, view.f1, cols, rows, n, S.range, ch].join();
  if (key === specKey) return;
  specKey = key; specView = {...view};
  spec.width = cols; spec.height = rows;
  const sctx = spec.getContext("2d"), image = sctx.createImageData(cols, rows), px = image.data;
  const binHz = rate / n, top = vmax[ch], range = S.range;
  const binOf = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    binOf[r] = (view.f1 - (r + 0.5) / rows * (view.f1 - view.f0)) / binHz;
  }
  for (let c = 0; c < cols; c++) {
    const t = view.t0 + (c + 0.5) / cols * (view.t1 - view.t0);
    const db = spectrum(x, (t - start) * rate, n);
    for (let r = 0; r < rows; r++) {
      const b = binOf[r], k = Math.floor(b), f = b - k;
      const v = k + 1 < db.length ? db[k] * (1 - f) + db[k + 1] * f : db[db.length - 1];
      const i = Math.max(0, Math.min(255, Math.round((v - (top - range)) / range * 255)));
      const o = (r * cols + c) * 4, rgb = MAGMA[i];
      px[o] = rgb[0]; px[o + 1] = rgb[1]; px[o + 2] = rgb[2]; px[o + 3] = 255;
    }
  }
  sctx.putImageData(image, 0, 0);
}

// --- Vue ---------------------------------------------------------------------------------
const H = () => D ? D.height : 300;
const W = () => canvas.clientWidth - M.l - M.r;
const PH = () => H() - M.t - M.b;
const fmax = () => buffers.rate ? Math.min(buffers.rate / 2, 24000) : D.fview;
const xOf = t => M.l + (t - view.t0) / (view.t1 - view.t0) * W();
const yOf = f => M.t + (1 - (f - view.f0) / (view.f1 - view.f0)) * PH();
const tOf = x => view.t0 + (x - M.l) / W() * (view.t1 - view.t0);
const fOf = y => view.f0 + (1 - (y - M.t) / PH()) * (view.f1 - view.f0);

function resize() {
  const r = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * r; canvas.height = H() * r;
  canvas.style.height = H() + "px";
  ctx.setTransform(r, 0, 0, r, 0, 0);
  draw();
}
function clampView() {
  const span = Math.min(Math.max(view.t1 - view.t0, 0.1), D.t1 - D.t0);
  const t0 = Math.min(Math.max(view.t0, D.t0), D.t1 - span);
  view.t0 = t0; view.t1 = t0 + span;
  const fspan = Math.min(Math.max(view.f1 - view.f0, 200), fmax());
  const f0 = Math.min(Math.max(view.f0, 0), fmax() - fspan);
  view.f0 = f0; view.f1 = f0 + fspan;
}
function niceStep(span, n) {
  const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 5, 10].map(m => m * p).find(s => s >= raw);
}
function minorStep(step) {  // 5 petites graduations par grande (4 pour un pas de 2)
  const lead = Math.round(step / Math.pow(10, Math.floor(Math.log10(step) + 1e-9)));
  return step / (lead === 2 ? 4 : 5);
}
function playheadTime() {
  const p = players[active];
  return p ? D.audios[active].start + p.currentTime : null;
}

function moved() {  // vue changée : image étirée tout de suite, recalculée une fois au repos
  settling = true; draw();
  clearTimeout(settle);
  settle = setTimeout(function rest() {
    if (drag) settle = setTimeout(rest, 150); else { settling = false; draw(); }
  }, 250);
}
function draw() {
  if (!D || !view) return;
  const w = canvas.clientWidth;
  ctx.fillStyle = "#0e1117"; ctx.fillRect(0, 0, w, H());
  if (buffers.length && vmax.length && settling && specView) {
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W(), PH()); ctx.clip();
    ctx.imageSmoothingEnabled = true;
    const x0 = xOf(specView.t0), y0 = yOf(specView.f1);
    ctx.drawImage(spec, x0, y0, xOf(specView.t1) - x0, yOf(specView.f0) - y0);
    ctx.restore();
  } else if (buffers.length && vmax.length) {
    renderSpectrogram(Math.max(1, Math.round(W())), Math.max(1, Math.round(PH())));
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(spec, M.l, M.t, W(), PH());
  } else {
    ctx.fillStyle = "#888"; ctx.fillText("calcul du spectrogramme…", M.l + 10, M.t + 20);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, W(), PH()); ctx.clip();
  (D.windows || []).forEach(win => {
    const x0 = xOf(win.t0), x1 = xOf(win.t1);
    if (win.current) {
      ctx.fillStyle = "rgba(0, 255, 255, 0.13)"; ctx.fillRect(x0, M.t, x1 - x0, PH());
      ctx.strokeStyle = "cyan"; ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, M.t + 0.5, x1 - x0, PH() - 1);
    } else if (!D.interval_mode) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, M.t); ctx.lineTo(x0, M.t + PH());
      ctx.moveTo(x1, M.t); ctx.lineTo(x1, M.t + PH()); ctx.stroke();
    }
    if (win.label) {
      ctx.fillStyle = "rgba(80, 200, 120, 0.85)"; ctx.fillRect(x0, M.t, x1 - x0, 14);
      ctx.fillStyle = "#000"; ctx.font = "11px system-ui";
      ctx.fillText(win.label, x0 + 3, M.t + 11, Math.max(x1 - x0 - 6, 1));
    }
    if (win.candidate) {
      ctx.fillStyle = "#ffd60a"; ctx.fillRect(x0, M.t + PH() - 3, x1 - x0, 3);
    }
  });
  const shown = drag && drag.mode === "draw" && drag.moved
    ? [...intervals, {t0: Math.min(drag.t, drag.t2), t1: Math.max(drag.t, drag.t2),
                      label: $("newlabel").value}]
    : intervals;
  shown.forEach((iv, i) => {
    const x0 = xOf(iv.t0), x1 = xOf(iv.t1), c = COLORS[iv.label] || [200, 200, 200];
    ctx.fillStyle = `rgba(${c}, 0.20)`; ctx.fillRect(x0, M.t, x1 - x0, PH());
    ctx.fillStyle = `rgba(${c}, 0.95)`; ctx.fillRect(x0, M.t, x1 - x0, 5);
    ctx.strokeStyle = `rgb(${c})`; ctx.lineWidth = i === selected ? 3 : 1.5;
    ctx.beginPath(); ctx.moveTo(x0, M.t); ctx.lineTo(x0, M.t + PH());
    ctx.moveTo(x1, M.t); ctx.lineTo(x1, M.t + PH()); ctx.stroke();
  });
  ctx.setLineDash([5, 4]); ctx.strokeStyle = "rgba(255, 255, 255, 0.85)"; ctx.lineWidth = 1.2;
  S.band.forEach(f => {
    const y = yOf(f); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + W(), y); ctx.stroke();
  });
  ctx.setLineDash([]);
  const t = playheadTime();
  if (t !== null) {
    const x = xOf(t);
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + PH()); ctx.stroke();
    ctx.strokeStyle = PLAYHEAD; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + PH()); ctx.stroke();
  }
  ctx.restore();
  axes();
}
function axes() {
  ctx.fillStyle = "#aaa"; ctx.strokeStyle = "#777"; ctx.lineWidth = 1; ctx.font = "11px system-ui";
  const ts = niceStep(view.t1 - view.t0, Math.max(2, W() / 90)), tm = minorStep(ts);
  for (let v = Math.ceil(view.t0 / tm) * tm; v <= view.t1 + 1e-9; v += tm) {
    const x = Math.round(xOf(v)) + 0.5, major = Math.abs(v / ts - Math.round(v / ts)) < 1e-6;
    ctx.beginPath(); ctx.moveTo(x, M.t + PH());
    ctx.lineTo(x, M.t + PH() + (major ? 6 : 3)); ctx.stroke();
    if (major) ctx.fillText(+v.toFixed(3) + " s", x - 10, H() - 5);
  }
  const fs = niceStep(view.f1 - view.f0, Math.max(2, PH() / 40)), fm = minorStep(fs);
  for (let v = Math.ceil(view.f0 / fm) * fm; v <= view.f1 + 1e-9; v += fm) {
    const y = Math.round(yOf(v)) + 0.5, major = Math.abs(v / fs - Math.round(v / fs)) < 1e-6;
    ctx.beginPath(); ctx.moveTo(M.l - (major ? 6 : 3), y); ctx.lineTo(M.l, y); ctx.stroke();
    if (major) ctx.fillText(+(v / 1000).toFixed(2) + "", 4, y + 4);
  }
  ctx.fillText("kHz", 4, H() - 5);
}

let raf = null;
function loop() {
  if (raf) return;
  const step = () => {
    const p = players[active], t = playheadTime();
    if (t !== null && $("follow").checked && !drag && (t > view.t1 || t < view.t0)) {
      const span = view.t1 - view.t0;
      view.t0 = t - 0.1 * span; view.t1 = view.t0 + span; clampView(); moved();
    } else draw();
    raf = p && !p.paused ? requestAnimationFrame(step) : null;
  };
  raf = requestAnimationFrame(step);
}
function zoomTime(factor, at) {
  at = at ?? (view.t0 + view.t1) / 2;
  view.t0 = at - (at - view.t0) * factor; view.t1 = at + (view.t1 - at) * factor;
  clampView(); moved();
}
function zoomFreq(factor, at) {
  at = at ?? (view.f0 + view.f1) / 2;
  view.f0 = at - (at - view.f0) * factor; view.f1 = at + (view.f1 - at) * factor;
  clampView(); moved();
}

// --- Intervalles -------------------------------------------------------------------------
function list() {
  const box = $("list");
  box.innerHTML = "";
  intervals.map((iv, i) => [iv, i]).sort((a, b) => a[0].t0 - b[0].t0).forEach(([iv, i]) => {
    const chip = document.createElement("span");
    chip.className = "chip" + (i === selected ? " sel" : "");
    const c = COLORS[iv.label] || [200, 200, 200];
    chip.style.background = `rgba(${c}, 0.18)`;
    chip.append(`${iv.t0.toFixed(1)}–${iv.t1.toFixed(1)} s`);
    const s = document.createElement("select");
    D.labels.forEach(([code, name]) => s.add(new Option(name, code, false, code === iv.label)));
    s.onchange = () => { iv.label = s.value; changed(); };
    const x = document.createElement("button");
    x.textContent = "×"; x.title = "Effacer";
    x.onclick = ev => { ev.stopPropagation(); remove(i); };
    chip.onclick = () => {
      selected = i;
      const span = view.t1 - view.t0;
      if (iv.t1 < view.t0 || iv.t0 > view.t1) {
        view.t0 = (iv.t0 + iv.t1) / 2 - span / 2; view.t1 = view.t0 + span; clampView();
      }
      list(); draw();
    };
    chip.append(s, x); box.appendChild(chip);
  });
  setHeight();
}
function changed() { list(); draw(); emit(); }
function remove(i) { intervals.splice(i, 1); selected = -1; changed(); }
const localX = e => e.clientX - canvas.getBoundingClientRect().left;
const localY = e => e.clientY - canvas.getBoundingClientRect().top;
function hit(x) {  // [indice, bord ("t0", "t1") ou null] de l'intervalle sous le curseur
  for (let i = intervals.length - 1; i >= 0; i--) {
    const iv = intervals[i];
    if (Math.abs(x - xOf(iv.t0)) < 6) return [i, "t0"];
    if (Math.abs(x - xOf(iv.t1)) < 6) return [i, "t1"];
  }
  const t = tOf(x);
  return [intervals.findIndex(iv => t >= iv.t0 && t <= iv.t1), null];
}
function bandLine(y) {  // 0 ou 1 : pointillé de la bande sous le curseur, sinon -1
  return S.band.findIndex(f => Math.abs(yOf(f) - y) < 5);
}

// --- Souris et clavier -------------------------------------------------------------------
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = Math.exp((e.deltaY || e.deltaX) * 0.0015);
  if (e.shiftKey) zoomFreq(factor, fOf(localY(e)));
  else zoomTime(factor, tOf(localX(e)));
}, {passive: false});

let drag = null;
canvas.addEventListener("mousedown", e => {
  if (e.button !== 0 || !D) return;
  canvas.focus();
  const x = localX(e), y = localY(e), t = tOf(x), line = bandLine(y);
  const [i, edge] = D.interval_mode ? hit(x) : [-1, null];
  let mode = "pan";
  if (line >= 0 && !e.shiftKey) mode = "band";
  else if (D.interval_mode && !e.shiftKey) mode = edge ? "edge" : "draw";
  drag = {mode, x: e.clientX, y: e.clientY, view: {...view}, moved: false, t, t2: t, i, edge, line};
});
canvas.addEventListener("mousemove", e => {
  if (drag || !D) return;
  const x = localX(e), y = localY(e);
  canvas.style.cursor = bandLine(y) >= 0 ? "ns-resize"
    : D.interval_mode && hit(x)[1] ? "ew-resize" : "crosshair";
});
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  if (!drag.moved) return;
  const t = Math.min(Math.max(tOf(localX(e)), D.t0), D.t1);
  if (drag.mode === "pan") {
    const dt = dx / W() * (drag.view.t1 - drag.view.t0);
    const df = dy / PH() * (drag.view.f1 - drag.view.f0);
    view = {t0: drag.view.t0 - dt, t1: drag.view.t1 - dt,
            f0: drag.view.f0 + df, f1: drag.view.f1 + df};
    clampView(); moved(); return;
  } else if (drag.mode === "band") {
    const f = Math.max(0, fOf(localY(e)));
    if (drag.line === 0) setBand(Math.min(f, S.band[1] - 100), S.band[1]);
    else setBand(S.band[0], Math.max(f, S.band[0] + 100));
  } else if (drag.mode === "edge") {
    intervals[drag.i][drag.edge] = t; selected = drag.i;
  } else {
    drag.t2 = t;
  }
  draw();
});
window.addEventListener("mouseup", () => {
  if (!drag) return;
  const d = drag; drag = null;
  if (d.mode === "pan" && d.moved) { clearTimeout(settle); settling = false; draw(); return; }
  if (!d.moved) {
    if (D.interval_mode && d.i >= 0) { selected = d.i; list(); }
    const p = players[active];  // la lecture reprendra de cet instant
    if (p && d.t >= D.t0 && d.t <= D.t1) p.currentTime = Math.max(0, d.t - D.audios[active].start);
    draw();
    return;
  }
  if (d.mode === "draw") {
    const t0 = Math.min(d.t, d.t2), t1 = Math.max(d.t, d.t2);
    if (t1 - t0 >= 0.05) {
      intervals.push({t0, t1, label: $("newlabel").value});
      selected = intervals.length - 1; changed();
    } else draw();
  } else if (d.mode === "edge") {
    const iv = intervals[d.i];
    [iv.t0, iv.t1] = [Math.min(iv.t0, iv.t1), Math.max(iv.t0, iv.t1)];
    if (iv.t1 - iv.t0 < 0.05) remove(d.i); else changed();
  }
});
canvas.addEventListener("contextmenu", e => {
  if (!D || !D.interval_mode) return;
  const [i] = hit(localX(e));
  if (i >= 0) { e.preventDefault(); remove(i); }
});
canvas.addEventListener("dblclick", () => {
  view = {t0: D.t0, t1: D.t1, f0: 0, f1: Math.min(D.fview, fmax())}; draw();
});
$("zin").onclick = () => zoomTime(0.5);
$("zout").onclick = () => zoomTime(2);
$("fin").onclick = () => zoomFreq(0.5);
$("fout").onclick = () => zoomFreq(2);
$("all").onclick = () => canvas.dispatchEvent(new Event("dblclick"));
$("win").onclick = () => {
  const w = (D.windows || []).find(w => w.current || w.candidate) || (D.windows || [])[0];
  if (!w) return;
  const pad = 0.5 * (w.t1 - w.t0);
  view.t0 = w.t0 - pad; view.t1 = w.t1 + pad; clampView(); draw();
};
document.addEventListener("keydown", e => {
  if (["SELECT", "INPUT"].includes(e.target.tagName)) return;
  if (e.code === "Space") {
    e.preventDefault();
    const p = players[active]; if (p) (p.paused ? p.play() : p.pause());
  } else if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0) {
    remove(selected);
  } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    e.preventDefault();
    const span = view.t1 - view.t0, s = (e.key === "ArrowLeft" ? -0.25 : 0.25) * span;
    view.t0 += s; view.t1 += s; clampView(); moved();
  }
});

window.addEventListener("message", e => {
  if (e.data && e.data.type === "streamlit:render") render(e.data.args);
});
window.addEventListener("resize", () => { if (D) resize(); });
send("streamlit:componentReady", {apiVersion: 1});
</script></body></html>
"""


def _page() -> str:
    from matplotlib import colormaps

    magma = [[round(255 * c) for c in colormaps["magma"](i / 255)[:3]] for i in range(256)]
    return INDEX.replace("__MAGMA__", json.dumps(magma))


def _prepare() -> Path:
    """Page du composant, réécrite si elle a changé (mise à jour du code)."""
    (VIEWER_DIR / "assets").mkdir(parents=True, exist_ok=True)
    index = VIEWER_DIR / "index.html"
    page = _page()
    if not index.exists() or index.read_text(encoding="utf-8") != page:
        index.write_text(page, encoding="utf-8")
    return VIEWER_DIR


def asset(data: bytes, suffix: str) -> str:
    """Range `data` à côté de la page (nom = empreinte du contenu) ; chemin relatif à la page.
    Seuls les `KEEP_ASSETS` fichiers les plus récents sont gardés."""
    folder = VIEWER_DIR / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(data).hexdigest()[:20] + suffix
    path = folder / name
    if not path.exists():
        path.write_bytes(data)
        old = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in old[KEEP_ASSETS:]:
            stale.unlink(missing_ok=True)
    return f"assets/{name}"


_component = None


def viewer_args(
    t0: float,
    t1: float,
    audios: list[tuple[str, str, float]],
    windows: list[dict],
    band_hz: tuple[float, float],
    key: str,
    labels: list[tuple[str, str]],
    intervals: list | None = None,
    interval_mode: bool = False,
    channel: int = 0,
    fview_hz: float = 10_000.0,
    height: int = 320,
) -> dict:
    """Arguments du composant. `audios` : (nom, chemin rendu par `asset`, début en s dans
    l'enregistrement), couvrant [t0, t1] ; `windows` : {t0, t1, label, current, candidate} ;
    `key` : identifie l'extrait (un autre extrait remet la vue et les intervalles à zéro) ;
    `labels` : (code, nom) des intervalles ; `intervals` : [début, fin, label] affichés à
    l'ouverture de l'extrait ; `band_hz`, `channel` : réglages de départ, la première fois."""
    return {
        "t0": float(t0),
        "t1": float(t1),
        "fview": float(fview_hz),
        "audios": [
            {"name": name, "src": src, "start": float(start)} for name, src, start in audios
        ],
        "windows": windows,
        "band": [float(f) for f in band_hz],
        "extract": key,
        "labels": [list(label) for label in labels],
        "intervals": [[float(a), float(b), str(c)] for a, b, c in intervals or []],
        "interval_mode": bool(interval_mode),
        "channel": int(channel),
        "height": int(height),
    }


def viewer(args: dict, component_key: str = "viewer") -> dict | None:
    """Affiche le visualiseur ; renvoie sa dernière valeur ({key, channel, intervals}) ou None."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("blanci_viewer", path=str(_prepare()))
    return _component(**args, key=component_key, default=None)
