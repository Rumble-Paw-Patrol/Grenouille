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
import threading
from pathlib import Path

from blanci.annotation.style import FONTS, LABEL_COLORS
from blanci.inputs.labels import QUALITIES

VIEWER_DIR = Path(tempfile.gettempdir()) / "blanci_viewer"
KEEP_ASSETS = 40  # fichiers d'écoute gardés (les plus récents)

HELP = """\
| Fonction | Geste |
|---|---|
| Placer le lecteur à un instant précis | **Clic** sur le spectrogramme (sans glisser), ou clic \
sur la piste audio |
| Lecture / pause | **Barre d'espace** |
| Zoom en temps autour du curseur | **Molette** |
| Zoom en fréquence autour du curseur | **Maj + molette** |
| Revenir à la vue de base (tout l'extrait, 0 à 10 kHz) | **Double-clic**, ou bouton « Vue de \
base » |
| Se déplacer dans le spectrogramme | **Clic droit + glisser** |
| Se déplacer dans le temps | **Flèches ← →** du clavier |
| Déplacer les frontières de la bande d'écoute | **Tirer le bord** de la bande (pointillé, \
curseur ↕) |
| Tracer un intervalle | **Clic gauche + glisser** sur le spectrogramme |
| Choisir l'étiquette des prochains intervalles | Touches **1** à **4** du clavier, ou **clic** \
sur une étiquette en haut (aucun intervalle sélectionné) |
| Ajuster un intervalle | **Tirer son bord** (curseur ↔) |
| Sélectionner un intervalle | **Clic** dessus, ou sur sa pastille sous le spectrogramme |
| Modifier l'étiquette d'un intervalle | Une fois sélectionné, **clic** sur une étiquette en haut |
| Supprimer un intervalle | Une fois sélectionné, touche **Suppr**, ou **clic** sur la petite \
croix de sa pastille |
"""

INDEX = r"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="__FONTS__">
<style>
  /* Couleurs et polices du tableau de bord (style.py) ; le cadre est posé par la page.
     Thème clair ou sombre : celui que Streamlit passe au composant (data-theme), avant lui
     celui du système. Le spectrogramme (la toile) reste sombre dans les deux. */
  :root { color-scheme: dark; --ink: #E9F2EC; --muted: #93A89D; --accent: #F5A53A;
          --accent-2: #E4573F; --accent-ink: #1A0B03; --gold: #FFE08A;
          --line: rgba(160, 210, 185, .14); --line-fort: rgba(160, 210, 185, .28);
          --surface-2: rgba(255, 255, 255, .055); --surface-3: rgba(255, 255, 255, .14);
          --fond: #08120F; --champ: #0F1C18; --sel: #FFFFFF; --anneau: rgba(255, 210, 110, .12);
          --mono: "JetBrains Mono", ui-monospace, Menlo, monospace; }
  @media (prefers-color-scheme: light) {
    html:not([data-theme="dark"]) {
      color-scheme: light; --ink: #11201A; --muted: #55675E; --accent: #B9530D;
        --accent-2: #C2352A; --accent-ink: #FFFFFF; --gold: #A86F00;
        --line: rgba(20, 70, 50, .14); --line-fort: rgba(20, 70, 50, .26);
        --surface-2: rgba(20, 60, 45, .06); --surface-3: rgba(20, 60, 45, .12); --fond: #E3ECE6;
        --champ: #FFFFFF; --sel: #11201A; --anneau: rgba(215, 150, 20, .16); }
  }
  html[data-theme="light"] {
    color-scheme: light; --ink: #11201A; --muted: #55675E; --accent: #B9530D;
    --accent-2: #C2352A; --accent-ink: #FFFFFF; --gold: #A86F00;
    --line: rgba(20, 70, 50, .14); --line-fort: rgba(20, 70, 50, .26);
    --surface-2: rgba(20, 60, 45, .06); --surface-3: rgba(20, 60, 45, .12); --fond: #E3ECE6;
    --champ: #FFFFFF; --sel: #11201A; --anneau: rgba(215, 150, 20, .16); }
  body { margin: 0; padding: 10px 12px 12px; color: var(--ink); background: var(--fond);
         font: 12.5px "Hanken Grotesk", "Segoe UI", system-ui, sans-serif;
         -webkit-font-smoothing: antialiased; }
  .bar { display: flex; gap: 6px 9px; align-items: center; flex-wrap: wrap; padding: 2px 0 7px; }
  .duo { display: flex; gap: 9px; align-items: center; }  /* filtre et contraste, inséparables */
  button, select, input { background: var(--champ); color: var(--ink); border-radius: 8px;
                          border: 1px solid var(--line-fort); padding: 3px 9px; font: inherit;
                          transition: color .2s, border-color .2s, box-shadow .2s; }
  button { cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--accent);
                                box-shadow: 0 0 14px -4px var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  input:hover { border-color: var(--accent); }
  input:focus { outline: none; border-color: var(--accent);
                box-shadow: 0 0 0 3px var(--anneau); }
  input[type=number] { width: 4.4em; font-family: var(--mono); font-size: 11.5px; }
  input[type=range] { width: 80px; vertical-align: middle; padding: 0; border: 0; background: none;
                      accent-color: var(--accent); box-shadow: none; }
  input[type=checkbox] { accent-color: var(--accent); vertical-align: -2px; }
  label { color: var(--muted); }
  .group { display: flex; gap: 5px; align-items: center; padding-right: 9px;
           border-right: 1px solid var(--line); }
  .lbl { font-family: var(--mono); font-size: 10px; letter-spacing: .1em; color: var(--muted);
         text-transform: uppercase; margin-right: 2px; }
  .val { font-family: var(--mono); font-size: 11px; color: var(--ink); min-width: 3.2em; }
  #contrastv { min-width: 1.2em; text-align: center; }
  canvas { display: block; width: 100%; cursor: crosshair; outline: none; border-radius: 10px;
           user-select: none; -webkit-user-select: none;
           box-shadow: 0 0 0 1px rgba(255, 210, 120, .14); transition: box-shadow .2s; }
  canvas:focus { box-shadow: 0 0 0 1px rgba(245, 165, 58, .7), 0 0 22px -8px var(--accent); }
  #status { color: var(--muted); margin-left: auto; font-family: var(--mono); font-size: 11px; }
  #players { display: flex; gap: 12px; margin-top: 8px; }
  #players div { flex: 1; display: grid; gap: 3px; }
  #players audio { width: 100%; height: 32px; }
  #players span { font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em;
                  color: var(--muted); display: inline-flex; align-items: center; gap: 7px; }
  #players span::before { content: ""; width: 7px; height: 7px; border-radius: 50%;
                          background: currentColor; opacity: .5; }
  #players .playing span { color: var(--gold); }
  #players .playing span::before { opacity: 1; box-shadow: 0 0 8px currentColor; }
  #list { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; min-height: 4px; }
  .chip { border: 1px solid var(--line-fort); border-radius: 999px; padding: 2px 3px 2px 10px;
          font-size: 12px;
          display: flex; gap: 4px; align-items: center; cursor: pointer;
          transition: box-shadow .2s; }
  .chip:hover { box-shadow: 0 0 14px -5px currentColor; }
  .chip.sel { border-color: var(--sel); box-shadow: 0 0 0 1px var(--sel) inset; }
  .chip button { padding: 0 6px; border: 0; background: transparent; color: var(--muted);
                 border-radius: 999px; }
  .chip button:hover:not(:disabled) { color: #F07A63; box-shadow: none; }
  .chip .q { font-weight: 700; }
  /* Intervalles des autres annotateurs : en lecture seule, cerclés de tirets. */
  #others { display: grid; gap: 5px; margin-top: 6px; }
  #others div { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  #others .who { font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em;
                 color: var(--muted); text-transform: uppercase; }
  .chip.ro { border-style: dashed; padding-right: 10px; }
  .pills { display: flex; gap: 5px; flex-wrap: wrap; }
  /* Étiquettes : pleine = celle des prochains intervalles, cerclée (de blanc en sombre, de
     noir en clair) = celle de l'intervalle sélectionné. */
  .types button { border-width: 1.5px; border-radius: 999px; padding: 3px 11px; }
  .types button.on { color: #0B0B0B; font-weight: 600; }
  .types button.sel { outline: 2px solid var(--sel); outline-offset: 2px; }
  .seg { display: inline-flex; padding: 2px; gap: 2px; background: var(--surface-2);
         border: 1px solid var(--line); border-radius: 10px; }
  .seg button { border: 0; background: transparent; color: var(--muted); padding: 3px 10px;
                border-radius: 7px; }
  .seg button:hover:not(:disabled) { color: var(--ink); box-shadow: none; }
  .seg button.on { background: linear-gradient(135deg, var(--accent), var(--accent-2));
                   color: var(--accent-ink); font-weight: 600;
                   box-shadow: 0 4px 16px -6px var(--accent-2); }
  /* Qualité en médailles : A or, B argent, C bronze ; « — » (non dite) reste neutre. */
  [data-q="A"] { --m: #F2C14E; --m1: #FFEDB0; --m2: #C8901A; }
  [data-q="B"] { --m: #C9D2DA; --m1: #FAFCFD; --m2: #8E9AA5; }
  [data-q="C"] { --m: #D9925A; --m1: #F5C59B; --m2: #9A5626; }
  /* En clair, les médailles foncent pour rester lisibles sur le fond. */
  @media (prefers-color-scheme: light) {
    html:not([data-theme="dark"]) [data-q="A"] { --m: #A87A00; }
    html:not([data-theme="dark"]) [data-q="B"] { --m: #6B7882; }
    html:not([data-theme="dark"]) [data-q="C"] { --m: #9A5626; }
  }
  html[data-theme="light"] [data-q="A"] { --m: #A87A00; }
  html[data-theme="light"] [data-q="B"] { --m: #6B7882; }
  html[data-theme="light"] [data-q="C"] { --m: #9A5626; }
  .medailles button[data-q], .chip .q { color: var(--m); font-weight: 700; }
  .medailles button[data-q]:hover:not(:disabled) { color: var(--m1); }
  .medailles button[data-q].on { background: linear-gradient(135deg, var(--m1), var(--m2));
                                 color: #1A1206; box-shadow: 0 0 14px -3px var(--m2); }
  .medailles button:not([data-q]).on { background: var(--surface-3); color: var(--ink);
                                       box-shadow: none; }
</style></head><body>
<div class="bar">
  <span class="group types" id="ivbar"
        title="Pleine : étiquette des prochains intervalles (touches 1 à 4).
Cerclée : celle de l'intervalle sélectionné (un clic la change).">
    <span class="lbl">Étiquette</span><span class="pills" id="ivbtns"></span>
    <button id="addlbl" title="Ajouter une étiquette (nom et couleur)">+</button>
    <span id="newlbl" hidden>
      <input id="newname" type="text" placeholder="Nouvelle étiquette" size="14">
      <input id="newcolor" type="color" value="#B08CF2">
      <button id="newok">Ajouter</button><button id="newno" title="Annuler">✕</button>
    </span></span>
  <span class="group" id="qbar"
        title="Qualité du chant : de l'intervalle sélectionné, et des prochains">
    <span class="lbl">Qualité</span><span class="seg medailles" id="qbtns"></span></span>
  <span class="group">
    <button id="home" title="Tout l'extrait, 0 à 10 kHz (double-clic)">Vue de base</button>
    <label><input type="checkbox" id="follow"> Suivre</label>
  </span>
  <span id="status"></span>
</div>
<div class="bar">
  <span class="group"><span class="lbl">Bande</span>
    <input type="number" id="lo" min="0" max="24" step="0.1"> –
    <input type="number" id="hi" min="0" max="24" step="0.1"> kHz
    <label><input type="checkbox" id="bandonly"> n'écouter qu'elle</label>
    <select id="slope" title="Raideur du filtre hors de la bande : plus elle est forte, plus les
 sons hors bande sont coupés, mais plus le filtre « sonne » sur les claquements">
      <option value="12">12 dB/oct</option><option value="24">24 dB/oct</option>
      <option value="48">48 dB/oct</option><option value="96">96 dB/oct</option>
      <option value="fft" title="Coupure franche : rien hors de la bande, mais pré-écho et
 traînée à la fréquence des frontières sur les claquements">FFT (coupure franche)</option>
    </select></span>
  <span class="group"><span class="lbl">Volume</span>
    <input type="range" id="gain" min="0" max="30" step="1"> <span class="val" id="gainv"></span>
  </span>
  <span class="duo">
    <span class="group" title="Seuls les sons à moins de N dB des plus forts gardent une couleur">
      <span class="lbl">Filtre dynamique</span>
      <input type="range" id="range" min="30" max="90" step="10">
      <span class="val" id="rangev"></span>
    </span>
    <span class="group" title="Assombrit les sons moyens (bruit de fond), garde les plus forts">
      <span class="lbl">Contraste</span> <button id="cless">－</button>
      <span class="val" id="contrastv"></span> <button id="cmore">＋</button></span>
  </span>
</div>
<div class="bar">
  <span class="group"><span class="lbl">Spectrogramme</span><span class="seg" id="chan"></span>
  </span>
</div>
<canvas id="c" tabindex="0"></canvas>
<div id="list"></div>
<div id="others"></div>
<div id="players"></div>
<script>
const MAGMA = __MAGMA__;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const M = {l: 44, r: 8, t: 8, b: 24};
// Couleurs du tableau de bord : tête de lecture crème à lueur ambrée, bande d'écoute en
// pointillés or ; étiquettes des intervalles : `style.LABEL_COLORS`.
const PLAYHEAD = "#FFF6DC", GLOW = "rgba(245, 165, 58, 0.9)", BAND = "rgba(255, 224, 138, 0.85)";
const BG = "#08120F", AXIS = "#93A89D", TICK = "rgba(160, 210, 185, 0.35)";
const MONO = '10.5px "JetBrains Mono", ui-monospace, Menlo, monospace';
const COLORS = __COLORS__;
const MEDALS = {A: "or : chant net", B: "argent : chant chevauché (pluie, bruit, autre espèce)",
                C: "bronze : chant lointain"};
const $ = id => document.getElementById(id);
let D = null, players = [], graphs = [], buffers = [], active = 0, actx = null;
let raw = [], decoding = [];  // fichiers chargés, décodages lancés (`decode`)
let view = null, intervals = [], selected = -1, vmax = [], loading = 0, newLabel = null;
let newQuality = null;  // qualité des prochains intervalles (null : non dite)
let spec = document.createElement("canvas"), specKey = "", specView = null;
const specCache = new Map();  // clé → toile : dernières vues calculées (et celle préparée)
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
  $("bandonly").checked = !!S.bandOnly; $("slope").value = String(S.slope);
  $("gain").value = S.gain;
  $("gainv").textContent = `+${S.gain} dB`; $("range").value = String(S.range);
  $("rangev").textContent = `${S.range} dB`; $("contrastv").textContent = String(S.contrast);
  $("cless").disabled = S.contrast <= 0; $("cmore").disabled = S.contrast >= 8;
  $("chan").querySelectorAll("button")
    .forEach(b => b.classList.toggle("on", +b.value === S.channel));
}
// Passe-haut et passe-bas de Butterworth d'ordre 2n, chacun fait de n biquads d'ordre 2 dont
// les Q sont ceux des paires de pôles : −3 dB pile sur la frontière, puis 12 n dB par octave.
// (Mettre des biquads à Q = 0,707 en cascade amollirait le coude.) Web Audio lit le Q d'un
// passe-haut ou d'un passe-bas en dB (α = sin ω0 / (2·10^(Q/20))) : on lui passe 20 log10 Q,
// sinon chaque étage bombe à la coupure (+1,9 dB à 24 dB/oct, +17 dB à 96 dB/oct). En mode
// FFT, ils laissent tout passer : c'est le son lu qui est déjà filtré (`applyFFT`).
const stages = () => S.slope === "fft" ? 1 : S.slope / 12;
function butterQ(n) {
  return Array.from({length: n}, (_, k) => 1 / (2 * Math.cos((2 * k + 1) * Math.PI / (4 * n))));
}
function wire(g) {
  g.source.disconnect(); [...g.hp, ...g.lp].forEach(f => f.disconnect());
  const qs = butterQ(stages()).map(q => 20 * Math.log10(q));  // Q en dB pour Web Audio
  g.hp = qs.map(Q => new BiquadFilterNode(actx, {type: "highpass", Q}));
  g.lp = qs.map(Q => new BiquadFilterNode(actx, {type: "lowpass", Q}));
  [g.source, ...g.hp, ...g.lp, g.gain].reduce((a, b) => (a.connect(b), b));
}
function applyAudio() {
  graphs.forEach(g => {
    if (g.hp.length !== stages()) wire(g);
    const nyq = actx.sampleRate / 2, on = S.bandOnly && S.slope !== "fft";
    const lo = on ? Math.max(S.band[0], 10) : 10;
    const hi = on ? Math.min(S.band[1], nyq * 0.99) : nyq * 0.99;
    g.hp.forEach(f => f.frequency.value = lo);
    g.lp.forEach(f => f.frequency.value = hi);
    g.gain.gain.value = Math.pow(10, S.gain / 20);
  });
  clearTimeout(fftTimer);  // pas à chaque pas d'un glisser
  fftTimer = setTimeout(() => { applyFFT(); prepareNext(); }, 250);
}

// --- Mode FFT : tout l'extrait passe en fréquences, ce qui sort de la bande est mis à zéro, on
// revient au signal et le lecteur bascule sur ce son, au même instant. Seule la piste écoutée
// (`active`) est filtrée ; une autre l'est quand on la lance.
let fftTimer = null;
function fftInPlace(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2, a0 = (inverse ? 2 : -2) * Math.PI / size;
    for (let j = 0; j < half; j++) {
      const c = Math.cos(a0 * j), sn = Math.sin(a0 * j);
      for (let a = j; a < n; a += size) {
        const b = a + half;
        const tr = c * re[b] - sn * im[b], ti = c * im[b] + sn * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}
function brickwall(x, rate, lo, hi) {
  const len = x.length;
  let n = 1; while (n < len + rate) n *= 2;  // au moins 1 s de zéros : pas de repliement
  const re = new Float64Array(n), im = new Float64Array(n);
  re.set(x);
  fftInPlace(re, im, false);
  for (let k = 0; k < n; k++) {
    const f = Math.min(k, n - k) * rate / n;
    if (f < lo || f > hi) { re[k] = 0; im[k] = 0; }
  }
  fftInPlace(re, im, true);
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) out[i] = re[i] / n;
  return out;
}
function wavURL(x, rate) {  // WAV 16 bits mono
  const v = new DataView(new ArrayBuffer(44 + 2 * x.length));
  const txt = (o, t) => [...t].forEach((ch, k) => v.setUint8(o + k, ch.charCodeAt(0)));
  txt(0, "RIFF"); v.setUint32(4, 36 + 2 * x.length, true); txt(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, 2 * rate, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); txt(36, "data"); v.setUint32(40, 2 * x.length, true);
  for (let i = 0; i < x.length; i++) {
    v.setInt16(44 + 2 * i, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  }
  return URL.createObjectURL(new Blob([v.buffer], {type: "audio/wav"}));
}
// Le dernier son filtré de chaque lecteur reste en mémoire (`el._fft`) : décocher puis recocher
// « n'écouter qu'elle » le remet aussitôt, sans refaire la FFT ; seule une nouvelle bande la
// relance (et libère l'ancien son).
function swapSource(el, url, key) {
  const t = el.currentTime, playing = !el.paused;
  if (key && el._fft?.key !== key) {
    if (el._fft) URL.revokeObjectURL(el._fft.url);
    el._fft = {key, url};
  }
  el._key = key; el.src = url;
  el.addEventListener("loadedmetadata", () => {
    el.currentTime = t; if (playing) el.play();
  }, {once: true});
}
// Bande filtrée d'avance (mode FFT), case « n'écouter qu'elle » cochée ou non : la cocher
// bascule aussitôt sur un son déjà calculé. `fftKey` : le son à faire jouer.
const fftWant = () => S.slope === "fft" ? S.band.join("-") : "";
const fftKey = () => S.bandOnly ? fftWant() : "";
const FILTERING = "filtrage FFT…";
function filteringDone() {
  if ($("status").textContent === FILTERING) $("status").textContent = "";
}
function applyFFT() {
  const key = fftKey(), want = fftWant(), i = active, el = players[i];
  if (!el) return;
  if (el._key !== key) {
    if (!key) swapSource(el, el._orig, "");
    else if (el._fft?.key === key) swapSource(el, el._fft.url, key);
  }
  if (!want || el._key === want || el._fft?.key === want) return;
  if (key) $("status").textContent = FILTERING;  // on l'attend pour écouter
  if (el._busy === want) return;
  if (!buffers[i]) { decode(i)?.then(applyFFT); return; }
  el._busy = want;
  const token = loading, rate = buffers.rate;
  filtered(buffers[i], rate, Math.max(S.band[0], 1), S.band[1]).then(out => {
    if (el._busy === want) el._busy = null;
    if (token !== loading) return;
    // Case décochée, autre bande, autre lecteur : le son filtré est gardé pour plus tard, mais
    // le lecteur n'y bascule pas.
    if (fftKey() !== want || active !== i) {
      if (!el._fft || el._fft.key !== el._key) {  // sans libérer le son en cours de lecture
        if (el._fft) URL.revokeObjectURL(el._fft.url);
        el._fft = {key: want, url: wavURL(out, rate)};
      }
      if (!el._busy) filteringDone();
      return;
    }
    swapSource(el, wavURL(out, rate), want);
    filteringDone();
  });
}
// La FFT tourne dans un fil à part (Web Worker) : l'interface ne gèle pas pendant le filtrage,
// ni pendant celui de l'extrait suivant, préparé en cours d'annotation. Sans Worker (refusé
// par le navigateur), elle se fait sur place.
let worker, jobId = 0;
const jobs = new Map();
function filtered(x, rate, lo, hi) {
  if (worker === undefined) {
    try {
      const code = `${fftInPlace}\n${brickwall}\nonmessage = e => {
        const {id, x, rate, lo, hi} = e.data, y = brickwall(x, rate, lo, hi);
        postMessage({id, y}, [y.buffer]);
      };`;
      worker = new Worker(URL.createObjectURL(new Blob([code], {type: "text/javascript"})));
      worker.onmessage = e => { jobs.get(e.data.id).ok(e.data.y); jobs.delete(e.data.id); };
      worker.onerror = () => {
        worker = null;
        jobs.forEach(({ok, args}) => ok(brickwall(...args))); jobs.clear();
      };
    } catch (e) { worker = null; }
  }
  if (!worker) return new Promise(ok => setTimeout(() => ok(brickwall(x, rate, lo, hi)), 30));
  const id = ++jobId, copy = x.slice();
  return new Promise(ok => {
    jobs.set(id, {ok, args: [x, rate, lo, hi]});
    worker.postMessage({id, x: copy, rate, lo, hi}, [copy.buffer]);
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
  S.bandOnly = $("bandonly").checked; applyAudio(); applyFFT(); saveSettings();
};
$("slope").onchange = () => {
  const v = $("slope").value; S.slope = v === "fft" ? v : +v; applyAudio(); saveSettings();
};
$("gain").oninput = () => {
  S.gain = +$("gain").value; showSettings(); applyAudio(); saveSettings();
};
$("range").oninput = () => {
  S.range = +$("range").value; showSettings(); saveSettings(); draw(); prepSoon();
};
function setContrast(c) {
  S.contrast = Math.max(0, Math.min(8, c)); showSettings(); saveSettings(); draw(); prepSoon();
}
$("cless").onclick = () => setContrast(S.contrast - 1);
$("cmore").onclick = () => setContrast(S.contrast + 1);
function setChannel(c) {
  S.channel = c; showSettings(); saveSettings(); draw(); decode(c); prepSoon();
}

// --- Étiquette des intervalles : un bouton par étiquette, toujours visibles ---------------
// Deux choses distinctes : l'étiquette des prochains intervalles (touches 1 à 4, ou un clic
// quand rien n'est sélectionné) et celle de l'intervalle sélectionné (un clic la change, sans
// toucher à celle des prochains).
const rgb = label => COLORS[label] || [200, 200, 200];
const labelName = code => (D.labels.find(([c]) => c === code) || [code, code])[1];
function showTypes() {
  const current = selected >= 0 ? intervals[selected].label : null;
  [...$("ivbtns").querySelectorAll("button")].forEach(b => {
    const on = b.value === newLabel;
    b.classList.toggle("on", on);
    b.classList.toggle("sel", b.value === current);
    b.style.background = on ? `rgb(${rgb(b.value)})` : `rgba(${rgb(b.value)}, 0.15)`;
    b.style.boxShadow = on ? `0 0 16px -4px rgb(${rgb(b.value)})` : "";
  });
}
function setNext(code) { newLabel = code; showTypes(); }  // prochains intervalles seulement
function setType(code) {  // clic : l'intervalle sélectionné s'il y en a un, sinon les prochains
  if (selected < 0) setNext(code);
  else if (intervals[selected].label !== code) { intervals[selected].label = code; changed(); }
}
// Qualité : un intervalle a la sienne (un même extrait mêle chants nets et lointains).
function showQualities() {
  const current = selected >= 0 ? intervals[selected].quality : newQuality;
  [...$("qbar").querySelectorAll("button")]
    .forEach(b => b.classList.toggle("on", (b.value || null) === current));
}
function setQuality(q) {  // qualité des prochains intervalles, et de l'intervalle sélectionné
  newQuality = q;
  if (selected >= 0 && intervals[selected].quality !== q) {
    intervals[selected].quality = q; changed();
  } else showQualities();
}

// --- Échanges avec Streamlit ----------------------------------------------------------------
function send(type, data) {
  window.parent.postMessage(Object.assign({isStreamlitMessage: true, type}, data), "*");
}
function setHeight() { send("streamlit:setFrameHeight", {height: document.body.scrollHeight}); }
const round = t => Math.round(t * 100) / 100;
function emit(addLabel) {
  send("streamlit:setComponentValue", {dataType: "json", value: {
    key: D.extract, ivkey: D.ivkey, channel: active,
    intervals: intervals.map(i => [round(i.t0), round(i.t1), i.label, i.quality]),
    add_label: addLabel || null,
  }});
}
// Nouvelle étiquette : nom et couleur saisis ici, ajoutées à la liste par le poste (`nonce`
// évite de la rajouter à chaque réaffichage).
function showNewLabel(on) {
  $("newlbl").hidden = !on; $("addlbl").hidden = on;
  if (on) $("newname").focus(); else $("newname").value = "";
  setHeight();
}
$("addlbl").onclick = () => showNewLabel(true);
$("newno").onclick = () => showNewLabel(false);
$("newok").onclick = () => {
  const name = $("newname").value.trim();
  if (!name) { $("newname").focus(); return; }
  emit({name, color: $("newcolor").value, nonce: Date.now()});
  showNewLabel(false);
};
$("newname").addEventListener("keydown", e => {
  if (e.key === "Enter") $("newok").click();
  else if (e.key === "Escape") $("newno").click();
});

function render(args) {
  const fresh = !D || args.extract !== D.extract;
  const nextChanged = !D || JSON.stringify(args.next) !== JSON.stringify(D.next);
  // Autre annotateur, ou autres masqués : mêmes son et vue, autres intervalles.
  const ivChanged = !D || args.ivkey !== D.ivkey;
  const othersChanged = !D || JSON.stringify(args.others) !== JSON.stringify(D.others);
  D = args;
  if (S.band === undefined) {
    S = {band: D.band, bandOnly: false, gain: 0, range: 60, channel: D.channel};
  }
  if (S.contrast === undefined) S.contrast = 0;
  if (![12, 24, 48, 96, "fft"].includes(S.slope)) S.slope = 24;
  $("ivbar").style.display = D.interval_mode ? "" : "none";
  $("qbar").style.display = D.interval_mode ? "" : "none";
  if (newLabel === null) {
    ["—", ...D.qualities].forEach((q, k) => {
      const b = document.createElement("button");
      b.value = k ? q : ""; b.textContent = q; b.onclick = () => setQuality(k ? q : null);
      if (k) { b.dataset.q = q; b.title = MEDALS[q] ? `${q}, ${MEDALS[q]}` : q; }
      $("qbtns").appendChild(b);
    });
    newLabel = D.labels[0][0];
    D.audios.forEach((a, i) => {
      const b = document.createElement("button");
      b.value = String(i); b.textContent = a.name; b.onclick = () => setChannel(i);
      $("chan").appendChild(b);
    });
  }
  // Étiquettes : reconstruites quand la liste ou les couleurs changent (étiquette ajoutée).
  Object.assign(COLORS, D.colors || {});
  const labelSig = JSON.stringify([D.labels, D.colors]);
  if (labelSig !== $("ivbtns").dataset.sig) {
    $("ivbtns").dataset.sig = labelSig;
    $("ivbtns").innerHTML = "";
    D.labels.forEach(([code, name], k) => {
      const b = document.createElement("button");
      b.value = code; b.textContent = name; b.title = k < 9 ? `Touche ${k + 1}` : name;
      b.style.borderColor = `rgb(${rgb(code)})`;
      b.onclick = () => setType(code);
      $("ivbtns").appendChild(b);
    });
    if (!D.labels.some(([c]) => c === newLabel)) newLabel = D.labels[0][0];
    showTypes();
  }
  showSettings();
  if (fresh) {
    view = {t0: D.t0, t1: D.t1, f0: 0, f1: D.fview};
    intervals = (D.intervals || [])
      .map(([t0, t1, l, q]) => ({t0, t1, label: l, quality: q || null}));
    selected = -1; vmax = []; specKey = ""; specView = null; settling = false; buffers = [];
    if (wide) clearTimeout(wide.timer);
    wide = null;
    load();
    list();
  } else {
    if (ivChanged) {
      intervals = (D.intervals || [])
        .map(([t0, t1, l, q]) => ({t0, t1, label: l, quality: q || null}));
      selected = -1;
    }
    if (ivChanged || othersChanged) list();
    if (nextChanged) prepareNext();
  }
  resize(); setHeight();
}

// --- Audio : chargé en mémoire (lecture n'importe où), décodé pour le spectrogramme --------
async function load() {
  const token = ++loading;
  players.forEach(p => {
    p.pause(); URL.revokeObjectURL(p._orig); if (p._fft) URL.revokeObjectURL(p._fft.url);
  });
  const box = $("players");
  box.innerHTML = ""; players = []; graphs = []; active = 0; raw = []; decoding = [];
  prepToken++;  // la préparation en cours visait cet extrait ou un autre : on reprend ce qu'elle a
  given = {};
  for (const [src, p] of Object.entries(ahead)) {
    if (D.audios.some(a => a.src === src)) given[src] = p;
    else if (p.fft) URL.revokeObjectURL(p.fft.url);
  }
  ahead = {};
  $("status").textContent = "chargement de l'audio…";
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  const data = await Promise.all(D.audios.map(a =>
    given[a.src]?.raw || fetch(a.src).then(r => r.arrayBuffer())));
  if (token !== loading) return;
  raw = data;
  D.audios.forEach((a, i) => {
    const div = document.createElement("div");
    div.innerHTML = `<span>${a.name}</span>`;
    const el = document.createElement("audio");
    el.controls = true; el.preload = "auto";
    // `a.type` : FLAC dans le paquet d'écoute en page web (`annotation/web/poste.html`).
    el.src = el._orig = URL.createObjectURL(new Blob([data[i]], {type: a.type || "audio/wav"}));
    el._key = ""; el._fft = given[a.src]?.fft;
    const source = actx.createMediaElementSource(el);
    const g = {source, hp: [], lp: [], gain: new GainNode(actx)};
    g.gain.connect(actx.destination); wire(g); graphs.push(g);
    el.addEventListener("play", () => {
      if (actx.state !== "running") actx.resume();
      players.forEach((p, j) => { if (j !== i) p.pause(); });
      if (active !== i) { active = i; emit(); applyFFT(); }
      mark(); loop();
    });
    el.addEventListener("seeked", draw);
    div.appendChild(el); box.appendChild(div); players.push(el);
  });
  applyAudio(); mark(); setHeight();
  await decode(shownTrack());
  if (token !== loading) return;
  $("status").textContent = "";
  applyFFT();
  prepareNext();
}
// Décodage et niveau de référence d'une piste, à la demande : au chargement, seulement la
// piste affichée ; une autre quand on l'affiche, ou quand on la lance en mode FFT.
const shownTrack = () => S.channel < D.audios.length ? S.channel : 0;
function decode(i) {
  if (!decoding[i] && raw[i]) {
    const token = loading, p = given[D.audios[i].src];
    decoding[i] = (p?.x ? Promise.resolve(p) : decodeTrack(raw[i])).then(r => {
      if (token !== loading) return;
      buffers[i] = r.x; buffers.rate = r.rate; vmax[i] = r.vmax;
      draw();
    });
  }
  return decoding[i];
}
async function decodeTrack(data) {
  const buf = await actx.decodeAudioData(data.slice(0));
  const x = buf.getChannelData(0), rate = buf.sampleRate;
  return {x, rate, vmax: reference(x, rate)};
}

// --- Extrait suivant, préparé pendant qu'on annote celui-ci ---------------------------------
// Python donne les fichiers du suivant (`D.next`) avant de les avoir écrits : on les guette,
// puis on les charge, on décode la piste affichée, on calcule son niveau de référence et son
// spectrogramme à la vue d'ouverture et, en mode FFT, le son filtré du premier lecteur.
// `load`, `decode` et `renderSpectrogram` reprennent tout cela à l'ouverture.
let ahead = {}, given = {}, prepToken = 0, prepTimer = null;
const idle = () => new Promise(ok => (window.requestIdleCallback || setTimeout)(() => ok()));
function prepSoon() { clearTimeout(prepTimer); prepTimer = setTimeout(prepareNext, 400); }
async function waitFile(src, token) {
  for (let k = 0; k < 300 && token === prepToken; k++) {  // 5 min au plus
    const r = await fetch(src, {cache: "no-store"}).catch(() => null);
    if (r?.ok) return r.arrayBuffer();
    await new Promise(ok => setTimeout(ok, 1000));
  }
  return null;
}
async function prepareNext() {
  const token = ++prepToken, next = D.next || [];
  for (const [src, p] of Object.entries(ahead)) {
    if (next.some(a => a.src === src)) continue;
    if (p.fft) URL.revokeObjectURL(p.fft.url);
    delete ahead[src];
  }
  if (!next.length || !actx) return;
  const raws = await Promise.all(next.map(a => ahead[a.src]?.raw || waitFile(a.src, token)));
  if (token !== prepToken || raws.some(r => !r)) return;
  next.forEach((a, i) => { ahead[a.src] = ahead[a.src] || {raw: raws[i]}; });
  const track = async i => {
    const p = ahead[next[i].src];
    if (!p.x) { await idle(); Object.assign(p, await decodeTrack(p.raw)); }
    return p;
  };
  const ch = S.channel < next.length ? S.channel : 0, a = next[ch], p = await track(ch);
  await idle();
  if (token !== prepToken) return;
  const v = {t0: a.start, t1: a.start + p.x.length / p.rate, f0: 0, f1: D.fview};
  const cols = Math.max(1, Math.round(W())), rows = Math.max(1, Math.round(PH()));
  const n = fftSize(v, p.rate), key = specKeyOf(a.src, v, cols, rows, n);
  if (!specCache.has(key)) remember(key, paint(p.x, p.rate, a.start, p.vmax, v, cols, rows, n));
  const fk = fftWant();
  if (!fk) return;
  const q = await track(0);
  if (token !== prepToken || q.fft?.key === fk) return;
  const y = await filtered(q.x, q.rate, Math.max(S.band[0], 1), S.band[1]);
  if (token !== prepToken || !ahead[next[0].src]) return;
  if (q.fft) URL.revokeObjectURL(q.fft.url);
  q.fft = {key: fk, url: wavURL(y, q.rate)};
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
function reference(x, rate) {
  const n = 1024, cols = 600, binHz = rate / n;
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
function fftSize(v, rate) {
  const tspan = v.t1 - v.t0, fspan = v.f1 - v.f0;
  if (fspan <= 1200) return rate > 30000 ? 8192 : 4096;
  if (fspan <= 3000) return rate > 30000 ? 4096 : 2048;
  if (tspan <= 2) return rate > 30000 ? 1024 : 512;
  return rate > 30000 ? 2048 : 1024;
}
// Clé d'une image : la piste (son fichier), la vue (au centième, pour retrouver celle préparée
// d'avance malgré les arrondis du décodage), la taille et les réglages.
function specKeyOf(src, v, cols, rows, n) {
  const t = [v.t0, v.t1, v.f0, v.f1].map(z => z.toFixed(2));
  return [src, ...t, cols, rows, n, S.range, S.contrast].join();
}
function remember(key, canvas) {
  specCache.delete(key); specCache.set(key, canvas);
  while (specCache.size > 6) specCache.delete(specCache.keys().next().value);
}
function renderSpectrogram(cols, rows) {
  const ch = shownTrack(), a = D.audios[ch], n = fftSize(view, buffers.rate);
  const key = specKeyOf(a.src, view, cols, rows, n);
  if (key === specKey) return;
  specKey = key; specView = {...view};
  spec = specCache.get(key)
    || paint(buffers[ch], buffers.rate, a.start, vmax[ch], view, cols, rows, n);
  remember(key, spec);
}
// Zone élargie (une demi-vue de plus de chaque côté), calculée en tâche de fond, par petits
// morceaux et du centre vers les bords, dès que la vue est au repos : un déplacement n'a plus
// rien à calculer, il montre ce qui est prêt (noir au-delà si l'on va plus vite que le calcul).
// Demi-résolution et FFT réduite : le rendu net revient au relâchement.
let wide = null;
function startWide() {
  const ch = shownTrack(), a = D.audios[ch];
  const v = {t0: Math.max(D.t0, view.t0 - (view.t1 - view.t0) / 2),
             t1: Math.min(D.t1, view.t1 + (view.t1 - view.t0) / 2),
             f0: Math.max(0, view.f0 - (view.f1 - view.f0) / 2),
             f1: Math.min(fmax(), view.f1 + (view.f1 - view.f0) / 2)};
  const cols = Math.max(1, Math.round(W() * (v.t1 - v.t0) / (view.t1 - view.t0) / 2));
  const rows = Math.max(1, Math.round(PH() * (v.f1 - v.f0) / (view.f1 - view.f0) / 2));
  const n = Math.min(fftSize(view, buffers.rate), 4096);
  const key = specKeyOf(a.src, v, cols, rows, n);
  if (wide && wide.key === key) return;
  if (wide) clearTimeout(wide.timer);
  const out = document.createElement("canvas"); out.width = cols; out.height = rows;
  const step = 16, order = [];
  for (let c = 0; c < cols; c += step) order.push(c);
  order.sort((p, q) => Math.abs(p + step / 2 - cols / 2) - Math.abs(q + step / 2 - cols / 2));
  const job = wide = {key, src: a.src, canvas: out, view: v, order, timer: null};
  const x = buffers[ch], rate = buffers.rate, top = vmax[ch];
  (function work() {
    const t = performance.now();
    while (job.order.length && performance.now() - t < 12) {
      const c0 = job.order.shift();
      paintCols(out, x, rate, a.start, top, v, cols, rows, n, c0, Math.min(cols, c0 + step));
    }
    if (drag) draw();
    if (job.order.length) job.timer = setTimeout(work, 0);
  })();
}
function paint(x, rate, start, top, v, cols, rows, n) {
  const out = document.createElement("canvas"); out.width = cols; out.height = rows;
  return paintCols(out, x, rate, start, top, v, cols, rows, n, 0, cols);
}
function paintCols(out, x, rate, start, top, v, cols, rows, n, c0, c1) {
  const sctx = out.getContext("2d"), image = sctx.createImageData(c1 - c0, rows), px = image.data;
  const binHz = rate / n, range = S.range;
  // Contraste : courbe en puissance sur l'échelle (0 = linéaire) ; le haut ne bouge pas, les
  // niveaux moyens (bruit de fond) s'assombrissent.
  const gamma = 1 + 0.35 * S.contrast, lut = new Uint8Array(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.round(255 * Math.pow(i / 255, gamma));
  const binOf = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    binOf[r] = (v.f1 - (r + 0.5) / rows * (v.f1 - v.f0)) / binHz;
  }
  for (let c = c0; c < c1; c++) {
    const t = v.t0 + (c + 0.5) / cols * (v.t1 - v.t0);
    const db = spectrum(x, (t - start) * rate, n);
    for (let r = 0; r < rows; r++) {
      const b = binOf[r], k = Math.floor(b), f = b - k;
      const l = k + 1 < db.length ? db[k] * (1 - f) + db[k + 1] * f : db[db.length - 1];
      const i = Math.max(0, Math.min(255, Math.round((l - (top - range)) / range * 255)));
      const o = (r * (c1 - c0) + c - c0) * 4, color = MAGMA[lut[i]];
      px[o] = color[0]; px[o + 1] = color[1]; px[o + 2] = color[2]; px[o + 3] = 255;
    }
  }
  sctx.putImageData(image, c0, 0);
  return out;
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
  ctx.fillStyle = BG; ctx.fillRect(0, 0, w, H());
  const ready = buffers[shownTrack()] && vmax[shownTrack()] !== undefined;
  if (ready && settling && specView) {
    ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t, W(), PH()); ctx.clip();
    ctx.imageSmoothingEnabled = true;
    const x0 = xOf(specView.t0), y0 = yOf(specView.f1);
    ctx.drawImage(spec, x0, y0, xOf(specView.t1) - x0, yOf(specView.f0) - y0);
    ctx.restore();
  } else if (ready) {
    renderSpectrogram(Math.max(1, Math.round(W())), Math.max(1, Math.round(PH())));
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(spec, M.l, M.t, W(), PH());
    if (!drag) setTimeout(startWide, 0);
  } else {
    ctx.fillStyle = AXIS; ctx.font = MONO;
    ctx.fillText("calcul du spectrogramme…", M.l + 10, M.t + 20);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, W(), PH()); ctx.clip();
  (D.windows || []).forEach(win => {
    const x0 = xOf(win.t0), x1 = xOf(win.t1);
    if (win.current) {
      ctx.fillStyle = "rgba(245, 165, 58, 0.10)"; ctx.fillRect(x0, M.t, x1 - x0, PH());
      ctx.strokeStyle = "#F5A53A"; ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, M.t + 0.5, x1 - x0, PH() - 1);
    } else if (!D.interval_mode) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.35)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, M.t); ctx.lineTo(x0, M.t + PH());
      ctx.moveTo(x1, M.t); ctx.lineTo(x1, M.t + PH()); ctx.stroke();
    }
    if (win.label) {
      ctx.fillStyle = "rgba(95, 211, 148, 0.85)"; ctx.fillRect(x0, M.t, x1 - x0, 14);
      ctx.fillStyle = "#06130D"; ctx.font = '600 11px "Hanken Grotesk", system-ui, sans-serif';
      ctx.fillText(win.label, x0 + 3, M.t + 11, Math.max(x1 - x0 - 6, 1));
    }
    if (win.candidate) {
      ctx.fillStyle = "#FFE08A"; ctx.fillRect(x0, M.t + PH() - 3, x1 - x0, 3);
    }
  });
  // Les autres annotateurs, sous les siens : bords en tirets, une bande nommée chacun en bas.
  (D.others || []).forEach((o, k) => {
    const y = M.t + PH() - 3 - 13 * (k + 1);
    o.intervals.forEach(([t0, t1, l]) => {
      const x0 = xOf(t0), x1 = xOf(t1), c = rgb(l);
      ctx.setLineDash([4, 3]); ctx.strokeStyle = `rgba(${c}, 0.8)`; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(x0, M.t); ctx.lineTo(x0, M.t + PH());
      ctx.moveTo(x1, M.t); ctx.lineTo(x1, M.t + PH()); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = `rgba(${c}, 0.8)`; ctx.fillRect(x0, y, x1 - x0, 11);
      ctx.fillStyle = "#0B0B0B"; ctx.font = '600 9.5px "Hanken Grotesk", system-ui, sans-serif';
      ctx.fillText(o.who, x0 + 3, y + 9, Math.max(x1 - x0 - 6, 1));
    });
  });
  const shown = drag && drag.mode === "draw" && drag.moved
    ? [...intervals, {t0: Math.min(drag.t, drag.t2), t1: Math.max(drag.t, drag.t2),
                      label: newLabel}]
    : intervals;
  shown.forEach((iv, i) => {
    const x0 = xOf(iv.t0), x1 = xOf(iv.t1), c = COLORS[iv.label] || [200, 200, 200];
    ctx.fillStyle = `rgba(${c}, 0.20)`; ctx.fillRect(x0, M.t, x1 - x0, PH());
    ctx.fillStyle = `rgba(${c}, 0.95)`; ctx.fillRect(x0, M.t, x1 - x0, 5);
    ctx.strokeStyle = `rgb(${c})`; ctx.lineWidth = i === selected ? 3 : 1.5;
    ctx.beginPath(); ctx.moveTo(x0, M.t); ctx.lineTo(x0, M.t + PH());
    ctx.moveTo(x1, M.t); ctx.lineTo(x1, M.t + PH()); ctx.stroke();
  });
  ctx.setLineDash([5, 4]); ctx.strokeStyle = BAND; ctx.lineWidth = 1.2;
  S.band.forEach(f => {
    const y = yOf(f); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + W(), y); ctx.stroke();
  });
  ctx.setLineDash([]);
  const t = playheadTime();
  if (t !== null) {
    const x = xOf(t);
    ctx.strokeStyle = "rgba(0, 0, 0, 0.7)"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + PH()); ctx.stroke();
    ctx.strokeStyle = PLAYHEAD; ctx.lineWidth = 2; ctx.shadowColor = GLOW; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + PH()); ctx.stroke();
    ctx.shadowBlur = 0;
  }
  ctx.restore();
  axes();
}
function axes() {
  ctx.fillStyle = AXIS; ctx.strokeStyle = TICK; ctx.lineWidth = 1; ctx.font = MONO;
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
// Pastilles sous le spectrogramme, sans menu déroulant : le type se change en sélectionnant
// l'intervalle puis en cliquant un bouton de type (ou 1 à 4).
function list() {
  const box = $("list");
  box.innerHTML = "";
  intervals.map((iv, i) => [iv, i]).sort((a, b) => a[0].t0 - b[0].t0).forEach(([iv, i]) => {
    const chip = document.createElement("span");
    chip.className = "chip" + (i === selected ? " sel" : "");
    chip.style.background = `rgba(${rgb(iv.label)}, 0.18)`;
    chip.style.borderColor = i === selected ? "var(--sel)" : `rgb(${rgb(iv.label)})`;
    chip.title = "Sélectionner, puis cliquer une étiquette en haut pour la changer";
    chip.append(`${iv.t0.toFixed(1)}–${iv.t1.toFixed(1)} s · ${labelName(iv.label)}`);
    if (iv.quality) {
      const q = document.createElement("b");
      q.className = "q"; q.dataset.q = iv.quality; q.textContent = iv.quality;
      q.title = MEDALS[iv.quality] ? `Qualité ${iv.quality}, ${MEDALS[iv.quality]}` : "";
      chip.append(" · ", q);
    }
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
    chip.append(x); box.appendChild(chip);
  });
  others();
  showTypes(); showQualities(); setHeight();
}
// Intervalles des autres annotateurs, une ligne chacun : un clic y amène la vue, rien ne
// s'y modifie.
function others() {
  const box = $("others");
  box.innerHTML = "";
  (D.others || []).forEach(o => {
    const row = document.createElement("div"), who = document.createElement("span");
    who.className = "who";
    who.textContent = o.intervals.length ? o.who : `${o.who} : aucun intervalle`;
    row.appendChild(who);
    o.intervals.forEach(([t0, t1, l, q]) => {
      const chip = document.createElement("span");
      chip.className = "chip ro";
      chip.style.background = `rgba(${rgb(l)}, 0.10)`; chip.style.borderColor = `rgb(${rgb(l)})`;
      chip.title = `Tracé par ${o.who} (lecture seule)`;
      chip.append(`${t0.toFixed(1)}–${t1.toFixed(1)} s · ${labelName(l)}`);
      if (q) {
        const b = document.createElement("b");
        b.className = "q"; b.dataset.q = q; b.textContent = q; chip.append(" · ", b);
      }
      chip.onclick = () => {
        const span = view.t1 - view.t0;
        if (t1 < view.t0 || t0 > view.t1) {
          view.t0 = (t0 + t1) / 2 - span / 2; view.t1 = view.t0 + span; clampView();
        }
        draw();
      };
      row.appendChild(chip);
    });
    box.appendChild(row);
  });
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

// Clic gauche : tracer un intervalle, en tirer un bord, ou tirer une limite de la bande.
// Clic droit + glisser : se déplacer. Le menu du clic droit ne s'ouvre pas sur le
// spectrogramme, ni au relâché d'un déplacement (Windows l'ouvre là).
let drag = null, noMenuUntil = 0;
canvas.addEventListener("mousedown", e => {
  if ((e.button !== 0 && e.button !== 2) || !D || drag) return;
  canvas.focus();
  const x = localX(e), y = localY(e), t = tOf(x), line = bandLine(y);
  const right = e.button === 2;
  const [i, edge] = D.interval_mode && !right ? hit(x) : [-1, null];
  let mode = "pan";
  if (!right && line >= 0) mode = "band";
  else if (!right && D.interval_mode) mode = edge ? "edge" : "draw";
  drag = {mode, right, x: e.clientX, y: e.clientY, view: {...view}, moved: false, t, t2: t, i,
          edge, line};
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
  if (drag.mode === "pan" && !drag.wide) {
    drag.wide = true;
    if (wide && wide.src === D.audios[shownTrack()].src) {
      spec = wide.canvas; specView = wide.view; specKey = "large";
    }
  }
  const t = Math.min(Math.max(tOf(localX(e)), D.t0), D.t1);
  if (drag.mode === "pan") {
    canvas.style.cursor = "grabbing";
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
    intervals[drag.i][drag.edge] = t;
  } else {
    drag.t2 = t;
  }
  draw();
});
window.addEventListener("mouseup", e => {
  if (!drag || (e.button === 2) !== drag.right) return;
  const d = drag; drag = null;
  if (d.right) noMenuUntil = performance.now() + 500;
  if (d.mode === "pan" && d.moved) {
    canvas.style.cursor = "crosshair"; clearTimeout(settle); settling = false; draw(); return;
  }
  if (!d.moved) {
    if (d.right) return;  // clic droit sans glisser : rien
    // Sélection par un clic seulement : un clic sur une étiquette change alors cet intervalle.
    if (D.interval_mode) { selected = d.i; list(); }
    const p = players[active];  // la lecture reprendra de cet instant
    if (p && d.t >= D.t0 && d.t <= D.t1) p.currentTime = Math.max(0, d.t - D.audios[active].start);
    draw();
    return;
  }
  if (d.mode === "draw") {
    const t0 = Math.min(d.t, d.t2), t1 = Math.max(d.t, d.t2);
    if (t1 - t0 >= 0.05) {
      intervals.push({t0, t1, label: newLabel, quality: newQuality});
      selected = -1; changed();
    } else draw();
  } else if (d.mode === "edge") {
    const iv = intervals[d.i];
    [iv.t0, iv.t1] = [Math.min(iv.t0, iv.t1), Math.max(iv.t0, iv.t1)];
    if (iv.t1 - iv.t0 < 0.05) remove(d.i); else changed();
  }
});
canvas.addEventListener("contextmenu", e => e.preventDefault());
window.addEventListener("contextmenu", e => {
  if (drag || performance.now() < noMenuUntil) e.preventDefault();
});
canvas.addEventListener("dblclick", () => {
  view = {t0: D.t0, t1: D.t1, f0: 0, f1: Math.min(D.fview, fmax())}; draw();
});
// Les boutons de la barre ne prennent pas la main : barre d'espace, Suppr et 1 à 4 restent
// au spectrogramme.
document.querySelectorAll(".bar").forEach(bar => bar.addEventListener("mousedown", e => {
  if (e.target.tagName === "BUTTON") e.preventDefault();
}));
$("home").onclick = () => canvas.dispatchEvent(new Event("dblclick"));
document.addEventListener("keydown", e => {
  if (["SELECT", "INPUT"].includes(e.target.tagName)) return;
  if (e.code === "Space") {
    e.preventDefault();
    const p = players[active]; if (p) (p.paused ? p.play() : p.pause());
  } else if (D && D.interval_mode && /^[1-9]$/.test(e.key) && D.labels[+e.key - 1]) {
    setNext(D.labels[+e.key - 1][0]);
  } else if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0) {
    remove(selected);
  } else if (view && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
    e.preventDefault();  // un quart de la vue par appui
    const span = view.t1 - view.t0, s = (e.key === "ArrowLeft" ? -0.25 : 0.25) * span;
    view.t0 += s; view.t1 += s; clampView(); moved();
  }
});

// Thème de la page, clair ou sombre : `base` s'il est donné, sinon d'après la couleur du fond.
function theme(t) {
  if (!t) return;
  let base = t.base;
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(t.backgroundColor || "");
  if (!base && m) {
    const [r, g, b] = m.slice(1).map(x => parseInt(x, 16));
    base = 0.299 * r + 0.587 * g + 0.114 * b > 128 ? "light" : "dark";
  }
  if (base === "light" || base === "dark") document.documentElement.dataset.theme = base;
}
window.addEventListener("message", e => {
  if (e.data && e.data.type === "streamlit:render") { theme(e.data.theme); render(e.data.args); }
});
window.addEventListener("resize", () => { if (D) resize(); });
// Polices chargées après le premier dessin : graduations et hauteur de la barre à refaire.
if (document.fonts) {
  document.fonts.addEventListener("loadingdone", () => { if (D) { resize(); setHeight(); } });
}
send("streamlit:componentReady", {apiVersion: 1});
</script></body></html>
"""


def _page() -> str:
    from matplotlib import colormaps

    magma = [[round(255 * c) for c in colormaps["magma"](i / 255)[:3]] for i in range(256)]
    colors = {k: [int(v[i : i + 2], 16) for i in (1, 3, 5)] for k, v in LABEL_COLORS.items()}
    return (
        INDEX.replace("__MAGMA__", json.dumps(magma))
        .replace("__COLORS__", json.dumps(colors))
        .replace("__FONTS__", FONTS)
    )


def _prepare() -> Path:
    """Page du composant, réécrite si elle a changé (mise à jour du code)."""
    (VIEWER_DIR / "assets").mkdir(parents=True, exist_ok=True)
    index = VIEWER_DIR / "index.html"
    page = _page()
    if not index.exists() or index.read_text(encoding="utf-8") != page:
        index.write_text(page, encoding="utf-8")
    return VIEWER_DIR


def asset_src(name_key: str, suffix: str) -> str:
    """Chemin, relatif à la page, du fichier que `asset(…, name_key=name_key)` écrira : connu
    avant d'avoir le contenu (le navigateur peut attendre le fichier de l'extrait suivant)."""
    return f"assets/{hashlib.sha1(name_key.encode()).hexdigest()[:20]}{suffix}"


def asset(data: bytes, suffix: str, name_key: str | None = None) -> str:
    """Range `data` à côté de la page (nom = empreinte du contenu, ou de `name_key`) ; chemin
    relatif à la page. Écriture atomique : le navigateur ne voit jamais un fichier à moitié
    écrit. Seuls les `KEEP_ASSETS` fichiers les plus récents sont gardés."""
    folder = VIEWER_DIR / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    if name_key is None:
        name = hashlib.sha1(data).hexdigest()[:20] + suffix
    else:
        name = asset_src(name_key, suffix).removeprefix("assets/")
    path = folder / name
    if not path.exists():
        tmp = path.with_name(f".{name}.{threading.get_ident()}")
        tmp.write_bytes(data)
        tmp.replace(path)
        old = sorted(  # sans les fichiers en cours d'écriture (« . ») d'un autre fil
            (p for p in folder.iterdir() if not p.name.startswith(".")),
            key=_mtime,
            reverse=True,
        )
        for stale in old[KEEP_ASSETS:]:
            stale.unlink(missing_ok=True)
    return f"assets/{name}"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:  # effacé entre-temps par un autre fil
        return 0.0


_component = None


def _intervals(intervals: list | None) -> list[list]:
    """[début, fin, label, qualité] de chaque intervalle (qualité None si absente)."""
    return [
        [float(i[0]), float(i[1]), str(i[2]), i[3] if len(i) > 3 else None] for i in intervals or []
    ]


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
    others: dict[str, list] | None = None,
    intervals_key: str = "",
    colors: dict[str, str] | None = None,
    fview_hz: float = 10_000.0,
    height: int = 320,
    next_audios: list[tuple[str, str, float]] | None = None,
) -> dict:
    """Arguments du composant. `audios` : (nom, chemin rendu par `asset`, début en s dans
    l'enregistrement), couvrant [t0, t1] ; `windows` : {t0, t1, label, current, candidate} ;
    `key` : identifie l'extrait (un autre extrait remet la vue et les intervalles à zéro) ;
    `labels` : (code, nom) des intervalles, `colors` : {code : #RRGGBB} de celles qui n'ont pas
    de couleur dans `style.LABEL_COLORS` ; `intervals` : [début, fin, label, qualité] affichés à
    l'ouverture de l'extrait, ou quand `intervals_key` change (autre annotateur) ; `others` :
    {annotateur : intervalles}, ceux des autres, affichés sans pouvoir être modifiés ;
    `band_hz`, `channel` : réglages de départ, la première fois ;
    `next_audios` : comme `audios`, pour l'extrait suivant, que le navigateur prépare pendant
    qu'on annote celui-ci (fichiers attendus s'ils ne sont pas encore écrits)."""
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
        "colors": {
            k: [int(v[i : i + 2], 16) for i in (1, 3, 5)] for k, v in (colors or {}).items()
        },
        "intervals": _intervals(intervals),
        "ivkey": str(intervals_key),
        "others": [
            {"who": str(who), "intervals": _intervals(ivs)} for who, ivs in (others or {}).items()
        ],
        "qualities": list(QUALITIES),
        "interval_mode": bool(interval_mode),
        "channel": int(channel),
        "height": int(height),
        "next": [
            {"name": name, "src": src, "start": float(start)}
            for name, src, start in next_audios or []
        ],
    }


def viewer(args: dict, component_key: str = "viewer") -> dict | None:
    """Affiche le visualiseur ; renvoie sa dernière valeur ({key, ivkey, channel, intervals}) ou
    None."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("blanci_viewer", path=str(_prepare()))
    return _component(**args, key=component_key, default=None)
