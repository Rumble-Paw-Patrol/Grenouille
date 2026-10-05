"""Visualiseur de spectrogramme du poste d'annotation : une page HTML autonome (affichée par
`streamlit.components.v1.html`), sans aller-retour avec Python.

- molette : zoom en temps autour du curseur ; Maj + molette : zoom en fréquence ;
- cliquer-glisser : se déplacer ; double-clic : tout revoir ; clic simple : la lecture saute là ;
- une barre rouge suit la lecture de l'un ou l'autre micro (la vue suit si « Suivre ») ;
- les fenêtres du découpage sont dessinées, la fenêtre en cours en cyan, les fenêtres déjà
  écoutées avec leur label ; la bande d'écoute est entre les pointillés.

La vue et la position de lecture sont gardées (sessionStorage) d'un affichage à l'autre du
même extrait : envoyer une réponse ne ramène pas au début.
"""

from __future__ import annotations

import base64
import json

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { margin: 0; font: 12px system-ui, sans-serif; color: #ddd; background: #0e1117; }
  #bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; padding: 2px 0 4px; }
  #bar button { background: #262730; color: #ddd; border: 1px solid #444; border-radius: 4px;
                padding: 2px 8px; cursor: pointer; }
  #bar button:hover { border-color: #ff4b4b; }
  #bar .help { color: #888; margin-left: auto; }
  canvas { display: block; width: 100%; cursor: crosshair; }
  #players { display: flex; gap: 12px; margin-top: 4px; }
  #players div { flex: 1; }
  #players audio { width: 100%; height: 32px; }
  #players span { font-weight: 600; }
  #players .playing span { color: #ff4b4b; }
</style></head><body>
<div id="bar">
  <button id="zin" title="Zoom avant en temps">＋ temps</button>
  <button id="zout" title="Zoom arrière en temps">－ temps</button>
  <button id="fin" title="Zoom avant en fréquence">＋ fréq.</button>
  <button id="fout" title="Zoom arrière en fréquence">－ fréq.</button>
  <button id="win" title="Cadrer la fenêtre en cours">Fenêtre</button>
  <button id="all" title="Tout voir (double-clic)">Tout</button>
  <label><input type="checkbox" id="follow" checked> Suivre la lecture</label>
  <span class="help">molette : zoom temps · Maj+molette : zoom fréq. · glisser : déplacer ·
    clic : lire depuis là</span>
</div>
<canvas id="c"></canvas>
<div id="players"></div>
<script>
const D = __DATA__;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const H = D.height, M = {l: 44, r: 8, t: 16, b: 22};
const img = new Image();
const store = (() => { try { return window.sessionStorage; } catch (e) { return null; } })();
const saved = (() => {
  try { return JSON.parse(store.getItem(D.key)) || {}; } catch (e) { return {}; }
})();
let view = saved.view || {t0: D.t0, t1: D.t1, f0: 0, f1: Math.min(D.fview, D.fmax)};
let active = saved.active || 0;

const players = [];
const box = document.getElementById("players");
D.audios.forEach((a, i) => {
  const div = document.createElement("div");
  div.innerHTML = `<span>${a.name}</span>`;
  const el = document.createElement("audio");
  el.controls = true; el.preload = "auto"; el.src = a.src;
  el.addEventListener("loadedmetadata", () => {
    if (i === active && saved.time) el.currentTime = Math.min(saved.time, el.duration || 0);
  }, {once: true});
  el.addEventListener("play", () => {
    players.forEach((p, j) => { if (j !== i) p.pause(); });
    active = i; mark(); loop();
  });
  el.addEventListener("pause", save);
  el.addEventListener("seeked", () => { draw(); save(); });
  div.appendChild(el); box.appendChild(div); players.push(el);
});
function mark() {
  [...box.children].forEach((d, j) => d.classList.toggle("playing", j === active));
}
mark();

function save() {
  try {
    store.setItem(D.key, JSON.stringify(
      {view, active, time: players[active] ? players[active].currentTime : 0}));
  } catch (e) {}
}

function resize() {
  const r = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * r; canvas.height = H * r;
  canvas.style.height = H + "px";
  ctx.setTransform(r, 0, 0, r, 0, 0);
  draw();
}
const W = () => canvas.clientWidth - M.l - M.r;
const PH = () => H - M.t - M.b;
const xOf = t => M.l + (t - view.t0) / (view.t1 - view.t0) * W();
const yOf = f => M.t + (1 - (f - view.f0) / (view.f1 - view.f0)) * PH();
const tOf = x => view.t0 + (x - M.l) / W() * (view.t1 - view.t0);
const fOf = y => view.f0 + (1 - (y - M.t) / PH()) * (view.f1 - view.f0);

function clampView() {
  const span = Math.min(Math.max(view.t1 - view.t0, 0.2), D.t1 - D.t0);
  let t0 = Math.min(Math.max(view.t0, D.t0), D.t1 - span);
  view.t0 = t0; view.t1 = t0 + span;
  const fspan = Math.min(Math.max(view.f1 - view.f0, 200), D.fmax);
  let f0 = Math.min(Math.max(view.f0, 0), D.fmax - fspan);
  view.f0 = f0; view.f1 = f0 + fspan;
}

function niceStep(span, n) {
  const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 5, 10].map(m => m * p).find(s => s >= raw);
}

function playheadTime() {
  const p = players[active];
  return p ? D.audios[active].start + p.currentTime : null;
}

function draw() {
  const w = canvas.clientWidth;
  ctx.fillStyle = "#0e1117"; ctx.fillRect(0, 0, w, H);
  if (img.complete && img.naturalWidth) {
    const iw = img.naturalWidth, ih = img.naturalHeight;
    const sx = (view.t0 - D.t0) / (D.t1 - D.t0) * iw;
    const sw = (view.t1 - view.t0) / (D.t1 - D.t0) * iw;
    const sy = (1 - view.f1 / D.fmax) * ih;
    const sh = (view.f1 - view.f0) / D.fmax * ih;
    ctx.imageSmoothingEnabled = sw < W();
    ctx.drawImage(img, sx, sy, sw, sh, M.l, M.t, W(), PH());
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, W(), PH()); ctx.clip();
  // Fenêtres du découpage
  D.windows.forEach(win => {
    const x0 = xOf(win.t0), x1 = xOf(win.t1);
    if (win.current) {
      ctx.fillStyle = "rgba(0, 255, 255, 0.13)"; ctx.fillRect(x0, M.t, x1 - x0, PH());
      ctx.strokeStyle = "cyan"; ctx.lineWidth = 1.5;
      ctx.strokeRect(x0, M.t + 0.5, x1 - x0, PH() - 1);
    } else {
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
  // Bande d'écoute
  ctx.setLineDash([5, 4]); ctx.strokeStyle = "rgba(255, 255, 255, 0.75)"; ctx.lineWidth = 1;
  D.band.forEach(f => {
    const y = yOf(f); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + W(), y); ctx.stroke();
  });
  ctx.setLineDash([]);
  // Barre de lecture
  const t = playheadTime();
  if (t !== null) {
    const x = xOf(t);
    ctx.strokeStyle = "#ff4b4b"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + PH()); ctx.stroke();
  }
  ctx.restore();
  // Axes
  ctx.fillStyle = "#aaa"; ctx.strokeStyle = "#666"; ctx.font = "11px system-ui";
  const ts = niceStep(view.t1 - view.t0, Math.max(2, W() / 90));
  for (let v = Math.ceil(view.t0 / ts) * ts; v <= view.t1 + 1e-9; v += ts) {
    const x = xOf(v);
    ctx.beginPath(); ctx.moveTo(x, M.t + PH()); ctx.lineTo(x, M.t + PH() + 4); ctx.stroke();
    ctx.fillText(+v.toFixed(3) + " s", x - 10, H - 6);
  }
  const fs = niceStep(view.f1 - view.f0, Math.max(2, PH() / 40));
  for (let v = Math.ceil(view.f0 / fs) * fs; v <= view.f1 + 1e-9; v += fs) {
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(M.l - 4, y); ctx.lineTo(M.l, y); ctx.stroke();
    ctx.fillText(+(v / 1000).toFixed(2) + "", 4, y + 4);
  }
  ctx.fillText("kHz", 4, 11);
}

let raf = null;
function loop() {
  if (raf) return;
  const step = () => {
    const p = players[active];
    const t = playheadTime();
    if (t !== null && document.getElementById("follow").checked && (t > view.t1 || t < view.t0)) {
      const span = view.t1 - view.t0;
      view.t0 = t - 0.1 * span; view.t1 = view.t0 + span; clampView();
    }
    draw();
    if (p && !p.paused) raf = requestAnimationFrame(step);
    else { raf = null; save(); }
  };
  raf = requestAnimationFrame(step);
}

function zoomTime(factor, at) {
  at = at ?? (view.t0 + view.t1) / 2;
  view.t0 = at - (at - view.t0) * factor; view.t1 = at + (view.t1 - at) * factor;
  clampView(); draw(); save();
}
function zoomFreq(factor, at) {
  at = at ?? (view.f0 + view.f1) / 2;
  view.f0 = at - (at - view.f0) * factor; view.f1 = at + (view.f1 - at) * factor;
  clampView(); draw(); save();
}

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  const factor = Math.exp((e.deltaY || e.deltaX) * 0.0015);
  if (e.shiftKey) zoomFreq(factor, fOf(e.clientY - r.top));
  else zoomTime(factor, tOf(e.clientX - r.left));
}, {passive: false});

let drag = null;
canvas.addEventListener("mousedown", e => {
  drag = {x: e.clientX, y: e.clientY, view: {...view}, moved: false};
});
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  if (!drag.moved) return;
  const dt = dx / W() * (drag.view.t1 - drag.view.t0);
  const df = dy / PH() * (drag.view.f1 - drag.view.f0);
  view = {t0: drag.view.t0 - dt, t1: drag.view.t1 - dt,
          f0: drag.view.f0 + df, f1: drag.view.f1 + df};
  clampView(); draw();
});
window.addEventListener("mouseup", e => {
  if (!drag) return;
  if (!drag.moved) {
    const r = canvas.getBoundingClientRect();
    const t = tOf(e.clientX - r.left), p = players[active];
    if (p && t >= D.t0 && t <= D.t1) {
      p.currentTime = Math.max(0, t - D.audios[active].start);
      if (p.paused) p.play().catch(() => {});
    }
  }
  drag = null; save();
});
canvas.addEventListener("dblclick", () => {
  view = {t0: D.t0, t1: D.t1, f0: 0, f1: Math.min(D.fview, D.fmax)}; draw(); save();
});
document.getElementById("zin").onclick = () => zoomTime(0.5);
document.getElementById("zout").onclick = () => zoomTime(2);
document.getElementById("fin").onclick = () => zoomFreq(0.5);
document.getElementById("fout").onclick = () => zoomFreq(2);
document.getElementById("all").onclick = () => canvas.dispatchEvent(new Event("dblclick"));
document.getElementById("win").onclick = () => {
  const w = D.windows.find(w => w.current);
  if (!w) return;
  const pad = 0.25 * (w.t1 - w.t0);
  view.t0 = w.t0 - pad; view.t1 = w.t1 + pad; clampView(); draw(); save();
};
document.addEventListener("keydown", e => {
  if (e.code === "Space") {
    e.preventDefault();
    const p = players[active]; if (p) (p.paused ? p.play() : p.pause());
  }
});

img.onload = draw;
img.src = D.image;
clampView();
window.addEventListener("resize", resize);
resize();
</script></body></html>
"""


def viewer_html(
    png: bytes,
    t0: float,
    t1: float,
    fmax_hz: float,
    audios: list[tuple[str, bytes, float]],
    windows: list[dict],
    band_hz: tuple[float, float],
    key: str,
    fview_hz: float = 10_000.0,
    height: int = 300,
) -> str:
    """Page du visualiseur. `png` : spectrogramme couvrant [t0, t1] s et [0, fmax_hz] Hz
    (`workbench.spectrogram_png`) ; `audios` : (nom, WAV, début en s dans l'enregistrement) ;
    `windows` : {t0, t1, label, current, candidate} ; `key` : identifie l'extrait, pour
    retrouver la vue et la position de lecture ; `fview_hz` : haut de la vue de départ."""

    def uri(mime: str, data: bytes) -> str:
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    data = {
        "image": uri("image/png", png),
        "t0": float(t0),
        "t1": float(t1),
        "fmax": float(fmax_hz),
        "fview": float(fview_hz),
        "audios": [
            {"name": name, "src": uri("audio/wav", wav), "start": float(start)}
            for name, wav, start in audios
        ],
        "windows": windows,
        "band": [float(f) for f in band_hz],
        "key": f"blanci-viewer::{key}",
        "height": int(height),
    }
    # « </ » échappé : une chaîne JSON ne peut pas fermer la balise <script>.
    return TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
