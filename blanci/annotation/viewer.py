"""Visualiseur de spectrogramme du poste d'annotation : composant Streamlit (page HTML et
JavaScript, sans outil de construction) qui renvoie à Python les intervalles tracés.

- molette : zoom en temps autour du curseur ; Maj + molette : zoom en fréquence ;
- glisser : tracer un intervalle d'A. blanci (mode intervalles) ; Maj + glisser, ou glisser en
  mode fenêtres : se déplacer ; tirer le bord d'un intervalle : l'ajuster ; clic droit ou
  Suppr : l'effacer ; double-clic : tout revoir ; clic simple : la lecture saute là ;
- une barre rouge suit la lecture de l'un ou l'autre micro (la vue suit si « Suivre ») ;
- les fenêtres du découpage (mode fenêtres) sont dessinées, la fenêtre en cours en cyan ; la
  bande d'écoute est entre les pointillés.

Le spectrogramme et l'écoute sont des fichiers servis à côté de la page (`VIEWER_DIR/assets`),
pas des arguments : un nouvel affichage de Streamlit ne renvoie que quelques octets, et la
page n'est pas rechargée (la lecture continue, la vue reste où elle est).
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

VIEWER_DIR = Path(tempfile.gettempdir()) / "blanci_viewer"
KEEP_ASSETS = 60  # fichiers d'écoute et de spectrogramme gardés (les plus récents)

INDEX = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  body { margin: 0; font: 12px "Source Sans Pro", system-ui, sans-serif; color: #ddd;
         background: #0e1117; }
  #bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; padding: 2px 0 4px; }
  button, select { background: #262730; color: #ddd; border: 1px solid #444;
                   border-radius: 4px; padding: 2px 8px; cursor: pointer; font: inherit; }
  button:hover { border-color: #ff4b4b; }
  #bar .help { color: #888; margin-left: auto; }
  canvas { display: block; width: 100%; cursor: crosshair; }
  #players { display: flex; gap: 12px; margin-top: 4px; }
  #players div { flex: 1; }
  #players audio { width: 100%; height: 32px; }
  #players span { font-weight: 600; }
  #players .playing span { color: #ff4b4b; }
  #list { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; min-height: 4px; }
  .chip { border: 1px solid #555; border-radius: 12px; padding: 2px 4px 2px 8px;
          display: flex; gap: 4px; align-items: center; cursor: pointer; }
  .chip.sel { border-color: #fff; }
  .chip select, .chip button { padding: 0 4px; }
</style></head><body>
<div id="bar">
  <span id="ivbar">Nouvel intervalle :
    <select id="newlabel"></select></span>
  <button id="zin" title="Zoom avant en temps">＋ temps</button>
  <button id="zout" title="Zoom arrière en temps">－ temps</button>
  <button id="fin" title="Zoom avant en fréquence">＋ fréq.</button>
  <button id="fout" title="Zoom arrière en fréquence">－ fréq.</button>
  <button id="win" title="Cadrer la fenêtre du candidat">Candidat</button>
  <button id="all" title="Tout voir (double-clic)">Tout</button>
  <label><input type="checkbox" id="follow" checked> Suivre la lecture</label>
  <span class="help" id="help"></span>
</div>
<canvas id="c"></canvas>
<div id="list"></div>
<div id="players"></div>
<script>
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const M = {l: 44, r: 8, t: 16, b: 22};
const COLORS = {blanci: [80, 220, 120], blanci_chorus: [190, 120, 255],
                blanci_uncertain: [255, 170, 40]};
let D = null, img = new Image(), players = [], active = 0;
let view = null, intervals = [], selected = -1;

function send(type, data) {
  window.parent.postMessage(Object.assign({isStreamlitMessage: true, type}, data), "*");
}
function setHeight() { send("streamlit:setFrameHeight", {height: document.body.scrollHeight}); }
function emit() {
  send("streamlit:setComponentValue", {
    dataType: "json",
    value: {key: D.extract, intervals: intervals.map(i => [round(i.t0), round(i.t1), i.label])},
  });
}
const round = t => Math.round(t * 100) / 100;
const H = () => D ? D.height : 300;
const W = () => canvas.clientWidth - M.l - M.r;
const PH = () => H() - M.t - M.b;
const xOf = t => M.l + (t - view.t0) / (view.t1 - view.t0) * W();
const yOf = f => M.t + (1 - (f - view.f0) / (view.f1 - view.f0)) * PH();
const tOf = x => view.t0 + (x - M.l) / W() * (view.t1 - view.t0);
const fOf = y => view.f0 + (1 - (y - M.t) / PH()) * (view.f1 - view.f0);
const label = code => (D.labels.find(l => l[0] === code) || [code, code])[1];

function render(args) {
  const fresh = !D || args.extract !== D.extract;
  const oldAudios = D ? D.audios : [];
  D = args;
  document.getElementById("ivbar").style.display = D.interval_mode ? "" : "none";
  document.getElementById("help").textContent = D.interval_mode
    ? "glisser : tracer · bord : ajuster · clic droit / Suppr : effacer · Maj+glisser : " +
      "déplacer · molette : zoom (Maj : fréq.) · clic : lire depuis là"
    : "glisser : déplacer · molette : zoom temps (Maj : fréq.) · clic : lire depuis là";
  const sel = document.getElementById("newlabel");
  if (!sel.options.length) {
    D.labels.forEach(([code, name]) => sel.add(new Option(name, code)));
  }
  if (fresh) {
    view = {t0: D.t0, t1: D.t1, f0: 0, f1: Math.min(D.fview, D.fmax)};
    intervals = (D.intervals || []).map(([t0, t1, l]) => ({t0, t1, label: l}));
    selected = -1;
    img = new Image(); img.onload = draw; img.src = D.image;
    buildPlayers();
    list();
  } else if (JSON.stringify(oldAudios) !== JSON.stringify(D.audios)) {
    swapAudio();  // volume ou bande changés : même extrait, même position
  }
  clampView(); resize(); setHeight();
}

function buildPlayers() {
  players.forEach(p => p.pause());
  const box = document.getElementById("players");
  box.innerHTML = ""; players = []; active = 0;
  D.audios.forEach((a, i) => {
    const div = document.createElement("div");
    div.innerHTML = `<span>${a.name}</span>`;
    const el = document.createElement("audio");
    el.controls = true; el.preload = "auto"; el.src = a.src;
    el.addEventListener("play", () => {
      players.forEach((p, j) => { if (j !== i) p.pause(); });
      active = i; mark(); loop();
    });
    el.addEventListener("seeked", draw);
    div.appendChild(el); box.appendChild(div); players.push(el);
  });
  mark();
}
function swapAudio() {
  players.forEach((p, i) => {
    const time = p.currentTime, playing = !p.paused;
    p.src = D.audios[i].src;
    p.addEventListener("loadedmetadata", () => {
      p.currentTime = time; if (playing) p.play().catch(() => {});
    }, {once: true});
  });
}
function mark() {
  [...document.getElementById("players").children]
    .forEach((d, j) => d.classList.toggle("playing", j === active));
}

function list() {
  const box = document.getElementById("list");
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

function resize() {
  const r = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * r; canvas.height = H() * r;
  canvas.style.height = H() + "px";
  ctx.setTransform(r, 0, 0, r, 0, 0);
  draw();
}
function clampView() {
  const span = Math.min(Math.max(view.t1 - view.t0, 0.2), D.t1 - D.t0);
  const t0 = Math.min(Math.max(view.t0, D.t0), D.t1 - span);
  view.t0 = t0; view.t1 = t0 + span;
  const fspan = Math.min(Math.max(view.f1 - view.f0, 200), D.fmax);
  const f0 = Math.min(Math.max(view.f0, 0), D.fmax - fspan);
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
  if (!D) return;
  const w = canvas.clientWidth;
  ctx.fillStyle = "#0e1117"; ctx.fillRect(0, 0, w, H());
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
                      label: document.getElementById("newlabel").value}]
    : intervals;
  shown.forEach((iv, i) => {
    const x0 = xOf(iv.t0), x1 = xOf(iv.t1), c = COLORS[iv.label] || [200, 200, 200];
    ctx.fillStyle = `rgba(${c}, 0.22)`; ctx.fillRect(x0, M.t, x1 - x0, PH());
    ctx.fillStyle = `rgba(${c}, 0.95)`; ctx.fillRect(x0, M.t, x1 - x0, 5);
    ctx.strokeStyle = `rgb(${c})`; ctx.lineWidth = i === selected ? 3 : 1.5;
    ctx.beginPath(); ctx.moveTo(x0, M.t); ctx.lineTo(x0, M.t + PH());
    ctx.moveTo(x1, M.t); ctx.lineTo(x1, M.t + PH()); ctx.stroke();
  });
  ctx.setLineDash([5, 4]); ctx.strokeStyle = "rgba(255, 255, 255, 0.75)"; ctx.lineWidth = 1;
  D.band.forEach(f => {
    const y = yOf(f); ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(M.l + W(), y); ctx.stroke();
  });
  ctx.setLineDash([]);
  const t = playheadTime();
  if (t !== null) {
    const x = xOf(t);
    ctx.strokeStyle = "#ff4b4b"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, M.t); ctx.lineTo(x, M.t + PH()); ctx.stroke();
  }
  ctx.restore();
  ctx.fillStyle = "#aaa"; ctx.strokeStyle = "#666"; ctx.font = "11px system-ui";
  const ts = niceStep(view.t1 - view.t0, Math.max(2, W() / 90));
  for (let v = Math.ceil(view.t0 / ts) * ts; v <= view.t1 + 1e-9; v += ts) {
    const x = xOf(v);
    ctx.beginPath(); ctx.moveTo(x, M.t + PH()); ctx.lineTo(x, M.t + PH() + 4); ctx.stroke();
    ctx.fillText(+v.toFixed(3) + " s", x - 10, H() - 6);
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
    const p = players[active], t = playheadTime();
    if (t !== null && document.getElementById("follow").checked && !drag &&
        (t > view.t1 || t < view.t0)) {
      const span = view.t1 - view.t0;
      view.t0 = t - 0.1 * span; view.t1 = view.t0 + span; clampView();
    }
    draw();
    raf = p && !p.paused ? requestAnimationFrame(step) : null;
  };
  raf = requestAnimationFrame(step);
}
function zoomTime(factor, at) {
  at = at ?? (view.t0 + view.t1) / 2;
  view.t0 = at - (at - view.t0) * factor; view.t1 = at + (view.t1 - at) * factor;
  clampView(); draw();
}
function zoomFreq(factor, at) {
  at = at ?? (view.f0 + view.f1) / 2;
  view.f0 = at - (at - view.f0) * factor; view.f1 = at + (view.f1 - at) * factor;
  clampView(); draw();
}
const localX = e => e.clientX - canvas.getBoundingClientRect().left;
const localY = e => e.clientY - canvas.getBoundingClientRect().top;
function hit(x) {  // [indice, bord ("t0", "t1") ou null] de l'intervalle sous le curseur
  for (let i = intervals.length - 1; i >= 0; i--) {
    const iv = intervals[i];
    if (Math.abs(x - xOf(iv.t0)) < 6) return [i, "t0"];
    if (Math.abs(x - xOf(iv.t1)) < 6) return [i, "t1"];
  }
  const t = tOf(x);
  const i = intervals.findIndex(iv => t >= iv.t0 && t <= iv.t1);
  return [i, null];
}

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = Math.exp((e.deltaY || e.deltaX) * 0.0015);
  if (e.shiftKey) zoomFreq(factor, fOf(localY(e)));
  else zoomTime(factor, tOf(localX(e)));
}, {passive: false});

let drag = null;
canvas.addEventListener("mousedown", e => {
  if (e.button !== 0 || !D) return;
  const x = localX(e), t = tOf(x);
  const [i, edge] = D.interval_mode ? hit(x) : [-1, null];
  let mode = "pan";
  if (D.interval_mode && !e.shiftKey) mode = edge ? "edge" : "draw";
  drag = {mode, x: e.clientX, y: e.clientY, view: {...view}, moved: false, t, t2: t, i, edge};
});
canvas.addEventListener("mousemove", e => {
  if (drag || !D || !D.interval_mode) return;
  canvas.style.cursor = hit(localX(e))[1] ? "ew-resize" : "crosshair";
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
    clampView();
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
  if (!d.moved) {
    if (D.interval_mode && d.i >= 0) { selected = d.i; list(); }
    const p = players[active];
    if (p && d.t >= D.t0 && d.t <= D.t1) {
      p.currentTime = Math.max(0, d.t - D.audios[active].start);
      if (p.paused) p.play().catch(() => {});
    }
    draw();
    return;
  }
  if (d.mode === "draw") {
    const t0 = Math.min(d.t, d.t2), t1 = Math.max(d.t, d.t2);
    if (t1 - t0 >= 0.05) {
      intervals.push({t0, t1, label: document.getElementById("newlabel").value});
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
  view = {t0: D.t0, t1: D.t1, f0: 0, f1: Math.min(D.fview, D.fmax)}; draw();
});
document.getElementById("zin").onclick = () => zoomTime(0.5);
document.getElementById("zout").onclick = () => zoomTime(2);
document.getElementById("fin").onclick = () => zoomFreq(0.5);
document.getElementById("fout").onclick = () => zoomFreq(2);
document.getElementById("all").onclick = () => canvas.dispatchEvent(new Event("dblclick"));
document.getElementById("win").onclick = () => {
  const w = (D.windows || []).find(w => w.current) || (D.windows || [])[0];
  if (!w) return;
  const pad = 0.5 * (w.t1 - w.t0);
  view.t0 = w.t0 - pad; view.t1 = w.t1 + pad; clampView(); draw();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.code === "Space") {
    e.preventDefault();
    const p = players[active]; if (p) (p.paused ? p.play() : p.pause());
  } else if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0) {
    remove(selected);
  } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
    const span = view.t1 - view.t0, s = (e.key === "ArrowLeft" ? -0.25 : 0.25) * span;
    view.t0 += s; view.t1 += s; clampView(); draw();
  }
});

window.addEventListener("message", e => {
  if (e.data && e.data.type === "streamlit:render") render(e.data.args);
});
window.addEventListener("resize", () => { if (D) resize(); });
send("streamlit:componentReady", {apiVersion: 1});
</script></body></html>
"""


def _prepare() -> Path:
    """Page du composant, réécrite si elle a changé (mise à jour du code)."""
    (VIEWER_DIR / "assets").mkdir(parents=True, exist_ok=True)
    index = VIEWER_DIR / "index.html"
    if not index.exists() or index.read_text(encoding="utf-8") != INDEX:
        index.write_text(INDEX, encoding="utf-8")
    return VIEWER_DIR


def asset(data: bytes, suffix: str) -> str:
    """Range `data` à côté de la page (nom = empreinte du contenu) ; chemin relatif à la page.
    Seuls les `KEEP_ASSETS` fichiers les plus récents sont gardés."""
    folder = _prepare() / "assets"
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
    png: bytes,
    t0: float,
    t1: float,
    fmax_hz: float,
    audios: list[tuple[str, bytes, float]],
    windows: list[dict],
    band_hz: tuple[float, float],
    key: str,
    labels: list[tuple[str, str]],
    intervals: list | None = None,
    interval_mode: bool = False,
    fview_hz: float = 10_000.0,
    height: int = 300,
) -> dict:
    """Arguments du composant. `png` : spectrogramme couvrant [t0, t1] s et [0, fmax_hz] Hz
    (`workbench.spectrogram_png`) ; `audios` : (nom, WAV, début en s dans l'enregistrement) ;
    `windows` : {t0, t1, label, current, candidate} ; `key` : identifie l'extrait (un autre
    extrait remet la vue et les intervalles à zéro) ; `labels` : (code, nom) des intervalles ;
    `intervals` : [début, fin, label] affichés à l'ouverture de l'extrait."""
    return {
        "image": asset(png, ".png"),
        "t0": float(t0),
        "t1": float(t1),
        "fmax": float(fmax_hz),
        "fview": float(fview_hz),
        "audios": [
            {"name": name, "src": asset(wav, ".wav"), "start": float(start)}
            for name, wav, start in audios
        ],
        "windows": windows,
        "band": [float(f) for f in band_hz],
        "extract": key,
        "labels": [list(label) for label in labels],
        "intervals": [[float(a), float(b), str(c)] for a, b, c in intervals or []],
        "interval_mode": bool(interval_mode),
        "height": int(height),
    }


def viewer(args: dict, component_key: str = "viewer") -> dict | None:
    """Affiche le visualiseur ; renvoie sa dernière valeur ({key, intervals}) ou None."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("blanci_viewer", path=str(_prepare()))
    return _component(**args, key=component_key, default=None)
