"""Bandeau du poste d'annotation, repris du tableau de bord : titre, A. blanci en 8-bit perchée
sur un vrai chant (spectrogramme balayé par une tête de lecture), mosaïque de pixels ambrés.

Composant Streamlit à part, comme le visualiseur : la page et ses fichiers sont servis une
fois, l'animation continue d'un clic à l'autre (rien n'est renvoyé à chaque réexécution).
Il est posé sur la page sans cadre : son fond (dégradé de nuit, lueurs, filet du bas) et ses
marges sont ceux du conteneur, dans la feuille de style du poste (`style.py`) ; la page du
composant reste transparente. Ses encres suivent le thème que Streamlit lui passe (clair ou
sombre) ; le spectrogramme reste sombre dans les deux.
L'image, le son et la grenouille sont ceux du tableau de bord (`documentation/tableau-de-bord/`,
produits par `spectrogramme.py` et `grenouille.py`), recopiés à côté de la page quand ils
changent ; sans eux, le bandeau garde son titre (sans `chant.mp3`, pas de bouton d'écoute).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from blanci.annotation.style import FONTS
from blanci.core.config import PROJECT_ROOT

BANDEAU_DIR = Path(tempfile.gettempdir()) / "blanci_bandeau"
SOURCES = PROJECT_ROOT / "documentation" / "tableau-de-bord"
FILES = ("spectrogramme.jpg", "chant.mp3", "grenouille.json")

INDEX = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<link rel="stylesheet" href="__FONTS__">
<style>
  /* Le bandeau du tableau de bord (modele.html), resserré pour laisser la place au poste. Pas
     de color-scheme sombre : différent de celui du cadre, il rendrait son fond opaque. */
  :root { --ink: #E9F2EC; --muted: #93A89D; --mono: "JetBrains Mono", ui-monospace, Menlo,
          monospace;
          --titre: linear-gradient(100deg, #FFFFFF 0%, #FFF3D6 40%, #FFD27A 70%, #F5A53A 100%);
          --titre-i: linear-gradient(100deg, #FFE08A, #F5A53A 45%, #E4573F 80%, #C0397A);
          --titre-ombre: drop-shadow(0 2px 10px rgba(4, 8, 7, .9)); }
  /* Thème clair (data-theme : celui que Streamlit passe au composant ; avant, le système). */
  @media (prefers-color-scheme: light) {
    html:not([data-theme="dark"]) {
      --ink: #11201A; --muted: #55675E; --titre-ombre: none;
      --titre: linear-gradient(100deg, #11201A 0%, #3A2414 45%, #8A3A0C 75%, #B9530D 100%);
      --titre-i: linear-gradient(100deg, #A86F00, #B9530D 45%, #C2352A 80%, #8A1F5C); }
  }
  html[data-theme="light"] {
    --ink: #11201A; --muted: #55675E; --titre-ombre: none;
    --titre: linear-gradient(100deg, #11201A 0%, #3A2414 45%, #8A3A0C 75%, #B9530D 100%);
    --titre-i: linear-gradient(100deg, #A86F00, #B9530D 45%, #C2352A 80%, #8A1F5C); }
  [hidden] { display: none !important; }
  html, body { margin: 0; background: transparent; }
  body { color: var(--ink); font: 15px/1.55 "Hanken Grotesk", "Segoe UI", system-ui, sans-serif;
         -webkit-font-smoothing: antialiased; }
  /* Ni fond ni bord : ils sont au conteneur, sur la page. */
  .bande { position: relative; isolation: isolate; overflow: hidden; display: grid; gap: 12px;
           padding: clamp(8px, 1.4vw, 16px) 0 22px; }
  h1, p { margin: 0; }
  /* Titre et sous-titre, devant la grenouille, assez hauts pour qu'elle tienne dans le cadre
     au-dessus du spectrogramme (--grenouille-h : hauteur de sa toile, posée par son script ;
     34 px : les rangs vides du haut de la toile, l'écart avant le perchoir, la marge du haut). */
  .texte { position: relative; z-index: 2; display: grid; gap: 12px; align-content: start;
           min-height: calc(var(--grenouille-h, 34px) - 34px); }
  h1 { font-family: "Unbounded", "Segoe UI", system-ui, sans-serif; font-weight: 700;
       font-size: clamp(24px, 3.6vw, 44px); line-height: 1.08; letter-spacing: -.025em;
       max-width: 19ch; text-wrap: balance;
       background: var(--titre); -webkit-background-clip: text; background-clip: text;
       color: transparent; filter: var(--titre-ombre); }
  h1 i { font-family: "Instrument Serif", Georgia, serif; font-style: italic; font-weight: 400;
         font-size: 1.12em; letter-spacing: 0; padding-right: .08em;
         background: var(--titre-i); -webkit-background-clip: text; background-clip: text;
         color: transparent; }
  /* Sous-titre et bouton d'écoute côte à côte, à gauche de la grenouille (--perchoir : la
     place qu'elle prend au bord droit, posée par son script) ; le bouton passe sous le
     sous-titre quand la place manque. */
  .accroche { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 24px;
              max-width: calc(100% - var(--perchoir, 0px) - 16px); }
  p { color: var(--muted); max-width: 62ch; flex: 1 1 300px; }
  .ecouter { flex: none; display: inline-flex; align-items: center; gap: 12px; height: 50px;
    padding: 0 22px 0 5px; border: 0; border-radius: 999px; cursor: pointer;
    font: 600 15px/1 "Hanken Grotesk", "Segoe UI", system-ui, sans-serif; letter-spacing: .01em;
    color: #1A0B03; background: linear-gradient(100deg, #FFE08A 0%, #F5A53A 48%, #E4573F 100%);
    box-shadow: 0 0 0 1px rgba(255, 224, 138, .4) inset, 0 12px 34px -12px rgba(228, 87, 63, .8),
                0 0 40px -14px rgba(245, 165, 58, .6);
    transition: transform .15s, filter .2s; }
  .ecouter:hover { transform: translateY(-1px); filter: brightness(1.07) saturate(1.05); }
  .ecouter:active { transform: none; filter: brightness(.96); }
  .ecouter:focus-visible { outline: 2px solid #FFE08A; outline-offset: 3px; }
  .ecouter .disque { width: 40px; height: 40px; border-radius: 50%; display: grid;
    place-items: center; background: #120A06; color: #FFE08A;
    box-shadow: 0 0 0 1px rgba(255, 224, 138, .25); }
  .ecouter .lire { width: 15px; height: 15px; margin-left: 2px; fill: currentColor; }
  .ecouter .egaliseur { display: none; align-items: flex-end; gap: 3px; height: 15px; }
  .ecouter .egaliseur i { width: 3px; height: 100%; border-radius: 2px; background: currentColor;
                          transform-origin: bottom; }
  .ecouter[aria-pressed="true"] .lire { display: none; }
  .ecouter[aria-pressed="true"] .egaliseur { display: flex; }
  /* A. blanci perchée sur le spectrogramme, sous le titre qui la chevauche, calée au bord
     droit du cadre (sur le tableau de bord, son tibia mord sur la marge de la page, que le
     cadre du composant ne couvre pas). */
  .perchoir { position: relative; z-index: 1; height: 0; margin-bottom: -12px;
              pointer-events: none; }
  #grenouille { position: absolute; right: 0; bottom: 0; max-width: none;
                image-rendering: pixelated; image-rendering: crisp-edges; }
  @media (max-width: 480px) {
    .texte { min-height: 0; }
    .accroche { max-width: none; }
    .perchoir { height: auto; display: flex; justify-content: flex-end; }
    #grenouille { position: static; }
  }
  .spectro { position: relative; border-radius: 12px; overflow: hidden; background: #000004;
             box-shadow: 0 0 0 1px rgba(255, 210, 120, .14),
                         0 30px 80px -30px rgba(228, 87, 63, .55),
                         0 0 60px -20px rgba(245, 165, 58, .35); }
  .spectro img { width: 100%; height: clamp(56px, 6vw, 80px); display: block; object-fit: fill;
                 filter: saturate(1.15) contrast(1.06); }
  .spectro::before { content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 1;
    background: linear-gradient(180deg, rgba(0, 0, 0, .25), transparent 22%, transparent 78%,
                                rgba(0, 0, 0, .35)),
                repeating-linear-gradient(90deg, rgba(255, 255, 255, .05) 0 1px,
                                          transparent 1px 10%); }
  .tete-lecture { position: absolute; top: 0; bottom: 0; left: 0; width: 2px; z-index: 2;
    background: linear-gradient(180deg, transparent, #FFF6DC 20%, #FFF6DC 80%, transparent);
    box-shadow: 0 0 12px 2px rgba(255, 224, 138, .8), 0 0 40px 8px rgba(245, 165, 58, .35);
    pointer-events: none; opacity: 0; }
  .joue .tete-lecture { opacity: 1; }
  .tete-lecture::before { content: ""; position: absolute; top: 0; bottom: 0; right: 2px;
    width: 90px; background: linear-gradient(90deg, transparent, rgba(255, 224, 138, .12)); }
  .bande-f { position: absolute; left: 0; right: 0; top: 16.7%; height: 36.7%; z-index: 1;
    border-block: 1px dashed rgba(255, 224, 138, .85); pointer-events: none;
    background: linear-gradient(180deg, rgba(255, 224, 138, .05), rgba(255, 224, 138, 0) 50%,
                                rgba(255, 224, 138, .05)); }
  .axe { position: absolute; z-index: 2; left: 8px; font: 10.5px/1.2 var(--mono); color: #FFF0C8;
         background: rgba(0, 0, 0, .45); padding: 2px 6px; border-radius: 4px; }
  .legende { font: 11px/1.4 var(--mono); color: var(--muted); }
  @media (prefers-reduced-motion: no-preference) {
    .tete-lecture { opacity: 1; animation: balayage 18s linear infinite; }
    .ecouter .egaliseur i { animation: egaliseur .9s ease-in-out infinite alternate; }
    .ecouter .egaliseur i:nth-child(2) { animation-delay: -.3s; }
    .ecouter .egaliseur i:nth-child(3) { animation-delay: -.6s; }
    @keyframes egaliseur { from { transform: scaleY(.25); } to { transform: scaleY(1); } }
    @keyframes balayage { from { left: 0; } to { left: 100%; } }
    .bande > * { animation: leve .9s cubic-bezier(.2, .7, .2, 1) both; }
    .bande > :nth-child(2) { animation-delay: .08s; }
    .bande > :nth-child(3) { animation-delay: .16s; }
    .bande > :nth-child(4) { animation-delay: .24s; }
    @keyframes leve { from { transform: translateY(14px); opacity: 0; } to { transform: none; } }
  }
</style></head><body>
<header class="bande">
  <div class="texte">
    <h1>Poste d'annotation d'<i>Anomaloglossus blanci</i></h1>
    <div class="accroche">
      <p>Écouter chaque extrait, tracer les chants sur le spectrogramme, nommer les faux amis :
        chaque réponse s'ajoute aux labels, aucune n'en efface.</p>
      <button type="button" class="ecouter" id="ecouter" aria-pressed="false" hidden>
        <span class="disque" aria-hidden="true"><svg class="lire" viewBox="0 0 16 16">
          <path d="M4.5 2.6v10.8L13.4 8z"/></svg><span class="egaliseur"><i></i><i></i><i></i>
        </span></span>
        <span class="libelle">Écouter le chant</span>
      </button>
    </div>
  </div>
  <div class="perchoir" aria-hidden="true"><canvas id="grenouille"></canvas></div>
  <div class="spectro">
    <img src="spectrogramme.jpg" alt="Spectrogramme de 18 s d'un chant d'A. blanci enregistré
      à Molokoi, de 3 à 6 kHz : notes brèves et régulières vers 5 kHz">
    <span class="bande-f" aria-hidden="true"></span>
    <span class="tete-lecture" aria-hidden="true"></span>
    <span class="axe" style="top: 6px" aria-hidden="true">6 kHz</span>
    <span class="axe" style="bottom: 6px" aria-hidden="true">3 kHz</span>
  </div>
  <div class="legende">chant d'A. blanci enregistré à Molokoi le 26/02/2024, 18 s : notes de
    0,09 s · bande 4,4–5,5 kHz en pointillés</div>
</header>
<script>
const $ = s => document.querySelector(s);
const bande = $(".bande");

// --- Échanges avec Streamlit : la hauteur du cadre suit celle du bandeau ------------------
function send(type, data) {
  window.parent.postMessage(Object.assign({isStreamlitMessage: true, type}, data), "*");
}
let hauteur = 0;
function setHeight(force) {
  const h = Math.ceil(bande.getBoundingClientRect().height);
  if (force || h !== hauteur) { hauteur = h; send("streamlit:setFrameHeight", {height: h}); }
}
new ResizeObserver(() => setHeight()).observe(bande);
$(".spectro img").addEventListener("error", () => {
  $(".spectro").hidden = true; $(".legende").hidden = true; $("#ecouter").hidden = true;
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
  if (e.data && e.data.type === "streamlit:render") { theme(e.data.theme); setHeight(true); }
});
send("streamlit:componentReady", {apiVersion: 1});

// --- Le chant à l'écoute -------------------------------------------------------------------
// Le code du tableau de bord : chant.mp3 (spectrogramme.py), les 18 s du spectrogramme ; le
// bouton n'apparaît que si le fichier est là. Le son part de l'endroit où passe la tête de
// lecture et tourne en boucle avec elle ; pendant l'écoute, c'est l'horloge du son qui la place
// (Web Audio : boucle sans blanc, position exacte, même sans requêtes partielles du serveur de
// Streamlit). « Couper le son » rend la main au balayage, là où il en est.
function ecoute(bouton, spectro) {
  const tete = spectro.querySelector(".tete-lecture"), CYCLE = 18000;  // ms, durée du balayage
  let octets = null, ctx = null, tampon = null, source = null, gain = null, t0 = 0, depart = 0;
  let occupe = false;
  fetch("chant.mp3").then(r => (r.ok ? r.arrayBuffer() : null)).then(b => {
    if (b && window.AudioContext && !spectro.hidden) { octets = b; bouton.hidden = false; }
  }).catch(() => {});
  // le balayage de la feuille de style (absent si l'on a demandé moins d'animations)
  const balayage = () => tete.getAnimations().find(a => a.animationName === "balayage");
  const placer = f => {  // f : fraction de l'extrait
    const a = balayage();
    if (a) a.currentTime = f * CYCLE; else tete.style.left = `${f * 100}%`;
  };
  const etat = joue => {
    bouton.setAttribute("aria-pressed", String(joue));
    bouton.querySelector(".libelle").textContent = joue ? "Couper le son" : "Écouter le chant";
    spectro.classList.toggle("joue", joue);
  };
  function suivre() {
    if (!source) return;
    const d = tampon.duration;
    const s = depart + Math.max(0, ctx.currentTime - t0 - (ctx.outputLatency || 0));
    placer((s % d) / d);
    requestAnimationFrame(suivre);
  }
  async function lancer() {
    if (!ctx) { ctx = new AudioContext(); tampon = await ctx.decodeAudioData(octets); }
    await ctx.resume();
    const a = balayage();
    depart = a ? (((a.currentTime || 0) % CYCLE) / CYCLE) * tampon.duration : 0;
    if (a) a.pause();
    gain = ctx.createGain(); gain.connect(ctx.destination);
    source = ctx.createBufferSource();
    source.buffer = tampon; source.loop = true; source.connect(gain);
    t0 = ctx.currentTime;
    // fondu de 40 ms à l'entrée, et à la sortie dans couper() : sans claquement
    gain.gain.setValueAtTime(0, t0); gain.gain.linearRampToValueAtTime(1, t0 + 0.04);
    source.start(t0, depart);
    etat(true); requestAnimationFrame(suivre);
  }
  function couper() {
    const s = source, g = gain, t = ctx.currentTime;
    source = null;
    g.gain.setValueAtTime(g.gain.value, t); g.gain.linearRampToValueAtTime(0, t + 0.04);
    s.stop(t + 0.05);
    const a = balayage();
    if (a) a.play();
    etat(false);
  }
  bouton.addEventListener("click", async () => {
    if (occupe) return;
    if (source) { couper(); return; }
    occupe = true;
    try { await lancer(); } catch { bouton.hidden = true; } finally { occupe = false; }
  });
}
ecoute($("#ecouter"), $(".spectro"));

// --- A. blanci en 8-bit -------------------------------------------------------------------
// Le code du tableau de bord : sprite de grenouille.py (images codées par plages : « O12 » =
// douze pixels de la couleur O), dans une toile de 32 pixels de plus en largeur et 8 en
// hauteur, sur une mosaïque de tuiles ambrées de 2 × 2 qui scintillent. Elle respire (flanc),
// sa gorge palpite, elle cligne, et de temps en temps elle baisse ou relève un peu la tête.
fetch("grenouille.json").then(r => (r.ok ? r.json() : null)).then(G => {
  if (G) grenouille(G); else $(".perchoir").hidden = true;
}).catch(() => { $(".perchoir").hidden = true; });

function grenouille(G) {
  const toile = $("#grenouille"), ctx = toile.getContext("2d");
  const IMG = Object.fromEntries(Object.entries(G.images).map(([k, rangs]) =>
    [k, rangs.map(r => [...r.matchAll(/(\D)(\d+)/g)].map(m => [m[1], +m[2]]))]));
  const DX = 24, DY = 8, TUILE = 2, COL = G.largeur + 32, LIG = G.hauteur + DY;
  const NC = COL / TUILE, NL = LIG / TUILE;
  const CX = (DX + G.largeur * 0.54) / TUILE, CY = (DY + G.hauteur / 2) / TUILE;
  const RX = G.largeur * 0.625 / TUILE, RY = G.hauteur * 0.525 / TUILE;
  const calme = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const phases = Array.from({length: NC * NL},
                            () => [Math.random() * 6.28, 0.6 + Math.random() * 1.6]);
  let p = 1;
  const dimension = () => {
    // 2 px d'écran par pixel du sprite (le bandeau du poste est plus bas que celui du tableau
    // de bord), arrondi aux pixels de l'appareil pour garder des pixels nets
    const dpr = devicePixelRatio || 1;
    p = Math.max(1, Math.floor(2 * dpr));
    toile.width = COL * p; toile.height = LIG * p;
    toile.style.width = (COL * p / dpr) + "px"; toile.style.height = (LIG * p / dpr) + "px";
    // place qu'elle prend au bord droit, jusqu'au bord gauche du sprite : le sous-titre et le
    // bouton d'écoute restent à sa gauche
    bande.style.setProperty("--perchoir", ((COL - DX) * p / dpr) + "px");
    bande.style.setProperty("--grenouille-h", (LIG * p / dpr) + "px");
  };
  let cligne = -9, prochainCligne = 2.5, tete = 1, prochaineTete = 4 + Math.random() * 3;
  function dessiner(t) {
    ctx.clearRect(0, 0, toile.width, toile.height);
    const q = p * TUILE, jeu = Math.max(1, Math.round(q / 7));
    for (let y = 0; y < NL; y++) for (let x = 0; x < NC; x++) {
      const d = Math.hypot((x - CX) / RX, (y - CY) / RY);
      if (d >= 1) continue;
      const [ph, v] = phases[y * NC + x];
      const a = (1 - d) * (1 - d) * (calme ? 0.16 : 0.1 + 0.1 * Math.sin(t * v + ph));
      if (a < 0.015) continue;
      ctx.fillStyle = `rgba(245, 165, 58, ${a.toFixed(3)})`;
      ctx.fillRect(x * q + jeu, y * q + jeu, q - jeu, q - jeu);
    }
    const souffle = !calme && t % 3.2 < 1.3 ? 1 : 0;
    const gorge = !calme && t % 7 < 2.4 && t % 0.6 < 0.3 ? 1 : 0;
    if (!calme && t > prochainCligne) {
      cligne = t;
      prochainCligne = t + 3 + Math.random() * 4.5 + (Math.random() < 0.25 ? -2.8 : 0);
    }
    const yeux = !calme && t - cligne < 0.13 ? 1 : 0;
    // tête : la plupart du temps droite, parfois un peu baissée ou relevée, quelques secondes
    if (!calme && t > prochaineTete) {
      tete = tete !== 1 ? 1 : Math.random() < 0.5 ? 0 : 2;
      prochaineTete = t + (tete === 1 ? 4 + Math.random() * 5 : 1.6 + Math.random() * 2.4);
    }
    const img = IMG[`${gorge}${souffle}${yeux}${tete}`] || IMG["0001"];
    for (let y = 0; y < img.length; y++) {
      let x = 0;
      for (const [c, n] of img[y]) {
        if (c !== ".") {
          ctx.fillStyle = G.palette[c]; ctx.fillRect((x + DX) * p, (y + DY) * p, n * p, p);
        }
        x += n;
      }
    }
  }
  dimension();
  addEventListener("resize", () => { dimension(); dessiner(performance.now() / 1000); });
  if (calme) dessiner(0);
  else {
    let dernier = -1;
    const boucle = ms => {
      const t = ms / 1000;
      if (t - dernier >= 1 / 12) { dernier = t; dessiner(t); }  // 12 images par seconde
      requestAnimationFrame(boucle);
    };
    requestAnimationFrame(boucle);
  }
}
</script></body></html>
"""


def _prepare() -> Path:
    """Page du composant (réécrite si elle a changé) et fichiers du tableau de bord (recopiés
    quand ils sont plus récents que la copie)."""
    BANDEAU_DIR.mkdir(parents=True, exist_ok=True)
    index = BANDEAU_DIR / "index.html"
    page = INDEX.replace("__FONTS__", FONTS)
    if not index.exists() or index.read_text(encoding="utf-8") != page:
        index.write_text(page, encoding="utf-8")
    for name in FILES:
        source, copy = SOURCES / name, BANDEAU_DIR / name
        if source.is_file() and (
            not copy.is_file()
            or copy.stat().st_size != source.stat().st_size
            or copy.stat().st_mtime < source.stat().st_mtime
        ):
            shutil.copyfile(source, copy)
    return BANDEAU_DIR


_component = None


def bandeau() -> None:
    """Affiche le bandeau en haut de la page."""
    global _component
    if _component is None:
        import streamlit.components.v1 as components

        _component = components.declare_component("blanci_bandeau", path=str(_prepare()))
    _component(key="bandeau", default=None)
