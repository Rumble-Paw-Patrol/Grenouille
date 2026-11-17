"""Habillage du poste d'annotation, repris du tableau de bord
(`documentation/tableau-de-bord/modele.html`) : « nuit guyanaise », fond de sous-bois la
nuit, lueurs braise et ambre, panneaux en verre dépoli, titres en Unbounded, texte en Hanken
Grotesk, étiquettes en JetBrains Mono. Clair ou sombre comme le tableau de bord, selon le
réglage du système (ou le choix fait dans le menu de Streamlit) ; le spectrogramme reste sombre
dans les deux.

Deux étages :
- `THEME` : options `[theme]` de Streamlit communes aux deux thèmes (polices, bords), et
  `THEME_LIGHT` / `THEME_DARK` pour `[theme.light]` / `[theme.dark]` (couleurs des cases,
  curseurs, menus, champs) : avec les deux, Streamlit suit le système. Passées par
  `blanci annotate` en variables d'environnement (`theme_env`). Une option que la version
  installée ne connaît pas est simplement ignorée, là où une option inconnue sur la ligne de
  commande ferait échouer le lancement.
- `CSS` : feuille injectée par `app.py` à chaque affichage, pour ce que le thème ne règle pas
  (fond, panneau de gauche, boutons, cartes, titres, cadres du bandeau et du visualiseur).
  Sombre par défaut ; le thème clair est pris quand la page porte `data-gr-theme="light"`
  (posé par le script de `app.py`, d'après le thème choisi par Streamlit), ou, avant ce
  script, quand le système est en clair. `css()` y ajoute le mode (jour / nuit) et le style
  choisis par l'utilisateur dans le poste (palettes et préférences : `habillage.py`).
- `LABEL_COLORS` : couleurs des étiquettes d'intervalle, pour le visualiseur et la carte.

Le bandeau (titre, grenouille, spectrogramme) est un composant à part : `bandeau.py`.

Pas d'animation de fond ni de grain en surimpression : c'est un poste de travail, le
navigateur garde ses forces pour le spectrogramme.
"""

from __future__ import annotations

import re

from blanci.annotation.habillage import (
    BRAISE_DARK,
    BRAISE_LIGHT,
    declarations,
    palette,
)

# Couleurs du tableau de bord, thème sombre puis thème clair.
BG = "#050B0A"
SURFACE = "#0F1C18"
INK = "#E9F2EC"
MUTED = "#93A89D"
ACCENT = "#F5A53A"
LINE = "#1B2722"  # rgba(160, 210, 185, .14) sur le fond
LINE_STRONG = "#30433B"  # rgba(160, 210, 185, .28) sur le fond

THEME = {
    "showWidgetBorder": "true",
    "showSidebarBorder": "true",
    "baseRadius": "10px",
    # Sans guillemets : selon les versions, Streamlit les ajoute lui-même.
    "font": "Hanken Grotesk, Segoe UI, system-ui, sans-serif",
    "headingFont": "Unbounded, Segoe UI, system-ui, sans-serif",
    "codeFont": "JetBrains Mono, ui-monospace, Menlo, monospace",
}

THEME_DARK = {
    "primaryColor": ACCENT,
    "backgroundColor": BG,
    "secondaryBackgroundColor": SURFACE,
    "textColor": INK,
    "linkColor": ACCENT,
    "codeBackgroundColor": SURFACE,
    "borderColor": LINE_STRONG,
    "dataframeBorderColor": LINE,
    "redColor": "#F07A63",
    "orangeColor": "#E4573F",
    "yellowColor": "#F5B547",
    "greenColor": "#5FD394",
    "blueColor": "#5AA2F0",
    "violetColor": "#B08CF2",
    "grayColor": "#8DA197",
    "sidebar.backgroundColor": "#060E0C",
    "sidebar.secondaryBackgroundColor": SURFACE,
    "sidebar.textColor": INK,
}

THEME_LIGHT = {
    "primaryColor": "#B9530D",
    "backgroundColor": "#EDF3EF",
    "secondaryBackgroundColor": "#E3ECE6",
    "textColor": "#11201A",
    "linkColor": "#B9530D",
    "codeBackgroundColor": "#E3ECE6",
    "borderColor": "#B7C6BD",  # rgba(20, 70, 50, .26) sur le fond
    "dataframeBorderColor": "#D2DDD6",  # rgba(20, 70, 50, .14) sur le fond
    "redColor": "#B33A28",
    "orangeColor": "#C2352A",
    "yellowColor": "#9A5E00",
    "greenColor": "#237A4A",
    "blueColor": "#2F6FB3",
    "violetColor": "#7C5CC0",
    "grayColor": "#637269",
    "sidebar.backgroundColor": "#F6F9F7",
    "sidebar.secondaryBackgroundColor": "#FFFFFF",
    "sidebar.textColor": "#11201A",
}

# Étiquettes des intervalles (boutons, intervalles et pastilles du visualiseur, carte des
# embeddings) : A. blanci en vert, plusieurs en bleu, incertaine en jaune orangé, faux ami en
# rouge.
LABEL_COLORS = {
    "blanci": "#3FC57D",
    "blanci_chorus": "#5AA2F0",
    "blanci_uncertain": "#F5B547",
    "false_friend": "#F06A5B",
}
# Les autres classes sur la carte des embeddings ; les fenêtres jamais écoutées restent grises.
CHART_COLORS = ("#B08CF2", "#4FD8C0", "#FFE08A", "#C0397A", "#E4573F", "#8DA197")
UNHEARD_COLOR = "#4A5A52"


def theme_env() -> dict[str, str]:
    """`THEME`, `THEME_LIGHT` et `THEME_DARK` en variables d'environnement de Streamlit :
    theme.font → STREAMLIT_THEME_FONT, theme.light.primaryColor →
    STREAMLIT_THEME_LIGHT_PRIMARY_COLOR, theme.dark.sidebar.backgroundColor →
    STREAMLIT_THEME_DARK_SIDEBAR_BACKGROUND_COLOR."""
    sections = {"THEME": THEME, "THEME_LIGHT": THEME_LIGHT, "THEME_DARK": THEME_DARK}
    return {
        f"STREAMLIT_{prefix}_"
        + re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).replace(".", "_").upper(): value
        for prefix, options in sections.items()
        for key, value in options.items()
    }


FONTS = (
    "https://fonts.googleapis.com/css2?family=Unbounded:wght@400;500;700"
    "&family=Hanken+Grotesk:wght@400;500;600;700&family=Instrument+Serif:ital@1"
    "&family=JetBrains+Mono:wght@400;500&display=swap"
)

# Grain de pellicule du tableau de bord, posé une fois dans le fond (pas de mélange de calques
# par-dessus la page).
_GRAIN = (
    "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' "
    "height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' "
    "numOctaves='3' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1  0 0 0 0 1  "
    "0 0 0 0 1  0 0 0 .035 0'/></filter><rect width='100%' height='100%' "
    "filter='url(%23n)'/></svg>\")"
)

CSS = (
    """<style>
@import url("__FONTS__");
:root {
__VARS_DARK__
  --marge: clamp(16px, 4vw, 56px);
  --f-display: "Unbounded", "Segoe UI", system-ui, sans-serif;
  --f-serif: "Instrument Serif", Georgia, serif;
  --f-body: "Hanken Grotesk", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --flamme: linear-gradient(100deg, var(--gold) 0%, var(--accent) 38%, var(--accent-2) 70%,
    var(--accent-3) 100%);
}
/* Thème clair : celui du tableau de bord. */
@media (prefers-color-scheme: light) {
  html:not([data-gr-theme="dark"]) {
__VARS_LIGHT__
  }
}
html[data-gr-theme="light"] {
__VARS_LIGHT__
}

/* Fond : les trois lueurs du tableau de bord (braise en haut à gauche, sarcelle en haut à
   droite, pourpre en bas), mêmes teintes et mêmes places, fixes, et le grain. */
.stApp {
  color: var(--ink); font-family: var(--f-body);
  -webkit-font-smoothing: antialiased;
  background:
    __GRAIN__,
    radial-gradient(44vmax 36vmax at 10% -12%, var(--aura-1), transparent 70%),
    radial-gradient(46vmax 38vmax at 98% 14%, var(--aura-2), transparent 70%),
    radial-gradient(44vmax 40vmax at 57% 122%, var(--aura-3), transparent 70%),
    var(--bg) !important;
  background-attachment: fixed !important;
}
/* Liseré « flamme » en haut de page, comme la barre de lecture du tableau de bord. */
.stApp::after {
  content: ""; position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 1000000;
  background: var(--flamme); box-shadow: 0 0 12px var(--accent); pointer-events: none;
}
[data-testid="stDecoration"] { display: none; }
/* Barre du haut de Streamlit : transparente en haut de page, le bandeau passe dessous comme
   en haut du tableau de bord ; elle se voile dès qu'on fait défiler la page (sans animation
   liée au défilement, dans Firefox, elle reste transparente). Pas de !important sur le fond :
   il l'emporterait sur l'animation. */
.stApp { timeline-scope: --gr-defile; }
.stApp section.stMain { scroll-timeline: --gr-defile block; }
.stApp [data-testid="stHeader"] { background: transparent; }
@supports (animation-timeline: scroll()) {
  .stApp [data-testid="stHeader"] {
    animation: gr-voile linear both; animation-timeline: --gr-defile;
    animation-range: 0 140px;
  }
  @keyframes gr-voile {
    from {
      background-color: rgba(var(--voile), 0);
      backdrop-filter: blur(0); -webkit-backdrop-filter: blur(0);
    }
    to {
      background-color: rgba(var(--voile), .72);
      backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    }
  }
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"] > div {
  background: transparent !important;
}
/* Colonne principale : marges du tableau de bord. */
[data-testid="stMainBlockContainer"], .block-container {
  padding-top: 4.25rem !important; padding-inline: var(--marge) !important;
}
/* Bandeau : posé sur la page comme celui du tableau de bord, pas dans un cadre. Il va d'un
   bord à l'autre de la colonne et remonte jusqu'en haut de la page, sous la barre de
   Streamlit (transparente), avec le fond, les lueurs et le filet du bas du tableau de bord.
   Le composant n'y met que le texte, la grenouille et le spectrogramme, sur fond transparent,
   aux marges de la page. */
.stMain [data-testid="stElementContainer"]:has(iframe[title*="blanci_bandeau"]) {
  position: relative; isolation: isolate; overflow: hidden; box-sizing: border-box;
  width: calc(100% + 2 * var(--marge)) !important; max-width: none !important;
  margin: -4.25rem calc(-1 * var(--marge)) 6px !important;
  padding: 3.5rem var(--marge) 0;
  background: var(--fond-bande);
  border-bottom: 1px solid var(--nuit-ligne);
}
.stMain [data-testid="stElementContainer"]:has(iframe[title*="blanci_bandeau"])::before {
  content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
  background: radial-gradient(50% 80% at 92% 10%, var(--halo-1), transparent 70%),
    radial-gradient(40% 60% at 70% 100%, var(--halo-2), transparent 70%);
}
/* Feuille de style et petit script des menus : hors du flux, sans laisser d'espace. */
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style),
[data-testid="stElementContainer"]:has(iframe[srcdoc*="__blanciMenus"]) {
  position: absolute !important; width: 1px; height: 1px; overflow: hidden; opacity: 0;
  pointer-events: none;
}
.stApp ::selection { background: var(--accent); color: var(--accent-ink); }
.stApp * { scrollbar-width: thin; scrollbar-color: var(--line-fort) transparent; }
.stApp a { color: var(--accent); text-underline-offset: 3px; }

/* Textes */
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp td, .stApp th, .stApp summary { font-family: var(--f-body); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: var(--f-display) !important; font-weight: 500 !important;
  letter-spacing: -.01em; text-wrap: balance;
}
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: var(--muted) !important;
}
.stApp :not(pre) > code {
  font-family: var(--f-mono); font-size: .84em; color: var(--ink); background: var(--surface-2);
  border: 1px solid var(--line); padding: 1px 6px; border-radius: 6px;
}

/* Titre du candidat : dégradé du titre du bandeau ; sa légende en chasse fixe. */
[data-testid="stMain"] h3:not([data-testid="stExpander"] h3) {
  font-size: clamp(19px, 2.1vw, 25px) !important; letter-spacing: -.02em; line-height: 1.2;
  width: fit-content; max-width: 100%; padding: 4px 0 2px;
  background: var(--titre);
  -webkit-background-clip: text; background-clip: text; color: transparent !important;
  filter: var(--titre-ombre);
}
[data-testid="stElementContainer"]:has(h3)
  + [data-testid="stElementContainer"] [data-testid="stCaptionContainer"] p {
  font-family: var(--f-mono); font-size: 11.5px; line-height: 1.5;
}
/* Enregistrement en cours d'écoute : point rouge « on enregistre » devant son nom. */
[data-testid="stMain"] :is(h3#en-ecoute, #en-ecoute h3)::before {
  content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin: 0 13px 0 3px; vertical-align: .16em; background: #FF5A4E;
  box-shadow: 0 0 0 0 rgba(255, 90, 78, .6);
}

/* Marque (haut du panneau de gauche). */
.gr-brand { padding: 0 2px; }
.gr-brand b {
  font-family: var(--f-display); font-size: 20px; font-weight: 700; letter-spacing: -.02em;
  color: var(--ink); display: flex; align-items: center; gap: 10px;
}
.gr-brand b::before {
  content: ""; width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
  background: radial-gradient(circle at 35% 35%, var(--gold), var(--accent) 55%, var(--accent-2));
  box-shadow: 0 0 14px var(--accent), 0 0 2px var(--gold);
}
@media (prefers-reduced-motion: no-preference) {
  [data-testid="stMain"] :is(h3#en-ecoute, #en-ecoute h3)::before {
    animation: gr-enregistre 1.8s ease-out infinite;
  }
  @keyframes gr-enregistre {
    0% { box-shadow: 0 0 0 0 rgba(255, 90, 78, .6); }
    80%, 100% { box-shadow: 0 0 0 9px rgba(255, 90, 78, 0); }
  }
}

/* Panneau de gauche : comme la barre latérale du tableau de bord. */
[data-testid="stSidebar"] {
  background: var(--fond-side) !important;
  border-right: 1px solid var(--nuit-ligne) !important;
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"], [data-testid="stSidebarHeader"] {
  background: transparent !important;
}
/* Titres de section (Session, Sélection, Écoute, Méthode) : surtitres ambrés, soulignés d'un
   filet qui part de l'accent, comme les en-têtes de section. */
[data-testid="stSidebar"] h2:not([data-testid="stExpander"] h2) {
  position: relative; font-family: var(--f-mono) !important; font-size: 14px !important;
  font-weight: 500 !important; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent) !important; padding: var(--avant-section) 0 8px !important;
  margin: 0 !important; line-height: 1.2 !important;
}
[data-testid="stSidebar"] h2:not([data-testid="stExpander"] h2)::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  border-radius: 1px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2) 24%, var(--line) 56%,
    transparent);
}
/* Panneau de gauche : un seul écart entre deux éléments (`--pas`), quels qu'ils soient, et
   `--pas` + `--avant-section` de part et d'autre d'un titre de section (au-dessus de son
   texte, au-dessous de son filet) : le contenu est centré entre son titre et le suivant.
   Streamlit donne aux textes une marge de 16 px qu'il reprend par une marge négative de leur
   cadre : un titre de section (sans marge) mangeait ainsi l'écart qui le suit. */
[data-testid="stSidebarUserContent"] { --pas: 12px; --avant-section: 8px; }
[data-testid="stSidebarUserContent"]
  [data-testid="stElementContainer"]:has(> [data-testid="stHeading"]) {
  margin-bottom: var(--avant-section);
}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: var(--pas) !important; }
[data-testid="stSidebarUserContent"] [data-testid="stMarkdownContainer"] {
  margin-bottom: 0 !important;
}
[data-testid="stSidebarUserContent"] [data-testid="stMarkdownContainer"] p:last-child {
  margin-bottom: 0 !important;
}
[data-testid="stSidebarUserContent"] [data-testid="stMarkdownContainer"] p { line-height: 1.4; }
[data-testid="stSidebarUserContent"] [data-testid="stCheckbox"] label { min-height: 0 !important; }
/* La case centrée sur la première ligne de son texte (Streamlit la décale de 2,5 px). */
[data-testid="stSidebarUserContent"] [data-testid="stCheckbox"] label > div:not([data-testid]) {
  margin-top: 0 !important;
}
[data-testid="stSidebarUserContent"] [data-testid="stCheckbox"] {
  margin-top: 0 !important; min-height: 0 !important;
}
[data-testid="stSidebarUserContent"]
  [data-testid="stWidgetLabel"]:not([data-testid="stCheckbox"] *) {
  min-height: 0 !important; margin-bottom: 6px !important;
}
[data-testid="stSidebarUserContent"] [data-testid="stWidgetLabel"] p {
  line-height: 1.25 !important;
}

/* Panneau de gauche resserré : la marque monte en haut, à hauteur du bouton qui replie le
   panneau (posé en surimpression à droite), et le bas ne garde qu'une petite marge. La ligne
   de la marque laisse la place de ce bouton. Le premier titre de section suit la marque sans
   grand blanc. */
[data-testid="stSidebarHeader"] {
  height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important;
  position: relative; z-index: 3; overflow: visible !important;
}
[data-testid="stSidebarHeader"] [data-testid="stLogoSpacer"] { display: none !important; }
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"] {
  position: absolute; top: 16px; right: 10px;
}
[data-testid="stSidebarUserContent"] {
  padding-top: 14px !important; padding-bottom: 18px !important;
}
/* Mode d'emploi à mi-chemin entre la marque et la première section (« Session »). */
[data-testid="stSidebarUserContent"] .st-key-marque { gap: 18px !important; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]
  > :is(.st-key-marque, :has(.st-key-marque)) + * h2 { padding-top: 6px !important; }
.st-key-marque [data-testid="stLayoutWrapper"] { width: 100% !important; }
.st-key-marque .gr-brand { padding-right: 40px; }
/* Bloc « commentaire + Valider » : carte en verre dépoli, comme l'ancien formulaire. */
.stMain [data-testid="stVerticalBlockBorderWrapper"]:has(input[aria-label="Commentaire"]) {
  background: var(--surface) !important; border: 1px solid var(--line) !important;
  border-radius: 14px !important; box-shadow: var(--ombre);
  backdrop-filter: blur(14px) saturate(1.2); -webkit-backdrop-filter: blur(14px) saturate(1.2);
}

/* Étiquettes des champs : chasse fixe, capitales espacées, comme les « champs » du tableau. */
[data-testid="stWidgetLabel"] p {
  font-family: var(--f-mono) !important; font-size: 10.5px !important; letter-spacing: .1em;
  text-transform: uppercase; color: var(--muted) !important; font-weight: 500;
}
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label p {
  font-family: var(--f-body) !important; font-size: 14px !important; letter-spacing: 0;
  text-transform: none; color: var(--ink) !important;
}

/* Champs et menus */
.stApp [data-baseweb="input"], .stApp [data-baseweb="textarea"],
.stApp [data-baseweb="select"] > div, .stApp [data-testid="stNumberInputContainer"] {
  background: var(--surface-solid) !important; border-color: var(--line-fort) !important;
  border-radius: 10px !important; transition: border-color .2s, box-shadow .2s;
}
.stApp [data-baseweb="input"]:hover, .stApp [data-baseweb="textarea"]:hover,
.stApp [data-baseweb="select"] > div:hover, .stApp [data-testid="stNumberInputContainer"]:hover {
  border-color: var(--accent) !important;
}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within,
.stApp [data-baseweb="select"] > div:focus-within,
.stApp [data-testid="stNumberInputContainer"]:focus-within {
  border-color: var(--accent) !important; box-shadow: 0 0 0 4px var(--gold-soft) !important;
}
.stApp [data-baseweb="input"] > div, .stApp [data-testid="stNumberInputContainer"] > div {
  background: transparent !important;
}
.stApp [data-testid="stNumberInputContainer"] [data-baseweb="input"] {
  border: 0 !important; box-shadow: none !important;
}
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] { color: var(--ink) !important; }
/* Menus : la main partout, pas le curseur de texte (on y choisit, on n'y tape rien). */
.stApp [data-baseweb="select"], .stApp [data-baseweb="select"] *,
.stApp [data-testid="stSelectbox"] .react-aria-ComboBox,
.stApp [data-testid="stSelectbox"] .react-aria-ComboBox * { cursor: pointer !important; }
.stApp input:disabled { color: var(--muted) !important; -webkit-text-fill-color: var(--muted); }
[data-baseweb="popover"] [role="listbox"] {
  background: var(--surface-solid) !important; border: 1px solid var(--line-fort);
  border-radius: 10px; box-shadow: var(--ombre-menu);
}
[data-baseweb="popover"] [role="option"] { font-family: var(--f-body); }
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
  background: linear-gradient(90deg, var(--gold-soft), transparent 90%) !important;
}

/* Boutons : l'action principale en dégradé braise, les autres en pastilles. */
.stApp button[kind^="primary"], .stApp [data-testid^="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color: var(--accent-ink) !important; border: 0 !important; font-weight: 600;
  box-shadow: 0 4px 16px -6px var(--accent-2);
  transition: box-shadow .2s, filter .2s, transform .2s;
}
.stApp button[kind^="primary"] p, .stApp [data-testid^="stBaseButton-primary"] p {
  color: inherit !important; font-weight: 600;
}
.stApp button[kind^="primary"]:hover:not(:disabled),
.stApp [data-testid^="stBaseButton-primary"]:hover:not(:disabled) {
  filter: brightness(1.08); transform: translateY(-1px);
  box-shadow: 0 8px 26px -8px var(--accent-2), 0 0 22px -8px var(--accent);
}
.stApp button[kind^="secondary"], .stApp [data-testid^="stBaseButton-secondary"] {
  background: var(--surface-2) !important; color: var(--ink) !important;
  border: 1px solid var(--line-fort) !important; border-radius: 999px !important;
  transition: color .2s, border-color .2s, box-shadow .2s;
}
.stApp button[kind^="secondary"]:hover:not(:disabled),
.stApp [data-testid^="stBaseButton-secondary"]:hover:not(:disabled) {
  color: var(--accent) !important; border-color: var(--accent) !important;
  box-shadow: 0 0 16px -4px var(--accent);
}
.stApp button[kind^="secondary"]:hover:not(:disabled) p,
.stApp [data-testid^="stBaseButton-secondary"]:hover:not(:disabled) p {
  color: var(--accent) !important;
}
.stApp button:disabled { opacity: .4; }
/* Mode d'emploi : bouton sur toute la largeur du panneau. Sa fenêtre s'ouvre comme une
   fenêtre surgissante, en haut de la page et non sous le bouton (elle sortait par le bas de
   l'écran) : à droite du panneau, qu'elle ne recouvre pas (le bouton reste cliquable pour la
   fermer), assez large pour le tableau des gestes, et défile toute seule si l'écran est trop
   bas. Un clic à côté ou Échap la ferme.
   Streamlit la place par des styles en ligne (position, décalage), d'où les !important. */
[data-testid="stSidebar"] [data-testid="stPopover"],
[data-testid="stSidebar"] [data-testid="stPopover"] > div,
[data-testid="stSidebar"] [data-testid="stPopover"] button { width: 100% !important; }
[data-testid="stSidebar"] [data-testid="stPopover"] button { justify-content: flex-start; }
[data-testid="stPopoverBody"] {
  position: fixed !important; top: 12px !important;
  left: calc(var(--panneau, 22rem) + 12px) !important;
  right: auto !important; bottom: auto !important; transform: none !important;
  margin: 0 !important; z-index: 1000001 !important;
  width: min(800px, calc(100vw - var(--panneau, 22rem) - 24px)) !important;
  max-width: none !important; max-height: calc(100dvh - 24px) !important;
  overflow-y: auto !important; padding: 18px 22px !important;
  background: var(--fond-fenetre) !important; border: 1px solid var(--line-fort) !important;
  border-radius: 14px !important;
  box-shadow: var(--ombre-fenetre), 0 0 50px -24px var(--accent) !important;
}
.stApp button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 3px; }
/* « Extrait précédent » et « Extrait suivant » : chacun une moitié de la largeur, en forme de
   flèche (pointe à gauche, pointe à droite) qu'on lit sans lire le texte ; la teinte monte vers
   la pointe. */
.stApp :is(.st-key-nav_prev, .st-key-nav_next) button {
  width: 100%; min-height: 56px; border: 0 !important; border-radius: 14px !important;
  font-family: var(--f-display) !important; color: var(--ink) !important;
  transition: background .2s, color .2s, filter .2s !important; box-shadow: none !important;
}
.stApp :is(.st-key-nav_prev, .st-key-nav_next) button p {
  font-family: var(--f-display) !important; font-size: 15px !important; font-weight: 500;
  letter-spacing: -.005em; color: inherit !important;
}
.stApp .st-key-nav_prev button {
  clip-path: polygon(34px 0, 100% 0, 100% 100%, 34px 100%, 0 50%);
  border-radius: 0 14px 14px 0 !important; justify-content: flex-start !important;
  padding-left: 52px !important;
  background: linear-gradient(90deg, var(--gold-soft), var(--surface-2) 70%) !important;
}
.stApp .st-key-nav_next button {
  clip-path: polygon(0 0, calc(100% - 34px) 0, 100% 50%, calc(100% - 34px) 100%, 0 100%);
  border-radius: 14px 0 0 14px !important; justify-content: flex-end !important;
  padding-right: 52px !important;
  background: linear-gradient(270deg, var(--gold-soft), var(--surface-2) 70%) !important;
}
.stApp :is(.st-key-nav_prev, .st-key-nav_next) button:hover:not(:disabled) {
  color: var(--accent-ink) !important; transform: none;
}
.stApp .st-key-nav_prev button:hover:not(:disabled) {
  background: linear-gradient(90deg, var(--accent), var(--accent-2) 80%) !important;
}
.stApp .st-key-nav_next button:hover:not(:disabled) {
  background: linear-gradient(270deg, var(--accent), var(--accent-2) 80%) !important;
}
.stApp :is(.st-key-nav_prev, .st-key-nav_next) button:hover:not(:disabled) p {
  color: var(--accent-ink) !important;
}

/* Cartes en verre dépoli : formulaire de réponse, rubriques repliables. */
[data-testid="stForm"] {
  background: var(--surface) !important; border: 1px solid var(--line) !important;
  border-radius: 14px !important; box-shadow: var(--ombre);
  backdrop-filter: blur(14px) saturate(1.2); -webkit-backdrop-filter: blur(14px) saturate(1.2);
  transition: border-color .3s;
}
[data-testid="stForm"]:hover { border-color: var(--line-fort) !important; }
[data-testid="stForm"] [data-testid="stMarkdownContainer"] strong {
  font-family: var(--f-display); font-weight: 500; letter-spacing: -.005em;
}
[data-testid="stExpander"] details {
  background: var(--surface) !important; border: 1px solid var(--line) !important;
  border-radius: 12px !important; transition: border-color .2s;
}
[data-testid="stExpander"] details:hover, [data-testid="stExpander"] details[open] {
  border-color: var(--line-fort) !important;
}
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover p {
  color: var(--accent) !important;
}
[data-testid="stExpander"] summary p { font-weight: 500; }

/* Messages : liseré coloré à gauche, fond à peine teinté, comme les consignes du tableau. */
[data-testid="stAlertContainer"], .stApp [data-baseweb="notification"] {
  --ton: var(--gold); --ton-doux: var(--gold-soft);
  border-radius: 0 12px 12px 0 !important; border-left: 3px solid var(--ton) !important;
  background: linear-gradient(100deg, var(--ton-doux), transparent) !important;
  color: var(--ink) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]),
[data-baseweb="notification"]:has([data-testid="stAlertContentSuccess"]) {
  --ton: var(--ok); --ton-doux: var(--ok-soft);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]),
[data-baseweb="notification"]:has([data-testid="stAlertContentWarning"]) {
  --ton: var(--warn); --ton-doux: var(--warn-soft);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
[data-baseweb="notification"]:has([data-testid="stAlertContentError"]) {
  --ton: var(--bad); --ton-doux: var(--bad-soft);
}
[data-testid="stAlertContainer"] p, .stApp [data-baseweb="notification"] p {
  color: var(--ink) !important;
}

/* Tableaux (mode d'emploi) : en-têtes en capitales, lignes séparées d'un filet. */
[data-testid="stMarkdownContainer"] table {
  border-collapse: collapse; width: 100%; font-size: 13.5px; border: 0 !important;
}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {
  border: 0 !important; border-bottom: 1px solid var(--line) !important;
  padding: 7px 10px !important;
  text-align: left; vertical-align: top; background: transparent !important;
}
[data-testid="stMarkdownContainer"] th {
  font-family: var(--f-mono); font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted); font-weight: 500;
}
[data-testid="stMarkdownContainer"] tbody tr:hover {
  background: linear-gradient(90deg, var(--gold-soft), transparent 80%);
}
[data-testid="stMarkdownContainer"] strong { color: var(--ink); }

/* Composants : le bandeau, sans cadre (voir plus haut), et le visualiseur, encadré comme le
   spectrogramme du bandeau (liseré or, lueur braise). */
.stApp iframe[data-testid="stCustomComponentV1"] { border-radius: 14px; }
.stApp iframe[title*="blanci_bandeau"] {
  width: 100% !important; border-radius: 0; vertical-align: top;
}
.stApp iframe[title*="blanci_viewer"] {
  background: var(--bg-2);
  box-shadow: 0 0 0 1px rgba(255, 210, 120, .14), 0 30px 80px -30px rgba(228, 87, 63, .5),
    0 0 60px -24px rgba(245, 165, 58, .3);
}

</style>""".replace("__FONTS__", FONTS)
    .replace("__GRAIN__", _GRAIN)
    .replace("__VARS_DARK__", declarations(BRAISE_DARK))
    .replace("__VARS_LIGHT__", declarations(BRAISE_LIGHT, "    "))
)

# Mode choisi : le texte de Streamlit suit son propre thème, pas nos variables ; on le force.
_TEXT = (
    ".stApp, .stApp :is(p, li, label, span, h1, h2, h3, h4, td, th, summary,"
    " [data-testid='stMarkdownContainer'], [data-testid='stWidgetLabel']) { color: var(--ink); }"
    # Champs : Streamlit les peint avec les couleurs de son thème, pas avec nos variables.
    " .stApp :is([data-testid='stSelectbox'] .react-aria-ComboBox > div,"
    " [data-testid='stTextInputRootElement'], [data-testid='stNumberInputContainer'],"
    " [data-testid='stFileUploaderDropzone']) { background: var(--surface-solid);"
    " border-color: var(--line-fort); }"
    " .stApp :is(input, textarea, [role='option'], [role='listbox']) { color: var(--ink); }"
    " .stApp :is(input, textarea)::placeholder { color: var(--muted); opacity: 1; }"
    " .stApp [data-testid='stSelectbox'] .react-aria-ComboBox { color: var(--ink); }"
    # Case non cochée : peinte par Streamlit aux couleurs de son thème (noire en mode jour quand
    # le système est en sombre) ; grisée ici. Cochée, elle garde la couleur d'accent.
    " .stApp [data-testid='stCheckbox'] label:not(:has(input:checked))"
    " > div:not([data-testid]) { background: var(--surface-2); border-color: var(--line-fort); }"
)


def css(theme: str | None, style: str) -> str:
    """`CSS` plus le choix de l'utilisateur. Un mode choisi (`theme`) l'emporte sur le système
    et sur le menu de Streamlit (`color-scheme` et texte forcés) ; sans choix, le système
    décide et seul le style est posé, dans les deux modes. `:root` répété : passer devant les
    règles `data-gr-theme`."""
    if theme is None and style == "braise":
        return CSS
    top = "html:root:root:root"
    if theme in ("light", "dark"):
        extra = [
            f"{top} {{ color-scheme: {theme};",
            declarations(palette(theme, style)),
            "}",
            _TEXT,
        ]
    else:
        extra = [
            f"{top} {{",
            declarations(palette("dark", style)),
            "}",
            f'@media (prefers-color-scheme: light) {{ {top}:not([data-gr-theme="dark"]) {{',
            declarations(palette("light", style)),
            "} }",
            f'{top}[data-gr-theme="light"] {{',
            declarations(palette("light", style)),
            "}",
        ]
    return CSS.replace("</style>", "\n".join(extra) + "\n</style>")


BRAND = '<div class="gr-brand"><b>Grenouille</b></div>'
