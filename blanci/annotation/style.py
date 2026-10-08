"""Habillage du poste d'annotation, repris du tableau de bord
(`documentation/tableau-de-bord/modele.html`) : « nuit guyanaise », fond de sous-bois la
nuit, lueurs braise et ambre, panneaux en verre dépoli, titres en Unbounded, texte en Hanken
Grotesk, étiquettes en JetBrains Mono. Sombre seulement : le spectrogramme l'est.

Deux étages :
- `THEME` : options `[theme]` de Streamlit (couleurs des cases, curseurs, menus, champs),
  passées par `blanci annotate` en variables d'environnement (`theme_env`). Une option que la
  version installée ne connaît pas est simplement ignorée, là où une option inconnue sur la
  ligne de commande ferait échouer le lancement.
- `CSS` : feuille injectée par `app.py` à chaque affichage, pour ce que le thème ne règle pas
  (fond, panneau de gauche, boutons, cartes, titres, cadre du visualiseur).

Pas d'animation de fond ni de grain en surimpression : c'est un poste de travail, le
navigateur garde ses forces pour le spectrogramme.
"""

from __future__ import annotations

import re

# Couleurs du tableau de bord (thème sombre).
BG = "#050B0A"
SURFACE = "#0F1C18"
INK = "#E9F2EC"
MUTED = "#93A89D"
ACCENT = "#F5A53A"
LINE = "#1B2722"  # rgba(160, 210, 185, .14) sur le fond
LINE_STRONG = "#30433B"  # rgba(160, 210, 185, .28) sur le fond

THEME = {
    "base": "dark",
    "primaryColor": ACCENT,
    "backgroundColor": BG,
    "secondaryBackgroundColor": SURFACE,
    "textColor": INK,
    "linkColor": ACCENT,
    "codeBackgroundColor": SURFACE,
    "borderColor": LINE_STRONG,
    "dataframeBorderColor": LINE,
    "showWidgetBorder": "true",
    "showSidebarBorder": "true",
    "baseRadius": "10px",
    # Sans guillemets : selon les versions, Streamlit les ajoute lui-même.
    "font": "Hanken Grotesk, Segoe UI, system-ui, sans-serif",
    "headingFont": "Unbounded, Segoe UI, system-ui, sans-serif",
    "codeFont": "JetBrains Mono, ui-monospace, Menlo, monospace",
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

# Couleurs des classes sur la carte des embeddings (espèces du tableau de bord) ; les fenêtres
# jamais écoutées restent grises.
CHART_COLORS = (
    "#5AA2F0",
    "#3FC57D",
    "#F29A3B",
    "#B08CF2",
    "#F06A5B",
    "#4FD8C0",
    "#FFE08A",
    "#E4573F",
)
UNHEARD_COLOR = "#4A5A52"


def theme_env() -> dict[str, str]:
    """`THEME` en variables d'environnement de Streamlit : theme.primaryColor →
    STREAMLIT_THEME_PRIMARY_COLOR, theme.sidebar.backgroundColor →
    STREAMLIT_THEME_SIDEBAR_BACKGROUND_COLOR."""
    return {
        "STREAMLIT_THEME_"
        + re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).replace(".", "_").upper(): value
        for key, value in THEME.items()
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
  --bg: #050B0A; --bg-2: #08120F; --surface: rgba(16, 29, 25, .72); --surface-solid: #0F1C18;
  --surface-2: rgba(255, 255, 255, .055); --ink: #E9F2EC; --muted: #93A89D;
  --line: rgba(160, 210, 185, .14); --line-fort: rgba(160, 210, 185, .28);
  --accent: #F5A53A; --accent-2: #E4573F; --accent-3: #9E2A6B; --accent-ink: #1A0B03;
  --gold: #FFE08A; --gold-soft: rgba(255, 210, 110, .12); --teal: #4FD8C0;
  --ok: #5FD394; --ok-soft: rgba(95, 211, 148, .13); --warn: #F5B547;
  --warn-soft: rgba(245, 181, 71, .13); --bad: #F07A63; --bad-soft: rgba(240, 122, 99, .14);
  --nuit-ligne: rgba(160, 210, 185, .16);
  --ombre: 0 20px 50px -24px rgba(0, 0, 0, .8), 0 1px 0 rgba(255, 255, 255, .04) inset;
  --f-display: "Unbounded", "Segoe UI", system-ui, sans-serif;
  --f-serif: "Instrument Serif", Georgia, serif;
  --f-body: "Hanken Grotesk", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --flamme: linear-gradient(100deg, var(--gold) 0%, var(--accent) 38%, var(--accent-2) 70%,
    var(--accent-3) 100%);
}

/* Fond : trois lueurs fixes (braise, sarcelle, pourpre) et le grain, sans animation. */
.stApp {
  color-scheme: dark; color: var(--ink); font-family: var(--f-body);
  -webkit-font-smoothing: antialiased;
  background:
    __GRAIN__,
    radial-gradient(38vmax 30vmax at 18% 8%, rgba(228, 87, 63, .13), transparent 70%),
    radial-gradient(42vmax 34vmax at 92% 30%, rgba(79, 216, 192, .08), transparent 70%),
    radial-gradient(40vmax 36vmax at 55% 105%, rgba(158, 42, 107, .13), transparent 70%),
    var(--bg) !important;
  background-attachment: fixed !important;
}
/* Liseré « flamme » en haut de page, comme la barre de lecture du tableau de bord. */
.stApp::after {
  content: ""; position: fixed; top: 0; left: 0; right: 0; height: 2px; z-index: 1000000;
  background: var(--flamme); box-shadow: 0 0 12px var(--accent); pointer-events: none;
}
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] {
  background: rgba(5, 11, 10, .55) !important;
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"] > div {
  background: transparent !important;
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

/* Titre du candidat : dégradé blanc → or → ambre du bandeau ; sa légende en chasse fixe. */
[data-testid="stMain"] h3:not([data-testid="stExpander"] h3) {
  font-size: clamp(19px, 2.1vw, 25px) !important; letter-spacing: -.02em; line-height: 1.2;
  width: fit-content; max-width: 100%; padding: 4px 0 2px;
  background: linear-gradient(100deg, #FFFFFF 0%, #FFF3D6 40%, #FFD27A 70%, #F5A53A 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent !important;
  filter: drop-shadow(0 2px 10px rgba(4, 8, 7, .9));
}
[data-testid="stElementContainer"]:has(h3)
  + [data-testid="stElementContainer"] [data-testid="stCaptionContainer"] p {
  font-family: var(--f-mono); font-size: 11.5px; line-height: 1.5;
}

/* Surtitre (au-dessus du titre) et marque (haut du panneau de gauche). */
[data-testid="stElementContainer"]:has(.gr-eyebrow) { margin-bottom: -10px; }
.gr-eyebrow {
  display: inline-flex; align-items: center; gap: 10px;
  font-family: var(--f-mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--gold); font-weight: 500;
}
.gr-eyebrow::before {
  content: ""; width: 8px; height: 8px; border-radius: 50%; background: #FF5A4E;
  box-shadow: 0 0 0 0 rgba(255, 90, 78, .6);
}
.gr-eyebrow b { color: var(--muted); font-weight: 500; }
.gr-brand { display: flex; flex-direction: column; gap: 2px; padding: 0 2px 4px; }
.gr-brand b {
  font-family: var(--f-display); font-size: 20px; font-weight: 700; letter-spacing: -.02em;
  color: var(--ink); display: flex; align-items: center; gap: 10px;
}
.gr-brand b::before {
  content: ""; width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
  background: radial-gradient(circle at 35% 35%, var(--gold), var(--accent) 55%, var(--accent-2));
  box-shadow: 0 0 14px var(--accent), 0 0 2px var(--gold);
}
.gr-brand span { color: var(--muted); font-size: 12.5px; }
.gr-brand i { font-family: var(--f-serif); font-size: 1.15em; color: var(--gold); }
@media (prefers-reduced-motion: no-preference) {
  .gr-eyebrow::before { animation: gr-enregistre 1.8s ease-out infinite; }
  @keyframes gr-enregistre {
    0% { box-shadow: 0 0 0 0 rgba(255, 90, 78, .6); }
    80%, 100% { box-shadow: 0 0 0 9px rgba(255, 90, 78, 0); }
  }
}

/* Panneau de gauche : nuit, comme la barre latérale du tableau de bord. */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(6, 14, 12, .97), rgba(4, 9, 8, .99)) !important;
  border-right: 1px solid var(--nuit-ligne) !important;
}
[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"], [data-testid="stSidebarHeader"] {
  background: transparent !important;
}
/* Titres de section (Session, Sélection, Écoute, Méthode) : surtitres ambrés, soulignés d'un
   filet qui part de l'accent, comme les en-têtes de section. */
[data-testid="stSidebar"] h2:not([data-testid="stExpander"] h2) {
  position: relative; font-family: var(--f-mono) !important; font-size: 11px !important;
  font-weight: 500 !important; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent) !important; padding: 18px 0 9px !important; margin: 0 !important;
}
[data-testid="stSidebar"] h2:not([data-testid="stExpander"] h2)::after {
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 1px;
  background: linear-gradient(90deg, var(--accent), var(--accent-2) 18%, var(--line) 42%,
    transparent);
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
.stApp input:disabled { color: var(--muted) !important; -webkit-text-fill-color: var(--muted); }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {
  background: var(--surface-solid) !important; border: 1px solid var(--line-fort);
  border-radius: 10px; box-shadow: 0 20px 50px -20px rgba(0, 0, 0, .9);
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
.stApp button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 3px; }

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

/* Visualiseur : cadre du spectrogramme du bandeau (liseré or, lueur braise). */
.stApp :is(iframe[title*="blanci_viewer"], iframe[data-testid="stCustomComponentV1"]) {
  border-radius: 14px; background: var(--bg-2);
  box-shadow: 0 0 0 1px rgba(255, 210, 120, .14), 0 30px 80px -30px rgba(228, 87, 63, .5),
    0 0 60px -24px rgba(245, 165, 58, .3);
}

</style>"""
    .replace("__FONTS__", FONTS)
    .replace("__GRAIN__", _GRAIN)
)


def eyebrow(text: str, detail: str = "") -> str:
    """Surtitre à point rouge (« on enregistre ») ; `detail` en gris après un point médian."""
    from html import escape

    tail = f" <b>· {escape(detail)}</b>" if detail else ""
    return f'<div class="gr-eyebrow">{escape(text)}{tail}</div>'


BRAND = (
    '<div class="gr-brand"><b>Grenouille</b>'
    "<span>Poste d'annotation · <i>A. blanci</i></span></div>"
)
