# ruff: noqa: E501
"""Présentation n° 5 v2 (annotation), charte de la présentation n° 4 « ultimate » : fond blanc,
pastille de rubrique en haut à gauche, titre Cambria 21, badges CODE en haut à droite, cartes
claires à liseré, intercalaires sombres entre les cinq parties. Notes orateur dans chaque diapo.

Figures : `presentation_5_figures.py` (dans `documentation/prez/figures/annotation/`), plus la
capture du poste et les deux pistes reprises de la version de travail de la n° 5.

    uv run --with python-pptx python documentation/prez/scripts/presentation_5_v2.py

Écrit `documentation/prez/presentation_5_annotation_v2.pptx` : ne jamais relancer par-dessus une
version que Léonard a retouchée à la main (changer SORTIE)."""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "documentation" / "prez" / "figures"
ANN = FIG / "annotation"
SORTIE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "documentation" / "prez" / "presentation_5_annotation_v2.pptx"

DARK, DARK2, INK, MUTED, SAGE = "0E2A24", "173D34", "1B2A24", "5B6B63", "A7BFB2"
PALE, PALER, GOLD, TEAL, TERRA, GOLD_DARK = "E4EEE8", "F3F7F4", "E9B44C", "2A7F72", "C8553D", "9A6B0F"
VERT, ORANGE, BLEU, ROUGE, VIOLET = "2E8B57", "D9822B", "3E7CB1", "C0492F", "8A6BBE"
PARTIES = ["Méthode du tirage", "Le poste", "Les coulisses", "Le résultat", "Doutes et erreurs"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
PAGE = [0]


def rgb(h):
    return RGBColor.from_string(h)


def nofill_line(sh):
    sh.line.fill.background()
    sh.shadow.inherit = False


def box(s, x, y, w, h, fill, radius=None, line=None, shape=None):
    kind = shape or (MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE)
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(0.75)
        sh.shadow.inherit = False
    else:
        nofill_line(sh)
    sh.text_frame.text = ""
    return sh


def R(text, **kw):
    return (text, kw)


def P(*runs, size=13, color=INK, bold=False, after=4, align="l", font="Calibri", line=None):
    return dict(runs=runs, size=size, color=color, bold=bold, after=after, align=align, font=font, line=line)


def text(s, x, y, w, h, paras, anchor="t", name=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, (str, dict)):
        paras = [paras]
    for k, p in enumerate(paras):
        if isinstance(p, str):
            p = P(p)
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p["align"]]
        para.space_after = Pt(p["after"])
        if p["line"]:
            para.line_spacing = p["line"]
        for r in p["runs"]:
            t, kw = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = t
            f = run.font
            f.name = kw.get("font", p["font"])
            f.size = Pt(kw.get("size", p["size"]))
            f.bold = kw.get("bold", p["bold"])
            f.italic = kw.get("italic", False)
            f.color.rgb = rgb(kw.get("color", p["color"]))
    return tb


def notes(s, txt):
    s.notes_slide.notes_text_frame.text = txt.strip()


def page_number(s, dark=False):
    PAGE[0] += 1
    text(s, 12.33, 7.08, 0.5, 0.25, P(str(PAGE[0]), size=9, color=SAGE, align="r"))


def code(s, modules):
    """Badges CODE en haut à droite, comme la n° 4 : le premier module le plus à droite."""
    x = 12.73
    for m in modules:
        w = 0.094 * len(m) + 0.24
        x -= w
        box(s, x, 0.4, w, 0.32, "FFFFFF", radius=0.2, line=SAGE)
        text(s, x, 0.4, w, 0.32, P(m, size=10.5, color=DARK, bold=True, font="Consolas", align="c"), anchor="m")
        x -= 0.1
    text(s, x - 0.62, 0.4, 0.62, 0.32, P("CODE", size=9, color=MUTED, bold=True, align="r"), anchor="m")


def content(rubrique, titre, modules=()):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb("FFFFFF")
    w = 0.083 * len(rubrique) + 0.32
    box(s, 0.6, 0.4, w, 0.32, PALE)
    text(s, 0.6, 0.4, w, 0.32, P(rubrique, size=10, color=TEAL, bold=True, align="c"), anchor="m")
    text(s, 0.6, 0.78, 12.13, 0.85, P(titre, size=21, color=INK, bold=True, font="Cambria", line=0.95), name="Titre")
    if modules:
        code(s, modules)
    page_number(s)
    return s


def label(s, x, y, w, t, color=TEAL):
    text(s, x, y, w, 0.24, P(t.upper(), size=11, color=color, bold=True))


def card(s, x, y, w, h, paras, bar=TEAL, fill=PALER, pad=0.26):
    box(s, x, y, w, h, fill, radius=0.06)
    box(s, x, y + 0.09, 0.06, h - 0.18, bar)
    return text(s, x + pad, y + 0.12, w - pad - 0.18, h - 0.22, paras)


def tile(s, x, y, w, h, big, small, color=TEAL):
    box(s, x, y, w, h, PALER, radius=0.06)
    text(s, x + 0.2, y + 0.1, w - 0.4, 0.56, P(big, size=28, color=color, bold=True, font="Cambria"))
    text(s, x + 0.2, y + 0.68, w - 0.4, h - 0.75, P(small, size=11, color=MUTED))


def picture(s, path, x, y, w=None, h=None):
    return s.shapes.add_picture(str(path), Inches(x), Inches(y),
                                Inches(w) if w else None, Inches(h) if h else None)


def arrow(s, x1, y1, x2, y2, color=SAGE, width=1.5):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def table(s, x, y, w, rows, widths, size=12, header_fill=DARK, row_h=0.42, first_col_bold=True, fills=None):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    for j, cw in enumerate(widths):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = rgb(header_fill)
            elif fills and fills.get((i, j)):
                cell.fill.fore_color.rgb = rgb(fills[(i, j)])
            else:
                cell.fill.fore_color.rgb = rgb(PALER if i % 2 else "FFFFFF")
            tf = cell.text_frame
            tf.word_wrap = True
            t, kw = (val, {}) if isinstance(val, str) else val
            tf.text = ""
            run = tf.paragraphs[0].add_run()
            run.text = t
            f = run.font
            f.name = "Calibri"
            f.size = Pt(kw.get("size", size))
            f.bold = kw.get("bold", i == 0 or (j == 0 and first_col_bold))
            f.color.rgb = rgb(kw.get("color", "FFFFFF" if i == 0 else INK))
    return gt


def divider(n, titre, sous_titre, etapes):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(DARK)
    pastille = f"STAGE ONF GUYANE · SUIVI N° 5 · PARTIE {n} SUR 5"
    w = 0.09 * len(pastille) + 0.3
    box(s, 0.6, 1.25, w, 0.36, DARK2)
    text(s, 0.6, 1.25, w, 0.36, P(pastille, size=11, color="FFFFFF", bold=True, align="c"), anchor="m")
    text(s, 0.6, 1.9, 12.0, 1.0, P(f"Partie {n} · {titre}", size=42, color="FFFFFF", bold=True, font="Cambria"))
    text(s, 0.6, 3.05, 11.0, 0.9, P(sous_titre, size=20, color=SAGE))
    xs = [0.6 + k * 2.45 for k in range(5)]
    for k, (x, nom) in enumerate(zip(xs, PARTIES, strict=True)):
        cur = k + 1 == n
        box(s, x, 4.6, 2.3, 1.35, TEAL if cur else DARK2)
        etat = "MAINTENANT" if cur else ("FAIT" if k + 1 < n else "ENSUITE")
        text(s, x + 0.18, 4.72, 1.95, 1.15, [
            P(f"{k + 1} · {nom.upper()}", size=11, color="FFFFFF" if cur else SAGE, bold=True, after=2),
            P(etat, size=9.5, color=GOLD if cur else SAGE, bold=True, after=4),
            P(etapes[k], size=10.5, color="FFFFFF" if cur else SAGE),
        ])
    page_number(s)
    return s


ETAPES = [
    "Partition des points, trois niveaux, strates, lots et test",
    "Une capture ; démonstration en direct ; intervalles et fenêtres",
    "La chaîne, les files, la base et les CSV",
    "Le bilan, la prévalence, la répartition des positifs",
    "La piste 1, la validité de la suite, les limites",
]

# ======================================================================== 1 · titre
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = rgb(DARK)
picture(s, FIG / "titre.png", 8.6, 0, 4.733, 7.5)
box(s, 0.6, 1.25, 3.9, 0.36, DARK2)
text(s, 0.6, 1.25, 3.9, 0.36, P("STAGE ONF GUYANE · SUIVI N° 5", size=11, color="FFFFFF", bold=True, align="c"), anchor="m")
text(s, 0.6, 1.9, 7.8, 1.8, [P("Détection acoustique d'", size=42, color="FFFFFF", bold=True, font="Cambria", after=0),
                             P(R("Anomaloglossus blanci", italic=True), size=42, color="FFFFFF", bold=True, font="Cambria")])
text(s, 0.6, 3.85, 7.6, 0.5, P("Annotation du jeu de données", size=20, color=SAGE))
for k, nom in enumerate(PARTIES):
    y = 4.5 + k * 0.36
    box(s, 0.6, y + 0.07, 0.22, 0.22, TEAL if k % 2 == 0 else GOLD, radius=0.3)
    text(s, 0.95, y, 6.5, 0.36, P(f"{k + 1} · {nom}", size=14, color="FFFFFF"), anchor="m")
text(s, 0.6, 6.55, 6.0, 0.35, P("Léonard Laplace-Palette · date à fixer", size=13, color=SAGE))
text(s, 8.75, 7.05, 4.4, 0.3, P("Fond : un chant d'A. blanci enregistré à Mataroni (spectrogramme, 1–9 kHz)", size=9, color="FFFFFF"))
PAGE[0] += 1
notes(s, """
Aujourd'hui, l'annotation : comment on a choisi quoi écouter, l'outil avec lequel j'ai écouté, ce qui se passe derrière, ce que donne le jeu annoté, et ce qui reste à corriger.

Cinq parties : la méthode du tirage, le poste d'annotation (que je montrerai en direct), les coulisses, le résultat, puis les doutes et les erreurs.
""")

# ======================================================================== 2 · sommaire
s = content("SOMMAIRE", "Cinq parties : de la façon de tirer les extraits jusqu'aux erreurs à rattraper")
items = [
    ("Méthode du tirage", "diapos 4 à 9", "Cinq façons de séparer entraînement et test ; pourquoi le point d'écoute ; les trois niveaux de généralisation ; la stratification par site, point, saison et heure ; lot 1, lot 2 et test v1."),
    ("Le poste d'annotation", "diapos 10 à 14", "Une capture, puis le poste en direct. Comment un intervalle tracé devient des labels de fenêtres : le chant l'emporte, deux notes éloignées font deux intervalles, une fenêtre qui effleure est écartée."),
    ("Les coulisses", "diapos 15 à 19", "La chaîne de bout en bout ; d'où viennent les files d'écoute ; ce que fait « Valider » dans la base SQLite ; pourquoi aussi des CSV ; une vraie ligne d'arrivée."),
    ("Le résultat", "diapos 20 à 23", "Le bilan chiffré de l'annotation ; la prévalence, et pourquoi il faut l'estimer ; des positifs inégalement répartis : gênant ou pas, pour l'entraînement et pour le test."),
    ("Doutes et erreurs", "diapos 24 à 28", "Les positifs entendus sur la piste 1 et comment les rattraper ; pourquoi la suite ne vaut que si les annotations sont bonnes ; les limites ; la suite."),
]
for k, (nom, ou, quoi) in enumerate(items):
    y = 1.8 + k * 1.02
    box(s, 0.6, y, 12.13, 0.9, PALER, radius=0.08)
    box(s, 0.6, y + 0.09, 0.06, 0.72, TEAL if k % 2 == 0 else GOLD_DARK)
    text(s, 0.88, y + 0.1, 0.5, 0.7, P(str(k + 1), size=26, color=TEAL, bold=True, font="Cambria"), anchor="m")
    text(s, 1.45, y + 0.1, 2.9, 0.7, [P(nom, size=15, bold=True, after=0), P(ou, size=11, color=MUTED)], anchor="m")
    text(s, 4.4, y + 0.08, 8.1, 0.74, P(quoi, size=12.5), anchor="m")
notes(s, """
Le fil de la présentation. On commence par la méthode, parce que c'est elle qui décide de ce que valent les chiffres à la fin. Ensuite le poste : une capture ici, et je l'ouvrirai en direct. Puis les coulisses, pour ceux qui reprendront la chaîne. Ensuite le résultat, avec deux diapos plus délicates sur la prévalence et sur la répartition des positifs. On finit par les erreurs, dont l'écoute sur la piste 1.
""")

# ======================================================================== 3 · repartir de zéro
s = content("ANNOTATION", "On repart de zéro : tout se décide avant la première écoute", ["plan.py"])
regles = [
    ("1", "Séparer les points d'écoute avant d'écouter", "Le jeu de test est fait de points d'écoute que l'entraînement ne verra jamais. La partition est tirée une fois, à graine fixe, et versionnée (config/plan/partition_v1.csv)."),
    ("2", "Tirer par un plan, sans détecteur", "Ni BirdNET, ni nos têtes, ni aucun score ne choisit le premier lot : sinon on n'écoute que ce qu'un modèle trouve déjà facile, et le test mesure ce biais."),
    ("3", "Ne jamais effacer", "La base est en ajout seul : une correction ajoute une ligne, l'ancienne reste, datée. Pour l'entraînement, la dernière version d'un extrait compte."),
]
for k, (n, t, d) in enumerate(regles):
    x = 0.6 + k * 4.1
    box(s, x, 1.85, 3.93, 2.9, PALER, radius=0.06)
    text(s, x + 0.25, 1.98, 0.7, 0.7, P(n, size=34, color=TEAL, bold=True, font="Cambria"))
    text(s, x + 0.25, 2.72, 3.45, 1.95, [P(t, size=15, bold=True, after=6), P(d, size=12.5, color=INK)])
card(s, 0.6, 4.95, 12.13, 1.95, [
    P("Le travail de Biophonia : ignoré, ni repris ni exclu", size=14, bold=True, color=GOLD_DARK, after=6),
    P("Leurs détections et annotations ne servent ni à tirer, ni à étiqueter, ni à évaluer : aucun de mes labels n'en vient. Mais je ne les ai pas non plus retirées du tirage : un enregistrement qu'ils ont annoté peut sortir comme n'importe quel autre, et je l'écoute sans voir leur avis.", size=12.5, after=6),
    P("La nuance compte : exclure leurs enregistrements aurait biaisé le tirage (ils ont travaillé là où la grenouille chante) ; les reprendre aurait fait entrer leurs choix dans le test.", size=12.5, color=MUTED),
], bar=GOLD_DARK)
notes(s, """
Trois règles, posées avant d'écouter quoi que ce soit.

Un : on sépare les points d'écoute avant la première écoute. Le jeu de test, c'est des points d'écoute que l'entraînement ne verra jamais.

Deux : aucun détecteur ne choisit ce que j'écoute dans le premier lot. Sinon on écoute ce que le modèle trouve déjà, et le test mesure ce biais-là.

Trois : on n'efface jamais. Une correction, c'est une ligne de plus.

Et une nuance sur Biophonia : on ignore leur travail. On ne s'en sert pas, mais on ne s'interdit pas non plus de retomber sur leurs enregistrements. Les exclure aurait biaisé le tirage, puisqu'ils ont travaillé là où l'espèce chante ; les reprendre aurait mis leurs choix dans notre test.
""")

# ======================================================================== partie 1
s = divider(1, "Méthode du tirage", "Avant d'écouter : quels points d'écoute vont au test, et comment tirer dedans pour que les chiffres veuillent dire quelque chose.", ETAPES)
notes(s, "Première partie : la méthode du tirage. C'est la partie la plus importante pour mes tuteurs : tout ce qu'on mesurera ensuite en dépend.")

# ======================================================================== 4 · cinq découpages
s = content("MÉTHODE DU TIRAGE · 1/5", "Cinq façons de séparer entraînement et test : plus l'unité de découpage est grande, plus le test est honnête", ["evaluation/"])
rows = [
    ["Unité de découpage", "Le test mesure…", "Fuite entre entraînement et test", "Coût en positifs", "Pour nous"],
    ["Fenêtre de 3–5 s, au hasard", "rien d'utile : la fenêtre voisine du même chant est à l'entraînement", "maximale : même chant, même fond", "nul", ("non", {"color": ROUGE, "bold": True})],
    ["Enregistrement, au hasard", "un enregistrement neuf d'un micro déjà entendu", "forte : même micro, même fond, même semaine", "faible", ("non", {"color": ROUGE, "bold": True})],
    ["Point d'écoute (site + micro)", "un micro jamais entendu, sur un site connu", "faible : même site, même semaine", "≈ 20 % des points", ("oui : niveau 1", {"color": VERT, "bold": True})],
    ["Site entier", "un site jamais vu", "aucune dans le site", "tout un site ; le test dépend de ce site", ("oui : niveau 2", {"color": VERT, "bold": True})],
    ["Période ou station sur l'année", "des saisons jamais vues à ce point", "aucune dans le temps", "les stations 2023 ne sont que 6", ("oui : niveau 3", {"color": VERT, "bold": True})],
]
table(s, 0.6, 1.8, 12.13, rows, [2.55, 3.1, 2.85, 2.13, 1.5], size=12, row_h=0.6)
card(s, 0.6, 5.65, 5.95, 1.27, [
    P("« Coût en positifs »", size=13, bold=True, color=TEAL, after=3),
    P("Tout positif mis au test manque à l'entraînement. Avec 128 extraits positifs en tout, retirer un site entier peut emporter une grosse part des chants.", size=12),
])
card(s, 6.78, 5.65, 5.95, 1.27, [
    P("Plus l'unité est grande, moins on triche", size=13, bold=True, color=TEAL, after=3),
    P("Une fenêtre est plus petite qu'un enregistrement, lui-même plus petit qu'un point, puis qu'un site. Plus le morceau mis de côté est grand, moins il ressemble à l'entraînement.", size=12),
])
notes(s, """
On peut couper entraînement et test de cinq façons, de la plus fine à la plus grossière.

Couper au hasard entre fenêtres, c'est le pire : la fenêtre d'à côté, du même chant, est à l'entraînement. Le modèle reconnaît le chant qu'il a déjà vu.

Couper entre enregistrements, c'est encore trop facile : deux enregistrements du même micro ont le même fond sonore.

Couper par point d'écoute, par site, ou dans le temps, c'est là qu'on mesure vraiment quelque chose. On en a gardé trois, ce sont nos trois niveaux.

Le « coût en positifs » : chaque positif qu'on met dans le test, on ne l'a plus pour entraîner. Avec 128 extraits positifs en tout, ça compte. Plus l'unité de découpage est grande, plus le test est honnête, mais plus il coûte.
""")

# ======================================================================== 5 · le point d'écoute
s = content("MÉTHODE DU TIRAGE · 2/5", "L'unité est le point d'écoute, pas l'enregistrement : on découpe les points avant de tirer dedans", ["plan.py : draw_partition()"])
label(s, 0.6, 1.78, 5.6, "Du plus grand au plus petit")
niveaux = [
    ("Site", "Mataroni, CDR, RNRT, Kaw…", 0.0),
    ("Point d'écoute", "un micro posé à un endroit (2026) ou une station (2023)", 0.35),
    ("Enregistrement", "2 min, deux pistes, toutes les 30 min de 5 h à 19 h 30", 0.7),
    ("Extrait", "30 s écoutées (entraînement) ou 2 min (test)", 1.05),
    ("Fenêtre", "3 s (BirdNET) ou 5 s (Perch) : ce que voit l'encodeur", 1.4),
]
for k, (nom, d, dx) in enumerate(niveaux):
    y = 2.12 + k * 0.78
    hi = nom == "Point d'écoute"
    box(s, 0.6 + dx, y, 5.6 - dx, 0.66, TEAL if hi else PALER, radius=0.1)
    text(s, 0.8 + dx, y + 0.05, 5.25 - dx, 0.58, [P(nom, size=13.5, bold=True, color="FFFFFF" if hi else INK, after=0),
                                                  P(d, size=11, color="FFFFFF" if hi else MUTED)], anchor="m")
text(s, 0.6, 6.12, 5.6, 0.8, P("162 points : 156 micros en 2026, 6 stations en 2023. Chaque point reçoit un statut (entraînement, niveau 1, 2 ou 3) avant toute écoute ; on tire ensuite les enregistrements dans chaque point.", size=12, color=MUTED))
card(s, 6.6, 1.78, 6.13, 2.45, [
    P("Pourquoi le point et pas l'enregistrement", size=14, bold=True, color=TEAL, after=5),
    P("Deux enregistrements du même micro se ressemblent : même fond d'insectes, même ruisseau, même orientation, même gain, même réverbération. L'embedding encode tout cela, pas seulement le chant.", size=13.5, after=6),
    P("L'un à l'entraînement, l'autre au test : le test mesure la mémoire du lieu. C'est tricher sans le vouloir.", size=13.5),
])
card(s, 6.6, 4.4, 6.13, 2.5, [
    P("Deux temps : prédécoupage, puis découpage en niveaux", size=14, bold=True, color=GOLD_DARK, after=5),
    P(R("1 · ", bold=True), "On fixe l'unité : le point d'écoute. Tout ce qui vient d'un point va du même côté.", size=13.5, after=6),
    P(R("2 · ", bold=True), "On range les points en quatre groupes : entraînement, niveaux 1, 2 et 3 (diapo suivante). Tirage à graine fixe, figé dans partition_v1.csv avant d'écouter.", size=13.5, after=6),
    P("Même enregistreur déplacé d'un site à l'autre : deux points différents.", size=12, color=MUTED),
], bar=GOLD_DARK)
notes(s, """
Avant de parler des trois niveaux, il faut fixer l'unité. Dans nos données, il y a des sites, dans chaque site des points d'écoute, c'est-à-dire un micro posé à un endroit, puis des enregistrements de deux minutes, des extraits de trente secondes, et des fenêtres de trois ou cinq secondes.

L'unité qui compte, c'est le point d'écoute. Deux enregistrements du même point se ressemblent énormément : même fond d'insectes, même cours d'eau, même gain. Et l'encodeur ne fait pas le tri : il encode tout ça avec le chant. Si un enregistrement d'un point est à l'entraînement et un autre au test, le test mesure surtout que le modèle reconnaît l'endroit.

Donc on procède en deux temps. D'abord on décide que tout ce qui vient d'un même point ira du même côté. Ensuite on répartit les points en quatre groupes. Et c'est fait une fois, avant d'écouter, à graine fixe.
""")

# ======================================================================== 6 · trois niveaux
s = content("MÉTHODE DU TIRAGE · 3/5", "Trois niveaux de généralisation : chacun pose une question qu'une future campagne posera", ["plan.py : draw_partition()", "partition_v1.csv"])
# barre des 162 points
tot, xs, x0, wtot = 162, [], 0.6, 12.13
parts = [("entraînement · 95 points", 95, PALE, INK), ("niveau 1 · 24", 24, VERT, "FFFFFF"), ("niveau 2 · 40 (RNRT)", 40, BLEU, "FFFFFF"), ("3", 3, ORANGE, "FFFFFF")]
x = x0
for nom, n, c, tc in parts:
    w = wtot * n / tot
    box(s, x, 1.78, w - 0.03, 0.42, c)
    text(s, x, 1.78, w - 0.03, 0.42, P(nom, size=11, color=tc, bold=True, align="c"), anchor="m")
    x += w
cols = [
    ("NIVEAU 1 · NOUVEAU MICRO, SITE CONNU", VERT, "Un micro de plus sur un site qu'on suit déjà.",
     "20 % des points de chaque site 2026 (CDR, Mataroni, Patawa), tirés au hasard.",
     "Neuf : le micro, son endroit exact. Connu : le site, la semaine, la faune.",
     "3 enregistrements positifs sur 84 : trop peu pour un rappel."),
    ("NIVEAU 2 · SITE ABSENT DE 2026", BLEU, "Une nouvelle campagne dans une réserve.",
     "RNRT entier : 40 points, aucun enregistrement 2026 à l'entraînement.",
     "RNRT = réserve naturelle régionale Trésor : la réserve est connue par la station Trésor B de 2023. Neuf : les points, l'année. Pas un site vierge.",
     "0 positif sur 83 : ne mesure que les fausses alarmes."),
    ("NIVEAU 3 · STATION NEUVE, TOUTE L'ANNÉE", ORANGE, "Le suivi d'une saison à l'autre (phénologie).",
     "Une station 2023 sur deux par site : Kaw D, Molokoï E, Trésor A.",
     "Neuf : la station et les saisons qu'elle a vécues. Connu : l'autre station du site.",
     "22 positifs sur 83 : le seul niveau où le rappel se mesure."),
]
for k, (t, c, q, quoi, neuf, mesure) in enumerate(cols):
    x = 0.6 + k * 4.1
    box(s, x, 2.38, 3.93, 3.36, PALER, radius=0.05)
    box(s, x, 2.38, 3.93, 0.08, c)
    text(s, x + 0.2, 2.55, 3.55, 3.15, [
        P(t, size=11, bold=True, color=c, after=5),
        P(q, size=14.5, bold=True, after=8),
        P(R("Points · ", bold=True, color=MUTED), quoi, size=12.5, after=6),
        P(R("Neuf / connu · ", bold=True, color=MUTED), neuf, size=12.5, after=6),
        P(R("Aujourd'hui · ", bold=True, color=MUTED), mesure, size=12.5, bold=True),
    ])
card(s, 0.6, 5.88, 12.13, 1.05, [
    P(R("Pourquoi trois niveaux · ", bold=True, color=TEAL), "la partition ne rend pas le modèle meilleur : elle mesure honnêtement jusqu'où il tient loin de ce qu'il connaît. Si le score chute du niveau 1 au niveau 2, il faut annoter quelques chants sur chaque nouveau site (sur AnuraSet, 5 à 10 enregistrements positifs d'un site neuf font passer l'AP de 0,81 à 0,89) ; s'il chute au niveau 3, ce sont les saisons qui manquent.", size=12),
])
notes(s, """
Les trois niveaux, c'est trois questions que l'ONF se posera.

Niveau 1 : j'ajoute un micro sur un site que je suis déjà. Est-ce que le détecteur marche sur ce micro qu'il n'a jamais entendu ? On a pris 20 % des points de chaque site 2026.

Niveau 2 : je pars en campagne dans une autre réserve. On a mis RNRT entier de côté. Attention, RNRT c'est la réserve naturelle régionale Trésor, et Trésor est un site de 2023 : la station Trésor B est à l'entraînement. Donc le niveau 2, c'est de nouveaux points d'une réserve déjà vue une autre année, pas un site vierge. Un vrai site vierge, avec des chants, on ne l'a pas : CDR et les Patawa n'ont presque pas de positifs.

Niveau 3 : je suis une station toute l'année. Est-ce que le détecteur tient sur les saisons ? Une station 2023 sur deux est mise de côté.

Pourquoi ça aide à « mieux généraliser » : la partition ne rend pas le modèle meilleur, elle dit où il casse. L'écart entre les niveaux dit quoi annoter ensuite : des chants sur un nouveau site, ou des saisons. Sur AnuraSet, 5 à 10 enregistrements positifs d'un site neuf suffisaient à faire passer l'AP de 0,81 à 0,89.

Aujourd'hui, seul le niveau 3 a assez de positifs pour mesurer un rappel ; on y revient dans la partie 4.
""")

# ======================================================================== 7 · stratification
s = content("MÉTHODE DU TIRAGE · 4/5", "Dans chaque point, on tire par strates de site, saison et heure, les heures de pic comptées double", ["plan.py", "blanci candidates --plan"])
card(s, 0.6, 1.78, 5.85, 2.2, [
    P("Les strates", size=14, bold=True, color=TEAL, after=5),
    P(R("Heure locale · ", bold=True), "aube 5–7 h, pic du matin 7–9 h, journée 9–15 h, pic du soir 15–17 h, soir 17–20 h.", size=12, after=3),
    P(R("Saison (2023) · ", bold=True), "haute de décembre à avril (février–mars à part), transition, basse. En 2026, une semaine par site : la saison se confond avec le site.", size=12, after=3),
    P(R("Dans une strate · ", bold=True), "un jour au hasard, puis un enregistrement ; au plus un par point, jour et tranche.", size=12),
])
card(s, 0.6, 4.13, 5.85, 2.8, [
    P("Pourquoi la tête apprendrait « matin = A. blanci »", size=14, bold=True, color=GOLD_DARK, after=5),
    P("Elle ne lit pas l'heure. Mais l'embedding entend le fond : chœur d'oiseaux de l'aube, cigales du soir, pluie de l'après-midi. Ce fond change avec l'heure, la saison et le site.", size=12, after=4),
    P("Si tous les positifs étaient des matins et les négatifs des midis, reconnaître un fond de matin suffirait à bien classer l'entraînement : la tête apprendrait l'heure par le fond, pas le chant.", size=12, after=4),
    P(R("Parade · ", bold=True), "chaque strate qui donne des positifs donne aussi des négatifs, aux mêmes points et aux mêmes heures.", size=12),
], bar=GOLD_DARK)
picture(s, ANN / "strates.png", 6.65, 1.75, w=6.08)
text(s, 6.65, 5.35, 6.08, 1.55, [
    P("Les pics font 27 % du programme d'enregistrement, mais 64 % des lots et 56 % du test : surreprésentés deux fois, exprès, pour récolter des chants.", size=12, after=4),
    P("Le test garde la probabilité de tirage de chaque enregistrement : on repondère pour revenir au programme réel (diapo 22).", size=12, color=MUTED),
])
notes(s, """
Une fois les points répartis, on tire dans chaque point par strates.

Les strates, c'est l'heure, en cinq tranches, et la saison pour 2023. En 2026, chaque site n'a été enregistré qu'une semaine, donc la saison et le site, c'est la même chose.

Pourquoi ne pas écouter seulement les pics, là où la grenouille chante ? Parce que la tête ne lit pas l'heure, mais l'encodeur entend le fond. Le matin il y a le chœur des oiseaux, le soir les cigales, l'après-midi la pluie. Si tous mes positifs étaient du matin et tous mes négatifs de midi, la tête pourrait se contenter de reconnaître un fond du matin. Elle aurait l'air excellente à l'entraînement et ferait n'importe quoi en saison sèche, ou à un autre site. D'où la règle : chaque strate qui donne des positifs donne aussi des négatifs.

On a quand même doublé les pics : 27 % du programme, mais 64 % des lots. Pour le test, on a noté la probabilité de tirage de chaque enregistrement, ce qui permet de corriger ce surplus après coup.
""")

# ======================================================================== 8 · lots
s = content("MÉTHODE DU TIRAGE · 5/5", "Lot 1, lot 2, jeu de test v1 : ce qui a été tiré, écouté, et ce qu'on en fait", ["blanci candidates --plan", "files_ecoute/"])
rows = [
    ["", "Lot 1", "Lot 2", "Jeu de test v1"],
    ["Points", "92 points 2026 d'entraînement", "3 stations 2023 d'entraînement", "59 des 67 points tenus à l'écart (niveaux 1, 2, 3)"],
    ["Unité écoutée", "extrait de 30 s, position au hasard", "extrait de 30 s, position au hasard", "enregistrement entier de 2 min (4 × 30 s)"],
    ["Écoutés", "274 extraits", "150 extraits", "250 enregistrements"],
    ["Avec A. blanci", "28 (10 %)", "43 (29 %)", "25 (10 %) : 3 niveau 1, 0 niveau 2, 22 niveau 3"],
    ["Usage", "entraînement", "entraînement", "test seulement, jamais entraîné ; probabilité de tirage notée"],
]
table(s, 0.6, 1.8, 12.13, rows, [1.9, 3.05, 3.05, 4.13], size=12.5, row_h=0.56)
card(s, 0.6, 5.35, 5.95, 1.58, [
    P("Lot 1 et lot 2 : un seul tirage", size=13.5, bold=True, color=TEAL, after=4),
    P("La file de 424 extraits a été coupée en deux le 9 octobre parce que les deux disques ne pouvaient pas être branchés ensemble. Même méthode, même graine, même partition.", size=13),
])
card(s, 6.78, 5.35, 5.95, 1.58, [
    P("30 s à l'entraînement, 2 min au test", size=13.5, bold=True, color=TEAL, after=4),
    P("30 s ≈ 20 notes, trois fois plus de points écoutés pour le même temps. Le test décide par enregistrement : on l'écoute en entier. Les lots suivants viendront de l'apprentissage actif, jamais sur les points de test.", size=13),
])
notes(s, """
Ce qui a été tiré concrètement.

Le lot 1, c'est 274 extraits de trente secondes sur les 92 points d'entraînement de 2026. Le lot 2, 150 extraits sur les trois stations d'entraînement de 2023. En fait c'est un seul tirage, coupé en deux le 9 octobre pour une raison bête : les deux disques ne pouvaient pas être branchés en même temps.

Le jeu de test, c'est 250 enregistrements entiers de deux minutes, sur les points tenus à l'écart. Entiers, parce qu'à la fin le détecteur décide enregistrement par enregistrement.

28 positifs dans le lot 1, 43 dans le lot 2, 25 dans le test, dont 22 au niveau 3. Les lots suivants viendront de l'apprentissage actif, après le go des experts, et jamais sur les points de test.
""")

# ======================================================================== partie 2
s = divider(2, "Le poste d'annotation", "Un outil d'écoute fait pour un chant bref et aigu : une capture, puis le poste en direct, puis ce que deviennent les intervalles tracés.", ETAPES)
notes(s, "Deuxième partie : le poste. Je montre une capture, puis je l'ouvre en direct. Ensuite trois diapos sur la règle qui transforme mes intervalles en labels de fenêtres.")

# ======================================================================== 9 · capture
s = content("LE POSTE · 1/4", "Le poste d'annotation, tout neuf, prêt à servir en dehors de blanci", ["blanci annotate", "annotation/app.py"])
picture(s, ROOT / "documentation" / "prez" / "figures" / "annotation" / "poste.png", 2.17, 1.72, h=5.25)
notes(s, """
Voilà le poste d'annotation. Je ne le détaille pas ici : je l'ouvre en direct.

Ce qu'il faut retenir : il est neuf, il a été écrit pour ce chant, bref et aigu, et il est déjà pensé pour d'autres projets : un projet règle l'espèce, la bande d'écoute, les étiquettes et les qualités. Et le paquet de vérification s'ouvre sans rien installer.

[Capture à remplacer par une capture du jour avant la réunion.]
""")

# ======================================================================== 10 · fenêtre / intervalle
s = content("LE POSTE · 2/4", "Une fenêtre prend le label de l'intervalle qu'elle croise ; A. blanci l'emporte sur tout le reste", ["dataset.py : interval_labels()"])
text(s, 0.6, 1.72, 12.13, 0.85, [
    P("On ne trace pas des fenêtres mais des intervalles, au ras des notes. Chaque encodeur découpe ensuite l'audio en fenêtres (3 s pour BirdNET, 5 s pour Perch) et chaque fenêtre reçoit un label, déduit par le code. Une fenêtre qui touche assez un chant d'A. blanci est positive, même si elle touche aussi un faux ami.", size=13),
])
picture(s, ANN / "fen_label.png", 0.6, 2.62, w=12.13)
text(s, 0.6, 6.6, 12.13, 0.35, P("Kaw C, 19/09/2024 à 5 h : trois notes d'A. blanci (vert) au milieu de faux amis (rouge). En gras, deux fenêtres qui touchent les deux : celle qui contient la note est positive.", size=11, color=MUTED))
notes(s, """
Pourquoi des intervalles ? Parce que le label ne doit pas dépendre de l'encodeur. BirdNET coupe en fenêtres de trois secondes, Perch en cinq. Si j'avais annoté des fenêtres de trois secondes, il aurait fallu tout réécouter pour Perch.

Donc je trace où la grenouille chante, et le code en déduit le label de n'importe quelle fenêtre. La règle de base : une fenêtre prend le label de l'intervalle qu'elle croise. Et quand elle en croise plusieurs, A. blanci l'emporte.

Sur cet exemple, à Kaw, trois notes de blanci au milieu de faux amis. La fenêtre de 82,5 à 85,5 secondes touche un long faux ami et une note de blanci : elle est positive. Celle de 81 à 84 ne touche que le faux ami : elle est « faux ami », c'est-à-dire un négatif difficile, très utile à l'entraînement.
""")

# ======================================================================== 11 · deux notes
s = content("LE POSTE · 3/4", "Deux notes à plus de 3 s l'une de l'autre : deux intervalles, sinon le silence entre elles devient positif", ["dataset.py : interval_labels()", "viewer.py"])
picture(s, ANN / "fen_deux_notes.png", 0.6, 1.75, w=8.25)
card(s, 9.05, 1.75, 3.68, 5.15, [
    P("Pourquoi 3 s", size=14, bold=True, color=TEAL, after=5),
    P("3 s, c'est la fenêtre la plus courte de nos encodeurs (BirdNET). Si deux notes sont séparées de plus de 3 s, une fenêtre peut tenir tout entière dans le silence entre elles.", size=12, after=5),
    P("Avec un seul intervalle, cette fenêtre est dans l'intervalle : elle devient positive alors qu'on n'y entend rien. Avec deux intervalles, elle est négative, ce qui est juste.", size=12, after=5),
    P("La grille bouge d'un encodeur à l'autre (3 s ou 5 s, décalée de moitié) : la règle doit tenir pour toute position de fenêtre.", size=12, after=5),
    P(R("Au poste · ", bold=True), "deux pointillés gris à ±3 s de l'intervalle sélectionné aident à voir s'il faut couper.", size=12),
])
text(s, 0.6, 6.55, 8.25, 0.4, P("Molokoï F, 09/03/2024 à 8 h 30 : 3,07 s entre deux notes. Labels calculés par interval_labels() sur la même fenêtre, avec les deux tracés.", size=11, color=MUTED))
notes(s, """
Deuxième cas : deux notes séparées de plus de trois secondes.

Ici, à Molokoï, il y a 3,07 secondes entre deux notes. J'ai tracé deux intervalles. La fenêtre de trois secondes qui tombe entre les deux ne contient aucun chant : elle est négative, c'est juste.

Si j'avais tracé un seul grand intervalle, cette même fenêtre serait tout entière dedans, donc positive : un positif sans chant.

Pourquoi trois secondes ? Parce que c'est la fenêtre la plus courte de nos encodeurs. Au-delà, une fenêtre peut tomber dans le silence. Et la grille change d'un encodeur à l'autre, donc il faut que la règle tienne pour n'importe quelle position.

Dans le poste, j'ai ajouté deux pointillés à plus ou moins trois secondes de l'intervalle sélectionné, pour voir tout de suite s'il faut couper.
""")

# ======================================================================== 12 · bord
s = content("LE POSTE · 4/4", "Une fenêtre qui ne fait qu'effleurer un chant n'est ni positive ni négative : elle est écartée", ["dataset.py : interval_labels()"])
picture(s, ANN / "fen_bord.png", 0.6, 1.75, w=8.25)
card(s, 9.05, 1.75, 3.68, 5.15, [
    P("La règle", size=14, bold=True, color=TEAL, after=5),
    P("Positive si le temps en commun avec un intervalle atteint le plus petit de : la moitié de l'intervalle, la moitié de la fenêtre, 0,07 s si elle coupe son début, 0,2 s si elle coupe sa fin.", size=12, after=5),
    P("En dessous, sans être nulle : « bord », écartée de l'entraînement. On ne sait pas si l'encodeur entend assez du chant.", size=12, after=5),
    P(R("Pourquoi 0,07 et 0,2 s · ", bold=True), "je pose le début sur l'attaque de la note, nette ; la fin dans sa décroissance, deux fois moins nette.", size=12, after=5),
    P(R("Garantie · ", bold=True), "avec des fenêtres décalées de moitié, chaque intervalle garde une fenêtre positive : écarter les bords ne perd aucun chant (test automatique, grilles 3 s et 5 s).", size=12),
])
text(s, 0.6, 6.55, 8.25, 0.4, P("Mataroni : 0,18 s en commun à la fin d'un intervalle, sous les 0,2 s. 29 bords sur 35 329 fenêtres de 3 s (0,1 %).", size=11, color=MUTED))
notes(s, """
Dernier cas : la fenêtre qui effleure un chant.

Ici la fenêtre de 31,5 à 34,5 secondes n'a que 0,18 seconde en commun avec la fin d'un intervalle. Est-ce que l'encodeur entend assez de chant ? On ne sait pas. Donc on ne la met ni en positif ni en négatif : on l'écarte. C'est un « bord ».

La règle complète est sur la droite. Les deux seuils, 0,07 seconde au début et 0,2 à la fin, viennent de la façon dont je trace : le début est posé sur l'attaque de la note, qui est très nette ; la fin dans la décroissance, beaucoup moins nette.

Et on ne perd aucun chant : avec des fenêtres qui se recouvrent de moitié, chaque intervalle a toujours au moins une fenêtre positive. C'est vérifié par un test automatique. En pratique, ça ne concerne que 29 fenêtres sur 35 000.
""")

# ======================================================================== partie 3
s = divider(3, "Les coulisses", "Comment les files arrivent au poste, où partent mes annotations, à quoi ressemblent les fichiers, et quelle fonction fait quoi.", ETAPES)
notes(s, "Troisième partie : les coulisses. C'est pour ceux qui reprendront la chaîne après moi.")

# ======================================================================== 13 · chaîne
s = content("COULISSES · 1/4", "La chaîne de l'annotation : du tirage à l'entraînement, chaque étape a sa commande", ["cli.py", "service.py", "dataset.py"])
etapes = [
    ("Plan de tirage", "blanci candidates --plan", "partition + strates, graine fixe"),
    ("File d'écoute", "candidats.csv + LISEZMOI.md", "des références, pas d'audio"),
    ("Poste", "blanci annotate", "Streamlit + visualiseur"),
    ("Base SQLite", "tables spans, intervals", "ajout seul"),
    ("CSV versionnés", "blanci export-labels", "annotations/*.csv, dans git"),
    ("Labels de fenêtres", "interval_labels()", "pour toute grille (3 s, 5 s)"),
]
w, gap, y = 1.85, 0.206, 2.15
for k, (t, c, d) in enumerate(etapes):
    x = 0.6 + k * (w + gap)
    box(s, x, y, w, 1.55, PALER if k != 2 else TEAL, radius=0.08)
    hi = k == 2
    text(s, x + 0.12, y + 0.12, w - 0.24, 1.35, [
        P(t, size=13, bold=True, color="FFFFFF" if hi else INK, after=4, align="c"),
        P(c, size=10, color="FFFFFF" if hi else DARK, font="Consolas", bold=True, after=4, align="c"),
        P(d, size=10.5, color="FFFFFF" if hi else MUTED, align="c"),
    ], anchor="m")
    if k < len(etapes) - 1:
        arrow(s, x + w + 0.02, y + 0.78, x + w + gap - 0.02, y + 0.78, color=TEAL)
label(s, 0.6, 1.78, 6, "Boucle principale")
# import et entraînement
box(s, 8.9, 4.0, 3.83, 0.95, PALER, radius=0.08)
text(s, 9.02, 4.05, 3.6, 0.85, [P("Entraînement et benchmark", size=13, bold=True, after=2, align="c"),
                                P("training_set() · blanci train · benchmark", size=10, font="Consolas", color=DARK, bold=True, align="c")], anchor="m")
arrow(s, 11.71, 3.7, 11.71, 4.0, color=TEAL)
box(s, 4.73, 4.0, 3.83, 0.95, PALER, radius=0.08)
text(s, 4.85, 4.05, 3.6, 0.85, [P("Autre machine", size=13, bold=True, after=2, align="c"),
                                P("git pull · blanci import-labels", size=10, font="Consolas", color=DARK, bold=True, align="c")], anchor="m")
arrow(s, 9.6, 3.7, 7.6, 4.0, color=SAGE)
label(s, 0.6, 5.2, 6, "Boucle de vérification", color=GOLD_DARK)
verif = [("Tirage à l'aveugle", "tirage_verification.py"), ("Paquet sans installation", "blanci paquet"), ("Réponses de l'expert", "<Nom>_reponses.json"), ("Retour en base", "blanci import-labels")]
for k, (t, c) in enumerate(verif):
    x = 0.6 + k * 3.1
    box(s, x, 5.55, 2.85, 1.0, "FFF8E8", radius=0.08)
    text(s, x + 0.1, 5.6, 2.65, 0.9, [P(t, size=12.5, bold=True, after=3, align="c"), P(c, size=10, font="Consolas", color=GOLD_DARK, bold=True, align="c")], anchor="m")
    if k < 3:
        arrow(s, x + 2.87, 6.05, x + 3.08, 6.05, color=GOLD)
text(s, 0.6, 6.65, 12.13, 0.3, P("La réponse d'un expert entre dans la même base, à son nom : c'est un envoi de plus, et le plus récent compte.", size=11, color=MUTED))
notes(s, """
La chaîne complète, de gauche à droite. Le plan de tirage écrit une file d'écoute : un CSV de références, enregistrement, début, durée, sans audio. Le poste lit la file, retrouve l'audio sur les disques branchés, et chaque envoi s'écrit dans la base SQLite. Après chaque séance, export-labels recopie la base dans des CSV versionnés dans git. Sur une autre machine, un git pull et un import-labels, et on reprend où on en était. Enfin, interval_labels transforme les intervalles en labels de fenêtres pour l'encodeur choisi, et c'est ce qui part à l'entraînement.

En bas, la boucle de vérification : un tirage à l'aveugle, un paquet que l'expert ouvre sans rien installer, ses réponses en JSON, qu'on réimporte. Sa réponse est un envoi de plus, à son nom.
""")

# ======================================================================== 14 · files
s = content("COULISSES · 2/4", "Les files d'écoute : des files déjà prêtes, ou une file générée par le poste", ["blanci candidates", "annotation/app.py", "selection.py"])
label(s, 0.6, 1.78, 6, "Files déjà prêtes : 18 dossiers dans files_ecoute/")
pretes = [
    ("Tirées par le plan", VERT, "lot1, lot2, test_v1", "blanci candidates --plan : partition, strates, graine. Les seules qui servent à l'entraînement et au test."),
    ("Écrites par un script du contrôle qualité", ORANGE, "muffled_douteux, etouffe_*, kaw_d_micro_sourd, blancinet_ecartes, coupes_programme, bascules_fines…", "pour trancher un drapeau à l'oreille ; leurs extraits ne vont pas à l'entraînement."),
    ("Retours d'experts", BLEU, "doutes", "les 20 extraits « A. blanci ? » renvoyés par Benoît, à réécouter d'après ses commentaires."),
]
for k, (t, c, q, d) in enumerate(pretes):
    y = 2.12 + k * 1.28
    card(s, 0.6, y, 6.0, 1.16, [P(t, size=13, bold=True, color=c, after=2), P(q, size=10.5, font="Consolas", color=DARK, bold=True, after=3), P(d, size=11.5)], bar=c)
text(s, 0.6, 6.0, 6.0, 0.9, P("Chaque file = candidats.csv (enregistrement, début, durée, raison, strate, probabilité de tirage) et un LISEZMOI.md qui dit d'où elle vient. Pas d'audio : le poste le retrouve sur le disque branché.", size=11.5, color=MUTED))
label(s, 6.83, 1.78, 6, "Les autres façons de faire une file", color=GOLD_DARK)
modes = [
    ("File 60-20-20", "60 % d'incertains, 20 % de meilleurs scores, 20 % au hasard"),
    ("Récolte par similarité", "les fenêtres proches des chants connus"),
    ("Negative mining", "les négatifs les mieux notés : les pièges"),
    ("Carte et couverture des embeddings", "écouter là où personne n'a écouté"),
    ("Groupes, puis étiquetage en bloc", "un regroupement, un label par groupe"),
    ("Audit, hasard, phénologie", "enregistrements entiers ou fenêtres au hasard"),
    ("Détections isolées, trous dans un chant", "faux positifs et faux négatifs suspects"),
]
box(s, 6.83, 2.12, 5.9, 3.8, "FFF8E8", radius=0.04)
text(s, 7.05, 2.25, 5.5, 3.6, [P(R(t + " · ", bold=True), R(d, color=MUTED), size=13, after=8) for t, d in modes])
card(s, 6.83, 6.05, 5.9, 0.88, [P("Elles demandent un encodeur ou un modèle : réservées aux lots d'après le go, jamais sur les points de test.", size=12)], bar=GOLD_DARK, fill=PALER)
notes(s, """
D'où viennent les files d'écoute.

Il y a d'abord les files déjà prêtes, rangées dans files_ecoute. Trois origines. Celles tirées par le plan : lot1, lot2, test_v1 ; ce sont les seules qui comptent pour l'entraînement et le test. Celles écrites par un script du contrôle qualité, pour trancher un drapeau à l'oreille, par exemple le micro sourd de Kaw D. Et celles qui viennent d'un retour d'expert, comme « doutes », les vingt extraits que Benoît m'a renvoyés.

Chaque file, c'est un CSV de références et un LISEZMOI qui dit d'où elle vient. Jamais d'audio : le poste le retrouve sur le disque branché.

À droite, les autres façons de faire une file, depuis le poste lui-même : apprentissage actif, similarité, negative mining, carte des embeddings… Elles ont toutes besoin d'un modèle ou d'un encodeur. Donc elles sont réservées à l'après-go, et jamais sur les points de test.
""")

# ======================================================================== 15 · SQLite et CSV
s = content("COULISSES · 3/4", "« Valider » ajoute des lignes à la base ; l'export les recopie dans des CSV versionnés", ["workbench.py : save_span()", "service.py"])
label(s, 0.6, 1.78, 4, "Au clic sur « Valider »")
box(s, 0.6, 2.12, 3.85, 3.25, PALER, radius=0.05)
text(s, 0.82, 2.25, 3.45, 3.05, [
    P(R("+ 1 ligne dans spans", bold=True, color=TEAL), size=13, after=2),
    P("l'extrait : enregistrement, début, fin, classes entendues, qualité, commentaire, piste affichée, annotateur, date", size=11.5, after=8),
    P(R("+ 1 ligne par intervalle dans intervals", bold=True, color=TEAL), size=13, after=2),
    P("début, fin, label (A. blanci, plusieurs, ?, faux ami), qualité", size=11.5, after=8),
    P(R("Rien n'est modifié ni effacé", bold=True, color=TERRA), size=13, after=2),
    P("réécouter un extrait ajoute un envoi ; le plus récent compte", size=11.5),
])
card(s, 4.65, 1.78, 3.95, 3.6, [
    P("Pourquoi SQLite", size=14, bold=True, color=TEAL, after=5),
    P("Une base de travail en un seul fichier, sans serveur.", size=13, after=7),
    P("Un envoi s'écrit en entier ou pas du tout, même si le PC plante au milieu.", size=13, after=7),
    P("Les annotations se croisent avec les 96 310 enregistrements, leurs drapeaux et les scores, en une requête.", size=13, after=7),
    P(R("Mais · ", bold=True, color=TERRA), "un fichier binaire de 6,7 Go avec l'inventaire : hors git, illisible hors Python.", size=13),
])
card(s, 8.78, 1.78, 3.95, 3.6, [
    P("Pourquoi aussi des CSV", size=14, bold=True, color=GOLD_DARK, after=5),
    P("Du texte : versionné dans git, avec l'historique et les différences d'un jour à l'autre.", size=13, after=7),
    P("Lisible dans Excel par les encadrants, sans rien installer.", size=13, after=7),
    P("C'est par eux qu'on change de machine : import-labels relit sans doublon.", size=13, after=7),
    P("Une sauvegarde : si la base est perdue, les annotations ne le sont pas.", size=13),
], bar=GOLD_DARK)
card(s, 0.6, 5.55, 12.13, 1.38, [
    P(R("En une phrase · ", bold=True, color=TEAL), "la base est la mémoire de travail, les CSV sont l'archive et le moyen de transport. On écrit dans la base, on exporte après chaque séance (blanci export-labels → export_annotations()), on relit ailleurs (blanci import-labels → import_annotations()).", size=12.5, after=4),
    P("Quatre fichiers : extraits_annotes.csv, intervalles_annotes.csv, fenetres_annotees.csv (59 coups d'œil sur des écartés, hors entraînement), reannotations.csv (première et dernière version, pour suivre les corrections).", size=11.5, color=MUTED),
])
notes(s, """
Qu'est-ce qui se passe quand j'appuie sur Valider ? Une ligne s'ajoute dans la table des extraits, et une ligne par intervalle dans la table des intervalles. Rien n'est jamais modifié ni effacé.

Pourquoi deux formats, une base et des CSV ? La base SQLite, c'est la mémoire de travail : un seul fichier, pas de serveur, un envoi s'écrit en entier ou pas du tout, et on croise les annotations avec l'inventaire, les drapeaux, les scores. Mais c'est un gros fichier binaire, qui ne peut pas aller dans git.

Les CSV, c'est l'archive et le transport : du texte, versionné dans git, lisible dans Excel, et c'est par eux qu'on passe d'une machine à l'autre. Et si la base est perdue, les annotations ne le sont pas.
""")

# ======================================================================== 16 · vraie ligne
s = content("COULISSES · 4/4", "À quoi ressemble ce qui arrive : un extrait, et ses intervalles", ["annotations/*.csv"])
label(s, 0.6, 1.78, 8, "extraits_annotes.csv · une ligne par envoi")
rows = [
    ["span_id", "site · micro", "début UTC", "extrait", "n_blanci", "classes", "qualité", "espèce", "commentaire", "remplace_par"],
    ["88", "Kaw · SMA13599", "19/09/2024 08:00", "77,4–107,4 s", "3", "blanci, false_friend", "—", "congénères probablement", "3 notes de blanci entourées de faux amis", "vide = à lire"],
]
table(s, 0.6, 2.1, 12.13, rows, [0.8, 1.35, 1.35, 1.1, 0.8, 1.5, 0.7, 1.45, 1.9, 1.18], size=11, row_h=0.5, first_col_bold=False)
label(s, 0.6, 3.35, 8, "intervalles_annotes.csv · une ligne par intervalle (3 sur 11)")
rows = [
    ["span_id", "début (s)", "fin (s)", "label", "qualité", "valable"],
    ["88", "80,46", "80,83", ("blanci", {"color": VERT, "bold": True}), "A", "True"],
    ["88", "81,12", "84,59", ("false_friend", {"color": ROUGE, "bold": True}), "B", "True"],
    ["88", "84,86", "85,24", ("blanci", {"color": VERT, "bold": True}), "A", "True"],
]
table(s, 0.6, 3.67, 6.4, rows, [0.9, 1.1, 1.1, 1.5, 0.9, 0.9], size=11.5, row_h=0.42, first_col_bold=False)
card(s, 7.25, 3.4, 5.48, 3.5, [
    P("Comment lire", size=14, bold=True, color=TEAL, after=5),
    P(R("remplace_par · ", bold=True), "vide sur la dernière version d'un extrait : ce sont les lignes à lire. Sinon, le span_id de la version qui la remplace.", size=12, after=4),
    P(R("valable · ", bold=True), "faux si un envoi plus récent couvre l'intervalle.", size=12, after=4),
    P(R("n_blanci · ", bold=True), "le nombre d'intervalles d'A. blanci, pour filtrer sans lire la liste.", size=12, after=4),
    P(R("source · ", bold=True), "plan (lots et test) ou flag (coup d'œil sur un écarté) ; l'entraînement ne prend que plan.", size=12, after=4),
    P("Plus le chemin du fichier, l'annotateur, la date d'envoi, le recording_id.", size=11.5, color=MUTED),
])
text(s, 0.6, 5.55, 6.4, 1.35, P("Les mêmes faits que dans la base, une ligne par envoi : l'extrait de Kaw de la diapo 12, trois notes de blanci au milieu de huit faux amis.", size=12, color=MUTED))
notes(s, """
Une vraie ligne. L'extrait 88, à Kaw, celui des trois notes de blanci au milieu des faux amis.

Dans le fichier des extraits : une ligne par envoi, avec le site, le micro, le moment, les bornes de l'extrait, le nombre d'intervalles de blanci, les classes entendues, mon commentaire. La colonne remplace_par est vide : c'est la dernière version, celle à lire.

Dans le fichier des intervalles, une ligne par intervalle, avec son label, sa qualité, et « valable » qui dit s'il compte encore.
""")

# ======================================================================== partie 4
s = divider(4, "Le résultat", "Ce que contient le jeu annoté, la part d'enregistrements où la grenouille chante, et ce que l'inégalité entre sites change pour la suite.", ETAPES)
notes(s, "Quatrième partie : le résultat. Deux diapos délicates : la prévalence, et la répartition inégale des positifs.")

# ======================================================================== 17 · bilan
s = content("RÉSULTAT · 1/3", "L'annotation est faite : 1 453 extraits, 13 h d'écoute, 373 intervalles d'A. blanci", ["annotations/*.csv"])
tiles = [("1 453", "extraits écoutés (dernière version), 13 séances du 8 oct. au 4 nov.", TEAL),
         ("13 h", "d'écoute : 3,5 h pour les lots, 8,3 h pour le test", TEAL),
         ("373", "intervalles d'A. blanci : 331 notes ou chants, 42 chœurs", VERT),
         ("375", "faux amis localisés : des négatifs difficiles, précieux", ROUGE)]
for k, (b, sm, c) in enumerate(tiles):
    tile(s, 0.6 + k * 3.08, 1.78, 2.93, 1.3, b, sm, color=c)
picture(s, ANN / "duree_chants.png", 0.6, 3.25, w=6.4)
card(s, 7.2, 3.25, 5.53, 3.65, [
    P("Ce que le jeu contient", size=14, bold=True, color=TEAL, after=5),
    P(R("Deux familles d'intervalles · ", bold=True), "des notes isolées (0,1 à 0,4 s, médiane 0,33 s) et des chants continus tracés d'un bloc (2 à 30 s).", size=12, after=4),
    P(R("Qualité · ", bold=True), "228 A, 92 B, 50 C ; 90 « A. blanci ? » écartés de l'entraînement.", size=12, after=4),
    P(R("Positifs · ", bold=True), "71 extraits sur 424 dans les lots, 25 enregistrements sur 250 dans le test (60 visés).", size=12, after=4),
    P(R("Machines · ", bold=True), "le PC de l'ONF et mon portable, un import-labels à chaque changement.", size=12, after=4),
    P("Un seul annotateur : moi. La vérification par Élodie et Benoît dira si c'est assez juste (diapo 26).", size=11.5, color=MUTED),
])
notes(s, """
Le bilan. 1 453 extraits écoutés, treize heures d'écoute en treize séances, du 8 octobre au 4 novembre. 373 intervalles d'A. blanci, et presque autant de faux amis localisés, 375 : des négatifs difficiles, ce sont eux qui apprennent à la tête ce qui n'est pas la grenouille.

Les durées montrent deux familles : des notes isolées, autour d'un tiers de seconde, et des chants continus tracés d'un seul bloc, jusqu'à trente secondes, c'est-à-dire tout l'extrait.

71 extraits positifs sur 424 dans les lots ; 25 enregistrements positifs sur 250 dans le test, alors qu'on en visait 60. On y revient tout de suite.
""")

# ======================================================================== 18 · prévalence
s = content("RÉSULTAT · 2/3", "La prévalence : sans elle, une précision ne veut rien dire, et on ne sait pas combien écouter", ["presentation_5_figures.py : prevalences()"])
card(s, 0.6, 1.78, 5.95, 2.0, [
    P("Ce que c'est, et comment on l'estime", size=14, bold=True, color=TEAL, after=4),
    P(R("Prévalence · ", bold=True), "la part des enregistrements de 2 min (5 h – 20 h) où A. blanci chante.", size=12, after=3),
    P(R("Pondérée · ", bold=True), "le test surreprésente les pics : chaque enregistrement compte pour 1 / sa probabilité de tirage (estimateur de Horvitz-Thompson), ce qui ramène au programme réel. Intervalles à 95 % par bootstrap.", size=12),
])
rows = [
    ["Niveau du test", "Écoutés", "Positifs", "Brute", "Pondérée [IC 95 %]"],
    ["1 · micro neuf, 2026", "84", "3", "3,6 %", "1,8 % [0 ; 4,3]"],
    ["2 · RNRT, 2026", "83", "0", "0 %", "0 %"],
    ["3 · station 2023, l'année", "83", "22", "26,5 %", "29,6 % [17 ; 43]"],
]
table(s, 0.6, 3.92, 5.95, rows, [1.95, 0.8, 0.8, 0.7, 1.7], size=11.5, row_h=0.38)
card(s, 0.6, 5.62, 5.95, 1.31, [
    P("Pour comprendre la courbe : 1 000 enregistrements à 1,8 %", size=12.5, bold=True, color=GOLD_DARK, after=3),
    P("18 ont le chant, le détecteur en trouve 16 ; 982 n'en ont pas, et 5 % d'entre eux déclenchent : 49 fausses alertes. 16 justes sur 65 : 25 % de précision.", size=11.5),
], bar=GOLD_DARK)
picture(s, ANN / "precision_prevalence.png", 6.75, 1.72, w=5.98)
text(s, 6.75, 5.45, 5.98, 1.5, [
    P(R("1 · La précision en dépend · ", bold=True, color=TEAL), "le même détecteur donne 88 % d'alertes justes au niveau 3, 25 % au niveau 1.", size=12, after=4),
    P(R("2 · La taille du test en dépend · ", bold=True, color=TEAL), "60 positifs à 1,8 % demandent ≈ 3 300 enregistrements tirés au hasard.", size=12, after=4),
    P(R("3 · C'est le résultat écologique · ", bold=True, color=TEAL), "l'activité par site et par saison est ce que l'ONF veut lire en sortie.", size=12),
])
notes(s, """
Une diapo un peu technique, mais importante.

La prévalence, c'est la part des enregistrements où la grenouille chante. Pour l'estimer, il ne suffit pas de compter les positifs du test : on a doublé les heures de pic, donc la proportion brute est trop haute. On corrige en donnant à chaque enregistrement un poids égal à un sur sa probabilité d'avoir été tiré : un enregistrement d'une strate qu'on a surreprésentée pèse moins.

Résultat : 1,8 % au niveau 1, en 2026, zéro à RNRT, et près de 30 % sur l'année 2023. Les intervalles sont larges, on a peu d'enregistrements.

Pourquoi c'est important, trois raisons.

Un, la précision, la part d'alertes justes, dépend de la prévalence. Sur la courbe, le même détecteur, 90 % de rappel et 5 % de fausses alarmes, donne 88 % d'alertes justes quand l'espèce chante dans un tiers des enregistrements, et 25 % quand elle chante dans 2 %. Trois alertes sur quatre seraient fausses. Une précision sans prévalence, ça ne dit rien.

Deux, elle dit combien écouter : pour 60 positifs à 1,8 %, il faudrait plus de 3 000 enregistrements au hasard.

Trois, c'est le résultat lui-même : l'ONF veut savoir où et quand la grenouille chante.
""")

# ======================================================================== 19 · répartition inégale
s = content("RÉSULTAT · 3/3", "Des positifs très inégalement répartis : supportable à l'entraînement, gênant pour le test", ["dataset.py : training_set()", "evaluation/"])
picture(s, ANN / "site_heure.png", 0.6, 1.72, w=6.1)
card(s, 6.9, 1.78, 5.83, 2.5, [
    P("À l'entraînement : supportable, à une condition", size=14, bold=True, color=VERT, after=5),
    P("Molokoï donne 39 positifs sur 137 extraits, CDR 3 sur 130, les Patawa et RNRT aucun. Le risque : apprendre « fond de Molokoï = A. blanci ».", size=13, after=6),
    P("La parade est déjà là : chaque site et chaque heure qui donne des positifs donne aussi des négatifs, et chaque enregistrement pèse autant (poids 1 / nombre de fenêtres). Les sites sans chant fournissent des fonds négatifs variés : c'est utile.", size=13),
], bar=VERT)
card(s, 6.9, 4.42, 5.83, 2.5, [
    P("Au test : gênant", size=14, bold=True, color=TERRA, after=5),
    P("22 des 25 positifs sont au niveau 3. Le niveau 2 n'en a aucun : il ne mesure que les fausses alarmes par heure. Le niveau 1 en a 3 : aucun rappel mesurable.", size=13, after=6),
    P("Avec 25 positifs, un rappel de 0,90 a pour intervalle [0,72 ; 0,97] ; avec 60, [0,80 ; 0,95]. On rapportera chaque niveau à part, jamais un chiffre global qui serait celui du niveau 3.", size=13),
], bar=TERRA)
text(s, 0.6, 6.15, 6.1, 0.8, P("Les chants sont à l'aube et aux pics ; la journée en est presque vide, le soir aussi en 2026. Doubler les pics était juste ; l'aube compte autant.", size=12, color=MUTED))
notes(s, """
Les positifs sont très mal répartis : beaucoup à Molokoï et Trésor, un peu à Kaw et Mataroni, presque rien à CDR, rien aux Patawa ni à RNRT.

Est-ce gênant pour entraîner ? Pas tant que ça, à une condition : que chaque site et chaque heure qui donne des positifs donne aussi des négatifs. Sinon la tête apprend « fond de Molokoï égale grenouille ». C'est ce que le tirage par strates garantit. Et les sites sans grenouille ne sont pas du gaspillage : ce sont des fonds négatifs variés, exactement ce qu'il faut pour éviter les fausses alarmes ailleurs.

Pour le test, c'est plus gênant. Presque tous les positifs sont au niveau 3. Le niveau 2 n'en a aucun, donc il ne mesure que les fausses alarmes. Le niveau 1 en a trois, on ne peut rien dire du rappel sur un micro neuf de 2026. Avec 25 positifs, un rappel de 0,9 a un intervalle de 0,72 à 0,97 : c'est large. D'où la question d'un test_v2, que je pose à la fin.
""")

# ======================================================================== partie 5
s = divider(5, "Doutes et erreurs", "Les chants entendus sur la mauvaise piste, ce qui conditionne toute la suite, et ce qui reste fragile.", ETAPES)
notes(s, "Dernière partie : les erreurs, et ce qu'on fait pour les rattraper.")

# ======================================================================== 20 · piste 1
s = content("ERREURS · 1/3", "J'ai parfois annoté en écoutant la piste 1 alors qu'on encode la piste 0 : trois façons de les rattraper", ["workbench.py", "inventaire.py", "spectre_pistes.py"])
picture(s, ROOT / "documentation" / "prez" / "figures" / "annotation" / "deux_pistes.png", 0.6, 1.72, w=6.3)
text(s, 0.6, 4.55, 6.3, 0.5, P("Même extrait de 30 s à Molokoï sur les deux micros : les notes sont plus nettes sur la piste 1.", size=11, color=MUTED))
tile(s, 0.6, 5.25, 2.0, 1.4, "128", "extraits positifs (dernière version)", color=VERT)
tile(s, 2.75, 5.25, 2.0, 1.4, "11", "validés avec le micro 2 affiché", color=ORANGE)
tile(s, 4.9, 5.25, 2.0, 1.4, "3", "où mon commentaire dit « micro 2 »", color=TERRA)
cases = [
    ("1 · Je l'ai écrit en commentaire", VERT, "« voir micro 2 », « écouter sur le micro 2 »… : on encode ces enregistrements sur la piste 1 (règle de piste par enregistrement, audio.channel_rules)."),
    ("2 · Je ne l'ai pas écrit", ORANGE, "Pour chaque fenêtre positive, comparer le niveau dans la bande du chant (4,4–5,5 kHz) sur les deux pistes, corrigé du gain réglé. Piste 0 nettement plus faible : réécouter, ou encoder la piste 1."),
    ("3 · Le clustering, sans trop y croire", MUTED, "Un positif dont l'embedding de la piste 0 tombe loin des autres positifs n'a peut-être été entendu que sur la piste 1. Mais en multi-site, le clustering regroupe surtout par site ou par micro : je pars du principe qu'il ne conclura pas."),
]
for k, (t, c, d) in enumerate(cases):
    card(s, 7.1, 1.78 + k * 1.73, 5.63, 1.6, [P(t, size=14, bold=True, color=c if c != MUTED else INK, after=4), P(d, size=12.5)], bar=c)
notes(s, """
Une erreur de ma part. Chaque enregistreur a deux micros. La chaîne encode la piste 0, le micro 1. Mais au poste, j'ai parfois écouté ou regardé le micro 2, parce que le chant y était plus net. Un chant entendu sur la piste 1 et presque absent de la piste 0 devient un positif que l'encodeur n'entend pas : c'est du bruit dans les labels.

Combien ? Le poste note le micro affiché au moment de valider : 11 des 128 extraits positifs ont été validés avec le micro 2 affiché, et trois fois je l'ai écrit en commentaire.

Trois façons de rattraper. Un : quand je l'ai écrit, c'est simple, on encode ces enregistrements sur la piste 1. Deux : quand je ne l'ai pas écrit, on mesure. Pour chaque fenêtre positive, on compare le niveau dans la bande du chant sur les deux pistes, en corrigeant du gain. Si la piste 0 est nettement plus faible, on réécoute, ou on encode la piste 1. Trois : le clustering. Un positif dont l'embedding tombe loin des autres positifs pourrait n'avoir été entendu que sur la piste 1. Mais je n'y crois pas trop : sur plusieurs sites, le clustering a tendance à faire des groupes par site, pas par présence de la grenouille.
""")

# ======================================================================== 21 · validité
s = content("ERREURS · 2/3", "Toute la suite de la chaîne ne vaut que si mes annotations sont assez bonnes", ["tirage_verification.py", "blanci paquet"])
text(s, 0.6, 1.85, 12.13, 0.9, P("Mes labels sont la vérité de référence de tout ce qui suit : ils entraînent les têtes, règlent les seuils et jugent les encodeurs. Une erreur d'annotation ne se voit plus ensuite : elle devient une erreur du modèle, ou pire, une réussite qui n'en est pas une.", size=15))
chain = ["Mes annotations", "Benchmark des encodeurs", "Têtes et seuils", "Détection sur la campagne", "Activité par site et saison"]
for k, t in enumerate(chain):
    x = 0.6 + k * 2.47
    box(s, x, 3.1, 2.2, 0.95, TEAL if k == 0 else PALER, radius=0.1)
    text(s, x + 0.1, 3.1, 2.0, 0.95, P(t, size=13, bold=True, color="FFFFFF" if k == 0 else INK, align="c"), anchor="m")
    if k < 4:
        arrow(s, x + 2.22, 3.575, x + 2.45, 3.575, color=TEAL)
card(s, 0.6, 4.4, 12.13, 2.5, [
    P("D'où la vérification, avant tout résultat", size=15, bold=True, color=TEAL, after=6),
    P("Élodie et Benoît réécoutent à l'aveugle un tirage au hasard de mes extraits (mon label caché, ordre mélangé), dans un paquet qui s'ouvre sans rien installer. Leurs réponses entrent dans la base à leur nom.", size=13, after=5),
    P("Tant que ce contrôle n'a pas dit que mes annotations sont assez justes, aucun chiffre de benchmark sur nos données n'est présenté comme un résultat.", size=13, bold=True),
])
notes(s, """
Je le dis simplement : toute la suite repose sur mes annotations. Elles entraînent les têtes, elles règlent les seuils, elles jugent les encodeurs. Une erreur d'annotation ne se voit plus après : elle devient une erreur du modèle, ou pire, un bon score qui n'en est pas un.

Donc avant de présenter le moindre résultat sur nos données, il faut que la vérification par Élodie et Benoît dise que mes annotations sont assez justes. C'est le sens du paquet que je leur prépare.
""")

# ======================================================================== 22 · limites
s = content("ERREURS · 3/3", "Ce qui reste fragile, et ce qui est déjà corrigé", ["qc.py", "embed.py", "dataset.py", "db.py"])
lim = [
    ("Contrôle qualité après l'annotation", TERRA,
     "Des enregistrements étouffés étaient dans mes files (4 dans le test, 2 dans le lot 1) : j'ai annoté avant un contrôle qualité sûr.",
     "Ce qui protège : un enregistrement écarté par un drapeau n'est jamais encodé. Ses labels ne trouvent pas d'embedding et sortent seuls de l'entraînement et du test ; rien n'est effacé. Un extrait cassé ne peut pas passer en silence."),
    ("Un test court en positifs", TERRA,
     "25 positifs au lieu de 60 ; 3 au niveau 1, 0 au niveau 2.",
     "Que faire : un test_v2 tiré aux heures de pic sur les points des niveaux 1 et 3, avec la probabilité de tirage notée pour repondérer ; ou garder le test tel quel et rapporter chaque niveau avec son intervalle de confiance. À décider."),
    ("Des seuils réglés à l'œil", GOLD_DARK,
     "0,07 s au début d'un intervalle, 0,2 s à la fin, 3 s entre deux notes.",
     "Ils viennent de 13 h de tracé : je sais où je pose un début (sur l'attaque) et une fin (dans la décroissance). Rien à voir avec un seuil posé sans données. À vérifier : retracer un échantillon et mesurer l'écart."),
    ("La piste écoutée", TEAL,
     "Avant : seul le micro affiché à l'envoi était noté, pas celui écouté en traçant.",
     "Corrigé : désormais chaque intervalle garde la piste écoutée au moment où il est tracé (colonne channel de la table intervals) ; c'est sur elle qu'on encodera."),
]
for k, (t, c, fait, rep) in enumerate(lim):
    x = 0.6 + (k % 2) * 6.17
    y = 1.78 + (k // 2) * 2.62
    card(s, x, y, 5.96, 2.48, [P(t, size=15, bold=True, color=c, after=5), P(fait, size=13, bold=True, after=7), P(rep, size=13)], bar=c)
notes(s, """
Ce qui reste fragile.

Un : le contrôle qualité. J'ai annoté avant d'avoir un contrôle qualité sûr, et des enregistrements étouffés sont passés dans mes files. Heureusement, dans ce projet, un enregistrement écarté par un drapeau n'est jamais encodé : ses labels ne trouvent pas d'embedding et sortent tout seuls de l'entraînement et du test. Rien n'est effacé, et rien ne passe en silence.

Deux : le test est court en positifs. Deux options : un test_v2 aux heures de pic, sur les points des niveaux 1 et 3, avec la probabilité de tirage notée ; ou on garde le test et on rapporte chaque niveau avec son intervalle. C'est à décider ensemble.

Trois : les seuils de 0,07 et 0,2 seconde. Ils sont réglés à l'œil, mais par treize heures de tracé : ce n'est pas un seuil au hasard. On pourrait le vérifier en retraçant un échantillon.

Quatre : la piste écoutée. C'est corrigé : désormais chaque intervalle garde la piste écoutée au moment où il est tracé.
""")

# ======================================================================== 23 · prochainement
s = content("PROCHAINEMENT", "D'ici la prochaine réunion : la vérification, la piste 1, puis le benchmark sur nos données")
todo = [
    ("Vérification par Élodie et Benoît", "tirage au hasard de mes extraits, un paquet pour chacun ; puis le go, ou les strates à réécouter"),
    ("Piste 1", "encoder sur la piste 1 les enregistrements commentés ; mesurer l'écart de niveau sur les autres positifs ; réécouter les cas nets"),
    ("Fin de l'encodage", "les stocks refaits sur la bonne piste, pour les quatre encodeurs"),
    ("Benchmark sur nos données", "dès le go : encodeurs et têtes, niveaux 1, 2 et 3 rapportés à part"),
    ("Présentation n° 6 : les encodeurs", "pourquoi ces quatre, une fiche par encodeur, ce qu'ils donnent sur AnuraSet"),
]
for k, (t, d) in enumerate(todo):
    y = 1.8 + k * 0.82
    box(s, 0.6, y, 7.4, 0.72, PALER, radius=0.1)
    text(s, 0.75, y, 0.45, 0.72, P(str(k + 1), size=22, color=TEAL, bold=True, font="Cambria"), anchor="m")
    text(s, 1.3, y + 0.04, 6.55, 0.66, [P(t, size=13, bold=True, after=0), P(d, size=11, color=MUTED)], anchor="m")
card(s, 8.2, 1.8, 4.53, 4.0, [
    P("Décisions attendues", size=15, bold=True, color=GOLD_DARK, after=6),
    P(R("Vérification · ", bold=True), "combien d'extraits chacun, et sous quel délai ?", size=12.5, after=5),
    P(R("Jeu de test · ", bold=True), "un test_v2 pour atteindre 60 positifs, tiré où ? Ou des intervalles plus larges ?", size=12.5, after=5),
    P(R("Niveau 2 · ", bold=True), "RNRT sans chant : le garder pour les fausses alarmes, ou chercher un autre site ?", size=12.5, after=5),
    P(R("Piste 1 · ", bold=True), "encoder la piste 1 des positifs entendus là, ou les réécouter sur la piste 0 ?", size=12.5),
], bar=GOLD_DARK)
notes(s, """
D'ici la prochaine fois : la vérification par Élodie et Benoît, le rattrapage de la piste 1, la fin de l'encodage, puis, dès le go, le benchmark sur nos données. Et la présentation sur les encodeurs.

J'ai besoin de vous sur quatre décisions : combien d'extraits chacun vérifie et sous quel délai ; s'il faut un test_v2 et où le tirer ; ce qu'on fait du niveau 2 sans chant ; et la piste 1, encoder ou réécouter.
""")

prs.save(SORTIE)
print(SORTIE, len(prs.slides), "diapos")
