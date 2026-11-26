# ruff: noqa: E501
"""Présentation PNA A. blanci, v2 (restitution de stage, 15 min), animée par la Morphose.

    uv run --with python-pptx python documentation/prez/scripts/presentation_7/deck.py [sortie.pptx]

Sortie par défaut : `documentation/prez/presentation_7_pna_v5.pptx` (v1 à v4 : versions de Léonard,
ne jamais écrire par-dessus). Les diapos sans animation entrent par une poussée latérale. Lit dans `rapports/presentation_7/` : `donnees.json`
(donnees.py, BirdNET 2.4), `onde.npz`, `fenetre_spectro.png`, les figures de figures_v4.py, figures_v5.py et animation_reseau.py,
`inv_sites_pna.png`, `chiffres_pna.json` (figures_pna.py), `spectro.png` et `extrait.wav`
(figures_audio.py) ; et `documentation/prez/figures/titre.png`.

Chaque forme animée porte un nom « !!… » : d'une diapo à l'autre, PowerPoint fait glisser,
grandir et recolorer les formes qui portent le même nom (Morphose, PowerPoint 2019 / Microsoft
365 ; un PowerPoint plus ancien fait un fondu). La forme d'onde est le vrai signal (minimum et
maximum par tranche) ; le zoom sur la fenêtre de 3 s chasse le reste de l'enregistrement hors de
l'écran. Les vues du nuage sont de vraies projections linéaires des 1 024 nombres de BirdNET.
Aucune diapo n'avance seule (une diapo à avance automatique empêche le retour en arrière) ; le
réseau de neurones de la diapo « dans BirdNET » est un GIF animé, que PowerPoint joue en boucle.
Écrit aussi `planche.png` (contrôle visuel des nuages) dans `rapports/presentation_7/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

DEPOT = Path(__file__).resolve().parents[4]
ICI = DEPOT / "rapports" / "presentation_7"  # données, jamais committées
FIG = DEPOT / "documentation" / "prez" / "figures"
SORTIE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEPOT / "documentation" / "prez" / "presentation_7_pna_v5.pptx"
D = json.loads((ICI / "donnees.json").read_text(encoding="utf-8"))
K = json.loads((ICI / "chiffres_pna.json").read_text(encoding="utf-8"))
N = D["nuage"]
DIM = D["dim"]

DARK, INK, MUTED, SAGE, PALE, PALER = "0E2A24", "1B2A24", "5B6B63", "A7BFB2", "E4EEE8", "F3F7F4"
GOLD, TEAL, TERRA = "E9B44C", "2A7F72", "C8553D"
GRIS, GRIS_PALE = "B9C3BD", "E3E8E5"
ECRAN, ONDE = "0D1A1F", "CFE6EE"  # fond et trait de la forme d'onde (comme un écran d'enregistreur)
POS, MID, NEG = "1E6B45", "E9B44C", "C8553D"  # vert foncé → or → rouge terre : la clarté change aussi
GROUPES = ["3E7CB1", "D9822B", "8A6BBE", "2A7F72"]
SITES = {"Kaw": "3E7CB1", "Molokoi": "D9822B", "Tresor": "8A6BBE", "Mataroni": "2A7F72"}
NOM_SITE = {"Kaw": "Kaw", "Molokoi": "Molokoï", "Tresor": "Trésor", "Mataroni": "Mataroni"}


def fr(x, d=0):
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace(" ", " ")


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "p14": "http://schemas.microsoft.com/office/powerpoint/2010/main",
    "p159": "http://schemas.microsoft.com/office/powerpoint/2015/09/main",
}
PAGE = [0]
PLANCHE = []


def rgb(h):
    return RGBColor.from_string(h)


def morph(slide, ms=1200, auto=False):
    """Transition Morphose (par objet) ; `auto` : la diapo suivante vient seule, sans clic."""
    adv = ' advClick="0" advTm="0"' if auto else ""
    xml = (
        f'<mc:AlternateContent xmlns:mc="{NS["mc"]}" xmlns:p="{NS["p"]}">'
        f'<mc:Choice xmlns:p159="{NS["p159"]}" xmlns:p14="{NS["p14"]}" Requires="p159">'
        f'<p:transition spd="slow" p14:dur="{ms}"{adv}><p159:morph option="byObject"/></p:transition>'
        f'</mc:Choice><mc:Fallback><p:transition spd="slow"{adv}><p:fade/></p:transition></mc:Fallback>'
        "</mc:AlternateContent>"
    )
    sld = slide._element
    clr = sld.find(qn("p:clrMapOvr"))
    el = etree.fromstring(xml)
    if clr is not None:
        clr.addnext(el)
    else:
        sld.append(el)


ZOOM = []  # [(cx, cy, k)] : tout ce qui est dessiné est agrandi k fois autour de (cx, cy)


def zt(x, y, w, h):
    if not ZOOM:
        return x, y, w, h
    cx, cy, k = ZOOM[0]
    return cx + (x - cx) * k, cy + (y - cy) * k, w * k, h * k


def pousse(slide):
    """Transition « Poussée » : la diapo précédente part à gauche, la nouvelle arrive de la droite."""
    sld = slide._element
    el = etree.fromstring(f'<p:transition xmlns:p="{NS["p"]}" spd="slow"><p:push dir="l"/></p:transition>')
    clr = sld.find(qn("p:clrMapOvr"))
    if clr is not None:
        clr.addnext(el)
    else:
        sld.append(el)


def boite(s, x, y, w, h, fill=None, line=None, forme=MSO_SHAPE.RECTANGLE, nom=None, lw=0.75, dash=False, radius=None):
    x, y, w, h = zt(x, y, w, h)
    sh = s.shapes.add_shape(forme, Inches(x), Inches(y), Inches(max(w, 0.001)), Inches(max(h, 0.001)))
    if nom:
        sh.name = nom
    if radius is not None and forme == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = 4
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def texte(s, x, y, w, h, lignes, size=14, color=INK, bold=False, font="Calibri", align="l", nom=None, anchor="t", after=8, italic=False):
    x, y, w, h = zt(x, y, w, h)
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if nom:
        tb.name = nom
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for k, ligne in enumerate([lignes] if isinstance(lignes, str) else lignes):
        para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        para.space_after = Pt(after)
        runs = ligne if isinstance(ligne, list) else [ligne]
        for r in runs:
            t, kw = (r, {}) if isinstance(r, str) else r
            run = para.add_run()
            run.text = t
            f = run.font
            f.name = kw.get("font", font)
            f.size = Pt(kw.get("size", size))
            f.bold = kw.get("bold", bold)
            f.italic = kw.get("italic", italic)
            f.color.rgb = rgb(kw.get("color", color))
    return tb


def B(t):
    return (t, {"bold": True})


def fleche(s, x1, y1, x2, y2, color=SAGE, width=1.5, nom=None, pointe=True):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    if nom:
        c.name = nom
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if pointe:
        ln = c.line._get_or_add_ln()
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def notes(s, t):
    s.notes_slide.notes_text_frame.text = " ".join(t.split())


def fond(s, couleur="FFFFFF"):
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(couleur)


def diapo(rubrique, titre, note, ms=1200, auto=False, transition=True):
    s = prs.slides.add_slide(BLANK)
    fond(s)
    w = 0.083 * len(rubrique) + 0.32
    boite(s, 0.6, 0.4, w, 0.32, PALE, nom="!!rub")
    texte(s, 0.6, 0.4, w, 0.32, rubrique, size=10, color=TEAL, bold=True, align="c", anchor="m", nom="!!rubt")
    texte(s, 0.6, 0.8, 12.1, 0.8, titre, size=22, bold=True, font="Cambria", nom="Titre")
    PAGE[0] += 1
    texte(s, 12.33, 7.08, 0.5, 0.25, str(PAGE[0]), size=9, color=SAGE, align="r")
    notes(s, note)
    if transition == "pousse":
        pousse(s)
    elif transition:
        morph(s, ms, auto)
    return s


def panneau(s, lignes, y=1.75, h=4.95, size=15):
    boite(s, 9.0, y, 3.75, h, PALER, nom="!!pan")
    texte(s, 9.25, y + 0.22, 3.3, h - 0.4, lignes, size=size, nom="Texte", after=10)


# ------------------------------------------------------------------ nuage
GX, GY, GW, GH = 0.7, 1.75, 7.9, 4.95
R_PT, LW_PT = 0.085, 0.75
NPTS = len(N["site"])
HERO = NPTS - 1


def ecran(u, v, box=(GX, GY, GW, GH)):
    x0, y0, w, h = box
    u, v = np.asarray(u), np.asarray(v)
    lo_u, hi_u = np.percentile(u, [1, 99])
    lo_v, hi_v = np.percentile(v, [1, 99])
    x = x0 + (np.clip(u, lo_u, hi_u) - lo_u) / (hi_u - lo_u) * (w - R_PT)
    y = y0 + (1 - (np.clip(v, lo_v, hi_v) - lo_v) / (hi_v - lo_v)) * (h - R_PT)
    return x, y, (lo_u, hi_u)


CX0, CY0 = GX + GW / 2, GY + GH / 2
VUES = [ecran(*v) for v in N["vues"]]
# chaque vue remplit le cadre (sinon la vue finale s'écrase : le nuage s'étale peu le long de la
# direction de la tête) ; le cube suit exactement la même mise à l'échelle que les points
ECH = []
for u, v in N["vues"]:
    lo_u, hi_u = np.percentile(u, [1, 99])
    lo_v, hi_v = np.percentile(v, [1, 99])
    ECH.append((lo_u, hi_u, lo_v, hi_v))


def vers_ecran(k, u, v):
    lo_u, hi_u, lo_v, hi_v = ECH[k]
    return GX + R_PT / 2 + (u - lo_u) / (hi_u - lo_u) * (GW - R_PT), GY + R_PT / 2 + (1 - (v - lo_v) / (hi_v - lo_v)) * (GH - R_PT)


def degrade(p):
    def mix(c1, c2, t):
        a_, b_ = [int(c1[k:k + 2], 16) for k in (0, 2, 4)], [int(c2[k:k + 2], 16) for k in (0, 2, 4)]
        return "".join(f"{round(u + (v - u) * t):02X}" for u, v in zip(a_, b_, strict=True))
    return mix(NEG, MID, p * 2) if p < 0.5 else mix(MID, POS, (p - 0.5) * 2)


def nuage(s, x, y, couleurs, contours=None, titre="", seulement=None, rayon=None):
    if titre:
        PLANCHE.append((titre, np.asarray(x), np.asarray(y), couleurs, contours))
    for i in range(NPTS) if seulement is None else seulement:
        r = R_PT if rayon is None else rayon.get(i, R_PT)
        line = contours[i] if contours is not None else None
        boite(s, x[i] - (r - R_PT) / 2, y[i] - (r - R_PT) / 2, r, r, couleurs[i], line=line, forme=MSO_SHAPE.OVAL, nom=f"!!p{i}", lw=LW_PT)


def legende(s, items, y0=6.82, x=GX):
    for coul, lib, contour in items:
        boite(s, x, y0 + 0.06, 0.13, 0.13, coul, line=INK if contour else None, forme=MSO_SHAPE.OVAL, nom=f"!!leg_{lib}", lw=LW_PT)
        texte(s, x + 0.22, y0, 2.6, 0.3, lib, size=12, color=MUTED, nom=f"!!legt_{lib}")
        x += 0.36 + 0.08 * len(lib) + 0.25


ANGLES = {}  # dernier angle de chaque segment : la Morphose tourne d'un angle à l'autre sans saut
COUL_AXE = {"X": "3E7CB1", "Y": "D9822B", "Z": "8A6BBE"}


def segment(s, x1, y1, x2, y2, nom, coul, ep=0.014):
    L = float(np.hypot(x2 - x1, y2 - y1))
    ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) if L > 1e-3 else ANGLES.get(nom, 0.0)
    prec = ANGLES.get(nom)
    if prec is None:  # même segment, tourné de 180° au besoin, entre 90 et 270 : loin du passage 360 → 0
        ang = ang % 360
        ang = ang + 180 if ang < 90 else ang - 180 if ang >= 270 else ang
    else:
        ang = min((ang + d for d in (-360, -180, 0, 180, 360)), key=lambda a: abs(a - prec))
    ANGLES[nom] = ang
    sh = boite(s, (x1 + x2) / 2 - L / 2, (y1 + y2) / 2 - ep / 2, max(L, 0.002), ep, coul, nom=nom)
    sh.rotation = ang % 360


def repere(s, k, cx=10.875, cy=5.72, c=0.5):
    """Cube et axes X, Y, Z (à échelle fixe, dans le panneau) de l'espace à 3 dimensions dont
    toutes les vues du nuage sont des ombres : X = ACP 1, Y = ACP 2, Z = direction de la tête
    rendue orthogonale aux deux. Il tourne avec exactement les mêmes vecteurs que le nuage."""
    T = N["triedre"][k]

    def proj(a, b, d):
        u = a * T["X"][0] + b * T["Y"][0] + d * T["Z"][0]
        v = a * T["X"][1] + b * T["Y"][1] + d * T["Z"][1]
        return cx + c * u, cy - c * v

    coins = [(a, b, d) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]
    n = 0
    for i, p_ in enumerate(coins):
        for q in coins[i + 1:]:
            if sum(x != y for x, y in zip(p_, q, strict=True)) == 1:
                segment(s, *proj(*p_), *proj(*q), f"!!cube{n}", "B9C3BD", ep=0.014)
                n += 1
    for nom, (a, b, d) in (("X", (1.6, 0, 0)), ("Y", (0, 1.6, 0)), ("Z", (0, 0, 1.6))):
        x2, y2 = proj(a, b, d)
        segment(s, cx, cy, x2, y2, f"!!axe{nom}", COUL_AXE[nom], ep=0.035)
        texte(s, x2 - 0.15 + (0.13 if x2 >= cx else -0.13), y2 - 0.17 + (0.11 if y2 >= cy else -0.11), 0.3, 0.3, nom, size=14, bold=True, color=COUL_AXE[nom], align="c", anchor="m", nom=f"!!axel{nom}", after=0)


SC = N["scores"]
Y = np.array(N["y"])
TOUR = np.array(N["tour"])
SITE = N["site"]


def etat(t_rev, version=None):
    """Couleurs et contours : annoté (révélé au tour ≤ t_rev) en vert/rouge cerclé ; sinon gris
    ou couleur du score de la version `version` de la tête."""
    coul, cont = [], []
    for i in range(NPTS):
        if TOUR[i] == -2 or (0 <= TOUR[i] <= t_rev):
            coul.append(POS if Y[i] == 1 else NEG)
            cont.append(INK)
        else:
            coul.append(GRIS if version is None else degrade(SC[version][i]))
            cont.append(None)
    return coul, cont


def jauge(s, t):
    """Score du modèle (AP sur des micros jamais vus) après le tour t du rejeu."""
    ap, n = N["ap_rejeu"][t], N["n_rejeu"][t]
    x, y, w = 9.25, 5.55, 3.25
    texte(s, x, y, w, 0.26, "Score du modèle (micros jamais entendus)", size=11, color=MUTED, nom="!!jt")
    boite(s, x, y + 0.32, w, 0.26, GRIS_PALE, nom="!!jf")
    boite(s, x, y + 0.32, w * ap, 0.26, POS, nom="!!jb")
    texte(s, x, y + 0.64, w, 0.3, [[B(fr(ap, 2)), f"   après {n} fenêtres annotées"]], size=12, color=INK, nom="!!jv")


# ================================================================== diapos
RUB_I, RUB_E, RUB_C, RUB_K, RUB_S = "Le projet", "1 · Encoder", "2 · Regrouper", "3 · Apprendre", "La suite"
C = K["corpus"]
heures = C["heures"]
jours = heures / 7

# 1. titre -----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
fond(s, DARK)
s.shapes.add_picture(str(FIG / "titre.png"), Inches(8.9), Inches(0), height=Inches(7.5))
texte(s, 0.8, 1.6, 7.6, 0.4, "PLAN NATIONAL D'ACTIONS", size=12, color=GOLD, bold=True)
texte(s, 0.8, 2.1, 7.7, 3.0, [[("Outil de détection automatique d'", {}), ("Anomaloglossus blanci", {"italic": True})]], size=40, color="FFFFFF", bold=True, font="Cambria", after=0)
texte(s, 0.8, 5.75, 7.6, 0.35, "Léonard Laplace-Palette", size=16, color="FFFFFF", bold=True)
texte(s, 0.8, 6.15, 7.6, 0.35, "Restitution de stage · PNA · Jeudi 26 Novembre 2026", size=13, color=SAGE)
texte(s, 9.1, 7.05, 4.0, 0.3, "Chant d'A. blanci : les notes en forme de virgule, vers 4,7 kHz", size=9, color="FFFFFF", italic=True)
notes(s, """Bonjour. Je présente mon stage : un outil qui détecte automatiquement le chant d'Anomaloglossus blanci dans les enregistrements des campagnes de suivi. L'image à droite est un spectrogramme : le temps de gauche à droite, la hauteur du son de bas en haut. Les petites virgules claires, ce sont les notes d'A. blanci.""")

# 2. l'idée ----------------------------------------------------------
s = diapo(RUB_I, "L'idée : automatiser l'écoute de dizaines de milliers d'enregistrements", "", transition="pousse")
for k, (gros, petit) in enumerate([
    (fr(C["enregistrements"]), "enregistrements de 2 minutes"),
    (f"{fr(heures)} h", f"d'audio · {C['sites']} sites · {C['points']} points d'écoute"),
    (f"{fr(jours)} jours", "d'écoute à 7 h par jour"),
]):
    x = 0.6 + k * 4.1
    boite(s, x, 1.85, 3.85, 1.6, PALER, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    texte(s, x + 0.25, 1.98, 3.4, 0.7, gros, size=30, color=TEAL if k < 2 else TERRA, bold=True, font="Cambria")
    texte(s, x + 0.25, 2.72, 3.4, 0.6, petit, size=13, color=MUTED)
texte(s, 0.6, 3.95, 12.1, 0.4, "L'outil : deux réseaux de neurones", size=18, bold=True, color=INK)
etapes2 = [
    ("Enregistrement", "2 minutes de forêt", PALER, INK),
    ("BirdNET", "réseau n° 1 : résume le son", DARK, "FFFFFF"),
    ("Ma tête de classification", "réseau n° 2 : reconnaît A. blanci", TEAL, "FFFFFF"),
    ("A. blanci ?", "oui / non, et avec quelle certitude", PALE, INK),
]
for k, (t, st, fill, coul) in enumerate(etapes2):
    x = 0.6 + k * 3.15
    boite(s, x, 4.6, 2.75, 1.55, fill, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    texte(s, x + 0.15, 4.78, 2.45, 0.5, t, size=17, bold=True, color=coul, align="c")
    texte(s, x + 0.15, 5.35, 2.45, 0.7, st, size=12, color=coul if fill in (DARK, TEAL) else MUTED, align="c")
    if k < 3:
        fleche(s, x + 2.78, 5.37, x + 3.12, 5.37, color=SAGE, width=2.2)
notes(s, f"""Depuis 2023, des enregistreurs captent deux minutes de forêt toutes les demi-heures. Aujourd'hui, {fr(C['enregistrements'])} enregistrements contrôlés, {fr(heures)} heures, sur {C['sites']} sites. Pour tout écouter, il faudrait {fr(jours)} journées de sept heures : plus de deux ans à plein temps. L'idée est donc d'automatiser l'écoute. L'outil enchaîne deux réseaux de neurones. Le premier, BirdNET, développé par l'université Cornell, résume chaque morceau de son en une liste de nombres. Le second, que j'entraîne moi-même sur mes écoutes, lit ces nombres et dit si A. blanci chante, avec une certitude.""")

# 3. ce qu'on cherche (audio) ----------------------------------------
s = diapo(RUB_I, "Ce qu'on cherche : quelques notes brèves dans le bruit de la forêt", "", transition="pousse")
pic = ICI / "spectro_grand.png"
try:
    s.shapes.add_movie(str(ICI / "extrait.wav"), Inches(0.6), Inches(1.65), Inches(12.1), Inches(4.7), poster_frame_image=str(pic), mime_type="audio/wav")
    audio_ok = True
except Exception as exc:  # noqa: BLE001
    print("audio non inséré :", exc)
    s.shapes.add_picture(str(pic), Inches(0.6), Inches(1.65), width=Inches(12.1))
    audio_ok = False
texte(s, 0.6, 6.45, 12.1, 0.4, "10 s enregistrées à Molokoï (point F, 26/02/2024, 13 h 30) ; triangles : les notes d'A. blanci, vers 4,7 kHz. Cliquer sur l'image pour écouter.", size=12, color=MUTED, italic=True)
notes(s, """Voici dix secondes à Molokoï. Je lance le son. Les petites virgules vers 4,7 kilohertz, ce sont les notes d'A. blanci : une toute petite note, toutes les trois ou quatre secondes. Au-dessus, la bande continue, ce sont des insectes ; en dessous, d'autres espèces. Retenez cette image : le chant prend très peu de place, dans le temps comme en fréquence. C'est toute la difficulté.""")

# 4. inventaire + contrôle qualité -----------------------------------
INV = K["inventaire"]
s = diapo(RUB_I, f"Avant toute IA : inventorier et contrôler {fr(round(INV['enregistrements'], -3))} enregistrements", "", transition="pousse")
s.shapes.add_picture(str(ICI / "inv_sites_pna.png"), Inches(0.5), Inches(1.75), width=Inches(8.2))
texte(s, 0.6, 6.15, 8.0, 0.6, "Heures enregistrées par lieu ; les relevés 2025-2026 (dont Gabaret et Pai, et Mataroni jusqu'en mai) empilés sur leur lieu.", size=11, color=MUTED, italic=True)
boite(s, 9.0, 1.75, 3.75, 3.35, PALER, nom="!!pan")
texte(s, 9.25, 1.95, 3.3, 3.0, [
    [B("Contrôle qualité")],
    "• Deux micros par enregistreur : pour chaque enregistrement, la piste exploitable (gain inversé, piste cassée, son étouffé).",
    f"• {fr(INV['enregistrements'] - C['enregistrements'])} enregistrements inutilisables écartés.",
    [B("Corpus figé le 23/11/2026")],
    f"{fr(C['enregistrements'])} enregistrements · {fr(heures)} h",
], size=13, after=6)
boite(s, 9.0, 5.3, 3.75, 1.4, "FFF6E3", forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
boite(s, 9.0, 5.39, 0.06, 1.22, GOLD)
texte(s, 9.25, 5.42, 3.35, 1.2, [[B("Vous avez des enregistrements, avec ou sans A. blanci ?")], ["Ils m'aideront à entraîner le modèle : je les prends volontiers."]], size=13, after=4)
notes(s, f"""Avant de parler d'intelligence artificielle, il a fallu savoir ce qu'on avait. L'inventaire : {fr(INV['enregistrements'])} fichiers, avec les relevés arrivés cette semaine, Gabaret, Pai et Mataroni jusqu'en mai. Des horloges à recaler, des doublons, des trous. Puis le contrôle qualité : chaque enregistreur a deux micros réglés différemment, et pour des milliers d'enregistrements la bonne piste n'est pas celle qu'on croit. {fr(INV['enregistrements'] - C['enregistrements'])} enregistrements sont écartés. Le corpus est figé depuis le 23 novembre : {fr(C['enregistrements'])} enregistrements. Et un appel : si vous avez des enregistrements, avec ou sans A. blanci, ils m'aideront à entraîner le modèle.""")

# ---------------------------------------------------------- 1 · encoder : forme d'onde et zoom
ONDE_NPZ = np.load(ICI / "onde.npz")
lo, hi, flo, fhi = ONDE_NPZ["lo"], ONDE_NPZ["hi"], ONDE_NPZ["flo"], ONDE_NPZ["fhi"]
PAS, FPAS = float(ONDE_NPZ["pas"]), float(ONDE_NPZ["fpas"])
F0, F1 = (float(v) for v in ONDE_NPZ["fen"])
NB, NF = len(lo), len(flo)
dans = set(range(int(round(F0 / PAS)), int(round(F1 / PAS))))
P = max(np.abs(lo).max(), np.abs(hi).max())
PW = max(np.abs(flo).max(), np.abs(fhi).max())
WX0, WX1 = 0.75, 12.58
PPS = (WX1 - WX0) / (NB * PAS)  # pouces par seconde, enregistrement entier
CY, HH = 3.85, 1.75
BW = 0.014
ZX0, ZW = 0.85, 5.45  # fenêtre zoomée : de ZX0 à ZX0 + ZW
ZPPS = ZW / (F1 - F0)
ZCY, ZHH = 4.2, 2.0


def barre(s, x, a, b, cy, hh, peak, nom, w=BW):
    top, h = cy - b / peak * hh, max(0.008, (b - a) / peak * hh)
    boite(s, x, top, w, h, ONDE, nom=nom)


def onde(s, mode):
    """mode : « entier » (2 min), « fenetres » (idem + grille de 3 s), « zoom » (la fenêtre seule)."""
    if mode == "zoom":
        boite(s, 0.6, 1.85, 5.95, 4.7, ECRAN, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03, nom="!!ecran")
    else:
        boite(s, 0.6, 1.85, 12.13, 4.0, ECRAN, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03, nom="!!ecran")
    for i in range(NB):
        if i in dans:
            continue
        t = i * PAS
        if mode == "zoom":  # zoom : le reste de l'enregistrement est chassé hors de l'écran
            x = ZX0 + (t - F0) * ZPPS
            x = min(x, -0.6) if t < F0 else max(x, 13.6)
            barre(s, x, lo[i], hi[i], ZCY, ZHH, PW, f"!!o{i}", w=BW * ZPPS / PPS)
        else:
            barre(s, WX0 + t * PPS, lo[i], hi[i], CY, HH, P, f"!!o{i}")
    for j in range(NF):
        t = F0 + j * FPAS
        if mode == "zoom":
            barre(s, ZX0 + (t - F0) * ZPPS, flo[j], fhi[j], ZCY, ZHH, PW, f"!!z{j}")
        else:
            barre(s, WX0 + t * PPS, flo[j], fhi[j], CY, HH, P, f"!!z{j}")
    if mode == "fenetres":
        for k in range(1, int(NB * PAS / 3)):
            boite(s, WX0 + 3 * k * PPS - 0.004, 1.95, 0.008, 3.8, "2C4048", nom=f"!!sep{k}")
    if mode == "fenetres":
        boite(s, WX0 + F0 * PPS - 0.03, 1.93, (F1 - F0) * PPS + 0.06, 3.84, None, line=GOLD, lw=2.5, nom="!!fen")
    elif mode == "zoom":
        boite(s, 0.68, 1.93, 5.79, 4.54, None, line=GOLD, lw=2.5, nom="!!fen")
    if mode == "zoom":
        texte(s, ZX0, 6.6, 1.0, 0.3, "0 s", size=12, color=MUTED, nom="!!t0")
        texte(s, ZX0 + ZW - 1.0, 6.6, 1.0, 0.3, "3 s", size=12, color=MUTED, align="r", nom="!!t1")
    else:
        texte(s, WX0, 5.95, 1.0, 0.3, "0 s", size=12, color=MUTED, nom="!!t0")
        texte(s, WX1 - 1.0, 5.95, 1.0, 0.3, "2 min", size=12, color=MUTED, align="r", nom="!!t1")


s = diapo(RUB_E, "Un enregistrement de 2 minutes", "", transition="pousse")
onde(s, "entier")
notes(s, """Voici un enregistrement de deux minutes, tel que l'ordinateur le reçoit : la forme du signal sonore. Quand la forêt est calme, c'est presque une ligne ; chaque son fait dresser des barres. Les pics réguliers, ce sont les notes d'A. blanci : dans cet enregistrement-là, c'est le son le plus fort.""")

s = diapo(RUB_E, "On le découpe en fenêtres de 3 secondes", "", ms=900)
onde(s, "fenetres")
notes(s, """BirdNET écoute trois secondes à la fois : on découpe donc l'enregistrement en quarante fenêtres. Tout le reste de la méthode travaille fenêtre par fenêtre ; à la fin, un enregistrement prend le score de sa meilleure fenêtre. Suivons celle-ci, encadrée en jaune.""")

V = D["vecteur"]
vals = V["haut"] + V["bas"]
VX, VW, VY0, VH = 9.3, 1.35, 1.95, 0.255


def case(v):
    t = min(1.0, max(0.0, (v - V["corpus_min"]) / (V["corpus_max"] - V["corpus_min"])))
    base = (0x9C, 0xCF, 0xC4)  # coloration légère
    return "".join(f"{round(255 + (c - 255) * t):02X}" for c in base)


def vecteur(s, x, crochet=INK, legende_coul=MUTED, legende=None, p="v"):
    rows = [("v", j) for j in range(8)] + [("e", None)] + [("v", 8 + j) for j in range(8)]
    for r, (kind, j) in enumerate(rows):
        y = VY0 + r * VH
        if kind == "e":
            texte(s, x, y - 0.04, VW, VH + 0.05, "⋮", size=16, color=MUTED, align="c", nom=f"!!{p}ell", anchor="m", after=0)
            continue
        boite(s, x, y, VW, VH, case(vals[j]), line="C9D3CE", lw=0.5, nom=f"!!{p}{j}")
        texte(s, x, y, VW, VH, fr(vals[j], 2), size=13, color=INK, align="c", anchor="m", nom=f"!!{p}t{j}", after=0)
    bas = VY0 + len(rows) * VH
    for side, xx in (("g", x - 0.08), ("d", x + VW + 0.06)):
        boite(s, xx, VY0 - 0.04, 0.02, bas - VY0 + 0.08, crochet, nom=f"!!{p}bk{side}")
        boite(s, min(xx, xx + 0.08 * (1 if side == "g" else -1)), VY0 - 0.04, 0.1, 0.02, crochet, nom=f"!!{p}bk{side}h")
        boite(s, min(xx, xx + 0.08 * (1 if side == "g" else -1)), bas + 0.02, 0.1, 0.02, crochet, nom=f"!!{p}bk{side}b")
    texte(s, x - 0.6, bas + 0.12, VW + 1.2, 0.3, legende or f"16 nombres affichés sur {fr(DIM)}", size=12, color=legende_coul, align="c", nom=f"!!{p}lab")
    return VY0 + (bas - VY0) / 2


BNB = (7.0, 3.8, 1.65, 0.8)  # la boîte BirdNET de la diapo 7
NOTE7 = f"""Zoomons sur cette fenêtre de trois secondes : la note d'A. blanci au milieu, et le bruit de la forêt autour. BirdNET la transforme en {fr(DIM)} nombres, rangés dans une colonne : on appelle ça un vecteur. J'en affiche seize. Je ne modifie pas BirdNET : je m'en sers comme d'une oreille déjà exercée. Regardons ce qu'il y a dans la boîte."""


def diapo7(note):
    s = diapo(RUB_E, f"Une fenêtre de 3 secondes devient {fr(DIM)} nombres", "", ms=1600)
    onde(s, "zoom")
    fleche(s, 6.62, 4.2, 6.95, 4.2, color=SAGE, width=2.2, nom="!!fa1")
    boite(s, *BNB, DARK, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, nom="!!bn")
    texte(s, *BNB, [[B("BirdNET")]], size=17, color="FFFFFF", align="c", anchor="m", nom="!!bnt")
    fleche(s, 8.7, 4.2, 9.1, 4.2, color=SAGE, width=2.2, nom="!!fa2")
    notes(s, note)
    return vecteur(s, VX)


vmid = diapo7(NOTE7)

# dans BirdNET : la boîte grandit jusqu'à remplir l'écran, le reste est chassé au loin
s = diapo(RUB_E, "BirdNET : un réseau de neurones qui reconnaît des motifs dans les spectrogrammes", "", ms=1700)
ZOOM.append((BNB[0] + BNB[2] / 2, BNB[1] + BNB[3] / 2, 9.0))
onde(s, "zoom")
vecteur(s, VX)  # chassé hors de l'écran, sous la boîte qui grandit
ZOOM.clear()
boite(s, 0.45, 1.6, 12.45, 5.4, DARK, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03, nom="!!bn")
texte(s, 0.75, 1.75, 3.0, 0.45, [[B("BirdNET")]], size=18, color="FFFFFF", nom="!!bnt")
s.shapes.add_picture(str(ICI / "fenetre_spectro_sombre.png"), Inches(0.75), Inches(2.4), Inches(2.85), Inches(2.67))
texte(s, 0.75, 5.12, 2.85, 0.3, "la fenêtre de 3 s, en spectrogramme", size=11, color=SAGE, align="c")
fleche(s, 3.68, 3.73, 4.05, 3.73, color=GOLD, width=2.2, nom="!!fa1")
s.shapes.add_picture(str(ICI / "reseau.gif"), Inches(4.1), Inches(2.2), Inches(5.05), Inches(3.03))
fleche(s, 9.2, 3.73, 9.6, 3.73, color=GOLD, width=2.2, nom="!!fa2")
vecteur(s, 10.0, crochet="FFFFFF", legende_coul=SAGE, legende=f"16 des {fr(DIM)} nombres", p="iv")
texte(s, 0.75, 5.55, 8.6, 1.3, [[
    ("Entraîné sur les chants de plus de 6 000 espèces, il a appris à reconnaître des motifs dans les spectrogrammes. Il en ressort une ", {}),
    ("signature", {"bold": True, "color": GOLD}),
    (f" : l'embedding, un vecteur qui place la fenêtre de 3 s dans un espace à {fr(DIM)} dimensions.", {}),
]], size=15, color="FFFFFF")
notes(s, f"""Ouvrons la boîte. BirdNET est un réseau de neurones, développé par l'université Cornell. On lui donne la fenêtre sous forme de spectrogramme, une image du son. Couche après couche, ses neurones s'activent : ils ont appris, sur les chants de plus de 6 000 espèces, à reconnaître des motifs dans ces images, une note qui monte, un trille, un bourdonnement. De tout ce travail ressort une signature : {fr(DIM)} nombres, l'embedding. C'est ce vecteur. Il place notre fenêtre de trois secondes dans un espace à {fr(DIM)} dimensions. Que veut dire chaque nombre, pris seul ? Personne ne sait le dire : c'est la boîte noire. Mais deux sons qui se ressemblent ont des signatures proches.""")

# retour : la boîte se referme
vmid = diapo7("""Refermons la boîte : une fenêtre de trois secondes, une signature de 1 024 nombres. Voyons maintenant ce qu'on fait de ces signatures.""".replace("1 024", fr(DIM)))

# ---------------------------------------------------------- 2 · regrouper
u0, v0 = N["vues"][0]
BOXR = (3.3, 1.85, 9.4, 4.75)
BOXL = (0.6, 1.85, 5.9, 4.75)
xr, yr, _ = ecran(u0, v0, BOXR)
xl, yl, _ = ecran(u0, v0, BOXL)
gris = [GRIS] * NPTS
gris[HERO] = GOLD
cont_h = [None] * NPTS
cont_h[HERO] = INK
gros = {HERO: 0.2}

s = diapo(RUB_C, "Chaque fenêtre devient un point dans l'espace", "", ms=1600)
vecteur(s, 0.75)
nuage(s, xr, yr, gris, cont_h, "nuage", rayon=gros)
fleche(s, 0.75 + VW + 0.12, vmid, xr[HERO] - 0.02, yr[HERO] + R_PT / 2, color=GOLD, width=1.5, nom="!!lien", pointe=True)
texte(s, 3.3, 6.68, 9.4, 0.3, f"Un point = une fenêtre de 3 s ; ici {NPTS} fenêtres de Kaw, Molokoï, Trésor et Mataroni, vues à plat.", size=12, color=MUTED, nom="!!cap")
notes(s, f"""Une colonne de {fr(DIM)} nombres, c'est une position dans un espace à {fr(DIM)} dimensions : chaque fenêtre devient un point. Notre fenêtre, c'est le point jaune. Les autres points sont {NPTS} fenêtres de quatre forêts. Un espace à {fr(DIM)} dimensions ne se dessine pas : on le met à plat, comme une ombre portée. Deux fenêtres qui se ressemblent sont deux points proches.""")

s = diapo(RUB_C, "Les fenêtres qui se ressemblent forment des groupes", "", ms=1000)
vecteur(s, 0.75)
coulg = [GROUPES[g] for g in N["groupe"]]
nuage(s, xr, yr, coulg, cont_h, "groupes", rayon=gros)
fleche(s, 0.75 + VW + 0.12, vmid, xr[HERO] - 0.02, yr[HERO] + R_PT / 2, color=GOLD, width=1.5, nom="!!lien", pointe=True)
texte(s, 3.3, 6.68, 9.4, 0.3, "Groupes formés par l'ordinateur sur les 1 024 nombres, sans rien savoir du contenu des fenêtres.", size=12, color=MUTED, nom="!!cap")
notes(s, """Demandons à l'ordinateur de faire des paquets de fenêtres qui se ressemblent, sans rien lui dire de leur contenu. C'est le clustering. Il trouve des groupes. Mais qu'est-ce qui les fait, ces groupes ?""")

VAR = D["variance"]
nom_fac = {"site": "le site", "point d'écoute (dans son site)": "le point d'écoute, dans son site", "moment de la journée": "le moment de la journée", "mois": "le mois", "présence d'A. blanci": "la présence d'A. blanci"}
premier = "le lieu" if {VAR[0]["facteur"], VAR[1]["facteur"]} == {"site", "point d'écoute (dans son site)"} else nom_fac[VAR[0]["facteur"]]
pos_b = next(i for i, v in enumerate(VAR) if v["facteur"] == "présence d'A. blanci")
ensuite = {"mois": "le mois", "moment de la journée": "l'heure"}.get(VAR[2]["facteur"]) if premier == "le lieu" else None
titre_var = f"Ce qui rend deux fenêtres semblables : d'abord {premier}" + (f", puis {ensuite}" if ensuite else "") + ", presque pas A. blanci"
s = diapo(RUB_C, titre_var if pos_b >= 2 else "Ce qui rend deux fenêtres semblables", "", ms=1400)
nuage(s, xl, yl, coulg, cont_h, "facteurs", rayon=gros)
texte(s, 7.0, 1.85, 5.7, 0.6, "Part des différences entre fenêtres expliquée par chaque facteur", size=14, color=MUTED, nom="!!vt")
vmax = max(v["part"] for v in VAR) * 1.05
for r, v in enumerate(VAR):
    yb = 2.65 + r * 0.78
    blanci = v["facteur"] == "présence d'A. blanci"
    coul = POS if blanci else TEAL
    lib = v["facteur"][0].upper() + v["facteur"][1:]
    texte(s, 7.0, yb - 0.02, 2.35, 0.6, lib, size=14, bold=blanci, color=POS if blanci else INK, nom=f"!!vf{r}")
    lw = 2.55 * v["part"] / vmax
    boite(s, 9.45, yb, max(lw, 0.02), 0.32, coul, nom=f"!!vb{r}")
    texte(s, 9.55 + 2.55 * v["part"] / vmax, yb - 0.01, 0.9, 0.35, f"{fr(100 * v['part'], 1)} %", size=14, bold=True, color=coul, nom=f"!!vv{r}")
texte(s, 7.0, 6.5, 5.7, 0.4, f"{fr(D['n_annotees'])} fenêtres écoutées, {D['n_sites_annotes']} sites.", size=11, color=MUTED, nom="!!vn")
notes(s, f"""Pour chaque facteur, je mesure quelle part des différences entre fenêtres il explique, sur les {fr(D['n_annotees'])} fenêtres que j'ai écoutées. Dans l'ordre : {', '.join(f"{nom_fac[v['facteur']]}, {fr(100 * v['part'], 1)} %" for v in VAR)}. Ce qui rapproche deux fenêtres, c'est d'abord le lieu et l'ambiance du moment. La présence d'A. blanci n'explique presque rien. Chaque facteur est pris seul ; ils se recouvrent en partie, le mois et le site surtout, puisque chaque relevé a ses dates. Le moment de la journée et la présence d'A. blanci sont à égalité, tout en bas : la différence entre les deux n'est pas significative.""")

# dilution
CH = K["chant"]
s = diapo(RUB_C, "Dans une fenêtre, le chant d'A. blanci ne pèse presque rien", "", transition="pousse")
s.shapes.add_picture(str(ICI / "fenetre_spectro.png"), Inches(0.5), Inches(1.75), height=Inches(4.9))
stats = [
    (f"{fr(100 * CH['temps_mediane'])} %", f"du temps : une note de {fr(CH['duree_note_mediane_s'], 1)} s ({fr(100 * CH['temps_p25'])} à {fr(100 * CH['temps_p75'])} % selon le chant)"),
    (f"{fr(100 * CH['bande'])} %", "des fréquences : 4,4 à 5,5 kHz sur les 15 kHz que lit BirdNET"),
    (f"{fr(100 * CH['surface_mediane'], 1)} %", "de l'image"),
]
for r, (gros_, petit) in enumerate(stats):
    yb = 1.85 + r * 1.3
    boite(s, 7.4, yb, 5.3, 1.12, PALER, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    texte(s, 7.6, yb + 0.15, 1.6, 0.8, gros_, size=30, bold=True, font="Cambria", color=GOLD if r == 2 else TEAL)
    texte(s, 9.25, yb + 0.18, 3.3, 0.85, petit, size=13, color=INK, anchor="m")
texte(s, 7.4, 5.85, 5.3, 0.8, "L'information « A. blanci » est diluée dans un océan de bruit.", size=15, bold=True, color=INK)
notes(s, f"""Pourquoi ? Revenons à notre fenêtre de trois secondes, en spectrogramme. La note d'A. blanci est dans le cadre jaune. Elle dure environ {fr(CH['duree_note_mediane_s'], 1)} seconde : {fr(100 * CH['temps_mediane'])} % du temps de la fenêtre, entre {fr(100 * CH['temps_p25'])} et {fr(100 * CH['temps_p75'])} % selon les chants, mesuré sur {CH['n_fenetres']} fenêtres où j'ai marqué les notes une à une. Elle n'occupe que {fr(100 * CH['bande'])} % des fréquences. Au total, à peine {fr(100 * CH['surface_mediane'], 1)} % de l'image. Tout le reste, c'est la forêt. Les nombres de BirdNET résument toute la fenêtre : l'information sur la grenouille y est diluée dans un océan de bruit. Le regroupement automatique ne peut pas la faire ressortir seul : il faut lui apprendre ce qu'on cherche.""")

# ---------------------------------------------------------- 3 · apprendre
connu0 = (TOUR == -2) | (TOUR == 0)
u0a, v0a = np.array(u0), np.array(v0)
xs, ys = np.zeros(NPTS), np.zeros(NPTS)
L = (0.7, 2.35, 5.75, 4.25)
R = (6.95, 2.35, 5.75, 4.25)
for masque, box in ((connu0, L), (~connu0, R)):
    xx, yy, _ = ecran(u0a[masque], v0a[masque], box)
    xs[masque], ys[masque] = xx, yy
coul0, cont_0 = etat(0)
n_ann0 = int(connu0.sum())

s = diapo(RUB_K, "J'écoute et j'annote : la machine apprend sur mes exemples", "", transition="pousse")
boite(s, 6.66, 1.85, 0.02, 4.85, SAGE, nom="!!split")
texte(s, 0.7, 1.85, 5.75, 0.35, f"Annotées ({n_ann0})", size=15, color=INK, bold=True, nom="!!hl")
texte(s, 6.95, 1.85, 5.75, 0.35, f"En attente ({NPTS - n_ann0})", size=15, color=MUTED, bold=True, nom="!!hr")
nuage(s, xs, ys, coul0, cont_0, "split")
legende(s, [(POS, "A. blanci", True), (NEG, "autre son", True), (GRIS, "en attente", False)])
notes(s, f"""Alors on apprend. J'écoute des fenêtres et je les étiquette : A. blanci, ou autre chose. À ce jour, {fr(D['n_annotees'])} fenêtres écoutées, dont {fr(D['n_positives'])} avec le chant, sur {D['n_micros_annotes']} points d'écoute. À gauche, les fenêtres déjà écoutées ; à droite, l'immense majorité, en attente. On part de {N['n_rejeu'][0]} fenêtres annotées.""")

# la tête de classification : un seul neurone, et son entraînement passe par passe (GIF)
EN = D["entrainement"]
score_hero = EN["score_hero"][-1]
s = diapo(RUB_K, "La tête de classification : un neurone qui apprend un poids par nombre", "", transition="pousse")
boite(s, 0.45, 1.6, 12.45, 5.4, DARK, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.03)
vecteur(s, 1.15, crochet="FFFFFF", legende_coul=SAGE, legende="la signature de notre fenêtre", p="tv")
s.shapes.add_picture(str(ICI / "tete_apprend.gif"), Inches(2.72), Inches(1.7), Inches(5.95), Inches(5.2))
texte(s, 8.95, 1.85, 3.8, 0.45, [[B("Ma tête de classification")]], size=18, color="FFFFFF")
texte(s, 8.95, 2.45, 3.75, 4.4, [
    ["Un seul neurone : chaque nombre de la signature est multiplié par son poids, puis tout est additionné. La somme, ramenée entre 0 et 1, donne le score."],
    ["Au départ, tous les poids valent 0 : le score vaut 0,5 partout."],
    ["À chaque passe sur les fenêtres annotées, chaque poids est corrigé un peu : le score monte vers 1 quand j'ai entendu A. blanci, descend vers 0 sinon."],
    [("Vert", {"bold": True, "color": "5CC98B"}), " : le poids pousse vers A. blanci. ", ("Rouge", {"bold": True, "color": "E07A5F"}), " : vers autre chose."],
], size=13, color="FFFFFF", after=9)
notes(s, f"""Au bout de BirdNET, j'ajoute ma tête de classification. C'est aussi un réseau de neurones, le plus petit possible : un seul neurone. À gauche, la signature de notre fenêtre de Molokoï. À côté, les poids : un par nombre de la signature, {fr(DIM)} en tout. Le neurone multiplie chaque nombre par son poids, additionne, et une fonction en forme de S, la sigmoïde, ramène la somme entre 0 et 1 : c'est le score. Au départ, tous les poids sont à zéro, et tout le monde a 0,5. Puis on entraîne : à chaque passe sur les {fr(EN['n_fenetres'])} fenêtres annotées qui servent à l'apprentissage, chaque poids est corrigé un peu, pour rapprocher le score de 1 sur les fenêtres où j'ai entendu A. blanci, de 0 sur les autres. En vert, les poids qui poussent vers A. blanci, en rouge ceux qui poussent vers autre chose. En bas, on voit mes fenêtres annotées se séparer. Au bout de {len(EN['poids']) - 1} passes, plus rien ne bouge : notre fenêtre a {fr(score_hero, 2)}. C'est la seule partie que j'entraîne, et elle s'entraîne en quelques secondes. Pour les curieux, c'est une régression logistique. Ces poids dessinent une direction dans l'espace des signatures. Et cette direction, on peut la regarder.""")

x0, y0, _ = VUES[0]
s = diapo(RUB_K, "Vu sous l'angle habituel, mes exemples sont mêlés", "", transition="pousse")
nuage(s, x0, y0, coul0, cont_0, "annotées, ACP")
legende(s, [(POS, "A. blanci", True), (NEG, "autre son", True), (GRIS, "en attente", False)])
panneau(s, ["Retour au nuage, vu à plat.", "Verts et rouges sont mêlés : aucune ligne droite ne les sépare sur ce dessin.", f"Le cube montre d'où l'on regarde : 3 directions parmi les {fr(DIM)}."])
repere(s, 0)
notes(s, """Revenons au nuage, vu sous le même angle qu'au début. Les verts et les rouges sont mêlés : aucune ligne ne les sépare. Le cube et les axes X, Y, Z sont là pour sentir la rotation. On ne peut dessiner que trois directions sur les 1 024 : X et Y, les deux directions où les fenêtres varient le plus, celles de ce dessin à plat ; Z, la direction apprise par la tête. Pour l'instant, Z pointe vers nous : on ne la voit pas.""")

for k in range(1, len(N["vues"])):
    xk, yk, _ = VUES[k]
    dernier = k == len(N["vues"]) - 1
    s = diapo(RUB_K, "On fait tourner le nuage vers la direction apprise" if not dernier else "Vu sous l'angle de la tête, les chants forment un groupe", "", ms=1300)
    nuage(s, xk, yk, coul0, cont_0, f"rotation {k}")
    legende(s, [(POS, "A. blanci", True), (NEG, "autre son", True), (GRIS, "en attente", False)])
    panneau(s, ["Mêmes points, même nuage : seul l'angle de vue change.", f"Z n'est pas l'un des {fr(DIM)} nombres : c'est la direction que dessinent ensemble les {fr(DIM)} poids de la tête."] if not dernier else [
        "Axe horizontal : la direction apprise par la tête.", "À droite, ce qui ressemble au chant ; à gauche, le reste.", "Les verts et les rouges mêlés tout à l'heure étaient déjà séparés : l'ombre le cachait."])
    repere(s, k)
    notes(s, f"Le nuage tourne : ce sont exactement les mêmes points, on change seulement l'angle de vue. Z n'est pas un des {fr(DIM)} nombres de BirdNET : c'est une direction qui les combine tous, chacun pesé par son poids. Le score de la tête ne dépend que de la position d'une fenêtre le long de cette direction." if not dernier else
          """Et voilà : regardé dans la direction apprise, le nuage se range. Les verts à droite, les rouges à gauche. Les points mêlés tout à l'heure étaient déjà séparés dans l'espace complet : c'est l'ombre en deux dimensions qui le cachait. Pour les curieux : l'axe est celui de la tête entraînée sur toutes mes annotations ; je le garde fixe pour la suite, pour que les points ne bougent plus et qu'on ne regarde que les couleurs.""")

xf, yf, _ = VUES[-1]
FRONT = N["frontieres"]
LO_U, HI_U, LO_V, HI_V = ECH[-1]


def frontiere(s, t):
    """Frontière de décision (score 0,5) de la version t de la tête : la droite où son hyperplan coupe
    le plan de la vue, tracée dans le cadre du nuage ; elle bouge à chaque réentraînement."""
    al, be, ga = FRONT[t]
    ku, kv = (HI_U - LO_U) / (GW - R_PT), (HI_V - LO_V) / (GH - R_PT)
    x0_, y0_ = GX + R_PT / 2, GY + R_PT / 2
    # α (LO_U + (x − x0) ku) + β (HI_V − (y − y0) kv) + γ = 0, soit a x + b y + c = 0
    a_, b_ = al * ku, -be * kv
    c_ = al * (LO_U - x0_ * ku) + be * (HI_V + y0_ * kv) + ga
    pts = []
    for xb in (GX, GX + GW):
        if abs(b_) > 1e-12 and GY - 1e-9 <= -(a_ * xb + c_) / b_ <= GY + GH + 1e-9:
            pts.append((xb, -(a_ * xb + c_) / b_))
    for yb in (GY, GY + GH):
        if abs(a_) > 1e-12 and GX - 1e-9 <= -(b_ * yb + c_) / a_ <= GX + GW + 1e-9:
            pts.append((-(b_ * yb + c_) / a_, yb))
    if len(pts) < 2:
        print("frontière hors du cadre, version", t)
        return
    (x1, y1), (x2, y2) = max(((p_, q_) for p_ in pts for q_ in pts), key=lambda pq: np.hypot(pq[0][0] - pq[1][0], pq[0][1] - pq[1][1]))
    long_ = float(np.hypot(x2 - x1, y2 - y1))
    ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    prec = ANGLES.get("!!seuil", 90.0)
    ang = min((ang + d for d in (-360, -180, 0, 180, 360)), key=lambda a2: abs(a2 - prec))
    ANGLES["!!seuil"] = ang
    sh = boite(s, (x1 + x2) / 2 - long_ / 2, (y1 + y2) / 2 - 0.0005, long_, 0.001, line=MUTED, nom="!!seuil", lw=1.25, dash=True)
    sh.rotation = ang % 360


def legende_scores(s, ou, attente=False):
    """Légende des diapos de scores : dans le panneau de droite (ou = « panneau »), puis en bas, une fois
    que la file de réécoute a pris la place du panneau. Mêmes noms : la Morphose la fait glisser."""
    ecoute = [("pos", POS, "A. blanci", "A. blanci (mon écoute)"), ("neg", NEG, "autre son", "autre son (mon écoute)")]
    modele = [("m0", POS, "sûr que c'est A. blanci"), ("m1", MID, "il hésite"), ("m2", NEG, "sûr que non")]
    if ou == "panneau":
        boite(s, 9.0, 1.75, 3.75, 3.6, PALER, nom="!!pan")
        texte(s, 9.25, 1.95, 3.3, 0.35, [[B("Légende")]], size=15, nom="!!lgtitre")
        texte(s, 9.25, 2.38, 3.3, 0.3, "Mon écoute (contour noir)", size=12, color=MUTED, nom="!!lgs1")
        y = 2.72
        for cle, coul, court, _ in ecoute:
            boite(s, 9.3, y + 0.07, 0.15, 0.15, coul, line=INK, forme=MSO_SHAPE.OVAL, nom=f"!!lg_{cle}", lw=LW_PT)
            texte(s, 9.6, y, 3.0, 0.3, court, size=13, nom=f"!!lgt_{cle}")
            y += 0.33
        texte(s, 9.25, y + 0.1, 3.3, 0.3, "Le modèle (sans contour)", size=12, color=MUTED, nom="!!lgs2")
        y += 0.45
        for cle, coul, lib in modele:
            boite(s, 9.3, y + 0.07, 0.15, 0.15, coul, forme=MSO_SHAPE.OVAL, nom=f"!!lg_{cle}")
            texte(s, 9.6, y, 3.0, 0.3, lib, size=13, nom=f"!!lgt_{cle}")
            y += 0.33
        boite(s, 9.25, y + 0.3, 0.25, 0.001, line=MUTED, lw=1.25, dash=True, nom="!!lg_f")
        texte(s, 9.6, y + 0.15, 3.0, 0.3, "frontière de décision (score 0,5)", size=13, nom="!!lgt_f")
        return
    x, y0 = GX, 6.82
    for cle, coul, _, long_ in ecoute:
        boite(s, x, y0 + 0.06, 0.13, 0.13, coul, line=INK, forme=MSO_SHAPE.OVAL, nom=f"!!lg_{cle}", lw=LW_PT)
        texte(s, x + 0.22, y0, 2.4, 0.3, long_, size=12, color=MUTED, nom=f"!!lgt_{cle}")
        x += 0.22 + 0.075 * len(long_) + 0.3
    for k, (cle, coul, _) in enumerate(modele):
        boite(s, x + k * 0.18, y0 + 0.06, 0.13, 0.13, coul, forme=MSO_SHAPE.OVAL, nom=f"!!lg_{cle}")
    x += 3 * 0.18 + 0.06
    lib = "score du modèle : oui, hésite, non"
    texte(s, x, y0, 3.0, 0.3, lib, size=12, color=MUTED, nom="!!lgt_m")
    x += 0.075 * len(lib) + 0.3
    boite(s, x, y0 + 0.125, 0.3, 0.001, line=MUTED, lw=1.25, dash=True, nom="!!lg_f")
    texte(s, x + 0.4, y0, 2.0, 0.3, "frontière (score 0,5)", size=12, color=MUTED, nom="!!lgt_f")
    x += 0.4 + 0.075 * 21 + 0.3
    if attente:
        boite(s, x, y0 + 0.06, 0.13, 0.13, GRIS, forme=MSO_SHAPE.OVAL, nom="!!lg_att")
        texte(s, x + 0.22, y0, 1.5, 0.3, "en attente", size=12, color=MUTED, nom="!!lgt_att")


TXT_TOUR = ["Chaque tour : le modèle propose, j'écoute, mes réponses entrent dans les annotations, il se réentraîne.", "Les hésitations se resserrent autour de la frontière."]
c, k_ = etat(0, 0)
s = diapo(RUB_K, "La tête note les fenêtres en attente, avec sa certitude", "", ms=1200)
nuage(s, xf, yf, c, k_, "scores v0")
frontiere(s, 0)
legende_scores(s, "panneau")
jauge(s, 0)
notes(s, f"""Les fenêtres en attente reçoivent une note de la tête. Sans contour, pour rappeler que ce n'est pas moi qui ai écouté. Vert foncé, le modèle est sûr qu'il y a le chant ; rouge terre, sûr que non ; doré, il hésite. La ligne pointillée, c'est la frontière de décision : là où le score vaut 0,5. D'un côté, le modèle dit A. blanci ; de l'autre, autre chose. En bas à droite, le score du modèle, mesuré sur des points d'écoute qu'il n'a jamais entendus : {fr(N['ap_rejeu'][0], 2)} sur 1 pour l'instant.""")

file1 = {"incertaine": [], "meilleur score": [], "hasard": []}
for i in range(NPTS):
    if TOUR[i] == 1 and N["cat1"][i] in file1:
        file1[N["cat1"][i]].append(i)
quota = {"incertaine": 6, "meilleur score": 2, "hasard": 2}
choisis = [(cat, i) for cat in file1 for i in file1[cat][: quota[cat]]]
POP = (9.0, 1.75, 3.75, 3.7)
xx, yy = xf.copy(), yf.copy()
rows = []
yrow = POP[1] + 0.5
for cat in file1:
    sel = [i for c_, i in choisis if c_ == cat]
    if not sel:
        continue
    rows.append((cat, yrow))
    for k, i in enumerate(sel):
        xx[i], yy[i] = POP[0] + 0.35 + k * 0.55, yrow + 0.32
    yrow += 1.02
ids_file = [i for _, i in choisis]
autres = [i for i in range(NPTS) if i not in set(ids_file)]


def encadre(s):
    boite(s, *POP, "FFFFFF", line=INK, lw=1.25, nom="!!popup")
    texte(s, POP[0] + 0.2, POP[1] + 0.12, POP[2] - 0.4, 0.35, "File de réécoute", size=14, bold=True, nom="!!popt")
    for cat, yr_ in rows:
        pct = {"incertaine": "60 % : là où il hésite", "meilleur score": "20 % : ses meilleurs scores", "hasard": "20 % : au hasard"}[cat]
        texte(s, POP[0] + 0.25, yr_, 3.3, 0.26, pct, size=11, color=MUTED, nom=f"!!pc_{cat}", after=0)


c, k_ = etat(0, 0)
# clic 1 : l'encadré s'ouvre à la place du panneau, la légende descend en bas ; les points n'ont pas encore bougé
s = diapo(RUB_K, "Le modèle me propose une file de réécoute : 60-20-20", "", ms=900)
nuage(s, xf, yf, c, k_, "file 1", seulement=autres)
frontiere(s, 0)
encadre(s)
nuage(s, xf, yf, c, k_, seulement=ids_file)
legende_scores(s, "bas")
jauge(s, 0)
notes(s, """Le modèle me propose une file de réécoute. Soixante pour cent de fenêtres où il hésite : c'est là que mon oreille lui apprend le plus. Vingt pour cent de ses meilleurs scores, pour attraper ses fausses alertes. Vingt pour cent au hasard, pour qu'il ne s'enferme pas dans ses certitudes.""")

# clic 2 : les fenêtres choisies viennent dans l'encadré
s = diapo(RUB_K, "Le modèle me propose une file de réécoute : 60-20-20", "", ms=1400)
nuage(s, xx, yy, c, k_, "file 2", seulement=autres)
frontiere(s, 0)
encadre(s)
for i in ids_file:  # sous chaque point, son double, qui sortira avec mon étiquette au clic suivant
    boite(s, xx[i], yy[i], R_PT, R_PT, c[i], forme=MSO_SHAPE.OVAL, nom=f"!!r{i}", lw=LW_PT)
nuage(s, xx, yy, c, k_, seulement=ids_file)
legende_scores(s, "bas")
jauge(s, 0)
notes(s, "Voici les fenêtres qu'il me propose, rangées par raison. Je les écoute une à une.")

# clic 3 : mon étiquette sort de chaque point
s = diapo(RUB_K, "J'écoute et je tranche", "", ms=1400)
nuage(s, xx, yy, c, k_, "file 3", seulement=autres)
frontiere(s, 0)
encadre(s)
for _cat, i in choisis:
    fleche(s, xx[i] + R_PT / 2, yy[i] + R_PT + 0.02, xx[i] + R_PT / 2, yy[i] + 0.3, color=SAGE, width=1.25, nom=f"!!fa{i}")
    boite(s, xx[i], yy[i] + 0.32, R_PT, R_PT, POS if Y[i] == 1 else NEG, line=INK, forme=MSO_SHAPE.OVAL, nom=f"!!r{i}", lw=LW_PT)
    texte(s, xx[i] + R_PT / 2 - 0.27, yy[i] + 0.43, 0.54, 0.22, "blanci" if Y[i] == 1 else "autre", size=9, color=INK, align="c", nom=f"!!rt{i}", after=0)
nuage(s, xx, yy, c, k_, seulement=ids_file)
legende_scores(s, "bas")
jauge(s, 0)
notes(s, """J'écoute, et je tranche : sous chaque fenêtre, ce que j'ai entendu, A. blanci ou autre chose. Ces fenêtres réécoutées ne servent jamais à noter le modèle.""")

s = diapo(RUB_K, "Mes réponses entrent dans les annotations ; on réentraîne", "", ms=1300)
c, k_ = etat(1)
nuage(s, xf, yf, c, k_, "réentraînement")
frontiere(s, 0)
legende_scores(s, "bas", attente=True)
panneau(s, TXT_TOUR, h=3.6)
jauge(s, 0)
notes(s, """Les fenêtres reprennent leur place, cette fois avec mon étiquette, gravée dans le marbre. On réentraîne la tête : quelques secondes. Les autres points repassent en attente.""")

s = diapo(RUB_K, "Nouvelle version du modèle, nouvelles certitudes", "", ms=1000)
c, k_ = etat(1, 1)
nuage(s, xf, yf, c, k_, "scores v1")
frontiere(s, 1)
legende_scores(s, "bas")
panneau(s, TXT_TOUR, h=3.6)
jauge(s, 1)
notes(s, f"""Nouvelle version, nouvelles notes. Le score passe à {fr(N['ap_rejeu'][1], 2)}. La frontière a bougé : les poids ont changé, la séparation aussi. Et on recommence : le modèle propose, j'écoute, il se réentraîne.""")

for t in range(2, 5):
    s = diapo(RUB_K, f"Tour {t} : j'annote la file", "", ms=700)
    c, k_ = etat(t, t - 1)
    nuage(s, xf, yf, c, k_, f"A{t}")
    frontiere(s, t - 1)
    legende_scores(s, "bas")
    panneau(s, TXT_TOUR, h=3.6)
    jauge(s, t - 1)
    notes(s, "J'écoute la file suivante.")
    s = diapo(RUB_K, f"Tour {t} : le modèle se réentraîne", "", ms=700)
    c, k_ = etat(t, t)
    nuage(s, xf, yf, c, k_, f"B{t}")
    frontiere(s, t)
    legende_scores(s, "bas")
    panneau(s, TXT_TOUR, h=3.6)
    jauge(s, t)
    notes(s, f"Nouvelle version : score {fr(N['ap_rejeu'][t], 2)}. La frontière se déplace encore.")

# regrouper ou apprendre : deux lectures de la même signature
s = diapo(RUB_K, "Regrouper ou apprendre : deux façons de lire la même signature", "", transition="pousse")
for k, (titre_, coul_, img, legende_) in enumerate([
    ("Regrouper (clustering)", TEAL, "fenetre_spectro_plein.png", "Compare les fenêtres dans leur ensemble : c'est le fond sonore (le lieu, la saison) qui décide de qui ressemble à qui."),
    ("Apprendre (classification)", POS, "fenetre_spectro.png", "Mes exemples désignent ce qui compte : les poids appris isolent, dans la signature, le signal faible d'A. blanci."),
]):
    x = 0.6 + k * 6.25
    texte(s, x, 1.75, 5.85, 0.4, titre_, size=18, bold=True, color=coul_)
    s.shapes.add_picture(str(ICI / img), Inches(x + 0.75), Inches(2.25), height=Inches(3.2))
    if k == 0:
        boite(s, x + 0.72, 2.22, 3.2 * 6.0 / 4.4 + 0.06, 3.26, None, line=GOLD, lw=2.5)
    texte(s, x, 5.6, 5.85, 0.9, legende_, size=14, color=INK)
texte(s, 0.6, 6.55, 12.1, 0.4, "Limite : la tête peut aussi apprendre un raccourci (un lieu, une saison où le chant est fréquent). C'est pourquoi on la note sur des micros jamais entendus.", size=12, color=MUTED, italic=True)
notes(s, """Pourquoi la classification réussit là où le regroupement échoue ? Le regroupement compare les signatures en entier : deux fenêtres se ressemblent si toute leur image se ressemble, et c'est le fond sonore, le lieu, la saison, qui domine. La note d'A. blanci n'y pèse presque rien. La classification, elle, reçoit mes exemples : des fenêtres avec la note, des fenêtres sans. Elle cherche ce qui les distingue, et ses poids vont sur les nombres de la signature qui portent la note, même s'ils ne représentent qu'une toute petite partie de l'information. On force la main au modèle : on lui fait trouver un signal faible. Deux précisions. La tête ne regarde pas l'image, elle ne voit que les 1 024 nombres de BirdNET ; l'image est une façon de se représenter ce qu'elle fait. Et elle peut apprendre un raccourci : si mes fenêtres avec chant viennent surtout d'un lieu ou d'une saison, elle peut apprendre ce lieu plutôt que la grenouille. C'est pour ça qu'on la note sur des micros qu'elle n'a jamais entendus.""")

# évaluation
SIM = D["simulation"]
cv = SIM["courbes"]
s = diapo(RUB_K, "Comment on saura que ça marche", "", transition="pousse")
s.shapes.add_picture(str(ICI / "courbe_v4.png"), Inches(0.5), Inches(1.7), height=Inches(4.6))
texte(s, 0.6, 6.4, 7.2, 0.6, f"Départ à {SIM['N0']} fenêtres annotées, +{SIM['lot']} par tour ; 10 répétitions (bande : min-max). Score mesuré sur {SIM['n_eval_micros']} points d'écoute jamais vus ({fr(SIM['n_eval'])} fenêtres, dont {SIM['n_eval_pos']} avec chant).", size=11, color=MUTED, italic=True)
mieux = all(a_ >= h_ for a_, h_ in zip(cv["actif"]["moy"][1:], cv["hasard"]["moy"][1:], strict=True)) and cv["actif"]["moy"][-1] > cv["hasard"]["moy"][-1] + 0.01
panneau(s, [
    [B("La métrique d'évaluation")],
    [B("1"), " : précision parfaite, le modèle classe parfaitement les fenêtres."],
    [B(fr(SIM["prevalence_eval"], 2)), " : le score d'un modèle qui répond au hasard."],
    [B("Ce qu'on voit")],
    "Réannoter avec la file 60-20-20 fait monter le score plus vite que réannoter au hasard." if mieux else "Réannoter avec la file 60-20-20 ne fait pas mieux que réannoter au hasard.",
    [B("Un premier jet")],
    "Des scores provisoires : je vais entraîner des modèles plus puissants, qui devraient faire mieux. Sur le futur jeu de test, tiré au sort, où le chant est plus rare, le score sera a priori un peu plus bas.",
], size=13)
if mieux:
    phrase_file = f"Et réannoter avec la file 60-20-20 le fait monter plus vite que réannoter au hasard : {fr(cv['actif']['moy'][-1], 2)} contre {fr(cv['hasard']['moy'][-1], 2)} au bout de {cv['actif']['n'][-1]} fenêtres."
else:
    phrase_file = "Réannoter avec la file 60-20-20 ne fait pas mieux que réannoter au hasard."
notes(s, f"""Comment saura-t-on que ça marche ? Le modèle est noté sur des points d'écoute qu'il n'a jamais entendus, avec la précision moyenne. Un, c'est la précision parfaite : le modèle classe parfaitement les fenêtres. Un modèle qui répondrait au hasard aurait ici {fr(SIM['prevalence_eval'], 2)}. Avec {SIM['N0']} fenêtres annotées au départ, le score est de {fr(cv['actif']['moy'][0], 2)} ; il monte à chaque tour. {phrase_file} C'est un premier jet : je vais entraîner des modèles plus puissants, qui devraient faire mieux. Mais ces scores sont mesurés sur des extraits où le chant est fréquent ; sur le futur jeu de test, des enregistrements tirés au sort et écoutés en entier, le chant sera plus rare et le score a priori un peu plus bas.""")

# la suite
s = diapo(RUB_S, "Février 2027 : de la carte SD à la carte de présence", "", transition="pousse")
# à gauche : la carte SD et ses fichiers bruts
sd = s.shapes.add_shape(MSO_SHAPE.SNIP_1_RECTANGLE, Inches(0.95), Inches(1.95), Inches(2.1), Inches(2.75))
sd.fill.solid()
sd.fill.fore_color.rgb = rgb("2F3B37")
sd.line.fill.background()
sd.shadow.inherit = False
for k in range(5):
    boite(s, 1.15 + k * 0.3, 2.1, 0.18, 0.5, GOLD)
texte(s, 0.95, 3.0, 2.1, 0.8, [[B("SD")]], size=34, color="FFFFFF", align="c", anchor="m", font="Cambria")
texte(s, 0.95, 3.85, 2.1, 0.4, "campagne de février", size=12, color=SAGE, align="c")
texte(s, 0.6, 4.95, 2.9, 1.5, ["SMA14163_20270212_053000.wav", "SMA14163_20270212_060000.wav", "SMA14163_20270212_063000.wav", "…"], size=10, color=MUTED, font="Consolas", after=2)
texte(s, 0.6, 6.3, 2.9, 0.6, "Les données brutes : des milliers de fichiers de 2 min.", size=12, color=INK, italic=True)
fleche(s, 3.45, 4.0, 3.95, 4.0, color=GOLD, width=3)
# au centre : la chaîne complète
boite(s, 4.05, 1.85, 4.75, 5.05, PALER, line=SAGE, lw=1, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
texte(s, 4.3, 1.95, 4.3, 0.4, [[B("La chaîne")]], size=16, color=TEAL)
chaine = [("Contrôle qualité", "pistes, durées, relevés", "FFFFFF", INK), ("BirdNET", "une signature par fenêtre de 3 s", DARK, "FFFFFF"), ("Tête de classification", "un score par fenêtre", TEAL, "FFFFFF"), ("Écoute ciblée", "les hésitations d'abord", "FFFFFF", INK)]
for k, (t_, st_, fill_, coul_) in enumerate(chaine):
    yb = 2.5 + k * 1.05
    boite(s, 4.35, yb, 3.7, 0.78, fill_, line=SAGE if fill_ == "FFFFFF" else None, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.1)
    texte(s, 4.5, yb + 0.06, 3.4, 0.35, t_, size=15, bold=True, color=coul_, align="c")
    texte(s, 4.5, yb + 0.4, 3.4, 0.3, st_, size=11, color=coul_ if fill_ in (DARK, TEAL) else MUTED, align="c")
    if k < 3:
        fleche(s, 6.2, yb + 0.8, 6.2, yb + 1.03, color=SAGE, width=2)
# réentraînement : de « Écoute ciblée », à droite, en haut, puis à gauche dans « Tête de classification »
y_tete, y_ecoute = 2.5 + 2 * 1.05 + 0.39, 2.5 + 3 * 1.05 + 0.39
ff = s.shapes.build_freeform(Inches(8.07), Inches(y_ecoute), scale=1.0)
ff.add_line_segments([(Inches(8.42), Inches(y_ecoute)), (Inches(8.42), Inches(y_tete)), (Inches(8.1), Inches(y_tete))], close=False)
coude = ff.convert_to_shape()
coude.fill.background()
coude.line.color.rgb = rgb(GOLD)
coude.line.width = Pt(2)
coude.shadow.inherit = False
ln_ = coude.line._get_or_add_ln()
ln_.append(ln_.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
rt = texte(s, 7.95, (y_tete + y_ecoute) / 2 - 0.11, 1.3, 0.22, "réentraînement", size=10, color="9A6B0F", align="c", italic=True, anchor="m", after=0)
rt.rotation = 270
fleche(s, 8.9, 4.0, 9.15, 4.0, color=GOLD, width=3)
# à droite : la carte de présence
s.shapes.add_picture(str(ICI / "carte_guyane.png"), Inches(9.2), Inches(1.85), height=Inches(4.15))
texte(s, 9.2, 6.05, 3.75, 0.9, "Carte de présence d'après mes écoutes, sur les trois sites dont la position est connue. Demain, d'après le modèle, sur toute la campagne.", size=11, color=MUTED, italic=True)
notes(s, f"""Concrètement, pour la campagne de février. À gauche, ce qu'on récupère : des cartes SD, des milliers de fichiers de deux minutes. Au centre, la chaîne : le contrôle qualité, puis BirdNET, qui donne une signature à chaque fenêtre de trois secondes, puis ma tête de classification, qui donne un score, et enfin l'écoute ciblée des cas où le modèle hésite ; ces nouvelles écoutes le réentraînent. À droite, ce qu'on veut produire : une carte de présence. Celle-ci est faite d'après mes écoutes, sur les trois sites dont je connais la position, Kaw, Molokoï et Trésor ; demain, elle sera faite par le modèle sur toute une campagne. Le stage livre un corpus contrôlé de {fr(C['enregistrements'])} enregistrements, un poste d'annotation, {fr(D['n_annotees'])} fenêtres écoutées et la chaîne complète. Il reste à geler un jeu de test pour donner une note officielle, à choisir l'encodeur sur nos données, et à suivre le modèle dans le temps : chaque nouvelle campagne sera sa vraie épreuve.""")

# merci
s = prs.slides.add_slide(BLANK)
fond(s, DARK)
s.shapes.add_picture(str(FIG / "titre.png"), Inches(8.9), Inches(0), height=Inches(7.5))
texte(s, 0.8, 2.6, 7.6, 1.2, "Merci", size=48, color="FFFFFF", bold=True, font="Cambria")
texte(s, 0.8, 3.8, 7.6, 1.2, "Questions, remarques, enregistrements à partager : je suis preneur.", size=18, color=SAGE)
texte(s, 0.8, 6.15, 7.6, 0.35, "Léonard Laplace-Palette", size=14, color="FFFFFF", bold=True)
notes(s, "Merci. Je prends vos questions.")
pousse(s)

prs.save(SORTIE)
print("écrit", SORTIE, len(prs.slides), "diapos ; audio inséré :", audio_ok)

# ------------------------------------------------------------------ planche de contrôle
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cols = 5
n = len(PLANCHE)
fig, axes = plt.subplots(-(-n // cols), cols, figsize=(4 * cols, 2.6 * -(-n // cols)))
for ax in axes.flat:
    ax.set_axis_off()
for ax, (titre, x, y, coul, cont) in zip(axes.flat, PLANCHE, strict=False):
    ax.set_axis_on()
    ax.scatter(x + R_PT / 2, -(y + R_PT / 2), s=6, c=["#" + c for c in coul], edgecolors=["#" + c if c else "none" for c in (cont or [None] * len(x))], linewidths=0.6)
    ax.set_xlim(0.4, 12.9)
    ax.set_ylim(-7.0, -1.6)
    ax.set_title(titre, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(ICI / "planche.png", dpi=70)

# rendu de contrôle des diapos d'encodage (forme d'onde) : positions exactes des barres
fig, axes = plt.subplots(1, 3, figsize=(18, 3.6))
for ax, mode in zip(axes, ("entier", "fenetres", "zoom"), strict=True):
    ax.set_facecolor("#" + ECRAN)
    for i in range(NB):
        if i in dans:
            continue
        t = i * PAS
        if mode == "zoom":
            xx_ = ZX0 + (t - F0) * ZPPS
            if xx_ < 0 or xx_ > 13.3:
                continue
            ax.vlines(xx_, ZCY - lo[i] / PW * ZHH, ZCY - hi[i] / PW * ZHH, color="r")
        else:
            ax.vlines(WX0 + t * PPS, CY - lo[i] / P * HH, CY - hi[i] / P * HH, color="#" + ONDE, lw=0.6)
    for j in range(NF):
        t = F0 + j * FPAS
        if mode == "zoom":
            ax.vlines(ZX0 + (t - F0) * ZPPS, ZCY - flo[j] / PW * ZHH, ZCY - fhi[j] / PW * ZHH, color="#" + ONDE, lw=0.8)
        else:
            ax.vlines(WX0 + t * PPS, CY - flo[j] / P * HH, CY - fhi[j] / P * HH, color="#" + GOLD if mode == "fenetres" else "#" + ONDE, lw=0.3)
    ax.set_xlim(0, 13.33)
    ax.set_ylim(7.5, 0)
    ax.set_title(mode)
fig.tight_layout()
fig.savefig(ICI / "planche_onde.png", dpi=70)
