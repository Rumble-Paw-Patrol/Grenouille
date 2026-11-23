# ruff: noqa: E501
"""Présentation PNA A. blanci (restitution de stage, 15 min), animée par la Morphose.

    uv run --with python-pptx python documentation/prez/scripts/presentation_7/deck.py [sortie.pptx]

Lit `donnees.json` et `courbe.png` (donnees.py), `spectro.png` et `extrait.wav`
(figures_audio.py), et deux figures du dépôt (titre.png, inventaire_qc/inv_sites.png).
Chaque point du nuage est une forme nommée « !!p<i> », chaque barre d'enveloppe « !!env<i »… :
d'une diapo à l'autre, PowerPoint fait glisser et recolore les formes qui portent le même nom
(Morphose, PowerPoint 2019 / Microsoft 365 ; un PowerPoint plus ancien fait un fondu). Les vues
du nuage sont de vraies projections linéaires des 1 536 nombres de Perch. Les diapos
intermédiaires de la rotation avancent seules (une seule frappe pour toute la rotation).
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

ICI = Path(__file__).resolve().parents[4] / "rapports" / "presentation_7"  # données, jamais committées
ICI.mkdir(parents=True, exist_ok=True)
DEPOT = Path(__file__).resolve().parents[4]
FIG = DEPOT / "documentation" / "prez" / "figures"
SORTIE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEPOT / "documentation" / "prez" / "presentation_7_pna.pptx"
D = json.loads((ICI / "donnees.json").read_text(encoding="utf-8"))
N = D["nuage"]

DARK, INK, MUTED, SAGE, PALE, PALER = "0E2A24", "1B2A24", "5B6B63", "A7BFB2", "E4EEE8", "F3F7F4"
GOLD, TEAL, TERRA = "E9B44C", "2A7F72", "C8553D"
GRIS, GRIS_PALE = "B9C3BD", "E3E8E5"
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


def boite(s, x, y, w, h, fill=None, line=None, forme=MSO_SHAPE.RECTANGLE, nom=None, lw=0.75, dash=False, radius=None):
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


def fleche(s, x1, y1, x2, y2, color=SAGE, width=1.5, nom=None):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    if nom:
        c.name = nom
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
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
    texte(s, 0.6, 0.4, w, 0.32, rubrique.upper() if False else rubrique, size=10, color=TEAL, bold=True, align="c", anchor="m", nom="!!rubt")
    texte(s, 0.6, 0.8, 12.1, 0.8, titre, size=22, bold=True, font="Cambria", nom="Titre")
    PAGE[0] += 1
    texte(s, 12.33, 7.08, 0.5, 0.25, str(PAGE[0]), size=9, color=SAGE, align="r")
    notes(s, note)
    if transition:
        morph(s, ms, auto)
    return s


def panneau(s, lignes, y=1.75, h=4.95, size=15):
    boite(s, 9.0, y, 3.75, h, PALER, nom="!!pan")
    texte(s, 9.25, y + 0.22, 3.3, h - 0.4, lignes, size=size, nom="Texte", after=10)


# ------------------------------------------------------------------ nuage
GX, GY, GW, GH = 0.7, 1.75, 7.9, 4.95
R_PT = 0.12
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


VUES = [ecran(*v) for v in N["vues"]]


def degrade(p):
    def mix(c1, c2, t):
        a_, b_ = [int(c1[k:k + 2], 16) for k in (0, 2, 4)], [int(c2[k:k + 2], 16) for k in (0, 2, 4)]
        return "".join(f"{round(u + (v - u) * t):02X}" for u, v in zip(a_, b_, strict=True))
    return mix(NEG, MID, p * 2) if p < 0.5 else mix(MID, POS, (p - 0.5) * 2)


def nuage(s, x, y, couleurs, contours=None, titre="", seulement=None):
    if titre:
        PLANCHE.append((titre, np.asarray(x), np.asarray(y), couleurs, contours))
    for i in range(NPTS) if seulement is None else seulement:
        line = contours[i] if contours is not None else None
        boite(s, x[i], y[i], R_PT, R_PT, couleurs[i], line=line, forme=MSO_SHAPE.OVAL, nom=f"!!p{i}", lw=1.5)


def legende(s, items, y0=6.82):
    x = GX
    for coul, lib, contour in items:
        boite(s, x, y0 + 0.05, 0.16, 0.16, coul, line=INK if contour else None, forme=MSO_SHAPE.OVAL, nom=f"!!leg_{lib}", lw=1.5)
        texte(s, x + 0.24, y0, 2.6, 0.3, lib, size=12, color=MUTED, nom=f"!!legt_{lib}")
        x += 0.36 + 0.08 * len(lib) + 0.25


def boussole(s, k):
    """Trois flèches : où pointent l'ACP 1, l'ACP 2 et la direction de la tête dans la vue k."""
    cx, cy, R = 10.9, 5.65, 0.62
    boite(s, cx - R - 0.12, cy - R - 0.12, 2 * R + 0.24, 2 * R + 0.24, "FFFFFF", line=GRIS_PALE, forme=MSO_SHAPE.OVAL, nom="!!bous")
    for nom, coul in (("ACP 1", MUTED), ("ACP 2", SAGE), ("tête", POS)):
        dx, dy = N["boussole"][k][nom]
        x2, y2 = cx + R * dx, cy - R * dy
        fleche(s, cx, cy, x2 if abs(x2 - cx) > 0.01 else cx + 0.01, y2, color=coul, width=2.2 if nom == "tête" else 1.5, nom=f"!!b_{nom}")
        lx = x2 + (0.06 if dx >= 0 else -0.5)
        texte(s, lx, y2 - 0.12, 0.5, 0.22, nom, size=9, color=coul, bold=nom == "tête", nom=f"!!bl_{nom}", align="l" if dx >= 0 else "r")
    texte(s, cx - 0.9, cy + R + 0.16, 1.8, 0.22, "angle de vue", size=9, color=MUTED, align="c", nom="!!bt")


SC = N["scores"]
Y = np.array(N["y"])
TOUR = np.array(N["tour"])
SITE = N["site"]


def etat(t_rev, version=None):
    """Couleurs et contours : annoté (révélé au tour ≤ t_rev) en vert/rouge cerclé ; sinon gris
    ou couleur du score de la version `version` de la tête."""
    coul, cont = [], []
    for i in range(NPTS):
        connu = TOUR[i] == -2 or (0 <= TOUR[i] <= t_rev)
        if connu:
            coul.append(POS if Y[i] == 1 else NEG)
            cont.append(INK)
        else:
            coul.append(GRIS if version is None else degrade(SC[version][i]))
            cont.append(None)
    return coul, cont


def jauge(s, t):
    """Score du modèle (AP sur des micros jamais vus) après le tour t du rejeu."""
    ap = N["ap_rejeu"][t]
    n = N["n_rejeu"][t]
    x, y, w = 9.25, 5.55, 3.25
    texte(s, x, y, w, 0.26, "Score du modèle (micros jamais entendus)", size=11, color=MUTED, nom="!!jt")
    boite(s, x, y + 0.32, w, 0.26, GRIS_PALE, nom="!!jf")
    boite(s, x, y + 0.32, w * ap, 0.26, POS, nom="!!jb")
    texte(s, x, y + 0.64, w, 0.3, [[B(fr(ap, 2)), f"   après {n} fenêtres annotées"]], size=12, color=INK, nom="!!jv")


# ================================================================== diapos
RUB_I, RUB_E, RUB_C, RUB_K, RUB_S = "Le projet", "1 · Encoder", "2 · Regrouper ?", "3 · Apprendre", "La suite"

# 1. titre -----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
fond(s, DARK)
s.shapes.add_picture(str(FIG / "titre.png"), Inches(8.9), Inches(0), height=Inches(7.5))
texte(s, 0.8, 1.55, 7.6, 0.4, "PLAN NATIONAL D'ACTIONS · ANOMALOGLOSSUS BLANCI", size=12, color=GOLD, bold=True)
texte(s, 0.8, 2.05, 7.7, 2.2, "Entendre A. blanci dans 3 200 heures de forêt", size=40, color="FFFFFF", bold=True, font="Cambria", after=0)
texte(s, 0.8, 4.2, 7.6, 0.9, "Détection acoustique automatique : où en est le stage, comment la méthode fonctionne, ce qu'elle apportera aux prochaines campagnes", size=16, color=SAGE)
texte(s, 0.8, 5.75, 7.6, 0.35, "Léonard Laplace-Palette", size=16, color="FFFFFF", bold=True)
texte(s, 0.8, 6.15, 7.6, 0.35, "Restitution de stage · réunion du PNA · [date]", size=13, color=SAGE)
texte(s, 9.1, 7.05, 4.0, 0.3, "Chant d'A. blanci : les notes en forme de virgule, vers 4,7 kHz", size=9, color="FFFFFF", italic=True)
notes(s, """Bonjour. Je présente mon stage : détecter automatiquement le chant d'Anomaloglossus blanci dans les enregistrements des campagnes de suivi. L'image à droite est un spectrogramme : le temps de gauche à droite, la hauteur du son de bas en haut. Les petites virgules claires, ce sont les notes d'A. blanci. Quinze minutes : le problème, les données, puis la méthode en trois étapes, et ce qu'elle donnera pour la campagne de février.""")

# 2. le problème -----------------------------------------------------
s = diapo(RUB_I, "Trop d'heures pour une oreille humaine", "", transition=False)
heures = 3187.9
jours = heures / 7
texte(s, 0.6, 1.75, 7.6, 2.4, [
    "Le PNA a besoin de savoir où et quand A. blanci chante : quels sites, quels mois, quelles heures.",
    "Les enregistreurs posés depuis 2023 ont capté la forêt toutes les 30 minutes, 2 minutes à chaque fois.",
    [B("Objectif du stage : "), "un outil qui dit, pour chaque enregistrement, si A. blanci y chante, et avec quelle certitude, pour n'écouter que ce qui compte."],
], size=17, after=14)
for k, (gros, petit) in enumerate([
    (fr(95633), "enregistrements de 2 min"),
    (f"{fr(heures)} h", "d'audio, 8 sites, 164 points d'écoute"),
    (f"{fr(jours)} jours", "d'écoute à 7 h par jour, soit 2 ans de travail"),
]):
    x = 0.6 + k * 4.1
    boite(s, x, 4.75, 3.85, 1.75, PALER, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    texte(s, x + 0.25, 4.9, 3.4, 0.7, gros, size=30, color=TEAL if k < 2 else TERRA, bold=True, font="Cambria")
    texte(s, x + 0.25, 5.65, 3.4, 0.7, petit, size=13, color=MUTED)
texte(s, 9.0, 1.85, 3.75, 2.6, [
    [B("Pourquoi c'est difficile")],
    "Des notes de 0,2 s, espacées de quelques secondes, au milieu du chœur des insectes, des oiseaux et des autres grenouilles.",
], size=14, color=INK)
notes(s, f"""Le Plan national d'actions a besoin de savoir où et quand l'espèce chante. Depuis 2023, des enregistreurs captent deux minutes de forêt toutes les demi-heures. Au total, {fr(95633)} enregistrements gardés, {fr(heures)} heures. Pour les écouter, il faudrait {fr(jours)} journées de sept heures : deux ans à plein temps. L'objectif du stage est donc un outil qui trie : il dit pour chaque enregistrement si A. blanci y chante, avec une certitude, et l'oreille humaine ne sert plus qu'à vérifier.""")

# 3. ce qu'on cherche (audio) ----------------------------------------
s = diapo(RUB_I, "Ce qu'on cherche : quelques notes brèves dans le bruit de la forêt", "", transition=False)
pic = ICI / "spectro.png"
try:
    s.shapes.add_movie(str(ICI / "extrait.wav"), Inches(0.6), Inches(1.8), Inches(8.0), Inches(3.2), poster_frame_image=str(pic), mime_type="audio/wav")
    audio_ok = True
except Exception as exc:  # noqa: BLE001
    print("audio non inséré :", exc)
    s.shapes.add_picture(str(pic), Inches(0.6), Inches(1.8), width=Inches(8.0))
    audio_ok = False
texte(s, 0.6, 5.1, 8.0, 0.6, "10 s enregistrées à Molokoï (point F, 26/02/2024, 13 h 30). Cliquer sur l'image pour écouter.", size=12, color=MUTED, italic=True)
panneau(s, [
    [B("Les virgules à 4,7 kHz : A. blanci.")],
    "Une note de 0,2 s toutes les 3 à 4 s.",
    "Au-dessus, la bande continue des insectes ; en dessous, d'autres grenouilles.",
    "Sur 2 minutes, le chant occupe moins de 5 % du temps et une bande étroite de fréquences.",
], y=1.8, h=4.6)
notes(s, """Voici dix secondes à Molokoï. Je lance le son. Les petites virgules vers 4,7 kilohertz, ce sont les notes d'A. blanci : une toute petite note, toutes les trois ou quatre secondes. Au-dessus, la bande continue, ce sont des insectes ; en dessous, d'autres espèces. Retenez cette image : le chant prend très peu de place, dans le temps comme en fréquence. C'est toute la difficulté.""")

# 4. inventaire + contrôle qualité -----------------------------------
s = diapo(RUB_I, "Avant toute IA : inventorier et contrôler 96 000 enregistrements", "", transition=False)
s.shapes.add_picture(str(FIG / "inventaire_qc" / "inv_sites.png"), Inches(0.6), Inches(1.75), width=Inches(7.6))
texte(s, 0.6, 5.85, 7.6, 0.8, "Inventaire : 4 lieux de suivi (2023-24) et 4 relevés d'une semaine (2025-26), empilés sur leur lieu.", size=11, color=MUTED, italic=True)
panneau(s, [
    [B("Inventaire")],
    "96 349 fichiers ; 95 enregistreurs ; horloges recalées, doublons et trous repérés.",
    [B("Contrôle qualité")],
    "Chaque enregistreur a deux micros (gains 6 et 18 dB) : la bonne piste est choisie pour chacun ; 7 900 enregistrements lisent l'autre piste (gain inversé, micro cassé, son étouffé).",
    "716 écartés (son étouffé, durée anormale, hors relevé).",
    [B(f"Corpus figé le 20/11/2026 : {fr(95633)} enregistrements.")],
], size=13)
notes(s, """Avant de parler d'intelligence artificielle, il a fallu savoir ce qu'on avait. L'inventaire : 96 349 fichiers, 95 enregistreurs, des horloges à recaler. Puis le contrôle qualité : chaque enregistreur a deux micros réglés différemment, et pour près de 8 000 enregistrements la bonne piste n'est pas celle qu'on croit, micro cassé ou étouffé. 716 enregistrements sont écartés. Le corpus est figé depuis le 20 novembre : tout ce qui suit travaille sur ces 95 633 enregistrements.""")

# ---------------------------------------------------------- 1 · encoder
env = np.array(D["enveloppe"])
NB = len(env)  # barres de 0,5 s
hero_off = D["hero"]["offset_s"]
f0, f1 = int(hero_off / 0.5), int((hero_off + 5) / 0.5)


def enveloppe(s, x0, x1, base, hmax, couleurs=None, cache=(), vers=None):
    bw = (x1 - x0) / NB
    for i in range(NB):
        h = max(0.012, env[i] * hmax)
        if vers is not None and i in cache:
            bx, by, bww, bh = vers
            boite(s, bx, by, bww, bh, (couleurs or {}).get(i, INK), nom=f"!!env{i}")
            continue
        boite(s, x0 + i * bw + bw * 0.15, base - h, bw * 0.7, h, (couleurs or {}).get(i, INK), nom=f"!!env{i}")
    return bw


s = diapo(RUB_E, "Un enregistrement de 2 minutes, tel que l'ordinateur le reçoit", "", ms=1000)
enveloppe(s, 0.7, 12.6, 5.6, 3.2)
texte(s, 0.7, 5.75, 11.9, 0.3, "0 s", size=11, color=MUTED, nom="!!t0")
texte(s, 11.6, 5.75, 1.0, 0.3, "120 s", size=11, color=MUTED, align="r", nom="!!t1")
texte(s, 0.7, 6.25, 11.9, 0.8, "Chaque barre : l'énergie du son pendant une demi-seconde. Fond plat : la forêt. Les pics réguliers : les notes d'A. blanci.", size=15, color=INK, nom="!!leg_env")
notes(s, """L'ordinateur n'a pas d'oreilles : il reçoit une suite de nombres. Ici, chaque barre est l'énergie du son pendant une demi-seconde, sur les deux minutes. Quand la forêt est calme, la ligne est plate. Les pics réguliers, ce sont les notes d'A. blanci : dans cet enregistrement-là, c'est le son le plus fort.""")

s = diapo(RUB_E, "On le découpe en fenêtres de 5 secondes", "", ms=900)
bw = enveloppe(s, 0.7, 12.6, 5.6, 3.2)
for k in range(1, 24):
    boite(s, 0.7 + k * 10 * bw, 2.2, 0.012, 3.45, GRIS_PALE, nom=f"!!sep{k}")
boite(s, 0.7 + f0 * bw, 2.15, 10 * bw, 3.55, None, line=GOLD, lw=3, nom="!!fen")
texte(s, 0.7, 5.75, 11.9, 0.3, "0 s", size=11, color=MUTED, nom="!!t0")
texte(s, 11.6, 5.75, 1.0, 0.3, "120 s", size=11, color=MUTED, align="r", nom="!!t1")
texte(s, 0.7, 6.25, 11.9, 0.8, "24 fenêtres par enregistrement. Tout ce qui suit travaille fenêtre par fenêtre ; un enregistrement prend le score de sa meilleure fenêtre.", size=15, color=INK, nom="!!leg_env")
notes(s, """Le modèle écoute cinq secondes à la fois : on découpe donc l'enregistrement en 24 fenêtres. Tout le reste de la méthode travaille fenêtre par fenêtre ; à la fin, un enregistrement prend le score de sa meilleure fenêtre. Suivons celle-ci, encadrée en jaune.""")

BOX = (5.1, 3.85, 3.0, 1.55)
s = diapo(RUB_E, "Chaque fenêtre passe dans un modèle déjà entraîné : Perch", "", ms=1400)
bwp = (12.6 - 0.7) / NB
coul = {i: GRIS for i in range(NB)}
enveloppe(s, 0.7, 12.6, 3.25, 1.2, couleurs=coul, cache=set(range(f0, f1)), vers=(BOX[0] + BOX[2] / 2 - 0.05, BOX[1] + BOX[3] / 2 - 0.05, 0.1, 0.1))
boite(s, 0.7 + f0 * bwp, 1.95, 10 * bwp, 1.4, None, line=GOLD, lw=3, nom="!!fen")
fleche(s, 0.7 + (f0 + 5) * bwp, 3.4, BOX[0] + BOX[2] / 2, BOX[1] - 0.05, color=GOLD, width=2, nom="!!fl1")
boite(s, *BOX, DARK, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, nom="!!perch")
texte(s, BOX[0], BOX[1] + 0.2, BOX[2], 1.2, [[B("Perch")], ["Google · 12 millions de réglages appris sur 15 000 espèces"]], size=14, color="FFFFFF", align="c", nom="!!percht", after=4)
texte(s, 0.7, 5.75, 11.9, 1.2, "Un réseau de neurones entraîné par Google sur des centaines de milliers d'enregistrements d'oiseaux, d'amphibiens et d'insectes. Je ne le modifie pas : je m'en sers comme d'une oreille.", size=15, color=INK, nom="!!leg_env")
notes(s, """Chaque fenêtre passe dans Perch, un modèle publié par Google, entraîné sur des centaines de milliers d'enregistrements et environ quinze mille espèces, oiseaux, amphibiens, insectes. Il a douze millions de réglages internes. Je ne le réentraîne pas : je l'utilise tel quel, comme une oreille déjà exercée. On utilise souvent BirdNET, de Cornell ; Perch est de la même taille, et il range mieux nos chants de grenouilles sur les tests que j'ai faits.""")

bande = D["bande"]
vals = np.array(bande["brut"])
vmax = np.abs(vals).max()
CW, CX, CY = 0.205, 1.0, 6.05


def cellule(v, pale=False):
    t = min(1.0, abs(v) / vmax)
    base = "2A7F72" if v >= 0 else "C8553D"
    if pale:
        t *= 0.25
    a_ = [int(base[k:k + 2], 16) for k in (0, 2, 4)]
    return "".join(f"{round(255 + (c - 255) * t):02X}" for c in a_)


s = diapo(RUB_E, "…et en ressort sous forme de 1 536 nombres", "", ms=1400)
enveloppe(s, 0.7, 12.6, 3.25, 1.2, couleurs=coul, cache=set(range(f0, f1)), vers=(BOX[0] + BOX[2] / 2 - 0.05, BOX[1] + BOX[3] / 2 - 0.05, 0.1, 0.1))
boite(s, 0.7 + f0 * bwp, 1.95, 10 * bwp, 1.4, None, line=GOLD, lw=3, nom="!!fen")
fleche(s, 0.7 + (f0 + 5) * bwp, 3.4, BOX[0] + BOX[2] / 2, BOX[1] - 0.05, color=GOLD, width=2, nom="!!fl1")
boite(s, *BOX, DARK, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12, nom="!!perch")
texte(s, BOX[0], BOX[1] + 0.2, BOX[2], 1.2, [[B("Perch")], ["Google · 12 millions de réglages appris sur 15 000 espèces"]], size=14, color="FFFFFF", align="c", nom="!!percht", after=4)
fleche(s, BOX[0] + BOX[2] / 2, BOX[1] + BOX[3] + 0.03, BOX[0] + BOX[2] / 2, CY - 0.08, color=GOLD, width=2, nom="!!fl2")
for j, v in enumerate(vals):
    boite(s, CX + j * CW, CY, CW - 0.02, 0.45, cellule(v), nom=f"!!c{j}")
texte(s, CX, CY + 0.52, 9.9, 0.3, "  ".join(fr(v, 3) for v in vals[:9]) + "  …  (48 des 1 536 nombres de cette fenêtre)", size=10, color=MUTED, nom="!!cval", font="Consolas")
texte(s, CX + 48 * CW + 0.25, CY - 0.05, 0.6, 0.55, "=", size=22, color=INK, bold=True, nom="!!egal")
boite(s, CX + 48 * CW + 0.75, CY + 0.17, R_PT, R_PT, POS, line=INK, forme=MSO_SHAPE.OVAL, nom=f"!!p{HERO}", lw=1.5)
texte(s, CX + 48 * CW + 0.95, CY + 0.08, 1.3, 0.3, "un point", size=13, color=INK, bold=True, nom="!!unpoint")
notes(s, """En sortie : 1 536 nombres. C'est la signature de la fenêtre. Que veut dire chaque nombre ? Personne ne sait le dire : c'est la boîte noire. Mais deux sons qui se ressemblent donnent deux listes de nombres proches. Et une liste de nombres, c'est une position : chaque fenêtre devient un point.""")

# ---------------------------------------------------------- 2 · regrouper
x0, y0, _ = VUES[0]
gris = [GRIS] * NPTS
gris[HERO] = POS
cont0 = [None] * NPTS
cont0[HERO] = INK

s = diapo(RUB_C, "Chaque fenêtre de 5 s devient un point", "", ms=1600)
nuage(s, x0, y0, gris, cont0, "nuage gris")
texte(s, x0[HERO] + 0.18, y0[HERO] - 0.08, 2.4, 0.3, "la fenêtre de Molokoï", size=11, color=POS, bold=True, nom="!!herot")
panneau(s, [
    f"{NPTS} fenêtres de Kaw, Molokoï, Trésor et Mataroni.",
    "1 536 nombres ne se dessinent pas : on montre leur ombre sur l'écran, selon les deux directions où les fenêtres diffèrent le plus.",
    [("Une ombre cache une partie du relief : on ne conclut jamais sur ce dessin seul.", {"italic": True, "color": MUTED})],
])
notes(s, """Voici notre fenêtre, et 400 autres, chacune un point. Un point à 1 536 coordonnées ne se dessine pas : on projette son ombre sur l'écran, selon les deux directions où les fenêtres diffèrent le plus. C'est une ombre : elle cache une partie du relief. On s'en sert pour voir, jamais pour conclure.""")

s = diapo(RUB_C, "L'ordinateur forme des groupes de fenêtres qui se ressemblent", "", ms=1000)
nuage(s, x0, y0, [GROUPES[g] for g in N["groupe"]], None, "groupes")
panneau(s, [
    "C'est le clustering : 4 groupes, calculés sur les 1 536 nombres, pas sur l'écran.",
    "Personne ne lui dit ce qu'il y a dans les fenêtres.",
    "Si le chant d'A. blanci marquait fortement la signature, un groupe devrait le rassembler.",
])
notes(s, """Première idée, la plus simple : demander à l'ordinateur de faire des paquets de fenêtres qui se ressemblent, sans rien lui dire. C'est le clustering. Si le chant d'A. blanci marquait fortement la signature, un des paquets devrait le rassembler.""")

s = diapo(RUB_C, "Les groupes ne suivent pas vraiment les forêts", "", ms=1000)
nuage(s, x0, y0, [SITES[k] for k in SITE], None, "sites")
legende(s, [(c, NOM_SITE[k], False) for k, c in SITES.items()])
panneau(s, [
    "Même nuage, coloré par forêt.",
    "Je m'attendais à ce que chaque groupe soit une forêt, avec son ambiance propre.",
    "Mataroni se détache un peu ; les autres forêts se mélangent.",
])
notes(s, """Recolorons par forêt. Je m'attendais à ce que les paquets soient les forêts, chacune avec son ambiance : sa rivière, ses insectes. Ce n'est vrai qu'en partie : Mataroni se détache un peu, le reste se mélange. Mais attention, c'est un dessin, une ombre : vérifions sur les vrais nombres.""")

# diapo mesurée : ce qui organise les groupes
C = D["clusters"]
s = diapo(RUB_C, "Mesuré : ni la forêt ni l'heure ne font les groupes, mais l'un d'eux attire le chant", "", transition=False)
fac = ["site", "point d'écoute", "moment de la journée", "mois"]
X0, Y0, LW, H = 0.9, 2.3, 6.6, 3.6
echelle = max(0.5, max(C[n][f]["hi"] for n in ("fenetre", "enregistrement") for f in fac) * 1.15)
texte(s, X0, 1.75, 7.5, 0.35, "Accord entre les groupes et chaque facteur (0 = aucun, 1 = les groupes sont ce facteur)", size=13, color=MUTED)
for k, f in enumerate(fac):
    yb = Y0 + k * 0.9
    texte(s, X0, yb + 0.05, 1.9, 0.6, f, size=13, color=INK, bold=True)
    for j, (niv, coul, _lib) in enumerate((("fenetre", TEAL, "fenêtres de 5 s"), ("enregistrement", GOLD, "enregistrements entiers"))):
        v = C[niv][f]
        yy = yb + j * 0.34
        boite(s, X0 + 2.0, yy, LW * v["ari"] / echelle, 0.26, coul)
        boite(s, X0 + 2.0 + LW * v["lo"] / echelle, yy + 0.12, LW * (v["hi"] - v["lo"]) / echelle, 0.02, INK)
        texte(s, X0 + 2.1 + LW * v["hi"] / echelle, yy - 0.02, 0.8, 0.3, fr(v["ari"], 2), size=11, color=INK)
boite(s, X0 + 2.0, 6.0, 0.2, 0.2, TEAL)
texte(s, X0 + 2.3, 5.97, 2.2, 0.3, f"{fr(C['n_fenetres'])} fenêtres", size=11, color=MUTED)
boite(s, X0 + 4.3, 6.0, 0.2, 0.2, GOLD)
texte(s, X0 + 4.6, 5.97, 4.2, 0.3, f"{fr(C['n_enregistrements'])} enregistrements (moyenne de leurs fenêtres)", size=11, color=MUTED)
pp = np.array(C["positifs_par_groupe"])
ph = np.array(C["hasard_par_groupe"])
gb = int(np.argmax(pp))
pos_g, tot_g = C["annotees_par_groupe"][gb]
pct = lambda x: f"{fr(100 * x)} %"  # noqa: E731
panneau(s, [
    [B("Méthode")],
    f"k-moyennes, k = {C['k']} (autant que de sites), sur les 8 sites ; trait noir : intervalle à 95 % (bootstrap, 200 tirages).",
    [B("Le groupe du chant")],
    f"Un groupe pèse {pct(ph[gb])} des fenêtres de la forêt, mais rassemble {pct(pp[gb])} de mes fenêtres avec chant ; {pct(pos_g / tot_g)} des fenêtres annotées qu'il contient ont du chant, sur tous les sites.",
    f"Mais {pct(1 - pp[gb])} des chants sont ailleurs, et ce que vaut le reste du groupe ne se saura qu'en l'écoutant.",
], size=12)
notes(s, f"""Je l'ai mesuré sur les 1 536 nombres, pas sur le dessin. Pour chaque facteur, un indice d'accord entre les groupes et ce facteur : zéro, aucun lien ; un, les groupes sont exactement ce facteur. En bleu-vert, au niveau des fenêtres de cinq secondes ; en jaune, au niveau des enregistrements entiers. Le site ressort le plus, mais l'accord reste faible : ni la forêt, ni l'heure, ni le mois n'organisent vraiment ces groupes. En revanche, surprise : un des {C['k']} groupes, qui ne pèse que {pct(ph[gb])} de la forêt, rassemble {pct(pp[gb])} de mes fenêtres avec chant, et presque toutes les fenêtres annotées qui y tombent ont du chant. Le clustering n'est donc pas aveugle au chant. Mais il en laisse plus de la moitié ailleurs, il ne donne aucune certitude, et il ne s'améliore pas quand j'écoute. Pour ça, il faut apprendre à partir de mes exemples. Et c'est une piste pour la suite : écouter une trentaine de fenêtres tirées au hasard dans ce groupe.""")

# ---------------------------------------------------------- 3 · apprendre
# split : annotées (tour 0 + héros) à gauche, en attente à droite
connu0 = (TOUR == -2) | (TOUR == 0)
u0, v0 = N["vues"][0]
u0, v0 = np.array(u0), np.array(v0)
xs, ys = np.zeros(NPTS), np.zeros(NPTS)
L = (0.75, 2.25, 3.6, 4.3)
R = (4.95, 2.25, 3.6, 4.3)
for masque, box in ((connu0, L), (~connu0, R)):
    xx, yy, _ = ecran(u0[masque], v0[masque], box)
    xs[masque], ys[masque] = xx, yy
coul0, cont_0 = etat(0)
n_ann0 = int(connu0.sum())

s = diapo(RUB_K, "J'écoute et j'annote : la machine apprend sur mes exemples", "", ms=1500)
boite(s, 4.7, 1.8, 0.02, 4.85, SAGE, nom="!!split")
texte(s, 0.75, 1.8, 3.6, 0.35, f"Annotées ({n_ann0})", size=14, color=INK, bold=True, nom="!!hl")
texte(s, 4.95, 1.8, 3.6, 0.35, f"En attente ({NPTS - n_ann0})", size=14, color=MUTED, bold=True, nom="!!hr")
nuage(s, xs, ys, coul0, cont_0, "split")
legende(s, [(POS, "A. blanci", True), (NEG, "autre son", True), (GRIS, "en attente", False)])
A_ = D
panneau(s, [
    "J'écoute des fenêtres et je dis : A. blanci, ou autre chose (oiseau, pluie, autre grenouille…).",
    f"À ce jour : {fr(A_['n_annotees'])} fenêtres annotées, dont {fr(A_['n_positives'])} avec A. blanci, sur {A_['n_micros_annotes']} points d'écoute.",
    "Ici, on rejoue l'apprentissage depuis le début, avec mes vraies annotations.",
])
notes(s, f"""Deuxième idée : apprendre. J'écoute des fenêtres et je les étiquette : A. blanci, ou autre chose. À ce jour, {fr(A_['n_annotees'])} fenêtres annotées, dont {fr(A_['n_positives'])} avec le chant. À gauche, les fenêtres déjà écoutées, à droite l'immense majorité, en attente. Ce que vous allez voir est un rejeu : on repart de quelques dizaines d'annotations et on refait le chemin, avec mes vraies étiquettes.""")

# scanner
poids = np.array(bande["poids"])
SY1, SY2 = 3.0, 4.35


def bandes(s, scan_x, allume):
    for j, v in enumerate(vals):
        pale = allume and poids[j] < 0.35
        boite(s, CX + j * CW, SY1, CW - 0.02, 0.5, cellule(v, pale), nom=f"!!c{j}")
        pj = poids[j] if allume else 0
        g = "".join(f"{round(255 + (c - 255) * pj):02X}" for c in (0x1E, 0x6B, 0x45))
        boite(s, CX + j * CW, SY2, CW - 0.02, 0.5, g if allume else GRIS_PALE, line=GRIS_PALE, nom=f"!!w{j}")
    boite(s, scan_x, SY1 - 0.2, 0.06, 1.85, GOLD, nom="!!scan")
    texte(s, CX, SY1 - 0.5, 9.0, 0.3, "Les nombres d'une fenêtre avec A. blanci (48 des 1 536)", size=12, color=MUTED, nom="!!sl1")
    texte(s, CX, SY2 + 0.58, 9.0, 0.3, "Poids appris par la tête : l'importance de chaque nombre", size=12, color=MUTED, nom="!!sl2")


s = diapo(RUB_K, "Une petite tête de classification cherche ce qui distingue mes exemples", "", ms=900)
bandes(s, CX - 0.15, False)
texte(s, 1.0, 5.6, 11.6, 1.2, "La tête est un petit calcul ajouté au bout de Perch : la seule partie que j'entraîne, en quelques secondes, sur mes annotations.", size=15, color=INK, nom="!!sct")
notes(s, """La machine reprend la signature de 1 536 nombres. Au bout de Perch, j'ajoute une petite tête de classification : c'est la seule partie que j'entraîne, et elle s'entraîne en quelques secondes sur un portable. Elle va passer les nombres au crible.""")

s = diapo(RUB_K, "Elle apprend quels nombres comptent, et ignore les autres", "", ms=2200)
bandes(s, CX + 48 * CW + 0.1, True)
texte(s, 1.0, 5.6, 11.6, 1.2, "Un poids pour chacun des 1 536 nombres : forts sur certains, presque nuls ailleurs. Ensemble, ils dessinent une direction « ressemble au chant d'A. blanci ». Aucun nombre n'est, à lui seul, « le nombre blanci ».", size=15, color=INK, nom="!!sct")
notes(s, """Elle apprend un poids pour chaque nombre : élevé là où mes exemples avec chant diffèrent des autres, presque nul ailleurs. Les nombres qui comptent s'allument, les autres s'éteignent. Attention : ce n'est pas un nombre magique qui serait la grenouille. C'est une combinaison de poids, une direction dans l'espace des signatures. Et cette direction, on peut la regarder.""")

# retour au nuage, vue ACP, annotations de départ
s = diapo(RUB_K, "Vu sous l'angle habituel, mes exemples sont mêlés", "", ms=1500)
nuage(s, x0, y0, coul0, cont_0, "annotées, ACP")
boussole(s, 0)
legende(s, [(POS, "A. blanci", True), (NEG, "autre son", True), (GRIS, "en attente", False)])
panneau(s, [
    "Retour à l'ombre du début.",
    "Verts et rouges sont mêlés : aucune ligne droite ne les sépare sur ce dessin.",
    "La boussole indique d'où l'on regarde le nuage.",
])
notes(s, """Revenons au nuage, vu sous le même angle qu'au début. Les verts et les rouges sont mêlés : aucune ligne ne les sépare. La boussole, en haut à droite, dit d'où l'on regarde. La flèche verte, c'est la direction apprise par la tête : pour l'instant, elle pointe presque vers nous, on ne la voit pas.""")

for k in range(1, len(N["vues"])):
    xk, yk, _ = VUES[k]
    dernier = k == len(N["vues"]) - 1
    s = diapo(RUB_K, "On fait tourner le nuage vers la direction apprise" if not dernier else "Vu sous l'angle de la tête, les chants forment un groupe", "", ms=1100, auto=not dernier)
    nuage(s, xk, yk, coul0, cont_0, f"rotation {k}")
    boussole(s, k)
    legende(s, [(POS, "A. blanci", True), (NEG, "autre son", True), (GRIS, "en attente", False)])
    panneau(s, [
        "Mêmes points, même nuage : seul l'angle de vue change.",
        "On tourne jusqu'à regarder le nuage dans la direction apprise par la tête.",
    ] if not dernier else [
        "Axe horizontal : la direction apprise par la tête.",
        "À droite, ce qui ressemble au chant ; à gauche, le reste.",
        "Les verts et les rouges mêlés tout à l'heure étaient déjà séparés : l'ombre le cachait.",
    ])
    notes(s, "Le nuage tourne : ce sont exactement les mêmes points, on change seulement l'angle de vue." if not dernier else
          """Et voilà : regardé dans la direction apprise, le nuage se range. Les verts à droite, les rouges à gauche. Les points mêlés tout à l'heure étaient déjà séparés dans l'espace complet : c'est l'ombre en deux dimensions qui le cachait. Précision pour les curieux : l'axe est celui de la tête entraînée sur toutes mes annotations ; je le garde fixe pour la suite, pour que les points ne bougent plus et qu'on ne regarde que les couleurs.""")

xf, yf, (lo_u, hi_u) = VUES[-1]
seuil_x = GX + (np.clip(N["seuil_u"], lo_u, hi_u) - lo_u) / (hi_u - lo_u) * (GW - R_PT)


def ligne_seuil(s):
    boite(s, float(seuil_x), GY, 0.001, GH, line=MUTED, nom="!!seuil", lw=1.25, dash=True)


LEG_SC = [(POS, "A. blanci (annoté)", True), (NEG, "autre son (annoté)", True), (MID, "classé par le modèle", False)]
c, k_ = etat(0, 0)
s = diapo(RUB_K, "La tête note les fenêtres en attente, avec sa certitude", "", ms=1200)
nuage(s, xf, yf, c, k_, "scores v0")
ligne_seuil(s)
legende(s, LEG_SC)
panneau(s, [
    "Sans contour : la couleur vient du modèle, pas de mon écoute.",
    "Vert foncé : sûr que c'est A. blanci. Rouge terre : sûr que non. Or : il hésite.",
    "Ligne : le seuil de décision.",
], h=3.6)
jauge(s, 0)
notes(s, f"""Les fenêtres en attente reçoivent une note de la tête. Sans contour, pour rappeler que ce n'est pas moi qui ai écouté. Vert foncé, le modèle est sûr qu'il y a le chant ; rouge terre, sûr que non ; doré, il hésite. En bas à droite, le score du modèle, mesuré sur des points d'écoute qu'il n'a jamais entendus : {fr(N['ap_rejeu'][0], 2)} sur 1 pour l'instant.""")

# file 60-20-20 du tour 1
file1 = {"incertaine": [], "meilleur score": [], "hasard": []}
for i in range(NPTS):
    if TOUR[i] == 1 and N["cat1"][i] in file1:
        file1[N["cat1"][i]].append(i)
quota = {"incertaine": 6, "meilleur score": 2, "hasard": 2}
choisis = [(cat, i) for cat in file1 for i in file1[cat][: quota[cat]]]
POP = (0.75, 1.85, 4.3, 4.75)
s = diapo(RUB_K, "Je réécoute ce que le modèle me propose : la file 60-20-20", "", ms=1500)
c, k_ = etat(0, 0)
xx, yy = xf.copy(), yf.copy()
rows = []
yrow = POP[1] + 0.55
for cat in file1:
    sel = [i for c_, i in choisis if c_ == cat]
    if not sel:
        continue
    rows.append((cat, yrow, len(sel)))
    for i in sel:
        xx[i], yy[i] = POP[0] + 0.35, yrow
        yrow += 0.3
    yrow += 0.3
ids_file = [i for _, i in choisis]
nuage(s, xx, yy, c, k_, "file", seulement=[i for i in range(NPTS) if i not in set(ids_file)])
boite(s, *POP, "FFFFFF", line=INK, lw=1.25, nom="!!popup")
nuage(s, xx, yy, c, k_, seulement=ids_file)
texte(s, POP[0] + 0.2, POP[1] + 0.12, POP[2] - 0.4, 0.35, "File de réécoute", size=14, bold=True, nom="!!popt")
for cat, yr, _n in rows:
    pct = {"incertaine": "60 % : là où il hésite", "meilleur score": "20 % : ses meilleurs scores", "hasard": "20 % : au hasard"}[cat]
    texte(s, POP[0] + 0.6, yr - 0.05, 2.0, 0.3, pct, size=10, color=MUTED, nom=f"!!pc_{cat}")
for _cat, i in choisis:
    fleche(s, POP[0] + 2.65, yy[i] + R_PT / 2, POP[0] + 3.2, yy[i] + R_PT / 2, color=SAGE, width=1.25, nom=f"!!fa{i}")
    boite(s, POP[0] + 3.3, yy[i], R_PT, R_PT, POS if Y[i] == 1 else NEG, line=INK, forme=MSO_SHAPE.OVAL, nom=f"!!r{i}", lw=1.5)
    texte(s, POP[0] + 3.5, yy[i] - 0.07, 0.8, 0.25, "blanci" if Y[i] == 1 else "autre", size=10, color=INK, nom=f"!!rt{i}")
legende(s, LEG_SC)
panneau(s, [
    "Le modèle choisit quoi me faire écouter :",
    "60 % de fenêtres où il hésite, 20 % de ses meilleurs scores (pour traquer ses fausses alertes), 20 % au hasard (pour ne rien oublier).",
    "À droite de chaque fenêtre : ce que j'ai entendu.",
], h=3.6)
jauge(s, 0)
notes(s, """Le modèle me propose une file de réécoute. Soixante pour cent de fenêtres où il hésite : c'est là que mon oreille lui apprend le plus. Vingt pour cent de ses meilleurs scores, pour attraper ses fausses alertes. Vingt pour cent au hasard, pour qu'il ne s'enferme pas dans ses certitudes. J'écoute, et je tranche : à droite de chaque fenêtre, ce que j'ai entendu. Ces fenêtres réécoutées ne servent jamais à noter le modèle.""")

s = diapo(RUB_K, "Mes réponses entrent dans les annotations ; on réentraîne", "", ms=1300)
c, k_ = etat(1)
nuage(s, xf, yf, c, k_, "réentraînement")
ligne_seuil(s)
legende(s, LEG_SC)
panneau(s, ["Les fenêtres réécoutées reprennent leur place, avec mon étiquette.", "Le modèle se réentraîne : les autres repassent en attente."], h=3.6)
jauge(s, 0)
notes(s, """Les fenêtres reprennent leur place, cette fois avec mon étiquette, gravée dans le marbre. On réentraîne la tête : quelques secondes. Les autres points repassent en attente.""")

s = diapo(RUB_K, "Nouvelle version du modèle, nouvelles certitudes", "", ms=1000)
c, k_ = etat(1, 1)
nuage(s, xf, yf, c, k_, "scores v1")
ligne_seuil(s)
legende(s, LEG_SC)
panneau(s, ["Chaque tour : le modèle propose, j'écoute, il se réentraîne.", "Les hésitations se resserrent autour du seuil."], h=3.6)
jauge(s, 1)
notes(s, f"""Nouvelle version, nouvelles notes. Le score passe à {fr(N['ap_rejeu'][1], 2)}. Et on recommence : le modèle propose, j'écoute, il se réentraîne.""")

for t in range(2, 5):
    s = diapo(RUB_K, f"Tour {t} : j'annote la file", "", ms=700)
    c, k_ = etat(t, t - 1)
    nuage(s, xf, yf, c, k_, f"A{t}")
    ligne_seuil(s)
    legende(s, LEG_SC)
    panneau(s, ["A : de nouvelles fenêtres reçoivent mon étiquette."], h=3.6)
    jauge(s, t - 1)
    notes(s, "J'écoute la file suivante.")
    s = diapo(RUB_K, f"Tour {t} : le modèle se réentraîne", "", ms=700)
    c, k_ = etat(t, t)
    nuage(s, xf, yf, c, k_, f"B{t}")
    ligne_seuil(s)
    legende(s, LEG_SC)
    panneau(s, ["B : toutes les fenêtres reçoivent leur nouveau score."], h=3.6)
    jauge(s, t)
    notes(s, f"Nouvelle version : score {fr(N['ap_rejeu'][t], 2)}.")

# évaluation
SIM = D["simulation"]
cv = SIM["courbes"]
s = diapo(RUB_K, "Comment on saura que ça marche", "", transition=False)
s.shapes.add_picture(str(ICI / "courbe.png"), Inches(0.6), Inches(1.75), height=Inches(4.6))
texte(s, 0.6, 6.4, 7.2, 0.6, f"Rejeu sur mes annotations : départ à {SIM['N0']} fenêtres, +{SIM['lot']} par tour ; 10 répétitions (bande : min-max). Noté sur {SIM['n_eval_micros']} points d'écoute jamais vus ({fr(SIM['n_eval'])} fenêtres, dont {SIM['n_eval_pos']} avec chant).", size=11, color=MUTED, italic=True)
panneau(s, [
    [B("La note : la précision moyenne")],
    f"1 : toutes les fenêtres avec chant classées avant les autres ; au hasard : {fr(SIM['prevalence_eval'], 2)} ici.",
    [B("Ce que dit le rejeu")],
    f"{SIM['N0']} fenêtres suffisent pour {fr(cv['actif']['moy'][0], 2)} ; la note plafonne vers {fr(cv['actif']['moy'][-1], 2)}. La file 60-20-20 ne bat pas le hasard ici.",
    [B("Pas encore de note officielle")],
    "Ici le chant est fréquent (extraits choisis) ; dans la forêt entière il est bien plus rare et la note sera plus basse. Il faut geler un jeu de test tiré au sort.",
], size=12)
notes(s, f"""Comment saura-t-on que ça marche ? Le modèle est noté sur des points d'écoute qu'il n'a jamais entendus, avec la précision moyenne : un, s'il classe toutes les fenêtres avec chant avant les autres ; un modèle qui répondrait au hasard aurait ici {fr(SIM['prevalence_eval'], 2)}. Sur le rejeu, avec seulement {SIM['N0']} fenêtres annotées, la note est déjà de {fr(cv['actif']['moy'][0], 2)} : la signature de Perch porte bien le chant. Elle monte puis plafonne vers {fr(cv['actif']['moy'][-1], 2)}. Et honnêtement, sur ce rejeu, choisir les fenêtres par la file 60-20-20 ne fait pas mieux que les tirer au hasard : mes annotations ont déjà été choisies par des files, le hasard parmi elles n'est pas le hasard dans la forêt. Deux limites, donc, avant d'annoncer un chiffre : ici le chant est fréquent, parce que j'ai écouté des extraits où il était probable ; dans la forêt entière il est bien plus rare, et la note sera plus basse. Il faut geler un jeu de test : des enregistrements tirés au sort, écoutés en entier, que le modèle ne verra jamais.""")

# la suite
s = diapo(RUB_S, "Février 2027 : de la carte SD à la carte de présence", "", transition=False)
etapes = [("Cartes SD", "nouvelle campagne"), ("Contrôle qualité", "pistes, durées, relevés"), ("Encodage Perch", "≈ 50 fois plus vite que l'écoute"), ("Scores", "par fenêtre et par enregistrement"), ("Écoute ciblée", "les hésitations d'abord")]
for k, (t, st) in enumerate(etapes):
    x = 0.6 + k * 2.5
    boite(s, x, 1.95, 2.2, 1.35, PALER if k < 4 else PALE, forme=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    texte(s, x + 0.15, 2.1, 1.9, 0.4, t, size=15, bold=True, color=TEAL if k < 4 else POS, align="c")
    texte(s, x + 0.15, 2.6, 1.9, 0.6, st, size=11, color=MUTED, align="c")
    if k < 4:
        fleche(s, x + 2.22, 2.62, x + 2.48, 2.62, color=SAGE, width=2)
fleche(s, 11.6, 3.35, 11.6, 3.75, color=SAGE, width=2)
texte(s, 7.8, 3.8, 4.95, 0.6, "réentraînement : les nouvelles données affinent le modèle", size=12, color=MUTED, align="r", italic=True)
texte(s, 0.6, 4.4, 5.9, 2.6, [
    [B("Ce que le stage livre")],
    f"Un corpus contrôlé et figé ({fr(95633)} enregistrements).",
    f"Un poste d'annotation et {fr(D['n_annotees'])} fenêtres annotées.",
    "La chaîne encodage → tête → file d'écoute, testée.",
], size=14, after=6)
texte(s, 6.9, 4.4, 5.9, 2.6, [
    [B("Ce qu'il reste à faire")],
    "Geler le jeu de test et donner une note officielle.",
    "Choisir l'encodeur sur nos données (Perch, BirdNET 3).",
    "Suivre le modèle dans le temps : chaque nouvelle campagne est sa vraie épreuve.",
], size=14, after=6)
notes(s, """Concrètement, pour la campagne de février : à la récupération des cartes, le contrôle qualité, puis l'encodage, une cinquantaine de fois plus rapide que l'écoute, puis les scores, et l'écoute ciblée des cas douteux. Ces nouvelles écoutes réentraînent le modèle. Le stage livre un corpus propre, un poste d'annotation, et la chaîne complète. Il reste à geler un jeu de test pour donner une vraie note, à choisir définitivement l'encodeur sur nos données, et à suivre le modèle dans le temps : chaque nouvelle campagne sera son vrai test.""")

# merci
s = prs.slides.add_slide(BLANK)
fond(s, DARK)
s.shapes.add_picture(str(FIG / "titre.png"), Inches(8.9), Inches(0), height=Inches(7.5))
texte(s, 0.8, 2.6, 7.6, 1.2, "Merci", size=48, color="FFFFFF", bold=True, font="Cambria")
texte(s, 0.8, 3.8, 7.6, 1.2, "Questions, remarques, enregistrements à partager : je suis preneur.", size=18, color=SAGE)
texte(s, 0.8, 6.15, 7.6, 0.35, "Léonard Laplace-Palette", size=14, color="FFFFFF", bold=True)
notes(s, "Merci. Je prends vos questions.")

prs.save(SORTIE)
print("écrit", SORTIE, len(prs.slides), "diapos ; audio inséré :", audio_ok)

# ------------------------------------------------------------------ planche de contrôle
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

n = len(PLANCHE)
cols = 5
fig, axes = plt.subplots(-(-n // cols), cols, figsize=(4 * cols, 2.6 * -(-n // cols)))
for ax in axes.flat:
    ax.set_axis_off()
for ax, (titre, x, y, coul, cont) in zip(axes.flat, PLANCHE, strict=False):
    ax.set_axis_on()
    ax.scatter(x + R_PT / 2, -(y + R_PT / 2), s=10, c=["#" + c for c in coul],
               edgecolors=["#" + c if c else "none" for c in (cont or [None] * len(x))], linewidths=1)
    ax.set_xlim(GX - 0.1, GX + GW + 0.1)
    ax.set_ylim(-(GY + GH + 0.1), -GY + 0.1)
    ax.set_title(titre, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(ICI / "planche.png", dpi=70)
