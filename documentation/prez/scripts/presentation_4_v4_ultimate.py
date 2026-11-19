"""Présentation n° 4, version « v4_ultimate » : la v3_fusion remise en page.

Lit `presentation_4_inventaire_qc_v3_fusion.pptx` (jamais modifié) et écrit
`presentation_4_inventaire_qc_v4_ultimate.pptx` à côté. Les notes orateur sont gardées.

Ce que fait la v4 :

- une pastille « CODE » en haut à droite de chaque diapo qui montre un résultat : `ingest.py`,
  `qc.py`, `inventaire.py`, sans chemin ni nom de fonction ;
- plus d'encart vert foncé à texte jaune : des cartes claires, toutes au même gabarit ;
- marges de 0,60 po, contenu de 1,75 à 6,92 po, mêmes écarts partout ;
- titres et légendes des anciens graphiques (notebook 06) retirés de l'image et réécrits dans la
  diapo, en Calibri : les images retouchées sont dans `figures/inventaire_qc/v4/` ;
- chiffres et trouvailles de `chiffres_presentation_4.md` et de `passation_inventaire.md`.

Aucune lecture de la base : les chiffres viennent du rejeu du 18/11/2026.

    python documentation/prez/scripts/presentation_4_v4_ultimate.py
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

PREZ = Path(__file__).resolve().parents[1]
V3 = PREZ / "presentation_4_inventaire_qc_v3_fusion.pptx"
V4 = PREZ / "presentation_4_inventaire_qc_v4_ultimate.pptx"
FIG = PREZ / "figures" / "inventaire_qc" / "v4"

INK, MUTED, TEAL, DEEP = "1B2A24", "5B6B63", "2A7F72", "0E2A24"
PALE, PALE2, SAGE, WHITE = "F3F7F4", "E4EEE8", "A7BFB2", "FFFFFF"
RED, GOLD, ORANGE, GREEN = "C8553D", "9A6B0F", "D9822B", "2E8B57"
L, R, TOP, BOTTOM, GAP = 0.60, 12.73, 1.75, 6.92, 0.20
W = R - L
COL = (W - GAP) / 2
X2 = L + COL + GAP
MONO, SANS, SERIF = "Consolas", "Calibri", "Cambria"

STYLES = {
    "t": {"bold": True, "color": TEAL}, "r": {"bold": True, "color": RED},
    "g": {"bold": True, "color": GOLD}, "b": {"bold": True}, "mu": {"color": MUTED},
    "m": {"font": MONO}, "mt": {"font": MONO, "bold": True, "color": TEAL},
    "ms": {"font": MONO, "color": MUTED, "dsize": -2, "bold": False},
    "mub": {"bold": True, "color": MUTED}, "mk": {"font": MONO, "bold": True},
    "y": {"color": "F2C14E"}, "i": {"italic": True}, "sm": {"color": MUTED, "dsize": -2.5},
}
MARK = re.compile(r"⟦(\w+)\|(.*?)⟧")


# ------------------------------------------------------------------ texte : mesure et écriture
def _police(nom: str, gras: bool) -> ImageFont.FreeTypeFont | None:
    noms = {(SANS, False): ["calibri.ttf", "Carlito-Regular.ttf"],
            (SANS, True): ["calibrib.ttf", "Carlito-Bold.ttf"],
            (MONO, False): ["consola.ttf", "DejaVuSansMono.ttf"],
            (MONO, True): ["consolab.ttf", "DejaVuSansMono-Bold.ttf"]}[(nom, gras)]
    for dossier in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/crosextra",
                    "/usr/share/fonts/truetype/dejavu"):
        for n in noms:
            if (Path(dossier) / n).is_file():
                return ImageFont.truetype(str(Path(dossier) / n), 100)
    return None


_POLICES: dict = {}


def largeur(texte: str, taille: float, font: str = SANS, gras: bool = False) -> float:
    """Largeur du texte en pouces."""
    cle = (font if font == MONO else SANS, gras)
    if cle not in _POLICES:
        _POLICES[cle] = _police(*cle)
    f = _POLICES[cle]
    if f is None:
        return len(texte) * taille * 0.5 / 72
    return f.getlength(texte) * taille / 100 / 72


def morceaux(texte: str) -> list[tuple[str, dict]]:
    out, pos = [], 0
    for m in MARK.finditer(texte):
        if m.start() > pos:
            out.append((texte[pos:m.start()], {}))
        out.append((m.group(2), STYLES[m.group(1)]))
        pos = m.end()
    if pos < len(texte):
        out.append((texte[pos:], {}))
    return out


def lignes(texte: str, w: float, taille: float, font: str = SANS, gras: bool = False) -> int:
    n, x = 1, 0.0
    for seg, st in morceaux(texte):
        f, g = st.get("font", font), st.get("bold", gras)
        t = taille + st.get("dsize", 0)
        for mot in re.findall(r"\S+\s*", seg):
            lw = largeur(mot.rstrip(), t, f, g)
            if x + lw > w and x > 0:
                n, x = n + 1, 0.0
            x += largeur(mot, t, f, g)
    return n


def hauteur(paras: list, w: float, taille: float, apres: float = 4, inter: float = 1.2) -> float:
    h = 0.0
    for p in paras:
        texte, o = p if isinstance(p, tuple) else (p, {})
        t = o.get("size", taille)
        h += lignes(texte, w, t, o.get("font", SANS), o.get("bold", False)) * t * inter / 72
        h += o.get("after", apres) / 72
    return h


def rgb(c: str) -> RGBColor:
    return RGBColor.from_string(c)


def tb(s, x, y, w, h, paras, size=12, color=INK, bold=False, font=SANS, align="l", anchor="t",
       after=4, italic=False):
    """Zone de texte sans marge. `paras` : textes (ou (texte, options)), balises ⟦style|…⟧."""
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(paras, str):
        paras = [paras]
    for k, p in enumerate(paras):
        texte, o = p if isinstance(p, tuple) else (p, {})
        par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        par.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[
            o.get("align", align)]
        par.space_after = Pt(o.get("after", after))
        for seg, st in morceaux(texte):
            r = par.add_run()
            r.text = seg
            r.font.name = st.get("font", o.get("font", font))
            r.font.size = Pt(o.get("size", size) + st.get("dsize", 0))
            r.font.bold = st.get("bold", o.get("bold", bold))
            r.font.italic = st.get("italic", o.get("italic", italic))
            r.font.color.rgb = rgb(st.get("color", o.get("color", color)))
    return box


def forme(s, kind, x, y, w, h, fill=None, line=None, lw=0.75, adj=None):
    sh = s.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if adj is not None:
        sh.adjustments[0] = adj
    if fill:
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(lw)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def carte(s, x, y, w, h, fill=PALE, bar=None, rayon=0.07):
    """Carte claire à coins de rayon constant, avec une barre d'accent à gauche si `bar`."""
    forme(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=fill, adj=min(0.5, rayon / min(w, h)))
    if bar:
        forme(s, MSO_SHAPE.RECTANGLE, x, y + 0.09, 0.06, h - 0.18, fill=bar)


def carte_texte(s, x, y, w, paras, size=12, bar=TEAL, fill=PALE, h=None, after=5, pad=0.13,
                anchor="t"):
    """Carte dont la hauteur suit le texte (ou `h` imposée). Renvoie la hauteur."""
    gauche = 0.26 if bar else 0.20
    wt = w - gauche - 0.20
    if h is None:
        h = hauteur(paras, wt, size, after) + 2 * pad
    carte(s, x, y, w, h, fill=fill, bar=bar)
    tb(s, x + gauche, y + pad, wt, h - 2 * pad, paras, size=size, after=after, anchor=anchor)
    return h


def etiquette(s, x, y, w, texte, color=TEAL):
    """Intertitre en petites capitales."""
    tb(s, x, y, w, 0.24, texte, size=11, color=color, bold=True)


def titre_fig(s, x, y, w, texte, sous=None, tag=None):
    """Titre de graphique écrit dans la diapo : Calibri 13 gras, sous-titre 11, fichier à droite."""
    tb(s, x, y, w, 0.26, texte, size=13, bold=True)
    if sous:
        tb(s, x, y + 0.26, w, 0.22, sous, size=11, color=MUTED)
    if tag:
        tb(s, x, y + 0.03, w, 0.2, tag, size=9.5, color=TEAL, font=MONO, align="r")


def legende(s, x, y, items, size=11, pas=0.24, horizontal=False):
    """Légende écrite dans la diapo. items : (marque, couleur, texte) ; marques : sq, line, dash,
    dot, tri, star."""
    for marque, c, texte in items:
        cy = y + pas / 2
        if marque == "sq":
            forme(s, MSO_SHAPE.RECTANGLE, x, cy - 0.065, 0.22, 0.13, fill=c)
        elif marque == "line":
            forme(s, MSO_SHAPE.RECTANGLE, x, cy - 0.018, 0.22, 0.036, fill=c)
        elif marque == "dash":
            for k in range(3):
                forme(s, MSO_SHAPE.RECTANGLE, x + k * 0.085, cy - 0.012, 0.05, 0.024, fill=c)
        elif marque == "dot":
            forme(s, MSO_SHAPE.OVAL, x + 0.06, cy - 0.05, 0.10, 0.10, fill=c)
        elif marque == "tri":
            forme(s, MSO_SHAPE.ISOSCELES_TRIANGLE, x + 0.05, cy - 0.06, 0.12, 0.11, line=c, lw=1.25)
        elif marque == "star":
            forme(s, MSO_SHAPE.STAR_5_POINT, x + 0.03, cy - 0.08, 0.16, 0.16, fill=c)
        wt = largeur(texte, size) + 0.1
        tb(s, x + 0.30, y, wt + 0.3, pas, texte, size=size, anchor="m")
        if horizontal:
            x += 0.30 + wt + 0.25
        else:
            y += pas


def pastilles(s, noms):
    """« CODE » et les fichiers Python dont la diapo montre les résultats, en haut à droite."""
    x = R
    for nom in reversed(noms):
        w = len(nom) * 0.088 + 0.30
        x -= w
        forme(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 0.40, w, 0.32, fill=WHITE, line=TEAL, lw=1, adj=0.5)
        tb(s, x, 0.40, w, 0.32, nom, size=10.5, color=DEEP, bold=True, font=MONO, align="c",
           anchor="m")
        x -= 0.10
    tb(s, x - 0.70, 0.40, 0.62, 0.32, "CODE", size=9, color=MUTED, bold=True, align="r", anchor="m")


def kpis(s, y, items, h=1.30):
    w = (W - 3 * GAP) / 4
    for k, (nombre, couleur, texte) in enumerate(items):
        x = L + k * (w + GAP)
        carte(s, x, y, w, h)
        tb(s, x + 0.20, y + 0.10, w - 0.4, 0.56, nombre, size=28, color=couleur, bold=True,
           font=SERIF, anchor="m")
        tb(s, x + 0.20, y + 0.70, w - 0.4, h - 0.76, texte, size=11, color=MUTED)


# ------------------------------------------------------------------ formes de la v3
def par_id(s, sid):
    return next(sh for sh in s.shapes if sh.shape_id == sid)


def place(sh, x, y, w=None, h=None):
    sh.left, sh.top = Inches(x), Inches(y)
    if w is not None:
        sh.width = Inches(w)
    if h is not None:
        sh.height = Inches(h)


def garder(s, ids):
    """Supprime toutes les formes sauf `ids` (et libère les images supprimées)."""
    for sh in list(s.shapes):
        if sh.shape_id in ids:
            continue
        rid = sh._element.blip_rId if sh.shape_type == 13 else None
        sh._element.getparent().remove(sh._element)
        if rid:
            s.part.drop_rel(rid)


def texte_forme(sh, texte):
    runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
    runs[0].text = texte
    for r in runs[1:]:
        r.text = ""


def note(s, ancien, nouveau):
    """Remplace un passage des notes orateur (il doit s'y trouver)."""
    for p in s.notes_slide.notes_text_frame.paragraphs:
        complet = "".join(r.text for r in p.runs)
        if ancien in complet:
            p.runs[0].text = complet.replace(ancien, nouveau)
            for r in p.runs[1:]:
                r.text = ""
            return
    raise ValueError(f"note introuvable : {ancien[:60]}")


def entete(s, numero, noms=()):
    """Pastille de partie, titre et numéro aux mêmes places partout ; pastilles de code."""
    place(par_id(s, 5), 12.33, 7.08, 0.50, 0.25)
    if noms:
        pastilles(s, noms)


# ------------------------------------------------------------------ images retouchées
def image_v3(prs, diapo, sid) -> Image.Image:
    return Image.open(io.BytesIO(par_id(prs.slides[diapo - 1], sid).image.blob)).convert("RGB")


def effacer(im: Image.Image, boite) -> None:
    """Blanchit une zone (légende) puis retrace les traits de grille qui la traversaient."""
    a = np.asarray(im).copy()
    x0, y0, x1, y1 = boite
    a[y0:y1, x0:x1] = 255
    for y in range(y0, y1):
        g, d = a[y, x0 - 2], a[y, x1 + 1]
        if g.sum() < 740 and np.abs(g.astype(int) - d).sum() < 30:
            a[y, x0:x1] = g
    for x in range(x0, x1):
        h, b = a[y0 - 2, x], a[y1 + 1, x]
        if h.sum() < 740 and np.abs(h.astype(int) - b).sum() < 30:
            a[y0:y1, x] = h
    im.paste(Image.fromarray(a))


def figures(prs) -> dict[str, Path]:
    """Les anciens graphiques sans leur titre ni leur légende (réécrits dans la diapo)."""
    FIG.mkdir(parents=True, exist_ok=True)
    plan = {  # nom : (diapo, forme, légende à effacer, rognage gauche-haut-droite-bas)
        "gains_mesure": (7, 32, (520, 70, 925, 150), (0, 56, 1348, 700)),
        "welch": (15, 6, (165, 452, 695, 533), (0, 56, 1456, 659)),
        "deux_pistes": (16, 6, (170, 115, 745, 230), (0, 80, 1210, 1034)),
        "suites": (17, 18, (755, 32, 1445, 112), None),
        "origine_cdr": (19, 6, None, (0, 56, 1432, 780)),
        "micro_sourd": (20, 10, (1775, 64, 2350, 132), (0, 54, 2380, 783)),
    }
    out = {}
    for nom, (diapo, sid, boite, rognage) in plan.items():
        im = image_v3(prs, diapo, sid)
        if boite:
            effacer(im, boite)
        if rognage:
            im = im.crop(rognage)
        out[nom] = FIG / f"{nom}.png"
        im.save(out[nom])
    # Bilan : « 2 fichiers illisibles » devient 26 (rapports d'erreurs de l'inventaire, 18/11).
    im = image_v3(prs, 21, 41)
    d = ImageDraw.Draw(im)
    d.rectangle((486, 562, 1000, 606), fill="white")
    italique = next((p for p in ("C:/Windows/Fonts/calibrii.ttf",
                                 "/usr/share/fonts/truetype/crosextra/Carlito-Italic.ttf")
                     if Path(p).is_file()), None)
    f = ImageFont.truetype(italique, 31) if italique else ImageFont.load_default()
    d.text((492, 584), "26, jamais inventoriés", fill="#" + MUTED, font=f, anchor="lm")
    out["recap"] = FIG / "recap_criteres.png"
    im.save(out["recap"])
    return out


def image(s, chemin, x, y, w=None, h=None):
    return s.shapes.add_picture(str(chemin), Inches(x), Inches(y),
                                Inches(w) if w else None, Inches(h) if h else None)


# ------------------------------------------------------------------ les diapos
def diapo_1(s):
    for r in par_id(s, 4).text_frame.paragraphs[0].runs:
        r.font.color.rgb = rgb(WHITE)


def diapo_2(s):
    for r in par_id(s, 24).text_frame.paragraphs[0].runs:
        r.font.name, r.font.size, r.font.color.rgb = SANS, Pt(13), rgb(MUTED)


def diapo_3(s):
    garder(s, {2, 3, 4, 5, 31, 32})
    entete(s, 3, ["inventaire.py"])
    kpis(s, TOP, [
        ("96 310", DEEP, "enregistrements de 2 min, deux pistes chacun (micro 1, micro 2)"),
        ("3 207 h", DEEP, "d'audio inventorié"),
        ("4 sites", DEEP, "164 points d'écoute, 95 enregistreurs"),
        ("30 min", DEEP, "entre deux enregistrements, de 5 h à 19 h 30 ou de 6 h à 18 h 30 selon "
                         "l'enregistreur ; rien la nuit")])
    place(par_id(s, 31), L, 3.30, 6.90, 3.55)
    place(par_id(s, 32), R - 5.0, 3.30, 5.0, 3.55)


def diapo_4(s):
    garder(s, {2, 3, 4, 5, 7})
    entete(s, 4, ["inventaire.py"])
    place(par_id(s, 7), L, TOP, W, W * 545 / 1998)
    carte_texte(s, L, 5.28, W, [
        "⟦t|Phénologie 2023-2024 · ⟧Kaw, Molokoï et Trésor, enregistrés quasi sans interruption de "
        "novembre 2023 à novembre 2024. Seule source des saisons.",
        "⟦g|Relevés 2025-2026 · ⟧retour à Kaw (CDR, Patawa Ouest, Patawa Est) et à Trésor (RNRT), et "
        "un nouveau site, Mataroni ; une semaine chacun entre fin décembre et début mars, avec des "
        "dizaines de micros par relevé. Toute la saison des pluies, aucune saison sèche."],
        size=13, h=BOTTOM - 5.28, anchor="m")


TERMINAL = [
    "cd F:\\audio\\⟦y|R1_2025-12_CDR⟧\\2LA04185\\Data",
    "ls 2LA04185_20251225_050000.wav",
    "Répertoire : F:\\audio\\R1_2025-12_CDR\\2LA04185\\Data",
    " ",
    "Mode                 LastWriteTime         Length Name",
    "----                 -------------         ------ ----",
    "-a----        25/12/2025     05:02       23041188 ⟦y|2LA04185_20251225_050000.wav⟧",
    "uv run python C:\\Users\\ll34272\\Grenouille\\scripts\\verification\\entete_wav.py "
    "2LA04185_20251225_050000.wav",
    "[fmt ] ⟦y|2 pistes · 48000 Hz⟧",
    "[junk] 460 octets",
    "[data] ⟦y|5760000 échantillons par piste = 120.0 s⟧",
    "[guan] GUANO|Version: 1.0",
    "Firmware Version: 4.5",
    "Make: Wildlife Acoustics, Inc.",
    "Model: Song Meter Mini",
    "⟦y|Serial: 2LA04185⟧",
    "WA|Song Meter|Prefix: 2LA04185",
    'WA|Song Meter|Audio settings: [{"rate":48000,"gain":6},{"rate":48000,"gain":18}]',
    "Length: 120.000",
    "Original Filename: 2LA04185_20251225_050000.wav",
    "⟦y|Timestamp: 2025-12-25 05:00:00-3:00⟧",
    "Loc Position: 4.92289 -52.29129",
    "Temperature Int: 23.25",
    "Samplerate: 48000",
    "[wamd] 276 octets",
]


def diapo_5(s):
    garder(s, {2, 3, 4, 5})
    entete(s, 5, ["ingest.py", "inventaire.py"])
    wl = 6.20
    etiquette(s, L, TOP, wl, "LE FICHIER TEL QU'IL EST SUR LE DISQUE")
    y0, ht = 2.07, BOTTOM - 2.07
    forme(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, y0, wl, ht, fill="1E1E1E", adj=0.10 / ht)
    forme(s, MSO_SHAPE.ROUNDED_RECTANGLE, L, y0, wl, 0.32, fill="3A3A3A", adj=0.10 / 0.32)
    forme(s, MSO_SHAPE.RECTANGLE, L, y0 + 0.20, wl, 0.12, fill="3A3A3A")
    for k, c in enumerate(("FF5F57", "FEBC2E", "28C840")):
        forme(s, MSO_SHAPE.OVAL, L + 0.13 + k * 0.20, y0 + 0.095, 0.13, 0.13, fill=c)
    tb(s, L + 0.9, y0, wl - 1.8, 0.32, "2LA04185_20251225_050000.wav — lecture seule", size=10,
       color="D4D4D4", font=MONO, align="c", anchor="m")
    tb(s, L + 0.15, y0 + 0.42, wl - 0.3, ht - 0.5, TERMINAL, size=9.5, color="D4D4D4", font=MONO,
       after=0)

    xr, wr = L + wl + GAP, R - (L + wl + GAP)
    tb(s, xr, TOP, wr, 0.24, "UNE LIGNE DE LA TABLE ⟦mt|recordings⟧", size=11, color=TEAL, bold=True)
    champs = ["recording_id", "path", "dataset", "site", "mic_id", "start_utc", "duration_s",
              "sample_rate", "channels", "qc_flags"]
    valeurs = ["ebe1c334f31d98b5", "…/RELEVE 1 CDR …/2LA04185_20251225_050000.wav", "2026",
               "⟦mt|CDR⟧", "⟦mt|2LA04185⟧", "⟦mt|2025-12-25T08:00:00Z⟧", "⟦mt|120.0⟧",
               "⟦mt|48000⟧", "⟦mt|2⟧", "{…} rempli au contrôle qualité (diapo 12)"]
    intro = ["Base SQLite blanci.sqlite, un seul fichier sur le PC (hors dépôt git). Une ligne par "
             "enregistrement (diapo suivante)."]
    hi = hauteur(intro, wr - 0.4, 11)
    h1 = 0.13 + hi + 0.04 + 10 * 0.155 + 0.13
    carte(s, xr, 2.07, wr, h1)
    tb(s, xr + 0.20, 2.20, wr - 0.4, hi, intro, size=11)
    tb(s, xr + 0.20, 2.24 + hi, 1.25, 1.6, champs, size=9.5, color=MUTED, font=MONO, after=0)
    tb(s, xr + 1.45, 2.24 + hi, wr - 1.6, 1.6, valeurs, size=9.5, color=MUTED, font=MONO, after=0)

    y2 = 2.07 + h1 + 0.18
    etiquette(s, xr, y2, wr, "CE QUE L'INVENTAIRE EN TIRE")
    carte_texte(s, xr, y2 + 0.30, wr, [
        "⟦b|Le micro. ⟧Il vient du numéro de série du GUANO, pas du nom ni du dossier : le même "
        "enregistreur est rangé dans « 2LA04185 » (CDR) ou « 2LA04185_GI19 » (Mataroni), et 893 "
        "fichiers de CDR portent un préfixe réglé à la main (LAGOTHRIX, SAIMIRI, ALOUATTA).",
        "⟦b|L'heure. ⟧Elle vient du GUANO, qui porte le fuseau : 5 h à −3:00 donne 08:00 UTC. Le nom "
        "du fichier ne porte pas le fuseau.",
        "⟦b|Le site. ⟧Il vient du dossier (« RELEVE 1 CDR »). Rien d'autre ne le vérifie : la "
        "position GPS du GUANO, essayée, est inutilisable en l'état (diapo 22)."],
        size=11, bar=None, h=BOTTOM - y2 - 0.30)


def diapo_6(s):
    garder(s, {2, 3, 4, 5, 33})
    place(par_id(s, 2), L, 0.40, 2.03)
    place(par_id(s, 3), L, 0.40, 2.03)
    texte_forme(par_id(s, 4), "Coup d'œil sur la base de données : la table recordings, une ligne "
                              "par fichier")
    entete(s, 6, ["ingest.py", "inventaire.py"])
    carte(s, L, TOP, W, 0.50)
    tb(s, L + 0.20, TOP, W - 0.4, 0.50,
       "⟦t|Chemin · ⟧⟦mk|C:\\Users\\ll34272\\base_blanci\\blanci.sqlite⟧", size=13, anchor="m")
    tb(s, L + 0.20, TOP, W - 0.4, 0.50, "6,7 Go · hors dépôt git", size=12, color=MUTED, align="r",
       anchor="m")
    place(par_id(s, 33), L, 2.42, W, 3.75)
    tb(s, L, 6.30, W, 0.6, "La même base tient aussi les annotations (tables spans, intervals, "
       "labels), les fenêtres d'encodage (windows, 11,9 millions) et les scores des têtes (scores, "
       "27,3 millions). Les 96 349 lignes de recordings comptent 39 fichiers de test (2018, "
       "référence) en plus des 96 310 de l'inventaire.", size=12)


def diapo_7(s, fig):
    garder(s, {2, 3, 4, 5, 31})
    entete(s, 7, ["inventaire.py", "qc.py"])
    place(par_id(s, 31), L, TOP, 5.40, 5.40 * 670 / 1096)
    tb(s, L, TOP + 0.10, 5.40, 0.2, "inventaire.py", size=9.5, color=TEAL, font=MONO, align="r")
    wd = 5.60
    hd = wd * 644 / 1348
    xd = R - wd
    titre_fig(s, xd, TOP + 0.07, wd, "Observation : écart mesuré entre les deux pistes", tag="qc.py")
    image(s, fig["gains_mesure"], xd, 5.05 - hd, wd)
    xl = xd + wd * 600 / 1348
    yl = 5.05 - hd + hd * 12 / 644
    legende(s, xl, yl, [("sq", GREEN, "réglage −12 dB (3 410 enr.)"),
                        ("sq", ORANGE, "réglage +12 dB (197 enr.)"),
                        ("dash", MUTED, "tirets : écart réglé")], pas=0.23)
    y = 5.20
    carte(s, L, y, W, BOTTOM - y, bar=TEAL)
    wt = COL - 0.36
    tb(s, L + 0.26, y + 0.12, wt, 1.5, [
        "⟦t|Réglage · ⟧143 points ont la piste 0 à 6 dB et la piste 1 à 18 dB ; 21 ont l'inverse : "
        "18 enregistreurs 2MA de Patawa Est et de RNRT, et 3 SMA de RNRT. Le gain est lu dans "
        "l'en-tête GUANO.",
        "⟦t|Pourquoi ça compte · ⟧on encode la piste 0, pensée comme le micro à 6 dB. Pour 21 points, "
        "c'est le micro à 18 dB : plus de bruit de fond, plus de saturation. À décider : encoder "
        "leurs enregistrements sur l'autre piste ?"], size=11.5)
    tb(s, X2 + 0.10, y + 0.12, COL - 0.30, 1.5, [
        "⟦t|Mesure · ⟧un enregistrement par micro et par jour (3 466), plus 141 lus aux bascules "
        "d'état : 3 607 enregistrements, 164 points. Niveau RMS de chaque piste sur les 2 min "
        "entières, en dBFS ; écart = piste 0 − piste 1. Il vaut la différence des gains : deux "
        "pics, à −12 et +12 dB.",
        "⟦t|Pourquoi un par jour · ⟧cela suffit à voir un micro qui dérive, sans relire 3 207 h de "
        "son."], size=11.5)


def diapo_8(s):
    garder(s, {2, 3, 4, 5, 9})
    entete(s, 8, ["inventaire.py"])
    h = 4.08
    w = h * 1999 / 824
    place(par_id(s, 9), L + (W - w) / 2, 1.52, w, h)
    carte_texte(s, L, 5.68, W, [
        "⟦t|Tenu · ⟧sur 99,7 % des jours entre la pose et la relève, le micro a fait tous les "
        "enregistrements prévus : 30 par jour de 5 h à 19 h 30, ou 26 de 6 h à 18 h 30, selon "
        "l'enregistreur (diapo 3). C'est le cas de 3 128 des 3 137 jours-micros ; les 9 autres sont "
        "partiels.",
        "⟦t|Trois trous · ⟧Molokoï E, 54 jours (22/11/2023 – 14/01/2024) : SMA14157 s'arrête le 21/11 "
        "à midi, son dernier fichier est tronqué à 256 Kio, jamais refermé (alimentation coupée en "
        "pleine écriture) ; Kaw D, 5 jours au changement d'enregistreur (retrait le 28/08, reprise "
        "le 03/09/2024) ; RNRT, cinq micros muets le 10/02/2026. L'absence d'enregistrement n'est "
        "pas l'absence de chant : ces trous comptent dans une étude phénologique d'A. blanci."],
        size=11.5, h=BOTTOM - 5.68, after=3, pad=0.10)


def diapo_9(s):
    garder(s, {2, 3, 4, 5, 31, 33, 35, 37, 39})
    entete(s, 9, ["inventaire.py", "ingest.py"])
    place(par_id(s, 31), L, 1.62, W, 2.45)
    ws = (W - 3 * GAP) / 4
    hs = ws * 234 / 558
    noms = ["SMA14163 · fichier 10 h 30", "SMA14163 · fichier 11 h 00 (l'averse)",
            "SMA14636 · fichier « 10 h 30 » (en-tête : 11 h 30)",
            "SMA14636 · fichier « 11 h 00 » (en-tête : 12 h 00)"]
    for k, sid in enumerate((33, 35, 37, 39)):
        x = L + k * (ws + GAP)
        tb(s, x, 4.10, ws, 0.2, noms[k], size=10, bold=True)
        place(par_id(s, sid), x, 4.31, ws, hs)
    yc = 4.31 + hs + 0.03
    tb(s, L, yc, W, 0.2, "Spectrogrammes de 0 à 12 kHz, 2 min, piste 0 : l'averse du fichier de "
       "11 h 00 est la bande claire sur toute la hauteur, sur les deux micros.", size=10.5,
       color=MUTED)
    y = yc + 0.25
    carte(s, L, y, W, BOTTOM + 0.03 - y, bar=TEAL)
    tb(s, L + 0.26, y + 0.09, COL - 0.36, 1.1, [
        "⟦t|Le symptôme · ⟧SMA14636, du 20/02 au 11/04/2024 : l'en-tête GUANO dit une heure de plus "
        "que le nom du fichier.",
        "⟦t|Lequel croire ? · ⟧le son : une averse s'entend sur les deux micros, à 150 m l'un de "
        "l'autre, en même temps. Elle tombe à 11 h avec l'heure du nom, à 12 h avec celle de "
        "l'en-tête."], size=11, after=2)
    tb(s, X2 + 0.10, y + 0.09, COL - 0.30, 1.1, [
        "⟦t|Mesure · ⟧corrélation des niveaux RMS des deux micros sur les 1 521 enregistrements : "
        "0,57 à l'heure du nom, 0,34 à celle de l'en-tête.",
        "⟦t|Vérifié le 18/11 · ⟧9 459 en-têtes relus sur le disque : exactement ces 1 521 sont écrits "
        "en −4:00 (du 20/02 16 h 30 au 11/04 11 h 30).",
        "⟦t|Correction · ⟧ils prennent l'heure de leur nom, à chaque inventaire."], size=11, after=2)


def diapo_10(s):
    place(par_id(s, 5), 12.33, 7.08, 0.50, 0.25)
    for r in par_id(s, 13).text_frame.paragraphs[0].runs:
        r.font.name = SERIF


def diapo_11(s):
    garder(s, {2, 3, 4, 5})
    entete(s, 11, ["ingest.py", "qc.py", "inventaire.py"])
    colonnes = [
        ("SANS LIRE LE SON : MÉTADONNÉES ET DATES", [
            (TEAL, "Les drapeaux, et où on les lit", "diapo 12",
             "⟦t|Vise · ⟧ce que le code note sur un enregistrement",
             "⟦t|Lecture · ⟧quatre drapeaux calculés écartent ; 680 écartés sur 96 310"),
            (RED, "Hors relevé", "diapo 13", "⟦t|Vise · ⟧test d'avant la pose, restes de carte SD",
             "⟦r|Écarte · ⟧88 écartés (106 levés, 18 gardés après écoute)"),
            (RED, "Durée anormale", "diapo 13", "⟦t|Vise · ⟧fichier qui ne dure pas 120 s, à 1 s près",
             "⟦r|Écarte · ⟧172 écartés (173 signalés, 1 gardé : A. blanci y chante)"),
            (RED, "Doublons", "diapo 14",
             "⟦t|Vise · ⟧même nom et même contenu qu'un fichier déjà inventorié",
             "⟦r|Écarte · ⟧3 673 copies de CDR sur les cartes de Mataroni")]),
        ("EN LISANT LE SON, PUIS À L'ÉCOUTE", [
            (RED, "Silence et son étouffé", "diapos 15 à 18",
             "⟦t|Vise · ⟧niveau RMS sous −70 dB, ou moins de 5 % de l'énergie au-dessus de 2 kHz",
             "⟦r|Écarte · ⟧506 écartés"),
            (TEAL, "D'où viennent les exclusions", "diapo 19", "⟦t|Vise · ⟧par site, micro et date",
             "⟦t|Constat · ⟧CDR : 6 micros portent 452 des 680"),
            (GOLD, "Micro défaillant", "diapo 20", "⟦t|Vise · ⟧dérive de la qualité audio",
             "⟦g|Remarque · ⟧Kaw D : 1 871 enr. encodés sur une piste devenue sourde"),
            (GOLD, "Les autres critères", "diapos 21 et 22",
             "⟦t|Vise · ⟧bascule, clics, trous numériques, mauvais site…",
             "⟦g|Remarque · ⟧11 bascules à CDR ; le reste à décider")])]
    hc, pas = 1.06, 1.06 + 0.14
    for k, (titre, cartes) in enumerate(colonnes):
        x = L if k == 0 else X2
        etiquette(s, x, TOP, COL, titre)
        for j, (barre, nom, ou, vise, effet) in enumerate(cartes):
            y = TOP + 0.32 + j * pas
            carte(s, x, y, COL, hc, bar=barre)
            tb(s, x + 0.26, y, COL - 0.46, hc, [(nom, {"size": 13, "bold": True, "after": 3}),
                                                vise, effet], size=11.5, after=1, anchor="m")
            tb(s, x + 0.26, y + 0.14, COL - 0.46, 0.2, ou, size=10, color=MUTED, align="r")


def diapo_12(s):
    garder(s, {2, 3, 4, 5, 31, 32})
    entete(s, 12, ["qc.py"])
    table = par_id(s, 31)
    place(table, L, TOP - 0.03)
    for row in table.table.rows:
        row.height = Inches(0.27)
    etiquette(s, L, 5.07, W, "UNE VRAIE LIGNE DE LA BASE")
    place(par_id(s, 32), L, 5.33, 10.30, 10.30 * 310 / 2426)
    tb(s, L + 10.30 + 0.18, 5.36, W - 10.48, 1.4, [
        "Ligne lue avant le 18/11.",
        "Depuis, elle porte aussi ⟦m|\"keep\"⟧ : A. blanci y chante, elle est gardée."],
        size=10.5, color=MUTED, after=3)


def diapo_13(s):
    garder(s, {2, 3, 4, 5})
    entete(s, 13, ["qc.py", "inventaire.py"])
    blocs = [
        ("HORS RELEVÉ", "off_campaign", "106 signalés, dont 18 gardés après écoute", [
            "⟦t|Ce que c'est · ⟧un fichier isolé de la série de son micro : test d'avant la pose, ou "
            "reste d'une carte SD d'ailleurs.",
            "⟦t|Comment c'est calculé · ⟧séries triées par date, coupées à chaque trou de plus de 7 "
            "jours (seuil choisi à la main) ; le plus gros bloc est le relevé, les autres sont hors "
            "relevé.",
            "⟦t|Combien · ⟧106 enregistrements ; 86 des 88 écartés sont aussi de durée anormale : des "
            "tests.",
            "⟦g|Écarte à tort · ⟧un micro reposé après plus de 7 jours : 18 enregistrements de "
            "Mataroni (2LA03595, 19/05/2026), écoutés puis gardés.",
            "⟦t|Laisse passer · ⟧un test à moins de 7 jours de la pose : 25 enregistrements de 19 "
            "micros, les 30 et 31/12/2025 à Mataroni, écartés quand même par leur durée.",
            "⟦mub|→ voir Difficultés rencontrées⟧"]),
        ("DURÉE ANORMALE", "duration_off", "173 signalés, dont 1 gardé après écoute", [
            "⟦t|Ce que c'est · ⟧un fichier qui ne dure pas 120 s à 1 s près : test, déclenchement "
            "manuel, ou enregistrement du programme coupé.",
            "⟦t|Comment c'est calculé · ⟧durée lue dans l'en-tête du WAV, sans lire le son ; "
            "tolérance d'1 s choisie à la main.",
            "⟦t|Combien · ⟧125 de moins de 30 s (tests), 34 de 30 à 119 s, 14 de plus de 2 min "
            "(manuels, jusqu'à 14 min).",
            "⟦g|Écarte à tort · ⟧les 34 coupés, du vrai terrain. Mataroni SMA13417, 104 s, A. blanci y "
            "chante (BlanciNet 0,97, écouté le 12/11) : écarté par une confirmation en bloc, gardé "
            "depuis le 18/11.",
            "⟦t|Laisse passer · ⟧rien : un fichier de 120 s vide ou étouffé reste gardé (contrôles "
            "suivants).",
            "⟦mub|→ voir Difficultés rencontrées⟧"])]
    wt = COL - 0.46
    hc = max(hauteur(p, wt, 13.5, 8) for *_, p in blocs) + 1.15
    for k, (nom, drapeau, gros, paras) in enumerate(blocs):
        x = L if k == 0 else X2
        carte(s, x, TOP, COL, hc, bar=TEAL)
        tb(s, x + 0.26, TOP + 0.13, wt, 0.26, f"{nom}   ⟦ms|{drapeau}⟧", size=13, color=TEAL,
           bold=True)
        tb(s, x + 0.26, TOP + 0.44, wt, 0.42, gros, size=20, bold=True, anchor="m")
        tb(s, x + 0.26, TOP + 1.00, wt, hc - 1.1, paras, size=13.5, after=8)


def diapo_14(s):
    garder(s, {2, 3, 4, 5})
    entete(s, 14, ["ingest.py", "inventaire.py"])
    h = carte_texte(s, L, TOP, W, [
        ("3 673 DOUBLONS ÉCARTÉS : LES CARTES SD DE CDR RAPPORTÉES À MATARONI",
         {"size": 11, "bold": True, "color": RED, "after": 3}),
        "Une carte SD rapportée d'un relevé contient encore les fichiers du précédent : à "
        "l'inventaire de Mataroni, 3 673 fichiers de CDR. Les compter deux fois, sous deux sites, "
        "fausserait tout : le contrôle écarte la copie, et garde l'original."], size=13.5, bar=RED,
        pad=0.17)
    y = TOP + h + 0.30
    etiquette(s, L, y, W, "LA RÈGLE : LE NOM ET LE CONTENU DOIVENT CONCORDER")
    tests = [("1 · Le nom", "même nom de fichier"), ("2 · La longueur", "même nombre d'échantillons"),
             ("3 · La fréquence et les pistes", "même fréquence, même nombre de pistes"),
             ("4 · L'en-tête", "même numéro de série et même heure, quand les deux fichiers en ont"),
             ("5 · Le son", "première seconde décodée, comparée échantillon par échantillon")]
    w5 = (W - 4 * 0.12) / 5
    y += 0.30
    for k, (nom, quoi) in enumerate(tests):
        x = L + k * (w5 + 0.12)
        carte(s, x, y, w5, 1.25, fill=PALE2, bar=TEAL)
        tb(s, x + 0.22, y + 0.14, w5 - 0.34, 1.0,
           [(nom, {"size": 12.5, "bold": True, "color": TEAL, "after": 4}), quoi], size=12.5)
    y += 1.25 + 0.14
    tb(s, L, y, W, 0.25, "Si un seul test échoue : un conflit est signalé et demande une "
       "vérification manuelle (aucun conflit à ce jour).", size=13, bold=True)
    y += 0.55
    gauche = [("Combien", {"size": 14, "bold": True, "color": TEAL, "after": 4}),
              "3 673 copies, chacune avec son original dans le relevé de CDR. Les copies écartées "
              "ne sont plus que sur le disque d'origine (Seagate Basic).",
              "Aucun autre fichier de même nom en base, tous jeux confondus."]
    droite = [("Limite", {"size": 14, "bold": True, "color": GOLD, "after": 4}),
              "Un fichier renommé à la main n'est pas vu : la règle ne compare que des fichiers du "
              "même nom. Un test par hachage du contenu, qui ne dépend pas du nom, est déjà "
              "programmé : il reste à le lancer (57 lignes seulement ont une empreinte "
              "aujourd'hui)."]
    hc = max(hauteur(gauche, COL - 0.46, 13, 5), hauteur(droite, COL - 0.46, 13, 5)) + 0.34
    carte_texte(s, L, y, COL, gauche, size=13, bar=TEAL, h=hc, after=5, pad=0.17)
    carte_texte(s, X2, y, COL, droite, size=13, bar=GOLD, h=hc, after=5, pad=0.17)


def points_numerotes(s, x, w, items, y0=1.95, pas=1.20):
    for k, texte in enumerate(items):
        y = y0 + k * pas
        forme(s, MSO_SHAPE.OVAL, x, y, 0.40, 0.40, fill=TEAL)
        tb(s, x, y, 0.40, 0.40, str(k + 1), size=14, color=WHITE, bold=True, align="c", anchor="m")
        tb(s, x + 0.56, y - 0.02, w - 0.56, pas - 0.1, texte, size=12.5)


def diapo_15(s, fig):
    garder(s, {2, 3, 4, 5})
    entete(s, 15, ["qc.py"])
    wf = 6.05
    hf = wf * 603 / 1456
    y = 2.45
    titre_fig(s, L, y, wf, "Un même enregistrement, deux micros : l'un étouffé, l'autre non")
    image(s, fig["welch"], L, y + 0.36, wf)
    legende(s, L + wf * 172 / 1456, y + 0.36 + hf * 388 / 603, [
        ("line", GREEN, "piste 1 (micro 2) : hf_ratio = 0,785"),
        ("line", "C0492F", "piste 0 (micro 1) : hf_ratio = 0,003")], pas=0.22)
    tb(s, L, y + 0.36 + hf + 0.08, wf, 0.4, "CDR, SMA14636, 25/12/2025 19 h : même enregistrement, "
       "deux micros. Piste 0 étouffée, piste 1 saine.", size=10.5, color=MUTED)
    points_numerotes(s, 7.0, R - 7.0, [
        "⟦b|Un ratio · ⟧la part de l'énergie au-dessus de 2 kHz. Il ne dépend pas du gain : les deux "
        "pistes ont 6 et 18 dB respectivement, un seuil en dBFS ne serait pas le même sur l'une et "
        "l'autre.",
        "⟦b|Un spectre moyen (Welch) · ⟧on découpe les 2 min en ~5 600 segments recouvrants, on "
        "moyenne leurs spectres : la variance de l'estimation baisse d'autant.",
        "⟦b|Une coupure à 2 kHz · ⟧un obstacle coupe d'abord les aigus ; A. blanci chante vers "
        "4,5 kHz. Une piste qui ne rend rien au-dessus de 2 kHz ne peut pas l'entendre.",
        "⟦b|Sa limite · ⟧un défaut qui ajoute des aigus (crépitement, souffle) passe : un "
        "enregistrement jugé inutilisable à l'oreille avait 0,61."])


def diapo_16(s, fig):
    garder(s, {2, 3, 4, 5})
    entete(s, 16, ["qc.py"])
    hf = 4.75
    wf = hf * 1210 / 954
    titre_fig(s, L, TOP, wf, "Part de l'énergie au-dessus de 2 kHz, en % de l'énergie totale")
    image(s, fig["deux_pistes"], L, TOP + 0.36, wf)
    legende(s, L + wf * 172 / 1210, TOP + 0.36 + hf * 36 / 954, [
        ("dot", "D0705C", "deux pistes étouffées : écartés d'office (399)"),
        ("tri", "C0492F", "douteux, réécoutés puis écartés (103)"),
        ("star", TEAL, "douteux, réintégré à l'écoute (1)")], pas=0.23)
    points_numerotes(s, 7.0, R - 7.0, [
        "⟦b|Un indice par piste · ⟧part de l'énergie au-dessus de 2 kHz dans l'énergie totale, sur le "
        "spectre moyen du fichier. Un sac laisse passer les basses, pas les aigus.",
        "⟦b|Un seuil : 5 % · ⟧en dessous, la piste est candidate. Le premier seuil, 20 %, était trop "
        "haut : il prenait des orages et des chœurs.",
        "⟦b|La règle des suites · ⟧au moins 4 enregistrements consécutifs du même micro (2 h), sans "
        "trou de plus d'une heure. Un sac est continu ; un enregistrement grave isolé n'est pas un "
        "sac.",
        "⟦b|Les deux pistes · ⟧étouffées toutes les deux : écarté. L'autre au-dessus de 10 % (deux "
        "fois le seuil) : gardé, encodé sur elle. Entre 5 % et 10 % : douteux, réécouté."])


def diapo_17(s, fig):
    garder(s, {2, 3, 4, 5})
    entete(s, 17, ["qc.py"])
    wf = 7.75
    hf = wf * 640 / 1480
    y = 2.15
    titre_fig(s, L, y, wf, "Suites de pistes candidates : observées, et attendues si c'était du hasard")
    image(s, fig["suites"], L, y + 0.36, wf)
    legende(s, L + wf * 770 / 1480, y + 0.36 + hf * 30 / 640, [
        ("line", ORANGE, "attendu si indépendants (taux de chaque micro)"),
        ("sq", GREEN, "suites observées (piste 0)")], pas=0.23)
    xc = L + wf + 0.30
    carte_texte(s, xc, y + 0.36, R - xc, [
        ("⟦t|La preuve · ⟧trois fois moins de suites courtes que si les candidats tombaient au "
         "hasard. L'étouffement est un état qui dure : exiger une suite a donc du sens.",
         {"size": 15, "after": 10}),
        "⟦t|On pourrait faire mieux · ⟧modéliser la persistance : carte CUSUM (Page 1954)."],
        size=12.5, pad=0.18)


def diapo_18(s):
    garder(s, {2, 3, 4, 5})
    entete(s, 18, ["qc.py", "inventaire.py"])
    titre_fig(s, L, TOP, W, "Ce que deviennent les pistes sous le seuil de 5 %")
    legende(s, R - 3.3, TOP, [("sq", RED, "écartés du corpus"), ("sq", GREEN, "gardés")],
            horizontal=True, pas=0.26)
    rangs = [("deux pistes étouffées (suites sur chacune)", 399, 0),
             ("piste 0 étouffée, piste 1 douteuse (5 à 10 %)", 85, 0),
             ("piste 0 étouffée, piste 1 bonne (≥ 10 %) : encodé sur la piste 1", 0, 407),
             ("piste 1 étouffée, piste 0 non retenue (suites)", 8, 38),
             ("candidat isolé (sauvé par la règle des suites)", 5, 267)]
    x0, ech, y0, pas, hb = 5.35, 4.9 / 407, 2.22, 0.47, 0.30
    forme(s, MSO_SHAPE.RECTANGLE, x0 - 0.03, y0 - 0.06, 0.012, 5 * pas - 0.05, fill=SAGE)
    for k, (nom, ecartes, gardes) in enumerate(rangs):
        y = y0 + k * pas
        tb(s, L, y, x0 - L - 0.15, hb, nom, size=11.5, align="r", anchor="m")
        x = x0
        for n, c in ((ecartes, RED), (gardes, GREEN)):
            if n:
                forme(s, MSO_SHAPE.RECTANGLE, x, y, max(n * ech, 0.03), hb, fill=c)
                x += max(n * ech, 0.03) + 0.02
        tb(s, x + 0.10, y, 2.4, hb, f"{ecartes} écartés · {gardes} gardés", size=11.5, bold=True,
           anchor="m")
    y = y0 + 5 * pas + 0.14
    carte_texte(s, L, y, W, [
        "⟦t|Lecture · ⟧399 enregistrements ont leurs deux pistes étouffées ; 85 ont une piste 0 "
        "étouffée et une piste 1 douteuse, écartés après écoute ; 407 sont récupérés et encodés sur "
        "la piste 1.",
        "⟦t|Candidats isolés · ⟧272 (pistes 0 et 1) sont sauvés par la règle des suites : c'est parmi "
        "eux que se cachent les étouffés passés.",
        "⟦t|Les 8 « piste 1 étouffée, piste 0 non retenue » · ⟧en fait étouffés des deux côtés : à "
        "CDR, du 24 au 27/12, leur piste 0 est sous 5 % (5 cas) ou à peine au-dessus (3 cas), sauvée "
        "par la règle des suites. Repérés dans la file des douteux, écartés à l'écoute."],
        size=12.5, h=BOTTOM - y, anchor="m")


def diapo_19(s, fig):
    garder(s, {2, 3, 4, 5})
    entete(s, 19, ["qc.py", "inventaire.py"])
    wf = 7.70
    hf = wf * 724 / 1432
    titre_fig(s, L, TOP + 0.25, wf, "CDR, 20 au 28 décembre 2025 : un carré par enregistrement")
    image(s, fig["origine_cdr"], L, TOP + 0.61, wf)
    xc = L + wf + 0.30
    legende(s, xc, TOP + 0.25, [("sq", "C0492F", "écarté"),
                                ("sq", ORANGE, "piste 0 étouffée, encodé sur la piste 1"),
                                ("sq", SAGE, "sain")], size=12, pas=0.27)
    carte_texte(s, xc, TOP + 1.30, R - xc, [
        "⟦t|Presque tout vient de CDR · ⟧6 micros portent 452 des 680 écartés. 9 pistes sont "
        "étouffées dès la pose (21/12) ; 14 pistes de 10 enregistreurs le deviennent dans la nuit "
        "du 24 au 25 décembre, à quelques heures d'intervalle.",
        "⟦t|Question à l'équipe de terrain · ⟧forte pluie, eau, chute, enregistreurs qui prennent "
        "leurs congés du 25 ?"], size=13, after=8)


def diapo_20(s, fig):
    garder(s, {2, 3, 4, 5})
    entete(s, 20, ["inventaire.py", "qc.py"])
    hf = W * 729 / 2380
    y = 1.68
    titre_fig(s, L, y, 5.0, "164 points : un micro sourd se voit tout de suite", tag="inventaire.py")
    xd = L + W * 1012 / 2380
    titre_fig(s, xd, y, R - xd, "Phénologie 2023-2024 : l'écart dérive-t-il en un an ?", tag="qc.py")
    image(s, fig["micro_sourd"], L, y + 0.32, W)
    yl = y + 0.32 + hf * 12 / 729
    legende(s, L + W * 1770 / 2380, yl, [("line", "C0492F", "Kaw SMA14079"),
                                         ("line", ORANGE, "Molokoï SMA14157")], size=10, pas=0.18)
    legende(s, L + W * 2070 / 2380, yl, [("line", SAGE, "autres points 2023")], size=10, pas=0.18)
    yc = y + 0.32 + hf + 0.10
    carte_texte(s, L, yc, W, [
        "⟦t|Kaw D, SMA14079 · ⟧du 23 mai au 23 juillet 2024, la piste 0 tombe à −62 dBFS, 30 dB sous "
        "la piste 1 ; son ratio d'aigus reste normal, le contrôle n'a rien vu. 1 871 enregistrements "
        "encodés sur la piste sourde, dont 6 dans le jeu de test.",
        "⟦t|Aussi à plus de 6 dB du réglage · ⟧SMA14636 (micro 1 sourd, à CDR puis à Patawa Ouest), "
        "Molokoï SMA14157, RNRT 2MA08067, Mataroni 2LA05191 : cinq points, lus sur l'échantillon "
        "des pistes (3 607 fichiers). À décider : encoder sur la piste saine, ou écarter."],
        size=12, h=BOTTOM + 0.03 - yc, after=4, pad=0.10, anchor="m")


def diapo_21(s, fig):
    garder(s, {2, 3, 4, 5})
    entete(s, 21, ["qc.py", "inventaire.py"])
    kpis(s, TOP, [
        ("95 630", TEAL, "encodables après contrôle qualité"),
        ("680", RED, "écartés : 506 étouffés, 172 durée anormale, 2 hors relevé"),
        ("407", TEAL, "gardés, encodés sur la piste 1 : micro 1 étouffé, micro 2 bon (compris dans "
                      "les 95 630)"),
        ("≈ 1 %", GOLD, "d'étouffés passés, mesuré sur 674 enregistrements tirés au hasard "
                        "(0,3 – 1,9 %)")])
    hf = 3.10
    wf = hf * 2426 / 700
    image(s, fig["recap"], L + (W - wf) / 2, 3.17, wf)
    y = 6.36
    carte(s, L, y, W, BOTTOM - y, bar=TEAL)
    tb(s, L + 0.26, y, COL - 0.36, BOTTOM - y, "⟦t|Gardés exprès, comme remarques · ⟧pluie, "
       "saturation : ils font partie du milieu.", size=12, anchor="m")
    tb(s, X2 + 0.10, y, COL - 0.30, BOTTOM - y, "⟦g|Essayé, inutilisable en l'état · ⟧vérifier le "
       "site par le GPS de l'en-tête (diapo 22).", size=12, anchor="m")


def grille(s, cartes, y, size, titre, after_titre=3):
    """Cartes sur deux colonnes, même hauteur par rangée. Renvoie le bas de la grille."""
    wt = COL - 0.46
    for k in range(0, len(cartes), 2):
        rang = cartes[k:k + 2]
        paras = [[(t, {"size": titre, "bold": True, "after": after_titre}), *corps]
                 for _, t, corps in rang]
        h = max(hauteur(p, wt, size, 4) for p in paras) + 0.26
        for j, ((barre, _, _), p) in enumerate(zip(rang, paras, strict=True)):
            carte_texte(s, L if j == 0 else X2, y, COL, p, size=size, bar=barre, h=h, after=4)
        y += h + 0.14
    return y - 0.14


def diapo_22(s):
    garder(s, {2, 3, 4, 5})
    texte_forme(par_id(s, 4), "Les limites du contrôle qualité : réglages à la main, défauts qu'il "
                              "ne voit pas encore")
    entete(s, 22, ["qc.py", "inventaire.py"])
    bas = grille(s, [
        (GOLD, "Réglages faits à la main : 5 %, 4 de suite, 7 jours", [
            "Seuil de 5 % calé sur 8 enregistrements ; « 4 de suite » : aucune loi ; blocs de 7 "
            "jours pour le hors relevé, faute de dates de pose.",
            "Résultat mesuré : environ 1 % d'étouffés passés, sur 674 enregistrements tirés au "
            "hasard."]),
        (GOLD, "Ce que l'indice ne voit pas", [
            "L'indice mesure la part d'énergie au-dessus de 2 kHz : un défaut qui ajoute des aigus "
            "lui échappe.",
            "Crépitement, souffle : un fichier inutilisable à l'oreille avait 61 % de son énergie "
            "au-dessus de 2 kHz."]),
        (GOLD, "Micro sourd : mesuré, pas encore dans le contrôle", [
            "Un micro perd 30 dB sans changer la part d'aigus : on ne le voit qu'à l'écart entre "
            "les deux pistes.",
            "Kaw D : 1 871 enregistrements encodés sur la piste sourde. À décider."]),
        (GOLD, "Bascule en plein enregistrement", [
            "Une piste passe de saine à étouffée (ou l'inverse) au milieu d'un fichier de 2 minutes.",
            "11 cas, tous à CDR, mesurés sur un échantillon. Pas encore écoutés."]),
        (GOLD, "Clics, parasites, trous numériques", [
            "Clics et parasites : l'indice réagit aux insectes, donc aucun critère n'est posé.",
            "Trous numériques (longue suite d'échantillons identiques) : aucun trouvé, mais mesuré "
            "sur 3 607 enregistrements seulement, pas sur tout le corpus."]),
        (GOLD, "Mauvais site, doublons renommés : pas encore contrôlés", [
            "Site : la position GPS de l'en-tête, essayée le 18/11, est inutilisable en l'état "
            "(longitude sans signe sur les 2MA, positions autour de Cayenne en 2026, stations "
            "confondues en 2023).",
            "Doublons renommés à la main : le hachage du contenu existe, il reste à le lancer."])],
        TOP, 11.5, 13)
    y = bas + 0.14
    carte_texte(s, L, y, W, [
        "⟦t|Déjà corrigé · ⟧l'heure de Molokoï (1 521 enregistrements) ; les 6 étouffés passés trouvés "
        "sur le training set, écartés ; le 18/11, un enregistrement où A. blanci chante, écarté à "
        "tort à Mataroni (gardé depuis), et un drapeau hors relevé périmé (Kaw 2023)."],
        size=12.5, fill=PALE2, h=max(BOTTOM - y, 0.6), anchor="m")


def diapo_23(s):
    garder(s, {2, 3, 4, 5})
    entete(s, 23)
    bas = grille(s, [
        (RED, "Ce qui s'est passé", [
            "Annotation et encodage lancés avant un inventaire et un contrôle qualité sûrs.",
            "Je n'ai pas attendu que l'étape précédente soit vérifiée pour passer à la suivante."]),
        (RED, "Conséquences", [
            "4 étouffés dans le jeu de test, 2 dans le lot 1.",
            "68 extraits écoutés sur une piste et encodés sur l'autre, dont 15 à A. blanci : le "
            "label décrit un son que l'encodeur n'a pas eu.",
            "L'heure fausse de Molokoï est entrée dans un tirage par tranche horaire.",
            "Une écoute écrasée par une confirmation en bloc (Mataroni, corrigé le 18/11)."]),
        (GOLD, "Ce qui manquait", [
            "Un suivi de chaque micro dans le temps.",
            "Kaw D est resté sourd deux mois sans que le contrôle le voie : 1 871 enregistrements "
            "encodés sur la piste sourde.",
            "L'horloge de Molokoï : SMA14636 avait une heure de trop dans l'en-tête ; on ne l'a vu "
            "qu'en comparant avec le micro voisin (corrélation 0,57 contre 0,34). Corrigé : 1 521 "
            "enregistrements prennent l'heure de leur nom."]),
        (TEAL, "Ce qui a sauvé la mise", [
            "L'audio n'est jamais touché : les drapeaux ne modifient aucun fichier.",
            "Les mesures sont gardées avec les drapeaux : tout se recalcule sans relire le son."])],
        TOP, 12.5, 14, after_titre=4)
    y = bas + 0.16
    carte_texte(s, L, y, W, [
        "⟦t|La règle désormais · ⟧une étape ne démarre que quand la précédente est vérifiée : chaque "
        "choix avec son effectif, sa justification, ce qu'il laisse passer, mesuré sur un tirage au "
        "hasard. Les prochaines présentations (annotation, encodeurs, têtes) suivront ce gabarit."],
        size=13, fill=PALE2, h=max(BOTTOM - y, 0.8), anchor="m")


def notes(prs):
    d = prs.slides
    note(d[3], "ce sont 25 enregistrements des 30 et 31 décembre, de 12 micros, rangés",
         "ce sont 26 enregistrements de 20 micros, les 30 et 31 décembre (dont 25, de 19 micros, à "
         "moins de sept jours de la pose), rangés")
    note(d[3], "Je dois encore vérifier ce que c'est : des tests, ou des fichiers restés sur les "
         "cartes.", "Tous sont écartés pour leur durée : ce sont des tests.")
    note(d[6], "soit 3 607 enregistrements pour 164 points",
         "soit 3 466 enregistrements, plus 141 lus aux bascules d'état : 3 607 pour 164 points")
    note(d[6], "sont ceux de la dernière exécution du notebook 05 avec les disques ; ils n'ont pas "
         "pu être recomptés ce jour (le gain est dans les WAV et le fichier echantillon_pistes.csv "
         "est sur le disque F: absent)",
         "ont été recomptés le 18/11 depuis echantillon_pistes.csv, par inventaire.py")
    note(d[11], "dans le même explorateur que la diapo 5 bis", "dans le même explorateur que la "
         "diapo 6")
    note(d[11], "est montré plus loin (diapo 12)", "est montré plus loin. Depuis le 18/11, cette "
         "ligne porte aussi keep : A. blanci y chante, la confirmation en bloc l'avait écartée à "
         "tort, elle est gardée")
    note(d[17], "Une quarantaine ont seulement la piste 1 étouffée : gardés sur la piste 0.",
         "46 ont seulement la piste 1 étouffée : 38 gardés sur la piste 0, 8 écartés.")


def main() -> None:
    prs = Presentation(str(V3))
    fig = figures(prs)
    d = prs.slides
    diapo_1(d[0]), diapo_2(d[1]), diapo_3(d[2]), diapo_4(d[3]), diapo_5(d[4]), diapo_6(d[5])
    diapo_7(d[6], fig), diapo_8(d[7]), diapo_9(d[8]), diapo_10(d[9]), diapo_11(d[10])
    diapo_12(d[11]), diapo_13(d[12]), diapo_14(d[13]), diapo_15(d[14], fig), diapo_16(d[15], fig)
    diapo_17(d[16], fig), diapo_18(d[17]), diapo_19(d[18], fig), diapo_20(d[19], fig)
    diapo_21(d[20], fig), diapo_22(d[21]), diapo_23(d[22])
    notes(prs)
    prs.save(str(V4))
    print(f"écrit : {V4.name} ({len(d)} diapos)")


if __name__ == "__main__":
    main()
