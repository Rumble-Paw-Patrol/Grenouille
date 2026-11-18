# ruff: noqa: E501
"""Paquet 2 (diapos 6, 7, 9, 9 bis, 9 ter, 11, 10, 8) de la présentation n° 4 : corrections demandées par
Léonard le 18/11/2026, appliquées au XML du paquet découpé (bibliothèque standard seulement ; Pillow, s'il
est là, sert à mesurer le texte). Les formes non touchées (dont ses retouches à la main) sont recopiées
telles quelles. Même fonctionnement que `presentation_4_paquet_1.py`, dont les aides sont reprises.

    python documentation/prez/scripts/presentation_4_paquet_2.py PAQUET_DECOUPE.pptx SORTIE.pptx \
        documentation/prez/figures/inventaire_qc

Ordre cible dans le .pptx : 6, 7, 9, 9 bis (changement de partie), 9 ter (sommaire du contrôle
qualité), 11, 10, 8. Ne jamais écrire SORTIE par-dessus un fichier que Léonard a modifié depuis :
relancer sur le paquet découpé écraserait ses retouches. Écrit aussi les sections 6 à 11 du script
des notes orateur (`presentation_4_inventaire_qc_v2_script.md`) si l'option `--script` est donnée."""

import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

SRC, DST, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
SCRIPT_MD = sys.argv[sys.argv.index("--script") + 1] if "--script" in sys.argv else None
EMU = 914400
INK, MUTED, TEAL, GOLD, PALE, PALER, SAGE = "1B2A24", "5B6B63", "2A7F72", "E9B44C", "E4EEE8", "F3F7F4", "A7BFB2"
GOLD_DARK, RED, DARK, DARK2, WHITE = "9A6B0F", "C8553D", "0E2A24", "173D34", "FFFFFF"
MONO = "Consolas"

try:
    from PIL import ImageFont
    _POLICES = {}

    def _largeur(t, sz, bold=False, font="Calibri"):
        fichier = {("Calibri", False): "calibri.ttf", ("Calibri", True): "calibrib.ttf",
                   ("Consolas", False): "consola.ttf", ("Consolas", True): "consolab.ttf"}[(font, bold)]
        k = (fichier, sz)
        if k not in _POLICES:
            _POLICES[k] = ImageFont.truetype(f"C:/Windows/Fonts/{fichier}", int(sz * 10))
        return _POLICES[k].getlength(t) / 10 / 72  # pouces
except Exception:  # pas de Pillow ou pas de polices : pas de mesure
    _largeur = None


def e(v):
    return int(round(v * EMU))


def rpr(sz, color, bold=False, font="Calibri", italic=False):
    return (f'<a:rPr lang="fr-FR" sz="{int(sz * 100)}" b="{int(bold)}" i="{int(italic)}" dirty="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{font}" pitchFamily="34" charset="0"/>'
            f'<a:ea typeface="{font}" pitchFamily="34" charset="-122"/>'
            f'<a:cs typeface="{font}" pitchFamily="34" charset="-120"/></a:rPr>')


def para(runs, algn="l", after=0, line=None):
    """runs : liste de (texte, sz, couleur, gras[, police])."""
    sp = f'<a:lnSpc><a:spcPct val="{int(line * 1000)}"/></a:lnSpc>' if line else ""
    out = f'<a:p><a:pPr marL="0" indent="0" algn="{algn}">{sp}<a:spcAft><a:spcPts val="{int(after * 100)}"/></a:spcAft><a:buNone/></a:pPr>'
    for r in runs:
        t, sz, c, b, *f = r
        out += f'<a:r>{rpr(sz, c, b, f[0] if f else "Calibri")}<a:t>{escape(t)}</a:t></a:r>'
    sz = runs[0][1] if runs else 10
    return out + f'<a:endParaRPr lang="fr-FR" sz="{int(sz * 100)}" dirty="0"/></a:p>'


class Ids:
    def __init__(self, start):
        self.n = start

    def __call__(self):
        self.n += 1
        return self.n


def rect(i, x, y, w, h, fill, line=None, radius=None, name="Shape"):
    geom = (f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val {radius}"/></a:avLst></a:prstGeom>'
            if radius else '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>')
    ln = (f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
          if line else "<a:ln><a:noFill/></a:ln>")
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{name} {i}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(w)}" cy="{e(h)}"/></a:xfrm>{geom}'
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{ln}</p:spPr>'
            f'<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="fr-FR"/></a:p></p:txBody></p:sp>')


def text(i, x, y, w, h, paras, anchor="t", name="Text"):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{name} {i}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(w)}" cy="{e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="{anchor}"/>'
            f'<a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')


def pic(i, rid, x, y, w, h, descr):
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name="Image {i}" descr="{escape(descr)}"/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(w)}" cy="{e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')


def shapes(xml):
    """Formes de premier niveau de l'arbre, avec leur id."""
    corps = xml[xml.index("</p:grpSpPr>") + len("</p:grpSpPr>"):xml.index("</p:spTree>")]
    out = re.findall(r"<p:(?:sp|pic|graphicFrame|cxnSp)>.*?</p:(?:sp|pic|graphicFrame|cxnSp)>", corps, flags=re.S)
    assert "".join(out) == corps, "forme imbriquée non prévue"
    return [(int(re.search(r'<p:cNvPr id="(\d+)"', s).group(1)), s) for s in out]


def remplacer_arbre(xml, formes):
    a = xml.index("</p:grpSpPr>") + len("</p:grpSpPr>")
    b = xml.index("</p:spTree>")
    return xml[:a] + "".join(formes) + xml[b:]


def remplacer_texte(forme, ancien, nouveau):
    assert f"<a:t>{escape(ancien)}</a:t>" in forme, ancien
    return forme.replace(f"<a:t>{escape(ancien)}</a:t>", f"<a:t>{escape(nouveau)}</a:t>")


def notes(xml, texte):
    a = xml.index('name="Notes Placeholder')
    b0 = xml.index("<p:txBody>", a)
    b1 = xml.index("</p:txBody>", b0)
    body = ('<p:txBody><a:bodyPr/><a:lstStyle/>'
            + "".join(f'<a:p><a:r><a:rPr lang="fr-FR" dirty="0"/><a:t>{escape(p)}</a:t></a:r></a:p>'
                      for p in texte.split("\n\n")) + "</p:txBody>")
    return xml[:b0] + body + xml[b1 + len("</p:txBody>"):]


def ajouter_rel(rels, rid, cible):
    return rels.replace("</Relationships>",
                        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{cible}"/></Relationships>')


def retirer_rel(rels, rid):
    return re.sub(rf'<Relationship Id="{rid}"[^>]*/>', "", rels)


# ------------------------------------------------------------------ mesure du texte (aperçu approximatif)
ALERTES = []


def hauteur(pps, w):
    """Hauteur en pouces d'une liste de paragraphes (runs, après) dans une boîte de largeur w, en simulant
    le retour à la ligne avec les polices Calibri / Consolas de Windows. None sans Pillow."""
    if _largeur is None:
        return None
    total = 0.0
    for runs, after, line in pps:
        mots = []  # (mot, largeur, taille)
        for t, sz, _c, b, *f in runs:
            font = f[0] if f else "Calibri"
            for m in re.findall(r"\S+\s*", t):
                mots.append((m, _largeur(m, sz, b, font), sz))
        lignes, cur, big = 1, 0.0, max((m[2] for m in mots), default=10)
        for m, lw, _sz in mots:
            if cur + _largeur(m.rstrip(), _sz, False) > w and cur > 0:
                lignes += 1
                cur = 0.0
            cur += lw
        total += lignes * big * 1.22 * (line or 100) / 100 / 72 + after / 72
    return total


class PP:
    """Un paragraphe : son XML et de quoi le mesurer."""

    def __init__(self, runs, algn="l", after=0, line=None):
        self.runs, self.after, self.line = runs, after, line
        self.xml = para(runs, algn, after, line)


def T(i, x, y, w, h, pps, anchor="t", name="Text", nom=""):
    """Zone de texte ; signale (ALERTES) un texte qui dépasserait sa boîte."""
    mesure = hauteur([(p.runs, p.after, p.line) for p in pps], w)
    if mesure is not None and mesure > h + 0.02:
        ALERTES.append(f"{nom or name} {i} : texte {mesure:.2f} po pour une boîte de {h:.2f} po")
    return text(i, x, y, w, h, [p.xml for p in pps], anchor, name)


def carte(ids, x, y, w, h, fond=PALER, barre=TEAL):
    return [rect(ids(), x, y, w, h, fond, radius=5000), rect(ids(), x, y + 0.08, 0.07, h - 0.16, barre)]


def pastille(f2, f3, libelle, large):
    """Reprend la pastille (formes 2 et 3) d'une diapo avec un autre libellé et une autre largeur."""
    r = f2.replace(re.search(r'<a:ext cx="\d+"', f2).group(0), f'<a:ext cx="{e(large)}"')
    t = f3.replace(re.search(r'<a:ext cx="\d+"', f3).group(0), f'<a:ext cx="{e(large)}"')
    ancien = re.search(r"<a:t>([^<]*)</a:t>", t).group(1)
    return r, t.replace(f"<a:t>{ancien}</a:t>", f"<a:t>{escape(libelle)}</a:t>")


def numero(f5, libelle):
    ancien = re.search(r"<a:t>([^<]*)</a:t>", f5).group(1)
    f = f5.replace(f"<a:t>{ancien}</a:t>", f"<a:t>{libelle}</a:t>")
    if " " in libelle:  # « 9 bis » : une boîte plus large, alignée à droite
        f = f.replace('<a:off x="11277295" y="6473952"/><a:ext cx="457200"', '<a:off x="10820095" y="6473952"/><a:ext cx="914400"')
    return f


# ------------------------------------------------------------------ tableau (diapo 11)
BORD = "".join(f'<a:{c} w="6350" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:srgbClr val="CFDDD5"/></a:solidFill>'
               f'<a:prstDash val="solid"/></a:{c}>' for c in ("lnL", "lnR", "lnT", "lnB"))


def cellule(runs, fond):
    return (f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{para(runs)}</a:txBody>'
            f'<a:tcPr marL="64008" marR="64008" marT="27432" marB="27432" anchor="ctr">{BORD}'
            f'<a:solidFill><a:srgbClr val="{fond}"/></a:solidFill></a:tcPr></a:tc>')


def tableau(i, x, y, largeurs, lignes, hauteurs):
    grille = "".join(f'<a:gridCol w="{e(w)}"/>' for w in largeurs)
    trs = "".join(f'<a:tr h="{e(h)}">{"".join(cs)}</a:tr>' for cs, h in zip(lignes, hauteurs, strict=True))
    return (f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{i}" name="Table {i}"/><p:cNvGraphicFramePr>'
            f'<a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>'
            f'<p:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(sum(largeurs))}" cy="{e(sum(hauteurs))}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl><a:tblPr/>'
            f'<a:tblGrid>{grille}</a:tblGrid>{trs}</a:tbl></a:graphicData></a:graphic></p:graphicFrame>')


NOTES = {}
TITRES = {}
z = zipfile.ZipFile(SRC)
parts = {n: z.read(n) for n in z.namelist()}
infos = {n: z.getinfo(n) for n in z.namelist()}
S = lambda n: parts[f"ppt/slides/slide{n}.xml"].decode("utf-8")  # noqa: E731
R = lambda n: parts[f"ppt/slides/_rels/slide{n}.xml.rels"].decode("utf-8")  # noqa: E731
CQ = 2.77  # largeur de la pastille « CONTRÔLE QUALITÉ · k/n »

# ================================================================= diapo 6 : les deux gains
s6 = S(6)
f6 = dict(shapes(s6))
f6[3] = remplacer_texte(f6[3], "INVENTAIRE · 4/7", "INVENTAIRE · 4/7")
ids = Ids(30)
garde = [f6[k] for k in (2, 3, 4, 5)]
r6 = ajouter_rel(ajouter_rel(retirer_rel(R(6), "rId3"), "rId10", "../media/inv_gains_reglage.png"), "rId11", "../media/inv_gains_mesure.png")
garde.append(pic(ids(), "rId10", 0.6, 1.75, 5.48, 3.35, "Gain réglé : 143 points à 6 / 18 dB, 21 points à 18 / 6 dB"))
garde.append(pic(ids(), "rId11", 6.28, 1.75, 3.35 * 1348 / 700, 3.35, "Écart de niveau mesuré entre les deux pistes : deux pics, à −12 et +12 dB"))
Y, H = 5.25, 1.65
garde += carte(ids, 0.6, Y, 12.13, H)
col1 = [PP([("Réglage · ", 11.5, TEAL, True),
            ("143 points ont la piste 0 à 6 dB et la piste 1 à 18 dB ; 21 ont l'inverse : 18 enregistreurs 2MA de Patawa Est et de RNRT, et 3 SMA de RNRT. Le gain est lu dans l'en-tête GUANO : il se règle, il ne se mesure pas.", 11.5, INK, False)], after=5),
        PP([("Mesure · ", 11.5, TEAL, True),
            ("un enregistrement par micro et par jour (3 607 enregistrements, 164 points). Niveau RMS de chaque piste sur les 2 min entières, en dBFS (racine de la moyenne des carrés, pas une moyenne de dB) ; écart = piste 0 − piste 1. Il vaut la différence des gains : deux pics, à −12 et +12 dB.", 11.5, INK, False)])]
col2 = [PP([("Pourquoi un par jour · ", 11.5, TEAL, True),
            ("l'écart ne dépend que des réglages : un échantillon par jour suffit à voir un micro qui dérive, sans relire 3 207 h de son. ", 11.5, INK, False),
            ("Limite · ", 11.5, GOLD_DARK, True),
            ("un chœur proche d'un seul micro décale l'écart de quelques dB : on lit la forme (deux pics), pas chaque valeur.", 11.5, INK, False)], after=5),
        PP([("Pourquoi ça compte · ", 11.5, TEAL, True),
            ("on encode la piste 0, pensée comme le micro à 6 dB. Pour 21 points, c'est le micro à 18 dB : plus de bruit de fond, plus de saturation. À décider : encoder leurs enregistrements sur l'autre piste ?", 11.5, INK, False)])]
garde.append(T(ids(), 0.9, Y + 0.12, 5.6, H - 0.2, col1, nom="diapo 6, colonne 1"))
garde.append(T(ids(), 6.75, Y + 0.12, 5.8, H - 0.2, col2, nom="diapo 6, colonne 2"))
parts["ppt/slides/slide6.xml"] = remplacer_arbre(s6, garde).encode("utf-8")
parts["ppt/slides/_rels/slide6.xml.rels"] = r6.encode("utf-8")
TITRES[6] = "Deux gains par enregistreur : 21 points ont la piste 0 au gain fort"
NOTES[6] = ("Les deux pistes. Chaque enregistreur a deux micros, et chaque micro son gain. Le gain ne se mesure pas, il se règle : c'est écrit dans l'en-tête GUANO, et il ne prend que deux valeurs, 6 ou 18 dB. À gauche, ce qu'on lit : 143 points ont la piste 0 à 6 dB et la piste 1 à 18 dB ; 21 points ont l'inverse : 18 enregistreurs 2MA de Patawa Est et de RNRT, et 3 SMA de RNRT (SMA13417, SMA13599, SMA14801). La barre orange, celle des 21, a la même couleur que le pic orange du graphique de droite : à gauche le réglage, à droite ce qu'on observe.\n\n"
            "Comment on a mesuré l'écart, puisque la question est posée : on ne fait pas une moyenne de décibels. Pour chaque piste d'un enregistrement, on calcule le niveau RMS sur les 2 minutes entières, c'est-à-dire la racine de la moyenne des carrés des échantillons, après retrait de la composante continue, puis on le convertit en dBFS : 20 fois le logarithme du rapport à la pleine échelle. L'écart est le niveau de la piste 0 moins celui de la piste 1. Si les deux micros entendent la même scène, il vaut la différence des gains : moins 12 dB pour un réglage 6 / 18, plus 12 pour 18 / 6, à quelques dB près selon le contenu. C'est ce qu'on voit : deux pics, centrés sur les écarts réglés.\n\n"
            "Effectif et choix : un enregistrement par micro et par jour, le plus proche de 10 h parmi ceux de durée normale, soit 3 607 enregistrements pour 164 points (scripts/controle_qualite/echantillon_pistes.py). Un par jour parce que l'écart ne dépend que des réglages : lire les 96 000 fichiers pour le même renseignement coûterait des jours de lecture de disque, et un point par jour suffit à voir une dérive (Turgeon et al. 2017 : un micro qui se dégrade sans être étouffé). Limite : un chœur proche d'un seul micro, ou de la pluie sur une seule bonnette, décale l'écart de quelques dB ; c'est pourquoi on lit la forme de l'histogramme, pas chaque valeur. Les enregistrements loin de leur pic sont ceux où un micro n'entend pas comme l'autre.\n\n"
            "Pourquoi ça compte : on encode la piste 0, en pensant que c'est le micro à 6 dB. Pour 21 points, c'est le micro à 18 dB : plus de bruit de fond, plus de saturation. À décider : encoder leurs enregistrements sur l'autre piste, c'est-à-dire choisir la piste par son gain et non par son numéro. Cet écart va nous servir plus loin, pour repérer un micro qui devient sourd.\n\n"
            "État des chiffres au 18/11/2026 : 143 et 21 points, 3 607 enregistrements et 164 points sont ceux de la dernière exécution du notebook 05 avec les disques ; ils n'ont pas pu être recomptés ce jour (le gain est dans les WAV et le fichier echantillon_pistes.csv est sur le disque F: absent). Le nombre de points (164) est bien celui de la base aujourd'hui. L'histogramme de droite est l'image du notebook, découpée ; elle se refait avec fig_gains() de presentation_4_figures.py dès que le CSV est là.")

# ================================================================= diapo 7 : complétude
s7 = S(7)
f7 = dict(shapes(s7))
trous = "".join(re.findall(r"<a:t>([^<]*)</a:t>", f7[8]))
assert "Trois trous · " in trous and "il faut prendre ces trous en compte dans une étude phénologique de Blanci." in trous
ids = Ids(30)
garde = [f7[k] for k in (2, 3, 4, 5)]
r7 = ajouter_rel(retirer_rel(R(7), "rId3"), "rId10", "../media/inv_completude_p2.png")
garde.append(pic(ids(), "rId10", 0.6, 1.72, 12.13, 3.2, "Complétude : 2023-2024 par station, 2025-2026 par site, un carré par jour"))
Y, H = 5.05, 1.85
garde += carte(ids, 0.6, Y, 12.13, H)
pps7 = [PP([("Tenu · ", 12, TEAL, True),
            ("sur 99,7 % des jours entre la pose et la relève, le micro a fait tous les enregistrements prévus pour la journée : 30 par jour de 5 h à 19 h 30, ou 26 de 6 h à 18 h 30, selon l'enregistreur (diapo 3). C'est le cas de 3 128 des 3 137 jours-micros ; les 9 autres sont partiels.", 12, INK, False)], after=5),
        PP([("Trois trous · ", 12, TEAL, True),
            ("Molokoï E, 54 jours sans rien (22/11/2023 – 14/01/2024 : micro arrêté au bout d'une semaine) ; Kaw D, 5 jours au changement d'enregistreur ; RNRT, cinq micros muets le 10/02/2026. L'absence d'enregistrement n'est pas l'absence de chant, il faut prendre ces trous en compte dans une étude phénologique de Blanci.", 12, INK, False)], after=5),
        PP([("Méthode · ", 12, GOLD_DARK, True),
            ("le programme attendu est lu dans les données, micro par micro : les créneaux présents sur la moitié au moins de ses jours (ni pose ni relève, partiels par construction). Un jour sans aucun enregistrement n'entre pas dans le 99,7 % : c'est un trou, en rouge. Limite : un micro qui n'a jamais fait son programme complet passerait pour tenu.", 12, INK, False)])]
garde.append(T(ids(), 0.9, Y + 0.12, 11.6, H - 0.2, pps7, nom="diapo 7"))
parts["ppt/slides/slide7.xml"] = remplacer_arbre(s7, garde).encode("utf-8")
parts["ppt/slides/_rels/slide7.xml.rels"] = r7.encode("utf-8")
TITRES[7] = "Complétude : le programme est tenu à 99,7 %, avec trois trous réels"
NOTES[7] = ("La complétude. Pour chaque micro, on lit son programme dans les données : 30 créneaux par jour de 5 h à 19 h 30, ou 26 de 6 h à 18 h 30 selon l'enregistreur (diapo 3). Puis on compte, jour par jour, ce qu'on a reçu. Sur 99,7 % des jours entre la pose et la relève, le micro a fait tous les enregistrements prévus : 3 128 jours-micros sur 3 137 (2023-2024 : 99,8 % sur 2 217 ; 2025-2026 : 99,5 % sur 920). Les 9 autres sont des jours partiels.\n\n"
            "À gauche, la phénologie, par station : un carré par jour, vert foncé quand tout le programme est reçu, plus clair quand il en manque, rouge quand il n'y a rien. À droite, les relevés de 2025-2026, tracés de la même façon : une ligne par site, un bloc par jour, la couleur est la moyenne des micros du site ce jour-là ; le rouge est un jour où des micros n'ont rien enregistré. Pose et relève sont plus clairs : ces jours-là sont partiels par construction.\n\n"
            "Il y a trois vrais trous. Le plus gros : la station E de Molokoï, 54 jours sans rien entre fin novembre 2023 et mi-janvier 2024, parce que le premier micro s'est arrêté au bout d'une semaine et n'a été remplacé qu'en janvier. Pour la phénologie, c'est en pleine saison haute. Kaw D, cinq jours au changement d'enregistreur. Et à RNRT, cinq micros muets le même jour, le 10 février 2026. Pourquoi c'est important : l'absence d'enregistrement n'est pas l'absence de chant. Un trou doit se voir avant toute analyse d'activité, sinon on lit une baisse d'activité là où il n'y a qu'une baisse d'effort.\n\n"
            "Méthode et limite : le programme de chaque micro est lu dans les données (créneaux présents sur au moins la moitié de ses jours, hors pose et relève) ; si un micro n'a jamais fait son programme complet, par exemple une batterie trop faible dès le premier jour, on prendrait ce programme réduit pour le programme et il passerait pour tenu. Un jour sans aucun enregistrement n'est pas une ligne de la base : il n'entre pas dans le 99,7 %, il apparaît en rouge. Chiffres recomptés dans la base le 18/11/2026.")

# ================================================================= diapo 9 : horloge de Molokoï
s9 = S(9)
f9 = dict(shapes(s9))
f9[3] = remplacer_texte(f9[3], "INVENTAIRE · 7/7", "INVENTAIRE · 6/7")
ids = Ids(30)
garde = [f9[k] for k in (2, 3, 4, 5)]
r9 = retirer_rel(R(9), "rId3")
rids = ["rId10", "rId11", "rId12", "rId13", "rId14"]
r9 = ajouter_rel(r9, rids[0], "../media/mol_niveaux.png")
for k in range(1, 5):
    r9 = ajouter_rel(r9, rids[k], f"../media/mol_spectro_{k}.png")
garde.append(pic(ids(), rids[0], 0.6, 1.72, 12.13, 2.45, "Niveau des deux micros de Molokoï le 03/04/2024, à l'heure du nom et à celle de l'en-tête, avec la corrélation"))
legendes = [("SMA14163 · fichier 10 h 30", 0), ("SMA14163 · fichier 11 h 00 (l'averse)", 1),
            ("SMA14636 · fichier « 10 h 30 » (en-tête : 11 h 30)", 2), ("SMA14636 · fichier « 11 h 00 » (en-tête : 12 h 00)", 3)]
pw, ph = 2.95, 2.95 * 234 / 558
for (lg, k) in legendes:
    x = 0.6 + k * (pw + (12.13 - 4 * pw) / 3)
    garde.append(T(ids(), x, 4.28, pw, 0.2, [PP([(lg, 10, INK, True)])], nom=f"légende spectro {k + 1}"))
    garde.append(pic(ids(), rids[k + 1], x, 4.5, pw, ph, f"Spectrogramme 0-12 kHz : {lg}"))
garde.append(T(ids(), 0.6, 4.5 + ph + 0.05, 12.13, 0.2, [PP([("Spectrogrammes de 0 à 12 kHz, 2 min, piste 0 : l'averse du fichier de 11 h 00 est la bande claire sur toute la hauteur, sur les deux micros.", 10.5, MUTED, False)])], nom="légende générale"))
Y = 4.5 + ph + 0.31
H = 6.92 - Y
garde += carte(ids, 0.6, Y, 12.13, H)
c1 = [PP([("Le symptôme · ", 11, TEAL, True),
          ("SMA14636, du 20/02 au 11/04/2024 : l'en-tête GUANO dit une heure de plus que le nom du fichier. ", 11, INK, False),
          ("Lequel croire ? · ", 11, TEAL, True),
          ("le son : une averse s'entend sur les deux micros, à 150 m l'un de l'autre, en même temps. Elle tombe à 11 h avec l'heure du nom, à 12 h avec celle de l'en-tête.", 11, INK, False)])]
c2 = [PP([("Mesure · ", 11, TEAL, True),
          ("corrélation du niveau des deux micros sur les 1 521 enregistrements, profil journalier retiré : 0,57 avec l'heure du nom, 0,34 avec celle de l'en-tête. ", 11, INK, False),
          ("Correction · ", 11, TEAL, True),
          ("ces 1 521 prennent l'heure de leur nom (config, rejouée à chaque inventaire). ", 11, INK, False),
          ("Effet · ", 11, GOLD_DARK, True),
          ("trois sont dans le jeu de test ; l'un change de tranche horaire.", 11, INK, False)])]
garde.append(T(ids(), 0.9, Y + 0.07, 5.65, H - 0.12, c1, nom="diapo 9, colonne 1"))
garde.append(T(ids(), 6.75, Y + 0.07, 5.8, H - 0.12, c2, nom="diapo 9, colonne 2"))
parts["ppt/slides/slide9.xml"] = remplacer_arbre(s9, garde).encode("utf-8")
parts["ppt/slides/_rels/slide9.xml.rels"] = r9.encode("utf-8")
TITRES[9] = "Horloge de Molokoï : le son du micro voisin a dit laquelle croire"
NOTES[9] = ("L'horloge de Molokoï. Pour un des enregistreurs de Molokoï, SMA14636, entre le 20 février et le 11 avril 2024, le nom du fichier et l'en-tête ne donnent pas la même heure : une heure d'écart. L'en-tête est écrit en UTC moins 4, alors que la Guyane est en UTC moins 3. La vraie question, c'est lequel croire. Si l'horloge était juste et que seul le fuseau était mal réglé, c'est l'en-tête qui aurait raison ; dans l'autre cas, c'est le nom. Plutôt que de raisonner, on regarde le son : un orage, une averse, le lever du jour, ça s'entend en même temps sur deux micros voisins.\n\n"
            "En haut, le niveau sonore des deux micros le 3 avril 2024, de 8 h à 15 h, un point toutes les demi-heures. SMA14163, le voisin à 150 mètres, a une horloge saine : c'est la référence, en vert. À gauche, SMA14636 à l'heure du nom de fichier : les deux courbes se superposent, le gros pic de 11 h et celui de 12 h 30 tombent aux mêmes heures. À droite, le même micro à l'heure de l'en-tête GUANO : ses pics sont décalés d'une heure, celui de 11 h arrive à 12 h. En bas, les spectrogrammes confirment à l'œil : l'averse est la bande claire sur toute la hauteur du spectre ; elle est dans le fichier nommé 11 h 00 des deux micros. Pour SMA14636, l'en-tête de ce fichier dit 12 h 00.\n\n"
            "Mesure et méthode : on corrèle le niveau de SMA14636 (RMS de la piste 0, rangé par blanci qc) avec celui de SMA14163 sur les 1 521 enregistrements de la période, en le décalant de moins trois à plus trois heures par pas d'une demi-heure, après avoir retiré le profil journalier moyen de chaque micro pour que la corrélation ne vienne pas seulement du cycle jour-nuit. La corrélation de Pearson culmine à moins une heure, 0,57, contre 0,34 sans décalage (pour les deux graphiques du haut, c'est la corrélation sur toute la période, pas sur la seule journée du 3 avril). Le décalage d'une heure est le seul où les 1 521 créneaux tombent tous sur un créneau du voisin. Limite : une seule paire de micros, pour une seule période ; une horloge fausse et cohérente partout (nom et en-tête se trompent ensemble) ne se verrait pas.\n\n"
            "La correction : ces 1 521 enregistrements prennent l'heure de leur nom (recorder.clock_corrections dans la config, appliquée par blanci flag et à chaque inventaire) ; c'est idempotent. C'est important parce que le tirage stratifié se fait par tranche horaire : trois enregistrements de SMA14636 sont dans le jeu de test, et l'un d'eux change de tranche. Je le garde et je le signale.\n\n"
            "État de la figure au 18/11/2026 : les deux graphiques du haut sont refaits depuis la base (lecture seule) à la taille de leur place ; les quatre spectrogrammes sont découpés dans l'image du 6 novembre (sources_paquet_2/molokoi_03avril.png), parce que le disque BLANCI_2023-24 n'était pas branché. Ils se refont avec fig_molokoi_spectros() dès qu'il l'est. Chiffres recalculés ce jour : 1 521 enregistrements corrigés, corrélations 0,57 et 0,34.")

# ================================================================= diapo 9 bis : changement de partie
s9b = S(8)
if "<p:bg>" not in s9b:
    s9b = s9b.replace("<p:spTree>", f'<p:bg><p:bgPr><a:solidFill><a:srgbClr val="{DARK}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>', 1)
f8 = dict(shapes(S(8)))
ids = Ids(10)
f = []
f.append(rect(ids(), 0.6, 1.25, 4.1, 0.36, DARK2, radius=50000))
f.append(T(ids(), 0.6, 1.25, 4.1, 0.36, [PP([("STAGE ONF GUYANE · SUIVI N° 4 · PARTIE 2 SUR 3", 11, WHITE, True)], algn="ctr")], anchor="ctr", name="Pastille"))
f.append(T(ids(), 0.6, 1.9, 12.0, 1.0, [PP([("Partie 2 · Contrôle qualité", 42, WHITE, True)])], name="Titre"))
f.append(T(ids(), 0.6, 3.05, 10.5, 0.9, [PP([("L'inventaire a dit ce qu'on a ; le contrôle qualité dit ce qu'on garde, ce qu'on écarte, et pourquoi.", 20, SAGE, False)])], name="Sous-titre"))
bw = 3.8
gap = (12.13 - 3 * bw) / 2
etapes = [("1 · INVENTAIRE · FAIT", "Ce qu'on a : 96 310 enregistrements, quatre sites, 164 points", DARK2),
          ("2 · CONTRÔLE QUALITÉ · MAINTENANT", "Ce qu'on garde, ce qu'on écarte, et pourquoi", TEAL),
          ("3 · DIFFICULTÉS RENCONTRÉES · À LA FIN", "Les erreurs en route et ce qu'elles m'ont appris", DARK2)]
for k, (a, b, fond) in enumerate(etapes):
    x = 0.6 + k * (bw + gap)
    f.append(rect(ids(), x, 4.6, bw, 1.35, fond, radius=6000))
    f.append(T(ids(), x + 0.2, 4.75, bw - 0.4, 1.1, [PP([(a, 11, SAGE if fond == DARK2 else WHITE, True)], after=6), PP([(b, 15, WHITE, True)])], name="Étape"))
f.append(numero(f8[5], "9 bis").replace('val="A7BFB2"', f'val="{SAGE}"'))
s9b = remplacer_arbre(s9b, f)
rels9b = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
          '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
          '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide28.xml"/>'
          '</Relationships>')
TITRES["9 bis"] = "Partie 2 · Contrôle qualité"
NOTES["9 bis"] = ("Changement de partie. L'inventaire est derrière nous : il a dit ce qu'on a, 96 310 enregistrements de deux minutes sur quatre sites. On entre dans la deuxième partie, le contrôle qualité : il dit ce qu'on garde, ce qu'on écarte, et pourquoi. Un enregistrement écarté sort du corpus pour toute la suite : il n'est jamais encodé, jamais écouté, jamais scoré. C'est donc la décision la plus lourde de l'entrée des données, et c'est pour ça qu'elle a sa partie.\n\n"
                  "Les trois étapes de la présentation, comme annoncé sur la première diapo : l'inventaire, fait ; le contrôle qualité, maintenant ; les difficultés rencontrées à la fin, les erreurs que j'ai faites en route et ce qu'elles m'ont appris. Plusieurs choses que j'ai corrigées au fil de l'eau, comme l'ancienne règle des doublons ou les enregistrements valables écartés à tort, sont renvoyées à cette troisième partie plutôt que montrées ici.")

# ================================================================= diapo 9 ter : sommaire du contrôle qualité
s9t = S(8)
ids = Ids(10)
f = []
r, t = pastille(f8[2], f8[3], "CONTRÔLE QUALITÉ · 1/16", CQ)
f += [r, t, remplacer_texte(f8[4], "Doublons : le nom ne suffit plus, on compare aussi le contenu", "Le contrôle qualité en dix contrôles : ce qu'ils visent, s'ils écartent, combien"), numero(f8[5], "9 ter")]
colw, chh, gapc = 5.98, 0.84, 0.09
X = [0.6, 0.6 + colw + 0.17]
ecarte = ("écarte", RED)
rem = ("remarque", GOLD_DARK)
col_a = ("SANS LIRE LE SON : MÉTADONNÉES ET DATES", [
    ("Les drapeaux, et où on les lit", "diapo 11", "ce que le code note sur un enregistrement, et comment", ("lecture", TEAL), "quatre drapeaux calculés écartent ; 681 écartés sur 96 310"),
    ("Hors relevé", "diapo 10", "test d'avant la pose, reste d'une carte d'un autre lieu", ecarte, "88 écartés (106 levés, 18 gardés après écoute)"),
    ("Durée anormale", "diapo 10", "fichier qui ne dure pas 120 s, à 1 s près", ecarte, "173 fichiers, dont 34 du programme coupés"),
    ("Doublons", "diapo 8", "même nom et même contenu qu'un fichier déjà inventorié", ecarte, "3 673 copies de CDR sur les cartes de Mataroni"),
    ("Silence", "diapos 13 à 17", "niveau RMS sous −70 dBFS : micro muet", ecarte, "aucun aujourd'hui"),
])
col_b = ("EN LISANT LE SON, PUIS À L'ÉCOUTE", [
    ("Son étouffé", "diapos 13 à 17", "énergie au-dessus de 2 kHz sous 5 %, 4 de suite, deux pistes", ecarte, "506 écartés, dont 399 par le calcul"),
    ("Ce qui passe", "diapo 18", "tirages au hasard dans ce qui est gardé", ("mesure", TEAL), "6 étouffés sur 674 écoutés (chiffre de la diapo)"),
    ("D'où viennent les exclusions", "diapo 19", "par site, micro et date", ("constat", TEAL), "CDR : 6 micros portent 452 des 681 (diapo)"),
    ("Micro sourd", "diapo 20", "écart entre pistes qui dérive, ratio d'aigus normal", rem, "Kaw D : 1 871 enr. encodés sur la piste sourde (diapo)"),
    ("Les autres critères", "diapo 21", "bascule, clics, trous numériques, mauvais site…", rem, "11 bascules à CDR (diapo) ; le reste à décider"),
])
y0 = 1.72
for (titre_col, cartes), x in zip((col_a, col_b), X, strict=True):
    f.append(T(ids(), x, y0, colw, 0.22, [PP([(titre_col, 10.5, TEAL, True)])], nom="en-tête de colonne"))
    for k, (nom, ref, vise, (eff, cc), nb) in enumerate(cartes):
        y = y0 + 0.27 + k * (chh + gapc)
        f += carte(ids, x, y, colw, chh, barre=cc)
        f.append(T(ids(), x + 0.2, y + 0.06, colw - 0.3, chh - 0.08, [
            PP([(nom, 12.5, INK, True), (f"   {ref}", 10, MUTED, False)], after=1),
            PP([("Vise · ", 11, TEAL, True), (vise, 11, INK, False)], after=1),
            PP([(eff.capitalize(), 11, cc, True), (f" · {nb}", 11, INK, False)])], nom=f"carte {nom}"))
f.append(T(ids(), 0.6, 6.66, 12.13, 0.28, [PP([("Nombres au 18/11/2026, recomptés dans la base pour les cinq premiers ; ceux des diapos 18 à 21 sont les leurs, à mettre à jour avec les paquets 4 et 5. « diapo » : numéros d'origine.", 10, MUTED, False)])], nom="note de bas"))
s9t = remplacer_arbre(S(8), f)
rels9t = rels9b.replace("notesSlide28", "notesSlide29")
TITRES["9 ter"] = "Le contrôle qualité en dix contrôles : ce qu'ils visent, s'ils écartent, combien"
NOTES["9 ter"] = ("Le plan du contrôle qualité. Dix contrôles, dans l'ordre où la suite les présente. À gauche, ceux qui ne lisent pas le son : les drapeaux eux-mêmes, d'abord, c'est-à-dire ce que le code note sur un enregistrement et comment ; puis le hors relevé, la durée anormale et les doublons, qui se calculent sur les dates, les durées et les noms. À droite, ceux qui lisent le son, puis l'écoute : le silence et le son étouffé, avec leur indice, leur seuil, la règle des suites et les deux pistes ; ce qui passe, c'est-à-dire ce que les tirages au hasard révèlent des enregistrements gardés ; d'où viennent les exclusions ; le micro sourd que le contrôle n'a pas vu ; et les autres critères, mesurés ou envisagés.\n\n"
                  "Pour chacun : ce qu'il vise, s'il écarte ou s'il n'est qu'une remarque, et combien aujourd'hui. Seuls cinq contrôles écartent : hors relevé, durée anormale, doublons, silence et son étouffé. Le total écarté est de 681 enregistrements sur 96 310, au 18 novembre : 506 pour le son étouffé (dont 399 par le calcul sur les deux pistes, les autres confirmés ou notés à l'écoute), 173 pour la durée, 2 seulement pour le hors relevé, les 86 autres hors relevé étant aussi de durée anormale. Rien pour le silence, et les doublons sont écartés à l'inventaire, ils ne sont pas dans les 681 : ce sont des fichiers qui n'entrent pas dans la base.\n\n"
                  "Les chiffres des cinq premiers contrôles sont recomptés dans la base (lecture seule) le 18/11/2026, sauf les 3 673 doublons, qui ne se recomptent qu'avec le disque d'origine. Ceux des quatre derniers sont ceux des diapos 18 à 21 telles qu'elles sont aujourd'hui, à mettre à jour avec les paquets 4 et 5. Les numéros de diapos sont ceux de la présentation d'origine, à refaire au réassemblage.")

# ================================================================= diapo 11 : les drapeaux
s11 = S(11)
f11 = dict(shapes(s11))
r, t = pastille(f11[2], f11[3], "CONTRÔLE QUALITÉ · 2/16", CQ)
ids = Ids(30)
garde = [r, t, f11[4], f11[5]]
r11 = ajouter_rel(R(11), "rId10", "../media/qc_ligne_drapeau.png")
SZ = 11


def c_txt(t, bold=False, color=INK):
    return [(t, SZ, color, bold)]


def c_flag(t):
    return [(t, 10.5, TEAL, True, MONO)]


def c_eff(t, color):
    return [(t, SZ, color, True)]


rows = [
    ("duration_off", "durée anormale", "lu dans les métadonnées : durée du WAV à plus de 1 s de 120 s", ("écarte", RED)),
    ("off_campaign", "hors des dates du relevé", "calculé sur les dates : série du micro coupée aux trous de plus de 7 jours", ("écarte", RED)),
    ("silent", "micro muet", "calculé sur le son : niveau RMS sous −70 dBFS", ("écarte", RED)),
    ("muffled", "son étouffé sur les deux pistes", "calculé sur le son, piste par piste : énergie au-dessus de 2 kHz sous 5 % du total, 4 enregistrements de suite", ("écarte", RED)),
    ("muffled_0 / muffled_1", "une piste étouffée, l'autre bonne", "même calcul, une piste seulement", ("gardé : encodé sur l'autre piste", TEAL)),
    ("clock_off", "horloge douteuse", "comparaison de l'heure du nom et de celle du GUANO, à plus de 5 min", ("remarque", GOLD_DARK)),
    ("clock_corrected", "heure corrigée (1 521, Molokoï)", "écrit quand l'heure est relue dans le nom (config)", ("remarque", GOLD_DARK)),
    ("rain", "pluie", "noté à l'écoute (label ou mention de pluie)", ("remarque", GOLD_DARK)),
    ("keep", "écouté et gardé malgré un drapeau", "noté à l'écoute : l'emporte sur tous les drapeaux qui écartent", ("garde", TEAL)),
    ("confirmed_…", "écartement confirmé à l'écoute", "noté à l'écoute (confirmation en bloc) : écarte comme le calcul", ("écarte", RED)),
]
largeurs = [1.9, 2.75, 5.0, 2.48]
tete = [cellule([(h, 11.5, WHITE, True)], DARK) for h in ("Drapeau", "Ce que c'est", "Comment on le pose", "Effet")]
lignes, hauteurs = [tete], [0.3]
for k, (a, b, c, (eff, col)) in enumerate(rows):
    fond = WHITE if k % 2 == 0 else PALER
    lignes.append([cellule(c_flag(a), fond), cellule(c_txt(b), fond), cellule(c_txt(c), fond), cellule(c_eff(eff, col), fond)])
    deux = _largeur is not None and _largeur(c, SZ) > largeurs[2] - 0.15
    if _largeur is not None:
        for t_, w_ in ((b, largeurs[1]), (eff, largeurs[3])):
            if _largeur(t_, SZ, t_ == eff) > w_ - 0.15:
                deux = True
    hauteurs.append(0.46 if deux else 0.3)
garde.append(tableau(ids(), 0.6, 1.72, largeurs, lignes, hauteurs))
ytab = 1.72 + sum(hauteurs)
yimg = ytab + 0.1
garde.append(pic(ids(), "rId10", 0.6, yimg, 12.13, 1.55, "Une vraie ligne de la table recordings avec un drapeau : Mataroni SMA13417, 104 s, duration_off"))
if yimg + 1.55 > 6.96:
    ALERTES.append(f"diapo 11 : image jusqu'à {yimg + 1.55:.2f} po")
parts["ppt/slides/slide11.xml"] = remplacer_arbre(s11, garde).encode("utf-8")
parts["ppt/slides/_rels/slide11.xml.rels"] = r11.encode("utf-8")
TITRES[11] = re.search(r"<a:t>([^<]*)</a:t>", f11[4]).group(1)
NOTES[11] = ("Le principe. Le contrôle qualité ne sert qu'à sortir du projet les fichiers cassés ou sans rapport avec lui. Il ne juge pas si l'enregistrement est beau ou facile. Un drapeau est une remarque posée sur l'enregistrement, dans la base ; le fichier audio, lui, n'est jamais touché ni déplacé. Une ligne par drapeau, et pour chacun, comment on le pose.\n\n"
             "Lus dans les métadonnées, sans rien calculer : duration_off, la durée du fichier, lue dans l'en-tête du WAV, à plus d'une seconde de 120 s. Calculés sur les dates : off_campaign, où l'on trie les dates de chaque micro et où l'on coupe la série à chaque trou de plus de sept jours ; le plus gros bloc est le relevé, les autres sont hors relevé. Calculés sur le son, en une seule passe avant tout encodage : silent, le niveau RMS sous moins 70 dBFS ; et muffled, le son étouffé : pour chaque piste, la part de l'énergie au-dessus de 2 kHz, qui doit rester sous 5 % pendant au moins quatre enregistrements consécutifs du même micro. L'enregistrement n'est écarté que si les deux pistes le sont ; si une seule l'est, il reste, avec muffled_0 ou muffled_1, et il est encodé sur l'autre piste. Par comparaison : clock_off compare l'heure du nom du fichier à celle de l'en-tête GUANO, à plus de cinq minutes près ; c'est une remarque, qui n'écarte rien. clock_corrected garde en mémoire les 1 521 enregistrements de Molokoï dont l'heure a été relue dans le nom. Et notés à l'écoute : rain, la pluie, quand j'annote un extrait ; keep, un enregistrement écouté et jugé bon, qui l'emporte sur tous les drapeaux qui écartent ; et confirmed_, la confirmation à l'écoute d'un écartement.\n\n"
             "Seuls quatre drapeaux calculés écartent : durée anormale, hors relevé, silencieux, étouffé sur les deux pistes, plus la confirmation à l'écoute. La pluie, la saturation et l'horloge douteuse restent dans le corpus : la pluie et la saturation font partie du milieu, et un détecteur qui ne les aurait jamais vues se tromperait dessus sur le terrain. L'horloge douteuse a été rétrogradée en remarque le 6 octobre. Dernière règle, la plus importante : un enregistrement où A. blanci a été entendu n'est jamais écarté, quel que soit le calcul. L'écoute prime.\n\n"
             "L'image du bas est une vraie ligne de la table recordings, dans le même explorateur que la diapo 5 bis : Mataroni, micro SMA13417, le 6 janvier 2026 à 11 h 30, 104 secondes au lieu de 120, donc duration_off vrai. Le dictionnaire qc_flags est lisible en entier : confirmed_duration_off a été posé à l'écoute, en bloc. La diapo suivante parle de ces deux drapeaux, et de cet enregistrement. Le dictionnaire d'un enregistrement étouffé, avec ses indices, est montré plus loin (diapo 12).\n\n"
             "Méthode : les seuils (1 s, 7 jours, −70 dBFS, 0,05, 4 de suite, 5 min) sont fixés à la main dans config/default.yaml ; aucun n'est estimé sur les données, et la limite de chacun est discutée là où on le présente. Vérifié dans blanci/inputs/qc.py le 18/11/2026.")

# ================================================================= diapo 10 : hors relevé et durée anormale
s10 = S(10)
f10 = dict(shapes(s10))
r, t = pastille(f10[2], f10[3], "CONTRÔLE QUALITÉ · 3/16", CQ)
ids = Ids(30)
garde = [r, t, f10[4], f10[5]]
bw, by, bh = 5.98, 1.72, 5.2


def boite(x, titre, code, gros, petit, blocs, renvoi):
    out = carte(ids, x, by, bw, bh)
    out.append(T(ids(), x + 0.25, by + 0.1, bw - 0.4, 0.3, [PP([(titre, 13, TEAL, True), (f"   {code}", 11, MUTED, False, MONO)])], nom=f"titre {titre}"))
    out.append(T(ids(), x + 0.25, by + 0.45, bw - 0.4, 0.5, [PP([(gros, 22, INK, True), (f"  {petit}", 11, MUTED, False)])], nom=f"nombre {titre}"))
    pps = []
    for it, tx in blocs:
        pps.append(PP([(it + " · ", 12.5, TEAL if it != "Écarte à tort" else GOLD_DARK, True), (tx, 12.5, INK, False)], after=5))
    pps.append(PP([(renvoi, 12.5, MUTED, True)]))
    out.append(T(ids(), x + 0.25, by + 1.0, bw - 0.4, bh - 1.08, pps, nom=f"corps {titre}"))
    return out


garde += boite(0.6, "HORS RELEVÉ", "off_campaign", "88 écartés", "106 levés, dont 18 gardés après écoute", [
    ("Ce que c'est", "un enregistrement isolé de la série de son micro : un test d'avant la pose, ou un reste de carte SD d'un autre lieu."),
    ("Comment c'est calculé", "pour chaque (jeu, site, micro), on trie les dates et on coupe la série à chaque trou de plus de 7 jours. Chaque morceau est un bloc ; le plus gros est le relevé, les autres sont hors relevé. Le seuil de 7 jours est choisi à la main (qc.campaign_gap_days) : on n'a pas les dates de pose."),
    ("Combien", "238 blocs, dont 74 petits ; 106 enregistrements, dont 20 seulement de 2 min. 86 des 88 écartés sont aussi de durée anormale : des tests."),
    ("Écarte à tort", "un micro reposé au même site après plus de 7 jours d'arrêt : 18 enregistrements du relevé 7 de Mataroni (2LA03595, 19/05/2026), écoutés puis gardés."),
    ("Laisse passer", "un test fait moins de 7 jours avant la pose : 26 enregistrements de 20 micros, les 30 et 31/12/2025, rangés dans le relevé de Mataroni six jours avant la pose ; tous de durée anormale, donc écartés quand même."),
], "→ voir Difficultés rencontrées")
garde += boite(6.75, "DURÉE ANORMALE", "duration_off", "173 écartés", "125 tests, 34 coupés, 14 manuels", [
    ("Ce que c'est", "un fichier qui ne dure pas 120 s à 1 s près (2 min : le programme des Song Meter) : test à la pose, déclenchement manuel, ou enregistrement du programme coupé."),
    ("Comment c'est calculé", "la durée du WAV est lue dans l'en-tête (échantillons ÷ fréquence), sans lire le son. Tolérance d'une seconde (qc.duration_tolerance_s), choisie à la main."),
    ("Combien", "173 fichiers : 125 de moins de 30 s (tests), 34 de 30 à 119 s, 14 de plus de 2 min (enregistrements manuels, jusqu'à 14 min). Tous écartés."),
    ("Écarte à tort", "les 34 enregistrements du programme coupés, qui sont du vrai terrain. Mataroni, SMA13417, 6/01/2026, 104 s : A. blanci y chante (BlanciNet 0,97, écouté le 12/11). Seuls 3 des 34 ont une détection de BlanciNet ≥ 0,5, dont une fausse alerte possible : un score seul ne tranche pas."),
    ("Laisse passer", "rien de plus : un fichier de 120 s est gardé, même s'il est vide ou étouffé (c'est le travail des contrôles suivants)."),
], "→ voir Difficultés rencontrées")
parts["ppt/slides/slide10.xml"] = remplacer_arbre(s10, garde).encode("utf-8")
TITRES[10] = "Hors relevé et durée anormale : des tests, et du vrai terrain coupé"
NOTES[10] = ("Deux drapeaux d'inventaire, qu'on peut calculer sans écouter, présentés de la même façon : ce que c'est, comment c'est calculé, combien, ce qu'ils écartent à tort ou laissent passer.\n\n"
             "Hors relevé. Vous m'aviez demandé ce qu'est un bloc : on prend toutes les dates d'un micro, on les trie, et on coupe la série partout où il y a un trou de plus de sept jours. Chaque morceau est un bloc. Le plus gros, c'est le relevé ; les autres sont hors relevé : des tests faits avant la pose, ou des fichiers d'un autre lieu restés sur la carte. Au 18 novembre : 238 blocs, dont 74 petits, 106 enregistrements, dont 20 seulement de deux minutes. 88 sont écartés, et 86 d'entre eux sont aussi de durée anormale : ce sont bien des tests. Le seuil de sept jours est choisi à la main, il n'est pas estimé : on n'a pas les dates de pose du terrain. Le risque est un micro reposé au même endroit après une longue pause : il fait deux vrais blocs, et le petit est écarté à tort. C'est arrivé : les 18 enregistrements du relevé 7 de Mataroni, micro 2LA03595, le 19 mai 2026, sont dans un petit bloc ; je les ai écoutés et marqués « gardés », c'est pour ça que 106 sont levés et 88 écartés. La règle a une seconde limite, dans l'autre sens : 26 enregistrements de 20 micros, les 30 et 31 décembre 2025, sont rangés dans le relevé de Mataroni, six jours avant la pose ; un trou de six jours ne dépasse pas sept, donc ils ne sont pas hors relevé. Mais ils sont tous de durée anormale, donc écartés par l'autre critère.\n\n"
             "Durée anormale : 173 fichiers, qui ne durent pas 120 secondes à une seconde près. 125 font moins de 30 secondes, ce sont des tests. 14 durent plus de deux minutes : des enregistrements manuels, jusqu'à 14 minutes. Mais 34 sont des enregistrements du programme coupés, à la pose, à la relève ou en fin de batterie, entre 30 secondes et 2 minutes. Ils sont parfaitement audibles. Le détecteur BlanciNet de Biophonia, dont les scores existent pour 8 349 enregistrements de 2026, en signale 3 à 0,5 ou plus : Mataroni SMA13417, 104 secondes, 0,97 ; CDR 2MA10210, 0,83 ; Mataroni 2LA05209, 0,57. Le premier, je l'ai écouté le 12 novembre : A. blanci chante, de 0 à 67 secondes. Les deux autres sont à écouter. La règle des 120 secondes écarte donc du vrai terrain. Ce qu'on peut dire : au moins un chant a été perdu par ce critère. Ce qu'on ne peut pas dire : un pourcentage de chants perdus, ni que BlanciNet est fiable (une fausse alerte à CDR, 2LA05017). Un score seul ne tranche pas : biais de vérification, on n'a écouté que ce que le test signalait.\n\n"
             "Les deux boîtes se terminent par le renvoi à la partie Difficultés rencontrées, où je reprends ces enregistrements valables écartés par ces critères, et ce que j'aurais dû faire.\n\n"
             "Chiffres recomptés dans la base (lecture seule) le 18/11/2026. Remarque de vocabulaire : on parle de « hors programme » à l'oral ; le drapeau s'appelle hors relevé, hors des dates de la campagne, pas hors des heures du programme.")

# ================================================================= diapo 8 : doublons
s8 = S(8)
r, t = pastille(f8[2], f8[3], "CONTRÔLE QUALITÉ · 4/16", CQ)
titre8 = remplacer_texte(f8[4], "Doublons : le nom ne suffit plus, on compare aussi le contenu", "Doublons : un fichier n'est un doublon que si le nom et le contenu concordent")
ids = Ids(30)
garde = [r, t, titre8, f8[5]]
garde += carte(ids, 0.6, 1.72, 12.13, 1.08, barre=RED)
garde.append(T(ids(), 0.9, 1.8, 11.6, 0.95, [PP([("3 673 DOUBLONS ÉCARTÉS : LES CARTES SD DE CDR RAPPORTÉES À MATARONI", 11, RED, True)], after=3),
                                              PP([("Une carte SD rapportée d'un relevé contient encore les fichiers du précédent : à l'inventaire de Mataroni, 3 673 fichiers de CDR. Les compter deux fois, sous deux sites, fausserait tout : le contrôle écarte la copie, et garde l'original.", 12.5, INK, False)])], nom="diapo 8, problème"))
garde.append(T(ids(), 0.6, 2.95, 12.13, 0.28, [PP([("LA RÈGLE : LE NOM ET LE CONTENU DOIVENT CONCORDER (ingest.same_recording)", 11, TEAL, True)])], nom="diapo 8, règle"))
tests = [("Le nom", "même nom de fichier"), ("La longueur", "même nombre d'échantillons"), ("La fréquence et les pistes", "même fréquence, même nombre de pistes"),
         ("L'en-tête", "même numéro de série et même heure, quand les deux fichiers en ont"), ("Le son", "première seconde décodée, comparée échantillon par échantillon")]
cw = (12.13 - 4 * 0.1) / 5
for k, (a, b) in enumerate(tests):
    x = 0.6 + k * (cw + 0.1)
    garde += carte(ids, x, 3.27, cw, 1.15, fond=PALE if k < 4 else PALE)
    garde.append(T(ids(), x + 0.18, 3.34, cw - 0.25, 1.0, [PP([(f"{k + 1} · {a}", 12, TEAL, True)], after=3), PP([(b, 11.5, INK, False)])], nom=f"test {a}"))
garde.append(T(ids(), 0.6, 4.5, 12.13, 0.3, [PP([("Un seul test échoue : « conflit », signalé à part (ingest_conflicts_*.csv), jamais compté en doublon : on garde les deux fichiers.", 12, INK, True)])], nom="conflit"))
garde += carte(ids, 0.6, 4.95, 5.98, 1.95)
garde.append(T(ids(), 0.85, 5.03, 5.6, 1.8, [PP([("Combien · à revérifier", 13, TEAL, True)], after=4),
                                             PP([("3 673 copies, chacune avec son original dans le relevé de CDR. Les copies écartées ne sont plus que sur le disque d'origine (Biophonia) : la vérification par le contenu (scripts/controle_qualite/verifier_doublons.py) attend qu'il soit branché. Sur les deux disques du stage, aucun nom n'est répété en 2025-2026, et un seul en 2023-2024 (même contenu).", 12.5, INK, False)])], nom="diapo 8, combien"))
garde += carte(ids, 6.75, 4.95, 5.98, 1.95, barre=GOLD_DARK)
garde.append(T(ids(), 7.0, 5.03, 5.6, 1.8, [PP([("Pourquoi ces tests · limite", 13, GOLD_DARK, True)], after=4),
                                            PP([("Une comparaison exacte, sans seuil ni statistique : deux fichiers de même nom, même longueur, même numéro de série, même heure d'en-tête et même première seconde sont, en pratique, le même enregistrement. Limite : un fichier renommé à la main n'est pas vu, la règle ne compare que des fichiers du même nom.", 12.5, INK, False)])], nom="diapo 8, limite"))
parts["ppt/slides/slide8.xml"] = remplacer_arbre(s8, garde).encode("utf-8")
TITRES[8] = "Doublons : un fichier n'est un doublon que si le nom et le contenu concordent"
NOTES[8] = ("Les doublons. Quand on rapporte une carte SD d'un relevé, elle contient souvent encore les fichiers du relevé d'avant. À l'inventaire de Mataroni, 3 673 fichiers étaient en fait des fichiers de CDR. Si on les gardait, ils compteraient deux fois, et sous deux sites : le contrôle écarte la copie et garde l'original. La place de cette diapo est ici, dans le contrôle qualité : c'est une décision d'écarter, pas une description de ce qu'on a.\n\n"
            "La règle : un fichier n'est un doublon que si le nom et le contenu concordent. On compare d'abord le nom ; puis la longueur, c'est-à-dire le nombre d'échantillons ; la fréquence d'échantillonnage et le nombre de pistes ; le numéro de série et l'heure de l'en-tête GUANO, quand les deux fichiers en ont ; et enfin la première seconde d'audio, décodée et comparée échantillon par échantillon. Si un seul de ces tests échoue, ce n'est pas un doublon : c'est un conflit, signalé à part dans un fichier de rapport, jamais compté en doublon ; on garde les deux fichiers.\n\n"
            "Effectif : 3 673 copies de CDR sur les cartes de Mataroni. Ce chiffre n'a pas pu être revérifié le 18/11/2026 : les copies écartées ne sont que sur le disque d'origine, celui de Biophonia, et la vérification par le contenu, scripts/controle_qualite/verifier_doublons.py, attend qu'il soit branché. Sur les deux disques du stage, aucun nom n'est répété en 2025-2026, et un seul en 2023-2024, avec le même contenu.\n\n"
            "Méthode : c'est une comparaison exacte, sans seuil ni loi statistique : la première seconde décodée à 48 kHz, ce sont 48 000 valeurs par piste comparées une à une ; ajoutée à la longueur, au numéro de série et à l'heure de l'en-tête, elle fait que deux fichiers qui passent tous les tests sont, en pratique, le même enregistrement. Limite : un fichier renommé à la main n'est pas vu, la règle ne compare que des fichiers du même nom. L'ancienne règle, qui ne regardait que le nom, et pourquoi elle était fausse, sont renvoyées à la partie Difficultés rencontrées.")

# ================================================================= nouvelles diapos : 9 bis et 9 ter
ct = parts["[Content_Types].xml"].decode("utf-8")
prels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
pres = parts["ppt/presentation.xml"].decode("utf-8")
note8 = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', R(8)).group(1)
for num, sx, rx, cle, rid in ((28, s9b, rels9b, "9 bis", "rId900"), (29, s9t, rels9t, "9 ter", "rId901")):
    parts[f"ppt/slides/slide{num}.xml"] = sx.encode("utf-8")
    parts[f"ppt/slides/_rels/slide{num}.xml.rels"] = rx.encode("utf-8")
    n = notes(parts[f"ppt/notesSlides/{note8}"].decode("utf-8"), NOTES[cle]).replace("<a:t>8</a:t></a:fld>", f"<a:t>{9}</a:t></a:fld>")
    parts[f"ppt/notesSlides/notesSlide{num}.xml"] = n.encode("utf-8")
    parts[f"ppt/notesSlides/_rels/notesSlide{num}.xml.rels"] = re.sub(
        r'Target="\.\./slides/slide\d+\.xml"', f'Target="../slides/slide{num}.xml"',
        parts[f"ppt/notesSlides/_rels/{note8}.rels"].decode("utf-8")).encode("utf-8")
    ct = ct.replace("</Types>",
                    f'<Override PartName="/ppt/slides/slide{num}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                    f'<Override PartName="/ppt/notesSlides/notesSlide{num}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>')
    prels = prels.replace("</Relationships>",
                          f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{num}.xml"/></Relationships>')
parts["[Content_Types].xml"] = ct.encode("utf-8")
parts["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")

# ordre : 6, 7, 9, 9 bis, 9 ter, 11, 10, 8
rid_de = {int(m.group(2)): m.group(1) for m in re.finditer(r'Id="(rId\d+)"[^>]*Target="slides/slide(\d+)\.xml"', prels)}
rid_de |= {int(m.group(1)): m.group(2) for m in re.finditer(r'Target="slides/slide(\d+)\.xml"[^>]*Id="(rId\d+)"', prels)}
sld = {m.group(2): m.group(1) for m in re.finditer(r'<p:sldId id="(\d+)" r:id="(rId\d+)"/>', pres)}  # rId -> id
ordre = [6, 7, 9, 28, 29, 11, 10, 8]
liste = ""
for k in ordre:
    rid = rid_de[k]
    liste += f'<p:sldId id="{sld.get(rid, 401 if k == 28 else 402)}" r:id="{rid}"/>'
pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", f"<p:sldIdLst>{liste}</p:sldIdLst>", pres, flags=re.S)
parts["ppt/presentation.xml"] = pres.encode("utf-8")
parts["docProps/app.xml"] = re.sub(rb"<Slides>\d+</Slides>", b"<Slides>8</Slides>", parts["docProps/app.xml"])

# ------------------------------------------------------------------ notes des diapos existantes, médias, types
for n, t in NOTES.items():
    if n in ("9 bis", "9 ter"):
        continue
    cible = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', R(n)).group(1)
    k = f"ppt/notesSlides/{cible}"
    parts[k] = notes(parts[k].decode("utf-8"), t).encode("utf-8")

for nom in ("inv_gains_reglage.png", "inv_gains_mesure.png", "inv_completude_p2.png", "mol_niveaux.png",
            "mol_spectro_1.png", "mol_spectro_2.png", "mol_spectro_3.png", "mol_spectro_4.png", "qc_ligne_drapeau.png"):
    parts[f"ppt/media/{nom}"] = open(f"{FIG}/{nom}", "rb").read()
utilises = set()
for k, v in parts.items():
    if k.endswith(".rels"):
        utilises |= {m.split("/")[-1] for m in re.findall(r'Target="[^"]*media/([^"]+)"', v.decode("utf-8"))}
for k in [k for k in parts if k.startswith("ppt/media/") and k.split("/")[-1] not in utilises]:
    del parts[k]

with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as w:
    for k, v in parts.items():
        w.writestr(infos.get(k, zipfile.ZipInfo(k)), v, compress_type=zipfile.ZIP_DEFLATED)
print("écrit", DST)
for a in ALERTES:
    print("ALERTE :", a)

# ------------------------------------------------------------------ script des notes orateur (sections 6 à 11)
if SCRIPT_MD:
    md = Path(SCRIPT_MD).read_text(encoding="utf-8")
    a = md.index("## 6. ")
    b = md.index("## 12. ")
    bloc = ""
    for cle in (6, 7, 9, "9 bis", "9 ter", 11, 10, 8):
        bloc += f"## {cle}. {TITRES[cle]}\n\n" + NOTES[cle].replace("\n\n", "\n\n") + "\n\n"
    Path(SCRIPT_MD).write_text(md[:a] + bloc + md[b:], encoding="utf-8")
    print("écrit", SCRIPT_MD, "(sections 6 à 11)")
