# ruff: noqa: E501
"""Paquet 1 (diapos 1-5, plus 5 bis) de la présentation n° 4 : corrections demandées par Léonard
le 18/11/2026, appliquées au XML du paquet découpé (bibliothèque standard seulement). Les formes
non touchées (dont ses retouches à la main) sont recopiées telles quelles. Aides réutilisables par
les autres paquets : rect, text, para, pic, shapes, remplacer_arbre, remplacer_texte, notes.

    python documentation/prez/scripts/presentation_4_paquet_1.py PAQUET_DECOUPE.pptx SORTIE.pptx \
        documentation/prez/figures/inventaire_qc

Ne jamais écrire SORTIE par-dessus un fichier que Léonard a modifié depuis : relancer sur le
paquet découpé écraserait ses retouches."""

import re
import sys
import zipfile
from xml.sax.saxutils import escape

SRC, DST, FIG = sys.argv[1], sys.argv[2], sys.argv[3]
EMU = 914400
INK, MUTED, TEAL, GOLD, PALE, PALER, SAGE = "1B2A24", "5B6B63", "2A7F72", "E9B44C", "E4EEE8", "F3F7F4", "A7BFB2"
GOLD_DARK = "9A6B0F"
TERM_BG, TERM_BAR, TERM_TXT, TERM_DIM, TERM_HI, TERM_PROMPT = (
    "1E1E1E", "3A3A3A", "D4D4D4", "8C8C8C", "F2C14E", "7FD48A")


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


def text(i, x, y, w, h, paras, anchor="t", name="Text", autofit=False):
    fit = "<a:normAutofit/>" if autofit else ""
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="{name} {i}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{e(x)}" y="{e(y)}"/><a:ext cx="{e(w)}" cy="{e(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="{anchor}">{fit}</a:bodyPr>'
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


NOTES = {}
z = zipfile.ZipFile(SRC)
parts = {n: z.read(n) for n in z.namelist()}
infos = {n: z.getinfo(n) for n in z.namelist()}
S = lambda n: parts[f"ppt/slides/slide{n}.xml"].decode("utf-8")  # noqa: E731
R = lambda n: parts[f"ppt/slides/_rels/slide{n}.xml.rels"].decode("utf-8")  # noqa: E731

# ---------------------------------------------------------------- diapo 1 : sous-titre, date
s1 = S(1)
s1 = s1.replace("<a:t>Inventaire du corpus · contrôle qualité : chaque étape, chaque choix, ses limites</a:t>",
                "<a:t>Inventaire du corpus · contrôle qualité · difficultés rencontrées : chaque étape, chaque choix, ses limites</a:t>")
s1 = s1.replace("<a:t>Léonard Laplace-Palette · 12 novembre 2026</a:t>",
                "<a:t>Léonard Laplace-Palette · 20 novembre 2026</a:t>")
assert "20 novembre" in s1 and "difficultés rencontrées" in s1
# sous-titre sur deux lignes : on descend un peu la ligne d'auteur
f1 = dict(shapes(s1))
f1[7] = f1[7].replace(f'<a:off x="{e(0.6)}" y="{e(4.75)}"/>', f'<a:off x="{e(0.6)}" y="{e(4.95)}"/>')
f1[6] = f1[6].replace(f'<a:ext cx="{e(7.6)}" cy="{e(0.7)}"/>', f'<a:ext cx="{e(7.6)}" cy="{e(0.9)}"/>')
s1 = remplacer_arbre(s1, [f1[k] for k, _ in shapes(s1)])
parts["ppt/slides/slide1.xml"] = s1.encode("utf-8")
NOTES[1] = ("Quatrième point de suivi, et premier d'une série : une présentation par étape de la chaîne, pour prendre le temps. "
            "Aujourd'hui, l'entrée des données, en trois temps : ce que contient le corpus, l'inventaire ; puis le contrôle qualité, "
            "c'est-à-dire quels enregistrements sont sortis du corpus et pourquoi ; et pour finir les difficultés rencontrées, "
            "les erreurs que j'ai faites en route et ce qu'elles m'ont appris. Pour chaque étape je vous dirai ce que fait le code, "
            "le choix qu'il fait, sur quoi ce choix repose, et ce qu'il laisse passer ; tout est rejouable dans deux notebooks, "
            "05_inventaire et 06_controle_qualite. L'image de fond est un vrai chant d'A. blanci enregistré à Mataroni : "
            "les petites notes vers 4,5 kHz.")

# ---------------------------------------------------------------- diapo 3 : deux graphiques
s3 = S(3)
f3 = dict(shapes(s3))
f3[4] = remplacer_texte(f3[4], "96 292 enregistrements, 3 206 h : trois sites suivis un an, cinq sites une semaine",
                        "96 310 enregistrements sur quatre sites : un an de suivi, puis des relevés d'une semaine")
f3[7] = remplacer_texte(f3[7], "96 292", "96 310")
f3[10] = remplacer_texte(f3[10], "3 206 h", "3 207 h")
f3[14] = remplacer_texte(f3[14], "162 points d'écoute, 95 enregistreurs", "164 points d'écoute, 95 enregistreurs")
ids = Ids(30)
garde = [f3[k] for k in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)]
r3 = retirer_rel(R(3), "rId3")
r3 = ajouter_rel(ajouter_rel(r3, "rId10", "../media/inv_sites.png"), "rId11", "../media/inv_programmes.png")
garde += [pic(ids(), "rId10", 0.6, 3.2, 6.9, 3.55, "inv_sites.png"),
          pic(ids(), "rId11", 7.73, 3.2, 5.0, 3.55, "inv_programmes.png")]
parts["ppt/slides/slide3.xml"] = remplacer_arbre(s3, garde).encode("utf-8")
parts["ppt/slides/_rels/slide3.xml.rels"] = r3.encode("utf-8")
NOTES[3] = ("D'abord, ce qu'on a. L'inventaire compte 96 310 enregistrements de deux minutes, 3 207 heures, chacun sur deux pistes : "
            "les deux micros de l'enregistreur. Quatre sites, 164 points d'écoute, 95 enregistreurs.\n\n"
            "À gauche, les heures par site. Les relevés de 2025-2026 ne sont pas des sites à part : CDR, Patawa Est et Patawa Ouest "
            "sont des secteurs de Kaw, et RNRT est la réserve de Trésor. Je les ai donc empilés, en jaune, sur leur site, à côté du "
            "suivi d'un an de 2023-2024, en vert. Kaw et Trésor ont ainsi les deux : une année entière sur deux ou trois micros, puis "
            "une semaine sur des dizaines de micros. Molokoï n'a que le suivi, Mataroni que le relevé.\n\n"
            "À droite, le programme. Tous les micros enregistrent deux minutes toutes les trente minutes, rien la nuit, mais pas sur la "
            "même plage. On ne le suppose pas : on le lit dans les données, micro par micro, comme les créneaux présents au moins un "
            "jour sur deux. 74 007 enregistrements viennent de micros programmés de 5 h à 19 h 30, 30 par jour : tout le suivi "
            "2023-2024, CDR, presque tout Patawa Ouest, et trois micros isolés de RNRT et de Patawa Est. 22 289 viennent de micros "
            "programmés de 6 h à 18 h 30, 26 par jour : Mataroni, et presque tout RNRT et Patawa Est. Le programme suit donc la pose, "
            "pas le modèle d'enregistreur. Pour la suite, ça veut dire que les heures de l'aube et du soir, 5 h et 19 h, n'existent "
            "que pour une partie des sites : il faudra en tenir compte dans le tirage par tranche horaire.")

# ---------------------------------------------------------------- diapo 4 : calendrier
s4 = S(4)
f4 = dict(shapes(s4))
ids = Ids(30)
r4 = ajouter_rel(retirer_rel(R(4), "rId3"), "rId10", "../media/inv_calendrier.png")
garde = [f4[k] for k in (2, 3, 4, 5)]
garde.append(pic(ids(), "rId10", 0.6, 1.75, 12.13, 3.308, "inv_calendrier.png"))
garde.append(rect(ids(), 0.6, 5.3, 12.13, 1.4, PALER, radius=6000))
garde.append(rect(ids(), 0.6, 5.3, 0.08, 1.4, TEAL))
garde.append(text(ids(), 0.95, 5.42, 11.6, 1.2, [
    para([("Phénologie 2023-2024 : ", 13, TEAL, True),
          ("Kaw, Molokoï et Trésor, enregistrés sans interruption de novembre 2023 à novembre 2024. Seule source des saisons.", 13, INK, False)], after=8),
    para([("Relevés 2025-2026 : ", 13, GOLD_DARK, True),
          ("retour à Kaw (CDR, Patawa Ouest, Patawa Est) et à Trésor (RNRT), et un nouveau site, Mataroni ; une semaine chacun "
           "entre fin décembre et début mars, avec des dizaines de micros par relevé. Toute la saison des pluies, aucune saison sèche.", 13, INK, False)]),
], anchor="ctr"))
parts["ppt/slides/slide4.xml"] = remplacer_arbre(s4, garde).encode("utf-8")
parts["ppt/slides/_rels/slide4.xml.rels"] = r4.encode("utf-8")
NOTES[4] = ("Le calendrier, un trait par jour enregistré, avec un axe coupé : à gauche 2023-2024, à droite 2025-2026, plus "
            "étiré pour qu'on voie chaque relevé. La phénologie couvre une année entière, de novembre 2023 à novembre 2024, sur "
            "trois sites : c'est notre seule source pour les saisons. Les relevés 2025-2026 sont des coups de sonde d'une semaine, "
            "tous en saison des pluies, de fin décembre à début mars, mais avec beaucoup de micros : c'est notre source de "
            "diversité des lieux. On voit qu'ils reviennent sur deux des sites de la phénologie : CDR, puis Patawa Ouest et, en "
            "février, Patawa Est sur la ligne de Kaw ; RNRT sur celle de Trésor. Mataroni est un site nouveau. Le petit trait "
            "isolé fin décembre sur sa ligne, ce sont 25 enregistrements des 30 et 31 décembre, de 12 micros, rangés dans le "
            "relevé de Mataroni une semaine avant sa pose : ils ne sont pas marqués hors relevé, parce que le trou qui les sépare "
            "de la campagne fait six jours et que la règle demande plus de sept. Je dois encore vérifier ce que c'est : des tests, "
            "ou des fichiers restés sur les cartes. Conséquence pour la suite : un détecteur "
            "entraîné surtout sur 2026 n'aura vu qu'une saison, et c'est pour ça que la partition garde des stations 2023 à "
            "l'écart pour le tester sur toute l'année. On en reparlera avec l'annotation.")

# ---------------------------------------------------------------- diapo 5 : avant / après
s5 = S(5)
f5 = dict(shapes(s5))
f5[4] = remplacer_texte(f5[4], "Un fichier devient une ligne : le nom, l'en-tête GUANO, le dossier",
                        "Un fichier WAV devient une ligne de la table recordings, dans la base SQLite")
ids = Ids(30)
garde = [f5[k] for k in (2, 3, 4, 5)]

# Carte de gauche : le fichier brut, dans un terminal.
X, Y, W, H = 0.6, 1.75, 6.3, 4.72
garde.append(text(ids(), X, Y, W, 0.3, [para([("AVANT · LE FICHIER TEL QU'IL EST SUR LE DISQUE", 11, TEAL, True)])], anchor="t"))
Y += 0.38
garde.append(rect(ids(), X, Y, W, H - 0.38, TERM_BG, radius=2500))
garde.append(rect(ids(), X, Y, W, 0.32, TERM_BAR, radius=2500))
garde.append(rect(ids(), X, Y + 0.2, W, 0.12, TERM_BAR))
for k, c in enumerate(("FF5F57", "FEBC2E", "28C840")):
    garde.append(f'<p:sp><p:nvSpPr><p:cNvPr id="{ids()}" name="Dot"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
                 f'<a:xfrm><a:off x="{e(X + 0.15 + k * 0.2)}" y="{e(Y + 0.1)}"/><a:ext cx="{e(0.12)}" cy="{e(0.12)}"/></a:xfrm>'
                 f'<a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{c}"/></a:solidFill>'
                 f'<a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="fr-FR"/></a:p></p:txBody></p:sp>')
garde.append(text(ids(), X + 0.8, Y, W - 1.6, 0.32,
                  [para([("2LA04185_20251225_050000.wav — lecture seule", 10, "BBBBBB", False, "Consolas")], algn="ctr")], anchor="ctr"))
M = "Consolas"
T = 10.5


def ligne(*morceaux):
    return para([(t, T, c, False, M) for t, c in morceaux], line=100)


lignes = [
    ligne(("cd F:\\audio\\", TERM_TXT), ("R1_2025-12_CDR", TERM_HI), ("\\2LA04185\\Data", TERM_TXT)),
    ligne(("ls 2LA04185_20251225_050000.wav", TERM_TXT)),
    ligne(("2LA04185_20251225_050000.wav", TERM_HI)),
    ligne(("uv run python C:\\Users\\ll34272\\Grenouille\\scripts\\verification\\entete_wav.py `", TERM_TXT)),
    ligne(("    ", TERM_PROMPT), ("2LA04185_20251225_050000.wav", TERM_TXT)),
    ligne(("[fmt ] ", TERM_DIM), ("2 pistes · 48000 Hz", TERM_HI)),
    ligne(("[data] ", TERM_DIM), ("5760000 échantillons par piste = 120.0 s", TERM_HI)),
    ligne(("[guan] ", TERM_DIM), ("GUANO|Version: 1.0", TERM_TXT)),
    ligne(("Firmware Version: 4.5", TERM_DIM)),
    ligne(("Make: Wildlife Acoustics, Inc.", TERM_DIM)),
    ligne(("Model: Song Meter Mini", TERM_DIM)),
    ligne(("Serial: 2LA04185", TERM_HI)),
    ligne(("WA|Song Meter|Prefix: 2LA04185", TERM_DIM)),
    ligne(("WA|Song Meter|Audio settings: [{\"rate\":48000,\"gain\":6},{\"rate\":48000,\"gain\":18}]", TERM_TXT)),
    ligne(("Length: 120.000", TERM_DIM)),
    ligne(("Original Filename: 2LA04185_20251225_050000.wav", TERM_DIM)),
    ligne(("Timestamp: 2025-12-25 05:00:00-3:00", TERM_HI)),
    ligne(("Loc Position: 4.92289 -52.29129", TERM_TXT)),
    ligne(("Temperature Int: 23.25", TERM_DIM)),
    ligne(("Samplerate: 48000", TERM_DIM)),
]
garde.append(text(ids(), X + 0.2, Y + 0.45, W - 0.4, H - 0.38 - 0.55, lignes, anchor="t"))
garde.append(text(ids(), X, 6.55, W, 0.42, [para([
    ("En jaune, ce que l'inventaire range dans la ligne ; le gain et la position servent plus loin (diapo 6, et la vérification du site).", 10, MUTED, False)])]))

# Carte en haut à droite : la ligne de la base.
X2, W2 = 7.15, 5.58
Y2, H2 = 1.75, 2.55
garde.append(rect(ids(), X2, Y2, W2, H2, PALER, radius=3000))
garde.append(text(ids(), X2 + 0.2, Y2 + 0.12, W2 - 0.4, 0.55, [
    para([("APRÈS · UNE LIGNE DE LA TABLE ", 11, TEAL, True), ("recordings", 11, TEAL, True, M)], after=2),
    para([("Base SQLite blanci.sqlite, un seul fichier sur le PC (hors dépôt git) ; pas de CSV. "
           "Une ligne par enregistrement (diapo suivante).", 10.5, INK, False)]),
]))
cles = ["recording_id", "path", "dataset", "site", "mic_id", "start_utc", "duration_s",
        "sample_rate", "channels", "qc_flags"]
vals = [("ebe1c334f31d98b5", MUTED), ("…/RELEVE 1 CDR …/2LA04185_20251225_050000.wav", MUTED),
        ("2026", MUTED), ("CDR", TEAL), ("2LA04185", TEAL), ("2025-12-25T08:00:00Z", TEAL),
        ("120.0", TEAL), ("48000", TEAL), ("2", TEAL), ("{…} rempli au contrôle qualité (diapo 12)", MUTED)]
ty = Y2 + 0.78
garde.append(text(ids(), X2 + 0.2, ty, 1.3, 1.6, [para([(k, 9.5, MUTED, False, M)], line=100) for k in cles]))
garde.append(text(ids(), X2 + 1.5, ty, W2 - 1.65, 1.6, [para([(v, 9.5, c, c == TEAL, M)], line=100) for v, c in vals]))

# Carte en bas à droite : ce que l'inventaire en tire.
Y3 = Y2 + H2 + 0.12
H3 = 6.7 - Y3
garde.append(rect(ids(), X2, Y3, W2, H3, PALER, radius=3000))
garde.append(text(ids(), X2 + 0.2, Y3 + 0.12, W2 - 0.4, H3 - 0.2, [
    para([("CE QUE L'INVENTAIRE EN TIRE", 11, TEAL, True)], after=5),
    para([("Le micro. ", 11.5, INK, True),
          ("Selon le relevé, le même enregistreur est rangé dans « 2LA04185 » (CDR) ou « 2LA04185_GI19 » (Mataroni) : "
           "on regarde donc son numéro de série, qui, lui, ne bouge pas.", 11.5, INK, False)], after=5),
    para([("L'heure. ", 11.5, INK, True),
          ("Elle vient du GUANO, qui porte le fuseau : 5 h à −3:00 donne 08:00 UTC. Le nom du fichier ne porte pas le fuseau.", 11.5, INK, False)], after=5),
    para([("Le site. ", 11.5, INK, True),
          ("Il vient du dossier (« RELEVE 1 CDR »). Rien d'autre ne le vérifie : la position GPS du GUANO le pourrait.", 11.5, INK, False)], after=5),
    para([("Rien n'est modifié : les disques sont lus, jamais écrits.", 11.5, MUTED, False)]),
]))
parts["ppt/slides/slide5.xml"] = remplacer_arbre(s5, garde).encode("utf-8")
NOTES[5] = ("Comment un fichier devient une ligne de la base. À gauche, l'avant : le fichier tel qu'il est sur le disque, ouvert "
            "dans un terminal. C'est un vrai fichier, celui du micro 2LA04185 à CDR, le 25 décembre 2025 à 5 h ; on le retrouvera "
            "au contrôle qualité. Son nom suit la convention des Song Meter : le numéro de série, la date et l'heure locale. "
            "Le fichier WAV est fait de blocs : fmt donne le nombre de pistes et la fréquence, data le son lui-même, donc la durée, "
            "et guan un bloc de texte que les Song Meter écrivent, au format GUANO, un standard de métadonnées pour l'acoustique : "
            "numéro de série, horodatage avec son fuseau, réglages audio dont le gain de chaque piste, position GPS, température. "
            "En jaune, ce que l'inventaire garde.\n\n"
            "En haut à droite, l'après : une ligne de la table recordings. Ce n'est pas un CSV : c'est une base SQLite, une petite "
            "base de données dans un seul fichier, blanci.sqlite, sur le PC, en dehors du dépôt git ; une ligne par enregistrement. "
            "La colonne qc_flags reste vide à ce stade : c'est le contrôle qualité qui la remplit.\n\n"
            "En bas, les trois choix. Le micro : selon le relevé, le même enregistreur est rangé dans un dossier 2LA04185 à CDR et "
            "2LA04185_GI19 à Mataroni ; on regarde donc son numéro de série, qui, lui, ne bouge pas. L'heure vient du GUANO plutôt "
            "que du nom, parce qu'il porte le fuseau : 5 h à moins 3 heures, c'est 8 h UTC. Ce choix a une limite, on la verra avec "
            "Molokoï : quand le fuseau est mal réglé, c'est l'en-tête qui se trompe. Et le site vient du dossier : rien ne le "
            "vérifie, alors que la position GPS du GUANO le pourrait.")

# ---------------------------------------------------------------- diapo 5 bis : la base
s5b = S(5)
f5b = dict(shapes(s5b))
pilule = f5b[2].replace('cx="1856231"', f'cx="{e(2.45)}"')
pilule_t = remplacer_texte(f5b[3].replace('cx="1856231"', f'cx="{e(2.45)}"'),
                           "INVENTAIRE · 3/7", "INVENTAIRE · 3 bis/7")
titre = remplacer_texte(f5b[4], "Un fichier WAV devient une ligne de la table recordings, dans la base SQLite",
                        "La base vue de l'intérieur : la table recordings, une ligne par fichier")
num = remplacer_texte(f5b[5], "5", "5 bis").replace(
    '<a:off x="11277295" y="6473952"/><a:ext cx="457200"', '<a:off x="10820095" y="6473952"/><a:ext cx="914400"')
ids = Ids(30)
garde = [pilule, pilule_t, titre, num]
garde.append(rect(ids(), 0.6, 1.75, 12.13, 0.55, PALER, radius=8000))
garde.append(text(ids(), 0.85, 1.75, 11.7, 0.55, [para([
    ("Chemin : ", 13, TEAL, True),
    ("C:\\Users\\ll34272\\base_blanci\\blanci.sqlite", 13, INK, True, M),
    ("   6,7 Go · réglé par paths.db dans config/local.yaml · hors dépôt git", 12, MUTED, False)])], anchor="ctr"))
garde.append(pic(ids(), "rId10", 0.6, 2.48, 12.13, 12.13 * 750 / 2426, "inv_base.png"))
garde.append(text(ids(), 0.6, 6.33, 12.13, 0.45, [para([
    ("La même base tient aussi les annotations (tables spans, intervals, labels), les fenêtres d'encodage "
     "(windows, 11,9 millions) et les scores des têtes (scores, 27,3 millions). Les 96 349 lignes de recordings "
     "comptent 39 fichiers de test (2018, référence) en plus des 96 310 de l'inventaire.", 12, INK, False)])]))
s5b = remplacer_arbre(s5b, garde)
r5 = R(5)
note5 = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', r5).group(1)
r5b = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
       '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
       '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
       '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide27.xml"/>'
       '<Relationship Id="rId10" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/inv_base.png"/>'
       '</Relationships>')
parts["ppt/slides/slide27.xml"] = s5b.encode("utf-8")
parts["ppt/slides/_rels/slide27.xml.rels"] = r5b.encode("utf-8")
n5b = notes(parts[f"ppt/notesSlides/{note5}"].decode("utf-8"),
            "Voilà la base pour de vrai, telle qu'on la voit en l'ouvrant avec un explorateur SQLite. Elle est sur le PC, dans "
            "C:\\Users\\ll34272\\base_blanci\\blanci.sqlite, un fichier de 6,7 Go ; son chemin est réglé dans la configuration "
            "locale, et elle n'est pas dans le dépôt git, parce qu'elle est trop grosse et qu'elle change tous les jours. "
            "La table recordings a une ligne par fichier : ici les dix premiers enregistrements du micro 2LA04185 à CDR le 25 "
            "décembre au matin, toutes les demi-heures. Surlignée, la ligne de la diapo précédente. Les heures sont en UTC : "
            "8 h UTC, c'est 5 h en Guyane. La dernière colonne, qc_flags, est remplie par le contrôle qualité : on y reviendra. "
            "La même base tient aussi tout le reste : les annotations, les fenêtres découpées pour l'encodage, et les scores.")
n5b = n5b.replace('<a:t>5</a:t></a:fld>', '<a:t>6</a:t></a:fld>')
parts["ppt/notesSlides/notesSlide27.xml"] = n5b.encode("utf-8")
parts["ppt/notesSlides/_rels/notesSlide27.xml.rels"] = re.sub(
    r'Target="\.\./slides/slide\d+\.xml"', 'Target="../slides/slide27.xml"',
    parts[f"ppt/notesSlides/_rels/{note5}.rels"].decode("utf-8")).encode("utf-8")
ct = parts["[Content_Types].xml"].decode("utf-8")
ct = ct.replace("</Types>",
                '<Override PartName="/ppt/slides/slide27.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                '<Override PartName="/ppt/notesSlides/notesSlide27.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/></Types>')
parts["[Content_Types].xml"] = ct.encode("utf-8")
prels = parts["ppt/_rels/presentation.xml.rels"].decode("utf-8")
rid5 = re.search(r'Id="(rId\d+)"[^>]*Target="slides/slide5\.xml"', prels).group(1)
prels = prels.replace("</Relationships>",
                      '<Relationship Id="rId900" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide27.xml"/></Relationships>')
parts["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")
pres = parts["ppt/presentation.xml"].decode("utf-8")
m = re.search(rf'<p:sldId id="\d+" r:id="{rid5}"/>', pres)
pres = pres[:m.end()] + '<p:sldId id="400" r:id="rId900"/>' + pres[m.end():]
parts["ppt/presentation.xml"] = pres.encode("utf-8")
parts["docProps/app.xml"] = re.sub(rb"<Slides>\d+</Slides>", b"<Slides>6</Slides>", parts["docProps/app.xml"])

# ---------------------------------------------------------------- notes, médias, types
for n, t in NOTES.items():
    rels = R(n)
    cible = re.search(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', rels).group(1)
    k = f"ppt/notesSlides/{cible}"
    parts[k] = notes(parts[k].decode("utf-8"), t).encode("utf-8")

for nom in ("inv_sites.png", "inv_programmes.png", "inv_calendrier.png", "inv_base.png"):
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
