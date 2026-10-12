"""La branche entière de la scène (C. tomopterna, photo d'Olivier Louguet), et son ménisque.

    python3 documentation/tableau-de-bord/bac-a-sable/tomopterna/branche.py [apercu.png]

La branche d'un bord à l'autre de la photo, sous les deux grenouilles validées. Ses bords
suivent la ligne de la session du 8/10/2026 (celle du morceau de la grenouille de droite, avec
sa bosse) et passent en douceur à celle du morceau de la grenouille de gauche, dans son cadre :
les grenouilles s'y posent comme sur leurs morceaux.

Sa matière, relevée sur la photo réduite à la toile, pixel par pixel (ECORCE_RELEVE) : en plus
du lichen gris-bleu, des plaques d'écorce noire et luisante, des déchirures où l'écorce
arrachée laisse voir un bois plus clair, veiné le long de la branche, et des éclats : la
branche est mouillée, elle brille (Léonard : « elle mérite plus de détails »). Chaque matière
a sa rampe, éclairée comme les morceaux (lumière d'en haut à gauche) ; le dessus, mouillé,
prend un reflet bleuté. La page fait scintiller quelques éclats.

Le ménisque : à gauche de la photo, la branche touche l'eau, et l'eau monte s'y coller
(tension superficielle). C'est un miroir courbe, pas un voile : ses couleurs sont relevées sur
la photo, colonne par colonne (sombre contre l'écorce, bleu ardoise, un liseré clair, la ligne
sombre de l'eau plate, qu'on ne garde pas). Il va de la branche à la ligne d'eau, son liseré
clair contre celui de son reflet : les deux ne font qu'un, et la branche et son reflet se
rejoignent presque ; court (40 pixels), il finit en coin mouillé, l'eau montant encore un peu
le long de la branche et de la surface, l'air entrant en arrondi entre les deux. C'est un
sprite à part : la page le pose sur l'eau et le reflète.
"""

import colorsys
import re
import sys

import numpy as np

ECHELLE = 2  # pixels de toile par unité de la scène
X0, Y0, W, H = 0, 40, 448, 124  # le cadre de la branche sur la toile de 448 × 303
_xs = (X0 + np.arange(W) + 0.5) / ECHELLE
X, Y = np.meshgrid(_xs, (Y0 + np.arange(H) + 0.5) / ECHELLE)
LUMIERE = (-0.6, -0.8)  # celle des morceaux
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
TRAME = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]
CONTOUR = "#24120A"
# les matières : écorce (e), écorce noire (n), lichen (l), bois mis à nu (b), éclats (g) ; la
# bosse sous la main droite de la grenouille de droite, sombre et violacée
MATIERES = {
    "e": "#42322C",  # écorce brune, mouillée : un peu plus sombre et franche que sèche
    "n": "#1D1922",  # écorce noire, luisante, bleutée
    "l": "#8E8D94",  # lichen gris argenté (celui des morceaux)
    "b": "#7C5E45",  # bois clair, sous l'écorce déchirée
}
ECLATS = ("#C9D3DC", "#E9F0F5")  # éclats mouillés : bleutés, et les plus vifs, presque blancs
NOEUD = "#35262E"
BOSSE = (129.6, 52.4, 4.6, 3.0)  # la petite bosse de l'écorce, sous la main droite

# la ligne médiane et la demi-épaisseur (unités), relevées sur la photo le 8/10/2026
BRANCHE = [
    (-2, 74.9),
    (10, 74.4),
    (20, 73.1),
    (30, 72.5),
    (40, 71.25),
    (50, 69.65),
    (60, 67.9),
    (72, 65.2),
    (84, 62.4),
    (98, 60.2),
    (112, 58.15),
    (126, 56.0),
    (140, 53.95),
    (154, 52.0),
    (168, 50.2),
    (182, 48.4),
    (196, 46.7),
    (210, 44.9),
    (226, 43.0),
]
EPAISSEUR = [
    (-2, 1.9),
    (10, 2.1),
    (20, 2.4),
    (30, 2.9),
    (40, 3.3),
    (50, 3.6),
    (84, 3.5),
    (112, 3.4),
    (140, 3.4),
    (168, 3.3),
    (196, 3.0),
    (226, 3.3),
]


# la matière relevée sur la photo, par rangée de la toile : (première colonne, plages « lettre +
# longueur »). Plus sombre que 41 sur 255 : écorce noire ; gris-bleu clair : lichen ; brun
# plus clair et chaud : bois ; un sous-pixel très clair et bleuté sur le dessus : éclat ; le
# reste, et ce que les grenouilles cachent sur la photo : écorce. Puis lissé (à chaque pixel, la
# matière la plus présente autour de lui, deux fois) : des plaques, pas du sel et du poivre ;
# les éclats restent où ils sont
ECORCE_RELEVE = {
    80: (443, "l5"),
    81: (436, "g1l8g2e1"),
    82: (428, "e2l7g1l1g1l5g2e1"),
    83: (420, "g2l3e1g1e3l4g1l3e2l5e3"),
    84: (413, "l10g1e6l9e1l5e3"),
    85: (406, "l2g1l1g1l7g2l4g1e3g1l9e4l3e2b1"),
    86: (398, "g1e1g1e4l2g1l7g1l9g1l6e2l5e4g1e2b2"),
    87: (391, "e4g1l1g1e6g2l16g1l8e3g1l2b3e2g1e3b2"),
    88: (382, "e1g1e3n1e7l3g1e4g1l1g1l5g1l18b1e4b5e7b1"),
    89: (373, "e1g2e3g2e6n1e2g1e2l4g2e1g1e3l2g3l1g3l5e1g2b4l3b4e3b5e8"),
    90: (364, "e1g3l3g2l1e13n1l2e2l6e1g1e3l7g2e1l1e1n2e4b12e3b1e11"),
    91: (356, "l2g1l4e5g1l5e10g1e3l2e3g1l4e1g2e1g1l2g3l4e3n3e4b11e15"),
    92: (347, "e3g4l2g1l6g1l3e4l2e16g1e1g1e2l3g1e8l6e3n3e4b11e15"),
    93: (339, "e4g2e9l7e1l6g1e1g1e3g1e10g1e3g1e10g1e6l2e13b11e11"),
    94: (330, "e4g4e1g1e2g1e11l5e3l6e2g1e3b3e11n3e31b11e2"),
    95: (322, "e6g1e4g1e19l6e2l4g1e5b7e8n5e31b3"),
    96: (314, "e2g2e5g1e9g1e2g1e16l3n3e2b4e6b10e5n4e26"),
    97: (307, "e1b4e16g2e1g1e11g2e6g1e2n5e2b5e7b10e25"),
    98: (299, "e4g3e3b3e2g1e13g1e18n4e2n6e4b3e8b7e19"),
    99: (255, "e3g3e48b1e5g1e21n3e3n8e1n5e6b4e7b7e1b2e7"),
    100: (253, "e3g1e37g1e17g1e5g1e13g1e2n4g1n11e2n4e7b3e7b6e2b2"),
    101: (252, "e4g2e25g1e2g1e15g1e5g1e9g1e16n14e5b3e7b4e6b2"),
    102: (251, "e20n4e10g2e13g1e1g1e1n1e2g1e4g1e21n12e7b4e8b3e1"),
    103: (250, "e7g1e11g3n2e24g1e2g2n3e2g3b3e10n2e9n7e11b4e4"),
    104: (250, "e4g1e19g1e4g1e1g1e16g4e1n3e4b5e1b2e35b1"),
    105: (250, "l2e2g1e20g2e23g1e3n2e4b9e28"),
    106: (244, "e3g1l4e4g1e4g1e13g1e1g1e5g1e13g2e5n1b2e4b8e21"),
    107: (237, "e12l3g1e16n1e3g1e5g1e1g1e23b3e7b4e13"),
    108: (228, "e7l1e14l1e11g1e5n3e5n2e28b2e8b3e6"),
    109: (216, "e18g2e1g2e14g1e3b3e8n4e3n4e37b1"),
    110: (208, "e18g2e4g1e1g1e11g1e1n2e2g1e4n1b2n4e4n4e3n5b2e12b2e14"),
    111: (203, "e20g1e1g3e1g2e8g1e6n2g1n2e5n9e2n4e5n3b4e11b2e6"),
    112: (198, "e46n7e1g2e3n9e1n3e9b3e13"),
    113: (192, "l1g1l1e13g2e14g1e18n8e6n8e2n2e9b4e6"),
    114: (187, "e5l3e8g1e7g2e13g1e1g2e2n1g1n2e4n11e7n4e15b4"),
    115: (182, "l1g1e3g3e2l2e1g1e3g3e2g1e10g5e9n8e2n11e25"),
    116: (176, "e4l4e12g1e9g1e14n2e5n10e2n8e20"),
    117: (171, "e10g1l1e1g1e9g1e1g2e6g1e9n10e1n12e24"),
    118: (166, "n3e4g1e6g3e3g1e3g1e4g1e12n1e5n9e2n7e22"),
    119: (160, "e4n4e9g1e1g1e2g2e4g1e7g2e10n1e38"),
    120: (155, "b2e1n3e3n4e9g1e8g1e54"),
    121: (150, "e4n1b1n6e2n3e3g1e6g1e1g3e1g1e50"),
    122: (144, "n1e8n9e3n1e11g1b4e1n3e2b3e4b1e10"),
    123: (139, "b1n6e6n11e2g1e6n2e5b4n3e3b1e4b3e4"),
    124: (134, "e2n15e1n11e7n4e5b5n1e8b3"),
    125: (128, "e8g1n26e7n4e7b3e8"),
    126: (123, "e12n28e7n3e8b4e2"),
    127: (118, "e18n10g1e3n6e2n6e18"),
    128: (112, "e5g1e21n6e5n6e2n5b2e12"),
    129: (107, "e4g1e1g1e10g1e1g2e23n3e9b2e7"),
    130: (102, "e14g1e10g1e1g1e3g1e5g2e6l1e15b3e1"),
    131: (97, "e10g3e2g1e3g1e1g1e8g1e9b2e8l1e14"),
    132: (92, "b3e5g3e1l3g1e15g1e1g1e12b1e18"),
    133: (87, "e5b3l3e1g1e1l7e1g2e14g2e25"),
    134: (83, "e9l7g1l3g1l3g1e19b2e18"),
    135: (80, "e8g1e1g1e1l9g1e4g1e7b3e9b4e13"),
    136: (75, "e7g1e1l4e1g1e2l9n1e1g1e5b1e3b4e9b3e9"),
    137: (70, "e7g1e1l10e3l5n8e3b2e3b3e11b1e5"),
    138: (64, "e5g1e2g1e4g1l11e6n10e9b1e13"),
    139: (57, "e1l4e1g1e12l11g1l1e6n10e18"),
    140: (48, "e9l5e15l7e1b1g1l1e7n6e17"),
    141: (39, "e16l6e20b5n1e3n2e4n4e13"),
    142: (33, "e13g1e1l3g1l9e9n6e5b4n3e2n2e16"),
    143: (27, "e14g1e3g1l14e5n12e4b3n4e15"),
    144: (21, "l7e6g1l2e1l21e4n14e8n2e11"),
    145: (5, "l1e6l11g2l27n7e2n16e17"),
    146: (0, "l6e5l22g1l14e3n20e2n2e15"),
    147: (0, "l7g1l29e3l4e7n13e22"),
    148: (0, "l28g1l7e5l2e12n4e22"),
    149: (0, "l23e2l10e40"),
    150: (0, "l15e11l5e33"),
    151: (0, "l1e2l11e13l4e4"),
    152: (0, "e4l10e8b2e1"),
}


def suivre(pts, x):
    return np.interp(x, [p[0] for p in pts], [p[1] for p in pts])


def lisse(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def bords(x):
    """Le dessus et le dessous de la branche (unités) : la ligne du 8/10, et dans le cadre de
    la grenouille de gauche (x de 48 à 103) les bords droits de son morceau, raccordés sur 14 et
    15 unités de part et d'autre."""
    c, e = suivre(BRANCHE, x), suivre(EPAISSEUR, x)
    w = lisse(34, 48, x) * (1 - lisse(103, 118, x))
    dessus_g = 65.8 - (x - 48.0) * 0.188
    dessous_g = 72.5 - (x - 48.0) * 0.204
    return w * dessus_g + (1 - w) * (c - e), w * dessous_g + (1 - w) * (c + e)


def flou(a, r):
    """Moyenne glissante (2r + 1)², bords prolongés (pas de bouclage d'un bord à l'autre)."""
    for axe in (0, 1):
        p = np.pad(a, [(r, r) if k == axe else (0, 0) for k in range(a.ndim)], mode="edge")
        a = sum(np.take(p, range(k, k + a.shape[axe]), axis=axe) for k in range(2 * r + 1))
        a = a / (2 * r + 1)
    return a


def hexa_de(r, g, b):
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def rampe(hexa, n=7):
    """Celle des morceaux : ombres plus sombres tirant vers le bleu, lumières vers le doré."""
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    out = []
    for k in range(n):
        d = -0.2 + 0.36 * k / (n - 1)
        cible = 0.62 if d < 0 else 0.13
        dh = ((cible - h + 0.5) % 1 - 0.5) * min(abs(d) * 0.5, 0.1)
        rr, gg, bb = colorsys.hls_to_rgb(
            (h + dh) % 1, min(max(clarte + d, 0.03), 0.97), min(s * (1 + 0.3 * -d), 1)
        )
        out.append(hexa_de(rr, gg, bb))
    return out


def decaler(hexa, dh):
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    return hexa_de(*colorsys.hls_to_rgb((h + dh) % 1, clarte, s))


def ellipse(cx, cy, rx, ry, x, y):
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def matiere():
    """Le relevé, lettre par pixel ("" hors de la branche)."""
    out = np.full((H, W), "", object)
    for y, (a, plages) in ECORCE_RELEVE.items():
        x = a
        for c, n in re.findall(r"([a-z])(\d+)", plages):
            out[y - Y0, x - X0 : x - X0 + int(n)] = c
            x += int(n)
    return out


def _plages(rang):
    out, k = [], 0
    while k < len(rang):
        j = k + 1
        while j < len(rang) and rang[j].isdigit():
            j += 1
        out.append(rang[k:j])
        k = j
    return out


def branche():
    """La branche peinte d'un bord à l'autre, matière par matière. Renvoie ses couleurs (hexa,
    "" hors d'elle) et ses éclats (les pixels que la page fait scintiller)."""
    dessus, dessous = bords(X)
    bande = (Y > dessus) & (Y < dessous)
    bosse = ellipse(*BOSSE, X, Y) & ~bande
    m = bande | bosse
    lx, ly = LUMIERE
    gy, gx = np.gradient(flou(m.astype(float), 3))
    face = -(gx * lx + gy * ly)  # bord tourné vers la lumière : positif
    # la hauteur se compte depuis le haut du cadre de la grenouille voisine, comme dans les
    # morceaux validés
    haut = np.interp(X, [103, 118], [30.0, 23.0])
    lum = 0.55 + 4.0 * face - 0.18 * ((Y - haut) / 45 - 0.4)
    t = np.clip((Y - dessus) / (dessous - dessus), 0, 1)  # 0 dessus, 1 dessous
    rng = np.random.default_rng
    plaques = np.clip(np.floor(flou(rng(3).random((H, W)), 4) * 6 - 2), -1, 1)
    grain = (rng(5).random((H, W)) - 0.5) * 0.08
    # le bois : veiné le long de la branche (des fibres claires et sombres, couchées)
    v = rng(7).random((H // 2 + 2, W // 9 + 2))
    fibres = np.repeat(np.repeat(v, 2, 0), 9, 1)[:H, :W]
    fibres = (flou(fibres, 1) - 0.5) * 0.5
    zone = matiere()
    zone[m & (zone == "")] = "e"
    zone[~m] = ""
    img = np.full((H, W), "", object)
    for c, coul in MATIERES.items():
        ici = (zone == c) | ((zone == "g") & (c == "e"))
        if c == "e":
            ici &= ~bosse
        if not ici.any():
            continue
        lu = lum + grain + (fibres if c == "b" else 0)
        lu = lu + (0.12 if c == "n" else 0) * (1 - t)  # le noir luit sur le dessus
        idx = np.clip(np.floor(lu * 6 + TRAME * 0.9), 0, 6).astype(int)
        for v_ in (-1, 0, 1):  # trois variantes de teinte, par plaques
            sous = ici & (plaques == v_)
            img[sous] = np.array(rampe(decaler(coul, 0.012 * v_) if v_ else coul), object)[
                idx[sous]
            ]
    tons_n = rampe(NOEUD)
    img[bosse] = np.array(tons_n, object)[np.clip(np.floor(lum * 6 + TRAME), 0, 6).astype(int)][
        bosse
    ]
    # mouillée : le dessus, tourné vers le ciel, prend un reflet bleuté par endroits
    ciel = m & ~bosse & (t < 0.3) & (lum > 0.62) & (rng(11).random((H, W)) < 0.35)
    for j, i in zip(*np.nonzero(ciel), strict=False):
        img[j, i] = melange(img[j, i], ECLATS[0], 0.35)
    # les éclats relevés : bleutés, et les plus éclairés presque blancs
    eclats = (zone == "g") & ~bosse
    img[eclats] = np.where(lum > 0.75, ECLATS[1], ECLATS[0])[eclats]
    v = np.pad(m, 1)
    autour = v[:-2, 1:-1] | v[2:, 1:-1] | v[1:-1, :-2] | v[1:-1, 2:]
    img[~m & autour] = CONTOUR
    return img, [[int(X0 + i), int(Y0 + j)] for j, i in zip(*np.nonzero(eclats), strict=False)]


def melange(a, b, t):
    ca = [int(a[k : k + 2], 16) for k in (1, 3, 5)]
    cb = [int(b[k : k + 2], 16) for k in (1, 3, 5)]
    return "#{:02X}{:02X}{:02X}".format(
        *(round(x + (y - x) * t) for x, y in zip(ca, cb, strict=True))
    )


# ------------------------------------------------------------------ le ménisque
# la ligne d'eau sous la branche (pixels de toile), a + b·x, relevée sur la photo : la page
# reflète tout autour d'elle (le reflet d'une rangée y tombe en round(2·(a + b·x)) − y)
AXE = (161.8, 0.0083)
COLLE, FIN = 26, 40  # plein jusqu'à x = COLLE ; son bout, creusé, finit en x = FIN
# le ménisque relevé sur la photo réduite à la toile, colonne par colonne (x = 0 à 40) : ses sept
# rangées, du contact avec la branche à l'eau plate (rangées 154 à 160 de la photo). C'est un
# miroir courbe : sombre contre l'écorce (il en reflète le dessous), bleu ardoise (le ciel),
# clair là où sa courbe renvoie la lumière, puis la ligne sombre où il rejoint l'eau plate ; il
# n'a rien d'un voile sur l'eau
MENISQUE_RELEVE = [
    "57575864687A50586D545C719197A69599A3403E3D",  # x = 0
    "59595B60657750586D575F74949AAA8588933F3B3B",  # x = 1
    "55575B5C6276535D71545E72949AA99195A0413C3D",  # x = 2
    "65666C575E73515C714E596D8D94A39C9FAA453F40",  # x = 3
    "7173775C63754F5B6E4F586D8C92A2A9ACB74C4648",  # x = 4
    "66676C595F714F5A6D4C55698A90A0ACB0BC494548",  # x = 5
    "6B6C7352586C545F734C5468868C9CB0B7C1505153",  # x = 6
    "72737A51576B5661754D556A8F95A4B0B6C1525053",  # x = 7
    "68696F5157694E596B4E586C838999A8ABB75A5458",  # x = 8
    "5354575A5F704D576D545F74828898ADB1BD676163",  # x = 9
    "393B3C5C60704D576D5761777B8294B1B5C1706C6E",  # x = 10
    "1F2120575C6C4A546A5962797A8092AFB3C1716C70",  # x = 11
    "1A1C1A4D52624852685963787F8496AFB4C2767377",  # x = 12
    "3235365157694651675A62797D8396ACB0BD5F5C60",  # x = 13
    "42464B4D556B49546A5A627A7D8498ADB0BC605C5D",  # x = 14
    "575A614B52694E586F5C647B7D8498ACAFB9595455",  # x = 15
    "5B5E644B53684E566C5D657B798093ADAFB95B5455",  # x = 16
    "46494D52596C5860735D657A7A8193ABACB75C5454",  # x = 17
    "434345596072545E725D667A7E8595AAABB3544D4A",  # x = 18
    "4342455B62744F596E687185818698A4A4AC4E4743",  # x = 19
    "454346606678535C7161697F7A7F92A6A5AD4E4742",  # x = 20
    "474448626679545B70585F76767A8D9F9DA5483F3A",  # x = 21
    "3C3B3D5C62744F576C555C72757A8D9F9FA449413B",  # x = 22
    "2625295C5F72565C72555C71767A8EA1A0A54B443D",  # x = 23
    "28262A5E5F73595E735A6076797C909E9CA14D433E",  # x = 24
    "2826285E607253586E60667C787A8F9F9CA14E4540",  # x = 25
    "2926255E61714D54695A60767C7D929F9CA14F4540",  # x = 26
    "2A29285D60724E556E595F75787B8CA09CA3564B47",  # x = 27
    "2C2A2A5B5E70505770565D74737788A29FA5594D48",  # x = 28
    "3D3C3C6164764E556E525A72707585A5A1A6594C45",  # x = 29
    "4140406367794A506A4F586F6F7484A4A0A453453D",  # x = 30
    "3534345F617053586F53596F7578899D989C463A31",  # x = 31
    "2C28276B6A785E6177555A707579899B9598473B34",  # x = 32
    "251E1B6E6C765B5D7254596F737687928B8F4B3F38",  # x = 33
    "1F18166F6E784B506451566C737686948E914B3F38",  # x = 34
    "1914125A5B65474D6253586D76788990898D4E423C",  # x = 35
    "15110F44454F464D5F5C607479798C867F854F453E",  # x = 36
    "1C181542444E42495C5F63777A798C827C824D433C",  # x = 37
    "1A151246485340485C5E627679788C817B814A403A",  # x = 38
    "2C2724474B553F485C55596D7676897E787E493F38",  # x = 39
    "1B1512484A573D455957596C7574867C777D463E36",  # x = 40
]


def ligne_eau(x):
    """La dernière rangée du ménisque à la colonne x : celle que son reflet touche (le reflet
    de cette rangée tombe sur elle ou juste dessous)."""
    return int(round(2 * (AXE[0] + AXE[1] * x))) // 2


def releve_menisque(x, t):
    """La couleur relevée du ménisque à la colonne x, à la hauteur t (0 contre la branche, 1 sur
    le liseré), un peu plus lumineuse, comme tout le décor (decor.py). Sans la ligne sombre
    de l'eau plate : le liseré touche celui du reflet, les deux ménisques n'en font qu'un."""
    r = MENISQUE_RELEVE[min(x, len(MENISQUE_RELEVE) - 1)]
    k = min(5, int(t * 6))  # la septième, la ligne sombre de l'eau plate, ne sert pas
    c = np.array([int(r[6 * k + j : 6 * k + j + 2], 16) for j in (0, 2, 4)], float)
    return tuple(int(v) for v in np.clip(c * 1.06 + 4, 0, 255))


def menisque(img):
    """Sous la branche (img, ses couleurs), l'eau qui s'y colle : colonne par colonne, de la
    branche à la ligne d'eau, aux couleurs relevées sur la photo (un miroir courbe : il reflète
    le dessous de la branche, puis le ciel, et s'éclaire sur sa courbe). Au bout (de COLLE à
    FIN), la surface est concave comme dans un coin mouillé : l'eau monte encore un peu le long
    de la branche et s'élève encore un peu de la surface, et entre les deux, l'air entre en
    arrondi (un quart de cercle) ; les deux pointes s'effilent jusqu'en x = FIN. Son reflet le
    touche : avec lui, la branche et son reflet se rejoignent presque."""
    x0, y0 = 0, 146
    w, h = FIN + 1, ligne_eau(FIN) + 1 - y0
    out = np.full((h, w), "", object)
    eclats, bout = [], None
    for i in range(w):
        bas = Y0 + np.nonzero(img[:, i - X0] != "")[0].max() + 1  # sous la branche
        fond = ligne_eau(x0 + i)
        n = fond - bas + 1
        if n <= 0:
            continue
        # le creux arrondi du bout : q = 0 (plein) à 1 (vide), en quart de cercle
        v = np.clip((i - COLLE) / (FIN - COLLE), 0, 1)
        q = np.sqrt(1 - (1 - v) ** 2) if i > COLLE else 0.0
        film = round((1 - q) * n / 2)  # l'eau le long de la branche, en haut
        pied = round((1 - q) * n / 2 + 0.49)  # l'eau qui monte de la surface, en bas
        for k in range(n):
            if q and film <= k < n - pied:
                continue
            t = k / max(n - 1, 1)
            if q and k == film - 1:  # le bord du creux, en haut : la courbe s'éclaire aussi
                t = 0.8
            out[bas + k - y0, i] = "#{:02X}{:02X}{:02X}".format(*releve_menisque(x0 + i, t))
            if k == n - 1:  # le liseré, qui scintille
                eclats.append([int(x0 + i), int(bas + k)])
                bout = [int(x0 + i), int(bas + k)]
    eclats.append(bout)  # le dernier : la pointe sur l'eau, d'où partent de petits ronds
    return out, (x0, y0), eclats


# ------------------------------------------------------------------ sortie
def coder(img):
    couleurs = sorted({c for c in img.ravel() if c} - {CONTOUR})
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    lettres += [chr(c) for c in range(0x100, 0x100 + max(0, len(couleurs) - len(lettres)))]
    palette = {"K": CONTOUR, **dict(zip(lettres, couleurs, strict=False))}
    code = {c: lettre for lettre, c in palette.items()} | {"": "."}
    rangs = []
    for rang in img:
        s, prec, n = "", None, 0
        for c in rang:
            ch = code[c]
            if ch == prec:
                n += 1
            else:
                s += f"{prec}{n}" if prec else ""
                prec, n = ch, 1
        rangs.append(s + f"{prec}{n}")
    return {
        "palette": palette,
        "contour": "K",
        "largeur": img.shape[1],
        "hauteur": img.shape[0],
        "images": {"0001": rangs},
    }


def produire():
    """{"branche": sprite, "menisque": sprite} (avec x, y : leur coin sur la toile), puis les
    pixels qui scintillent dans la page : ceux du liseré du ménisque (le dernier : sa pointe),
    et les éclats de la branche mouillée."""
    img, eclats_branche = branche()
    men, (mx, my), eclats = menisque(img)
    return (
        {
            "branche": {"x": X0, "y": Y0, **coder(img)},
            "menisque": {"x": mx, "y": my, **coder(men)},
        },
        eclats,
        eclats_branche,
    )


if __name__ == "__main__":
    sprites, *_ = produire()
    for nom, s in sprites.items():
        print(f"{nom} : {s['largeur']} × {s['hauteur']}, {len(s['palette'])} couleurs")
    if len(sys.argv) > 1:
        from PIL import Image

        toile = np.zeros((303, 448, 4), np.uint8)
        toile[..., :3] = (40, 44, 40)
        toile[..., 3] = 255
        for s in sprites.values():
            for j, rang in enumerate(s["images"]["0001"]):
                i = 0
                for p in _plages(rang):
                    c, n = p[0], int(p[1:])
                    if c != ".":
                        h = s["palette"][c]
                        toile[s["y"] + j, s["x"] + i : s["x"] + i + n, :3] = [
                            int(h[k : k + 2], 16) for k in (1, 3, 5)
                        ]
                    i += n
        Image.fromarray(toile).resize((448 * 3, 303 * 3), Image.NEAREST).save(sys.argv[1])
