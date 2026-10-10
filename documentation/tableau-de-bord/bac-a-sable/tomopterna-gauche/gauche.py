"""C. tomopterna, grenouille de gauche de la photo d'Olivier Louguet, en sprite 8-bit (mode image).

    uv run python documentation/tableau-de-bord/bac-a-sable/tomopterna-gauche/gauche.py ETAPE
        ETAPE : ebauche | croquis | aplats | details | animation

Copie du modèle du skill sprite-grenouille. La grenouille est perchée sur une branche au-dessus
de l'eau : le corps presque de face (on voit le ventre), la tête tournée vers la droite, de
profil, vers l'autre grenouille. Sa droite est donc à gauche de l'image : les membres droits
(_d) sont devant, les gauches (_g) derrière le ventre.

Repère : celui de la scène de la session précédente, en unités de deux pixels de toile, soit
7,14 pixels de la photo de 1600 × 1081 (échelle 0,28), origine en haut à gauche de la photo. Le
sprite en couvre le rectangle de 55 × 45 unités qui commence en (48, 30) : 110 × 90 pixels. La
lecture des membres reprend celle que Léonard a validée en cinq tours (decisions.md).
"""

import colorsys
import json
import math
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
NOM = "gauche"  # sorties gauche_<etape>.json
UNITES = (55, 45)  # 110 × 90 pixels : rien ne touche les bords
# ORIGINE : en mode image, coin haut gauche du sprite dans le repère de la scène (en unités) ;
# les formes et le squelette s'écrivent alors directement en unités de la scène
ORIGINE = (48.0, 30.0)  # coin du sprite dans la scène (photo / 7,14)
ECHELLE = 2
W, H = UNITES[0] * ECHELLE, UNITES[1] * ECHELLE
# MIROIR : tête à droite. Les formes restent décrites tête à gauche ; elles sont retournées avant
# le rendu, si bien que la lumière (LUMIERE, à l'écran) garde son côté.
MIROIR = False
LUMIERE = (-0.35, -0.94)  # d'en haut, un peu de la gauche, comme sur la photo
PIVOT, PAS = (78.0, 42.0), 0.06  # cou (unités) et rotation de la tête par cran (radians)
_xs = (np.arange(W) + 0.5) / ECHELLE
X, Y = np.meshgrid(UNITES[0] - _xs if MIROIR else _xs, (np.arange(H) + 0.5) / ECHELLE)
X, Y = X + ORIGINE[0], Y + ORIGINE[1]
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
TRAME = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]

# ------------------------------------------------------------------ matières et couleurs
# une couleur à plat par matière ; les rampes du rendu détaillé en sont tirées
MATIERES = {
    "dos": "#D07A35",
    "flanc": "#C2804A",
    "ventre": "#E8DCC4",
    "gorge": "#E8DCC4",
    "membre": "#D88A40",
    "fond": "#9C5223",  # membres de l'autre côté
    "levre": "#F2CC98",
}
MOTIFS = {"bande": "#4A230E", "barre": "#7E3F17"}
# œil : iris, pupille, cercle autour du globe, matière de la paupière fermée ; la forme de la
# pupille (et sa taille, dans rendu_oeil) vient des photos
OEIL = {"iris": "#C9A05A", "pupille": "#0B0705", "cercle": "#24120A", "paupiere": "dos"}
PUPILLE = "verticale"  # fente verticale des phyllomédusines
CONTOUR, CONTOUR_CLAIR = "#24120A", "#5A2B14"
# CONTOUR_TEINTE : au rendu détaillé, le contour prend la couleur qu'il borde, assombrie, au lieu
# d'un trait sombre uniforme (choix de l'étape 3b ; C. tomopterna l'a pris, A. blanci non)
CONTOUR_TEINTE = False
SILHOUETTE = {
    "fond": "#56625E",
    "corps": "#8E9B96",
    "devant": "#C3CEC9",
    "globe": "#E6ECEA",
    "pupille": "#141918",
}
# le croquis : du papier, les plans en gris légers, des traits sombres
CROQUIS = {
    "fond": "#C9CCC8",
    "corps": "#E4E6E2",
    "devant": "#F6F7F4",
    "trait": "#5E6663",
    "globe": "#FFFFFF",
    "pupille": "#141918",
}

# ------------------------------------------------------------------ squelette
# Les repères anatomiques, en unités, relevés sur la photo (étape 1) : articulations, bouts des
# doigts et traits du visage. Les membres en sont tirés : une correction de posture se fait en
# déplaçant un point. La planche les trace et les nomme (bouton « Squelette ») ; outils.py
# lecture les pose sur la photo.
SQUELETTE = {
    # tête, de profil, tournée vers la droite
    "museau": (99.75, 41.8),
    "narine": (96.4, 38.4),
    # la narine gauche, cachée, au coin du museau : du bout du museau jusqu'à elle, puis jusqu'à
    # la nuque, le profil de la tête est fait de droites ; les yeux dépassent du crâne
    "narine_g": (98.4, 36.9),
    "oeil": (87.25, 38.0),
    # l'autre œil : toute la bosse verte sur le crâne (paupière et globe) ; on n'en voit qu'un
    # mince croissant gris, au bout à droite
    "autre_oeil": (93.4, 34.4),
    "commissure": (79.5, 46.6),
    "tympan": (81.6, 41.8),
    # tronc : le dos se voit jusqu'en (64,6 ; 50) ; sacrum et cloaque cachés, estimés
    "nuque": (82.0, 35.6),
    "sacrum": (67.0, 46.5),
    "cloaque": (69.0, 60.2),
    # bras droit (devant) : le bras, long et fin, à l'horizontale, va de l'épaule, presque à la
    # verticale de la commissure, au coude levé, en haut à gauche ; l'avant-bras, vert, descend
    # à la verticale jusqu'à la branche
    "epaule_d": (79.2, 50.6),
    "coude_d": (63.6, 50.6),  # l'avant-bras, large, penche un peu : le haut à droite
    "poignet_d": (62.6, 63.4),
    "main_d": (64.7, 65.4),
    "doigt1_d": (66.7, 62.4),  # le pouce, court, opposable, remonte le long de la branche
    "doigt2_d": (73.4, 66.0),
    "doigt3_d": (74.2, 69.2),
    "doigt4_d": (68.4, 70.6),
    # patte arrière droite (fond) : la cuisse descend en diagonale vers la gauche jusqu'au
    # genou ; le tibia, gros, revient à l'horizontale et passe derrière l'avant-bras ; deux
    # orteils s'enroulent sur la branche ; le troisième, vertical, passe sous le poignet
    "hanche_d": (64.6, 51.6),
    "genou_d": (54.0, 60.0),
    "talon_d": (62.2, 59.5),
    "tarse_d": (61.25, 61.6),  # caché derrière l'avant-bras, juste à l'intérieur de son bord
    "orteil_a_d": (57.8, 70.4),
    "orteil_b_d": (61.0, 70.8),
    "orteil_c_d": (63.8, 70.8),
    # bras gauche (2e plan) : il sort de derrière le ventre et vient vers nous, raccourci ;
    # trois doigts visibles, dont un part vers la gauche sur la branche
    "epaule_g": (88.6, 49.0),
    "coude_g": (90.2, 53.2),
    "poignet_g": (90.4, 57.6),
    "main_g": (90.4, 58.8),
    "doigt1_g": (83.2, 64.9),
    "doigt3_g": (88.6, 66.4),
    "doigt4_g": (91.6, 66.0),
    # patte arrière gauche (3e plan, à l'ombre) : genou en haut, sous la mâchoire ; la cuisse,
    # large, part du genou en diagonale vers le bas et la gauche, passe derrière le bras et se
    # voit entre le bras et le ventre ; le tibia descend à la verticale jusqu'à la branche
    "hanche_g": (85.6, 52.6),
    "genou_g": (92.4, 49.0),
    "talon_g": (93.8, 56.4),
    "tarse_g": (94.0, 58.4),  # l'orteil b en part à la verticale, parallèle au doigt voisin
    "orteil_a_g": (97.4, 60.4),
    "orteil_b_g": (94.0, 64.8),  # son disque touche celui du doigt voisin
}
MEMBRES = {
    "patte_g": (
        ["hanche_g", "genou_g", "talon_g", "tarse_g"],
        "fond",
        ["orteil_a_g", "orteil_b_g"],
    ),
    "bras_g": (
        ["epaule_g", "coude_g", "poignet_g", "main_g"],
        "corps",
        ["doigt1_g", "doigt3_g", "doigt4_g"],
    ),
    "patte_d": (
        ["hanche_d", "genou_d", "talon_d", "tarse_d"],
        "fond",
        ["orteil_a_d", "orteil_b_d", "orteil_c_d"],
    ),
    "bras_d": (
        ["epaule_d", "coude_d", "poignet_d", "main_d"],
        "devant",
        ["doigt1_d", "doigt2_d", "doigt3_d", "doigt4_d"],
    ),
}
AXES = [
    ["museau", "commissure"],
    ["museau", "narine_g", "nuque", "sacrum", "cloaque"],
    ["oeil", "autre_oeil"],
]


# ------------------------------------------------------------------ outils de forme
def catmull(pts, n=10):
    """Courbe fermée lisse passant par les points (Catmull-Rom)."""
    out, m = [], len(pts)
    for i in range(m):
        p0, p1, p2, p3 = (np.array(pts[(i + k) % m], float) for k in (-1, 0, 1, 2))
        for k in range(n):
            t = k / n
            out.append(
                0.5
                * (
                    2 * p1
                    + (p2 - p0) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                    + (3 * p1 - p0 - 3 * p2 + p3) * t**3
                )
            )
    return out


def dans(poly, x, y):
    """Points (x, y) à l'intérieur du polygone (pair-impair) : tracer des polygones simples."""
    p = np.array(poly)
    x1, y1 = p[:, 0][:, None, None], p[:, 1][:, None, None]
    x2, y2 = np.roll(p[:, 0], 1)[:, None, None], np.roll(p[:, 1], 1)[:, None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        coupe = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return np.logical_xor.reduce(coupe, axis=0)


def ellipse(cx, cy, rx, ry, x, y, rot=0.0):
    dx, dy = x - cx, y - cy
    c, s = math.cos(rot), math.sin(rot)
    return ((dx * c + dy * s) / rx) ** 2 + ((-dx * s + dy * c) / ry) ** 2 <= 1


def membre(pts, rayons, x, y):
    """Un membre en formes simples : segments de rayon variable, d'articulation en articulation."""
    m = np.zeros(x.shape, bool)
    for k in range(len(pts) - 1):
        a, b, ra, rb = (
            np.array(pts[k], float),
            np.array(pts[k + 1], float),
            rayons[k],
            rayons[k + 1],
        )
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        m |= np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]) <= ra + (rb - ra) * t
    return m


def os_(noms, rayons, x=None, y=None):
    """Un membre tiré du squelette : membre() sur les repères nommés."""
    return membre(
        [SQUELETTE[n] for n in noms], rayons, X if x is None else x, Y if y is None else y
    )


def forme(nom, masque, matiere, plan="corps", cernee=False, volume=None):
    """plan : fond (membres de l'autre côté), corps, devant (membres proches) ; cernee : un
    contour la sépare de ce qu'elle recouvre ; volume : la forme dont elle prend le relief
    (une matière dessinée dans le corps, comme le flanc, prend celui du corps)."""
    return {
        "nom": nom,
        "masque": masque,
        "matiere": matiere,
        "plan": plan,
        "cernee": cernee,
        "volume": volume or nom,
    }


def tete_tournee(tete):
    """Coordonnées vues par la tête, tournée de tete × PAS autour du cou ; l'effet s'estompe
    vers le corps (rien ne bouge en arrière du pivot). Une tête décrite tournée vers la droite
    (en mode image, sans MIROIR : museau à droite de la nuque) tourne dans l'autre sens."""
    sens = -1 if SQUELETTE and SQUELETTE["museau"][0] > SQUELETTE["nuque"][0] else 1
    poids = np.clip((sens * (PIVOT[0] - X) + 3) / 6, 0, 1)
    a = -sens * tete * PAS * poids
    dx, dy = X - PIVOT[0], Y - PIVOT[1]
    return PIVOT[0] + np.cos(a) * dx - np.sin(a) * dy, PIVOT[1] + np.sin(a) * dx + np.cos(a) * dy


# ------------------------------------------------------------------ la grenouille
def px(*pts):
    """Points relevés en pixels du sprite, sur la photo quadrillée au pixel, en unités de la
    scène : le croquis se relève au pixel près."""
    return [(ORIGINE[0] + i / ECHELLE, ORIGINE[1] + j / ECHELLE) for i, j in pts]


def courbe(pts, n=8):
    """Courbe ouverte lisse (Catmull-Rom) par les points, extrémités comprises : un galbe au
    milieu d'un contour fait aussi de droites."""
    pts = [pts[0], *pts, pts[-1]]
    out = []
    for k in range(1, len(pts) - 2):
        p0, p1, p2, p3 = (np.array(pts[k + d], float) for d in (-1, 0, 1, 2))
        for i in range(n):
            t = i / n
            q = 0.5 * (
                2 * p1
                + (p2 - p0) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                + (3 * p1 - p0 - 3 * p2 + p3) * t**3
            )
            out.append((q[0], q[1]))
    return out + [tuple(pts[-2])]


def fuseau(a, b, rayons, milieu=0.5, decale=(0.0, 0.0)):
    """Un segment du squelette, de a à b, en fuseau : trois rayons (a, milieu, b) ; le milieu,
    à la fraction milieu du segment, peut être décalé (en unités) pour bomber un côté."""
    (xa, ya), (xb, yb) = SQUELETTE[a], SQUELETTE[b]
    m = (xa + (xb - xa) * milieu + decale[0], ya + (yb - ya) * milieu + decale[1])
    return membre([(xa, ya), m, (xb, yb)], rayons, X, Y)


# un disque en boule, de 6 pixels de diamètre (doigts de C. tomopterna à l'échelle 0,28)
DISQUE = [".####.", "######", "######", "######", "######", ".####."]


def doigt(base, bout, largeur=3, via=()):
    """Un doigt posé au pixel près, de la base au bout (repères du squelette), en passant par
    les points via (unités : l'éventail des métatarses, caché sous un autre membre) : un trait
    de largeur pixels, en escalier (chaque pixel touche le suivant par un côté, sinon le
    contour le coupe), et son disque en boule centré sur le bout. Un masque par doigt."""
    m = np.zeros((H, W), bool)

    def pixel(x, y):
        x -= ORIGINE[0]
        return ((UNITES[0] - x) if MIROIR else x) * ECHELLE, (y - ORIGINE[1]) * ECHELLE

    def poser(i, j):
        if 0 <= j < H and 0 <= i < W:
            m[j, i] = True

    pts = [pixel(*p) for p in (SQUELETTE[base], *via, SQUELETTE[bout])]
    prec = None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        debout = abs(y1 - y0) >= abs(x1 - x0)  # le trait s'épaissit en travers de son sens
        n = int(max(abs(x1 - x0), abs(y1 - y0))) * 2 + 1
        for k in range(n + 1):
            i, j = math.floor(x0 + (x1 - x0) * k / n), math.floor(y0 + (y1 - y0) * k / n)
            cases = [(i, j)]
            if prec and prec[0] != i and prec[1] != j:
                cases.append((i, prec[1]))  # la marche de l'escalier
            for a, b in cases:
                for e in range(largeur):
                    d = e - largeur // 2
                    poser(a + d, b) if debout else poser(a, b + d)
            prec = (i, j)
    x1, y1 = pts[-1]
    bi, bj = round(x1) - 3, round(y1) - 3
    for dj, rang in enumerate(DISQUE):
        for di, c in enumerate(rang):
            if c == "#":
                poser(bi + di, bj + dj)
    return m


def cernes(nom, base, bouts, plan, via=None):
    """Une forme cernée par doigt, pour que deux disques voisins restent séparés ; via : pour
    chaque doigt, ses points de passage (voir doigt)."""
    via = via or [()] * len(bouts)
    return [
        forme(f"{nom}{k}", doigt(base, b, via=v), "membre", plan, cernee=True)
        for k, (b, v) in enumerate(zip(bouts, via, strict=True))
    ]


def formes(gorge=0.0, souffle=0.0, cligne=False, tete=0):
    """Étape 2, croquis : les contours relevés sur la photo, quadrillée au pixel du sprite.
    Le profil de la tête, le dos et la mâchoire en droites ; le ventre en galbe ; les membres
    en fuseaux sur le squelette validé à l'ébauche ; les doigts posés au pixel près."""
    S = SQUELETTE
    xt, yt = tete_tournee(tete)
    s, g = souffle * 2, gorge * 2  # en pixels
    corps = dans(
        [
            # le profil de la tête : deux droites, du bout du museau à la narine gauche, puis
            # jusqu'à la nuque ; le crâne ne suit pas la courbure des yeux, qui en dépassent
            S["museau"],
            S["narine_g"],
            S["nuque"],
            # le dos, relevé colonne par colonne : trois droites, en pente douce, puis de plus
            # en plus raide jusqu'au postérieur, caché derrière le coude droit
            *px((64.5, 11.0), (46.5, 20.0), (40.5, 25.0), (34.5, 36.0), (33.0, 38.8)),
            *px((33.2, 41.0), (33.2, 55.2)),
            # le ventre, en galbe : il s'arrête au-dessus de la branche et passe derrière les
            # membres gauches
            *courbe(
                px(
                    (33.2, 55.2),
                    (34.8, 58.6),
                    (38.0, 60.6),
                    (48.0, 61.6 + s),
                    (57.0, 61.2 + s),
                    (64.0, 59.2 + s * 0.8),
                    (70.0, 56.8),
                    (74.4, 53.6),
                )
            ),
            # la gorge, puis le menton : son bord est coupé plus bas, au pixel, sous la lèvre
            *px((74.8, 43.2), (77.2, 40.0), (80.8, 36.8 + g * 0.3), (83.2, 34.0 + g * 0.6)),
            *px((84.8, 30.0 + g * 0.4), (94.0, 28.8), (103.5, 26.4)),
        ],
        xt,
        yt,
    )
    # la lèvre : une droite de la commissure au bout du museau ; sous elle, au menton, une seule
    # rangée de pixels, puis le contour (coupe au pixel près, colonne par colonne)
    (mx, my), (cx, cy) = S["museau"], S["commissure"]
    ligne = cy + (xt - cx) * (my - cy) / (mx - cx)
    corps &= ~((xt > 90.4) & (yt > ligne + 0.75))
    ox, oy = S["oeil"]
    corps |= ellipse(ox, oy, 4.0, 4.0, xt, yt) & (yt < oy)  # la paupière : l'œil dépasse
    autre = ellipse(*S["autre_oeil"], 3.3, 1.95, xt, yt)  # l'autre œil : toute la bosse
    corps |= autre
    # membres : des fuseaux sur le squelette ; rayons en unités = largeur en pixels / 4
    patte_g = fuseau("hanche_g", "genou_g", [2.0, 2.2, 1.9])  # la cuisse, large
    patte_g |= os_(["genou_g", "talon_g", "tarse_g"], [1.9, 1.6, 1.2])  # tibia vertical
    bras_g = os_(MEMBRES["bras_g"][0], [1.2, 1.35, 1.3, 1.1])
    # la cuisse droite en fuseau ; elle comble le coin sous le coude
    cuisse_d = fuseau("hanche_d", "genou_d", [1.8, 2.0, 1.5], decale=(-0.2, -0.25))
    # le tibia : dessus presque droit, mollet bombé dessous
    tibia_d = fuseau("genou_d", "talon_d", [1.95, 2.25, 2.0], decale=(0.0, 0.3))
    pied_d = os_(["talon_d", "tarse_d"], [1.2, 1.0])
    # le bras droit, d'une seule forme (un coude n'a pas de frontière) : le bras, attaché en
    # arrondi à l'épaule, fin jusqu'au coude ; l'avant-bras, large, un peu renflé sous le coude
    bras_d = fuseau("epaule_d", "coude_d", [1.5, 0.95, 1.05])
    bras_d |= os_(["coude_d", "poignet_d", "main_d"], [1.85, 1.55, 1.2])
    bras_d |= fuseau("coude_d", "poignet_d", [1.85, 1.95, 1.55], milieu=0.4)
    # les orteils droits partent du tarse, caché sous l'avant-bras, et s'écartent en éventail
    # (les métatarses) pour sortir de sous lui à 4 pixels l'un de l'autre
    eventail_d = [px((22.5, 71.0)), px((26.5, 71.5)), px((30.5, 71.5))]
    f = [
        forme("patte_g", patte_g, "membre", "fond", cernee=True),
        *cernes("orteil_g", "tarse_g", MEMBRES["patte_g"][2], "fond"),
        forme("bras_g", bras_g, "membre", "corps", cernee=True),
        *cernes("doigt_g", "main_g", MEMBRES["bras_g"][2], "corps"),
        forme("cuisse_d", cuisse_d, "membre", "fond", cernee=True),
        forme("pied_d", pied_d, "membre", "fond", cernee=True),
        *cernes("orteil_d", "tarse_d", MEMBRES["patte_d"][2], "fond", eventail_d),
        forme("tibia_d", tibia_d, "membre", "fond", cernee=True),
        forme("corps", corps, "dos", cernee=True),
        forme("bras_d", bras_d, "membre", "devant", cernee=True),
        # l'épaule : le bras s'y attache en arrondi, sans trait ; il se fond dans la poitrine (la
        # peau de la poitrine recouvre son contour)
        forme(
            "epaule_d",
            ellipse(*S["epaule_d"], 2.2, 2.2, X, Y) & dilater(bras_d) & ~bras_d,
            "dos",
            volume="corps",
        ),
        *cernes("doigt_d", "main_d", MEMBRES["bras_d"][2], "devant"),
    ]
    oeil = {"x": ox, "y": oy, "r": 3.4, "ferme": cligne}
    oeil["masque"] = (xt - ox) ** 2 + (yt - oy) ** 2 <= oeil["r"] ** 2
    oeil["u"], oeil["v"] = (xt - ox) / oeil["r"], (yt - oy) / oeil["r"]
    ax, ay = S["autre_oeil"]
    oeil["autre"] = autre & ellipse(ax + 3.9, ay + 0.2, 1.5, 1.7, xt, yt)  # croissant gris
    # traits du visage : la lèvre (un pixel par colonne, sur la droite) ; la narine ; le tympan ;
    # le pli sous l'autre œil et le bord de son croissant de globe ; sous le globe, à un pixel,
    # le pli de la paupière inférieure
    tx, ty = S["tympan"]
    rt = np.hypot(xt - tx, yt - ty)
    pli = ellipse(ax, ay, 3.3, 1.95, xt, yt) & ~ellipse(ax, ay - 0.5, 3.3, 1.95, xt, yt)
    autour = dilater(oeil["masque"])
    traits = {
        "bouche": corps & (xt > cx) & (xt < mx) & (np.floor(2 * yt) == np.floor(2 * ligne)),
        "pli_autre_oeil": corps & pli & (yt > ay) & (xt < ax + 2.6),
        "croissant": autre & dilater(oeil["autre"]) & ~oeil["autre"],
        "narine": (np.abs(xt - S["narine"][0]) < 0.5) & (np.abs(yt - S["narine"][1]) < 0.5),
        "tympan": (rt > 0.9) & (rt < 1.4),
        "pli_oeil": dilater(autour) & ~autour & (yt > oy + 1.7) & (np.abs(xt - ox) < 2.6),
    }
    motifs = {}
    return f, traits, motifs, oeil


# ------------------------------------------------------------------ rendus
def etiqueter(f):
    """Matière, plan et numéro de forme de chaque pixel (le plus proche l'emporte)."""
    mat, plan, num = np.full((H, W), "", object), np.full((H, W), "", object), np.full((H, W), -1)
    for k, fo in enumerate(f):
        m = fo["masque"]
        mat[m], plan[m], num[m] = fo["matiere"], fo["plan"], k
    return mat, plan, num


def contour(img, f, num):
    """Contour sombre autour de la silhouette, et autour des formes cernées là où elles
    recouvrent une forme plus lointaine."""
    plein = num >= 0
    v = np.pad(plein, 1)
    voisin = v[:-2, 1:-1] | v[2:, 1:-1] | v[1:-1, :-2] | v[1:-1, 2:]
    img[~plein & voisin] = CONTOUR
    for k, fo in enumerate(f):
        if not fo["cernee"]:
            continue
        mm = np.pad(fo["masque"] & (num == k), 1)
        autour = mm[:-2, 1:-1] | mm[2:, 1:-1] | mm[1:-1, :-2] | mm[1:-1, 2:]
        img[autour & plein & (num < k) & (num >= 0)] = CONTOUR
    return miettes(img)


def miettes(img, n=4, bord=CONTOUR):
    """Les groupes de moins de n pixels isolés par les contours ou les traits de couleur bord
    (voisins par un côté) prennent cette couleur : sinon un pixel flotte, seul, entre deux
    traits (entre la lèvre et le bord du menton, par exemple)."""
    plein = (img != "") & (img != CONTOUR) & (img != bord)
    vus = np.zeros(plein.shape, bool)
    for j0, i0 in zip(*np.nonzero(plein), strict=False):
        if vus[j0, i0]:
            continue
        pile, groupe = [(j0, i0)], []
        vus[j0, i0] = True
        while pile:
            j, i = pile.pop()
            groupe.append((j, i))
            for v in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if 0 <= v[0] < H and 0 <= v[1] < W and plein[v] and not vus[v]:
                    vus[v] = True
                    pile.append(v)
        if len(groupe) < n:
            for p in groupe:
                img[p] = bord
    return img


def dilater(m):
    """Le masque grossi d'un pixel (voisins par un côté)."""
    v = np.pad(m, 1)
    return m | v[:-2, 1:-1] | v[2:, 1:-1] | v[1:-1, :-2] | v[1:-1, 2:]


def bord(m):
    """Les pixels du masque qui touchent l'extérieur par un côté : un trait continu."""
    return m & dilater(~m)[: m.shape[0], : m.shape[1]]


def rendu_oeil(img, oeil, couleurs, ferme_couleur):
    if oeil is None:  # un support (branche, feuille) n'a pas d'œil
        return img
    m, u, v = oeil["masque"], oeil["u"], oeil["v"]
    if oeil.get("autre") is not None and not oeil["ferme"]:
        # l'autre œil est une bosse de peau ; on n'en voit au plus qu'un mince croissant de globe
        img[oeil["autre"] & (img != "") & (img != CONTOUR)] = couleurs["iris"]
    if oeil["ferme"]:
        img[m] = ferme_couleur
        img[m & (np.abs(v - 0.15 * u * u) < 0.18)] = CONTOUR
        return img
    img[m] = couleurs["iris"]
    img[bord(m)] = couleurs.get("cercle", CONTOUR)  # un cercle continu d'un pixel
    p = {
        "ronde": u * u + v * v < 0.3,
        "horizontale": (u / 0.75) ** 2 + (v / 0.32) ** 2 < 1,
        # une fente en amande : 3 pixels au milieu, pointue aux deux bouts
        "verticale": np.abs(u) < 0.26 * (1 - (v / 0.82) ** 2),
    }[PUPILLE]
    img[m & p] = couleurs["pupille"]
    return img


def tracer(img, traits, num, oeil, couleur=None):
    """Les traits (bouche, narine, tympan, plis) : d'une couleur donnée (ébauche, croquis) ou,
    sans couleur donnée, de la couleur du dessous assombrie (aplats, détails)."""
    hors_oeil = ~oeil["masque"] if oeil else True
    for m in traits.values():
        ici = m & (num >= 0) & hors_oeil & (img != CONTOUR)
        for j, i in zip(*np.nonzero(ici), strict=False):
            img[j, i] = couleur or assombrir(img[j, i])
    return miettes(img, bord=couleur) if couleur else img


def ebauche(f, traits, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom, coul in SILHOUETTE.items():
        img[plan == nom] = coul
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil, CONTOUR)
    return rendu_oeil(
        img,
        oeil,
        {"iris": SILHOUETTE["globe"], "pupille": SILHOUETTE["pupille"]},
        SILHOUETTE["corps"],
    )


def croquis(f, traits, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom in ("fond", "corps", "devant"):
        img[plan == nom] = CROQUIS[nom]
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil, CROQUIS["trait"])
    return rendu_oeil(
        img,
        oeil,
        {"iris": CROQUIS["globe"], "pupille": CROQUIS["pupille"], "cercle": CROQUIS["trait"]},
        CROQUIS["corps"],
    )


def aplats(f, traits, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom, coul in MATIERES.items():
        img[mat == nom] = coul
    for nom, m in motifs.items():
        img[m & (num >= 0)] = MOTIFS[nom]
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil)
    return rendu_oeil(img, oeil, OEIL, MATIERES[OEIL["paupiere"]])


def _flou(a, r):
    for axe in (0, 1):
        a = sum(np.roll(a, k, axis=axe) for k in range(-r, r + 1)) / (2 * r + 1)
    return a


def rampe(hexa, n=7):
    """n tons d'une couleur : ombres plus sombres tirant vers le bleu, lumières vers le doré."""
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


def details(f, traits, motifs, oeil):
    """Volume par forme (bords tournés vers la lumière plus clairs), rampes tramées, plaques
    de teinte, grain, œil brillant, contour clair côté lumière. Point de départ : affiner à la
    main (ombres de contact, reflets humides, épaule, pustules…)."""
    mat, plan, num = etiqueter(f)
    lx, ly = LUMIERE
    lum = np.full((H, W), 0.5)
    volumes = {fo["nom"]: fo["masque"] for fo in f}
    for k, fo in enumerate(f):
        b = _flou(volumes[fo["volume"]].astype(float), 3)
        gy, gx = np.gradient(b)
        face = gx * lx + gy * ly  # bord tourné vers la lumière : positif
        ici = num == k
        lum[ici] = (0.55 + 4.0 * face - 0.18 * ((Y - ORIGINE[1]) / UNITES[1] - 0.4))[ici]
    plaques = np.clip(np.floor(_flou(np.random.default_rng(3).random((H, W)), 4) * 6 - 2), -1, 1)
    grain = (np.random.default_rng(5).random((H, W)) - 0.5) * 0.08
    img = np.full((H, W), "", object)
    couleurs = {**MATIERES, **{"motif_" + n: c for n, c in MOTIFS.items()}}
    zone = mat.copy()
    for nom, m in motifs.items():
        zone[m & (num >= 0)] = "motif_" + nom
    for nom, coul in couleurs.items():
        ici = zone == nom
        if not ici.any():
            continue
        tons = rampe(coul)
        idx = np.clip(
            np.floor((lum + grain) * (len(tons) - 1) + TRAME * 0.9), 0, len(tons) - 1
        ).astype(int)
        for v in (-1, 0, 1):  # trois variantes de teinte, par plaques
            sous = ici & (plaques == v)
            if v:
                tons_v = rampe(_decaler(coul, 0.012 * v))
                img[sous] = np.array(tons_v, object)[idx[sous]]
            else:
                img[sous] = np.array(tons, object)[idx[sous]]
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil)
    if CONTOUR_TEINTE:
        img = teinter_contour(img)
    # contour clair du côté de la lumière, autour des membres proches
    devant = plan == "devant"
    sx, sy = (-1 if LUMIERE[0] < 0 else 1), (-1 if LUMIERE[1] < 0 else 1)
    cote_lumiere = np.roll(devant, sy, axis=0) | np.roll(devant, sx, axis=1)
    img[(img == CONTOUR) & (num < 0) & cote_lumiere] = CONTOUR_CLAIR
    img = rendu_oeil(img, oeil, OEIL, rampe(MATIERES[OEIL["paupiere"]])[4])
    if oeil and not oeil["ferme"]:  # reflet blanc, petit reflet bleuté, bas du globe
        u, v = oeil["u"], oeil["v"]
        m = oeil["masque"]
        img[m & ((u + 0.35) ** 2 + (v + 0.35) ** 2 < 0.05)] = "#FFFFFF"
        img[m & ((u - 0.3) ** 2 + (v - 0.4) ** 2 < 0.03)] = "#9FB4C0"
    return img


def _decaler(hexa, dh):
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    return hexa_de(*colorsys.hls_to_rgb((h + dh) % 1, clarte, s))


def hexa_de(r, g, b):
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def assombrir(hexa, f=0.55):
    """La même teinte, plus sombre et un peu plus saturée : traits et contours teintés."""
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    return hexa_de(*colorsys.hls_to_rgb(h, clarte * f, min(s * 1.15, 1)))


TEINTES = set()  # couleurs des contours teintés : coder() les range avec le contour


def teinter_contour(img):
    """Chaque pixel de contour prend la couleur, assombrie, du premier voisin qu'il borde."""
    out = img.copy()
    for j, i in zip(*np.nonzero(img == CONTOUR), strict=False):
        for v in ((j, i + 1), (j, i - 1), (j + 1, i), (j - 1, i)):
            if 0 <= v[0] < H and 0 <= v[1] < W and img[v] not in ("", CONTOUR, CONTOUR_CLAIR):
                out[j, i] = assombrir(img[v], 0.4)
                TEINTES.add(out[j, i])
                break
    return out


RENDUS = {"ebauche": ebauche, "croquis": croquis, "aplats": aplats, "details": details}


# ------------------------------------------------------------------ sortie
def coder(images):
    """Images de couleurs hexa → palette de lettres (K et k réservés au contour) et plages."""
    couleurs = sorted(
        {c for im in images.values() for c in im.ravel() if c} - {CONTOUR, CONTOUR_CLAIR}
    )
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    lettres += [chr(0x100 + i) for i in range(max(0, len(couleurs) - len(lettres)))]
    palette = {"K": CONTOUR, "k": CONTOUR_CLAIR, **dict(zip(lettres, couleurs, strict=False))}
    # lettres du contour (K, k et les contours teintés) : outils.py controle les ignore
    contour_lettres = "Kk" + "".join(lt for lt, c in palette.items() if c in TEINTES)
    code = {c: lettre for lettre, c in palette.items()} | {"": "."}
    out = {}
    for cle, im in images.items():
        rangs = []
        for rang in im:
            s, prec, n = "", None, 0
            for c in rang:
                ch = code[c]
                if ch == prec:
                    n += 1
                else:
                    s += f"{prec}{n}" if prec else ""
                    prec, n = ch, 1
            rangs.append(s + f"{prec}{n}")
        out[cle] = rangs
    return {
        "palette": palette,
        "contour": contour_lettres,
        "largeur": W,
        "hauteur": H,
        "images": out,
    }


def squelette():
    """Les repères et les os en pixels de la toile, pour la planche et outils.py lecture. Avec
    MIROIR, l'image est retournée : les côtés de l'animal s'échangent (_g ↔ _d)."""

    def px(nom):
        x, y = SQUELETTE[nom][0] - ORIGINE[0], SQUELETTE[nom][1] - ORIGINE[1]
        return [round((UNITES[0] - x if MIROIR else x) * ECHELLE, 1), round(y * ECHELLE, 1)]

    def cote(nom):
        if MIROIR and nom[-2:] in ("_g", "_d"):
            return nom[:-2] + ("_d" if nom.endswith("_g") else "_g")
        return nom

    plans = {n: "corps" for n in SQUELETTE}
    os_liste = [[a, b, "corps"] for axe in AXES for a, b in zip(axe, axe[1:], strict=False)]
    for chaine, plan, bouts in MEMBRES.values():
        for n in chaine + bouts:
            plans[n] = plan
        os_liste += [[a, b, plan] for a, b in zip(chaine, chaine[1:], strict=False)]
        os_liste += [[chaine[-1], b, plan] for b in bouts]
    return {
        "reperes": {cote(n): px(n) + [plans[n]] for n in SQUELETTE},
        "os": [[cote(a), cote(b), p] for a, b, p in os_liste],
    }


def produire(etape):
    if etape == "animation":
        images = {}
        for g in (0, 1):
            for s in (0, 1):
                for c in (0, 1):
                    for t in (-1, 0, 1):
                        images[f"{g}{s}{c}{t + 1}"] = details(*formes(g * 0.9, s * 0.7, bool(c), t))
        return coder(images)
    sprite = coder({"0001": RENDUS[etape](*formes())})
    if etape in ("ebauche", "croquis") and SQUELETTE:
        # la planche trace le squelette ; visible d'emblée à l'ébauche, sur demande au croquis
        sprite |= squelette() | {"squelette_visible": etape == "ebauche"}
    return sprite


if __name__ == "__main__":
    etape = sys.argv[1] if len(sys.argv) > 1 else "ebauche"
    sortie = Path(sys.argv[2]) if len(sys.argv) > 2 else ICI / f"{NOM}_{etape}.json"
    sprite = produire(etape)
    sortie.write_text(json.dumps(sprite, separators=(",", ":")), encoding="utf-8")
    print(f"{sortie.name} : {len(sprite['images'])} image(s), {len(sprite['palette'])} couleurs")
