"""C. tomopterna, grenouille de droite de la photo d'Olivier Louguet, en sprite 8-bit (mode image).

    python3 documentation/tableau-de-bord/bac-a-sable/tomopterna-droite/droite.py ETAPE
        ETAPE : ebauche | croquis | aplats | details | animation

Copie de gauche.py (la grenouille de gauche, validée), elle-même copie du modèle du skill
sprite-grenouille. La grenouille est perchée sur la branche, de profil, tournée vers la gauche,
vers l'autre grenouille : on voit son flanc gauche, les membres gauches (_g) devant, les droits
(_d) de l'autre côté. Point de départ : la grenouille de droite de la session du 8/10/2026,
reprise étape par étape (decisions.md).

Repère : celui de la scène, en unités de deux pixels de toile, soit 7,14 pixels de la photo de
1600 × 1081 (échelle 0,28), origine en haut à gauche de la photo. Le sprite en couvre le
rectangle de 55 × 45 unités qui commence en (118, 23) : 110 × 90 pixels. La branche est dans
branche.py, au même cadre.
"""

import colorsys
import json
import math
import sys
import unicodedata
from pathlib import Path

import numpy as np
from branche import dans_branche

ICI = Path(__file__).resolve().parent
NOM = "droite"  # sorties droite_<etape>.json
UNITES = (55, 45)  # 110 × 90 pixels : rien ne touche les bords
# ORIGINE : en mode image, coin haut gauche du sprite dans le repère de la scène (en unités) ;
# les formes et le squelette s'écrivent alors directement en unités de la scène
ORIGINE = (118.0, 23.0)  # coin du sprite dans la scène (photo / 7,14)
ECHELLE = 2
W, H = UNITES[0] * ECHELLE, UNITES[1] * ECHELLE
# MIROIR : tête à droite. Les formes restent décrites tête à gauche ; elles sont retournées avant
# le rendu, si bien que la lumière (LUMIERE, à l'écran) garde son côté.
MIROIR = False
LUMIERE = (-0.35, -0.94)  # d'en haut, un peu de la gauche, comme sur la photo
PIVOT, PAS = (138.0, 36.0), 0.06  # cou (unités) et rotation de la tête par cran (radians)
_xs = (np.arange(W) + 0.5) / ECHELLE
X, Y = np.meshgrid(UNITES[0] - _xs if MIROIR else _xs, (np.arange(H) + 0.5) / ECHELLE)
X, Y = X + ORIGINE[0], Y + ORIGINE[1]
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
TRAME = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]

# ------------------------------------------------------------------ matières et couleurs
# une couleur à plat par matière ; les rampes du rendu détaillé en sont tirées. Étape 3a :
# teintes prises à la pipette sur la photo, dans les zones éclairées, sans reflet ni ombre
MATIERES = {
    "dos": "#4F9C43",  # vert feuille : dos, tête, faces externes des membres
    "ventre": "#C4A9AA",  # poitrine et ventre, blanc rosé
    "bas_ventre": "#B9866A",  # bas du ventre, pêche orangé
    "gorge": "#B3A3B0",  # gorge, gris lilas
    "levre": "#E4DCDB",  # liseré blanc de la lèvre
    "membre": "#AE6C38",  # faces internes des membres, mains, pieds : orangé
    "pale": "#B3A4AC",  # dessus du bras droit, pâle
    "disque": "#C2884F",  # disques, plus clairs que le doigt
    "membrane": "#A9A3B8",  # paupière inférieure, lilas (œil fermé)
}
MOTIFS = {
    "barre": "#2B1E33",  # barres violet-noir des membres, mouchetures sous la lèvre
    "tache": "#6B3F50",  # marbrures violettes du ventre et du bras gauche
}
# les traits du croquis prennent la couleur qu'ils traversent, assombrie ; sauf ceux-ci, qui ont
# leur matière : chez cette grenouille, la lèvre est un liseré blanc
TRAITS_MATIERE = {"bouche": "levre"}
# œil : iris, pupille, cercle autour du globe, bord pâle de la paupière autour de lui, matière de
# la paupière fermée ; la forme de la pupille (et sa taille, dans rendu_oeil) vient des photos
OEIL = {
    "iris": "#B4B0BE",  # argenté, lavande
    "pupille": "#16141E",
    "cercle": "#2C2833",
    "bord": "#A9C99E",  # le bord de la paupière autour du globe, vert pâle
    "paupiere": "membrane",
}
PUPILLE = "verticale"  # fente verticale des phyllomédusines
# le regard : la fente décalée dans le globe, en part du rayon (négatif : vers la gauche, vers
# l'avant de cette grenouille ; Léonard : « très sensiblement vers la gauche »)
REGARD = -0.1
CONTOUR, CONTOUR_CLAIR = "#24120A", "#5A2B14"
# CONTOUR_TEINTE : au rendu détaillé, le contour prend la couleur qu'il borde, assombrie, au lieu
# d'un trait sombre uniforme (choix de l'étape 3b ; C. tomopterna l'a pris, A. blanci non)
CONTOUR_TEINTE = True  # choix de tomopterna : un contour teinté, sans trait noir
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
    # tête, de profil, tournée vers la gauche ; le museau tronqué : une droite verticale, du bout
    # du museau (au liseré de la lèvre) au coin de la narine, puis une droite jusqu'à la nuque
    "museau": (122.25, 36.5),
    "narine": (123.75, 33.15),
    "narine_g": (122.5, 32.5),  # le coin du museau, au-dessus de la narine
    "oeil": (131.15, 32.15),  # le centre du globe, très gros
    "autre_oeil": (127.5, 30.25),  # une petite bosse verte en avant de l'œil
    "commissure": (138.5, 38.4),  # la mâchoire va jusqu'au-dessus de l'épaule
    "tympan": (136.75, 34.25),
    # tronc : le dos en pente douce ; sacrum et cloaque cachés par la cuisse, estimés
    "nuque": (138, 30.65),
    "sacrum": (158, 39.5),
    "cloaque": (160, 46),
    # bras gauche (devant) : le bras, fin, à l'horizontale de l'épaule au coude ; l'avant-bras,
    # vert, épais, part du coude, juste sous le dos, sans bosse au-dessus, et descend sur la
    # branche : un membre à part, cerné, même s'il a la couleur du dos
    "epaule_g": (140, 42.25),
    "coude_g": (151.25, 40),
    "poignet_g": (149, 51.25),
    "main_g": (146.75, 53),
    "doigt1_g": (146.25, 49.25),  # quasi invisible : il monte sur la branche, derrière l'avant-bras
    "doigt2_g": (140.75, 51.65),  # le long du dessus de la branche, vers la main droite
    "doigt3_g": (139.25, 58.25),
    "doigt4_g": (144.75, 59.15),
    # bras droit : de l'autre côté, mais pas caché derrière le corps : de l'épaule, une légère
    # bosse sur la poitrine, le bras descend jusqu'au coude, pointu, bien visible sous le ventre ;
    # l'avant-bras revient vers la gauche jusqu'à la main posée sur la branche
    "epaule_d": (136.5, 43.25),
    "coude_d": (141.5, 46.25),  # déplacé par Léonard sur la planche
    "poignet_d": (131, 49.5),
    "main_d": (129.75, 50.25),
    "doigt1_d": (137.5, 50.75),  # le long de la branche vers la droite ; il touche presque doigt2_g
    "doigt2_d": (123.25, 52.3),  # le long de la branche vers la gauche
    "doigt3_d": (127.25, 56.5),
    "doigt4_d": (132.75, 55),
    # patte arrière gauche (devant) : la cuisse, à peine visible, monte à la verticale derrière
    # le tibia ; le genou en haut ; le tibia, vertical, descend sur la branche ; trois orteils
    "hanche_g": (160, 45),
    "genou_g": (161.5, 39),
    "talon_g": (162.25, 46.25),
    "tarse_g": (162, 46.75),  # remonté par Léonard : des orteils plus longs
    "orteil_a_g": (158.25, 51),
    "orteil_b_g": (161.5, 54),
    "orteil_c_g": (165.75, 52.25),
    # patte arrière droite (fond) : la cuisse et le genou cachés ; le tibia descend derrière la
    # branche puis derrière le pied jusqu'au talon ; le tarse est le point le plus bas, d'où les
    # trois orteils partent en éventail : deux remontent derrière la branche, le troisième part à
    # l'horizontale vers la droite, dans le vide (correction de Léonard)
    "hanche_d": (158, 46),
    "genou_d": (154.5, 52),
    "talon_d": (149.75, 60.25),
    "tarse_d": (150.75, 61.5),
    "orteil_a_d": (149, 55.5),
    "orteil_b_d": (154, 55.5),
    "orteil_c_d": (156.75, 60.5),
}
MEMBRES = {
    "patte_d": (
        ["hanche_d", "genou_d", "talon_d", "tarse_d"],
        "fond",
        ["orteil_a_d", "orteil_b_d", "orteil_c_d"],
    ),
    "bras_d": (
        ["epaule_d", "coude_d", "poignet_d", "main_d"],
        "fond",
        ["doigt1_d", "doigt2_d", "doigt3_d", "doigt4_d"],
    ),
    "patte_g": (
        ["hanche_g", "genou_g", "talon_g", "tarse_g"],
        "devant",
        ["orteil_a_g", "orteil_b_g", "orteil_c_g"],
    ),
    "bras_g": (
        ["epaule_g", "coude_g", "poignet_g", "main_g"],
        "devant",
        ["doigt1_g", "doigt2_g", "doigt3_g", "doigt4_g"],
    ),
}
# les mains, relevées sur la photo, en pixels du sprite (les conventions de la grenouille de
# gauche) : la paume, d'où part chaque doigt, de sa base, sur son bord ; les doigts voisins
# reliés à leur départ par une petite palmure (paires de doigts, dans l'ordre de MEMBRES)
MAINS = {
    "bras_d": {
        "paume": [
            (21.0, 52.0),
            (26.0, 51.0),
            (29.0, 53.0),
            (28.5, 56.0),
            (25.0, 57.5),
            (21.0, 57.0),
            (19.5, 55.0),
        ],
        "bases": [(28.5, 54.0), (20.0, 54.5), (21.5, 57.0), (26.5, 57.3)],
        "palmes": [(1, 2), (2, 3)],
    },
    "patte_g": {
        "paume": [(85.5, 46.5), (90.5, 46.5), (92.5, 49.5), (90.5, 52.0), (86.0, 52.0), (83.5, 49.5)],
        "bases": [(84.3, 50.0), (87.5, 52.0), (91.8, 50.0)],
        "palmes": [(0, 1), (1, 2)],
    },
    "bras_g": {
        "paume": [
            (55.0, 57.0),
            (60.0, 56.5),
            (62.5, 59.0),
            (60.5, 62.5),
            (57.0, 63.5),
            (54.0, 62.0),
            (53.5, 59.5),
        ],
        "bases": [(57.5, 57.0), (54.0, 58.5), (54.5, 62.0), (57.5, 63.5)],
        "palmes": [(1, 2), (2, 3)],
    },
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


def forme(nom, masque, matiere, plan="corps", cernee=False, volume=None, zones=None, soude=()):
    """plan : fond (membres de l'autre côté), corps, devant (membres proches) ; cernee : un
    contour la sépare de ce qu'elle recouvre ; volume : la forme dont elle prend le relief
    (une matière dessinée dans le corps, comme le flanc, prend celui du corps) ; zones : les
    autres matières et les motifs de la forme, {nom: masque}, posés dans l'ordre (étape 3a) ;
    soude : les formes (début de leur nom) dont elle ne se sépare ni par un contour ni par une
    ombre portée, parce qu'elle les continue (la main, son bras et ses doigts)."""
    return {
        "nom": nom,
        "masque": masque,
        "matiere": matiere,
        "plan": plan,
        "cernee": cernee,
        "volume": volume or nom,
        "zones": zones or {},
        "soude": soude,
    }


def soudees(f, k):
    """Les numéros des formes auxquelles la forme k est soudée."""
    debuts = tuple(f[k]["soude"])
    return [j for j, g in enumerate(f) if debuts and g["nom"].startswith(debuts)]


def tete_tournee(tete, x=None, y=None):
    """Coordonnées vues par la tête, tournée de tete × PAS autour du cou ; l'effet s'estompe
    vers le corps (rien ne bouge en arrière du pivot). Une tête décrite tournée vers la droite
    (en mode image, sans MIROIR : museau à droite de la nuque) tourne dans l'autre sens."""
    x, y = (X, Y) if x is None else (x, y)  # une autre grille : la pupille en demi-pixels
    sens = -1 if SQUELETTE and SQUELETTE["museau"][0] > SQUELETTE["nuque"][0] else 1
    poids = np.clip((sens * (PIVOT[0] - x) + 3) / 6, 0, 1)
    a = -sens * tete * PAS * poids
    dx, dy = x - PIVOT[0], y - PIVOT[1]
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


# le bout de chaque doigt, relevé sur la photo : largeur, hauteur (pixels), inclinaison (degrés).
# Pas des boules toutes pareilles (correction de Léonard) : vu de face, un disque est un ovale de
# 7 pixels ; enroulé autour de la branche, il se présente par la tranche, aplati ; le pouce est
# plus petit. Sans relevé, un disque rond de 6 pixels
BOUTS = {
    "doigt1_d": (4, 3, 0),  # le long de la branche, vers la main gauche
    "doigt2_d": (6, 4, -10),  # couché sur la branche
    "doigt3_d": (6, 6, 0),  # de face
    "doigt4_d": (5, 5, 0),
    "doigt1_g": (4, 3, 0),  # le doigt qui monte, quasi invisible
    "doigt2_g": (6, 4, 0),
    "doigt3_g": (7, 5, -25),  # de face, de biais
    "doigt4_g": (6, 5, 0),
    "orteil_a_g": (5, 5, 0),
    "orteil_b_g": (6, 7, 0),  # le grand disque, de face
    "orteil_c_g": (6, 6, 0),
    "orteil_a_d": (4, 4, 0),  # derrière la branche
    "orteil_b_d": (4, 4, 0),
    "orteil_c_d": (4, 5, 0),  # au bout de l'orteil horizontal, dans le vide
}


def _pixel(x, y):
    """Un point en unités de la scène → ses coordonnées en pixels de la toile."""
    x -= ORIGINE[0]
    return ((UNITES[0] - x) if MIROIR else x) * ECHELLE, (y - ORIGINE[1]) * ECHELLE


def disque(bout):
    """Le disque du bout d'un doigt (BOUTS), un ovale centré sur le bout : sa zone de matière.
    Rond de 6 pixels, c'est l'ancien disque en boule, au pixel près."""
    larg, haut, incl = BOUTS.get(bout, (6, 6, 0))
    x1, y1 = _pixel(*SQUELETTE[bout])
    cx, cy = round(x1 - larg / 2) + larg / 2, round(y1 - haut / 2) + haut / 2
    i, j = np.meshgrid(np.arange(W) + 0.5 - cx, np.arange(H) + 0.5 - cy)
    c, s = math.cos(math.radians(incl)), math.sin(math.radians(incl))
    return ((i * c + j * s) / (larg / 2)) ** 2 + ((j * c - i * s) / (haut / 2)) ** 2 <= 1


def doigt(base, bout, largeur=3, via=()):
    """Un doigt posé au pixel près, de la base au bout (repères du squelette), en passant par
    les points via (unités : l'éventail des métatarses, caché sous un autre membre) : un trait
    de largeur pixels, en escalier (chaque pixel touche le suivant par un côté, sinon le
    contour le coupe), et son disque en boule centré sur le bout. Un masque par doigt."""
    m = np.zeros((H, W), bool)

    def poser(i, j):
        if 0 <= j < H and 0 <= i < W:
            m[j, i] = True

    pts = [_pixel(*p) for p in (SQUELETTE[base], *via, SQUELETTE[bout])]
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
    return m | disque(bout)


def main(membre_, palmure=2.0):
    """La main d'un membre (MAINS) : la paume, et entre deux doigts voisins une petite palmure,
    de leur base jusqu'à palmure pixels le long de chacun, au bord libre creusé vers la paume.
    Elle se pose par-dessus la racine des doigts, qui partent ainsi chacun de sa base."""
    m_ = MAINS[membre_]
    m = dans(px(*m_["paume"]), X, Y)
    bases = [np.array(b) for b in m_["bases"]]
    bouts = [np.array(_pixel(*SQUELETTE[b])) for b in MEMBRES[membre_][2]]
    for a, b in m_["palmes"]:
        p1, p2 = (
            bases[k] + (bouts[k] - bases[k]) * palmure / np.hypot(*(bouts[k] - bases[k]))
            for k in (a, b)
        )
        creux = 0.6 * (p1 + p2) / 2 + 0.4 * (bases[a] + bases[b]) / 2
        m |= dans(px(*(tuple(q) for q in (bases[a], p1, creux, p2, bases[b]))), X, Y)
    return m


# ------------------------------------------------------------------ couleurs et motifs (3a)
def pixels(*pts):
    """Un motif posé au pixel près : les pixels (i, j) du sprite, relevés sur la photo."""
    m = np.zeros((H, W), bool)
    for i, j in pts:
        m[j, i] = True
    return m


def barres(chemin, centres, largeur=1.0, portee=3.0):
    """Des barres en travers d'un membre ou d'un doigt : chemin, ses points (os ou tracé du
    doigt) ; centres, le milieu de chaque barre, relevé sur la photo (outils.py profil le long
    de l'os : chaque creux est une barre) ; largeur, le long du membre, et portée, de part et
    d'autre de son axe. Tout en unités. Chaque barre suit le segment le plus proche."""
    m = np.zeros((H, W), bool)
    for cx, cy in centres:
        meilleur = None
        for (xa, ya), (xb, yb) in zip(chemin, chemin[1:], strict=False):
            dx, dy = xb - xa, yb - ya
            t = np.clip(((cx - xa) * dx + (cy - ya) * dy) / (dx * dx + dy * dy), 0, 1)
            d = math.hypot(xa + t * dx - cx, ya + t * dy - cy)
            if meilleur is None or d < meilleur[0]:
                meilleur = (d, dx / math.hypot(dx, dy), dy / math.hypot(dx, dy))
        _, ux, uy = meilleur
        long = (X - cx) * ux + (Y - cy) * uy
        trav = -(X - cx) * uy + (Y - cy) * ux
        m |= (np.abs(long) < largeur / 2) & (np.abs(trav) < portee)
    return m


def bande(m, dx, dy, n=2):
    """La bande de n pixels de la forme m tournée vers la direction (dx, dy) : les pixels dont
    un des n voisins dans cette direction sort de la forme (la face verte d'un membre)."""
    out = np.zeros(m.shape, bool)
    for k in range(1, n + 1):
        di, dj = round(k * dx), round(k * dy)
        v = np.zeros(m.shape, bool)
        src = m[max(dj, 0) : H + min(dj, 0), max(di, 0) : W + min(di, 0)]
        v[max(-dj, 0) : H + min(-dj, 0), max(-di, 0) : W + min(-di, 0)] = src
        out |= m & ~v
    return out


def cernes(nom, base, bouts, plan, via=None, motifs=None):
    """Une forme cernée par doigt, pour que deux disques voisins restent séparés ; via : pour
    chaque doigt, ses points de passage (voir doigt) ; motifs : pour chaque doigt, ses barres
    (centres en pixels) et les taches de son disque (pixels). Le disque est d'une matière à
    part, plus claire."""
    via = via or [()] * len(bouts)
    motifs = motifs or [((), ())] * len(bouts)
    out = []
    for k, (b, v, (centres, taches)) in enumerate(zip(bouts, via, motifs, strict=True)):
        chemin = [SQUELETTE[base], *v, SQUELETTE[b]]
        zones = {"barre": barres(chemin, px(*centres), 1.0, 1.0), "disque": disque(b)}
        zones["tache"] = pixels(*taches)
        out.append(forme(f"{nom}{k}", doigt(base, b, via=v), "membre", plan, True, zones=zones))
    return out


def formes(gorge=0.0, souffle=0.0, cligne=False, tete=0):
    """Étape 1, ébauche : le corps et la tête en polygones relevés sur la photo, quadrillée au
    pixel du sprite ; le profil de la tête en droites, comme la grenouille de gauche ; les
    membres en os à leur largeur mesurée ; les mains et les doigts selon les conventions de la
    grenouille de gauche (paume, bases, palmure, disques relevés)."""
    S = SQUELETTE
    xt, yt = tete_tournee(tete)
    s, g = souffle * 2, gorge * 2  # en pixels
    Py = (Y - ORIGINE[1]) * ECHELLE - 0.5
    corps = dans(
        [
            # le museau tronqué, en droites, comme la grenouille de gauche : une verticale du bout
            # (à la lèvre) au coin de la narine ; puis le dessus du museau monte en droite jusqu'à
            # la bosse de l'autre œil, sans creux ; puis le crâne, droit, jusqu'à la nuque
            S["museau"],
            S["narine_g"],
            *px((17.5, 15.8)),
            S["nuque"],
            # le dos, en pente douce puis plus raide jusqu'à la croupe, cachée par la cuisse
            *px((47.0, 16.6), (54.0, 18.4), (60.5, 20.8), (67.0, 24.2), (73.5, 28.2)),
            *px((78.5, 32.0), (82.0, 36.0), (83.0, 41.0), (82.0, 47.0)),
            # le ventre, svelte : une droite qui remonte du flanc, derrière l'avant-bras, jusqu'à
            # la poitrine, derrière le coude droit ; puis la gorge, en pente, jusqu'au menton
            *px((77.0, 49.5 + s * 0.5), (66.0, 49.5 + s * 0.5), (59.0, 48.5 + s)),
            *px((50.0, 44.6 + s), (42.0, 42.6), (37.5, 41.5), (33.5, 38.2 + g * 0.5)),
            *px((29.5, 34.6 + g), (25.0, 31.4 + g * 0.8), (20.0, 30.2 + g * 0.4), (14.0, 29.4)),
            *px((9.0, 28.5)),
        ],
        xt,
        yt,
    )
    ox, oy = S["oeil"]
    # l'œil entier de la grenouille de gauche, jugé parfait (correction de Léonard) : le globe de
    # 3,4 unités de rayon, la paupière en cercle de 4 unités sur sa moitié haute ; ici décalée
    # vers la droite et un peu vers le bas, pour que la peau verte se voie à droite et en haut à
    # droite du globe, presque pas à gauche : l'œil regarde un peu vers la gauche
    corps |= ellipse(ox + 0.35, oy + 0.15, 4.0, 4.0, xt, yt) & (yt < oy + 0.15)
    autre = ellipse(*S["autre_oeil"], 1.5, 1.05, xt, yt)  # l'autre œil : une bosse verte
    corps |= autre
    br = dans_branche(X, Y)
    # patte arrière droite (fond), derrière la branche : le tibia en sort derrière le pied et
    # descend au talon ; du tarse, tout en bas, les orteils s'ouvrent en éventail vers le haut,
    # reliés par la palmure : la plante, en trapèze, pas en rond (bord gauche en diagonale le
    # long de l'orteil a, dessous presque droit le long de l'orteil c)
    patte_d = os_(["genou_d", "talon_d", "tarse_d"], [1.5, 1.2, 1.1]) & ~br
    pied_d = dans(
        [S["tarse_d"], *px((61.0, 68.0), (61.5, 62.0), (72.0, 62.0), (73.5, 70.5), (74.5, 73.5))]
        + px((75.0, 77.0), (68.0, 78.0)),
        X,
        Y,
    )
    pied_d &= ~br
    orteils_d = cernes("orteil_d", "tarse_d", MEMBRES["patte_d"][2], "fond")
    for o in orteils_d:
        o["masque"] &= ~br
    # les orteils continuent le pied, sans contour : les deux qui remontent derrière la branche
    # n'en sortent que par leur racine, le troisième part de son bord droit
    for o in orteils_d:
        o["soude"] = ("pied_d", "patte_d", "orteil_d")
    # bras droit, de l'autre côté mais devant le corps (correction de Léonard) : le bras part de
    # l'épaule, sur la poitrine, jusqu'au coude, pointu ; l'avant-bras, plus épais, revient à la
    # main posée sur la branche
    bras_d = fuseau("epaule_d", "coude_d", [1.0, 1.05, 0.7])
    bras_d |= fuseau("coude_d", "poignet_d", [0.7, 1.3, 1.05], milieu=0.35)
    bras_d |= os_(["poignet_d", "main_d"], [1.05, 0.9])
    # le bras, de l'épaule au coude, rogné d'un pixel sur son bord droit : un petit vide entre le
    # coude et le ventre (correction de Léonard)
    haut_d = fuseau("epaule_d", "coude_d", [1.0, 1.05, 0.7])
    rogne = haut_d & ~np.roll(haut_d, -1, axis=1) & (Y <= S["coude_d"][1] + 0.6)
    bras_d &= ~rogne
    # patte arrière gauche (devant) : la cuisse, verticale, derrière le tibia mais devant la
    # croupe, se voit à peine : un liseré à gauche du tibia, du vert à sa droite ; à gauche, on
    # voit nettement le flanc (corrections de Léonard) ; le tibia, gros, du genou à la branche
    cuisse_g = dans(
        px((79.0, 34.0), (82.0, 30.0), (88.0, 28.8), (93.0, 30.0), (97.0, 33.5), (97.5, 46.0))
        + px((94.0, 48.5), (80.0, 49.0), (78.5, 42.0)),
        X,
        Y,
    )
    # le tibia : un genou large et rond en haut (5 pixels de rayon), un mollet bombé vers la
    # droite, puis il s'affine vers la cheville
    tibia_g = fuseau("genou_g", "talon_g", [2.5, 2.2, 1.6], milieu=0.35, decale=(0.25, 0.0))
    tibia_g |= os_(["talon_g", "tarse_g"], [1.6, 1.3])
    # les orteils partent du pied, de leur base, reliés par une petite palmure (correction de
    # Léonard : ils se superposaient à leur départ)
    eventail_g = [px(b) for b in MAINS["patte_g"]["bases"]]
    # bras gauche (devant) : le bras, fin, à l'horizontale ; l'avant-bras, épais, part du coude,
    # juste sous le dos : son haut est coupé droit, parallèle au dos, sans capuchon qui ferait une
    # bosse ; un membre à part, cerné, même de la couleur du dos (corrections de Léonard)
    bras_g = os_(["epaule_g", "coude_g"], [0.9, 0.85])
    bras_g |= fuseau("coude_g", "poignet_g", [2.4, 2.1, 1.6], milieu=0.4)
    bras_g |= os_(["poignet_g", "main_g"], [1.6, 1.2])
    Px = (X - ORIGINE[0]) * ECHELLE - 0.5
    bras_g &= Py > 31.0 + 0.12 * (Px - 66.0)
    bases_d, bases_g = ([px(b) for b in MAINS[m]["bases"]] for m in ("bras_d", "bras_g"))
    doigts_g = cernes("doigt_g", "main_g", MEMBRES["bras_g"][2], "devant", bases_g)
    # le doigt qui monte sur la branche, derrière l'avant-bras : fin, quasi invisible
    doigt1_g = doigts_g.pop(0)
    doigt1_g["masque"] = doigt("main_g", "doigt1_g", largeur=2, via=bases_g[0])
    doigt1_g["plan"] = "corps"
    f = [
        forme("patte_d", patte_d, "membre", "fond", cernee=True),
        forme("pied_d", pied_d, "membre", "fond", cernee=True, soude=("patte_d",)),
        *orteils_d,
        forme("corps", corps, "dos", cernee=True),
        forme("bras_d", bras_d, "membre", "corps", cernee=True),
        *cernes("doigt_d", "main_d", MEMBRES["bras_d"][2], "corps", bases_d),
        forme("main_d", main("bras_d"), "membre", "corps", True, soude=("doigt_d", "bras_d")),
        forme("cuisse_g", cuisse_g, "membre", "corps", cernee=True),
        forme("tibia_g", tibia_g, "membre", "devant", cernee=True),
        *cernes("orteil_g", "tarse_g", MEMBRES["patte_g"][2], "devant", eventail_g),
        forme("pied_g", main("patte_g"), "membre", "devant", True, soude=("orteil_g", "tibia_g")),
        doigt1_g,
        forme("bras_g", bras_g, "dos", "devant", cernee=True),
        *doigts_g,
        forme("main_g", main("bras_g"), "membre", "devant", True, soude=("doigt_g", "bras_g")),
    ]
    oeil = {"x": ox, "y": oy, "r": 3.4, "ferme": cligne, "tete": tete}
    oeil["masque"] = (xt - ox) ** 2 + (yt - oy) ** 2 <= oeil["r"] ** 2
    oeil["u"], oeil["v"] = (xt - ox) / oeil["r"], (yt - oy) / oeil["r"]
    # l'autre œil : on n'en voit que la bosse verte, pas le globe (correction de Léonard)
    # traits du visage : la bouche (un pixel par colonne, sur la droite du museau à la
    # commissure) ; la narine ; le tympan ; le pli sous la bosse de l'autre œil
    (mx, my), (cx, cy) = S["museau"], S["commissure"]
    ligne = cy + (xt - cx) * (my - cy) / (mx - cx)
    tx, ty = S["tympan"]
    rt = np.hypot(xt - tx, yt - ty)
    ax, ay = S["autre_oeil"]
    traits = {
        "bouche": corps & (xt > mx) & (xt < cx) & (np.floor(2 * yt) == np.floor(2 * ligne)),
        "narine": (np.abs(xt - S["narine"][0]) < 0.5) & (np.abs(yt - S["narine"][1]) < 0.5),
        "tympan": (rt > 0.9) & (rt < 1.4),
        "pli_autre_oeil": bord(autre) & (yt > ay) & ~oeil["masque"],
    }
    return f, traits, {}, oeil


# ------------------------------------------------------------------ rendus
def etiqueter(f):
    """Matière, plan et numéro de forme de chaque pixel (le plus proche l'emporte)."""
    mat, plan, num = np.full((H, W), "", object), np.full((H, W), "", object), np.full((H, W), -1)
    for k, fo in enumerate(f):
        m = fo["masque"]
        mat[m], plan[m], num[m] = fo["matiere"], fo["plan"], k
        for nom, zone in fo["zones"].items():  # matières et motifs de la forme, dans l'ordre
            mat[m & zone] = nom
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
        img[autour & plein & (num < k) & (num >= 0) & ~np.isin(num, soudees(f, k))] = CONTOUR
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
    if "bord" in couleurs:  # le bord pâle de la paupière, autour du globe, sauf en haut
        autour = dilater(m) & ~m & (v > -0.3) & (img != "") & (img != CONTOUR)
        img[autour] = couleurs["bord"]
    img[m] = couleurs["iris"]
    img[bord(m)] = couleurs.get("cercle", CONTOUR)  # un cercle continu d'un pixel
    p = {
        "ronde": u * u + v * v < 0.3,
        "horizontale": (u / 0.75) ** 2 + (v / 0.32) ** 2 < 1,
        # une fente en amande : 3 pixels au milieu, pointue aux deux bouts
        "verticale": np.abs(u - REGARD) < 0.26 * (1 - (v / 0.82) ** 2),
    }[PUPILLE]
    img[m & p] = couleurs["pupille"]
    return img


def tracer(img, traits, num, oeil, couleur=None):
    """Les traits (bouche, narine, tympan, plis) : d'une couleur donnée (ébauche, croquis) ou,
    sans couleur donnée, de la couleur du dessous assombrie (aplats, détails)."""
    hors_oeil = ~oeil["masque"] if oeil else True
    for nom, m in traits.items():
        ici = m & (num >= 0) & hors_oeil & (img != CONTOUR)
        if couleur is None and nom in TRAITS_MATIERE:
            img[ici] = MATIERES[TRAITS_MATIERE[nom]]
            continue
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
    for nom, coul in (MATIERES | MOTIFS).items():  # les motifs sont des zones des formes
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


# étape 3b : les formes à l'ombre sur la photo (la patte et le bras gauches, derrière le ventre),
# de combien ; les granules blanches de tomopterna, sur le vert
OMBRES = {"patte_g": 0.16, "orteil_g": 0.12, "bras_g": 0.08, "main_g": 0.07, "doigt_g": 0.06}
GRANULE = "#E2EEDA"
# l'éclat (carte blanche de Léonard, avant l'étape 4) : des teintes au-delà de la photo, qui font
# ressortir la brillance et la beauté. Par matière : la teinte où glissent les ombres, celle où
# glissent les lumières, et l'écart de teinte d'une plaque à l'autre (du jauni au rougi)
ECLAT = {
    "dos": (0.47, 0.18, 0.04),  # citron dans la lumière, émeraude puis bleu-vert dans l'ombre
    "membre": (0.975, 0.125, 0.05),  # de l'or au cramoisi
    "disque": (0.01, 0.13, 0.035),  # ambre et miel
    "ventre": (0.80, 0.07, 0.025),  # nacre : crème dans la lumière, lilas dans l'ombre
    "bas_ventre": (0.985, 0.095, 0.035),  # abricot et rose
    "gorge": (0.68, 0.88, 0.02),  # lavande, bleutée dans l'ombre de la tête
    "pale": (0.70, 0.90, 0.02),
    "levre": (0.72, 0.12, 0.0),
    "barre": (0.72, 0.87, 0.02),  # violet-noir, indigo dans l'ombre, prune dans la lumière
    "tache": (0.85, 0.97, 0.02),  # marbrures prune et vieux rose
}
# les motifs restent sombres : seule la part basse de leur rampe sert, même en pleine lumière
PLAFOND = {"barre": 0.5, "tache": 0.7}
ECLAT_ACTIF = True  # False : le modelé fidèle à la photo, gelé dans gauche_details_valide.json
REFLET_FROID = "#86E8E0"  # la lumière du ciel renvoyée sur les bords à l'ombre
ECLAT_HUMIDE = "#FFFFFF"  # les points de brillance de la peau mouillée


def rampe_eclat(hexa, ombre, lumiere, n=11):
    """n tons d'une couleur, du plus sombre au plus clair : les ombres glissent vers la teinte
    ombre et se saturent, les lumières vers la teinte lumiere et s'éclaircissent fort."""
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, sat = colorsys.rgb_to_hls(r, g, b)
    out = []
    for k in range(n):
        t = 2 * k / (n - 1) - 1  # de -1 (ombre) à 1 (lumière)
        cible = ombre if t < 0 else lumiere
        hh = (h + ((cible - h + 0.5) % 1 - 0.5) * abs(t) * 0.6) % 1
        if t < 0:
            ll = max(clarte - 0.3 * -t, 0.06)
            ss = min(max(sat * (1 + 0.35 * -t), 0.3 * -t), 1)
        else:  # des lumières vives, pas crayeuses
            ll = clarte + (0.95 - clarte) * 0.5 * t
            ss = min(sat * (1 + 0.3 * t), 1)
        out.append(hexa_de(*colorsys.hls_to_rgb(hh, ll, ss)))
    return out


def continue_(tons, pos):
    """La rampe lue en continu : pos, position flottante dans la rampe (0 = le plus sombre) ;
    les tons voisins se mélangent au lieu de se tramer. Renvoie des RVB (…, 3)."""
    rvb = np.array([[int(h[i : i + 2], 16) for i in (1, 3, 5)] for h in tons], float)
    pos = np.clip(pos, 0, len(tons) - 1)
    bas = np.floor(pos).astype(int)
    haut = np.minimum(bas + 1, len(tons) - 1)
    fr = (pos - bas)[..., None]
    return rvb[bas] * (1 - fr) + rvb[haut] * fr


def lisser(valeurs, m, r=1):
    """La moyenne des valeurs sur m seulement, autour de chaque pixel (rayon r)."""
    return _flou(np.where(m, valeurs, 0.0), r) / np.maximum(_flou(m.astype(float), r), 1e-6)


def melanger(a, b, t):
    """La couleur a, tirée vers b de la part t."""
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return hexa_de(*((x + (y - x) * t) / 255 for x, y in zip(ca, cb, strict=True)))


def decaler(m, di, dj):
    """Le tableau décalé de (di, dj) pixels, sans retour par les bords."""
    v = np.zeros(m.shape, m.dtype)
    src = m[max(-dj, 0) : H + min(-dj, 0), max(-di, 0) : W + min(-di, 0)]
    v[max(dj, 0) : H + min(dj, 0), max(di, 0) : W + min(di, 0)] = src
    return v


def semis(m, densite, graine, pris=None):
    """Des points semés au hasard dans m, à places fixes (graine), jamais deux voisins (même en
    diagonale, ni d'un point déjà pris)."""
    r = np.random.default_rng(graine).random((H, W))
    pris = np.zeros((H, W), bool) if pris is None else pris.copy()
    out = np.zeros((H, W), bool)
    for j, i in sorted(zip(*np.nonzero(m & (r < densite)), strict=False), key=lambda q: r[q]):
        if not pris[max(j - 1, 0) : j + 2, max(i - 1, 0) : i + 2].any():
            out[j, i] = pris[j, i] = True
    return out


def fondre(zone, a, b):
    """Le passage d'une matière à l'autre sur une même face (ventre, gorge) : deux rangées
    tramées au lieu d'une ligne."""
    autre = zone == b
    pres = dilater(autre)[:H, :W]
    zone[(zone == a) & pres & (TRAME > 0.5)] = b
    zone[(zone == a) & dilater(pres)[:H, :W] & ~pres & (TRAME > 0.85)] = b
    return zone


def details(f, traits, motifs, oeil, eclat=None):
    """Étape 3b, le modelé : volume par forme (lumière d'en haut, un peu de la gauche, comme sur
    la photo), ombre portée par les formes plus proches, patte et bras gauches à l'ombre, ventre
    bombé, gorge dans l'ombre de la tête, pustules du ventre, rampes tramées en trois variantes
    de teinte, passages tramés entre les zones du ventre, granules blanches sur le vert et en
    liseré au bord du vert, reflets humides des disques, œil bombé et réticulé, contour
    teinté. Avec l'éclat (ECLAT_ACTIF), les teintes de ECLAT, des plaques du jauni au rougi, des
    points de brillance et un reflet froid sur les bords à l'ombre."""
    eclat = ECLAT_ACTIF if eclat is None else eclat
    mat, plan, num = etiqueter(f)
    lx, ly = LUMIERE
    lum = np.full((H, W), 0.5)
    for k, fo in enumerate(f):
        ici = num == k
        volume = next(g["masque"] for g in f if g["nom"] == fo["volume"])
        gy, gx = np.gradient(_flou(volume.astype(float), 3))
        # le gradient du masque flouté pointe vers l'intérieur : un bord tourné vers la lumière
        # (LUMIERE, d'où elle vient) a un gradient opposé à elle
        face = -(gx * lx + gy * ly)
        lum[ici] = (0.55 + 4.0 * face - 0.18 * ((Y - ORIGINE[1]) / UNITES[1] - 0.4))[ici]
        # l'ombre portée par les formes plus proches, du côté opposé à la lumière
        proches = [
            j
            for j in range(k + 1, len(f))
            if f[j]["volume"] == f[j]["nom"] and k not in soudees(f, j) + soudees(f, k)
        ]
        porte = decaler(np.isin(num, proches), round(-lx * 2), round(-ly * 2))
        lum[ici] -= 0.32 * _flou(porte.astype(float), 1)[ici]
        lum[ici] -= next((v for n, v in OMBRES.items() if fo["nom"].startswith(n)), 0.0)
    # le ventre, bombé, plus clair au milieu ; la gorge, dans l'ombre de la tête
    ventre = np.isin(mat, ["ventre", "bas_ventre", "tache"]) & (num == noms(f).index("corps"))
    bombe = np.clip(1 - ((X - 76.0) / 10) ** 2 - ((Y - 56.0) / 5.5) ** 2, 0, 1)
    lum[ventre] += 0.2 * bombe[ventre]
    lum[ventre] = np.maximum(lum[ventre], 0.4)  # le bas du ventre, face à nous, reste éclairé
    lum[mat == "gorge"] -= 0.1
    # pustules du ventre et de la gorge : un pixel plus clair, son ombre dessous
    peau = np.isin(mat, ["ventre", "bas_ventre", "gorge"])
    pustules = semis(peau, 0.16, 11)
    lum[pustules] += 0.14
    dessous = decaler(pustules, 0, 1) & peau
    lum[dessous] -= 0.1
    plaques = np.clip(np.floor(_flou(np.random.default_rng(3).random((H, W)), 4) * 6 - 2), -1, 1)
    if eclat:  # cinq niveaux de plaques, plus larges
        champ = _flou(np.random.default_rng(17).random((H, W)), 5)
        plaques = np.clip(np.round((champ - champ.mean()) / champ.std() * 1.1), -2, 2)
    grain = (np.random.default_rng(5).random((H, W)) - 0.5) * 0.08
    img = np.full((H, W), "", object)
    couleurs = MATIERES | MOTIFS
    zone = mat.copy()
    for nom, m in motifs.items():
        zone[m & (num >= 0)] = nom
    if not eclat:
        zone = fondre(fondre(zone, "ventre", "bas_ventre"), "gorge", "ventre")
    for nom, coul in couleurs.items():
        ici = zone == nom
        if not ici.any() or (eclat and nom in FONDUS):
            continue
        if eclat and nom in ECLAT:
            ombre, lumiere, ecart = ECLAT[nom]
            haut = 10 * PLAFOND.get(nom, 1.0)
            idx = np.clip(np.floor((lum + grain) * haut + TRAME * 0.9), 0, 10).astype(int)
            for v in (-2, -1, 0, 1, 2):
                sous = ici & (plaques == v)
                tons_v = rampe_eclat(_decaler(coul, ecart * v / 2), ombre, lumiere)
                img[sous] = np.array(tons_v, object)[idx[sous]]
            continue
        tons = rampe(coul)
        idx = np.clip(
            np.floor((lum + grain) * (len(tons) - 1) + TRAME * 0.9), 0, len(tons) - 1
        ).astype(int)
        # trois variantes de teinte, par plaques, sur les grandes matières seulement
        for v in (-1, 0, 1) if nom in ("dos", "ventre", "membre") else (0,):
            sous = ici & (plaques == v) if nom in ("dos", "ventre", "membre") else ici
            tons_v = rampe(_decaler(coul, 0.012 * v)) if v else tons
            img[sous] = np.array(tons_v, object)[idx[sous]]
    if eclat:
        peindre_fondus(img, zone, lum, mat)
    # granules blanches : clairsemées sur le vert, serrées en liseré là où le vert touche une
    # autre matière de la même forme (le bord des membres verts, la limite du ventre)
    vert = mat == "dos"
    lisere = vert & dilater(~vert & (num >= 0))[:H, :W] & (decaler(num, 1, 0) == num)
    lisere |= vert & dilater(~vert & (num >= 0))[:H, :W] & (decaler(num, 0, 1) == num)
    granules = semis(lisere, 0.5, 7)
    granules |= semis(vert & ~lisere, 0.035, 8, pris=granules)
    img[granules & ~oeil["masque"]] = GRANULE
    # reflets humides : un point clair en haut à gauche de chaque disque
    for k, fo in enumerate(f):
        if "disque" in fo["zones"]:
            d = fo["zones"]["disque"] & (num == k)
            if d.any():
                jj, ii = np.nonzero(fo["zones"]["disque"])
                q = (jj.min() + 1, ii.min() + 1)
                if d[q]:
                    img[q] = "#FFF6DF" if eclat else rampe(MATIERES["disque"])[-1]
    if eclat:
        img[granules & ~oeil["masque"]] = "#F2FFEE"
        # des points de brillance, là où la lumière frappe la peau mouillée
        brille = np.isin(zone, ["dos", "membre", "disque", "ventre", "bas_ventre"]) & (lum > 0.88)
        for j, i in zip(*np.nonzero(semis(brille & ~granules, 0.3, 31)), strict=False):
            img[j, i] = melanger(img[j, i], ECLAT_HUMIDE, 0.55)
        # un reflet froid sur les bords à l'ombre, à droite et en bas de la silhouette
        sil = num >= 0
        bords = sil & (~decaler(sil, -1, 0) | ~decaler(sil, 0, -1)) & (lum < 0.62)
        bords &= np.isin(zone, ["dos", "membre", "disque", "ventre", "bas_ventre", "gorge"])
        for j, i in zip(*np.nonzero(bords), strict=False):
            img[j, i] = melanger(img[j, i], REFLET_FROID, 0.3)
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil)
    if eclat:  # la lèvre nacrée, claire dans la lumière, lilas dans l'ombre
        levre = traits["bouche"] & (num >= 0) & (img == MATIERES["levre"])
        img[levre & (lum > 0.5)] = "#FBF5F3"
        img[levre & (lum <= 0.5)] = "#E6DAEB"
    if CONTOUR_TEINTE:  # la matière qu'il borde, assombrie : une teinte par matière
        teinte = {nom: assombrir(c, 0.4) for nom, c in couleurs.items()}
        if eclat:  # le plus sombre de sa rampe, assombri encore : vert sapin, cramoisi, indigo
            teinte |= {n: assombrir(rampe_eclat(couleurs[n], *ECLAT[n][:2])[0], 0.7) for n in ECLAT}
        TEINTES.update(teinte.values())
        for j, i in zip(*np.nonzero(img == CONTOUR), strict=False):
            voisins = [(j, i)] if num[j, i] >= 0 else []
            voisins += [(j, i + 1), (j, i - 1), (j + 1, i), (j - 1, i)]
            for q in voisins:
                if 0 <= q[0] < H and 0 <= q[1] < W and zone[q] in teinte:
                    img[j, i] = teinte[zone[q]]
                    break
    return oeil_detaille(img, oeil, eclat)


# avec l'éclat, ces matières se peignent sans grain ni trame : le vert en grandes zones plus
# jaunes ou plus bleues qui se mêlent ; gorge, ventre et bas du ventre fondus l'un dans l'autre
# sur plusieurs pixels, par une limite irrégulière (corrections de Léonard)
FONDUS = ("dos", "gorge", "ventre", "bas_ventre")


# l'entre-deux (correction de Léonard : 1 500 couleurs, c'était trop) : les transitions se font
# en quelques paliers, sans trame ; 5 paliers entre les verts, 7 dans le ventre, et les tons de
# chaque rampe pris tels quels (11)
PALIERS_VERT, PALIERS_VENTRE = 5, 7


def peindre_fondus(img, zone, lum, mat):
    def poser(m, rvb):
        for (j, i), c in zip(zip(*np.nonzero(m), strict=False), rvb[m], strict=True):
            img[j, i] = hexa_de(*(c / 255))

    def ton(rampe_, pos):  # le ton de la rampe le plus proche : pas de mélange entre tons
        return continue_(rampe_, np.round(pos))

    # le vert : la lumière lissée, lue dans trois rampes bien distinctes (vert citron, émeraude,
    # bleu-vert), mêlées en paliers selon un champ lisse en grandes taches
    vert = zone == "dos"
    pos = lisser(lum, vert) * 10
    # (décalage de teinte : négatif vers le jaune, positif vers le bleu ; teinte des ombres, des
    # lumières ; de combien plus clair)
    verts = [
        ton(rampe_eclat(_decaler(MATIERES["dos"], dh), o, li), pos + dp)
        for dh, o, li, dp in (
            (-0.07, 0.36, 0.15, 1.2),  # vert citron, plus clair
            (0.0, 0.45, 0.22, 0.0),  # émeraude
            (0.1, 0.53, 0.40, -0.6),  # bleu-vert, menthe dans la lumière
        )
    ]
    champ = _flou(_flou(np.random.default_rng(23).random((H, W)), 6), 6)
    t = np.clip((champ - champ.mean()) / (1.6 * champ.std()), -1, 1)
    t = (np.round((t + 1) / 2 * (PALIERS_VERT - 1)) / (PALIERS_VERT - 1) * 2 - 1)[..., None]
    rvb = np.where(t < 0, verts[1] * (1 + t) - verts[0] * t, verts[1] * (1 - t) + verts[2] * t)
    poser(vert, rvb)
    # le ventre : gorge, ventre et bas du ventre mêlés selon leurs zones floutées (sur 6 à 8
    # pixels) et un peu de bruit, pour une limite qui ondule ; en paliers, de la gorge (-1) au
    # bas du ventre (+1) en passant par le ventre (0)
    peau = np.isin(zone, FONDUS[1:])
    poids = []
    for k, n in enumerate(FONDUS[1:]):
        bruit = _flou(np.random.default_rng(40 + k).random((H, W)), 3)
        poids.append(_flou(_flou((mat == n).astype(float), 3), 3) * (0.6 + bruit))
    total = np.maximum(sum(poids), 1e-6)
    sv = (poids[2] - poids[0]) / total
    m = (PALIERS_VENTRE - 1) / 2
    sv = (np.round(sv * m) / m)[..., None]
    gorge, ventre, bas = (
        ton(rampe_eclat(MATIERES[n], *ECLAT[n][:2]), lum * 10) for n in FONDUS[1:]
    )
    rvb = np.where(sv < 0, ventre * (1 + sv) - gorge * sv, ventre * (1 - sv) + bas * sv)
    poser(peau, rvb)


def noms(f):
    return [fo["nom"] for fo in f]


def oeil_detaille(img, oeil, eclat=False):
    """L'œil du modelé : le bord pâle de la paupière, le globe bombé (clair en haut à gauche),
    l'iris argenté finement réticulé, le cercle, la fente, un reflet blanc et un petit reflet
    bleuté ; fermé, la paupière inférieure, membrane lilas au réseau doré irrégulier."""
    m, u, v = oeil["masque"], oeil["u"], oeil["v"]
    # avec l'éclat : un iris de nacre, bleu dans l'ombre, chaud dans la lumière, et 9 tons
    tons = rampe_eclat(OEIL["iris"], 0.62, 0.11, 9) if eclat else rampe(OEIL["iris"])
    n = len(tons) - 1
    if oeil.get("autre") is not None:  # l'autre œil ne cligne jamais
        img[oeil["autre"] & (img != "") & (img != CONTOUR)] = tons[n // 2]
    if oeil["ferme"]:  # sans cercle ni bord autour : pas d'effet lunettes
        membrane = MATIERES["membrane"]
        tons_m = rampe_eclat(membrane, 0.70, 0.95, 9) if eclat else rampe(membrane)
        k = len(tons_m) - 1
        clair = k * (0.53 - 0.33 * v) if eclat else 3.2 - 2.0 * v
        idx = np.clip(np.floor(clair + TRAME * 0.9), 0, k).astype(int)
        img[m] = np.array(tons_m, object)[idx][m]
        img[semis(m & ~bord(m), 0.2, 21)] = "#D9B55E" if eclat else "#B49C6E"
        return img
    autour = dilater(m)[:H, :W] & ~m & (v > -0.3) & (img != "") & (img != CONTOUR)
    img[autour] = "#BDEBB2" if eclat else OEIL["bord"]
    clair = n * (0.57 - 0.33 * v - 0.15 * u) if eclat else 3.4 - 2.0 * v - 0.9 * u
    idx = np.clip(np.floor(clair + TRAME * 0.9), 0, n).astype(int)
    img[m] = np.array(tons, object)[idx][m]
    reticule = semis(m & ~bord(m), 0.22, 13)
    img[reticule] = np.array(tons, object)[np.clip(idx - 2, 0, n)][reticule]
    img[bord(m)] = "#211A33" if eclat else OEIL["cercle"]
    fente = np.abs(u - REGARD) < 0.26 * (1 - (v / 0.82) ** 2)
    img[m & fente] = "#120E22" if eclat else OEIL["pupille"]
    # le haut de la fente bleuté du modelé validé ; avec l'éclat, une fente d'un seul noir
    # (correction de Léonard)
    bleu = NOIRS[0] if eclat else "#2E3550"
    img[m & fente & (v < -0.35) & (np.abs(u) < 0.12)] = bleu  # le haut de la fente bleuté
    img[m & ((u + 0.42) ** 2 + (v + 0.42) ** 2 < 0.035)] = "#FFFFFF"
    img[m & ((u - 0.4) ** 2 + (v - 0.45) ** 2 < 0.02) & ~fente] = "#9FB4C0"
    return img


# la pupille en double résolution (essai de Léonard) : au rendu détaillé, une pièce à
# demi-pixels posée sur le globe, où seule la fente est redessinée, plus fine et plus nette
PUPILLE_FINE = True
NOIRS = ("#120E22",)  # la fente, avec l'éclat


def fente_fine(img, oeil, iris, noir):
    """La pupille en demi-pixels dès l'ébauche (Léonard : la résolution de l'œil est doublée à
    toutes les étapes), celle de la grenouille de gauche, jugée parfaite, telle quelle : une
    pièce deux fois plus fine posée sur le globe, où l'iris est repeint et la fente, verticale,
    en amande, redessinée à cette résolution. Renvoie (colonne, rangée) de son coin et la pièce."""
    m = oeil["masque"]
    jj, ii = np.nonzero(m)
    j0, j1, i0, i1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1

    def double(a):
        return np.repeat(np.repeat(a[j0:j1, i0:i1], 2, 0), 2, 1)

    piece = double(img)
    hj, hi = np.mgrid[2 * j0 : 2 * j1, 2 * i0 : 2 * i1]
    x2 = ORIGINE[0] + (hi + 0.5) / (2 * ECHELLE)
    y2 = ORIGINE[1] + (hj + 0.5) / (2 * ECHELLE)
    xt, yt = tete_tournee(oeil["tete"], x2, y2)
    u, v = (xt - oeil["x"]) / oeil["r"], (yt - oeil["y"]) / oeil["r"]
    dedans = double(m & ~bord(m))
    piece[dedans] = iris
    piece[dedans & (np.abs(u - REGARD) < 0.25 * np.clip(1 - (v / 0.9) ** 2, 0, 1) ** 1.4)] = noir
    return int(i0), int(j0), piece


def pupille_fine(img, oeil):
    """La pièce en demi-pixels du globe : (colonne, rangée) de son coin en pixels du sprite et
    l'image deux fois plus fine ; les pixels de l'image, sauf la fente, refaite à cette
    résolution, et l'iris redessiné là où l'ancienne fente, plus grossière, débordait."""
    m = oeil["masque"]
    jj, ii = np.nonzero(m)
    j0, j1, i0, i1 = jj.min(), jj.max() + 1, ii.min(), ii.max() + 1

    def double(a):
        return np.repeat(np.repeat(a[j0:j1, i0:i1], 2, 0), 2, 1)

    piece = double(img)
    hj, hi = np.mgrid[2 * j0 : 2 * j1, 2 * i0 : 2 * i1]
    x2 = ORIGINE[0] + (hi + 0.5) / (2 * ECHELLE)
    y2 = ORIGINE[1] + (hj + 0.5) / (2 * ECHELLE)
    xt, yt = tete_tournee(oeil["tete"], x2, y2)
    u, v = (xt - oeil["x"]) / oeil["r"], (yt - oeil["y"]) / oeil["r"]
    dedans = double(m & ~bord(m))
    fente = dedans & (np.abs(u - REGARD) < 0.25 * np.clip(1 - (v / 0.9) ** 2, 0, 1) ** 1.4)
    tons = rampe_eclat(OEIL["iris"], 0.62, 0.11, 9)
    trame = np.tile(BAYER, (piece.shape[0] // 4 + 1, piece.shape[1] // 4 + 1))
    trame = trame[: piece.shape[0], : piece.shape[1]]
    idx = np.clip(np.floor(8 * (0.57 - 0.33 * v - 0.15 * u) + trame * 0.9), 0, 8).astype(int)
    iris = dedans & np.isin(piece, NOIRS) & ~fente
    piece[iris] = np.array(tons, object)[idx][iris]
    piece[fente] = NOIRS[0]
    return int(i0), int(j0), piece


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


RENDUS = {
    "ebauche": ebauche,
    "croquis": croquis,
    "aplats": aplats,
    "details": details,
    "fidele": lambda *a: details(*a, eclat=False),  # le modelé validé, fidèle à la photo
}


# ------------------------------------------------------------------ sortie
def coder(images):
    """Images de couleurs hexa → palette de lettres (K et k réservés au contour) et plages."""
    couleurs = sorted(
        {c for im in images.values() for c in im.ravel() if c} - {CONTOUR, CONTOUR_CLAIR}
    )
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    # au-delà, des lettres et des symboles d'Unicode, jamais un chiffre (les plages s'écrivent
    # lettre puis nombre ; certains chiffres d'autres écritures passeraient pour des nombres)
    lettres += [
        c
        for c in map(chr, range(0x100, 0x3000))
        if unicodedata.category(c)[0] in "LS" and not c.isdigit()
    ][: max(0, len(couleurs) - len(lettres))]
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
        poses = {
            f"{g}{s}{c}{t + 1}": (g * 0.9, s * 0.7, bool(c), t)
            for g in (0, 1)
            for s in (0, 1)
            for c in (0, 1)
            for t in (-1, 0, 1)
        }
    else:
        poses = {"0001": (0.0, 0.0, False, 0)}
    rendu = details if etape == "animation" else RENDUS[etape]
    images, coins = {}, {}
    for cle, reglage in poses.items():
        f, traits, motifs, oeil = formes(*reglage)
        images[cle] = rendu(f, traits, motifs, oeil)
        fine = PUPILLE_FINE and ECLAT_ACTIF and etape in ("details", "animation")
        if fine and not oeil["ferme"]:
            *coins[cle], images["fin " + cle] = pupille_fine(images[cle], oeil)
        elif etape in ("ebauche", "croquis") and not oeil["ferme"]:
            c = SILHOUETTE if etape == "ebauche" else CROQUIS
            *coins[cle], images["fin " + cle] = fente_fine(images[cle], oeil, c["globe"], c["pupille"])
    sprite = coder(images)
    if coins:  # les pièces en demi-pixels, à part : la planche les pose sur l'image
        sprite["fins"] = {
            cle: {"x": x, "y": y, "rangs": sprite["images"].pop("fin " + cle)}
            for cle, (x, y) in coins.items()
        }
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
