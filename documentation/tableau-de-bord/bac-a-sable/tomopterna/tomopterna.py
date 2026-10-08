"""Deux Callimedusa tomopterna face à face sur une branche, en sprites 8-bit (mode image).

    uv run python documentation/tableau-de-bord/bac-a-sable/tomopterna/tomopterna.py ETAPE
        ETAPE : silhouette | aplats | details | animation

D'après une photo d'Olivier Louguet : deux phyllomédusines tigrées perchées sur une branche
au-dessus de l'eau, celle de gauche tournée vers la droite et vers nous (trois quarts), celle de
droite de profil, tournée vers la gauche ; elles se regardent. La photo est reprise telle
quelle (pose, angle, cadrage) : on décalque, on ne réinvente pas l'anatomie.

Toile de 448 × 303 pixels : la photo de 1600 × 1081 à l'échelle 0,28, où chaque grenouille
garde un peu plus de 90 pixels de large. Les formes sont décrites dans le repère de la scène,
en unités de deux pixels de toile (7,14 pixels de photo), origine en haut à gauche de la photo.
Trois sprites, chacun dans son cadre sur la toile : la branche (derrière), la grenouille de
droite et celle de gauche. Ce qui passe derrière la branche (le pied qui pend dessous) est
découpé par la branche.

Une étape par rendu, sur les mêmes formes (modèle du skill sprite-grenouille, modele.py) :
silhouette (trois gris selon le plan, contour, œil), aplats (une couleur par matière et par
motif), details (volume, nuances, œil brillant), animation (24 images par grenouille : gorge,
flanc, paupière, tête). Écrit <sprite>_<etape>.json et la composition scene_<etape>.json, que
lit la planche du skill.
"""

import colorsys
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
TOILE = (448, 303)  # la photo de 1600 × 1081 à l'échelle 0,28
ECHELLE = 2  # pixels de toile par unité
LUMIERE = (-0.35, -0.94)  # d'en haut, un peu de la gauche, comme sur la photo
PAS = 0.06  # rotation de la tête par cran (radians)
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16


class Cadre:
    """Le rectangle d'un sprite sur la toile (en pixels, coin pair) et les coordonnées, dans le
    repère de la scène, du centre de chacun de ses pixels."""

    def __init__(self, x0, y0, largeur, hauteur):
        self.x0, self.y0, self.W, self.H = x0, y0, largeur, hauteur
        xs = (x0 + np.arange(largeur) + 0.5) / ECHELLE
        ys = (y0 + np.arange(hauteur) + 0.5) / ECHELLE
        self.X, self.Y = np.meshgrid(xs, ys)
        self.trame = np.tile(BAYER, (hauteur // 4 + 1, largeur // 4 + 1))[:hauteur, :largeur]


# ------------------------------------------------------------------ matières et couleurs
# couleurs médianes de la photo sous chaque forme, dans les parties éclairées (la photo est
# assez sombre : on garde la teinte et on prend la clarté des zones au soleil)
MATIERES = {
    "dos": "#3E9A4C",  # vert feuille : dos, tête, faces externes des membres
    "flanc": "#C98A62",  # flanc orangé
    "flanc_pale": "#D8B4A6",  # avant du flanc, blanc rosé près de la gorge
    "gorge": "#CDBDC0",  # blanc lilas, dans l'ombre de la tête
    "ventre": "#DCCBC8",  # blanc lilas, rosé (grenouille de gauche, de face)
    "levre": "#E6DAD4",
    "membre": "#D08A3C",  # faces cachées des membres, mains, pieds
    "disque": "#E3A862",  # disques des doigts, plus clairs
    "plante": "#E2A866",  # plante du pied, claire
    "bras_pale": "#D4C3CA",  # face interne du bras, pâle, tachetée
    "fond": "#B86E3E",  # membres de l'autre côté, un peu moins éclairés
    "ecorce": "#4A3B32",
}
MOTIFS = {
    "barre": "#2A0A16",  # barres et taches violet-noir des flancs et des membres
    "tache": "#7E4A5A",  # marbrure violette du ventre et de la gorge
    "tubercule": "#E6DAD4",  # rangée de tubercules clairs, granules du talon
    "lichen": "#8C8680",  # taches claires de l'écorce
    "creux": "#2A201C",  # creux sombres de l'écorce
    "bouche": "#3E2228",  # ligne de la bouche, sous la lèvre blanche
}
OEIL = {"iris": "#C2C4D4", "pupille": "#07070B", "cercle": "#17131C", "paupiere": "dos"}
CONTOUR, CONTOUR_CLAIR = "#140B10", "#3A2A20"
SILHOUETTE = {
    "fond": "#56625E",
    "corps": "#8E9B96",
    "devant": "#C3CEC9",
    "globe": "#E6ECEA",
    "pupille": "#141918",
}


# ------------------------------------------------------------------ décalque des motifs
# Les motifs de chaque individu, relevés sur la photo réduite à la toile (pixels nettement plus
# sombres ou plus clairs que leur voisinage, sous chaque forme, bord exclu), puis gardés ici
# comme un dessin : (première rangée du cadre, rangées codées par plages). B : barre ou tache
# violet-noir ; t : marbrure violette (ventre, gorge) ; l : lichen clair de l'écorce.
DECALQUE = {
    "droite": (
        28,
        [
            ".37B6.1B4.2B1.36B1.16",
            ".50B2.4B2.29B1.16",
            ".48B3.11B1.41",
            ".46B4.11B2.22B1.18",
            ".47B2.6B3.2B3.22B2.1B2.14",
            ".54B5.1B2.25B2.15",
            ".54B4.4B1.13B1.9B3.15",
            ".37B1.14B3.1B2.3B2.12B3.26",
            ".37B4.2B9.4B2.3B3.11B2.3B3.21",
            ".56B1.5B1.12B2.3B2.7B1.14",
            ".40B2.32B2.4B2.6B2.14",
            ".45B1.28B2.4B2.5B2.15",
            ".44B2.27B3.4B2.22",
            ".49B2.4B2.17B2.4B2.22",
            ".42B1.7B3.4B1.16B2.28",
            ".42B2.13B1.46",
            ".42B2.31B2.27",
            ".35B3.35B1.4B2.24",
            ".36B1.36B2.4B1.24",
            ".75B2.27",
            ".104",
            ".104",
            ".104",
            ".104",
            ".104",
            ".104",
            ".104",
            ".104",
            ".56B3.45",
            ".56B1.47",
            ".104",
            ".104",
            ".104",
            ".104",
            ".104",
            ".72B2.30",
            ".70B1.33",
            ".68B1.1B1.33",
            ".64B1.2B2.35",
            ".64B1.8B2.29",
            ".73B2.1B2.26",
            ".73B5.26",
            ".74B2.28",
        ],
    ),
    "gauche": (
        29,
        [
            ".84t2.1t3.20",
            ".81t2.2t1.1t2.21",
            ".77t4.29",
            ".59t1.10t5.2t3.30",
            ".57t6.1t11.2t1.1t1.4B3.23",
            ".57t5.2t3.2t1.3t2.8B2.25",
            ".53t2.7t2.2t1.43",
            ".40t6.6t2.56",
            ".38t4.2t1.3t2.1t4.20t1.34",
            ".52t2.8t3.9t2.7B3.24",
            ".43t1.18t2.20B1.7B1.17",
            ".43t2.47B2.16",
            ".37t3.4t4.6t2.37B1.16",
            ".24B2.11t3.5t3.6t2.2t3.9t1.1t4.10B4.3B1.16",
            ".23B3.12t3.5t1.1t4.6t3.6t6.14B4.19",
            ".22B2.14t15.6t5.1t6.5t1.11B3.3B1.15",
            ".21B2.15t1.5t2.5t5.8t3.2t1.6t1.3B2.7B2.3B1.15",
            ".20B2.2B1.20t2.16t2.11t1.13B1.3B1.15",
            ".19B2.2B2.12t1.4t1.2t2.28t2.17B1.15",
            ".18B2.3B3.9t3.4t1.1t4.4t1.22t2.13B1.3B1.15",
            ".18B1.5B2.9t4.3t2.3t2.3t3.20t2.4B3.6B2.2B2.14",
            ".20B2.2B1.9t4.4t1.9t5.18t1.6B1.8B2.1B2.14",
            ".20B3.11t3.5t1.9t3.19t2.18B1.15",
            ".21B2.19t2.30t2.18B1.15",
            ".21B1.20t4.27t3.17B1.16",
            ".16B2.18t1.5t2.9t2.18t2.1t2.5B1.9B1.16",
            ".16B1.18t2.13t2.3t2.15t2.9B1.26",
            ".17B1.17t2.12t4.18t2.10B1.6B2.18",
            ".16B2.16t4.5t3.1t1.3t2.29B2.6B2.18",
            ".16B1.17t1.1t4.5t3.15t2.17B2.26",
            ".34t1.3t4.5t2.33B1.27",
            ".39t3.17t1.50",
            ".35t2.4t2.16t1.50",
            ".34t1.1t4.8t1.6t2.53",
            ".34t2.2t3.6t3.33B1.1B2.23",
            ".39t2.4t7.31B1.2B1.23",
            ".10B1.3B1.24t1.2t6.35B2.25",
            ".10B6.23t1.2t1.67",
            ".11B4.95",
            ".11B4.95",
            ".11B3.96",
            ".12B2.96",
            ".110",
            ".110",
            ".17B1.75B1.16",
            ".14B1.2B2.74B2.15",
            ".14B1.81B1.13",
            ".16B2.1B1.76B1.13",
            ".19B1.66B2.7B2.13",
            ".86B2.7B2.13",
            ".85B3.6B3.13",
            ".89B3.2B2.14",
            ".89B7.14",
        ],
    ),
    "branche": (
        5,
        [
            ".443l2.3",
            ".448",
            ".430B1.5B2.2B1.1l1.2B2.1",
            ".424B1.1B4.13l1.4",
            ".421B3.5l2.6l1B2.5l1.2",
            ".406B1.17l1.2l1.1l3.2l3.2B3.1l1.1B1.2",
            ".398B1.2B2.3l4.10l1.1l5.1l2.2B2l1.7l1.5",
            ".393B2.4l3.3l2.3l1.1B1.5l3.2l1.2B4.1B2.2B1.6B3.1B1.1",
            ".387B2l2.7l2.9B1.5l1.6B1.4B1.5B1.14",
            ".373B2.2B1.6B4.1l1B1.1l2.7B1.2l4B2.9B1.7B2.3B2.2B1.11",
            ".365B1.8B1.1B2.6l1.3l3.1l1.3l2B1.2B1.1l1.3l6.2B8.6l2.4l1B1.6B1.1l2.1",
            ".362B1.3B2.5l1.1B2.3l1.2l1.1l2B2l1.4l1.1l2.2l3B3.1l4.3B4.9l2.4l1.3B1.4B1.5",
            ".358B5.1B3.3l2.2l3.2l1.1B1.2l2.2l2.3l1.1B2.1l1.1l2.1B2.3B1.2B1.21l1.14",
            ".342l1.1B2.4B3.9l3.1l3.1l1.1B2.5B2.1B2.4B4.7B6.2B2.2B2.4B1.30",
            ".336l1.2B3.1B6.1B2.3l2.1l2.1l5.2B2.6l2.3l1.2B5.12B3.1l2.4B4.33",
            ".341B2.1l2.2l1.4l2.2B3.5B1.3B1.7B1.12B1.5B2.1l1.1B1.1l3.42",
            ".333B2.4l3.1l5.2l2.1B2.2B2.10l1.2B4.6l2.11l2B1l1.49",
            ".308l1.19l1.5l2.1l1.4l1.2l1.1B2.1B2l1.1B4.2B1.2B1.1B1.1B2.7l1.11l1.59",
            ".328l1.5l1.1B1.1B4.4l1B3.2B5.3B1.1B1.1l1.4B3.2l1.4B2.2l1.64",
            ".292B1.12B1.2B1.19l1.5l1B3.1l1B1.1B3.1B1.1B2.1l1.1B1l1.1B3.1l2.3l2.2B2.4l1.72",
            ".284l1.2l1.4B1.13B1.3B2l1.8l1.4l1B1.1B1.10B3.1B3.3l3.1l2.11l1.80",
            ".277l1.10B1.1B1.1B1.11B2.2l5.8l1.4l1.2B1.7B1.1B3.1l1.3l2.1l1B1.1l1.1l6.87",
            ".270l2.1B3.1B2.23l1B2.2l2.12l1.7l2.4B5.2l2.1B1.2B2.98",
            ".268B1.2B1.1B1.30B2.2B1.7B2.9B2.6l1B1.1B1.3l2.2l1.101",
            ".269l1.1B1l2.2l1.1B2.22B2.4B1.4B1.11l1.1B2.6l1.1l2.109",
            ".252B1.2l1.5l1B1.5B1.1B1.2l1.1l3B3.2l1.1B2.14B1.1B2.6l1.3l3.7B1.1B2l1.118",
            ".251l1.1l1.1l1.4l3.6B3.1l1.1l2.1l2.1l2.4B1.11l1B2l3.3l1.1B2.1B1.3l1.1B1.128",
            ".241l1.1l2.4l2.11l1B1.8B3.1B1.1l2.1B4.13l1B1.2l3.2l3B1.1l1B1.2l1.131",
            ".235l4.4l1.3l1.1l1.2l1.17l2.1l1.1B4.2B1.14l2B1.1B2.1l2.1l2.1B1.138",
            ".226l2.2l7.8l2B1l1B3.7l1.2B2.5B1.2B1l1.1l1B4.15l1.1B5.146",
            ".219l1.2B1l2.1l2.1l2.10l1.1B1.1l1.1B1l1.1B1.1l1.6l1.1B1.1B1.5B4.1l2.1B1l4.7B1.158",
            ".217B1.2B1.6B2.1B1.5B1.2B1l4.1B1.2B1.1B1.7B6.7l1.3B2l1.2l1.167",
            ".203B1.4l4.4l4.2l2.1l1.3B2.1l1.2l1B1.1l1B1.1l1.1B2.1B1.2B2l1.5B6.1B1.3B1.2l2.1B1.1l1B1.170",
            ".196B3.4l2.1B2.1l7.1l4.1l8.1l1.1l1B3.1l1.2B5.4B1.6B2.1B1.2l1B3.1B1.1B1.1l1.175",
            ".192B1l1.1l1.2l7.4B2.3l3.1l2.1B1.6B8l2.4B1.2B1.3B1.1B1l1.4B2.2l4.183",
            ".192B1.8l1.1B1.2l2.2B2.2B2.4B4.1B1.3B1.1B4.1B2.1l1B2l2B1l1.1B1l1.2B8l2.187",
            ".191B1.1l1.8B1.1l2.2l2.1B1.2B6.1B2.1B8.1l1B2.2l2B1.1l3.1B1l2.1B2.196",
            ".172B1.1B1.16B1.4l1.6B2.1l1B2.1l2.2B2.2B1.2B1.7B2.1l2.1B1.1B1.1B2l1B2.1B2.201",
            ".175l1.21l1.4l1.1B4.1B3.1B1.4l4.8l1.1l2.1B1.1B4.207",
            ".168B1l3.25l1B1.1B2.2l1.1B1.5B1.5B2l2.8l3.215",
            ".167B1.1l3.3B1.21l1.2l2.1B1.2B1.1B1.1B2.2B1.3B3.6B1.220",
            ".160B1.1l2.3B1l3.1B2.22l2.2B1.1B1.4l1.3l1.6B2.228",
            ".157B1.1B3.3l1.2l3.1B2.18l1B1.1l2.1B2.3B1.1B1l1.1B2.1B1.1B1.234",
            ".155B1.4B1l5B1.2l1.1B1.15l1.6l2.1B4.1l1B4.241",
            ".150B4l5.1B2.4B3.1B1.16l1.6B6.248",
            ".148l3.2l3.1B1.2B3.2B1l1.10l1.15B2.253",
            ".145B1l1.1B1.1B1.1B3.2l1.6l1.10l1.1l1.8B2.260",
            ".140B1.2B2.2l1.1B1.2B1.3B1.4B2l1.9B2l3.270",
            ".149l1.1B1.4l2.3B1.1l1.8B3.273",
            ".116l1.1B1.1B1.1l1.25l1.1B1.7l1.4l1.284",
            ".116l3.1B2l2.26B1.4l1.3B4l2.283",
            ".105l1.11l2.1B1.3l1.24l1B1.3B1.2B4.287",
            ".100l5.13l1.1B1.2B1l2.13B2.8l1B2.2B1.293",
            ".97l3.2l4.11B2.2B3.1l1.15B3l1.5B2.296",
            ".91l3.2l1.6B1.14B1.1l1.1B1.4l1.320",
            ".87l2.2l4.8l2.13l3.1B2l1.2l1.5B1.314",
            ".77l2.7l2.4l1.1l2.2B1.1B1.2B3.1l1.11l2B3.1l1.2l1.7B1.311",
            ".76l2.1B1.5l3.5B2.1B10.1l1.12l1.1B4.1B1.6l1.313",
            ".64l1.2l3.1l3.5l1.1l3B1l1.1l1.2B2.2B1.1B5.2l1.1B1.14l1.1B1.1B5.319",
            ".57B1l2.1l3.9l1.2l1.2l1.1l1.1B2.1l2.1l1.1B2.1B6.1l3.1l1B2.13l1.1B3.322",
            ".62B2.4l1.1l1.2l1.4B1.1B2.2l1.5B1.1l2.1l1.1B2.1B2.1l1.2B2.1l1.9B2l1.326",
            ".39B3.4l2.2l2.2l2.8l1.2l1.2B1l1.2B4.1B1.3l1.1B2.1l2B2.1l4B1.1B4l1.1B3.1l1.338",
            ".34B2.5l6.7l1.2l3.2l2.1l1B10.4l1.5B3l1.1B2.1l1.2B5.1l1.1l1.1B1.340",
            ".27B2.1B1.5l2.11l1.2l2.2l5.1B1.2B2.1B5.3l1.3l1.4B6.2B3.1B2.1l1.1l2.344",
            ".25l1.8l2.8l2.1l4B1.2B3.5B4.1l1.1B2l2B1.1l2.1l2.2B1.1B1.1l1B3.1B3.2l1.1l1.349",
            ".5B5.10l1.2l2.6l1.2l1.7l2.7B2l1.1B6.2B3.1l2.2l2B2l1.1B1.4B2.3B2.1B2.356",
            ".2l2.4l1.4B1.16l2.4B2.3l2.1B2.9B2.1B1.1B3.1l1.1l1.1l1.4B3.2B3.2B1.364",
            ".1l2.10l1.13l1.2l1.2l2.3B1.3l2.3l1.1B2.2B1.1l1.5l1.4B1.2B2.2B2.373",
            ".10l5.11l1.1B1.5l1.5B2.3B2.2B1.2B2l2.392",
            ".14l1B2.1l2.1B3.424",
            ".2B1.4B1.8B1.1B1.1B3.425",
        ],
    ),
}


def decalque(nom, c, xt, yt):
    """Les motifs relevés de ce sprite, lus aux coordonnées (xt, yt) : ils suivent la tête."""
    j0, rangs = DECALQUE[nom]
    g = np.full((c.H, c.W), ".")
    for dj, r in enumerate(rangs):
        g[j0 + dj] = list("".join(ch * int(n) for ch, n in re.findall(r"(\D)(\d+)", r)))
    i = np.clip(np.floor(xt * ECHELLE).astype(int) - c.x0, 0, c.W - 1)
    j = np.clip(np.floor(yt * ECHELLE).astype(int) - c.y0, 0, c.H - 1)
    return g[j, i]


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
        a, b = np.array(pts[k], float), np.array(pts[k + 1], float)
        ra, rb = rayons[k], rayons[k + 1]
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        m |= np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]) <= ra + (rb - ra) * t
    return m


def doigt(c, pts):
    """Un doigt posé au pixel près, d'articulation en articulation (unités) : un trait de deux
    pixels, en escalier (chaque pixel touche le suivant par un côté, sinon le contour le coupe),
    et son disque au bout, de 3 × 3. Les doigts des phyllomédusines sont épais, à grands disques."""
    m = np.zeros((c.H, c.W), bool)
    q = [(x * ECHELLE - c.x0, y * ECHELLE - c.y0) for x, y in pts]

    def poser(i, j):
        if 0 <= j < c.H and 0 <= i < c.W:
            m[j, i] = True

    for (x0, y0), (x1, y1) in zip(q, q[1:], strict=False):
        horizontal = abs(x1 - x0) >= abs(y1 - y0)
        n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 1
        prec = None
        for k in range(n + 1):
            i = math.floor(x0 + (x1 - x0) * k / n)
            j = math.floor(y0 + (y1 - y0) * k / n)
            cases = [(i, j)]
            if prec and prec[0] != i and prec[1] != j:
                cases.append((i, prec[1]))  # la marche de l'escalier
            for a, b in cases:
                poser(a, b)
                if horizontal:
                    poser(a, b + 1)
                else:
                    poser(a + 1, b)
            prec = (i, j)
    bi, bj = (math.floor(v) for v in q[-1])
    disque = np.zeros((c.H, c.W), bool)
    disque[max(bj - 1, 0) : bj + 2, max(bi - 1, 0) : bi + 2] = True
    return m | disque, disque


def suivre(pts, x):
    xs, ys = zip(*pts, strict=True)
    return np.interp(x, xs, ys)


def forme(nom, masque, matiere, plan="corps", cernee=False, volume=None, lum=None):
    """plan : fond (membres de l'autre côté, ce qui pend derrière la branche), corps, devant
    (membres proches) ; cernee : un contour la sépare de ce qu'elle recouvre ; volume : la forme
    dont elle prend le relief (le flanc prend celui du corps) ; lum : la clarté imposée pixel
    par pixel (un gabarit relevé sur la photo), au lieu du relief calculé."""
    return {
        "nom": nom,
        "masque": masque,
        "matiere": matiere,
        "plan": plan,
        "cernee": cernee,
        "volume": volume or nom,
        "lum": lum,
    }


def tete_tournee(c, pivot, sens, tete):
    """Coordonnées vues par la tête, tournée de tete × PAS autour du cou (pivot) ; sens = +1
    pour une tête à gauche, -1 à droite. L'effet s'estompe vers le corps."""
    poids = np.clip((sens * (pivot[0] - c.X) + 3) / 6, 0, 1) * np.clip(
        (pivot[1] + 4 - c.Y) / 4, 0, 1
    )
    a = -sens * tete * PAS * poids
    dx, dy = c.X - pivot[0], c.Y - pivot[1]
    return pivot[0] + np.cos(a) * dx - np.sin(a) * dy, pivot[1] + np.sin(a) * dx + np.cos(a) * dy


# ------------------------------------------------------------------ la branche
# ligne médiane (unités) et demi-épaisseur, relevées sur la photo
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


def branche(x, y):
    """Masque de la branche, aux coordonnées de la scène."""
    return np.abs(y - suivre(BRANCHE, x)) <= suivre(EPAISSEUR, x)


def formes_branche(c, **_):
    m = branche(c.X, c.Y)
    d = decalque("branche", c, c.X, c.Y)
    return (
        [forme("branche", m, "ecorce")],
        {"lichen": m & (d == "l"), "creux": m & (d == "B")},
        None,
    )


# ------------------------------------------------------------------ la grenouille de droite
# de profil, tournée vers la gauche
DROITE = {"cadre": (236, 50, 104, 80), "pivot": (138.0, 36.0), "sens": 1}

# pied droit de la grenouille de droite, sous la branche, relevé au pixel sur la photo réduite :
# (coin haut gauche sur la toile, rangées). Z, S, O : du plus sombre au plus clair (le pied qui
# part en diagonale, dans l'ombre) ; H, I : la plante, claire, en éventail, et le disque de
# l'orteil qui part dans le vide à droite ; K : la tache sombre
PIED_DROIT = (
    (296, 113),
    [
        "...............SKKKKOZ..",
        "........ZKKZSSZOKKKKZ...",
        ".......ZKZZSOOKZZKKKSKS.",
        "...SKZSZZSSSHSKZZKZSZ...",
        "..ZKZSSSSSIHOKKSZKZZSS..",
        "..SZSOOOOOIIOKKSZKSSZ...",
        "..SKZOOHOHOOOKKZZS......",
        "..SZKOHHHHHHHOSSHIO.....",
        "...OKOIHHHHIHIIHIII.....",
        "...SOOHOHOOHHIIHIII.....",
        "....SOSOOOOOOHHOHIH.....",
    ],
)
CLARTE_PIED = {"Z": 0.05, "S": 0.25, "O": 0.45, "H": 0.75, "I": 0.98, "K": 0.1}


def gabarit(c, gab):
    """Les pixels d'un gabarit posé sur la toile, par code : {code: masque du cadre}."""
    (x0, y0), rangs = gab
    out = {}
    for dj, rang in enumerate(rangs):
        for di, ch in enumerate(rang):
            j, i = y0 + dj - c.y0, x0 + di - c.x0
            if ch != "." and 0 <= j < c.H and 0 <= i < c.W:
                out.setdefault(ch, np.zeros((c.H, c.W), bool))[j, i] = True
    return out


def bord(pts, rayons, X, Y, dx, dy):
    """Le liseré d'un membre du côté (dx, dy) : ses pixels dont le voisin décalé n'en est pas."""
    return membre(pts, rayons, X, Y) & ~membre(pts, rayons, X + dx, Y + dy)


def epaule(c, cx, cy, rx, ry):
    """L'épaule s'attache en arrondi : un capuchon dont le pourtour se fond dans le corps en
    tramage ordonné (Bayer), sans trait."""
    r = np.sqrt(((c.X - cx) / rx) ** 2 + ((c.Y - cy) / ry) ** 2)
    return (r <= 1) & (c.trame < np.clip((1 - r) / 0.45, 0, 1))


def formes_droite(c, gorge=0.0, souffle=0.0, cligne=False, tete=0):
    X, Y = c.X, c.Y
    xt, yt = tete_tournee(c, DROITE["pivot"], 1, tete)
    # tête plus petite et museau plus long : le crâne passe bas, l'œil globuleux dépasse
    # nettement ; petite bosse de l'épaule droite sous la poitrine
    corps = dans(
        catmull(
            [
                (121.5, 35.9),
                (121.5, 34.4),
                (121.6, 32.8),
                (122.1, 32.0),
                (123.2, 31.6),
                (125.0, 31.1),
                (126.6, 30.6),
                (128.6, 30.1),
                (131.4, 29.9),
                (134.2, 30.1),
                (136.6, 30.5),
                (139.0, 30.9),
                (141.6, 31.3),
                (145.0, 32.2),
                (148.2, 33.4),
                (151.5, 35.1),
                (154.8, 37.1),
                (157.2, 39.2),
                (158.6, 41.2),
                (159.4, 43.6),
                (159.0, 46.6),
                (156.4, 47.8 + souffle * 0.4),
                (151.0, 47.8 + souffle * 0.6),
                (146.0, 47.4 + souffle * 0.6),
                (142.0, 46.6 + souffle * 0.4),
                (139.0, 45.4),
                (136.4, 44.5),
                (134.8, 43.7 + gorge * 0.5),
                (133.4, 42.2 + gorge * 0.9),
                (132.2, 40.5 + gorge),
                (129.6, 38.7 + gorge * 0.6),
                (126.0, 37.6 + gorge * 0.2),
                (123.0, 36.9),
            ]
        ),
        xt,
        yt,
    )
    # l'œil, globuleux, et sa paupière font saillie sur le crâne
    corps |= ellipse(131.3, 31.6, 4.15, 3.9, xt, yt) & (yt < 32.4)
    limite = suivre(
        [
            (120.0, 35.9),
            (125.0, 36.6),
            (129.8, 37.2),
            (134.0, 37.7),
            (137.8, 38.3),
            (141.0, 38.7),
            (147.0, 39.0),
            (151.5, 39.4),
            (155.0, 40.2),
            (158.0, 41.5),
            (161.0, 43.0),
        ],
        xt,
    )
    dessous = corps & (yt >= limite)
    gorge_ = dessous & (xt < 135.6)
    # ligne de la mâchoire : la lèvre blanche et la bouche sombre dessous, du museau jusqu'au-
    # dessus de l'épaule
    levre = dessous & (yt < limite + 0.75) & (xt < 141.5)
    bouche = dessous & (yt >= limite + 0.75) & (yt < limite + 1.25) & (xt > 122.0) & (xt < 140.4)
    alea = np.random.default_rng(29).random(X.shape)
    avant = dessous & ~gorge_ & (xt < 141.0 + 1.2 * np.sin(yt * 2.1) + 0.6 * (yt - 41) + 1.6 * alea)
    # patte arrière gauche : la cuisse, verte, en arrière du tibia mais devant le postérieur, se
    # voit des deux côtés du tibia ; le genou levé, granuleux ; le tibia barré redescend vers
    # la branche ; trois orteils en éventail sur la branche
    cuisse = dans(
        [
            (156.2, 42.2),
            (157.4, 40.4),
            (159.4, 39.0),
            (162.0, 38.5),
            (164.6, 38.9),
            (166.2, 40.6),
            (166.5, 44.0),
            (166.2, 47.8),
            (163.6, 48.2),
            (160.6, 47.8),
            (158.4, 47.0),
            (156.8, 45.2),
        ],
        X,
        Y,
    )
    genou = ellipse(161.6, 39.4, 2.4, 1.8, X, Y)
    tibia = membre([(161.6, 40.6), (162.0, 43.6), (162.4, 46.8)], [1.9, 1.7, 1.3], X, Y)
    orteils, disques_o = np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for pts in (
        [(162.2, 46.6), (160.0, 48.6), (158.0, 50.6)],
        [(162.4, 46.8), (161.6, 50.4), (161.1, 53.6)],
        [(162.8, 46.6), (164.8, 49.2), (166.2, 52.6)],
    ):
        o, d = doigt(c, pts)
        orteils |= o
        disques_o |= d
    # bras gauche : l'avant-bras, gros, vert, descend du coude (en haut) au poignet sur la
    # branche ; le bras, fin, part du coude à l'horizontale vers l'avant, sous la limite du dos
    # (face interne pâle, tachetée), et s'implante dans l'épaule, une légère bosse
    avant_bras = membre(
        [(152.4, 40.0), (151.6, 44.6), (149.8, 49.6), (148.6, 52.6)], [1.5, 2.2, 1.8, 1.2], X, Y
    )
    pts_bh = [(141.4, 40.7), (146.0, 40.5), (151.6, 40.2)]
    bras_haut = membre(pts_bh, [1.0, 0.85, 0.95], X, Y)
    epaule_g = epaule(c, 140.8, 41.1, 1.6, 1.4)
    main, disques_m = ellipse(146.8, 53.2, 1.5, 1.0, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(146.0, 52.8), (143.4, 51.8), (140.8, 51.6)],
        [(146.0, 53.6), (142.4, 55.6), (139.2, 58.2)],
        [(146.6, 53.8), (145.6, 56.6), (145.0, 59.2)],
    ):
        o, d = doigt(c, pts)
        main |= o
        disques_m |= d
    # 4e doigt de la main gauche : discret, il monte collé à la branche, derrière l'avant-bras
    doigt_g, disque_g = doigt(c, [(147.4, 53.0), (148.4, 51.6), (149.2, 50.2)])
    # bras droit, de l'autre côté : l'avant-bras part du poignet en arrière-plan, vers la droite
    # et un peu vers le haut, jusqu'au coude, sous le ventre ; le bras, fin, remonte vers le
    # haut et la gauche s'attacher à l'épaule, une très légère bosse sous la poitrine
    avant_bras_f = membre([(130.2, 49.9), (134.6, 47.4), (139.2, 45.9)], [1.05, 1.45, 1.1], X, Y)
    bras_haut_f = membre([(139.2, 45.8), (137.0, 44.6), (135.4, 43.7)], [0.75, 0.7, 0.8], X, Y)
    main_fond, disques_f = ellipse(131.4, 50.8, 1.4, 1.0, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(130.4, 50.8), (127.2, 51.4), (124.6, 52.6)],
        [(130.6, 51.4), (128.6, 54.0), (127.4, 56.6)],
        [(131.6, 51.4), (132.6, 53.4), (133.2, 55.6)],
        [(132.4, 50.6), (136.0, 50.4), (139.2, 50.7)],  # 4e doigt, le long de la branche
    ):
        o, d = doigt(c, pts)
        main_fond |= o
        disques_f |= d
    # pied droit, sous la branche : relevé au pixel (PIED_DROIT), découpé par la branche
    pied = gabarit(c, PIED_DROIT)
    vide = np.zeros(X.shape, bool)
    sous_branche = ~branche(X, Y)
    ombre_pied = (pied.get("Z", vide) | pied.get("S", vide) | pied.get("O", vide)) & sous_branche
    plante = (pied.get("H", vide) | pied.get("I", vide)) & sous_branche
    clarte_pied = np.zeros(X.shape)
    for ch, val in CLARTE_PIED.items():
        clarte_pied[pied.get(ch, vide)] = val
    f = [
        forme(
            "pied_fond",
            ombre_pied | (pied.get("K", vide) & sous_branche),
            "fond",
            "fond",
            cernee=True,
            lum=clarte_pied,
        ),
        forme("plante", plante, "plante", "fond", lum=clarte_pied),
        forme("bras_fond", bras_haut_f, "fond", "fond", cernee=True),
        forme("avant_bras_fond", avant_bras_f | main_fond, "fond", "fond", cernee=True),
        forme("corps", corps | ellipse(127.0, 30.0, 1.4, 1.0, xt, yt), "dos"),
        forme("flanc", dessous & ~gorge_ & ~avant, "flanc", volume="corps"),
        forme("flanc_pale", avant, "flanc_pale", volume="corps"),
        forme("gorge", gorge_ & ~levre, "gorge", volume="corps"),
        forme("levre", levre, "levre", volume="corps"),
        forme("cuisse", cuisse, "dos", cernee=True),
        forme("tibia", tibia | genou, "membre", cernee=True),
        forme("orteils", orteils, "membre", "devant", cernee=True),
        forme("epaule", epaule_g & corps & ~bras_haut, "bras_pale", volume="bras"),
        forme("bras", bras_haut, "bras_pale", "devant", cernee=True),
        forme("doigt_g", doigt_g, "membre", "devant", cernee=True),
        forme("avant_bras", avant_bras, "dos", "devant", cernee=True),
        forme("main", main, "membre", "devant", cernee=True),
    ]
    d = decalque("droite", c, xt, yt)
    # tubercules clairs : une rangée, un pixel sur deux, entre le vert du dos et le flanc (le
    # dessus du bras) ; quelques granules sur le genou
    pair = (np.floor(X * ECHELLE) + np.floor(Y * ECHELLE)) % 2 == 0
    rangee = dessous & ~gorge_ & ~levre & (yt < limite + 0.5) & (xt < 155) & pair
    granules = genou & (np.random.default_rng(31).random(X.shape) < 0.2) & (X < 163.0)
    motifs = {
        "barre": ((d == "B") & ~levre & ~bouche & ~(cuisse & ~tibia & ~genou))
        | (pied.get("K", vide) & sous_branche),
        "tache": d == "t",
        "tubercule": rangee | granules,
        "bouche": bouche,
    }
    oeil = oeil_de(
        xt, yt, 131.3, 31.5, 3.5, 3.5, cligne, dome=(127.0, 30.0, 1.4, 1.0), pupille=-0.24
    )
    oeil["narine"] = ellipse(122.9, 32.9, 0.4, 0.35, xt, yt)
    oeil["dome_globe"] = ellipse(127.2, 29.6, 0.85, 0.55, xt, yt)
    oeil["dome_pupille"] = np.zeros(X.shape, bool)
    # traits du visage : arête du museau (de la narine à l'œil), rebord de la narine
    oeil["arete"] = membre([(122.6, 32.2), (125.2, 31.3), (127.6, 30.8)], [0.3] * 3, xt, yt)
    oeil["arete"] |= ellipse(123.0, 32.5, 0.6, 0.35, xt, yt)
    oeil["joue"] = corps & (yt > 33.4) & (yt < limite - 0.3) & (xt < 128.2) & (xt > 122.2)
    oeil["disques"] = disques_o | disques_m | disques_f | disque_g
    oeil["coudes"] = ellipse(152.6, 39.8, 1.0, 0.8, X, Y) | ellipse(139.4, 45.8, 0.8, 0.7, X, Y)
    oeil["ombres"] = (membre(pts_bh, [1.5, 1.4, 1.5], X, Y - 0.9) & corps & ~bras_haut) | (
        cuisse & ~membre([(161.6, 40.6), (162.0, 43.6)], [2.6, 2.4], X, Y) & (X < 161)
    )
    return f, motifs, oeil


# ------------------------------------------------------------------ la grenouille de gauche
# presque de face, tournée vers la droite : sa droite est à gauche sur l'image
GAUCHE = {"cadre": (96, 60, 110, 90), "pivot": (78.0, 42.0), "sens": -1}


def formes_gauche(c, gorge=0.0, souffle=0.0, cligne=False, tete=0):
    X, Y = c.X, c.Y
    xt, yt = tete_tournee(c, GAUCHE["pivot"], -1, tete)
    # lignes marquées, presque droites : dos, museau tronqué, mâchoire ; polygone simple, sans
    # lissage. Le ventre fuit à gauche derrière l'avant-bras droit et à droite derrière la
    # patte arrière gauche
    corps = dans(
        [
            (99.9, 41.8),
            (99.6, 39.6),
            (99.1, 37.2),
            (98.4, 36.6),
            (96.8, 36.1),
            (95.2, 35.5),
            (92.4, 34.7),
            (90.2, 33.8),
            (87.0, 33.2),
            (84.0, 33.6),
            (82.4, 34.4),
            (66.6, 45.4),
            (65.0, 47.8),
            (64.6, 50.4),
            (64.4, 54.0),
            (64.6, 58.0),
            (65.6, 61.4),
            (67.4, 62.6),
            (70.0, 63.2 + souffle * 0.3),
            (73.0, 63.0 + souffle * 0.6),
            (77.0, 61.8 + souffle * 0.6),
            (80.6, 60.6 + souffle * 0.4),
            (83.4, 59.2),
            (86.4, 58.4),
            (88.4, 55.8),
            (89.2, 52.2),
            (89.6, 49.0 + gorge * 0.3),
            (91.0, 46.6 + gorge * 0.9),
            (94.5, 45.3 + gorge),
            (97.4, 44.0 + gorge * 0.5),
            (99.6, 42.8),
        ],
        xt,
        yt,
    )
    corps |= ellipse(87.15, 37.6, 4.3, 4.2, xt, yt) & (yt < 37.6)  # l'œil fait saillie
    limite = suivre(
        [
            (60.0, 52.6),
            (64.0, 50.0),
            (68.0, 47.6),
            (72.0, 46.1),
            (77.8, 45.3),
            (84.8, 44.4),
            (89.5, 43.5),
            (95.0, 42.7),
            (101.0, 41.8),
        ],
        xt,
    )
    dessous = corps & (yt >= limite)
    gorge_ = dessous & (xt > 86.0) & (yt < 47.5)
    levre = dessous & (xt > 78.0) & (yt < limite + 0.75)
    bouche = dessous & (yt >= limite + 0.75) & (yt < limite + 1.25) & (xt > 79.0) & (xt < 99.4)
    # les côtés du ventre qui fuient : peau orangée, dans l'ombre
    cote_d = dessous & (X < 67.6) & (Y > 49.0)
    cote_g = dessous & (X > 84.4) & (Y > 50.4)
    pli = dessous & ~cote_d & (Y > 59.4) & (X < 71.0)
    # patte arrière droite (à gauche) : la cuisse descend en diagonale vers la gauche jusqu'au
    # genou, vert et granuleux ; le tibia, gros, horizontal, revient du genou vers la droite et
    # passe derrière l'avant-bras droit ; deux gros orteils barrés s'enroulent sur la branche
    pts_c, r_c = [(62.6, 52.2), (58.8, 55.2), (55.4, 58.0)], [1.3, 1.4, 1.5]
    cuisse_d = membre(pts_c, r_c, X, Y)
    cuisse_d_vert = bord(pts_c, r_c, X, Y, -0.8, -0.8)
    genou_d = ellipse(53.9, 60.0, 1.8, 2.0, X, Y)
    tibia_d = membre([(54.6, 59.8), (58.0, 59.7), (61.8, 59.5)], [2.0, 2.3, 2.1], X, Y)
    orteils_d = membre([(56.2, 61.6), (56.5, 66.0), (56.7, 70.2)], [1.15, 1.2, 1.05], X, Y)
    orteils_d |= membre([(59.4, 61.8), (59.7, 66.2), (60.0, 70.6)], [1.15, 1.2, 1.05], X, Y)
    disques_od = ellipse(56.8, 70.8, 1.3, 1.0, X, Y) | ellipse(60.1, 71.4, 1.3, 1.0, X, Y)
    # bras droit : l'avant-bras, vert, descend du coude (en haut à gauche) à la main ; le bras,
    # fin, part du coude à l'horizontale et s'implante dans l'épaule, une boule orangée
    avant_bras_d = membre([(63.2, 50.6), (62.9, 57.5), (63.1, 64.4)], [1.7, 1.65, 1.5], X, Y)
    pts_bh = [(64.4, 50.0), (65.8, 50.1), (67.2, 50.3)]
    bras_haut_d = membre(pts_bh, [0.6, 0.55, 0.6], X, Y)
    epaule_d = ellipse(66.8, 52.3, 1.7, 1.5, X, Y)
    # barres noires : trois en travers du tibia, trois en travers de la cuisse
    barres_tibia = tibia_d & (
        (np.abs(X - 56.4) < 0.55) | (np.abs(X - 58.7) < 0.6) | (np.abs(X - 60.8) < 0.45)
    )
    barres_cuisse = cuisse_d & (
        (np.abs(X - 61.2 + (Y - 53.0) * 0.75) < 0.5) | (np.abs(X - 58.6 + (Y - 55.2) * 0.75) < 0.5)
    )
    main_d, disques_m = ellipse(64.9, 65.6, 1.5, 1.1, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(65.4, 65.4), (69.4, 65.2), (73.4, 66.0)],
        [(65.4, 66.2), (69.6, 67.8), (74.2, 69.2)],
        [(65.0, 66.6), (66.6, 68.6), (68.4, 70.6)],
    ):
        o, d = doigt(c, pts)
        main_d |= o
        disques_m |= d
    # le pouce, quatrième doigt, court et gros, part vers l'arrière, collé à la branche
    pouce = membre([(64.8, 64.8), (65.9, 63.5), (66.7, 62.5)], [0.85, 0.85, 0.9], X, Y)
    # patte arrière gauche (à droite), en arrière-plan, à l'ombre : colonne barrée bordée de vert
    # qui descend vers la branche ; deux orteils, un vers la gauche, un vers le bas
    pts_j, r_j = [(88.6, 50.6), (89.8, 56.0), (90.8, 62.0)], [1.3, 1.45, 1.25]
    jambe_g = membre(pts_j, r_j, X, Y)
    jambe_g_vert = bord(pts_j, r_j, X, Y, -0.9, 0.0)
    orteil_g1, disque_g1 = doigt(c, [(89.8, 61.6), (86.4, 63.6), (83.2, 64.9)])
    orteil_g2, disque_g2 = doigt(c, [(90.4, 62.2), (89.4, 64.6), (88.6, 66.6)])
    # bras gauche (à droite), devant : il vient vers nous et paraît court ; l'avant-bras
    # granuleux, bordé de vert, descend du coude à la main, qui a trois longs doigts
    pts_ab, r_ab = [(90.4, 48.0), (93.4, 51.0), (94.6, 55.0), (93.8, 59.0)], [1.7, 2.2, 1.7, 1.3]
    avant_bras_g = membre(pts_ab, r_ab, X, Y)
    avant_bras_g_vert = bord(pts_ab, r_ab, X, Y, 1.0, 0.0) & (Y < 56)
    main_g, disques_g = ellipse(93.8, 60.4, 1.4, 1.0, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(94.4, 60.2), (96.0, 60.0), (97.3, 60.4)],
        [(94.2, 61.0), (94.4, 62.4), (94.4, 63.8)],
        [(93.2, 61.0), (92.2, 63.4), (91.6, 66.2)],
    ):
        o, d = doigt(c, pts)
        main_g |= o
        disques_g |= d
    f = [
        forme("cuisse_d", cuisse_d, "membre", "fond", cernee=True),
        forme("cuisse_d_vert", cuisse_d_vert, "dos", "fond", volume="cuisse_d"),
        forme("orteils_d", orteils_d | disques_od, "membre", "fond", cernee=True),
        forme("tibia_d", tibia_d, "membre", "fond", cernee=True),
        forme("genou_d", genou_d, "dos", "fond", cernee=True),
        forme("jambe_g", jambe_g | orteil_g1 | orteil_g2, "fond", "fond", cernee=True),
        forme("jambe_g_vert", jambe_g_vert, "dos", "fond", volume="jambe_g"),
        forme("corps", corps | ellipse(93.4, 34.4, 3.4, 1.9, xt, yt), "dos"),
        forme(
            "ventre",
            dessous & ~gorge_ & ~levre & ~cote_d & ~cote_g & ~pli,
            "ventre",
            volume="corps",
        ),
        forme("cotes", (cote_d | cote_g | pli) & ~levre, "flanc", volume="corps"),
        forme("gorge", gorge_ & ~levre, "gorge", volume="corps"),
        forme("levre", levre, "levre", volume="corps"),
        forme("epaule_d", epaule_d, "membre", cernee=True),
        forme("pouce", pouce, "membre", cernee=True),
        forme("avant_bras_d", avant_bras_d, "dos", "devant", cernee=True),
        forme("bras_haut_d", bras_haut_d, "bras_pale", "devant"),
        forme("main_d", main_d, "membre", "devant", cernee=True),
        forme("avant_bras_g", avant_bras_g | main_g, "membre", "devant", cernee=True),
        forme("avant_bras_g_vert", avant_bras_g_vert, "dos", "devant", volume="avant_bras_g"),
    ]
    d = decalque("gauche", c, xt, yt)
    granules = genou_d & (np.random.default_rng(37).random(X.shape) < 0.22)
    motifs = {
        "barre": ((d == "B") & ~levre & ~genou_d & ~bras_haut_d & ~avant_bras_d & ~tibia_d)
        | barres_tibia
        | barres_cuisse,
        "tache": (d == "t") & ~cote_d & ~cote_g,
        "tubercule": granules,
        "bouche": bouche,
    }
    oeil = oeil_de(xt, yt, 87.15, 37.9, 3.75, 3.8, cligne, dome=(93.4, 34.4, 3.4, 1.9), pupille=0.1)
    oeil["narine"] = ellipse(96.8, 38.2, 0.4, 0.35, xt, yt)
    # l'autre œil dépasse du crâne : on voit un peu de son globe pâle et de sa pupille
    oeil["dome_globe"] = ellipse(95.5, 34.5, 1.5, 1.35, xt, yt)
    oeil["dome_pupille"] = (np.abs(xt - 96.25) < 0.26) & (np.abs(yt - 34.6) < 0.55)
    # arête du museau, droite, de l'œil à la narine ; rebord de la narine ; coin du museau
    oeil["arete"] = membre([(90.8, 35.8), (93.8, 36.6), (96.4, 37.6)], [0.3] * 3, xt, yt)
    oeil["arete"] |= ellipse(96.8, 37.6, 0.6, 0.35, xt, yt) | membre(
        [(98.2, 36.8), (99.3, 37.6)], [0.3, 0.3], xt, yt
    )
    oeil["joue"] = corps & (yt > 39.6) & (yt < limite - 0.3) & (xt > 90.0)
    oeil["disques"] = disques_m | disques_g | disques_od | disque_g1 | disque_g2
    oeil["coudes"] = ellipse(90.6, 47.9, 1.0, 0.8, X, Y) | ellipse(62.6, 50.2, 1.0, 0.7, X, Y)
    oeil["ombres"] = (
        (cote_d | cote_g | pli)
        | (membre(pts_bh, [1.5, 1.4, 1.5], X, Y - 0.9) & corps & ~bras_haut_d)
        | (tibia_d & (X > 60.4))
    )
    return f, motifs, oeil


def oeil_de(xt, yt, cx, cy, rx, ry, ferme, dome, pupille=0.0):
    """L'œil visible (globe pâle sur le pourtour, cerne noir, iris, pupille en fente décalée de
    `pupille` rayon) et la bosse de l'autre œil."""
    u, v = (xt - cx) / rx, (yt - cy) / ry
    return {
        "x": cx,
        "y": cy,
        "masque": u * u + v * v <= 1,
        "u": u,
        "v": v,
        "ferme": ferme,
        "dome": ellipse(*dome, xt, yt),
        "pupille": pupille,
        "rx": rx,
    }


def fente(oeil):
    """La pupille en fente, posée au pixel : une colonne sur toute la hauteur, trois au milieu,
    centrée sur la colonne la plus proche de son axe (pas d'escalier quand la tête tourne)."""
    u, v = oeil["u"], oeil["v"]
    col = np.abs(np.floor((u - oeil["pupille"]) * oeil["rx"] * ECHELLE + 0.5))
    return oeil["masque"] & (np.abs(v) < 0.52) & (col <= np.where(np.abs(v) < 0.24, 1, 0))


SPRITES = {
    "branche": ((0, 76, 448, 84), formes_branche),
    "droite": (DROITE["cadre"], formes_droite),
    "gauche": (GAUCHE["cadre"], formes_gauche),
}


# ------------------------------------------------------------------ rendus
def etiqueter(c, f):
    """Matière, plan et numéro de forme de chaque pixel (le plus proche l'emporte)."""
    mat = np.full((c.H, c.W), "", object)
    plan = np.full((c.H, c.W), "", object)
    num = np.full((c.H, c.W), -1)
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


def miettes(img, n=4):
    """Les groupes de moins de n pixels isolés par les contours (voisins par un côté)
    deviennent du contour : sinon un pixel flotte, seul, entre deux traits."""
    plein = (img != "") & (img != CONTOUR)
    vus = np.zeros(plein.shape, bool)
    h, w = img.shape
    for j0, i0 in zip(*np.nonzero(plein), strict=False):
        if vus[j0, i0]:
            continue
        pile, groupe = [(j0, i0)], []
        vus[j0, i0] = True
        while pile:
            j, i = pile.pop()
            groupe.append((j, i))
            for v in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if 0 <= v[0] < h and 0 <= v[1] < w and plein[v] and not vus[v]:
                    vus[v] = True
                    pile.append(v)
        if len(groupe) < n:
            for p in groupe:
                img[p] = CONTOUR
    return img


def rendu_oeil(img, oeil, couleurs, ferme_couleur, bosse_couleur):
    if oeil is None:  # la branche n'a pas d'œil
        return img
    img[oeil["dome"] & (img != "")] = bosse_couleur
    img[oeil["narine"]] = CONTOUR
    m, u, v = oeil["masque"], oeil["u"], oeil["v"]
    if oeil["ferme"]:
        img[m] = ferme_couleur
        img[m & (np.abs(v - 0.15 * u * u) < 0.18)] = CONTOUR
        return img
    img[m] = couleurs["iris"]
    img[m & (u * u + v * v > 0.78)] = couleurs.get("cercle", CONTOUR)
    img[fente(oeil)] = couleurs["pupille"]
    return img


def silhouette(c, f, motifs, oeil):
    mat, plan, num = etiqueter(c, f)
    img = np.full((c.H, c.W), "", object)
    for nom, coul in SILHOUETTE.items():
        img[plan == nom] = coul
    img = contour(img, f, num)
    return rendu_oeil(
        img,
        oeil,
        {"iris": SILHOUETTE["globe"], "pupille": SILHOUETTE["pupille"]},
        SILHOUETTE["corps"],
        SILHOUETTE["corps"],
    )


def aplats(c, f, motifs, oeil):
    mat, plan, num = etiqueter(c, f)
    img = np.full((c.H, c.W), "", object)
    for nom, coul in MATIERES.items():
        img[mat == nom] = coul
    if oeil is not None:
        img[oeil["disques"] & (num >= 0)] = MATIERES["disque"]
    for nom, m in motifs.items():
        img[m & (num >= 0)] = MOTIFS[nom]
    img = contour(img, f, num)
    img = rendu_oeil(img, oeil, OEIL, MATIERES[OEIL["paupiere"]], MATIERES["dos"])
    if oeil is not None and not oeil["ferme"]:
        img[oeil["dome_globe"] & oeil["dome"] & (num >= 0)] = OEIL["iris"]
        img[oeil["dome_pupille"] & (num >= 0)] = OEIL["pupille"]
    return img


def _flou(a, r):
    for axe in (0, 1):
        a = sum(np.roll(a, k, axis=axe) for k in range(-r, r + 1)) / (2 * r + 1)
    return a


def hexa_de(r, g, b):
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


# étendue des rampes (clarté de l'ombre et de la lumière, autour de la couleur à plat), nombre
# de tons, et si la matière prend les trois variantes de teinte par plaques
RAMPES = {
    "dos": (-0.26, 0.26, 10, True),
    "flanc": (-0.26, 0.17, 9, True),
    "flanc_pale": (-0.26, 0.1, 8, True),
    "membre": (-0.28, 0.18, 9, True),
    "fond": (-0.2, 0.1, 7, True),
    "plante": (-0.16, 0.1, 5, False),
    "bras_pale": (-0.24, 0.1, 7, True),
    "gorge": (-0.26, 0.1, 8, True),
    "ventre": (-0.26, 0.1, 8, True),
    "levre": (-0.14, 0.06, 4, False),
    "disque": (-0.16, 0.1, 4, False),
    "ecorce": (-0.14, 0.12, 6, True),
    "barre": (-0.04, 0.1, 3, False),
    "tache": (-0.1, 0.1, 4, False),
    "tubercule": (-0.12, 0.04, 3, False),
    "lichen": (-0.12, 0.08, 3, False),
    "creux": (-0.04, 0.04, 2, False),
    "bouche": (-0.04, 0.06, 2, False),
}


def rampe(hexa, bas, haut, n, dh=0.0):
    """n tons d'une couleur, de l'ombre à la lumière : ombres plus sombres qui tirent vers le
    bleu-violet, lumières vers le doré (lumière chaude d'en haut)."""
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, sat = colorsys.rgb_to_hls(r, g, b)
    out = []
    for k in range(n):
        d = bas + (haut - bas) * k / (n - 1)
        cible = 0.68 if d < 0 else 0.13
        glisse = ((cible - h + 0.5) % 1 - 0.5) * min(abs(d) * 0.35, 0.06)
        rr, gg, bb = colorsys.hls_to_rgb(
            (h + glisse + dh) % 1,
            min(max(clarte + d, 0.03), 0.95),
            min(sat * (1 + 0.25 * -d), 1),
        )
        out.append(hexa_de(rr, gg, bb))
    return out


def _voisin(m, dy, dx):
    """m décalé : out[j, i] = m[j + dy, i + dx] (faux hors du cadre)."""
    k = max(abs(dy), abs(dx))
    v = np.pad(m, k)
    return v[k + dy : k + dy + m.shape[0], k + dx : k + dx + m.shape[1]]


def profondeur(m, rayon):
    """Distance au bord (en pixels, plafonnée à rayon), par érosions successives : le relief
    d'une forme est un dôme, haut au milieu, qui retombe vers le bord."""
    d, cur = np.zeros(m.shape), m.copy()
    for _ in range(rayon):
        d += cur
        cur = (
            cur
            & _voisin(cur, 1, 0)
            & _voisin(cur, -1, 0)
            & _voisin(cur, 0, 1)
            & _voisin(cur, 0, -1)
        )
        if not cur.any():
            break
    return d


def details(c, f, motifs, oeil):
    """Volume par forme (lumière LUMIERE, d'en haut), rampes tramées en trois variantes de
    teinte par plaques, grain léger, ombres de contact, granules claires sur le vert, contour
    plus clair côté lumière des membres, œil brillant."""
    mat, plan, num = etiqueter(c, f)
    lx, ly = LUMIERE
    lum = np.full((c.H, c.W), 0.5)
    volumes = {fo["nom"]: fo["masque"] for fo in f}
    hauteur = (c.Y - c.Y.min()) / max(c.Y.max() - c.Y.min(), 1)
    for k, fo in enumerate(f):
        vol = volumes[fo["volume"]].astype(float)
        # relief : le masque adouci, à deux échelles ; la normale sortante est -gradient
        gy, gx = np.gradient(_flou(vol, 2))
        bord = -(gx * lx + gy * ly)
        dome = np.sqrt(_flou(profondeur(vol > 0, 10), 1) / 10)
        gy, gx = np.gradient(dome)
        face = -(gx * lx + gy * ly)
        ici = num == k
        lum[ici] = (0.5 + 1.6 * bord + 2.4 * face - 0.22 * (hauteur - 0.5))[ici]
        if fo["lum"] is not None:
            lum[ici] = fo["lum"][ici]
    # ombre de contact : la forme plus lointaine juste sous ou à droite d'une forme cernée
    devant = np.zeros((c.H, c.W), bool)
    for k, fo in enumerate(f):
        if fo["cernee"] and fo["plan"] != "fond":
            m = num == k
            pres = _voisin(m, -1, 0) | _voisin(m, -2, 0) | _voisin(m, 0, -1) | _voisin(m, -1, -1)
            ombre = pres & (num >= 0) & (num < k)
            lum[ombre] -= 0.22
            devant |= m
    if oeil is not None:
        lum[oeil["ombres"] & (num >= 0)] -= 0.2  # ombre du bras et de l'épaule sur le corps
        lum[oeil["coudes"] & (num >= 0)] += 0.24  # coudes saillants
        lum[oeil["arete"] & (num >= 0)] += 0.26  # arête du museau, rebord de la narine
        lum[oeil["joue"] & ~oeil["arete"]] -= 0.08  # joue, sous l'arête
    grain = (np.random.default_rng(5).random((c.H, c.W)) - 0.5) * 0.08
    plaques = np.clip(
        np.floor(_flou(np.random.default_rng(3).random((c.H, c.W)), 4) * 6 - 2), -1, 1
    ).astype(int)
    zone = mat.copy()
    if oeil is not None:
        zone[oeil["disques"] & (num >= 0)] = "disque"
    for nom, m in motifs.items():
        zone[m & (num >= 0)] = nom
    couleurs = {**MATIERES, **MOTIFS}
    img = np.full((c.H, c.W), "", object)
    for nom, coul in couleurs.items():
        ici = zone == nom
        if not ici.any():
            continue
        bas, haut, n, variantes = RAMPES[nom]
        idx = (lum + grain) * (n - 1) + 0.5 + (c.trame - 0.5) * 0.55  # tramage léger
        idx = np.clip(np.floor(idx), 0, n - 1).astype(int)
        for v in (-1, 0, 1) if variantes else (0,):
            sous = ici & ((plaques == v) if variantes else True)
            tons = np.array(rampe(coul, bas, haut, n, 0.014 * v), object)
            img[sous] = tons[idx[sous]]
    # granules claires sur le vert (dos, tête, bras), à places fixes : un pixel plus clair,
    # son ombre dessous ; jamais blanc
    vert = (mat == "dos") & (zone == "dos")
    interieur = vert & _voisin(vert, 1, 0) & _voisin(vert, -1, 0) & _voisin(vert, 0, 1)
    tirage = np.random.default_rng(17).random((c.H, c.W))
    for j, i in zip(*np.nonzero(interieur & (tirage < 0.022)), strict=False):
        img[j, i] = _retoucher(img[j, i], clair=0.32)
        img[j + 1, i] = _retoucher(img[j + 1, i], sombre=0.85)
    img = contour(img, f, num)
    # contour plus clair du côté de la lumière, autour des membres proches
    proche = plan == "devant"
    cote = _voisin(proche, 1, 0) | _voisin(proche, 0, 1)
    img[(img == CONTOUR) & (num < 0) & cote] = CONTOUR_CLAIR
    return oeil_details(img, oeil, num)


def _retoucher(hexa, sombre=1.0, clair=0.0):
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, cl, s = colorsys.rgb_to_hls(r, g, b)
    cl *= sombre
    return hexa_de(*colorsys.hls_to_rgb(h, cl + (1 - cl) * clair, s))


def oeil_details(img, oeil, num):
    """L'œil brillant : iris gris lavande plus clair en haut, finement réticulé, cerne noir,
    pupille en fente, reflet blanc et petit reflet bleuté, paupière supérieure en relief, pli
    sombre dessous. Fermé : la paupière inférieure, membrane pâle réticulée d'or, remonte sur le
    globe ; la paupière supérieure verte, éclairée, la rejoint en une fente sombre. La bosse de
    l'autre œil : éclairée par-dessus, ombrée à sa base."""
    if oeil is None:
        return img
    vert = rampe(MATIERES["dos"], *RAMPES["dos"][:3])
    d, u, v = oeil["dome"] & (num >= 0), oeil["u"], oeil["v"]
    du = np.gradient(d.astype(float))[0]
    img[d] = vert[3]
    img[d & (du > 0)] = vert[5]
    img[d & (du < 0)] = vert[2]
    img[oeil["narine"]] = CONTOUR
    m = oeil["masque"]
    r2 = u * u + v * v
    autour = (r2 > 1) & (r2 < 1.55) & (num >= 0)
    img[autour & (v < -0.2)] = vert[5]  # paupière supérieure en relief
    img[autour & (v < -0.55) & (u < 0.2)] = vert[6]
    img[autour & (v > 0.45)] = vert[1]  # pli sous le globe
    if oeil["ferme"]:
        ligne = -0.15 - 0.12 * u * u
        img[m] = "#A9A3B8"  # membrane pâle, translucide
        ii, jj = np.arange(img.shape[1])[None, :], np.arange(img.shape[0])[:, None]
        maille = ((ii + jj) % 4 == 0) ^ ((ii - jj) % 4 == 0)  # réseau lâche, en losanges
        img[m & maille & (v > ligne + 0.3) & (u * u + v * v < 0.7)] = (
            "#B49C6E"  # réticulation dorée
        )
        img[m & (v > 0.72)] = "#8A8498"
        img[m & (v < ligne)] = vert[4]  # paupière supérieure
        img[m & (v < ligne - 0.4)] = vert[5]
        img[m & (np.abs(v - ligne) < 0.16)] = vert[0]
        return img
    # globe pâle sur le pourtour, cerne noir, iris gris lavande réticulé, pupille en fente
    anneau = 0.86
    img[m] = "#A6ABB7"
    img[m & (v < 0.1)] = "#D5D9E2"
    iris = m & (r2 < (anneau - 0.13) ** 2)
    img[m & ~iris & (r2 < anneau**2)] = OEIL["cercle"]
    img[iris] = "#A9A8BC"
    img[iris & (v < 0.2)] = OEIL["iris"]
    img[iris & (v < -0.3) & (u < 0.3)] = "#D8DAE8"
    img[iris & (v > 0.5)] = "#8D8AA0"
    reticule = np.random.default_rng(23).random(img.shape) < 0.13
    img[iris & reticule] = "#8F8DA2"
    img[iris & fente(oeil)] = OEIL["pupille"]
    img[iris & ((u + 0.4) ** 2 + (v + 0.36) ** 2 < 0.035)] = "#FFFFFF"
    img[iris & ((u - 0.46) ** 2 + (v - 0.4) ** 2 < 0.014) & ~fente(oeil)] = "#9FB4C0"
    # l'autre œil : un peu de son globe pâle, et de sa pupille chez la grenouille de gauche
    g = oeil["dome_globe"] & d
    img[g] = "#C3C8D3"
    img[g & (np.gradient(g.astype(float))[0] < 0)] = "#9EA3AF"
    img[oeil["dome_pupille"] & d] = OEIL["pupille"]
    return img


RENDUS = {"silhouette": silhouette, "aplats": aplats, "details": details}


# ------------------------------------------------------------------ sortie
def coder(c, images):
    """Images de couleurs hexa → palette de lettres (K et k réservés au contour) et plages."""
    couleurs = sorted(
        {x for im in images.values() for x in im.ravel() if x} - {CONTOUR, CONTOUR_CLAIR}
    )
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    lettres += [chr(0x100 + i) for i in range(max(0, len(couleurs) - len(lettres)))]
    palette = {"K": CONTOUR, "k": CONTOUR_CLAIR, **dict(zip(lettres, couleurs, strict=False))}
    code = {x: lettre for lettre, x in palette.items()} | {"": "."}
    out = {}
    for cle, im in images.items():
        rangs = []
        for rang in im:
            s, prec, n = "", None, 0
            for x in rang:
                ch = code[x]
                if ch == prec:
                    n += 1
                else:
                    s += f"{prec}{n}" if prec else ""
                    prec, n = ch, 1
            rangs.append(s + f"{prec}{n}")
        out[cle] = rangs
    return {"palette": palette, "largeur": c.W, "hauteur": c.H, "images": out}


def produire(etape, nom):
    cadre, formes = SPRITES[nom]
    c = Cadre(*cadre)
    if etape == "animation" and nom != "branche":
        images = {}
        for g in (0, 1):
            for s in (0, 1):
                for k in (0, 1):
                    for t in (-1, 0, 1):
                        args = formes(c, g * 0.9, s * 0.7, bool(k), t)
                        images[f"{g}{s}{k}{t + 1}"] = RENDUS["details"](c, *args)
        return c, coder(c, images)
    rendu = RENDUS["details" if etape == "animation" else etape]
    return c, coder(c, {"0001": rendu(c, *formes(c))})


if __name__ == "__main__":
    etape = sys.argv[1] if len(sys.argv) > 1 else "silhouette"
    grenouilles = []
    for nom in SPRITES:
        c, sprite = produire(etape, nom)
        (ICI / f"{nom}_{etape}.json").write_text(
            json.dumps(sprite, separators=(",", ":")), encoding="utf-8"
        )
        grenouilles.append({"nom": nom, "x": c.x0, "y": c.y0, "sprite": sprite})
        print(f"{nom} : {len(sprite['images'])} image(s), {len(sprite['palette'])} couleurs")
    scene = {"largeur": TOILE[0], "hauteur": TOILE[1], "grenouilles": grenouilles}
    (ICI / f"scene_{etape}.json").write_text(
        json.dumps(scene, separators=(",", ":")), encoding="utf-8"
    )
