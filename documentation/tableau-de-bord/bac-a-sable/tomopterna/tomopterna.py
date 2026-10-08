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
    "dos": "#4F9E58",  # vert feuille : dos, tête, faces externes des membres
    "flanc": "#C98A62",  # flanc orangé, plus pâle vers la gorge
    "gorge": "#CDBDC0",  # blanc lilas, dans l'ombre de la tête
    "ventre": "#D2B6AA",  # blanc rosé (grenouille de gauche, vue de face)
    "levre": "#E6DAD4",
    "membre": "#C8844A",  # faces cachées des membres, mains, pieds
    "disque": "#E3A862",  # disques des doigts, plus clairs
    "fond": "#8A4E33",  # membres de l'autre côté, dans l'ombre
    "ecorce": "#4A3B32",
}
MOTIFS = {
    "barre": "#2A0A16",  # barres et taches violet-noir des flancs et des membres
    "tache": "#6A3442",  # marbrure violette du ventre et de la gorge
    "tubercule": "#E6DAD4",  # rangée de tubercules clairs, granules du talon
    "lichen": "#8C8680",  # taches claires de l'écorce
    "creux": "#2A201C",  # creux sombres de l'écorce
}
OEIL = {"iris": "#C2C4D4", "pupille": "#07070B", "cercle": "#17131C", "paupiere": "dos"}
PUPILLE = "verticale"
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
        7,
        [
            ".436B2.2B1.1l1.2B2.1",
            ".427B3.13l1.4",
            ".421B3.5l2.6l1B2.5l1.2",
            ".421l1.2l1.2l5.2l3.2B3.1l1.1B1.2",
            ".406l2.1l1.10l1.1l5.1l2.2B2l1.7l1.5",
            ".399l3.8l1.1B1.5l3.2l1.2B4.1B2.2B1.6B3.1B1.1",
            ".389l2.7l2.9B1.5l1.6B1.4B1.5B1.14",
            ".385B3.1l1B1.1l2.7B1.2l4B2.9B1.7B2.3B2.2B1.3B1.7",
            ".376B2.6l2.2l3.1l1.3l2B1.2B1.1l1.3l6.2B8.6l2.4l1B1.2B1.3B1.4",
            ".373l1.1B2.3l1.2l1.1l2B2l1.4l1.1l2.2l3B3.1l4.3B4.9l1.20",
            ".361B2.1B3.3l2.2l3.2l1.1B1.2l2.2l2.3l1.1B2.1l1.1l2.1B2.3B1.2B1.36",
            ".358l1.3l3.1l3.1l1.1B2.5B2.1B2.4B4.7B6.2B2.2B2.4B1.30",
            ".346B3.1B2l14.2B2.6l2.3l1.2B5.12B3.1l2.4B4.33",
            ".341B2.1l2.2l1.3l3.2B3.5B1.3B1.7B1.18B2.1l1.1B1.1l3.42",
            ".331l1.1B1.1l1.1l5.1l5.2l2.1B2.2B2.10l1.2B4.6l2.11l2B1l1.49",
            ".326B1l1.6l2.1l1.4l1.2l1.1B2.1B2l1.1B4.2B1.2B1.1B1.1B2.7l1.11l1.4B1.54",
            ".328l1.5l1.1B1.1B4.4l1B3.2B5.3B1.1B1.1l1.4B3.2l1.4B2.2l1.64",
            ".308B1.10l1.12l1B1l1B3.1l1B1.1B3.1B1.1B2.1l1.1B1l1.1B3l4.2l2.2B2.4l1.2l2.68",
            ".306B1.3B2l1.5l1.2l1.4l1B1.1B1.3B2.5B3.1B3.3l3.1l3.10l1.3l2.75",
            ".304B2.2l5.1l1.4B1.1l1.4l1.2B1.4B1.2B1.1B3.1l1.3l2.1l1B1.1l9.1B1.84",
            ".302l1B2.2l2.10B1.1l1.3B1.3l2.3l1B5.2l2.1B1.3B1.98",
            ".280l1.23B2.2B1.6B3.9B2.1l1.5B1.1B1.3l2.2l3.1B1.2B1.94",
            ".273l1.2l1.1B2.7l1.14B2.4B1.4B1.10l2.1B2.6l1.1l2.1B1l1.1B2.1l1.101",
            ".268B1.1B1.2l1.1l3B3.4B3.1B1.11B1.1B2.6l1.3l3.7B1.1B2l1.1B1.2l2B2.110",
            ".260l3.4l1B4.1l1.1l2.1l2.1l1.5B2.10l1B2l3.3l1.1B2.1B1.3l1.1B1.7B1.1l2.1B1.115",
            ".253l1.8l1B1.3l1B2.2B3.1B1.1l2.1B6.11l1B1.2l3.2l3B1.1l1B1.2l2.3B1.2l1.123",
            ".246l4.2l1.6l1.7l1B2l2.1l1.1B4.2B1.2B1.9l1.1l2.3B1.1l2.1l2.1B2.2B1.2l1.131",
            ".245l2B1l1B3.6l2.2B2.5B1.2B1l1.1l1B4.3B1.5l1.4l3B2.1B3l1B3.1B3.137",
            ".237l1.3l1.1B1.1l1.1B1l1.1B1.1l1.4B1.1l1.1B1.1B1.5B4.1l2B2.1l3.7B1.4l1.1B8.144",
            ".227B2.1B1.5B1.2B1l4.1B1.2B1.1B1.1l1.4B8.3B1.2l1.3B2l1.2l1.5l1.1B2.4B3.151",
            ".222l4.4B1l2.2l1B1.1l1B1.1l1.1B2.1B1.2B2l2.4B6.1B5.2l2.1B1.1l1B1.7l1B4.158",
            ".212l4.1l4.1l8.1l1.1l1B3.1l1.2B5.4B1.1l2.2l1B2.1B1.2l1B5.1B1.1l1.10B2.163",
            ".205l1.3B2.3l3.1l2.1B1.6B8l2.4B1.2B1.3B1.1B1l4B3.2l3.2B3.2l3.1B1.172",
            ".199B1.3B1.2l2.2B2.2B2.4B4.1B1.3B1.1B4.1B2.1l1B2l2B1l1.1B1l1.2B8l2.1l4B2l2B1.177",
            ".192l2.2l1.5B1.1l2.2l2.1B1.2B6.1B2.1B8.1l1B2.2l2B1.1l3.1B1l2.1B3.2B3.1B1.1l2B1.184",
            ".193l1.2l1.6B2.1l1B2.1l2.2B2.2B1.2B1.7B2.1l3B1.1B1.2B1l1B1.5B2.3l5.190",
            ".197l1.4l1.1B4.1B3.1B1.4l4.8l4.1B1.1B11.1B1.198",
            ".169l3.25l1B1.1B2.2l1.1B1.5B1.5B2l2.8l3.1B1.1B2.1l1B3.205",
            ".167B1.1l3.3B1.21l1.2l2.1B1.2B1.1B1.1B2.2B1.3B4.5B1.1B8.211",
            ".160B1.1l2.3B1l3.1B2.17l2.3l2.2B1.1B1.4l1.3l1.6B2.1B2.4B4.217",
            ".157B1.1B3.3l1.2l3.1B2.17l2B1.1l2.1B2.3B1.1B1l1.1B2.1B1.1B1.6B2.226",
            ".155B1.4B1l5B1.2l1.1B2.13l2.4l1.1l2.1B4l2B4.2B2.1B1.1l1.233",
            ".150B4l5.1B2.1B1.2B3.1B1.5l1.9l3.3B1.1B3.1B2.1B6.1B1.239",
            ".148l3.2l3.1B1.2B3.2B1l1.8l3.9l1.4B4.2B2.248",
            ".121B1.23B1l1.1B1.1B1.1B3.2l1.6l1.1l1.6B2l1.1l1.8B1.1B1.4B1.254",
            ".114B2l3.1B2.18B2.1B2.2l1.1B1.2B1.3B1.4B2l2.7B3l3.8B1.261",
            ".115B1.1B2.1B1.21B1.6l1.1B1.4l2.3B1.1l1.6B5.1l2.270",
            ".102B4.10l1.1B1.1B1.1l1.25l1.1B1.7l1.4l1.6B1.277",
            ".97B9.10l3.1B2l2.26B1.4l1.3B4.285",
            ".95B5.5l2.9l3.1B1.3l1.11l1.13B2.2B1.293",
            ".85B7.2B2.1l8.11B1.1l1.1B1.2B1l2.12B4.306",
            ".80B1.2B4.1B3.2l1.3l3.1l5.11B2.2B3.1l1.322",
            ".81B3.4l6.1l3.5B1.3B1.9B2.1l1.1B2.324",
            ".77B1.3l3.3l3.1l4.8l2.13l3B3.324",
            ".68B2.1B2.1B2.1l2.2l1.4l3.3l1.1l2.2B1.1B1.2B3.1l1.10l2.328",
            ".61B1.4B1.2l1.3l1.1l3.1B1.5l3.5B2.1B10.1l1.1l1.338",
            ".49B1.2B1.11l1.1l4.1l3.5l1.1l3B1l1.1l1.2B2.2B1.1B5.4B1.342",
            ".40B12.6l6.9l1.2l1.2l1.1l1.1B2.1l2.1l1B3.1B6.348",
            ".38B10.9l1.4B2.4l1.1l1.2l1.4B1.1B2.4B1.3B2.1l1.354",
            ".22B7.17l2.2l2.1l3.1l1.6l1.2l1.2B1l1.2B4.1B1.4B3.361",
            ".16B10.15l6.7l1.2l3.2l2.1l1B10.2B1.369",
            ".1B2.11B4.6B1.9l1.1l3.10l1.2l2.2l5.1B1.2B2.1B5.375",
            ".1B5.1B1.2B1.4B3.1l4.1l3.3l1.3l2.8l2.1l4B1.2B4.1B1.1B5.382",
            ".6B2.3l1.7l2.2l3.5l1.2l1.7l1.8B2l1.1B5.388",
            ".1l4.3l1.7l1.13l2.4B2.4l1.1B4.1B1.1B1.396",
            ".1l2.10l1.13l1.2l1.3l1.3B1.409",
            ".10l5.11l1.1B1.419",
            ".14l1B2.1l1.1B4.424",
            ".2B1.4B1.440",
            ".1B1.1B1.444",
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


def doigt(pts, x, y):
    """Un doigt d'un pixel, d'articulation en articulation, et son disque au bout (2 × 2)."""
    trait = membre(pts, [0.26] * len(pts), x, y)
    bx, by = pts[-1]
    disque = (np.abs(x - bx) < 0.55) & (np.abs(y - by) < 0.55)
    return trait | disque, disque


def suivre(pts, x):
    xs, ys = zip(*pts, strict=True)
    return np.interp(x, xs, ys)


def forme(nom, masque, matiere, plan="corps", cernee=False, volume=None):
    """plan : fond (membres de l'autre côté, ce qui pend derrière la branche), corps, devant
    (membres proches) ; cernee : un contour la sépare de ce qu'elle recouvre ; volume : la forme
    dont elle prend le relief (le flanc prend celui du corps)."""
    return {
        "nom": nom,
        "masque": masque,
        "matiere": matiere,
        "plan": plan,
        "cernee": cernee,
        "volume": volume or nom,
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
    (-2, 74.6),
    (14, 73.0),
    (28, 71.2),
    (42, 69.1),
    (56, 66.9),
    (70, 64.8),
    (84, 62.9),
    (98, 61.0),
    (112, 59.1),
    (126, 57.0),
    (140, 54.9),
    (154, 52.9),
    (168, 51.0),
    (182, 49.0),
    (196, 47.0),
    (210, 45.0),
    (226, 43.0),
]
EPAISSEUR = [(-2, 2.6), (40, 2.9), (90, 3.1), (140, 3.2), (180, 3.0), (226, 2.8)]


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


def formes_droite(c, gorge=0.0, souffle=0.0, cligne=False, tete=0):
    X, Y = c.X, c.Y
    xt, yt = tete_tournee(c, DROITE["pivot"], 1, tete)
    corps = dans(
        catmull(
            [
                (122.3, 34.6),
                (122.6, 33.2),
                (123.4, 32.1),
                (124.8, 31.3),
                (126.4, 30.4),
                (128.6, 28.7),
                (131.0, 28.0),
                (133.5, 28.3),
                (135.6, 29.4),
                (138.0, 30.1),
                (141.2, 30.9),
                (145.0, 32.0),
                (148.2, 33.3),
                (151.5, 35.1),
                (154.8, 37.1),
                (157.2, 39.2),
                (159.0, 41.6),
                (159.6, 43.8),
                (158.6, 45.6),
                (155.5, 46.8 + souffle * 0.4),
                (151.0, 47.6 + souffle * 0.6),
                (146.0, 47.4 + souffle * 0.6),
                (142.0, 46.6 + souffle * 0.4),
                (139.0, 45.4),
                (136.2, 43.9 + gorge * 0.3),
                (134.2, 42.4 + gorge * 0.8),
                (132.7, 40.7 + gorge),
                (130.0, 38.8 + gorge * 0.6),
                (126.2, 37.8 + gorge * 0.2),
                (123.4, 37.2),
                (122.4, 36.4),
            ]
        ),
        xt,
        yt,
    )
    limite = suivre(
        [
            (120.0, 36.2),
            (125.0, 36.7),
            (129.8, 37.2),
            (134.0, 37.7),
            (137.8, 38.3),
            (141.0, 38.6),
            (147.0, 39.0),
            (151.5, 39.4),
            (155.0, 40.2),
            (158.0, 41.5),
            (161.0, 43.0),
        ],
        xt,
    )
    dessous = corps & (yt >= limite)
    gorge_ = dessous & (xt < 136.0)
    levre = gorge_ & (yt < limite + 0.75)
    # patte arrière proche : entre le bras et le pied, la face cachée de la cuisse, orangée et
    # barrée, jusqu'à la branche ; talon levé, granuleux ; tarse orangé qui redescend vers la
    # branche, bordé de vert ; orteils en éventail sur la branche
    cuisse = dans(
        catmull(
            [
                (154.4, 40.8),
                (158.0, 41.4),
                (160.0, 43.0),
                (160.4, 46.4),
                (159.0, 49.0),
                (156.0, 49.4),
                (153.6, 47.6),
            ]
        ),
        X,
        Y,
    )
    talon = ellipse(161.8, 39.4, 2.6, 2.0, X, Y)
    tarse = membre([(161.8, 40.6), (162.2, 43.6), (162.6, 46.8)], [1.9, 1.7, 1.3], X, Y)
    lisere = membre([(164.8, 39.6), (165.4, 47.2)], [0.9, 0.7], X, Y)  # bord externe vert
    orteils, disques_o = np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for pts in (
        [(162.4, 46.6), (160.2, 48.6), (158.2, 50.6)],
        [(162.6, 46.8), (161.8, 50.4), (161.3, 53.6)],
        [(163.0, 46.6), (165.0, 49.2), (166.4, 52.6)],
    ):
        o, d = doigt(pts, X, Y)
        orteils |= o
        disques_o |= d
    # bras proche : colonne verte de l'épaule, sous le dos, au poignet sur la branche
    bras = membre(
        [(153.8, 39.6), (152.0, 45.0), (149.8, 50.0), (148.6, 52.6)], [2.2, 2.1, 1.7, 1.2], X, Y
    )
    main, disques_m = ellipse(146.8, 53.2, 1.5, 1.0, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(146.0, 52.8), (143.4, 51.8), (140.8, 51.6)],
        [(146.0, 53.6), (142.4, 55.6), (139.2, 58.2)],
        [(146.6, 53.8), (145.6, 56.6), (145.0, 59.2)],
    ):
        o, d = doigt(pts, X, Y)
        main |= o
        disques_m |= d
    # bras de l'autre côté : sa face cachée, orangée, sous la gorge ; sa main sur la branche
    bras_fond = membre(
        [(142.8, 44.6), (138.8, 46.8), (135.8, 48.8), (133.2, 50.2)], [1.7, 1.6, 1.3, 1.0], X, Y
    )
    main_fond, disques_f = ellipse(131.4, 50.8, 1.4, 1.0, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(130.4, 50.8), (127.2, 51.4), (124.6, 52.6)],
        [(130.6, 51.4), (128.6, 54.0), (127.4, 56.6)],
        [(131.6, 51.4), (132.6, 53.4), (133.2, 55.6)],
    ):
        o, d = doigt(pts, X, Y)
        main_fond |= o
        disques_f |= d
    # pied de l'autre côté, qui pend sous la branche : découpé par elle
    pend = dans(
        catmull(
            [
                (150.0, 56.0),
                (154.0, 55.6),
                (157.0, 58.2),
                (157.6, 61.2),
                (154.0, 61.6),
                (151.2, 60.4),
                (149.6, 58.4),
            ]
        ),
        X,
        Y,
    ) & ~branche(X, Y)
    f = [
        forme("pied_fond", pend, "fond", "fond", cernee=True),
        forme("bras_fond", bras_fond | main_fond, "fond", "fond", cernee=True),
        forme("corps", corps | ellipse(127.4, 29.6, 1.3, 1.0, xt, yt), "dos"),
        forme("flanc", dessous & ~gorge_, "flanc", volume="corps"),
        forme("gorge", gorge_ & ~levre, "gorge", volume="corps"),
        forme("levre", levre, "levre", volume="corps"),
        forme("cuisse", cuisse, "membre", cernee=True),
        forme("tarse", tarse | talon, "membre", cernee=True),
        forme("lisere", (lisere & ~tarse) | (talon & (X > 163.4)), "dos", volume="tarse"),
        forme("orteils", orteils, "membre", "devant", cernee=True),
        forme("bras", bras, "dos", "devant", cernee=True),
        forme("main", main, "membre", "devant", cernee=True),
    ]
    d = decalque("droite", c, xt, yt)
    # tubercules clairs : une rangée, un pixel sur deux, entre le vert du dos et le flanc ;
    # quelques granules sur le talon
    pair = (np.floor(X * ECHELLE) + np.floor(Y * ECHELLE)) % 2 == 0
    rangee = dessous & ~gorge_ & (yt < limite + 0.5) & (xt < 155) & pair
    granules = talon & (np.floor(X * ECHELLE) % 3 == 0) & (np.floor(Y * ECHELLE) % 2 == 0)
    motifs = {
        "barre": d == "B",
        "tache": d == "t",
        "tubercule": rangee | (granules & (X < 163.0)),
    }
    oeil = oeil_de(xt, yt, 131.15, 31.8, 2.75, 3.0, cligne, dome=(127.4, 29.6, 1.3, 1.0))
    oeil["narine"] = ellipse(123.9, 33.1, 0.35, 0.35, xt, yt)
    oeil["disques"] = disques_o | disques_m | disques_f
    return f, motifs, oeil


# ------------------------------------------------------------------ la grenouille de gauche
# de trois quarts, tournée vers la droite et vers nous
GAUCHE = {"cadre": (96, 60, 110, 90), "pivot": (78.0, 42.0), "sens": -1}


def formes_gauche(c, gorge=0.0, souffle=0.0, cligne=False, tete=0):
    X, Y = c.X, c.Y
    xt, yt = tete_tournee(c, GAUCHE["pivot"], -1, tete)
    corps = dans(
        catmull(
            [
                (99.7, 41.2),
                (99.3, 39.6),
                (98.2, 38.0),
                (96.8, 36.8),
                (95.0, 35.6),
                (92.4, 34.8),
                (90.2, 33.6),
                (87.0, 33.0),
                (84.0, 33.4),
                (81.8, 34.6),
                (79.0, 35.9),
                (75.9, 37.4),
                (71.6, 39.6),
                (68.9, 41.9),
                (66.8, 45.0),
                (65.0, 48.0),
                (62.6, 50.2),
                (61.4, 53.0),
                (61.4, 58.0),
                (62.6, 62.0),
                (65.0, 63.6 + souffle * 0.3),
                (69.0, 63.8 + souffle * 0.5),
                (73.0, 63.0 + souffle * 0.6),
                (77.0, 61.6 + souffle * 0.6),
                (81.0, 60.4 + souffle * 0.4),
                (85.0, 59.4),
                (87.4, 57.0),
                (88.4, 52.0),
                (89.0, 48.0 + gorge * 0.4),
                (91.0, 46.4 + gorge * 0.9),
                (94.5, 45.3 + gorge),
                (97.4, 43.9 + gorge * 0.5),
                (99.3, 42.4),
            ]
        ),
        xt,
        yt,
    )
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
    # patte arrière proche, à gauche : cuisse (dessus vert) jusqu'au genou, tibia orangé qui
    # redescend devant la branche, pied replié dessous
    cuisse = membre([(61.0, 52.4), (57.0, 56.6), (53.4, 60.6)], [1.9, 1.8, 1.7], X, Y)
    tibia = membre([(53.6, 61.0), (55.4, 65.6), (57.4, 70.2)], [1.7, 1.7, 1.5], X, Y)
    pied, disques_p = np.zeros(X.shape, bool), np.zeros(X.shape, bool)
    for pts in (
        [(57.2, 70.4), (55.8, 71.8), (54.6, 72.8)],
        [(57.6, 70.6), (59.0, 71.8), (59.8, 72.8)],
    ):
        o, d = doigt(pts, X, Y)
        pied |= o
        disques_p |= d
    # bras gauche : colonne verte ; main sur la branche, deux longs doigts vers la droite
    bras = membre([(63.2, 50.6), (62.9, 57.5), (63.1, 64.4)], [1.65, 1.6, 1.45], X, Y)
    main, disques_m = ellipse(64.9, 65.6, 1.5, 1.1, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(65.4, 65.4), (69.4, 65.2), (73.4, 66.0)],
        [(65.4, 66.2), (69.6, 67.8), (74.2, 69.2)],
        [(64.6, 66.4), (65.0, 68.2), (65.6, 69.6)],
    ):
        o, d = doigt(pts, X, Y)
        main |= o
        disques_m |= d
    # à droite, sous la tête : le bras droit (épaule granuleuse barrée, bord vert), sa main
    # sur la branche ; la patte arrière droite, colonne barrée bordée de vert, un orteil vers la
    # gauche, et son pied qui pend sous la branche
    bras_d = membre(
        [(90.0, 47.6), (93.4, 51.0), (94.6, 55.0), (93.8, 59.0)], [1.9, 2.2, 1.7, 1.3], X, Y
    )
    main_d, disques_d = ellipse(93.8, 60.4, 1.4, 1.0, X, Y), np.zeros(X.shape, bool)
    for pts in (
        [(94.4, 60.2), (96.0, 60.0), (97.3, 60.4)],
        [(94.2, 61.0), (95.0, 62.6), (95.4, 64.2)],
        [(93.2, 61.0), (92.0, 62.6), (91.2, 64.2)],
    ):
        o, d = doigt(pts, X, Y)
        main_d |= o
        disques_d |= d
    jambe_d = membre([(88.0, 50.4), (89.6, 57.5), (91.4, 66.6)], [1.3, 1.5, 1.3], X, Y)
    # faces externes vertes, en liseré : bord gauche de la jambe, bord droit du bras, dessus de la
    # cuisse et genou (le masque décalé dit de quel côté est le bord)
    jambe_d_g = membre([(88.0, 50.4), (89.6, 57.5), (91.4, 66.6)], [1.3, 1.5, 1.3], X - 0.9, Y)
    bras_d_d = membre(
        [(90.0, 47.6), (93.4, 51.0), (94.6, 55.0), (93.8, 59.0)], [1.9, 2.2, 1.7, 1.3], X + 1.0, Y
    )
    cuisse_hg = membre(
        [(61.0, 52.4), (57.0, 56.6), (53.4, 60.6)], [1.9, 1.8, 1.7], X - 0.8, Y - 0.8
    )
    genou = cuisse & ellipse(53.4, 60.4, 1.5, 1.3, X, Y)
    orteil_d, disque_od = doigt([(89.0, 61.4), (85.8, 63.4), (83.1, 64.9)], X, Y)
    pend = dans(
        catmull(
            [
                (90.0, 66.2),
                (94.0, 66.0),
                (97.2, 67.6),
                (97.0, 70.8),
                (93.6, 71.6),
                (90.6, 70.6),
                (89.6, 68.2),
            ]
        ),
        X,
        Y,
    ) & ~branche(X, Y)
    f = [
        forme("pied_fond", pend, "fond", "fond", cernee=True),
        forme("corps", corps | ellipse(93.6, 34.5, 3.0, 1.7, xt, yt), "dos"),
        forme("ventre", dessous & ~gorge_ & ~levre, "ventre", volume="corps"),
        forme("gorge", gorge_ & ~levre, "gorge", volume="corps"),
        forme("levre", levre, "levre", volume="corps"),
        forme("jambe_d", jambe_d | orteil_d, "membre", cernee=True),
        forme("jambe_d_vert", jambe_d & ~jambe_d_g, "dos", volume="jambe_d"),
        forme("cuisse", cuisse, "membre", cernee=True),
        forme("cuisse_vert", (cuisse & ~cuisse_hg) | genou, "dos", volume="cuisse"),
        forme("tibia", tibia | pied, "membre", "devant", cernee=True),
        forme("bras", bras, "dos", "devant", cernee=True),
        forme("main", main, "membre", "devant", cernee=True),
        forme("bras_d", bras_d | main_d, "membre", "devant", cernee=True),
        forme("bras_d_vert", bras_d & ~bras_d_d & (Y < 56), "dos", "devant", volume="bras_d"),
    ]
    d = decalque("gauche", c, xt, yt)
    motifs = {"barre": d == "B", "tache": d == "t"}
    oeil = oeil_de(xt, yt, 87.1, 37.9, 3.3, 3.4, cligne, dome=(93.6, 34.5, 3.0, 1.7))
    oeil["narine"] = ellipse(96.3, 38.4, 0.35, 0.35, xt, yt)
    oeil["disques"] = disques_p | disques_m | disques_d | disque_od
    return f, motifs, oeil


def oeil_de(xt, yt, cx, cy, rx, ry, ferme, dome):
    """L'œil visible (globe cerné, iris, pupille en fente) et la bosse de l'autre œil."""
    u, v = (xt - cx) / rx, (yt - cy) / ry
    return {
        "x": cx,
        "y": cy,
        "masque": u * u + v * v <= 1,
        "u": u,
        "v": v,
        "ferme": ferme,
        "dome": ellipse(*dome, xt, yt),
    }


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
    p = {
        "ronde": u * u + v * v < 0.3,
        "horizontale": (u / 0.75) ** 2 + (v / 0.32) ** 2 < 1,
        "verticale": np.abs(u + 0.1) < 0.22 * np.clip(1 - np.abs(v) / 0.62, 0, 1) + 0.09,
    }[PUPILLE]
    img[m & p & (np.abs(v) < 0.62)] = couleurs["pupille"]
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
    return rendu_oeil(img, oeil, OEIL, MATIERES[OEIL["paupiere"]], MATIERES["dos"])


RENDUS = {"silhouette": silhouette, "aplats": aplats}


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
