"""Sprite 8-bit de Callimedusa tomopterna (phyllomédusine tigrée), sur le modèle d'A. blanci.

    uv run python documentation/tableau-de-bord/tomopterna.py

Grenouille de profil, tournée vers le titre (à gauche), assise, accroupie, dessinée d'après une
photo de deux C. tomopterna perchées sur une branche (auteur à créditer) : celle de droite, de
profil, pour la silhouette, la tête et l'œil, celle de gauche, de trois quarts, pour la gorge, le
ventre et le bras proche. Posture et membres repris d'A. blanci (grenouille.py), validés par
Léonard : patte arrière en Z (cuisse arrondie tournée vers nous, tibia long et bombé qui dépasse
derrière la croupe, talon arrondi, tarse au sol, orteils en éventail), bras pliés dont l'épaule
s'attache en arrondi, bras de l'autre côté en arrière-plan, mains et pieds en gabarits posés au
pixel près (ORTEILS, MAIN_PROCHE, MAIN_LOINTAINE).

Robe de l'espèce, d'après la photo : dos, tête et faces externes des membres vert feuille,
luisants ; flancs orangés barrés de violet-noir, semés de granules claires ; ligne blanche de la
lèvre, prolongée par une rangée de tubercules clairs entre le vert du dos et l'orangé du flanc ;
gorge blanc lilas, mouchetée ; faces cachées des membres, mains et pieds orangés, disques pâles ;
grand œil saillant à l'iris gris lavande finement réticulé, pupille verticale en fente, cerné de
noir ; l'autre œil n'est qu'une bosse verte sur le crâne. L'épaule tourne au vert-jaune avant de
se fondre dans le flanc.

Les formes sont décrites dans un repère de 57 × 44 unités et tracées sur une grille de 114 × 88
pixels (deux pixels par unité), sans anticrénelage, puis cernées d'un contour, en tons de base.
Une passe de nuances (nuancer) affine ensuite chaque matière (peau verte, peau orangée, barres,
flanc, gorge et ventre, épaule) : ses tons de base sont adoucis par un flou limité à la matière,
puis redistribués sur une rampe plus fine (de 3 à 9 niveaux de clarté), en trois variantes de
teinte réparties par plaques. Le flanc juste derrière le bras garde sa teinte, assombrie et un
peu grisée (ombre de contact), et une quarantaine de pustules parsèment le corps sauf la tête :
un pixel de la couleur d'origine, plus clair, et son ombre juste en dessous, à des places fixes.

Une image par combinaison gorge (gonflée ou non) × flanc (gonflé ou non) × clignement ×
tête (baissée, droite, relevée) : la page les enchaîne pour que la grenouille respire, cligne
et bouge un peu la tête. Écrit tomopterna.json (palette, taille et images codées par plages :
« V12 » = douze pixels de la couleur V), au format de grenouille.json.
"""

import colorsys
import json
import math
import random
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
PALETTE = {
    "K": "#140B10",  # contour
    "k": "#4A2216",  # contour côté lumière des membres orangés
    "j": "#173A1E",  # contour côté lumière des membres verts
    # peau verte : dos, tête, faces externes des membres
    "L": "#B6EC8A",  # reflet vif, peau mouillée
    "l": "#7CCF6A",  # reflet
    "V": "#3FA448",  # vert feuille
    "N": "#2E8040",  # mouchetures
    "D": "#256A38",  # ombre
    "X": "#123A24",  # ombre profonde, plis
    # peau orangée : faces cachées des membres, mains, pieds
    "I": "#F7C987",  # reflet vif, disques
    "H": "#EFA25A",  # reflet
    "O": "#DB7A33",  # orangé
    "S": "#A9501F",  # ombre
    "Z": "#6E2C14",  # ombre profonde
    # barres violet-noir
    "B": "#2C0F1E",  # barre
    "b": "#4E2434",  # bord de barre, taches
    # flanc
    "F": "#D88752",  # flanc orangé
    "f": "#EDB58A",  # granules claires
    "v": "#E09A68",  # grain clair : un cran plus clair que le flanc alentour
    "u": "#C87442",  # grain sombre : un cran plus sombre
    "s": "#9C5032",  # flanc dans l'ombre (croupe)
    "J": "#A9D25A",  # pourtour de l'épaule, qui tourne au vert-jaune
    "T": "#EFE7E8",  # lèvre, tubercules
    # gorge et ventre, blanc lilas
    "C": "#E6DBE2",  # gorge, ventre
    "g": "#C7B8C6",  # gorge grisée
    "c": "#9E8C9E",  # ventre ombré, points de la gorge, bouche
    "q": "#B9A6B4",  # ventre grisé
    # œil
    "E": "#07070B",  # pupille
    "e": "#17131C",  # cerne noir autour de l'iris
    "G": "#CFCFDC",  # iris, gris lavande
    "A": "#9693A6",  # bas de l'iris, plus sombre
    "r": "#ABA9BB",  # réticulation de l'iris
    "W": "#FFFFFF",  # reflet de l'œil
    "w": "#9FB4C0",  # second reflet, bleuté
}
VERT, ORANGE = "LlVNDX", "IHOSZ"
UNITES = (57, 44)
MARGE = 1  # unité laissée vide au-dessus de la tête, qui monte quand la grenouille la relève
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
ECHELLE = 2
W, H = UNITES[0] * ECHELLE, UNITES[1] * ECHELLE
# centres des pixels, en unités
X, Y = np.meshgrid((np.arange(W) + 0.5) / ECHELLE, (np.arange(H) + 0.5) / ECHELLE - MARGE)
TRAME = (BAYER4[(np.arange(H) % 4)[:, None], (np.arange(W) % 4)[None, :]] + 0.5) / 16


def catmull(pts, n=14):
    """Courbe fermée lisse passant par les points (Catmull-Rom)."""
    out, m = [], len(pts)
    for i in range(m):
        p0, p1, p2, p3 = pts[(i - 1) % m], pts[i], pts[(i + 1) % m], pts[(i + 2) % m]
        for k in range(n):
            t = k / n
            out.append(
                tuple(
                    0.5
                    * (
                        2 * p1[j]
                        + (-p0[j] + p2[j]) * t
                        + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                        + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t**3
                    )
                    for j in (0, 1)
                )
            )
    return out


def dans(poly, x, y):
    """Masque des points (x, y) à l'intérieur du polygone (règle pair-impair)."""
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


# Mains et pieds, dessinés au pixel près (tons de base de la peau orangée ; « . » = rien ;
# « w » = reflet bleuté de peau humide sur un disque). Chaque doigt : phalanges d'un pixel,
# disque terminal de 2 × 2 plus clair que le doigt, un pixel vide entre deux doigts (le contour
# le noircit). Doigts longs, comme chez les phyllomédusines ; les lointains, plus sombres.
ORTEILS = (
    (58, 78),
    [
        "........HHSSOS..........",
        "........SO...SSSSOSSSSS.",
        "..wI....................",
        "..HOOOOOOHOOOOOHOOOOOO..",
        "..................SSSSS.",
        "............wI.....HHHH.",
        "............HOOOOHOOOOO.",
    ],
)
MAIN_PROCHE = (
    (29, 77),
    [
        "....IH.............",
        "....HOOO...........",
        ".......OO....HOOS..",
        "........OOHHHHOOOS.",
        "..IwOOOHOOOOOOOOOS.",
        "..HO......OOOOOSS..",
        ".......IHOOSOOSS...",
        ".......HO..........",
    ],
)
MAIN_LOINTAINE = (
    (26, 67),
    [
        "....HO.........",
        "....SSSS...OSSZ",
        ".......SSOOOSSZ",
        "........SOSSSSZ",
        "...HOSOSSSSSSZ.",
        "...SS...SSSZZ..",
        "......HOS......",
        "......SS.......",
    ],
)


def poser(g, gabarit):
    """Pose un gabarit de pixels ; renvoie le masque des pixels posés."""
    (i0, j0), rangs = gabarit
    j0 += MARGE * ECHELLE
    m = np.zeros(g.shape, bool)
    for dj, rang in enumerate(rangs):
        for di, ton in enumerate(rang):
            if ton != ".":
                g[j0 + dj, i0 + di] = ton
                m[j0 + dj, i0 + di] = True
    return m


def separer(g, dessus, dessous):
    """Un pixel vide, que le contour noircit, entre ce qui est devant (`dessus`) et ce
    qui est derrière (`dessous`) : la main proche se détache de la main lointaine."""
    voisin = np.zeros_like(dessus)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin |= np.roll(np.roll(dessus, dy, 0), dx, 1)
    g[voisin & dessous & ~dessus] = "."


def miettes(m, n=12):
    """Groupes de pixels de m (4-connexes) de moins de n pixels : restes isolés à effacer."""
    out, vu = np.zeros_like(m), np.zeros_like(m)
    for j0, i0 in zip(*np.nonzero(m), strict=True):
        if vu[j0, i0]:
            continue
        pile, groupe = [(j0, i0)], []
        vu[j0, i0] = True
        while pile:
            j, i = pile.pop()
            groupe.append((j, i))
            for a, b in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if 0 <= a < m.shape[0] and 0 <= b < m.shape[1] and m[a, b] and not vu[a, b]:
                    vu[a, b] = True
                    pile.append((a, b))
        if len(groupe) < n:
            for j, i in groupe:
                out[j, i] = True
    return out


def suivre(pts, x):
    xs, ys = zip(*pts, strict=True)
    return np.interp(x, xs, ys)


def bruit(x, y, graine):
    """Valeur pseudo-aléatoire stable par demi-unité (grain de la peau), entre 0 et 1."""
    xi, yi = np.floor(x * ECHELLE), np.floor(y * ECHELLE)
    v = np.sin(xi * 12.9898 + yi * 78.233 + graine * 37.719) * 43758.5453
    return v - np.floor(v)


def barres(x, y, pieds, pente, largeur):
    """Barres sombres passant par les pieds (x0, y0), obliques (x avance de `pente` par unité
    de y), de demi-largeur `largeur` : le cœur et un liseré autour."""
    coeur, bord = np.zeros(x.shape, bool), np.zeros(x.shape, bool)
    for x0, y0 in pieds:
        d = np.abs(x - x0 - pente * (y - y0))
        coeur |= d < largeur
        bord |= d < largeur + 0.45
    return coeur, bord & ~coeur


# tibia, du genou (en haut, à l'avant) au talon : connu avant la cuisse, qui en reçoit l'ombre
JAMBE = catmull(
    [
        (32.48, 32.19),
        (34.6, 30.23),
        (37.69, 29.57),
        (40.77, 29.81),
        (44.3, 30.92),
        (47.83, 32.32),
        (51.0, 33.87),
        (52.59, 34.84),
        (53.65, 35.78),
        (54.0, 37.2),
        (53.47, 39.18),
        (51.35, 38.68),
        (47.83, 38.52),
        (43.42, 38.33),
        (39.01, 37.93),
        (35.48, 37.37),
        (33.19, 36.47),
        (31.78, 35.21),
        (31.6, 33.6),
    ]
)


def image(gorge=0.0, souffle=0.0, cligne=False, tete=0):
    """Tons de base d'une image, le masque de l'ombre de contact derrière le bras et celui de
    la peau où poser les pustules."""
    g = np.full((H, W), ".", dtype="<U1")
    # Tête baissée ou relevée : rotation de quelques degrés autour du cou, qui s'estompe
    # vers le corps et ne touche pas les pattes. On échantillonne à l'envers.
    poids = np.clip((21 - X) / 8, 0, 1) * np.clip((19.5 - Y) / 3.5, 0, 1)
    a0 = -tete * 0.06
    a = a0 * poids
    px, py = 18.0, 10.0
    x = px + (X - px) * np.cos(a) - (Y - py) * np.sin(a)
    y = py + (X - px) * np.sin(a) + (Y - py) * np.cos(a)
    n1, n2, n3 = bruit(x, y, 1), bruit(x, y, 2), bruit(X, Y, 3)
    alea = random.Random(7)

    # --- bras droit, de l'autre côté : on en voit la face cachée, orangée, sous la gorge, dans
    # l'ombre ; sa main se pose un peu plus loin, donc un peu plus haut, et en retrait
    loin = catmull(
        [
            (19.0, 18.6),
            (22.6, 18.8),
            (24.0, 23.0),
            (25.0, 26.8),
            (25.2, 28.4),
            (24.61, 29.4),
            (23.01, 32.31),
            (21.6, 33.9),
            (20.66, 34.9),
            (19.14, 33.9),
            (19.6, 32.6),
            (20.49, 30.69),
            (21.6, 29.2),
            (22.59, 27.95),
            (21.6, 25.4),
            (19.6, 21.6),
        ]
    )
    lm = dans(loin, X, Y)
    g[lm] = "S"
    g[lm & (n3 < 0.12)] = "Z"
    g[lm & ~dans(loin, X - 0.9, Y)] = "O"
    g[lm & ~dans(loin, X + 0.6, Y)] = "Z"
    g[lm & barres(X, Y, [(22.4, 30.4), (24.0, 26.4)], 1.33, 0.45)[0]] = "B"
    bras_loin = lm | poser(g, MAIN_LOINTAINE)

    # --- corps et tête : l'œil saillant et la bosse de l'autre œil dépassent du crâne
    corps = catmull(
        [
            (1.0, 9.4),
            (1.5, 7.6),
            (2.6, 6.2),
            (4.4, 5.2),
            (6.2, 4.2),
            (8.3, 2.6),
            (10.2, 1.2),
            (12.4, 0.8),
            (14.8, 1.2),
            (16.9, 2.6),
            (19.4, 3.5),
            (23.0, 4.8),
            (27.5, 6.6),
            (32.0, 8.9),
            (36.5, 11.4),
            (40.6, 14.0),
            (44.4, 16.9),
            (47.6, 20.2),
            (49.6, 24.4),
            (50.6, 28.8),
            (50.6, 33.0),
            (49.0, 36.4),
            (45.6, 37.6),
            (41.2, 37.0),
            (37.6, 35.0),
            (33.0, 33.0 + souffle),
            (29.0, 31.2 + souffle),
            (26.0, 29.0 + souffle * 0.6),
            (23.6, 26.4),
            (22.4, 24.6),
            (19.6, 21.0),
            (16.8, 18.2),
            (13.6, 16.6 + gorge * 0.4),
            (10.0, 15.4 + gorge),
            (6.4, 14.1 + gorge * 0.8),
            (3.4, 12.7 + gorge * 0.3),
            (1.6, 11.6),
            (1.0, 10.6),
        ]
    )
    dedans = dans(corps, x, y)
    # limite du vert : la lèvre sur la tête, puis la frontière entre le dos et le flanc
    limite = [
        (0, 10.4),
        (4, 11.3),
        (8, 12.1),
        (12, 12.9),
        (15.5, 13.8),
        (19, 15.2),
        (24, 16.9),
        (29, 18.7),
        (34, 20.6),
        (38, 22.1),
        (42, 23.4),
        (46, 24.9),
        (50, 26.8),
    ]
    yb = suivre(limite, x) + (n2 - 0.5) * 0.35 * (x > 16)  # bord irrégulier sur le flanc
    haut1, haut2 = ~dans(corps, x, y - 0.6), ~dans(corps, x, y - 1.3)
    bas1, bas2 = ~dans(corps, x, y + 0.6), ~dans(corps, x, y + 1.6)

    # dos et tête, vert feuille : grain de peau, mouchetures, reflet le long du dos, peau
    # luisante (un trait de lumière parallèle au dos), ombre au bas du vert
    dos = dedans & (y < yb)
    g[dos] = "V"
    g[dos & (n1 < 0.07)] = "N"
    g[dos & (n1 > 0.94)] = "l"
    for _ in range(14):
        mx, my, r = alea.uniform(18, 42), alea.uniform(6, 20), alea.uniform(0.4, 0.7)
        g[ellipse(mx, my, r, r * 0.8, x, y) & dos & (n2 > 0.35)] = "N"
    g[dos & (x > 15) & (TRAME < np.clip((y - yb + 3.4) / 2.6, 0, 1) * 0.8)] = "N"
    g[dos & (y > yb - 1.1) & (x > 15) & (n2 > 0.3)] = "D"
    g[dos & (y > yb - 0.5) & (x > 15)] = "D"
    luisant = dos & ~dans(corps, x, y - 2.6) & dans(corps, x, y - 2.0) & (x > 18) & (x < 40)
    g[luisant & (n1 > 0.3)] = "l"
    g[dos & haut2 & (n1 > 0.25)] = "l"
    g[dos & haut1] = "L"
    g[dos & haut1 & (n1 < 0.12)] = "l"
    # joue un peu plus claire sous l'œil
    g[dos & (x < 17) & (y > 9.5) & (y < yb - 0.4) & (n1 > 0.62)] = "l"

    # lèvre blanche, bouche, gorge mouchetée ; tubercules clairs le long de la limite
    dessous = dedans & (y >= yb)
    levre = dessous & (y < yb + 0.9) & (x < 15.5)
    g[dessous] = "C"
    tete_ = dessous & (x < 16.5)
    g[tete_ & ((n1 < 0.14) | (TRAME < np.clip((y - yb - 2.0) / 2.5, 0, 0.7)))] = "g"
    g[tete_ & (n2 < 0.07) & (x > 6)] = "c"
    g[levre] = "T"
    g[dessous & (y >= yb + 0.9) & (y < yb + 1.6) & (x > 2.2) & (x < 14.2)] = "b"
    # flanc orangé, semé de granules claires ; l'avant, près de la gorge, reste blanc lilas
    # et se fond dans l'orangé en tramage
    flanc = dessous & (x >= 15.5) & ~levre
    vers_peche = np.clip((x - 18.0) / 2.5, 0, 1)
    vers_orange = np.clip((x - 20.5) / 3.0, 0, 1)
    orange_ = flanc & (TRAME < vers_peche)
    g[orange_] = "f"
    g[orange_ & (TRAME < vers_orange)] = "F"
    g[orange_ & (n1 > 0.74)] = "f"
    g[orange_ & (n2 > 0.86)] = "v"
    g[orange_ & (n2 < 0.1)] = "u"
    g[flanc & ~orange_ & (n1 < 0.18)] = "g"
    # taches violettes de l'avant du flanc
    taches = ((18.8, 17.6, 0.55), (21.2, 19.6, 0.6), (17.6, 19.8, 0.45), (20.6, 22.4, 0.5))
    taches += ((26.2, 19.4, 0.5), (31.0, 28.6, 0.55), (35.4, 22.6, 0.45), (25.6, 26.6, 0.45))
    for mx, my, r in taches:
        g[ellipse(mx, my, r, r * 0.8, x, y) & flanc] = "b"
    # barres violet-noir, irrégulières, un peu obliques, du dos vers le ventre
    coeur, bord = barres(x + (n2 - 0.5) * 0.9, y, [(28.8, 24), (33.4, 25.5), (38.0, 27)], -0.3, 0.5)
    g[flanc & bord & (x > 23.5) & (n1 > 0.35)] = "b"
    g[flanc & coeur & (x > 23.5) & (n1 > 0.06)] = "B"
    # tubercules clairs entre le vert et l'orangé, un pixel sur deux
    tub = flanc & (y < yb + 0.5) & (x < 44)
    g[tub & (((X * ECHELLE).astype(int) + (Y * ECHELLE).astype(int)) % 2 == 0)] = "T"
    # ventre : blanc lilas en bas, ombré au bord, marbré de violet
    g[dessous & bas2 & (x >= 16)] = "C"
    g[dessous & bas2 & (x >= 16) & (n1 < 0.22)] = "q"
    g[dessous & bas2 & (x >= 20) & (n2 > 0.84)] = "c"
    g[dessous & bas1] = "c"
    # arrière-train : croupe et postérieur, orangés dans l'ombre, barrés
    croupe = dedans & (x > 44) & (y > yb)
    g[croupe] = "F"
    g[croupe & (n1 < 0.1)] = "u"
    g[croupe & (y > 30.5)] = "s"
    g[dedans & (x > 40) & (y > 36)] = "s"
    cb, _ = barres(x, y, [(46.5, 30), (49.5, 30)], -0.3, 0.75)
    g[croupe & cb] = "B"
    g[croupe & (x > 48.6) & (y < 34)] = "D"  # le vert du dos tourne autour de la croupe
    g[tete_ & bas1] = "g"

    # --- cuisse : on n'en voit que le dessus, vert, du genou (à l'avant) vers la croupe dont
    # elle dépasse ; le tibia la recouvre par-dessous. Pli là où elle touche le corps
    cuisse = catmull(
        [
            (33.4, 29.07),
            (36.0, 26.81),
            (40.0, 25.5),
            (44.0, 24.4),
            (47.4, 23.7),
            (49.8, 25.1),
            (50.8, 28.0),
            (50.7, 30.8),
            (50.4, 33.0),
            (48.6, 34.9),
            (44.0, 35.45),
            (39.0, 34.97),
            (35.0, 34.15),
            (32.6, 33.02),
            (32.2, 31.0),
        ]
    )
    cm = dans(cuisse, X, Y)
    avant = g != "."
    g[cm] = "V"
    g[cm & (n3 < 0.1)] = "N"
    g[cm & (n3 > 0.94)] = "l"
    g[cm & ~dans(cuisse, X + 0.4, Y - 2.4)] = "l"
    g[cm & ~dans(cuisse, X + 0.2, Y - 1.0)] = "L"
    g[cm & ~dans(cuisse, X - 0.6, Y + 1.2) & (X > 47)] = "D"
    g[ellipse(39.6, 27.0, 2.0, 0.5, X, Y, -0.05)] = "L"  # reflet de la peau humide
    g[cm & dans(JAMBE, X, Y - 1.2) & (n3 > 0.2)] = "D"  # ombre du tibia sur la cuisse
    bord = np.zeros_like(cm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bord |= cm & ~np.roll(np.roll(cm, dy, 0), dx, 1) & np.roll(np.roll(avant, dy, 0), dx, 1)
    g[bord] = "X"

    # --- pied : le tarse part du talon, à l'arrière contre le sol, et file vers l'avant
    # sous le tibia, orangé et barré ; trois orteils longs et fins s'en écartent vers l'avant,
    # chacun avec son disque. Les plus lointains sont plus hauts et plus sombres.
    tarse = catmull(
        [
            (52.97, 38.0),
            (52.47, 39.6),
            (49.15, 40.4),
            (44.99, 40.8),
            (41.66, 40.4),
            (40.6, 39.8),
            (39.8, 40.6),
            (40.2, 41.8),
            (42.49, 42.4),
            (46.65, 42.4),
            (50.81, 42.0),
            (52.3, 41.2),
            (53.63, 40.0),
            (53.8, 38.8),
        ]
    )
    tm = dans(tarse, X, Y)
    g[tm] = "O"
    g[tm & ~dans(tarse, X, Y - 0.6)] = "H"
    g[tm & ~dans(tarse, X, Y + 0.6)] = "S"
    tb, _ = barres(X, Y, [(44.2, 41.5), (48.6, 41.5)], 0.0, 0.7)
    g[tm & tb & dans(tarse, X, Y - 0.6)] = "B"
    orteils = poser(g, ORTEILS)

    # --- tibia : au premier plan, grosse masse ronde du genou (en haut, à l'avant) au talon
    # (en bas, à l'arrière, contre le sol), par-dessus la cuisse, vert ; reflet de peau humide
    # au genou
    jambe = JAMBE
    jm = dans(jambe, X, Y)
    sous_jambe = g != "."
    g[jm] = "V"
    g[jm & (n3 < 0.06)] = "N"
    g[jm & (n3 > 0.97)] = "l"
    g[jm & ~dans(jambe, X + 0.4, Y - 1.4)] = "l"
    g[jm & ~dans(jambe, X + 0.2, Y - 0.7) & (X < 44)] = "L"
    g[jm & ~dans(jambe, X - 0.2, Y + 2.2)] = "D"
    g[jm & ~dans(jambe, X, Y + 0.7)] = "X"
    g[ellipse(36.4, 31.0, 2.2, 0.5, X, Y, -0.05)] = "L"  # reflet au genou
    pli = np.zeros_like(jm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin_hors = ~np.roll(np.roll(jm, dy, 0), dx, 1)
        voisin_dessous = np.roll(np.roll(sous_jambe, dy, 0), dx, 1)
        pli |= jm & voisin_hors & voisin_dessous & (X < 52)  # pas de pli au talon
    g[pli] = "X"
    # cheville : talon arrondi, assez épais, qui joint le bout du tibia au tarse
    talon = ellipse(52.9, 38.9, 1.3, 1.15, X, Y)
    g[talon] = "O"
    g[talon & ~ellipse(52.9, 39.5, 1.3, 1.15, X, Y)] = "H"
    g[talon & ~ellipse(52.9, 38.4, 1.3, 1.15, X, Y)] = "S"

    # --- patte avant : sort du flanc sous la limite du vert, coude en arrière, avant-bras
    # plus épais en son milieu, poignet fin ; face externe verte, poignet orangé
    patte = catmull(
        [
            (18.6, 17.6),
            (23.2, 18.2),
            (25.6, 23.0),
            (27.6, 27.8),
            (28.5, 30.2),
            (28.3, 31.2),
            (27.4, 33.4),
            (26.35, 36.0),
            (25.0, 38.4),
            (23.5, 40.6),
            (21.7, 39.8),
            (22.4, 37.6),
            (23.25, 34.6),
            (24.4, 32.0),
            (25.7, 29.8),
            (23.4, 27.0),
            (20.8, 22.6),
        ]
    )
    pa = dans(patte, X, Y)
    # L'épaule s'attache en arrondi : un capuchon (demi-ellipse) coiffe le haut du bras ; sur
    # son pourtour, la part de pixels du bras décroît en tramage ordonné (Bayer 4 × 4) et le
    # flanc reprend. Aucun trait dans le corps : le contour n'apparaît que là où le bras sort
    # du corps. Le pourtour du capuchon tourne au vert-jaune avant de se fondre dans le flanc.
    corps_dessous = g != "."
    cx_e, cy_e, rx_e, ry_e = 22.6, 22.6, 2.2, 2.8
    rayon = np.sqrt(((X - cx_e) / rx_e) ** 2 + ((Y - cy_e) / ry_e) ** 2)
    capuchon = (rayon <= 1) & (Y < cy_e)
    fondu = capuchon & corps_dessous & (TRAME < np.clip((1 - rayon) / 0.45, 0, 1))
    bras = (pa & (Y >= cy_e)) | fondu | (pa & ~corps_dessous)
    g[bras] = "V"
    g[bras & (n3 < 0.06)] = "N"
    g[bras & (n3 > 0.97)] = "l"
    # lumière d'en haut à gauche : bord avant éclairé, bord arrière dans l'ombre
    g[bras & ~dans(patte, X - 1.1, Y)] = "l"
    g[bras & ~dans(patte, X - 0.6, Y)] = "L"
    g[bras & ~dans(patte, X + 1.1, Y) & (Y > 21.0)] = "D"
    g[bras & ~dans(patte, X + 0.6, Y) & (Y > 22.0)] = "X"
    g[ellipse(26.4, 30.6, 0.6, 0.9, X, Y, 0.4)] = "L"  # reflet au coude
    # poignet : le vert cède à l'orangé de la main, en tramage
    poignet = bras & (TRAME < np.clip((Y - 36.4) / 1.6, 0, 1))
    g[poignet] = "O"
    g[poignet & ~dans(patte, X - 0.6, Y)] = "H"
    g[poignet & ~dans(patte, X + 0.6, Y)] = "S"
    g[bras & capuchon & (rayon > 0.55)] = "J"  # pourtour de l'épaule, vert-jaune
    g[bras & capuchon & (rayon <= 0.55) & (Y < cy_e - 0.6)] = "L"
    # ombre de contact : le flanc juste derrière le bras, sous l'épaule (assombri plus tard)
    contact = corps_dessous & ~bras & (Y > 21.5) & (np.roll(bras, 1, 1) | np.roll(bras, 2, 1))
    separer(g, bras, bras_loin & ~dedans)  # contour entre les deux bras
    g[miettes(bras_loin & ~dedans & ~bras & (g != "."))] = "."  # restes entre gorge et bras
    # main tournée vers l'intérieur : poignet, paume et trois doigts longs vers l'avant
    proche = poser(g, MAIN_PROCHE)
    separer(g, proche, bras_loin & ~proche)

    # --- l'autre œil : une simple bosse de peau verte sur le crâne, éclairée par-dessus et
    # soulignée d'une ombre à sa base, qui ne bouge pas quand l'œil visible cligne
    dome = ellipse(7.9, 3.3, 2.1, 1.45, x, y)
    g[dome] = "V"
    g[dome & (n1 > 0.9)] = "l"
    g[dome & ~ellipse(7.9, 3.6, 2.1, 1.45, x, y)] = "L"
    g[dome & ellipse(7.4, 2.9, 1.1, 0.7, x, y) & ~ellipse(7.9, 3.6, 2.1, 1.45, x, y)] = "l"
    g[dome & ~ellipse(7.9, 2.8, 2.1, 1.45, x, y)] = "D"

    # --- l'œil visible : globe saillant cerné de noir, paupière supérieure en relief, pli
    cx, cy = 12.4, 6.0
    oeil = ellipse(cx, cy, 3.8, 3.95, x, y)
    iris = ellipse(cx, cy, 3.25, 3.4, x, y)
    paup = ellipse(cx, cy - 0.4, 4.7, 4.75, x, y) & ~oeil & dedans
    g[paup & (y < cy - 1.0)] = "l"
    g[paup & (y < cy - 1.0) & (x < cx)] = "L"
    g[paup & (y >= cy - 1.0) & (y < cy + 2.0) & (x > cx)] = "D"
    g[ellipse(cx, cy + 0.45, 3.85, 4.0, x, y) & ~oeil & (y > cy + 1.0)] = "X"
    if cligne:
        # paupières closes : la peau verte couvre le globe, fente sombre en arc, paupière
        # basse pâle (la membrane, gris lilas) ; le cerne noir disparaît
        fente = cy + 1.0 - 0.07 * (x - cx) ** 2
        g[oeil] = "V"
        g[oeil & (y < cy - 1.6)] = "l"
        g[oeil & (y < cy - 2.4) & (x < cx)] = "L"
        g[oeil & (n1 < 0.12) & (y < fente - 0.6)] = "N"
        g[oeil & (y > fente + 0.5)] = "l"
        g[oeil & (y > fente + 1.3)] = "V"
        g[oeil & (y > fente + 2.2)] = "D"
        g[oeil & (np.abs(y - fente) <= 0.5)] = "X"
        g[oeil & ~ellipse(cx, cy, 3.3, 3.45, x, y) & (y > cy)] = "D"
    else:
        # iris gris lavande, plus sombre vers le bas et le bord, finement réticulé ;
        # pupille verticale en fente, pointue aux deux bouts
        g[oeil] = "e"
        g[iris] = "A"
        g[iris & (y < cy + 1.0) & ellipse(cx - 0.3, cy - 0.3, 2.9, 3.1, x, y)] = "G"
        g[iris & (n1 < 0.1)] = "r"
        # (une colonne de pixels sur toute la hauteur, trois au milieu), et reflets, tracés à
        # l'écran autour du centre de l'œil tourné avec la tête, pour rester nets
        ox = px + (cx - px) * math.cos(a0) + (cy - py) * math.sin(a0)
        oy = py - (cx - px) * math.sin(a0) + (cy - py) * math.cos(a0)
        ox = math.floor(ox * ECHELLE) / ECHELLE + 0.25
        oy = math.floor((oy + MARGE) * ECHELLE) / ECHELLE - MARGE
        demi = np.where(np.abs(Y - oy) < 1.1, 0.76, 0.26)
        g[iris & (np.abs(Y - oy) < 2.9) & (np.abs(X - ox) < demi)] = "E"
        g[(np.abs(X - ox + 1.75) < 0.5) & (np.abs(Y - oy + 1.5) < 0.5)] = "W"
        g[(np.abs(X - ox - 1.5) < 0.1) & (np.abs(Y - oy - 2.25) < 0.1)] = "w"
    # narine et bout du museau
    g[ellipse(2.9, 7.6, 0.4, 0.4, x, y)] = "X"
    g[ellipse(3.6, 5.8, 0.7, 0.5, x, y) & dedans] = "L"

    # où poser des pustules : la peau du corps restée visible (dos, flanc, cuisse, tibia, bras
    # proche), sauf la tête, les mains, les pieds, le bras du fond et l'ombre de contact
    peau = np.isin(g, list(VERT + ORANGE + "Ffvus")) & (X > 17.0) & ~contact
    peau &= ~(orteils | proche | tm | (bras_loin & ~bras))
    for _ in range(2):  # loin des bords
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            peau &= np.roll(np.roll(peau, dy, 0), dx, 1)

    # contour : presque noir, sauf côté lumière des membres, où il prend la couleur sombre
    # de la peau qu'il cerne (brun sous l'orangé, vert sous le vert)
    plein = g != "."
    membres = (jm | talon | tm | orteils | bras | proche | (bras_loin & ~dedans)) & plein
    autour = np.zeros_like(plein)
    eclaire, ombre = np.zeros_like(plein), np.zeros_like(plein)
    vert_voisin = np.zeros_like(plein)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        decale = np.roll(np.roll(plein, dy, 0), dx, 1)
        if dy == 1:
            decale[0, :] = False
        if dy == -1:
            decale[-1, :] = False
        if dx == 1:
            decale[:, 0] = False
        if dx == -1:
            decale[:, -1] = False
        autour |= decale
        if dy == -1 or dx == -1:
            eclaire |= decale & np.roll(np.roll(membres, dy, 0), dx, 1)
            vert_voisin |= decale & np.roll(np.roll(np.isin(g, list(VERT)), dy, 0), dx, 1)
        else:
            ombre |= decale
    g[~plein & autour] = "K"
    lumiere = ~plein & eclaire & ~ombre
    g[lumiere] = np.where(vert_voisin[lumiere], "j", "k")
    return ["".join(r) for r in g], contact, peau


def rampe(n, h0, h1, s0, s1, l0, l1, dh=0.0):
    """n couleurs du sombre au clair : teinte, saturation et clarté interpolées (TSL)."""
    out = []
    for i in range(n):
        t = i / (n - 1)
        r, v, b = colorsys.hls_to_rgb(
            ((h0 + (h1 - h0) * t + dh) % 360) / 360, l0 + (l1 - l0) * t, s0 + (s1 - s0) * t
        )
        out.append(f"#{round(r * 255):02X}{round(v * 255):02X}{round(b * 255):02X}")
    return out


# matière : (clarté de chaque ton de base, une rampe par variante de teinte). Le vert tire au
# bleu dans l'ombre et au jaune dans la lumière ; l'orangé, au rouge dans l'ombre.
MATIERES = {
    "vert": (
        {"L": 1.0, "l": 0.78, "V": 0.55, "N": 0.38, "D": 0.3, "X": 0.1},
        [rampe(9, 165, 92, 0.46, 0.44, 0.09, 0.7, dh) for dh in (-6, 0, 6)],
    ),
    "orange": (
        {"I": 1.0, "H": 0.78, "O": 0.55, "S": 0.32, "Z": 0.12},
        [rampe(8, 8, 38, 0.66, 0.9, 0.2, 0.78, dh) for dh in (-4, 0, 5)],
    ),
    "jaune": (
        {"J": 0.6},
        [rampe(3, 78, 92, 0.5, 0.56, 0.42, 0.7, dh) for dh in (-3, 0, 3)],
    ),
    "barre": (
        {"B": 0.25, "b": 0.65},
        [rampe(4, 326, 342, 0.42, 0.3, 0.07, 0.3, dh) for dh in (-4, 0, 4)],
    ),
    "flanc": (
        {"F": 0.5, "f": 0.8, "v": 0.5, "u": 0.5, "s": 0.2},
        [rampe(7, 8, 30, 0.5, 0.72, 0.3, 0.78, dh) for dh in (-4, 0, 5)],
    ),
    "creme": (
        {"C": 0.85, "g": 0.62, "c": 0.35, "q": 0.5},
        [rampe(6, 290, 335, 0.12, 0.3, 0.48, 0.92, dh) for dh in (-8, 0, 8)],
    ),
}
# codes des nouvelles teintes : caractères libres, ni chiffre, ni « . », ni guillemet
_LIBRES = [
    chr(c)
    for c in list(range(0x21, 0x7F)) + list(range(0xC0, 0x24F))
    if not chr(c).isdigit() and chr(c) not in '."\\' and chr(c) not in PALETTE
]
PALETTE_NUANCES = dict(PALETTE)
_PAR_TEINTE = {}


def code(hexa):
    """Code d'une teinte de la palette nuancée (créé à la première demande)."""
    if hexa not in _PAR_TEINTE:
        _PAR_TEINTE[hexa] = _LIBRES.pop(0)
        PALETTE_NUANCES[_PAR_TEINTE[hexa]] = hexa
    return _PAR_TEINTE[hexa]


CODES = {
    (nom, v, i): code(hexa)
    for nom, (_, rampes) in MATIERES.items()
    for v, r in enumerate(rampes)
    for i, hexa in enumerate(r)
}


def retoucher(ch, sombre=1.0, gris=1.0, clair=0.0):
    """Le même pixel, de la même teinte, plus sombre (et grisé) ou plus clair."""
    hexa = PALETTE_NUANCES[ch]
    r, v, b = (int(hexa[k : k + 2], 16) / 255 for k in (1, 3, 5))
    t, cl, sa = colorsys.rgb_to_hls(r, v, b)
    cl = cl * sombre
    r, v, b = colorsys.hls_to_rgb(t, cl + (1 - cl) * clair, sa * gris)
    return code(f"#{round(r * 255):02X}{round(v * 255):02X}{round(b * 255):02X}")


def assombrir(ch):
    """Le même pixel dans l'ombre de contact : sa teinte, plus sombre et un peu grisée."""
    return retoucher(ch, sombre=0.84, gris=0.72)


_K = np.array([1, 4, 6, 4, 1], float) / 16
_YY, _XX = np.mgrid[0:H, 0:W]
_LUMIERE = 0.07 * (0.5 - (_XX / W * 0.6 + _YY / H * 0.4))  # un peu plus clair en haut à gauche
_v = np.sin(_XX * 12.9898 + _YY * 78.233) * 43758.5453
_GRAIN = _v - np.floor(_v) - 0.5
GRAIN = 0.045
_champ = np.kron(np.random.default_rng(11).random((H // 8 + 2, W // 8 + 2)), np.ones((8, 8)))


def flou(a):
    a = np.apply_along_axis(lambda r: np.convolve(r, _K, mode="same"), 1, a)
    return np.apply_along_axis(lambda c: np.convolve(c, _K, mode="same"), 0, a)


_champ = _champ[:H, :W]
for _ in range(3):
    _champ = flou(_champ)
_champ = (_champ - _champ.min()) / (_champ.max() - _champ.min())
_BAYER2 = (np.array([[0, 2], [3, 1]])[_YY % 2, _XX % 2] + 0.5) / 4
# plaques de teinte : basse fréquence, bords tramés
VARIANTE = np.clip(np.floor(_champ * 3 + (_BAYER2 - 0.5) * 0.7), 0, 2).astype(int)


def nuancer(rangs):
    """Rangées en tons de base → tableau de teintes fines (voir MATIERES)."""
    g = np.array([list(r) for r in rangs])
    out = g.copy()
    for nom, (tons, rampes) in MATIERES.items():
        m = np.isin(g, list(tons))
        if not m.any():
            continue
        clarte = np.zeros(g.shape)
        for ch, val in tons.items():
            clarte[g == ch] = val
        fm, fc = flou(m.astype(float)), flou(clarte * m)
        lisse = np.where(fm > 0, fc / np.maximum(fm, 1e-6), clarte)
        final = 0.5 + (0.45 * clarte + 0.55 * lisse - 0.5) * 1.22 + _LUMIERE + _GRAIN * GRAIN
        n = len(rampes[0])
        idx = np.clip(np.rint(final * (n - 1)), 0, n - 1).astype(int)
        # grains du flanc : la teinte alentour, un seul cran plus clair ou plus sombre
        idx = np.clip(idx + (g == "v") - (g == "u"), 0, n - 1)
        for v in range(len(rampes)):
            for i in range(n):
                out[m & (idx == i) & (VARIANTE == v)] = CODES[nom, v, i]
    return out


_PUSTULES = []


def pustules():
    """Places des pustules, tirées une fois sur l'image au repos pour qu'elles ne bougent pas
    d'une image à l'autre : espacées, sur la peau du corps hors tête (voir image)."""
    if not _PUSTULES:
        peau = image()[2]
        ok = (
            peau
            & np.roll(peau, -1, 0)
            & np.roll(peau, -1, 1)
            & np.roll(np.roll(peau, -1, 0), -1, 1)
        )
        candidats = list(zip(*np.nonzero(ok), strict=True))
        random.Random(13).shuffle(candidats)
        for j, i in candidats:
            if all(abs(j - a) + abs(i - b) >= 7 for a, b in _PUSTULES):
                _PUSTULES.append((int(j), int(i)))
    return _PUSTULES


def composer(gorge, souffle, cligne, tete):
    """L'image finale : tons nuancés, flanc derrière le bras assombri, pixel par pixel, et
    pustules : un pixel de la couleur d'origine, plus clair, son ombre juste en dessous."""
    rangs, contact, peau = image(gorge, souffle, cligne, tete)
    out = nuancer(rangs)
    for j, i in zip(*np.nonzero(contact & (out != ".") & (out != "K")), strict=True):
        out[j, i] = assombrir(out[j, i])
    for j, i in pustules():
        if peau[j, i] and peau[j + 1, i] and peau[j + 1, i + 1]:
            out[j, i] = retoucher(out[j, i], clair=0.3)
            out[j, i + 1] = retoucher(out[j, i + 1], clair=0.12)
            out[j + 1, i] = retoucher(out[j + 1, i], sombre=0.78, gris=0.9)
            out[j + 1, i + 1] = retoucher(out[j + 1, i + 1], sombre=0.85, gris=0.9)
    return ["".join(r) for r in out]


def plages(ligne):
    """« ...VVVK » → « .3V3K1 »."""
    out, i = [], 0
    while i < len(ligne):
        j = i
        while j < len(ligne) and ligne[j] == ligne[i]:
            j += 1
        out.append(f"{ligne[i]}{j - i}")
        i = j
    return "".join(out)


if __name__ == "__main__":
    images = {}
    for g_ in (0, 1):
        for s in (0, 1):
            for c in (0, 1):
                for t in (-1, 0, 1):
                    rangs = composer(g_ * 0.9, s * 0.7, bool(c), t)
                    images[f"{g_}{s}{c}{t + 1}"] = [plages(r) for r in rangs]
    (ICI / "tomopterna.json").write_text(
        json.dumps(
            {"palette": PALETTE_NUANCES, "largeur": W, "hauteur": H, "images": images},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    teintes = {ch for rangs in images.values() for r in rangs for ch in r if not ch.isdigit()}
    print(f"{len(images)} images, {len(teintes - {'.'})} couleurs")
