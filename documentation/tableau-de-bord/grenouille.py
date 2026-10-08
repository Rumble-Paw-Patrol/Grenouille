"""Sprite 8-bit d'A. blanci pour le bandeau du tableau de bord.

    uv run python documentation/tableau-de-bord/grenouille.py

Grenouille de profil, tournée vers le titre (à gauche), assise, accroupie, dessinée d'après les
photos d'A. blanci de Benoît Villette et d'Arnaud Aury pour la tête, la robe et les couleurs,
et d'après des photos d'Anomaloglossus (dont M. Dewynter) pour la posture et les membres :
patte arrière en Z (cuisse arrondie tournée vers nous, dont le haut cache la bande du dos ;
tibia long et bombé qui dépasse derrière la croupe ; talon arrondi ; tarse au sol, orteils en
éventail), bras pliés dont l'épaule s'attache en arrondi et tourne au jaune avant de se fondre
dans le flanc, bras de l'autre côté en arrière-plan, mains et pieds en gabarits posés au pixel
près (ORTEILS, MAIN_PROCHE, MAIN_LOINTAINE).

Deux rendus sont calculés et assemblés pixel par pixel (composer) : le corps vient du rendu
« ancien », le pied, le tibia, les bras et les mains, avec leur contour, du rendu « nouveau »
(ombres de la peau plus rouges et lumières plus dorées, moins de grain, contour brun côté
lumière). Le flanc juste derrière le bras garde sa teinte, assombrie et un peu grisée (ombre de
contact) ; le ventre est semé d'un grain doux, un cran plus clair ou plus sombre que le flanc
alentour.

Grenouille de profil, tournée vers le titre (à gauche), dessinée d'après les photos d'A. blanci
de Benoît Villette et d'Arnaud Aury : museau court, grand œil noir cerclé de doré, bande sombre
du museau au flanc, gorge claire, longue patte avant aux doigts fins, cuisse repliée et
arrière-train posé presque au sol. Les formes sont décrites dans un repère de 57 × 43 unités
et tracées sur une grille de 114 × 86 pixels (deux pixels par unité), sans anticrénelage, puis
cernées d'un contour, avec 22 tons de base.

Une passe de nuances (nuancer) affine ensuite chaque matière (peau orangée, bande sombre,
flanc, gorge et ventre) : ses tons de base sont adoucis par un flou limité à la matière, puis
redistribués sur une rampe plus fine (de 7 à 12 niveaux de clarté), en trois variantes de
teinte (plus rouge, neutre, plus dorée) réparties par plaques. Les passages du clair au foncé
gagnent des teintes intermédiaires, le relief garde son grain : environ 80 couleurs en tout.

Une image par combinaison gorge (gonflée ou non) × flanc (gonflé ou non) × clignement ×
tête (baissée, droite, relevée) : la page les enchaîne pour que la grenouille respire, cligne
et bouge un peu la tête. Écrit grenouille.json (palette, taille et images codées par
plages : « O12 » = douze pixels de la couleur O), au format que lit construire.py.
"""

import colorsys
import json
import math
import random
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
PALETTE = {
    "K": "#24120A",  # contour
    "k": "#5A2B14",  # contour côté lumière des membres (style « nouveau »)
    "I": "#F8C68A",  # reflet vif, peau brillante
    "H": "#E89F55",  # reflet
    "O": "#D07A35",  # dos orangé
    "S": "#A8591F",  # ombre
    "Z": "#7E3F17",  # ombre profonde, plis
    "M": "#9C5223",  # mouchetures
    "B": "#4A230E",  # bande sombre
    "b": "#6E3818",  # bord de la bande
    "F": "#C2804A",  # flanc
    "f": "#D9A574",  # flanc clair
    "J": "#F2C46A",  # haut de l'épaule, qui tourne au jaune
    "v": "#CE9561",  # grain clair du ventre : un cran plus clair que le flanc alentour
    "u": "#B57643",  # grain sombre du ventre : un cran plus sombre que le flanc alentour
    "T": "#F2CC98",  # lèvre
    "C": "#E8DCC4",  # gorge, ventre
    "g": "#CFC4AE",  # gorge grisée
    "c": "#ADA08A",  # ventre ombré, points de la gorge
    "q": "#C8B49A",  # ventre grisé
    "E": "#0B0705",  # œil
    "e": "#22150E",  # bas du globe de l'œil
    "G": "#C9A05A",  # cercle doré de l'œil
    "A": "#776647",  # cercle de l'œil, en bas
    "W": "#FFFFFF",  # reflet de l'œil
    "w": "#9FB4C0",  # second reflet, bleuté
}
UNITES = (57, 43)
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
ECHELLE = 2
W, H = UNITES[0] * ECHELLE, UNITES[1] * ECHELLE
# centres des pixels, en unités
X, Y = np.meshgrid((np.arange(W) + 0.5) / ECHELLE, (np.arange(H) + 0.5) / ECHELLE)


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


# Mains et pieds, dessinés au pixel près (tons de base ; « . » = rien ; « w » = reflet
# bleuté de peau humide sur un disque). Chaque doigt : phalanges d'un pixel, articulation
# plus claire, disque terminal de 2 × 2 plus clair que le doigt, un pixel vide entre deux
# doigts (le contour le noircit). Les doigts lointains sont plus sombres et plus hauts.
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
    (31, 77),
    [
        ".....IH..........",
        ".....HOOO........",
        "........OO..HOOS.",
        ".........OHHHOOOS",
        "...IwOOHOOOOOOOOS",
        "...HO....OOOOOSS.",
        "......IHOOSOOSS..",
        "......HO.........",
    ],
)
MAIN_LOINTAINE = (
    (27, 67),
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


def image(gorge=0.0, souffle=0.0, cligne=False, tete=0, style="ancien"):
    """Tons de base d'une image, le masque des membres (pied, tibia, bras, mains, avec
    leur contour) et celui de l'ombre de contact derrière le bras. Le style « nouveau »
    sème moins de grain sur les membres et cerne leur côté lumière de brun."""
    neuf = style == "nouveau"
    g = np.full((H, W), ".", dtype="<U1")
    # Tête baissée ou relevée : rotation de quelques degrés autour du cou, qui s'estompe
    # vers le corps et ne touche pas les pattes. On échantillonne à l'envers.
    poids = np.clip((21 - X) / 8, 0, 1) * np.clip((19.5 - Y) / 3.5, 0, 1)
    a = -tete * 0.06 * poids
    px, py = 18.0, 9.5
    x = px + (X - px) * np.cos(a) - (Y - py) * np.sin(a)
    y = py + (X - px) * np.sin(a) + (Y - py) * np.cos(a)
    n1, n2, n3 = bruit(x, y, 1), bruit(x, y, 2), bruit(X, Y, 3)
    alea = random.Random(7)

    # --- bras droit, de l'autre côté : on le voit sous la gorge, devant le bras gauche,
    # dans l'ombre ; sa main se pose un peu plus loin, donc un peu plus haut
    loin = catmull(
        [
            (19.0, 18.2),
            (22.6, 18.4),
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
    bras_loin = lm | poser(g, MAIN_LOINTAINE)

    # --- corps et tête : la bosse de l'autre œil dépasse du crâne
    corps = catmull(
        [
            (1.2, 5.6),
            (2.2, 4.1),
            (3.8, 2.8),
            (5.6, 1.7),
            (7.2, 1.0),
            (8.8, 0.9),
            (10.2, 1.5),
            (11.6, 2.4),
            (13.6, 3.6),
            (16.5, 5.2),
            (21, 7.8),
            (25.5, 10.1),
            (30, 12.8),
            (34, 14.8),
            (38, 16.6),
            (41.5, 18.2),
            (44.6, 20.0),
            (47.4, 22.2),
            (49.6, 25.0),
            (50.8, 28.8),
            (51.2, 33.0),
            (50.4, 36.4),
            (48.4, 37.6),
            (44.4, 37.8),
            (41.0, 37.0),
            (38, 35.0),
            (33, 33.2 + souffle),
            (29, 31.4 + souffle),
            (26, 29.2 + souffle * 0.6),
            (23.6, 26.4),
            (22.4, 24.6),
            (19.4, 20.6),
            (16.6, 17.8),
            (13.5, 15.9 + gorge * 0.4),
            (10, 14.5 + gorge),
            (6.5, 12.9 + gorge * 0.8),
            (3.4, 11.1 + gorge * 0.3),
            (1.6, 9.4),
            (0.9, 7.4),
        ]
    )
    dedans = dans(corps, x, y)
    bande = [
        (0, 7.2),
        (4, 7.6),
        (7.5, 8.2),
        (11, 8.8),
        (15, 10.3),
        (19, 12.5),
        (24, 15.1),
        (29, 18.0),
        (34, 21.0),
        (38, 23.4),
        (42, 24.6),
        (46, 25.8),
        (50, 27.4),
    ]
    yb = suivre(bande, x) + (n2 - 0.5) * 0.35 * (x > 15)  # bord irrégulier sur le flanc
    e = np.where(x < 7.5, 0.6, np.where(x < 18, 1.0, 1.25))
    haut1, haut2 = ~dans(corps, x, y - 0.6), ~dans(corps, x, y - 1.3)
    bas1, bas2 = ~dans(corps, x, y + 0.6), ~dans(corps, x, y + 1.6)

    # dos : grain de peau (granules claires et sombres), quelques taches, reflet le long du dos
    dos = dedans & (y < yb - e)
    g[dos] = "O"
    g[dos & (n1 < 0.06)] = "M"
    g[dos & (n1 > 0.94)] = "H"
    for _ in range(22):
        mx, my, r = alea.uniform(13, 38), alea.uniform(5, 22), alea.uniform(0.45, 0.85)
        g[ellipse(mx, my, r, r * 0.8, x, y) & dos & (n2 > 0.25)] = "M"
    g[dos & (y > yb - e - 0.5) & (x > 13) & (n2 > 0.3)] = "S"
    g[dos & haut2 & (n1 > 0.25)] = "H"
    g[dos & haut1] = "I"
    g[dos & haut1 & (n1 < 0.12)] = "H"

    # bande sombre, du museau au flanc, bords effilochés
    sur_bande = dedans & (np.abs(y - yb) <= e) & (x >= 2.6)
    g[dedans & (np.abs(y - yb) <= e) & (x < 2.6)] = "O"
    g[sur_bande] = "B"
    g[sur_bande & (np.abs(y - yb) > e - 0.5) & (x > 14) & (n1 > 0.35)] = "b"

    # sous la bande : lèvre claire jusqu'à l'épaule, joue, gorge mouchetée, flanc marbré
    dessous = dedans & (y > yb + e)
    tete_ = dessous & (x < 16)
    g[tete_] = "g"
    g[tete_ & (n1 < 0.16)] = "c"
    joue = tete_ & (y <= yb + e + 2.3 - np.clip(x - 10, 0, None) * 0.55)
    g[joue] = "O"
    g[joue & (n1 > 0.85)] = "H"
    flanc = dessous & (x >= 16)
    g[flanc] = "F"
    g[flanc & (n1 > 0.72)] = "f"
    g[flanc & (n2 > 0.86)] = "v"
    g[flanc & (n2 < 0.1)] = "u"
    levre = dessous & (y <= yb + e + 0.8) & (x < 20)
    g[levre] = np.where(x[levre] < 9, "T", "C")
    g[dessous & (y > yb + e + 0.8) & (y <= yb + e + 1.5) & (x < 9)] = "f"
    g[dessous & bas2 & (x >= 14)] = "C"
    g[dessous & bas2 & (x >= 14) & (n1 < 0.2)] = "g"
    g[dessous & bas1] = "c"
    # arrière-train : postérieur posé au sol et croupe, dans l'ombre, sans ventre clair
    croupe = dedans & (x > 44) & (y > yb + e)
    g[croupe] = "O"
    g[croupe & (n1 < 0.08)] = "M"
    g[croupe & (y > 31)] = "S"
    g[dedans & (x > 40) & (y > 36)] = "S"
    g[tete_ & bas1] = "g"

    # --- cuisse : longue, du genou (à l'avant) vers l'arrière, couchée sur la croupe dont
    # elle dépasse ; le tibia la recouvre par-dessous. Pli là où elle touche le corps
    cuisse = catmull(
        [
            (33.4, 29.07),
            (36.0, 26.81),
            (40.0, 25.5),
            (44.0, 24.6),
            (47.4, 24.1),
            (49.8, 25.3),
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
    g[cm] = "O"
    g[cm & (n3 < 0.1)] = "M"
    g[cm & (n3 > 0.94)] = "H"
    g[cm & ~dans(cuisse, X + 0.4, Y - 2.4)] = "H"
    g[cm & ~dans(cuisse, X + 0.2, Y - 1.0)] = "I"
    g[cm & (np.abs((X - 43.0) + (Y - 30) * 0.3) < 0.7) & (n3 > 0.3)] = "M"  # barre
    g[ellipse(39.6, 27.2, 2.0, 0.5, X, Y, -0.05)] = "I"  # reflet de la peau humide
    bord = np.zeros_like(cm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bord |= cm & ~np.roll(np.roll(cm, dy, 0), dx, 1) & np.roll(np.roll(avant, dy, 0), dx, 1)
    g[bord] = "Z"

    # --- pied : le tarse part du talon, à l'arrière contre le sol, et file vers l'avant
    # sous le tibia ; trois orteils longs et fins s'en écartent vers l'avant, chacun avec
    # son disque. Les plus lointains sont plus hauts et plus sombres.
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
    orteils = poser(g, ORTEILS)

    # --- tibia : au premier plan, grosse masse ronde du genou (en haut, à l'avant) au talon
    # (en bas, à l'arrière, contre le sol), par-dessus la cuisse ; reflet de peau humide
    jambe = catmull(
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
    jm = dans(jambe, X, Y)
    sous_jambe = g != "."
    g[jm] = "O"
    g[jm & (n3 < (0.06 if neuf else 0.1))] = "M"
    g[jm & (n3 > (0.97 if neuf else 0.95))] = "H"
    g[jm & ~dans(jambe, X + 0.4, Y - 1.4)] = "H"
    g[jm & ~dans(jambe, X + 0.2, Y - 0.7) & (X < 44)] = "I"
    g[jm & ~dans(jambe, X - 0.2, Y + 1.6)] = "S"
    g[jm & ~dans(jambe, X, Y + 0.7)] = "Z"
    g[jm & (np.abs((X - 43.0) + (Y - 37) * 0.3) < 0.7) & (n3 > 0.3) & (g == "O")] = "M"
    g[ellipse(36.4, 31.1, 2.2, 0.5, X, Y, -0.05)] = "I"  # reflet au genou
    pli = np.zeros_like(jm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin_hors = ~np.roll(np.roll(jm, dy, 0), dx, 1)
        voisin_dessous = np.roll(np.roll(sous_jambe, dy, 0), dx, 1)
        pli |= jm & voisin_hors & voisin_dessous & (X < 52)  # pas de pli au talon
    g[pli] = "Z"
    # cheville : talon arrondi, assez épais, qui joint le bout du tibia au tarse
    talon = ellipse(52.9, 38.9, 1.3, 1.15, X, Y)
    g[talon] = "O"
    g[talon & ~ellipse(52.9, 39.5, 1.3, 1.15, X, Y)] = "H"
    g[talon & ~ellipse(52.9, 38.4, 1.3, 1.15, X, Y)] = "S"

    # --- patte avant : sort du flanc sous la bande, coude en arrière, avant-bras plus
    # épais près du coude, poignet fin
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
    # du corps. Le haut du capuchon tourne au jaune avant de se fondre dans le flanc.
    corps_dessous = g != "."
    cx_e, cy_e, rx_e, ry_e = 22.6, 22.6, 2.2, 2.8
    rayon = np.sqrt(((X - cx_e) / rx_e) ** 2 + ((Y - cy_e) / ry_e) ** 2)
    capuchon = (rayon <= 1) & (Y < cy_e)
    trame = (BAYER4[(np.arange(H) % 4)[:, None], (np.arange(W) % 4)[None, :]] + 0.5) / 16
    fondu = capuchon & corps_dessous & (trame < np.clip((1 - rayon) / 0.45, 0, 1))
    bras = (pa & (Y >= cy_e)) | fondu | (pa & ~corps_dessous)
    g[bras] = "O"
    g[bras & (n3 < (0.06 if neuf else 0.1))] = "M"
    g[bras & (n3 > (0.97 if neuf else 0.93))] = "f"
    # lumière d'en haut à gauche : bord avant éclairé, bord arrière dans l'ombre
    g[bras & ~dans(patte, X - 1.1, Y)] = "H"
    g[bras & ~dans(patte, X - 0.6, Y)] = "I"
    g[bras & ~dans(patte, X + 1.1, Y) & (Y > 21.0)] = "S"
    g[bras & ~dans(patte, X + 0.6, Y) & (Y > 22.0)] = "Z"
    g[ellipse(26.4, 30.6, 0.6, 0.9, X, Y, 0.4)] = "I"  # reflet au coude
    g[bras & capuchon & (rayon > 0.55)] = "J"  # haut de l'épaule, jaune, sur son pourtour
    g[bras & capuchon & (rayon <= 0.55) & (Y < cy_e - 0.6)] = "I"
    # ombre de contact : le flanc juste derrière le bras, sous l'épaule (assombri plus tard)
    contact = corps_dessous & ~bras & (Y > 21.5) & (np.roll(bras, 1, 1) | np.roll(bras, 2, 1))
    separer(g, bras, bras_loin & ~dedans)  # contour entre les deux bras
    g[miettes(bras_loin & ~dedans & ~bras & (g != "."))] = "."  # restes entre gorge et bras
    # main tournée vers l'intérieur : poignet, paume et trois doigts vers l'avant qui filent
    # un peu vers le fond (gabarit MAIN_PROCHE)
    proche = poser(g, MAIN_PROCHE)
    separer(g, proche, bras_loin & ~proche)

    # --- l'autre œil : une simple bosse de peau sur le crâne, éclairée par-dessus et
    # soulignée d'une ombre à sa base, qui ne bouge pas quand l'œil visible cligne
    dome = ellipse(8.4, 2.2, 2.5, 1.75, x, y)
    g[dome] = "O"
    g[dome & (n1 > 0.9)] = "H"
    g[dome & ~ellipse(8.4, 2.5, 2.5, 1.75, x, y)] = "I"
    g[dome & ellipse(7.9, 1.8, 1.3, 0.8, x, y) & ~ellipse(8.4, 2.5, 2.5, 1.75, x, y)] = "H"
    g[dome & ~ellipse(8.4, 1.6, 2.5, 1.75, x, y)] = "S"

    # --- l'œil visible : globe saillant, paupière supérieure en relief, pli dessous
    cx, cy = 11.2, 8.4
    oeil = ellipse(cx, cy, 3.8, 3.65, x, y)
    iris = ellipse(cx, cy, 2.9, 2.8, x, y)
    paup = ellipse(cx, cy - 0.5, 4.7, 4.5, x, y) & ~oeil & dedans
    g[paup & (y < cy - 1.0)] = "H"
    g[paup & (y < cy - 1.0) & (x < cx)] = "I"
    g[paup & (y >= cy - 1.0) & (y < cy + 2.0) & (x > cx)] = "S"
    g[ellipse(cx, cy + 0.45, 3.85, 3.7, x, y) & ~oeil & (y > cy + 1.0)] = "Z"
    if cligne:
        # paupières closes : la peau couvre le globe, fente sombre en arc, paupière basse pâle
        fente = cy + 1.0 - 0.07 * (x - cx) ** 2
        g[oeil] = "O"
        g[oeil & (y < cy - 1.6)] = "H"
        g[oeil & (y < cy - 2.4) & (x < cx)] = "I"
        g[oeil & (n1 < 0.12) & (y < fente - 0.6)] = "M"
        g[oeil & (y > fente + 0.5)] = "f"
        g[oeil & (y > fente + 1.6)] = "S"
        g[oeil & (np.abs(y - fente) <= 0.5)] = "B"
        g[oeil & (np.abs(y - fente - 0.7) <= 0.2) & (np.abs(x - cx) < 2.2)] = "Z"
        g[oeil & ~ellipse(cx, cy, 3.3, 3.15, x, y) & (y > cy)] = "S"
    else:
        g[oeil] = "A"
        g[oeil & (y < cy + 0.6)] = "G"
        g[iris] = "E"
        g[
            iris
            & ellipse(cx + 0.7, cy + 0.9, 2.3, 1.9, x, y)
            & ~ellipse(cx - 0.4, cy - 0.5, 2.4, 2.3, x, y)
        ] = "e"
        g[ellipse(cx - 1.1, cy - 1.2, 0.85, 0.8, x, y)] = "W"
        g[ellipse(cx + 1.2, cy + 1.5, 0.4, 0.4, x, y)] = "w"
    # narine et bout du museau
    g[ellipse(2.1, 5.9, 0.4, 0.4, x, y)] = "B"
    g[ellipse(2.8, 4.0, 0.7, 0.5, x, y) & dedans] = "I"

    # contour
    plein = g != "."
    membres = (jm | talon | tm | orteils | bras | proche | (bras_loin & ~dedans)) & plein
    autour = np.zeros_like(plein)
    eclaire, ombre = np.zeros_like(plein), np.zeros_like(plein)
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
            eclaire |= decale
        else:
            ombre |= decale
    g[~plein & autour] = "K"
    if neuf:  # brun chaud côté lumière, presque noir côté ombre et entre deux formes
        g[~plein & eclaire & ~ombre] = "k"
    anneau = np.zeros_like(membres)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        anneau |= np.roll(np.roll(membres, dy, 0), dx, 1)
    membres |= anneau & ~plein & autour
    return ["".join(r) for r in g], membres, contact & ~membres


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


# matière : (clarté de chaque ton de base, une rampe par variante de teinte, rouge → dorée).
# Deux styles : « ancien » pour le corps, « nouveau » pour les membres (pied, tibia, bras,
# mains), dont la peau a des ombres plus rouges et des lumières plus dorées.
_COMMUNES = {
    "jaune": (
        {"J": 0.6},
        [rampe(5, 36, 46, 0.72, 0.82, 0.5, 0.78, dh) for dh in (-3, 0, 3)],
    ),
    "bande": (
        {"B": 0.25, "b": 0.65},
        [rampe(7, 18, 24, 0.62, 0.55, 0.11, 0.33, dh) for dh in (-3, 0, 4)],
    ),
    "flanc": (
        {"F": 0.45, "f": 0.78, "v": 0.45, "u": 0.45},
        [rampe(8, 22, 32, 0.45, 0.6, 0.42, 0.72, dh) for dh in (-4, 0, 6)],
    ),
    "creme": (
        {"C": 0.85, "g": 0.62, "c": 0.35, "q": 0.5},
        [rampe(8, 30, 40, 0.14, 0.38, 0.52, 0.9, dh) for dh in (-6, 0, 6)],
    ),
}
_PEAU = {"I": 1.0, "H": 0.78, "O": 0.55, "M": 0.34, "S": 0.32, "Z": 0.12}
MATIERES = {
    "ancien": {
        "orange": (_PEAU, [rampe(12, 17, 34, 0.66, 0.86, 0.22, 0.78, dh) for dh in (-4, 0, 5)])
    }
    | _COMMUNES,
    "nouveau": {
        "orange": (_PEAU, [rampe(12, 13, 37, 0.66, 0.86, 0.22, 0.78, dh) for dh in (-4, 0, 5)])
    }
    | _COMMUNES,
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
    (style, nom, v, i): code(hexa)
    for style, matieres in MATIERES.items()
    for nom, (_, rampes) in matieres.items()
    for v, r in enumerate(rampes)
    for i, hexa in enumerate(r)
}


def assombrir(ch):
    """Le même pixel dans l'ombre : sa teinte, plus sombre et un peu grisée."""
    hexa = PALETTE_NUANCES[ch]
    r, v, b = (int(hexa[k : k + 2], 16) / 255 for k in (1, 3, 5))
    t, cl, sa = colorsys.rgb_to_hls(r, v, b)
    r, v, b = colorsys.hls_to_rgb(t, cl * 0.84, sa * 0.72)
    return code(f"#{round(r * 255):02X}{round(v * 255):02X}{round(b * 255):02X}")


_K = np.array([1, 4, 6, 4, 1], float) / 16
_YY, _XX = np.mgrid[0:H, 0:W]
_LUMIERE = 0.07 * (0.5 - (_XX / W * 0.6 + _YY / H * 0.4))  # un peu plus clair en haut à gauche
_v = np.sin(_XX * 12.9898 + _YY * 78.233) * 43758.5453
_GRAIN = _v - np.floor(_v) - 0.5
GRAIN = {"ancien": 0.05, "nouveau": 0.035}
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


def nuancer(rangs, style="ancien"):
    """Rangées en tons de base → tableau de teintes fines (voir MATIERES)."""
    g = np.array([list(r) for r in rangs])
    out = g.copy()
    for nom, (tons, rampes) in MATIERES[style].items():
        m = np.isin(g, list(tons))
        if not m.any():
            continue
        clarte = np.zeros(g.shape)
        for ch, val in tons.items():
            clarte[g == ch] = val
        fm, fc = flou(m.astype(float)), flou(clarte * m)
        lisse = np.where(fm > 0, fc / np.maximum(fm, 1e-6), clarte)
        final = 0.5 + (0.45 * clarte + 0.55 * lisse - 0.5) * 1.22 + _LUMIERE + _GRAIN * GRAIN[style]
        n = len(rampes[0])
        idx = np.clip(np.rint(final * (n - 1)), 0, n - 1).astype(int)
        # grains du ventre : la teinte du flanc alentour, un seul cran plus clair ou plus sombre
        idx = np.clip(idx + (g == "v") - (g == "u"), 0, n - 1)
        for v in range(len(rampes)):
            for i in range(n):
                out[m & (idx == i) & (VARIANTE == v)] = CODES[style, nom, v, i]
    return out


def composer(gorge, souffle, cligne, tete):
    """L'image finale : le corps du rendu « ancien », les membres et leur contour copiés tels
    quels du rendu « nouveau », et le flanc derrière le bras assombri, pixel par pixel."""
    rangs, membres, contact = image(gorge, souffle, cligne, tete, "ancien")
    neufs, _, _ = image(gorge, souffle, cligne, tete, "nouveau")
    out = np.where(membres, nuancer(neufs, "nouveau"), nuancer(rangs, "ancien"))
    for j, i in zip(*np.nonzero(contact & (out != ".") & (out != "K")), strict=True):
        out[j, i] = assombrir(out[j, i])
    return ["".join(r) for r in out]


def plages(ligne):
    """« ...OOOK » → « .3O3K1 »."""
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
    (ICI / "grenouille.json").write_text(
        json.dumps(
            {"palette": PALETTE_NUANCES, "largeur": W, "hauteur": H, "images": images},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    teintes = {ch for rangs in images.values() for r in rangs for ch in r if not ch.isdigit()}
    print(f"{len(images)} images, {len(teintes - {'.'})} couleurs")
