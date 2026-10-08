"""Bac à sable : sprite 8-bit d'A. blanci aux pattes revues, avant reprise dans le bandeau.

    uv run python documentation/tableau-de-bord/bac-a-sable/grenouille.py

Copie de ../grenouille.py dont seules les pattes changent, d'après la morphologie réelle :
cuisse en fuseau de la hanche au genou, tibia au mollet bombé qui s'affine vers la cheville,
talon, tarse allongé posé au sol, orteils et doigts en éventail, articulés, à disques
terminaux. Écrit grenouille.json dans ce dossier (non suivi) ; le tableau de bord n'en lit rien.

Grenouille de profil, tournée vers le titre (à gauche), dessinée d'après les photos d'A. blanci
de Benoît Villette et d'Arnaud Aury : museau court, grand œil noir cerclé de doré, bande sombre
du museau au flanc, gorge claire, longue patte avant aux doigts fins, cuisse repliée et
arrière-train posé presque au sol. Les formes sont décrites dans un repère de 48 × 40 unités
et tracées sur une grille de 96 × 80 pixels (deux pixels par unité), sans anticrénelage, puis
cernées d'un contour, avec 22 tons de base.

Une passe de nuances (nuancer) affine ensuite chaque matière (peau orangée, bande sombre,
flanc, gorge et ventre) : ses tons de base sont adoucis par un flou limité à la matière, puis
redistribués sur une rampe plus fine (de 7 à 12 niveaux de clarté), en trois variantes de
teinte (plus rouge, neutre, plus dorée) réparties par plaques. Les passages du clair au foncé
gagnent des teintes intermédiaires, le relief garde son grain : environ 80 couleurs en tout.

Une image par combinaison gorge (gonflée ou non) × flanc (gonflé ou non) × clignement ×
tête (baissée, droite, relevée) : la page les enchaîne pour que la grenouille respire, cligne
et bouge un peu la tête. Écrit grenouille.json (palette, taille et images codées par
plages : « O12 » = douze pixels de la couleur O), lu par construire.py.
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
    "I": "#F8C68A",  # reflet vif, peau brillante
    "H": "#E89F55",  # reflet
    "O": "#D07A35",  # dos orangé
    "S": "#A8591F",  # ombre
    "Z": "#7E3F17",  # ombre profonde, plis
    "M": "#9C5223",  # mouchetures
    "B": "#4A230E",  # bande sombre
    "b": "#6E3818",  # bord de la bande
    "F": "#C2804A",  # flanc
    "f": "#D9A574",  # flanc clair, doigts
    "T": "#F2CC98",  # lèvre, pelotes des doigts
    "C": "#E8DCC4",  # gorge, ventre
    "g": "#CFC4AE",  # gorge grisée
    "c": "#ADA08A",  # ventre ombré, points de la gorge
    "q": "#C8B49A",  # dessous des orteils
    "E": "#0B0705",  # œil
    "e": "#22150E",  # bas du globe de l'œil
    "G": "#C9A05A",  # cercle doré de l'œil
    "A": "#776647",  # cercle de l'œil, en bas
    "W": "#FFFFFF",  # reflet de l'œil
    "w": "#9FB4C0",  # second reflet, bleuté
}
UNITES = (48, 40)
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


def segment(a, b, larg, x, y):
    (ax, ay), (bx, by) = a, b
    vx, vy = bx - ax, by - ay
    t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(x - ax - t * vx, y - ay - t * vy) <= larg / 2


def px(i, j):
    """Centre du pixel (colonne i, rangée j), en unités : pour placer doigts et orteils
    au pixel près."""
    return ((i + 0.5) / ECHELLE, (j + 0.5) / ECHELLE)


def doigt(g, pts, larg, ton, disque):
    """Doigt ou orteil (points en pixels, de la base au bout) : phalanges qui s'affinent,
    articulations plus claires, disque terminal élargi à pelote claire, dessous sombre.
    Les doigts voisins sont séparés d'un pixel vide, que le contour noircit."""
    pts = [px(*p) for p in pts]
    m = np.zeros(g.shape, bool)
    for a, b in zip(pts, pts[1:], strict=False):
        m |= segment(a, b, larg, X, Y)
    g[m] = ton
    if larg >= 0.9:
        g[m & ~np.roll(m, -1, 0)] = "q"
    for jx, jy in pts[1:-1]:
        g[ellipse(jx, jy, 0.3, 0.3, X, Y) & m] = "T" if ton == "f" else "f"
    (bx, by), (rx, ry) = pts[-1], disque
    d = ellipse(bx, by, rx, ry, X, Y)
    g[d] = "T"
    g[d & ~ellipse(bx, by - 0.5, rx, ry, X, Y)] = "q"


def suivre(pts, x):
    xs, ys = zip(*pts, strict=True)
    return np.interp(x, xs, ys)


def bruit(x, y, graine):
    """Valeur pseudo-aléatoire stable par demi-unité (grain de la peau), entre 0 et 1."""
    xi, yi = np.floor(x * ECHELLE), np.floor(y * ECHELLE)
    v = np.sin(xi * 12.9898 + yi * 78.233 + graine * 37.719) * 43758.5453
    return v - np.floor(v)


def image(gorge=0.0, souffle=0.0, cligne=False, tete=0):
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
            (34, 15.8),
            (37, 19),
            (39.5, 22.5),
            (40, 30),
            (35, 31.5),
            (31.5, 31.6 + souffle),
            (28, 31 + souffle),
            (25, 29),
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
    g[flanc & (n2 > 0.96)] = "C"
    g[flanc & (n2 < 0.05)] = "M"
    levre = dessous & (y <= yb + e + 0.8) & (x < 20)
    g[levre] = np.where(x[levre] < 9, "T", "C")
    g[dessous & (y > yb + e + 0.8) & (y <= yb + e + 1.5) & (x < 9)] = "f"
    g[dessous & bas2 & (x >= 14)] = "C"
    g[dessous & bas2 & (x >= 14) & (n1 < 0.2)] = "g"
    g[dessous & bas1] = "c"
    g[tete_ & bas1] = "g"

    # --- cuisse : fuseau musclé de la hanche (haut de l'arrière-train) au genou (sous le
    # flanc, vers l'avant), éclairé par-dessus, pli de l'aine le long du corps
    cuisse = catmull(
        [
            (36.2, 20.6),
            (39.6, 20.4),
            (42.2, 22.4),
            (43.4, 25.8),
            (42.6, 29.4),
            (40.0, 31.8),
            (35.6, 33.0),
            (31.2, 33.2),
            (28.6, 32.6),
            (27.9, 30.9),
            (29.0, 29.2),
            (31.8, 27.2),
            (34.0, 23.8),
        ]
    )
    cm = dans(cuisse, X, Y)
    avant = g != "."
    # axe de la cuisse, de la hanche au genou, et normale vers le haut
    ux, uy, nx, ny = -0.83, 0.55, -0.55, -0.83
    le_long = (X - 40.5) * ux + (Y - 24.5) * uy
    g[cm] = "O"
    g[cm & (n3 < 0.1)] = "M"
    g[cm & (n3 > 0.94)] = "H"
    for k, ton in ((3.0, "H"), (1.2, "I")):
        g[cm & ~dans(cuisse, X + nx * k, Y + ny * k)] = ton
    for k, ton in ((3.4, "S"), (1.6, "Z")):
        g[cm & ~dans(cuisse, X - nx * k, Y - ny * k)] = ton
    barres = cm & ((np.abs(le_long - 3.4) < 0.75) | (np.abs(le_long - 7.6) < 0.7)) & (n3 > 0.3)
    g[barres] = np.where(g[barres] == "S", "Z", "M")
    g[ellipse(37.0, 23.8, 1.8, 0.45, X, Y, -0.585)] = "I"  # reflet de la peau humide
    bord = np.zeros_like(cm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        bord |= cm & ~np.roll(np.roll(cm, dy, 0), dx, 1) & np.roll(np.roll(avant, dy, 0), dx, 1)
    g[bord] = "Z"

    # --- pied : tarse allongé posé au sol du talon vers l'avant, puis orteils articulés
    # en éventail ; les plus lointains, plus sombres, dépassent au-dessus des plus proches
    tarse = catmull(
        [
            (45.8, 37.2),
            (43.0, 37.9),
            (39.5, 38.4),
            (37.0, 38.6),
            (36.8, 39.3),
            (40, 39.6),
            (45.6, 39.6),
        ]
    )
    tm = dans(tarse, X, Y)
    g[tm] = "O"
    g[tm & (n3 < 0.12)] = "M"
    g[tm & ~dans(tarse, X, Y - 0.6)] = "H"
    g[tm & ~dans(tarse, X, Y + 0.6)] = "S"
    g[tm & ~dans(tarse, X, Y + 1.2) & (X < 40)] = "f"  # plante, claire vers les orteils
    doigt(g, [(75, 76), (71, 74.5), (67, 73.5)], 0.5, "S", (0.8, 0.6))
    doigt(g, [(74, 77), (69, 76.5), (65, 76), (62, 76)], 0.5, "F", (0.8, 0.6))
    doigt(g, [(75, 78.5), (70, 78.5), (64, 78.5), (58, 78.5)], 1.0, "f", (0.9, 0.7))

    # --- jambe (tibia) : du genou, sous l'avant de la cuisse, jusqu'au talon. Mollet bombé
    # près du genou, cheville fine, talon en bosse. Elle passe par-dessus la cuisse
    jambe = catmull(
        [
            (27.8, 31.2),
            (30.5, 30.7),
            (34.0, 31.2),
            (38.0, 32.6),
            (42.0, 34.4),
            (44.4, 35.2),
            (45.8, 35.6),
            (46.4, 36.8),
            (45.8, 37.9),
            (44.4, 38.1),
            (42.4, 37.2),
            (39.0, 36.4),
            (35.0, 35.2),
            (31.5, 34.5),
            (28.6, 34.1),
            (27.3, 32.8),
        ]
    )
    jm = dans(jambe, X, Y)
    sous_jambe = g != "."
    g[jm] = "O"
    g[jm & (n3 < 0.12)] = "M"
    g[jm & ~dans(jambe, X + 0.3, Y - 1.3)] = "H"
    g[jm & ~dans(jambe, X + 0.15, Y - 0.6)] = "I"
    g[jm & ~dans(jambe, X, Y + 1.4)] = "S"
    g[jm & ~dans(jambe, X, Y + 0.6)] = "Z"
    # barre sombre en travers du tibia, comme sur la cuisse
    g[jm & (np.abs((X - 36.5) + (Y - 33.5) * 0.4) < 0.7) & (n3 > 0.3) & (g == "O")] = "M"
    g[ellipse(29.6, 31.9, 1.1, 0.5, X, Y, 0.1)] = "I"  # reflet au genou
    g[ellipse(45.4, 36.2, 0.5, 0.4, X, Y)] = "H"  # talon
    pli = np.zeros_like(jm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin_hors = ~np.roll(np.roll(jm, dy, 0), dx, 1)
        voisin_dessous = np.roll(np.roll(sous_jambe, dy, 0), dx, 1)
        pli |= jm & voisin_hors & voisin_dessous
    g[pli] = "Z"

    # --- patte avant : sort du flanc sous la bande, coude en arrière, avant-bras plus
    # épais près du coude, poignet fin
    patte = catmull(
        [
            (18.4, 17.6),
            (23.6, 18.2),
            (25.0, 22.6),
            (25.2, 26.6),
            (24.4, 29.4),
            (22.6, 32.2),
            (21.0, 35.0),
            (20.6, 36.4),
            (18.6, 36.4),
            (18.7, 34.6),
            (19.6, 31.0),
            (20.6, 27.6),
            (19.6, 22.6),
        ]
    )
    pa = dans(patte, X, Y)
    # L'épaule se fond dans le flanc : de 19,2 à 22,4 unités de haut, la part de pixels du
    # bras croît en tramage ordonné (Bayer 4 × 4) ; au-dessus, c'est le flanc. Aucun trait
    # dans le corps : le contour n'apparaît que là où le bras sort du corps.
    corps_dessous = g != "."
    t = np.clip((Y - 19.2) / 3.2, 0, 1)
    trame = (BAYER4[(np.arange(H) % 4)[:, None], (np.arange(W) % 4)[None, :]] + 0.5) / 16
    bras = pa & ((trame < t) | ~corps_dessous)
    g[bras] = "O"
    g[bras & (n3 < 0.1)] = "M"
    g[bras & (n3 > 0.93)] = "f"
    # lumière d'en haut à gauche : bord avant éclairé, bord arrière dans l'ombre
    g[bras & ~dans(patte, X - 1.1, Y)] = "H"
    g[bras & ~dans(patte, X - 0.6, Y)] = "I"
    g[bras & ~dans(patte, X + 1.1, Y) & (Y > 21.0)] = "S"
    g[bras & ~dans(patte, X + 0.6, Y) & (Y > 22.0)] = "Z"
    g[ellipse(22.6, 26.6, 0.5, 0.9, X, Y, 0.3)] = "I"  # reflet au coude
    # main : paume sur le sol, tubercule sous le poignet, doigts articulés à disques
    doigt(g, [(37, 73), (33, 71.5), (30, 70.5)], 0.5, "S", (0.8, 0.6))
    doigt(g, [(36, 75), (32, 74.5), (28, 74.5)], 0.5, "F", (0.8, 0.6))
    doigt(g, [(37, 77), (34, 78.5), (30, 78.5)], 1.0, "f", (0.9, 0.7))
    doigt(g, [(41, 76.5), (44, 78), (47, 78.5)], 1.0, "f", (0.8, 0.7))
    main = catmull(
        [(18.4, 35.8), (20.8, 35.8), (21.4, 37.0), (20.6, 38.0), (18.6, 38.0), (17.8, 37.0)]
    )
    mm = dans(main, X, Y)
    g[mm] = "f"
    g[mm & ~dans(main, X, Y - 0.6)] = "H"
    g[mm & ~dans(main, X, Y + 0.6)] = "q"
    g[ellipse(20.0, 37.6, 0.35, 0.3, X, Y)] = "T"  # tubercule de la paume

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
    autour = np.zeros_like(plein)
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
    g[~plein & autour] = "K"
    return ["".join(r) for r in g]


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


# matière : (clarté de chaque ton de base, une rampe par variante de teinte, rouge → dorée)
MATIERES = {
    "orange": (
        {"I": 1.0, "H": 0.78, "O": 0.55, "M": 0.34, "S": 0.32, "Z": 0.12},
        [rampe(12, 17, 34, 0.66, 0.86, 0.22, 0.78, dh) for dh in (-4, 0, 5)],
    ),
    "bande": (
        {"B": 0.25, "b": 0.65},
        [rampe(7, 18, 24, 0.62, 0.55, 0.11, 0.33, dh) for dh in (-3, 0, 4)],
    ),
    "flanc": (
        {"F": 0.45, "f": 0.78},
        [rampe(8, 22, 32, 0.45, 0.6, 0.42, 0.72, dh) for dh in (-4, 0, 6)],
    ),
    "creme": (
        {"C": 0.85, "g": 0.62, "c": 0.35, "q": 0.5},
        [rampe(8, 30, 40, 0.14, 0.38, 0.52, 0.9, dh) for dh in (-6, 0, 6)],
    ),
}
# codes des nouvelles teintes : caractères libres, ni chiffre, ni « . », ni guillemet
_LIBRES = [
    chr(c)
    for c in list(range(0x21, 0x7F)) + list(range(0xC0, 0x17F))
    if not chr(c).isdigit() and chr(c) not in '."\\' and chr(c) not in PALETTE
]
PALETTE_NUANCES = dict(PALETTE)
CODES = {}
for _nom, (_, _rampes) in MATIERES.items():
    for _v, _r in enumerate(_rampes):
        for _i, _hexa in enumerate(_r):
            CODES[_nom, _v, _i] = _LIBRES.pop(0)
            PALETTE_NUANCES[CODES[_nom, _v, _i]] = _hexa

_K = np.array([1, 4, 6, 4, 1], float) / 16
_YY, _XX = np.mgrid[0:H, 0:W]
_LUMIERE = 0.07 * (0.5 - (_XX / W * 0.6 + _YY / H * 0.4))  # un peu plus clair en haut à gauche
_v = np.sin(_XX * 12.9898 + _YY * 78.233) * 43758.5453
_GRAIN = (_v - np.floor(_v) - 0.5) * 0.05
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
    """Rangées en tons de base → rangées en teintes fines (voir MATIERES)."""
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
        final = 0.5 + (0.45 * clarte + 0.55 * lisse - 0.5) * 1.22 + _LUMIERE + _GRAIN
        n = len(rampes[0])
        idx = np.clip(np.rint(final * (n - 1)), 0, n - 1).astype(int)
        for v in range(len(rampes)):
            for i in range(n):
                out[m & (idx == i) & (VARIANTE == v)] = CODES[nom, v, i]
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
                    rangs = nuancer(image(g_ * 0.9, s * 0.7, bool(c), t))
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
