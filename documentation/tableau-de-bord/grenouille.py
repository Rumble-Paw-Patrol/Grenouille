"""Sprite 8-bit d'A. blanci pour le bandeau du tableau de bord.

    uv run python documentation/tableau-de-bord/grenouille.py

Grenouille de profil, tournée vers le titre (à gauche), dessinée d'après les photos d'A. blanci
de Benoît Villette et d'Arnaud Aury : museau court, grand œil noir cerclé de doré, bande sombre
du museau au flanc, gorge claire, longue patte avant aux doigts fins, cuisse repliée et
arrière-train posé presque au sol. Les formes sont décrites dans un repère de 48 × 40 unités
et tracées sur une grille de 96 × 80 pixels (deux pixels par unité), sans anticrénelage, puis
cernées d'un contour.

Une image par combinaison gorge (gonflée ou non) × flanc (gonflé ou non) × clignement ×
tête (baissée, droite, relevée) : la page les enchaîne pour que la grenouille respire, cligne
et bouge un peu la tête. Écrit grenouille.json (palette, taille et images codées par
plages : « O12 » = douze pixels de la couleur O), lu par construire.py.
"""

import json
import math
import random
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
PALETTE = {
    "K": "#24120A",  # contour
    "I": "#F6BF7E",  # reflet vif
    "H": "#E89F55",  # reflet
    "O": "#D07A35",  # dos orangé
    "S": "#A8591F",  # ombre
    "Z": "#85431A",  # ombre profonde
    "M": "#A65A27",  # mouchetures du dos
    "L": "#6B3414",  # trait entre deux membres
    "B": "#4E2610",  # bande sombre
    "b": "#6E3818",  # bord de la bande
    "F": "#C2804A",  # flanc
    "f": "#D6A272",  # flanc clair
    "T": "#F0C48A",  # lèvre, bouts des doigts
    "C": "#E8DCC4",  # gorge, ventre
    "g": "#CFC4AE",  # gorge grisée
    "c": "#B2A58E",  # ventre ombré
    "E": "#0B0705",  # œil
    "G": "#C29A55",  # cercle doré de l'œil, en haut
    "A": "#7A6A4A",  # cercle de l'œil, en bas
    "W": "#FFFFFF",  # reflet de l'œil
    "w": "#9FB4C0",  # second reflet, bleuté
}
UNITES = (48, 40)
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


def suivre(pts, x):
    xs, ys = zip(*pts, strict=True)
    return np.interp(x, xs, ys)


def image(gorge=0.0, souffle=0.0, cligne=False, tete=0):
    g = np.full((H, W), ".", dtype="<U1")
    # Tête baissée ou relevée : rotation de quelques degrés autour du cou, qui s'estompe
    # vers le corps et ne touche pas les pattes. On échantillonne à l'envers.
    poids = np.clip((21 - X) / 8, 0, 1) * np.clip((19.5 - Y) / 3.5, 0, 1)
    a = -tete * 0.06 * poids
    px, py = 18.0, 9.5
    x = px + (X - px) * np.cos(a) - (Y - py) * np.sin(a)
    y = py + (X - px) * np.sin(a) + (Y - py) * np.cos(a)
    alea = random.Random(7)

    # jambe arrière très pliée : du genou au talon, puis le pied à plat vers l'avant
    jambe = catmull(
        [
            (30.6, 31.6),
            (35, 33.2),
            (41, 35.4),
            (46.3, 36.6),
            (46.3, 38.6),
            (43, 38.9),
            (37.5, 37.2),
            (31.2, 34.4),
        ]
    )
    g[dans(jambe, X, Y)] = "S"
    g[dans(jambe, X, Y - 0.6) & ~dans(jambe, X, Y)] = "S"
    pied = catmull([(46.4, 37.6), (46.6, 39.3), (40, 39.4), (35, 39.3), (35, 38.4), (40, 38.2)])
    g[dans(pied, X, Y)] = "Z"
    for bout in [(29.6, 39.0), (31.2, 39.5), (33.0, 39.6)]:
        g[segment((35.6, 38.9), bout, 0.6, X, Y)] = "S"
        g[ellipse(*bout, 0.45, 0.45, X, Y)] = "T"

    # corps et tête : tête haute, dos en pente jusqu'à l'arrière-train, presque au sol
    corps = catmull(
        [
            (1.2, 5.6),
            (2.6, 3.8),
            (5, 2.4),
            (7.6, 1.3),
            (9.6, 1.4),
            (11.6, 2.6),
            (14, 4.2),
            (17, 5.8),
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
    yb = suivre(bande, x)
    e = np.where(x < 7.5, 0.6, np.where(x < 18, 1.0, 1.25))
    dos = dedans & (y < yb - e)
    g[dos] = "O"
    g[dos & (y > yb - e - 0.45) & (x > 13)] = "S"  # ombre au-dessus de la bande
    haut1, haut2 = ~dans(corps, x, y - 0.6), ~dans(corps, x, y - 1.3)
    g[dos & haut2] = "H"
    g[dos & haut1] = "I"
    sur_bande = dedans & (np.abs(y - yb) <= e)
    g[sur_bande] = np.where(x[sur_bande] < 2.6, "O", "B")
    g[sur_bande & (np.abs(y - yb) > e - 0.5) & (x > 14)] = "b"
    dessous = dedans & (y > yb + e)
    tete_ = dessous & (x < 16)
    g[tete_] = "g"
    joue = tete_ & (y <= yb + e + 2.3 - np.clip(x - 10, 0, None) * 0.55)
    g[joue] = "O"
    g[tete_ & (y <= yb + e + 0.8)] = "T"
    g[tete_ & (y <= yb + e + 0.8) & (x > 9)] = "C"
    flanc = dessous & (x >= 16)
    g[flanc] = "F"
    g[flanc & (y < yb + e + 1.4)] = "f"
    bas1, bas2 = ~dans(corps, x, y + 0.6), ~dans(corps, x, y + 1.6)
    g[dessous & bas2] = "C"
    g[dessous & bas1] = "c"
    g[tete_ & bas1] = "g"
    # mouchetures : brunes sur le dos, claires sur le flanc
    for _ in range(46):
        mx, my = alea.uniform(12, 38), alea.uniform(4, 24)
        tache = ellipse(mx, my, 0.4, 0.4, x, y)
        g[tache & dos & ~haut1] = "M"
        g[tache & flanc & ~bas2] = "f"
    for _ in range(16):
        mx, my = alea.uniform(17, 33), alea.uniform(13, 29)
        g[ellipse(mx, my, 0.4, 0.4, x, y) & flanc & ~bas2] = "C"

    # cuisse, par-dessus le corps, cernée de brun là où elle le touche
    cuisse = ellipse(38.4, 27.6, 7.7, 7.0, X, Y, -0.22)
    d = (X - 36.5) * 0.5 + (Y - 25) * 0.85
    avant = g != "."
    g[cuisse] = "O"
    g[cuisse & (d < -2.4)] = "H"
    g[cuisse & (d < -4.6)] = "I"
    g[cuisse & (d > 5.4)] = "S"
    g[cuisse & (d > 7.4)] = "Z"
    for _ in range(9):
        mx, my = alea.uniform(33, 45), alea.uniform(22, 32)
        g[ellipse(mx, my, 0.4, 0.4, X, Y) & cuisse & (d > -2.4) & (d < 5.4)] = "M"
    bord = np.zeros_like(cuisse)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin_hors = ~np.roll(np.roll(cuisse, dy, 0), dx, 1)
        voisin_corps = np.roll(np.roll(avant, dy, 0), dx, 1)
        bord |= cuisse & voisin_hors & voisin_corps
    g[bord] = "L"

    # patte avant : longue, coude en arrière, main aux doigts fins
    patte = catmull(
        [
            (18.6, 19.6),
            (23.6, 19.8),
            (25, 23.2),
            (24.8, 27.2),
            (23.4, 30),
            (21.6, 33.2),
            (20.6, 36.2),
            (17.8, 36.2),
            (18.3, 32.4),
            (19.6, 29),
            (20.4, 26.3),
            (19.4, 22.8),
        ]
    )
    pm = dans(patte, X, Y)
    g[pm] = "O"
    g[pm & ~dans(patte, X + 0.6, Y)] = "H"
    g[pm & ~dans(patte, X - 0.6, Y)] = "S"
    g[pm & ~dans(patte, X - 0.6, Y) & (Y < 27)] = "L"
    main = catmull([(16.6, 36.0), (20.8, 35.8), (21.4, 37.4), (19.6, 38.2), (16.4, 38.0)])
    g[dans(main, X, Y)] = "O"
    for base, bout in [
        ((17.4, 37.4), (13.0, 38.9)),
        ((18.2, 37.7), (15.4, 39.5)),
        ((19.3, 37.8), (18.4, 39.6)),
        ((20.6, 37.6), (21.9, 39.3)),
    ]:
        g[segment(base, bout, 0.6, X, Y)] = "O"
        g[ellipse(*bout, 0.45, 0.45, X, Y)] = "T"

    # œil : grand, noir, cerclé de doré en haut et de gris en bas, deux reflets
    oeil = ellipse(11, 8.6, 3.75, 3.6, x, y)
    iris = ellipse(11, 8.6, 2.85, 2.75, x, y)
    arcade = ellipse(11, 8.6, 4.5, 4.35, x, y) & ~oeil & (y < 6.4) & (x > 8.2) & (x < 13.8)
    g[arcade & dedans] = "H"
    g[oeil] = "A"
    g[oeil & (y < 8.6)] = "G"
    g[iris] = "E"
    if cligne:
        # paupière fermée dans le cercle de l'œil : haut orangé, fente sombre, bas pâle
        g[iris] = "S"
        g[iris & (y < 8.4)] = "O"
        g[iris & (y < 6.8)] = "H"
        g[iris & (np.abs(y - 8.9) < 0.5)] = "B"
        g[iris & (y > 9.4)] = "f"
    else:
        g[ellipse(9.9, 7.4, 0.8, 0.75, x, y)] = "W"
        g[ellipse(12.2, 10.0, 0.4, 0.4, x, y)] = "w"
    # narine et bout du museau
    g[ellipse(2.1, 5.9, 0.4, 0.4, x, y)] = "B"
    g[ellipse(2.6, 4.2, 0.6, 0.5, x, y) & dedans] = "I"

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
                    rangs = image(g_ * 0.9, s * 0.7, bool(c), t)
                    images[f"{g_}{s}{c}{t + 1}"] = [plages(r) for r in rangs]
    (ICI / "grenouille.json").write_text(
        json.dumps(
            {"palette": PALETTE, "largeur": W, "hauteur": H, "images": images},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print("\n".join(image()))
