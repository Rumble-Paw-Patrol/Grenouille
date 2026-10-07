"""Sprite 8-bit d'A. blanci pour le bandeau du tableau de bord.

    uv run python documentation/tableau-de-bord/grenouille.py

Grenouille de profil, tournée vers le titre (à gauche), dessinée d'après les photos d'A. blanci
de Benoît Villette et d'Arnaud Aury : museau court, grand œil cerclé de doré, bande sombre du
museau au flanc, longue patte avant, cuisse repliée. Les formes sont tracées sur une grille de
48 × 40, sans anticrénelage, puis cernées d'un contour. Une image par combinaison
gorge (gonflée ou non) × souffle (flanc) × clignement : la page les enchaîne pour que la
grenouille respire et cligne. Écrit grenouille.json (palette et images), lu par construire.py.
"""

import json
import math
from pathlib import Path

ICI = Path(__file__).resolve().parent
PALETTE = {
    "K": "#24120A",
    "O": "#D07A35",
    "H": "#EDA35C",
    "S": "#A3561F",
    "L": "#6B3414",
    "B": "#5A2C14",
    "b": "#7A3E1C",
    "F": "#C2804A",
    "C": "#E6D8BE",
    "c": "#BFB09A",
    "E": "#0B0705",
    "G": "#A08C5C",
    "W": "#FFFFFF",
    "T": "#F0C48A",
}

W, H = 48, 40


def catmull(pts, n=12, ferme=True):
    out = []
    m = len(pts)
    rng = range(m) if ferme else range(m - 1)
    for i in rng:
        p0, p1, p2, p3 = pts[(i - 1) % m], pts[i], pts[(i + 1) % m], pts[(i + 2) % m]
        if not ferme:
            p0 = pts[max(i - 1, 0)]
            p3 = pts[min(i + 2, m - 1)]
        for k in range(n):
            t = k / n
            out.append(
                tuple(
                    0.5
                    * (
                        (2 * p1[j])
                        + (-p0[j] + p2[j]) * t
                        + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t * t
                        + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t**3
                    )
                    for j in (0, 1)
                )
            )
    return out


def dedans(poly, x, y):
    c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def remplir(g, poly, coul, test=None):
    for y in range(H):
        for x in range(W):
            if dedans(poly, x + 0.5, y + 0.5) and (test is None or test(x + 0.5, y + 0.5)):
                g[y][x] = coul


def ellipse_poly(cx, cy, rx, ry, rot=0, n=48):
    return [
        (
            cx + rx * math.cos(a) * math.cos(rot) - ry * math.sin(a) * math.sin(rot),
            cy + rx * math.cos(a) * math.sin(rot) + ry * math.sin(a) * math.cos(rot),
        )
        for a in (2 * math.pi * k / n for k in range(n))
    ]


def interp(pts, x):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:], strict=False):
        if x1 <= x <= x2:
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return pts[0][1] if x < pts[0][0] else pts[-1][1]


def image(gorge=0.0, souffle=0.0, cligne=False):
    g = [["." for _ in range(W)] for _ in range(H)]
    # corps et tête
    corps = catmull(
        [
            (1.2, 5.6),
            (2.6, 3.8),
            (5, 2.4),
            (7.6, 1.3),
            (9.6, 1.4),
            (11.6, 2.6),
            (14, 4.2),
            (17, 5.6),
            (21, 7.1),
            (25.5, 8.6),
            (30, 10.6),
            (33.5, 12.8),
            (36, 15),
            (37, 22),
            (33, 26),
            (30.5, 27.8 + souffle),
            (27, 27.6 + souffle),
            (24.5, 26),
            (22, 22),
            (19, 19.8),
            (16.5, 17.6),
            (13.5, 15.8 + gorge * 0.4),
            (10, 14.4 + gorge),
            (6.5, 12.8 + gorge * 0.8),
            (3.4, 11 + gorge * 0.3),
            (1.6, 9.4),
            (0.9, 7.4),
        ]
    )
    bande = [
        (0, 7.2),
        (4, 7.6),
        (7.5, 8.2),
        (11, 8.8),
        (15, 9.8),
        (19, 11),
        (24, 12.6),
        (29, 14.2),
        (34, 16),
    ]

    def epaisseur(x):
        return 0.6 if x < 7.5 else 1.2 if x < 18 else 1.5

    def zone(x, y):
        yb = interp(bande, x)
        e = epaisseur(x)
        if y < yb - e:
            return "O"
        if y <= yb + e:
            return "B" if x > 2.5 else "O"
        if x < 16:
            if y <= yb + e + 1.3:
                return "T" if x < 9 else "C"
            return "O" if y <= yb + e + 2.6 and x < 13 else "C"
        return "F"

    for y in range(H):
        for x in range(W):
            if dedans(corps, x + 0.5, y + 0.5):
                g[y][x] = zone(x + 0.5, y + 0.5)
    # reflet du dos : une rangée sous le contour supérieur
    for x in range(2, 35):
        for y in range(H):
            if g[y][x] != ".":
                if g[y][x] == "O" and g[y + 1][x] == "O":
                    g[y + 1][x] = "H"
                break
    # bas du ventre clair, puis ombré
    for x in range(16, W):
        bas = [y for y in range(H) if g[y][x] in "CF"]
        if bas:
            yb = bas[-1]
            g[yb][x] = "c"
            for k in (1,):
                if yb - k in bas:
                    g[yb - k][x] = "C"
    # jambe et pied arrière, cuisse par-dessus : la jambe descend du genou au talon, le pied
    # revient vers l'avant le long du sol (patte repliée en Z)
    remplir(
        g,
        catmull(
            [
                (31.5, 27),
                (35, 28),
                (41, 31.5),
                (45.6, 34.6),
                (45.2, 37.4),
                (42.6, 37.2),
                (38, 33.6),
                (32.5, 30),
            ]
        ),
        "S",
    )
    remplir(
        g,
        catmull(
            [
                (45.2, 36.2),
                (45.6, 38.2),
                (40, 38.4),
                (34, 38.4),
                (30.4, 38.5),
                (30.6, 37.2),
                (34, 37.2),
                (40, 36.8),
            ]
        ),
        "S",
    )
    for x, y in [(30, 38), (32, 38), (34, 38), (36, 38)]:
        g[y][x] = "T"
    cuisse = ellipse_poly(39.2, 22.2, 7.9, 7.2, -0.15)
    avant = [r[:] for r in g]
    for y in range(H):
        for x in range(W):
            if dedans(cuisse, x + 0.5, y + 0.5):
                d = (x + 0.5 - 37) * 0.5 + (y + 0.5 - 19.5) * 0.85
                g[y][x] = "S" if d > 7.2 else "H" if d < -2.6 else "O"
                # jambe repliée sous la cuisse : un pli brun, la jambe en dessous
                pli = 26.2 + (x + 0.5 - 33) * 0.12
                if y + 0.5 > pli + 1:
                    g[y][x] = "S"
                elif y + 0.5 > pli:
                    g[y][x] = "L"
    for y in range(H):
        for x in range(W):
            if dedans(cuisse, x + 0.5, y + 0.5) and any(
                not dedans(cuisse, x + dx + 0.5, y + dy + 0.5) and avant[y + dy][x + dx] not in ".S"
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= y + dy < H and 0 <= x + dx < W
            ):
                g[y][x] = "L"
    # mouchetures du flanc
    for x, y in [
        (20, 15),
        (23, 17),
        (26, 16),
        (28, 19),
        (24, 20),
        (31, 18),
        (21, 13),
        (27, 13),
        (17, 8),
        (22, 9),
    ]:
        if g[y][x] == "F":
            g[y][x] = "C"
        elif g[y][x] in "OH":
            g[y][x] = "S"
    # patte avant : longue, coude en arrière, main aux doigts écartés
    patte = catmull(
        [
            (18.6, 19.6),
            (23.6, 19.6),
            (25, 23),
            (24.8, 26.8),
            (23.4, 29.5),
            (21.4, 32.8),
            (20.4, 35.8),
            (17.6, 35.8),
            (18.2, 32),
            (19.6, 28.6),
            (20.4, 26),
            (19.4, 22.6),
        ]
    )
    for y in range(H):
        xs = [x for x in range(W) if dedans(patte, x + 0.5, y + 0.5)]
        for x in xs:
            g[y][x] = "O"
        if len(xs) >= 2:
            g[y][xs[0]] = "H"
            g[y][xs[-1]] = "L" if y < 27 else "S"
    remplir(
        g,
        catmull([(13.4, 37.4), (16, 36.2), (20.6, 35.6), (21.2, 37.4), (19, 38.4), (13.6, 38.6)]),
        "O",
    )
    for x, y in [(12, 38), (14, 39), (16, 39), (19, 39), (21, 38)]:
        g[y][x] = "T"
    # œil : grand, cerclé de doré, reflet
    oeil = ellipse_poly(11, 8.6, 3.75, 3.6)
    iris = ellipse_poly(11, 8.6, 2.85, 2.75)
    for y in range(H):
        for x in range(W):
            if dedans(oeil, x + 0.5, y + 0.5):
                g[y][x] = "E" if dedans(iris, x + 0.5, y + 0.5) else "G"
    if cligne:
        for y in range(H):
            for x in range(W):
                if dedans(oeil, x + 0.5, y + 0.5):
                    g[y][x] = "O" if y < 8 else "B" if y == 8 else "b"
    else:
        g[7][9] = "W"
        g[7][10] = "W"
        g[8][9] = "W"
    # arcade claire au-dessus de l'œil, narine
    for x in range(8, 14):
        if g[4][x] in "OH":
            g[4][x] = "H"
    g[6][2] = "B"
    # contour
    plein = [[c != "." for c in r] for r in g]
    for y in range(H):
        for x in range(W):
            if not plein[y][x] and any(
                0 <= y + dy < H and 0 <= x + dx < W and plein[y + dy][x + dx]
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                g[y][x] = "K"
    return ["".join(r) for r in g]


if __name__ == "__main__":
    images = {
        f"{g}{s}{c}": image(g * 0.9, s * 0.7, bool(c))
        for g in (0, 1)
        for s in (0, 1)
        for c in (0, 1)
    }
    (ICI / "grenouille.json").write_text(
        json.dumps({"palette": PALETTE, "images": images}, indent=0), encoding="utf-8"
    )
    print("\n".join(images["000"]))
