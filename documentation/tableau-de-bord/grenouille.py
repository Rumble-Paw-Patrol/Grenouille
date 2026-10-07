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

    # --- pied arrière, à plat vers l'avant (la jambe est tracée plus bas, sur la cuisse)
    pied = catmull([(46.4, 37.6), (46.6, 39.3), (40, 39.4), (35, 39.3), (35, 38.4), (40, 38.2)])
    pm = dans(pied, X, Y)
    g[pm] = "F"
    g[pm & ~dans(pied, X, Y - 0.6)] = "f"
    g[pm & ~dans(pied, X, Y + 0.6)] = "q"
    for bout in [(29.6, 39.0), (31.2, 39.5), (33.0, 39.6)]:
        g[segment((35.6, 38.9), bout, 0.6, X, Y)] = "f"
        g[ellipse(*bout, 0.45, 0.45, X, Y)] = "T"

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

    # --- cuisse : orange marbré, deux barres sombres, peau brillante, pli au bord du corps
    cuisse = ellipse(38.4, 27.6, 7.7, 7.0, X, Y, -0.22)
    d = (X - 36.5) * 0.5 + (Y - 25) * 0.85
    avant = g != "."
    g[cuisse] = "O"
    g[cuisse & (d < -2.4)] = "H"
    g[cuisse & (d < -4.6)] = "I"
    g[cuisse & (d > 4.6)] = "S"
    g[cuisse & (d > 6.8)] = "Z"
    axe = (X - 38.4) * 0.8 - (Y - 27.6) * 0.6  # le long de la cuisse, du genou à la hanche
    barres = cuisse & ((np.abs(axe + 2.2) < 0.7) | (np.abs(axe - 2.6) < 0.7)) & (n3 > 0.35)
    g[barres] = np.where(d[barres] > 4.6, "Z", "M")
    g[cuisse & (n3 < 0.08) & (d < 4.6)] = "M"
    g[cuisse & (n3 > 0.95) & (d < 2)] = "H"
    g[ellipse(34.8, 23.4, 0.9, 0.5, X, Y, -0.5)] = "I"  # reflet de la peau humide
    bord = np.zeros_like(cuisse)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin_hors = ~np.roll(np.roll(cuisse, dy, 0), dx, 1)
        voisin_corps = np.roll(np.roll(avant, dy, 0), dx, 1)
        bord |= cuisse & voisin_hors & voisin_corps
    g[bord] = "Z"

    # --- jambe : du genou, à l'avant du bas de la cuisse, jusqu'au talon. Elle passe
    # par-dessus la cuisse (et devant le ventre), soulignée d'un pli sombre là où elle la
    # recouvre ; bord du haut éclairé, dessous dans l'ombre.
    jambe = catmull(
        [
            (30.2, 30.8),
            (33, 30.9),
            (38.5, 32.4),
            (44, 34.4),
            (46.6, 35.8),
            (46.6, 38.4),
            (43, 38.9),
            (37.5, 37.4),
            (31.4, 34.6),
            (29.8, 32.6),
        ]
    )
    jm = dans(jambe, X, Y)
    sous_jambe = g != "."
    g[jm] = "O"
    g[jm & (n3 < 0.12)] = "M"
    g[jm & ~dans(jambe, X, Y - 1.2)] = "H"
    g[jm & ~dans(jambe, X, Y + 1.2)] = "S"
    g[jm & ~dans(jambe, X, Y + 0.6)] = "Z"
    g[ellipse(33.2, 32.0, 1.0, 0.45, X, Y, 0.25)] = "I"  # reflet au genou
    pli = np.zeros_like(jm)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        voisin_hors = ~np.roll(np.roll(jm, dy, 0), dx, 1)
        voisin_dessous = np.roll(np.roll(sous_jambe, dy, 0), dx, 1)
        pli |= jm & voisin_hors & voisin_dessous & (Y < 37.6)
    g[pli] = "Z"

    # --- patte avant : sort du flanc sous la bande, coude en arrière, longue main fine
    patte = catmull(
        [
            (18.4, 17.6),
            (23.6, 18.2),
            (25.0, 22.6),
            (25.0, 26.8),
            (23.6, 29.8),
            (21.8, 33.0),
            (20.8, 36.2),
            (17.8, 36.2),
            (18.4, 32.4),
            (19.8, 29.0),
            (20.6, 26.2),
            (19.6, 22.6),
        ]
    )
    pa = dans(patte, X, Y)
    epaule = pa & (Y < 20.6)  # l'épaule se fond dans le flanc : seulement une ombre portée
    bras = pa & ~epaule
    g[epaule & ~dans(patte, X - 0.6, Y)] = "S"
    g[epaule & ~dans(patte, X - 0.6, Y) & (Y > 19.4)] = "Z"
    g[bras] = "O"
    g[bras & (n3 < 0.1)] = "M"
    g[bras & (n3 > 0.93)] = "f"
    g[bras & ~dans(patte, X + 1.1, Y)] = "H"
    g[bras & ~dans(patte, X + 0.6, Y)] = "I"
    g[bras & ~dans(patte, X - 1.1, Y)] = "S"
    g[bras & ~dans(patte, X - 0.6, Y)] = "Z"
    g[bras & (Y < 21.2) & dans(patte, X - 0.6, Y) & dans(patte, X + 0.6, Y)] = "F"
    g[ellipse(22.6, 26.6, 0.5, 0.9, X, Y, 0.3)] = "I"  # reflet au coude
    main = catmull([(16.6, 36.0), (20.8, 35.8), (21.4, 37.4), (19.6, 38.2), (16.4, 38.0)])
    mm = dans(main, X, Y)
    g[mm] = "f"
    g[mm & ~dans(main, X, Y + 0.6)] = "q"
    g[mm & ~dans(main, X, Y - 0.6)] = "H"
    for base, bout in [
        ((17.4, 37.4), (13.0, 38.9)),
        ((18.2, 37.7), (15.4, 39.5)),
        ((19.3, 37.8), (18.4, 39.6)),
        ((20.6, 37.6), (21.9, 39.3)),
    ]:
        g[segment(base, bout, 0.6, X, Y)] = "f"
        g[ellipse(*bout, 0.45, 0.45, X, Y)] = "T"

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
