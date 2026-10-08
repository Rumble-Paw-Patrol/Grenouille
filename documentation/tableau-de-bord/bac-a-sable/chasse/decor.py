"""Le décor de la scène de chasse : sous-bois guyanais, au pixel, en trois calques.

    python3 documentation/tableau-de-bord/bac-a-sable/chasse/decor.py [apercu.png]

Le fond (opaque) superpose, de loin en près : la canopée et ses trouées de ciel, la brume
verte et ses troncs lointains, deux troncs à mi-distance, un sous-étage de palmes, la litière
au loin et, à gauche, un grand tronc à contrefort couvert de mousse. Tout le vert vient d'une
seule rampe : plus un plan est loin, moins il s'écarte de la brume (perspective aérienne). Le
sol (transparent hors des feuilles) est la grande feuille morte où la grenouille est assise,
avec son ombre, et la litière proche : la page s'en sert aussi pour éclairer ces feuilles à la
lueur des lucioles. L'avant-plan est flou : une fronde de fougère en haut à droite, le bord
d'une feuille en bas à gauche. Les dégradés sont tramés (Bayer 4 × 4) sur des rampes courtes,
comme la grenouille. Le hasard est tiré d'une graine fixe : le décor est le même à chaque fois.
"""

import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ICI = Path(__file__).resolve().parent
W, H = 256, 144
FX, FY = 136, 46                     # coin haut gauche de la grenouille (114 × 86)
SOL = FY + 84                        # rangée de ses pieds
X, Y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
_B = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], float)
BAYER = (_B[Y.astype(int) % 4, X.astype(int) % 4] + 0.5) / 16
HASARD = np.random.default_rng(20261008)


def hexa(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def rampe(v, couleurs, seuil=None):
    """Valeurs 0..1 → couleurs de la rampe, tramées entre deux crans voisins."""
    pal = np.array([hexa(c) for c in couleurs])
    n = len(pal)
    b = BAYER if seuil is None else seuil
    i = np.clip(np.floor(np.clip(v, 0, 1) * (n - 1) + b), 0, n - 1).astype(int)
    return pal[i]


def lisse(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def bruit(echelle, graine, octaves=1):
    """Bruit de valeur lissé, 0..1, à l'échelle donnée (en pixels)."""
    rng = np.random.default_rng(graine)
    total, poids = np.zeros((H, W)), 0.0
    for o in range(octaves):
        e = echelle / 2**o
        g = rng.random((int(H / e) + 3, int(W / e) + 3))
        xs, ys = X / e, Y / e
        x0, y0 = xs.astype(int), ys.astype(int)
        fx, fy = xs - x0, ys - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a, b, c, d = g[y0, x0], g[y0, x0 + 1], g[y0 + 1, x0], g[y0 + 1, x0 + 1]
        total += ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy) * 0.5**o
        poids += 0.5**o
    return total / poids


def poly(pts):
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(float(a), float(b)) for a, b in pts], fill=255)
    return np.array(im) > 0


def courbe(pts, n=12):
    """Courbe ouverte lisse par les points (Catmull-Rom, extrémités doublées)."""
    p = [pts[0], *pts, pts[-1]]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = (np.array(q, float) for q in p[i - 1:i + 3])
        for k in range(n):
            t = k / n
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t**3))
    out.append(np.array(pts[-1], float))
    return np.array(out)


def trait(pts, epaisseur=1.0):
    """Masque d'un trait le long d'une polyligne (distance au segment le plus proche)."""
    d = np.full((H, W), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        l2 = max(float(ab @ ab), 1e-9)
        t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / l2, 0, 1)
        d = np.minimum(d, np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1]))
    return d <= epaisseur / 2, d


def flou(m, r):
    """Flou boîte séparable (r pixels) d'un champ 2D."""
    a = m.astype(float)
    for axe in (0, 1):
        acc = np.zeros_like(a)
        for k in range(-r, r + 1):
            acc += np.roll(a, k, axis=axe)
        a = acc / (2 * r + 1)
    return a


# ---------------------------------------------------------------- les rampes
BRUME = ["#13241A", "#182C1F", "#1E3524", "#253F2A", "#2D4A30", "#365537", "#40603D", "#4A6B43",
         "#55764A", "#618152", "#6F8C5B", "#7F9865", "#91A570", "#A6B47D"]
CIEL = ["#9FB574", "#BCCB8C", "#D8DFA8", "#EEEDC6"]
ECORCE = ["#0A120D", "#0F1913", "#142119", "#1A2A1F", "#223426", "#2C402F", "#384C37",
          "#46593F", "#566A4A", "#687C57"]
MOUSSE = ["#2A4020", "#37522A", "#476633", "#5A7A3B", "#728F45", "#8CA452"]
LITIERE = ["#1B150E", "#251B12", "#312316", "#3D2C1A", "#4A341E", "#573D22", "#654626",
           "#734F2A", "#80592F", "#8C6435"]


def ellipses(nb, centre, rayon, taille, graine):
    """Masque et valeur relative (-1 dessous .. 1 dessus) d'un amas de petites feuilles."""
    rng = np.random.default_rng(graine)
    m, w_ = np.zeros((H, W), bool), np.zeros((H, W))
    for _ in range(nb):
        cx, cy = centre[0] + rng.normal(0, rayon[0]), centre[1] + rng.normal(0, rayon[1])
        rx, ry, an = rng.uniform(*taille), rng.uniform(0.9, 1.6), rng.uniform(-1.0, 1.0)
        dx, dy = X - cx, Y - cy
        u = (dx * math.cos(an) + dy * math.sin(an)) / rx
        w = (-dx * math.sin(an) + dy * math.cos(an)) / ry
        e = u * u + w * w < 1
        m |= e
        w_ = np.where(e, -w, w_)
    return m, w_


def palme(base, portee, angles, graine, foliole=5.0):
    """Masque d'une palme vue de loin : rachis arqués et folioles pendantes."""
    m = np.zeros((H, W), bool)
    rng = np.random.default_rng(graine)
    for a, lg in angles:
        pts, u = [], 0.0
        for j in range(10):
            u = j / 9 * portee * lg
            pts.append((base[0] + math.cos(a) * u, base[1] - math.sin(a) * u + 0.018 * u * u))
        ligne = courbe(pts, 3)
        m |= trait(ligne, 1.1)[0]
        for j in range(2, len(ligne) - 1, 2):
            p = ligne[j]
            tg = ligne[j + 1] - ligne[j - 1]
            tg = tg / (np.linalg.norm(tg) + 1e-9)
            for sens in (-1, 1):
                nrm = np.array([-tg[1], tg[0]]) * sens
                lg_f = foliole * (0.6 + 0.4 * math.sin(math.pi * j / len(ligne))) * rng.uniform(0.8, 1.1)
                bout = p + (nrm * 0.55 + np.array([0, 1.0]) * 0.8 + tg * 0.3) * lg_f
                m |= trait(np.array([p, bout]), 1.0)[0]
    return m


def fond():
    n1, n2 = bruit(24, 1, 3), bruit(9, 2, 2)
    lum = np.exp(-(((X - 196) / 64) ** 2))
    # la brume : sombre sous la canopée, claire à mi-hauteur, plus sombre vers le sol ; un puits
    # de lumière en haut à droite, là où la canopée s'ouvre
    vb = 0.28 + 0.42 * lisse(4, 66, Y) - 0.24 * lisse(80, 120, Y)
    vb += 0.20 * np.exp(-(((X - 196) / 58) ** 2) - (((Y - 34) / 46) ** 2))
    vb += 0.08 * (n1 - 0.5)
    v = vb.copy()

    def poser(m, val):
        nonlocal v
        v = np.where(m, val, v)

    # troncs lointains, à peine plus sombres que la brume
    for x0, l, pente, d in [(30, 3, 0.02, 0.09), (49, 2, -0.01, 0.07), (88, 4, 0.015, 0.11),
                            (120, 2, -0.02, 0.07), (152, 3, 0.01, 0.09), (178, 5, -0.01, 0.12),
                            (197, 2, 0.02, 0.07), (244, 4, 0.0, 0.11)]:
        bord = x0 + pente * Y + 0.8 * np.sin(Y * 0.045 + x0)
        m = (X >= bord) & (X < bord + l) & (Y < 116)
        poser(m, vb - d + 0.04 * (X - bord > l - 1.5))
    # une palme lointaine à gauche, une autre derrière la croupe
    poser(palme((74, 116), 40, [(2.5, 1.0), (2.0, 0.9), (1.25, 0.75), (0.7, 0.95)], 3) & (Y < 117), vb - 0.13)
    poser(palme((236, 118), 30, [(2.3, 0.9), (1.6, 0.7), (0.9, 0.85)], 4) & (Y < 118), vb - 0.12)
    # deux troncs à mi-distance, plus sombres, liseré de lumière à droite, racines évasées
    for x0, l, pente in [(62, 6, 0.03), (208, 10, -0.025)]:
        bord = x0 + pente * Y + 1.1 * np.sin(Y * 0.035 + x0)
        evase = 5 * lisse(98, 118, Y) ** 2
        m = (X >= bord - evase) & (X < bord + l + evase) & (Y < 118)
        lum_t = (X - bord + evase) / (l + 2 * evase)
        poser(m, vb - 0.32 + 0.12 * lisse(0.6, 1.0, lum_t) + 0.04 * (n2 - 0.5))
    # une liane qui pend de la canopée, et ses feuilles en cœur
    liane = courbe([(116, 0), (111, 14), (114, 30), (109, 46), (107, 58)], 8)
    m, _ = trait(liane, 1.1)
    for k, p in enumerate(liane[8::8]):
        s_ = 1 if k % 2 else -1
        m |= ((X - p[0] - 1.8 * s_) ** 2 / 4.5 + (Y - p[1] - 1.2) ** 2 / 2.4) < 1
    poser(m, vb - 0.26)
    # la canopée : des amas de feuilles sur des rameaux, avec le ciel entre eux
    ciel = Y < 8 + 2 * np.sin(X * 0.05)
    can = np.zeros((H, W), bool)
    dessus = np.zeros((H, W))
    rng = np.random.default_rng(31)
    for k in range(100):
        cx, cy = rng.uniform(-10, W + 10), rng.exponential(3.6) - 1
        mm, ww = ellipses(int(rng.integers(7, 14)), (cx, cy), (5, 2.4), (2.2, 4.2), 100 + k)
        can |= mm
        dessus = np.where(mm, ww, dessus)
    can |= Y < 1.5 + 2.5 * bruit(6, 13)
    vc = 0.07 + 0.08 * lisse(2, 24, Y) + 0.08 * lum + 0.05 * (dessus > 0.4)
    poser(can, vc)
    img = rampe(v, BRUME)
    trous = ciel & ~can
    img = np.where(trous[..., None], rampe(0.1 + 0.8 * lum - 0.05 * Y + 0.1 * (n2 - 0.5), CIEL), img)

    # la litière au loin : un tapis brun noyé dans la brume, semé de feuilles
    sol_loin = Y >= 114 + 1.5 * np.sin(X * 0.05) + 1.0 * np.sin(X * 0.13 + 1)
    brun = rampe(0.30 + 0.18 * (bruit(6, 4) - 0.5), LITIERE)
    rng = np.random.default_rng(41)
    for _ in range(170):
        cx, cy = rng.uniform(-8, W + 8), rng.uniform(112, 146)
        prof = (cy - 112) / 32
        rx, ry, an = (2.5 + 4.5 * prof) * rng.uniform(0.7, 1.3), 0.9 + 1.2 * prof, rng.uniform(-0.4, 0.4)
        dx, dy = X - cx, Y - cy
        u = (dx * math.cos(an) + dy * math.sin(an)) / rx
        w = (-dx * math.sin(an) + dy * math.cos(an)) / ry
        m = (u * u + w * w) < 1
        val = rng.uniform(0.35, 0.85) - 0.18 * (w > 0.2) + 0.1 * (w < -0.5)
        brun = np.where(m[..., None], rampe(np.full((H, W), val), LITIERE), brun)
        sol_loin |= m
    # quelques semis qui dépassent de la litière : une tige, deux feuilles
    semis = np.zeros((H, W), bool)
    for sx, sy, hs in [(78, 114, 9), (126, 113, 6), (168, 114, 8), (232, 116, 5)]:
        semis |= (np.abs(X - sx - 0.15 * (sy - Y)) < 0.5) & (Y > sy - hs) & (Y <= sy)
        for cote in (-1, 1):
            semis |= ((X - sx - 0.15 * hs - cote * 2.2) ** 2 / 5 + (Y - sy + hs + 0.6 * cote) ** 2 / 1.1) < 1
    brun = np.where(semis[..., None], rampe(vb - 0.16, BRUME), brun)
    sol_loin |= semis
    voile = (1 - lisse(112, 134, Y))[..., None] * 0.72
    brun = brun * (1 - voile) + rampe(vb - 0.04, BRUME) * voile
    img = np.where(sol_loin[..., None], brun, img)

    # le grand tronc de gauche et son contrefort, une lame de racine éclairée par le haut
    bord = 15 + 1.2 * np.sin(Y * 0.05)
    lame = courbe([(14, 50), (21, 74), (31, 97), (44, 116), (58, 128), (63, 132)], 10)
    haut_lame = np.interp(Y, lame[:, 1], lame[:, 0], left=-99, right=63)
    contre = (X < haut_lame) & (Y > 50) & (Y < 133 + 0.04 * X)
    tronc = (X < bord) | contre
    stries = np.sin(X * 1.7 + 5 * bruit(10, 5)) * 0.5 + 0.5
    ve = 0.26 + 0.16 * stries ** 3 + 0.06 * (bruit(4, 6) - 0.5) + 0.08 * lisse(0, 14, X)
    ve += 0.34 * lisse(-2.5, -0.5, X - bord) * (X < bord) * (Y < 66)       # liseré du tronc
    face = contre & (X >= bord)
    ve = np.where(face, 0.36 - 0.18 * lisse(96, 132, Y) + 0.14 * stries ** 3 + 0.05 * (bruit(4, 7) - 0.5), ve)
    arete = contre & (X >= haut_lame - 2.2)
    ve = np.where(arete, 0.7, ve)
    img = np.where(tronc[..., None], rampe(ve, ECORCE), img)
    # la mousse sur l'arête du contrefort et en plaques sur le tronc
    nm = bruit(5, 8, 2)
    mousse = tronc & ((contre & (X >= haut_lame - 3.6) & (nm > 0.42)) | ((nm > 0.66) & (X < bord) & (Y > 70)))
    vm = 0.25 + 0.55 * lisse(0.42, 0.8, nm) + 0.25 * (X >= haut_lame - 1.4)
    img = np.where(mousse[..., None], rampe(vm, MOUSSE), img)
    return img.astype(np.uint8), tronc, haut_lame


# ------------------------------------------------- la feuille morte et la litière proche
# nervure : du bout (à gauche) au pétiole (sous la croupe, hors champ à droite) ; vue d'un peu
# au-dessus du sol, la moitié lointaine de la feuille paraît plus étroite que la moitié proche
NERVURE = courbe([(40, 133), (100, 131.4), (160, 129.6), (215, 127.6), (275, 125.4)], 10)
FEUILLE = ["#25120A", "#36190D", "#482211", "#5A2C15", "#6B371A", "#7B421F", "#8A4E25",
           "#985A2C", "#A56734", "#B1743D", "#BC8248", "#C69055", "#D09F63", "#DAAE75"]


def feuille_geo():
    a, b = NERVURE[0], NERVURE[-1]
    ab = b - a
    t = ((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab)
    d = Y - np.interp(X, NERVURE[:, 0], NERVURE[:, 1])
    hl = 13.0 * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.55
    haut, bas = hl * 0.68, hl
    dedans = (t > 0) & (t < 1) & (((d < 0) & (-d < haut)) | ((d >= 0) & (d < bas)))
    r = np.where(d < 0, -d / np.maximum(haut, 0.01), d / np.maximum(bas, 0.01))
    return t, d, r, dedans, bas


def sol():
    rgba = np.zeros((H, W, 4))
    t, d, r, dedans, bas = feuille_geo()
    # la feuille sèche, un peu plus claire côté soleil ; grain fin et quelques taches
    v = 0.50 + 0.10 * (t - 0.5) - 0.08 * r + 0.06 * (bruit(2, 9) - 0.5) + 0.05 * (bruit(8, 8) - 0.5)
    v = np.where(bruit(3, 10) > 0.78, v - 0.16, v)
    # bords : le lointain accroche la lumière, le proche se recourbe et laisse voir son revers
    v = np.where(dedans & (d < 0) & (r > 0.86), 0.86, v)
    v = np.where(dedans & (d > 0) & (r > 0.76) & (r <= 0.9), v + 0.14, v)
    v = np.where(dedans & (d > 0) & (r > 0.9), 0.10, v)
    # nervures secondaires : de la nervure centrale vers le bord, en filant vers le bout
    nerv = np.zeros((H, W), bool)
    a, b = NERVURE[0], NERVURE[-1]
    for tk in np.linspace(0.07, 0.93, 21):
        p0 = a + (b - a) * tk
        p0[1] = np.interp(p0[0], NERVURE[:, 0], NERVURE[:, 1])
        for cote, f in ((-1, 0.68), (1, 1.0)):
            pts = []
            for s_ in np.linspace(0, 1, 6):
                tt = tk - 0.04 * s_ ** 1.3
                x_ = a[0] + (b[0] - a[0]) * tt
                hl_ = 13.0 * math.sin(math.pi * min(max(tt, 0), 1)) ** 0.55
                y_ = np.interp(x_, NERVURE[:, 0], NERVURE[:, 1]) + cote * f * hl_ * 0.84 * s_
                pts.append((x_, y_))
            nerv |= trait(np.array(pts), 0.9)[0]
    v = np.where(nerv & dedans & (r < 0.86), v + 0.17, v)
    v = np.where(np.abs(d) < 0.55, 0.88 - 0.1 * (t < 0.15), v)    # nervure centrale, claire
    v = np.where((d >= 0.55) & (d < 1.5), v - 0.12, v)             # et son ombre
    couleur = rampe(v, FEUILLE)
    # deux trous rongés
    trous = (((X - 96) / 2.6) ** 2 + ((Y - 135) / 1.4) ** 2 < 1) | (((X - 128) / 1.6) ** 2 + ((Y - 124.5) / 0.9) ** 2 < 1)
    couleur = np.where(trous[..., None], hexa("#1B120A"), couleur)
    # ombre de la grenouille : large et douce sous le corps, dense sous les pieds et la main
    o = np.exp(-(((X - (FX + 66)) / 48) ** 2) - (((Y - (SOL + 0.5)) / 3.4) ** 2)) * 0.6
    for cx, rx in [(FX + 27, 7), (FX + 52, 9), (FX + 84, 20)]:
        o = np.maximum(o, np.exp(-(((X - cx) / rx) ** 2) - (((Y - (SOL + 0.8)) / 1.7) ** 2)) * 0.85)
    ombre = o > BAYER * 0.9 + 0.08
    couleur = np.where(ombre[..., None], couleur * np.array([0.55, 0.5, 0.5]), couleur)
    # gouttes posées sur la feuille
    for gx, gy in [(84, 133), (114, 126), (238, 131)]:
        couleur[gy, gx] = hexa("#F4ECD8")
        couleur[gy, gx + 1] = couleur[gy, gx + 1] * 0.7 + hexa("#F4ECD8") * 0.3
        couleur[gy + 1, gx] = couleur[gy + 1, gx] * 0.55
    rgba[..., :3] = couleur
    rgba[..., 3] = dedans * 255

    # litière proche, dans l'ombre : trois feuilles devant la grande
    for cx, cy, rx, ry, an, val in [(26, 142, 15, 3.0, 0.06, 0.40), (122, 145, 13, 2.6, 0.03, 0.30),
                                    (206, 146, 11, 2.6, -0.05, 0.26)]:
        dx, dy = X - cx, Y - cy
        u = (dx * math.cos(an) + dy * math.sin(an)) / rx
        w = (-dx * math.sin(an) + dy * math.cos(an)) / ry
        m = (u * u + w * w) < 1
        vv = val + 0.2 * (w < -0.45) - 0.1 * (w > 0.4) + 0.08 * (bruit(3, int(cx)) - 0.5)
        vv = np.where(np.abs(w) < 0.2, vv + 0.12, vv)
        rgba[m] = np.concatenate([rampe(vv, LITIERE), np.full((H, W, 1), 255)], axis=2)[m]
    # petits champignons au pied du contrefort (Marasmius, chapeaux orangés)
    for cx, cy, lg, haut in [(65, 127, 2, 5), (69, 127, 3, 7), (74, 128, 1, 3)]:
        for k in range(haut):
            rgba[cy - k, cx] = [*hexa("#D9CDB0" if k > 1 else "#A99A7C"), 255]
        top = cy - haut
        for i in range(-lg, lg + 1):
            rgba[top, cx + i] = [*hexa("#B4552A" if i <= 0 else "#8C3E1E"), 255]
        for i in range(-lg + 1, lg):
            rgba[top - 1, cx + i] = [*hexa("#E5904A" if i < 0 else "#C76A30"), 255]
        if lg > 1:
            rgba[top - 1, cx - lg + 1] = [*hexa("#F2B070"), 255]
    sol_y = np.array([int(np.argmax(rgba[:, x, 3] > 0)) if rgba[:, x, 3].any() else 139 for x in range(W)])
    return rgba.astype(np.uint8), sol_y


def ombre_portee():
    """L'ombre que le bord proche, relevé, jette sur la litière juste en dessous."""
    t, d, r, dedans, bas = feuille_geo()
    return (t > 0.02) & (t < 1) & (d >= bas) & (d < bas + 2.2) & (BAYER < 0.85 - 0.3 * (d - bas))


# ------------------------------------------------------------------- l'avant-plan
def avant():
    """Une fronde de fougère qui pend du coin haut droit, à contre-jour."""
    rgba = np.zeros((H, W, 4))
    m = np.zeros((H, W), bool)
    eclaire = np.zeros((H, W), bool)
    rachis = courbe([(268, -10), (250, 2), (231, 10), (212, 15), (196, 17), (186, 16)], 10)
    m |= trait(rachis, 1.5)[0]
    n = len(rachis)
    for i in range(3, n - 1, 3):
        p = rachis[i]
        tg = rachis[min(i + 1, n - 1)] - rachis[i - 1]
        tg = tg / (np.linalg.norm(tg) + 1e-9)
        nrm = np.array([-tg[1], tg[0]])
        reste = 1 - i / n
        for sens in (-1, 1):
            lg = (3.0 + 9.0 * reste) * (1.0 if sens > 0 else 0.75)
            axe = nrm * sens * 0.85 + tg * 0.55 + np.array([0, 0.35 if sens > 0 else -0.1])
            axe = axe / np.linalg.norm(axe)
            dx, dy = X - p[0], Y - p[1]
            u = dx * axe[0] + dy * axe[1]
            w = -dx * axe[1] + dy * axe[0]
            larg = 1.4 * np.sin(np.pi * np.clip(u / lg, 0, 1)) ** 0.7 + 0.2
            pinnule = (u > 0) & (u < lg) & (np.abs(w) < larg)
            m |= pinnule
            eclaire |= pinnule & (w * sens < -0.3) & (u > 1)
    rgba[m, :3] = hexa("#0B1610")
    rgba[m & eclaire, :3] = hexa("#1A2E1F")
    rgba[m, 3] = 255
    return rgba.astype(np.uint8)


def png(rgba):
    tampon = io.BytesIO()
    Image.fromarray(rgba).save(tampon, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(tampon.getvalue()).decode()


def calques():
    f, _, _ = fond()
    f[ombre_portee()] = (f[ombre_portee()] * 0.5).astype(np.uint8)
    s, sol_y = sol()
    a = avant()
    return {"fond": png(f), "sol": png(s), "avant": png(a), "solY": sol_y.tolist()}, (f, s, a)


def grenouille_rgba(cle="0002"):
    d = json.loads((ICI / "chasse.json").read_text())["grenouille"]
    pal = {c: hexa(h) for c, h in d["palette"].items()}
    out = np.zeros((d["hauteur"], d["largeur"], 4), np.uint8)
    import re
    for y, r in enumerate(d["images"][cle]):
        i = 0
        for ch, n in re.findall(r"(\D)(\d+)", r):
            n = int(n)
            if ch != ".":
                out[y, i:i + n, :3] = pal[ch]
                out[y, i:i + n, 3] = 255
            i += n
    return out


if __name__ == "__main__":
    _, (f, s, a) = calques()
    img = Image.fromarray(f).convert("RGBA")
    img.alpha_composite(Image.fromarray(s))
    img.alpha_composite(Image.fromarray(grenouille_rgba()), (FX, FY))
    img.alpha_composite(Image.fromarray(a))
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else ICI / "apercu.png"
    img.convert("RGB").resize((W * 4, H * 4), Image.NEAREST).save(sortie)
    print(sortie)
