"""La branche entière de la scène (C. tomopterna, photo d'Olivier Louguet), et son ménisque.

    python3 documentation/tableau-de-bord/bac-a-sable/tomopterna/branche.py [apercu.png]

Les deux grenouilles validées ont chacune le morceau de branche de leur cadre
(../tomopterna-gauche/branche_details_valide.json et ../tomopterna-droite/…). La scène veut la
branche d'un bord à l'autre de la photo : hors des cadres, elle est peinte comme les morceaux
(même écorce, même lumière, même lichen gris argenté, relevé sur la photo) ; dans les cadres,
ce sont les morceaux validés, pixel pour pixel. Ses bords suivent la ligne de la session du
8/10/2026 (celle du morceau de droite) et passent en douceur à celle du morceau de gauche.

Le ménisque : à gauche de la photo, la branche touche l'eau, et l'eau monte s'y coller
(tension superficielle). Sous la branche, un liseré d'eau bleu-gris, qui reflète le ciel, et
plus clair en bas, où sa courbe prend la lumière ; il s'amincit et s'éteint là où la branche
quitte l'eau. Relevé sur la photo : de x = 0 à 60 pixels de toile, jusqu'à la rangée 159, sous
laquelle l'eau est plate. C'est un sprite à part : la page le pose sur l'eau et le reflète.
"""

import colorsys
import json
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
ECHELLE = 2  # pixels de toile par unité de la scène
X0, Y0, W, H = 0, 40, 448, 124  # le cadre de la branche sur la toile de 448 × 303
_xs = (X0 + np.arange(W) + 0.5) / ECHELLE
X, Y = np.meshgrid(_xs, (Y0 + np.arange(H) + 0.5) / ECHELLE)
LUMIERE = (-0.6, -0.8)  # celle des morceaux
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
TRAME = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]
ECORCE, LICHEN_GRIS = "#463733", "#8E8D94"  # les matières des morceaux
CONTOUR = "#24120A"
BORD = 4  # colonnes de bord des morceaux, repeintes

# les morceaux validés : fichier, coin sur la toile, et la rangée (en unités) d'où leur lumière
# compte la hauteur (l'ORIGINE de leur cadre)
MORCEAUX = {
    "gauche": (ICI.parent / "tomopterna-gauche/branche_details_valide.json", 96, 60, 30.0),
    "droite": (ICI.parent / "tomopterna-droite/branche_details_valide.json", 236, 46, 23.0),
}

# la ligne médiane et la demi-épaisseur (unités), relevées sur la photo le 8/10/2026
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

# le lichen hors des cadres, décalqué sur la photo réduite à la toile : les pixels clairs et
# gris du dessus de la branche. Par rangée de la toile, les colonnes ou plages de colonnes
LICHEN = {
    81: "442-444",
    82: "431-435 437-439",
    83: "420-423 431-435 441-442",
    84: "413-419 421-422 432 434-437",
    85: "408-410 412-419 429-430 433-435 437",
    86: "400-402 408-409 412-414 418-424 430-431 434-436",
    87: "396-397 405-411 419-420 422-425 428-429 434",
    88: "382-383 399-402 404-410 413 415-416 418-420 423",
    89: "374-375 380-381 398-399 403-407 410-411 413 415",
    90: "364-371 403-407",
    91: "356-358 360-361 368 390-392 395-396 400 403-404 406-412",
    92: "347-355 359-360 396-397 400-402 407-410",
    93: "370-371 374-376 400 407",
    94: "362-364 366-368 370",
    95: "361-365",
    109: "235",
    110: "230-235",
    111: "223-226 229-230",
    113: "216-219",
    114: "209-212 227-228",
    134: "93-95",
    135: "89-95",
    136: "82-83 87-89 91-95",
    137: "77-81 85-88 92 94-95",
    138: "75-77 85-88",
    139: "67-69 71-72 85 87",
    140: "58-59 61-63",
    142: "46-47",
    143: "41-46",
    144: "22 34 36-38",
    145: "19-26 29-30 32-35 44-45 47-50",
    146: "11 17-25 30-31 34-35 42-43",
    147: "0-5 7-8 10-12 15-20 27-31 33-34",
    148: "0-5 7-14 17-18 21-34",
    149: "0 4-15 25-27",
    150: "4 6-7 9-12 14 17-18",
}


def suivre(pts, x):
    return np.interp(x, [p[0] for p in pts], [p[1] for p in pts])


def lisse(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def bords(x):
    """Le dessus et le dessous de la branche (unités) : la ligne du 8/10, et dans le cadre de
    la grenouille de gauche (x de 48 à 103) les bords droits de son morceau, raccordés sur 14 et
    15 unités de part et d'autre."""
    c, e = suivre(BRANCHE, x), suivre(EPAISSEUR, x)
    w = lisse(34, 48, x) * (1 - lisse(103, 118, x))
    dessus_g = 65.8 - (x - 48.0) * 0.188
    dessous_g = 72.5 - (x - 48.0) * 0.204
    return w * dessus_g + (1 - w) * (c - e), w * dessous_g + (1 - w) * (c + e)


def flou(a, r):
    """Moyenne glissante (2r + 1)², bords prolongés (pas de bouclage d'un bord à l'autre)."""
    for axe in (0, 1):
        p = np.pad(a, [(r, r) if k == axe else (0, 0) for k in range(a.ndim)], mode="edge")
        a = sum(np.take(p, range(k, k + a.shape[axe]), axis=axe) for k in range(2 * r + 1))
        a = a / (2 * r + 1)
    return a


def hexa_de(r, g, b):
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


def rampe(hexa, n=7):
    """Celle des morceaux : ombres plus sombres tirant vers le bleu, lumières vers le doré."""
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


def decaler(hexa, dh):
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    return hexa_de(*colorsys.hls_to_rgb((h + dh) % 1, clarte, s))


def pixels_lichen():
    m = np.zeros((H, W), bool)
    for y, plages in LICHEN.items():
        for p in plages.split():
            a, _, b = p.partition("-")
            m[y - Y0, int(a) - X0 : int(b or a) + 1 - X0] = True
    return m


def morceau(nom):
    """Le morceau validé : ses couleurs (hexa, "" où il est vide) sur toute la toile de la
    branche, et le rectangle de son cadre."""
    fichier, x, y, _ = MORCEAUX[nom]
    s = json.loads(fichier.read_text(encoding="utf-8"))
    img = np.full((H, W), "", object)
    for j, rang in enumerate(s["images"]["0001"]):
        i = 0
        for c, n in ((r[0], int(r[1:])) for r in _plages(rang)):
            if c != ".":
                img[y + j - Y0, x + i - X0 : x + i + n - X0] = s["palette"][c]
            i += n
    # sur leurs quatre colonnes de bord, les morceaux ont une lumière faussée (leur flou y
    # boucle d'un bord du cadre à l'autre) : la branche entière les remplace
    cadre = np.zeros((H, W), bool)
    cadre[y - Y0 : y + s["hauteur"] - Y0, x + BORD - X0 : x + s["largeur"] - BORD - X0] = True
    return img, cadre


def _plages(rang):
    out, k = [], 0
    while k < len(rang):
        j = k + 1
        while j < len(rang) and rang[j].isdigit():
            j += 1
        out.append(rang[k:j])
        k = j
    return out


def branche():
    """La branche peinte d'un bord à l'autre, puis les morceaux validés dans leurs cadres."""
    dessus, dessous = bords(X)
    m = (Y > dessus) & (Y < dessous)
    lx, ly = LUMIERE
    gy, gx = np.gradient(flou(m.astype(float), 3))
    face = -(gx * lx + gy * ly)  # bord tourné vers la lumière : positif
    # la hauteur se compte depuis le haut du cadre du morceau voisin, comme dans les morceaux
    haut = np.interp(X, [103, 118], [MORCEAUX["gauche"][3], MORCEAUX["droite"][3]])
    lum = 0.55 + 4.0 * face - 0.18 * ((Y - haut) / 45 - 0.4)
    rng = np.random.default_rng
    plaques = np.clip(np.floor(flou(rng(3).random((H, W)), 4) * 6 - 2), -1, 1)
    grain = (rng(5).random((H, W)) - 0.5) * 0.08
    lichen = pixels_lichen() & m
    img = np.full((H, W), "", object)
    for zone, coul in ((m & ~lichen, ECORCE), (lichen, LICHEN_GRIS)):
        idx = np.clip(np.floor((lum + grain) * 6 + TRAME * 0.9), 0, 6).astype(int)
        for v in (-1, 0, 1):  # trois variantes de teinte, par plaques
            sous = zone & (plaques == v)
            img[sous] = np.array(rampe(decaler(coul, 0.012 * v) if v else coul), object)[idx[sous]]
    v = np.pad(m, 1)
    autour = v[:-2, 1:-1] | v[2:, 1:-1] | v[1:-1, :-2] | v[1:-1, 2:]
    img[~m & autour] = CONTOUR
    for nom in MORCEAUX:
        piece, cadre = morceau(nom)
        img[cadre] = piece[cadre]
    return img


# ------------------------------------------------------------------ le ménisque
EAU_PLATE = 160  # la première rangée d'eau plate, sous le ménisque (toile)
PLEIN, FIN = 40, 60  # le ménisque est plein jusqu'à x = PLEIN, puis remonte vers la branche
# relevés sur la photo, du contact avec la branche à la courbe éclairée : bleu-gris (il
# reflète le dessous de la branche), bleu-gris du ciel, plus clair, le liseré argenté ; un peu
# plus clairs que la photo, comme tout le décor, pour qu'il se lise à côté des grenouilles
# quelques stries claires dans le liseré : (colonne, rangée depuis la branche, longueur)
STRIES = [(3, 2, 7), (17, 1, 5), (27, 3, 6), (36, 2, 4)]
TONS_MENISQUE = ["#56607A", "#66718A", "#727E96", "#949DB0", "#BCC4D0", "#D6DCE4"]


def menisque(img):
    """Sous la branche (img, ses couleurs), l'eau collée à elle, colonne par colonne, du
    dernier pixel de la branche jusqu'à son bord inférieur : la rangée EAU_PLATE jusqu'à
    x = PLEIN, puis une courbe qui remonte jusqu'à la branche en x = FIN (là où elle quitte
    l'eau). Le bas de chaque colonne est le liseré clair, le haut le contact sombre."""
    x0, y0 = 0, 146
    w, h = FIN + 1, EAU_PLATE - y0
    out = np.full((h, w), "", object)
    rng = np.random.default_rng(17)
    eclats = []
    for i in range(w):
        bas = Y0 + np.nonzero(img[:, i - X0] != "")[0].max() + 1  # sous la branche
        t = lisse(PLEIN, FIN + 1, i + 0.5)
        # le bord inférieur remonte, lentement puis de plus en plus vite, jusqu'à la branche
        fond = EAU_PLATE - (EAU_PLATE - bas) * t**2
        n = int(round(fond)) - bas
        if n <= 0:
            continue
        for k in range(n):
            if k == n - 1:  # le liseré, argenté, plus brillant par endroits
                ton = 5 if t < 0.5 and rng.random() < 0.3 else 4
            elif k == n - 2:
                ton = 3
            elif k == 0 and n > 3:  # contre l'écorce, l'eau prend son ombre
                ton = 0
            elif any(a <= i < a + n_ and k == r for a, r, n_ in STRIES):
                ton = 3  # une strie claire : le ciel sur la courbe de l'eau
            else:  # bleu-gris, en deux tons tramés, plus clair vers le bas, plus sombre au bout
                ton = 1 + int(TRAME[(bas + k - Y0) % H, i] < (k + 0.5) / n - 0.6 * t)
            out[bas + k - y0, i] = TONS_MENISQUE[ton]
            if ton >= 4:
                eclats.append([int(x0 + i), int(bas + k)])
    return out, (x0, y0), eclats


# ------------------------------------------------------------------ sortie
def coder(img):
    couleurs = sorted({c for c in img.ravel() if c} - {CONTOUR})
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    lettres += [chr(c) for c in range(0x100, 0x100 + max(0, len(couleurs) - len(lettres)))]
    palette = {"K": CONTOUR, **dict(zip(lettres, couleurs, strict=False))}
    code = {c: lettre for lettre, c in palette.items()} | {"": "."}
    rangs = []
    for rang in img:
        s, prec, n = "", None, 0
        for c in rang:
            ch = code[c]
            if ch == prec:
                n += 1
            else:
                s += f"{prec}{n}" if prec else ""
                prec, n = ch, 1
        rangs.append(s + f"{prec}{n}")
    return {
        "palette": palette,
        "contour": "K",
        "largeur": img.shape[1],
        "hauteur": img.shape[0],
        "images": {"0001": rangs},
    }


def produire():
    """{"branche": sprite, "menisque": sprite} (avec x, y : leur coin sur la toile), et les
    pixels clairs du ménisque, que la page fait scintiller."""
    img = branche()
    men, (mx, my), eclats = menisque(img)
    return {
        "branche": {"x": X0, "y": Y0, **coder(img)},
        "menisque": {"x": mx, "y": my, **coder(men)},
    }, eclats


if __name__ == "__main__":
    sprites, _ = produire()
    for nom, s in sprites.items():
        print(f"{nom} : {s['largeur']} × {s['hauteur']}, {len(s['palette'])} couleurs")
    if len(sys.argv) > 1:
        from PIL import Image

        toile = np.zeros((303, 448, 4), np.uint8)
        toile[..., :3] = (40, 44, 40)
        toile[..., 3] = 255
        for s in sprites.values():
            for j, rang in enumerate(s["images"]["0001"]):
                i = 0
                for p in _plages(rang):
                    c, n = p[0], int(p[1:])
                    if c != ".":
                        h = s["palette"][c]
                        toile[s["y"] + j, s["x"] + i : s["x"] + i + n, :3] = [
                            int(h[k : k + 2], 16) for k in (1, 3, 5)
                        ]
                    i += n
        Image.fromarray(toile).resize((448 * 3, 303 * 3), Image.NEAREST).save(sys.argv[1])
