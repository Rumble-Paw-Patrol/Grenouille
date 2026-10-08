"""Modèle de sprite par étapes (skill sprite-grenouille) : à copier, puis à remplir.

    python3 <code_espece>.py ETAPE [SORTIE.json]
        ETAPE : silhouette | aplats | details | animation

Une grenouille est un jeu de FORMES (masques sur la grille du sprite), rangées du plan le plus
lointain au plus proche, chacune dans une MATIÈRE (peau du dos, flanc, ventre, membres…), et de
MOTIFS (bandes, barres, taches) peints par-dessus certaines matières. Chaque étape est un rendu
de ces mêmes formes, et n'en modifie aucune :

    silhouette  trois gris selon le plan (fond, corps, devant), le contour, l'œil (globe, iris,
                pupille) : la morphologie et la posture, rien d'autre ;
    aplats      une couleur par matière et par motif : ni dégradé, ni ombre, ni lumière, ni
                reflet ; les motifs à leur place exacte ;
    details     volume (lumière LUMIERE), rampes de 7 tons par matière, plaques de
                teinte, grain, reflets humides, détails de l'œil, contour clair côté lumière ;
                à affiner ensuite à la main selon les règles du skill (grenouille.py du tableau
                de bord est la référence de ce niveau de détail) ;
    animation   les 24 images du rendu détaillé : gorge × flanc × œil × tête.

Une correction de forme demandée à l'étape 2 ou 3 se fait dans formes() : les rendus suivent.
Un support (branche, feuille) se fait avec une copie de ce modèle dont formes() rend une seule
forme et None pour l'œil.
La grenouille de démonstration ci-dessous (une silhouette assise quelconque, en formes simples)
ne sert qu'à vérifier la chaîne : tout remplacer d'après les photos.
"""

import colorsys
import json
import math
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
NOM = "demo"  # code de l'espèce : sorties <NOM>_<etape>.json
UNITES = (57, 43)  # repère en unités (2 pixels par unité) : rien ne touche les bords
ECHELLE = 2
W, H = UNITES[0] * ECHELLE, UNITES[1] * ECHELLE
# MIROIR : tête à droite. Les formes restent décrites tête à gauche ; elles sont retournées avant
# le rendu, si bien que la lumière (LUMIERE, à l'écran) garde son côté.
MIROIR = False
LUMIERE = (
    -0.6,
    -0.8,
)  # d'où vient la lumière : en haut à gauche ; en mode image, celle de la photo
PIVOT, PAS = (18.0, 9.5), 0.06  # cou (unités) et rotation de la tête par cran (radians)
_xs = (np.arange(W) + 0.5) / ECHELLE
X, Y = np.meshgrid(UNITES[0] - _xs if MIROIR else _xs, (np.arange(H) + 0.5) / ECHELLE)
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16
TRAME = np.tile(BAYER, (H // 4 + 1, W // 4 + 1))[:H, :W]

# ------------------------------------------------------------------ matières et couleurs
# une couleur à plat par matière ; les rampes du rendu détaillé en sont tirées
MATIERES = {
    "dos": "#D07A35",
    "flanc": "#C2804A",
    "ventre": "#E8DCC4",
    "gorge": "#E8DCC4",
    "membre": "#D88A40",
    "fond": "#9C5223",  # membres de l'autre côté
    "levre": "#F2CC98",
}
MOTIFS = {"bande": "#4A230E", "barre": "#7E3F17"}
# œil : iris, pupille, cercle autour du globe, matière de la paupière fermée ; la forme de la
# pupille (et sa taille, dans rendu_oeil) vient des photos
OEIL = {"iris": "#C9A05A", "pupille": "#0B0705", "cercle": "#24120A", "paupiere": "dos"}
PUPILLE = "ronde"  # ronde | horizontale | verticale (fente)
CONTOUR, CONTOUR_CLAIR = "#24120A", "#5A2B14"
SILHOUETTE = {
    "fond": "#56625E",
    "corps": "#8E9B96",
    "devant": "#C3CEC9",
    "globe": "#E6ECEA",
    "pupille": "#141918",
}


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
        a, b, ra, rb = (
            np.array(pts[k], float),
            np.array(pts[k + 1], float),
            rayons[k],
            rayons[k + 1],
        )
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
        m |= np.hypot(x - a[0] - t * ab[0], y - a[1] - t * ab[1]) <= ra + (rb - ra) * t
    return m


def forme(nom, masque, matiere, plan="corps", cernee=False, volume=None):
    """plan : fond (membres de l'autre côté), corps, devant (membres proches) ; cernee : un
    contour la sépare de ce qu'elle recouvre ; volume : la forme dont elle prend le relief
    (une matière dessinée dans le corps, comme le flanc, prend celui du corps)."""
    return {
        "nom": nom,
        "masque": masque,
        "matiere": matiere,
        "plan": plan,
        "cernee": cernee,
        "volume": volume or nom,
    }


def tete_tournee(tete):
    """Coordonnées vues par la tête, tournée de tete × PAS autour du cou ; l'effet s'estompe
    vers le corps (rien ne bouge en arrière du pivot)."""
    poids = np.clip((PIVOT[0] + 3 - X) / 6, 0, 1)
    a = -tete * PAS * poids
    dx, dy = X - PIVOT[0], Y - PIVOT[1]
    return PIVOT[0] + np.cos(a) * dx - np.sin(a) * dy, PIVOT[1] + np.sin(a) * dx + np.cos(a) * dy


# ------------------------------------------------------------------ la grenouille
def formes(gorge=0.0, souffle=0.0, cligne=False, tete=0):
    """Les formes, du plus loin au plus près, les motifs et l'œil. À REMPLACER d'après les
    photos : ici, une grenouille assise quelconque, pour vérifier la chaîne."""
    xt, yt = tete_tournee(tete)
    corps = dans(
        catmull(
            [
                (2.2, 9.8),
                (4.5, 6.2),
                (10, 4.4),
                (18, 6.0),
                (30, 10.5),
                (40, 17),
                (44.5, 24),
                (42, 31),
                (33, 33.5 + souffle * 0.4),
                (22, 31),
                (13, 24 + gorge * 0.6),
                (6, 14.5),
            ]
        ),
        xt,
        yt,
    )
    tibia = membre([(28, 30), (40, 31.5), (42.5, 35)], [2.4, 3, 2.4], X, Y)
    f = [
        forme(
            "bras_fond",
            membre([(18, 22), (16, 29), (12, 33.5), (10, 34.5)], [2, 1.5, 1, 1], X, Y),
            "fond",
            "fond",
            cernee=True,
        ),
        forme("corps", corps, "dos"),
        # matières dessinées dans le corps : elles en gardent le relief
        forme("flanc", corps & (yt > 12 + 0.25 * (xt - 10)), "flanc", volume="corps"),
        forme("ventre", corps & (yt > 24 + 0.12 * (xt - 20)), "ventre", volume="corps"),
        forme("gorge", corps & (xt < 16) & (yt > 13.2), "gorge", volume="corps"),
        forme(
            "levre",
            corps & (xt < 10) & (np.abs(yt - (11.3 + 0.05 * xt)) < 0.45),
            "levre",
            volume="corps",
        ),
        forme("cuisse", ellipse(36, 27, 8.5, 6, X, Y, -0.3), "membre", cernee=True),
        forme(
            "pied",
            membre([(41, 37), (32, 38.5), (24, 38.8)], [1.6, 1.2, 0.8], X, Y),
            "membre",
            "devant",
            cernee=True,
        ),
        forme("tibia", tibia, "membre", "devant", cernee=True),
        forme(
            "bras",
            membre([(21, 21), (20, 30), (15, 36.5), (13, 38.5)], [2.6, 1.8, 1.2, 1], X, Y),
            "membre",
            "devant",
            cernee=True,
        ),
    ]
    motifs = {
        "bande": corps & (np.abs(yt - (9.0 + 0.32 * (xt - 6))) < 1.2) & (xt > 6) & (xt < 36),
        "barre": tibia & (np.sin(X * 1.6) > 0.55),
    }
    oeil = {"x": 11.2, "y": 8.4, "r": 2.6, "ferme": cligne}
    oeil["masque"] = (xt - oeil["x"]) ** 2 + (yt - oeil["y"]) ** 2 <= oeil["r"] ** 2
    oeil["u"], oeil["v"] = (xt - oeil["x"]) / oeil["r"], (yt - oeil["y"]) / oeil["r"]
    if MIROIR:
        oeil["u"] = -oeil["u"]
    return f, motifs, oeil


# ------------------------------------------------------------------ rendus
def etiqueter(f):
    """Matière, plan et numéro de forme de chaque pixel (le plus proche l'emporte)."""
    mat, plan, num = np.full((H, W), "", object), np.full((H, W), "", object), np.full((H, W), -1)
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
    for j0, i0 in zip(*np.nonzero(plein), strict=False):
        if vus[j0, i0]:
            continue
        pile, groupe = [(j0, i0)], []
        vus[j0, i0] = True
        while pile:
            j, i = pile.pop()
            groupe.append((j, i))
            for v in ((j + 1, i), (j - 1, i), (j, i + 1), (j, i - 1)):
                if 0 <= v[0] < H and 0 <= v[1] < W and plein[v] and not vus[v]:
                    vus[v] = True
                    pile.append(v)
        if len(groupe) < n:
            for p in groupe:
                img[p] = CONTOUR
    return img


def rendu_oeil(img, oeil, couleurs, ferme_couleur):
    if oeil is None:  # un support (branche, feuille) n'a pas d'œil
        return img
    m, u, v = oeil["masque"], oeil["u"], oeil["v"]
    if oeil["ferme"]:
        img[m] = ferme_couleur
        img[m & (np.abs(v - 0.15 * u * u) < 0.18)] = CONTOUR
        return img
    img[m] = couleurs["iris"]
    img[m & (u * u + v * v > 0.8)] = couleurs.get("cercle", CONTOUR)
    p = {
        "ronde": u * u + v * v < 0.3,
        "horizontale": (u / 0.75) ** 2 + (v / 0.32) ** 2 < 1,
        "verticale": (u / 0.28) ** 2 + (v / 0.8) ** 2 < 1,
    }[PUPILLE]
    img[m & p] = couleurs["pupille"]
    return img


def silhouette(f, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom, coul in SILHOUETTE.items():
        img[plan == nom] = coul
    img = contour(img, f, num)
    return rendu_oeil(
        img,
        oeil,
        {"iris": SILHOUETTE["globe"], "pupille": SILHOUETTE["pupille"]},
        SILHOUETTE["corps"],
    )


def aplats(f, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom, coul in MATIERES.items():
        img[mat == nom] = coul
    for nom, m in motifs.items():
        img[m & (num >= 0)] = MOTIFS[nom]
    img = contour(img, f, num)
    return rendu_oeil(img, oeil, OEIL, MATIERES[OEIL["paupiere"]])


def _flou(a, r):
    for axe in (0, 1):
        a = sum(np.roll(a, k, axis=axe) for k in range(-r, r + 1)) / (2 * r + 1)
    return a


def rampe(hexa, n=7):
    """n tons d'une couleur : ombres plus sombres tirant vers le bleu, lumières vers le doré."""
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


def details(f, motifs, oeil):
    """Volume par forme (bords tournés vers la lumière plus clairs), rampes tramées, plaques
    de teinte, grain, œil brillant, contour clair côté lumière. Point de départ : affiner à la
    main (ombres de contact, reflets humides, épaule, pustules…)."""
    mat, plan, num = etiqueter(f)
    lx, ly = LUMIERE
    lum = np.full((H, W), 0.5)
    volumes = {fo["nom"]: fo["masque"] for fo in f}
    for k, fo in enumerate(f):
        b = _flou(volumes[fo["volume"]].astype(float), 3)
        gy, gx = np.gradient(b)
        face = gx * lx + gy * ly  # bord tourné vers la lumière : positif
        ici = num == k
        lum[ici] = (0.55 + 4.0 * face - 0.18 * (Y / UNITES[1] - 0.4))[ici]
    plaques = np.clip(np.floor(_flou(np.random.default_rng(3).random((H, W)), 4) * 6 - 2), -1, 1)
    grain = (np.random.default_rng(5).random((H, W)) - 0.5) * 0.08
    img = np.full((H, W), "", object)
    couleurs = {**MATIERES, **{"motif_" + n: c for n, c in MOTIFS.items()}}
    zone = mat.copy()
    for nom, m in motifs.items():
        zone[m & (num >= 0)] = "motif_" + nom
    for nom, coul in couleurs.items():
        ici = zone == nom
        if not ici.any():
            continue
        tons = rampe(coul)
        idx = np.clip(
            np.floor((lum + grain) * (len(tons) - 1) + TRAME * 0.9), 0, len(tons) - 1
        ).astype(int)
        for v in (-1, 0, 1):  # trois variantes de teinte, par plaques
            sous = ici & (plaques == v)
            if v:
                tons_v = rampe(_decaler(coul, 0.012 * v))
                img[sous] = np.array(tons_v, object)[idx[sous]]
            else:
                img[sous] = np.array(tons, object)[idx[sous]]
    img = contour(img, f, num)
    # contour clair du côté de la lumière, autour des membres proches
    devant = plan == "devant"
    sx, sy = (-1 if LUMIERE[0] < 0 else 1), (-1 if LUMIERE[1] < 0 else 1)
    cote_lumiere = np.roll(devant, sy, axis=0) | np.roll(devant, sx, axis=1)
    img[(img == CONTOUR) & (num < 0) & cote_lumiere] = CONTOUR_CLAIR
    img = rendu_oeil(img, oeil, OEIL, rampe(MATIERES[OEIL["paupiere"]])[4])
    if oeil and not oeil["ferme"]:  # reflet blanc, petit reflet bleuté, bas du globe
        u, v = oeil["u"], oeil["v"]
        m = oeil["masque"]
        img[m & ((u + 0.35) ** 2 + (v + 0.35) ** 2 < 0.05)] = "#FFFFFF"
        img[m & ((u - 0.3) ** 2 + (v - 0.4) ** 2 < 0.03)] = "#9FB4C0"
    return img


def _decaler(hexa, dh):
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    return hexa_de(*colorsys.hls_to_rgb((h + dh) % 1, clarte, s))


def hexa_de(r, g, b):
    return f"#{round(r * 255):02X}{round(g * 255):02X}{round(b * 255):02X}"


RENDUS = {"silhouette": silhouette, "aplats": aplats, "details": details}


# ------------------------------------------------------------------ sortie
def coder(images):
    """Images de couleurs hexa → palette de lettres (K et k réservés au contour) et plages."""
    couleurs = sorted(
        {c for im in images.values() for c in im.ravel() if c} - {CONTOUR, CONTOUR_CLAIR}
    )
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    lettres += [chr(0x100 + i) for i in range(max(0, len(couleurs) - len(lettres)))]
    palette = {"K": CONTOUR, "k": CONTOUR_CLAIR, **dict(zip(lettres, couleurs, strict=False))}
    code = {c: lettre for lettre, c in palette.items()} | {"": "."}
    out = {}
    for cle, im in images.items():
        rangs = []
        for rang in im:
            s, prec, n = "", None, 0
            for c in rang:
                ch = code[c]
                if ch == prec:
                    n += 1
                else:
                    s += f"{prec}{n}" if prec else ""
                    prec, n = ch, 1
            rangs.append(s + f"{prec}{n}")
        out[cle] = rangs
    return {"palette": palette, "largeur": W, "hauteur": H, "images": out}


def produire(etape):
    if etape == "animation":
        images = {}
        for g in (0, 1):
            for s in (0, 1):
                for c in (0, 1):
                    for t in (-1, 0, 1):
                        images[f"{g}{s}{c}{t + 1}"] = details(*formes(g * 0.9, s * 0.7, bool(c), t))
        return coder(images)
    return coder({"0001": RENDUS[etape](*formes())})


if __name__ == "__main__":
    etape = sys.argv[1] if len(sys.argv) > 1 else "silhouette"
    sortie = Path(sys.argv[2]) if len(sys.argv) > 2 else ICI / f"{NOM}_{etape}.json"
    sprite = produire(etape)
    sortie.write_text(json.dumps(sprite, separators=(",", ":")), encoding="utf-8")
    print(f"{sortie.name} : {len(sprite['images'])} image(s), {len(sprite['palette'])} couleurs")
