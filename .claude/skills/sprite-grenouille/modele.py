"""Modèle de sprite par étapes (skill sprite-grenouille) : à copier, puis à remplir.

    python3 <code_espece>.py ETAPE [SORTIE.json]
        ETAPE : ebauche | croquis | aplats | details | animation

Une grenouille est un SQUELETTE (repères anatomiques nommés : articulations, museau, œil,
tympan…), un jeu de FORMES tirées de ce squelette (masques sur la grille du sprite), rangées du
plan le plus lointain au plus proche, chacune dans une MATIÈRE (dos, flanc, ventre, membres…),
des TRAITS (bouche, narine, tympan, plis) et des MOTIFS (bandes, barres, taches) peints
par-dessus certaines matières. Chaque étape est un rendu de ces mêmes formes, et n'en modifie
aucune :

    ebauche     étape 1 : formes simples en trois gris selon le plan (fond, corps, devant), le
                contour, l'œil, les traits du visage, et le squelette nommé, tracé par la
                planche (bouton « Squelette ») : la morphologie et la posture, rien d'autre ;
    croquis     étape 2 : les formes affinées, en dessin sans couleur (papier clair, plans en
                gris légers, contour, traits, doigts au pixel) ;
    aplats      étape 3a : une couleur par matière et par motif : ni dégradé, ni ombre, ni
                lumière, ni reflet ; les motifs à leur place exacte ;
    details     étape 3b : volume (lumière LUMIERE), rampes de 7 tons par matière, plaques de
                teinte, grain, reflets humides, détails de l'œil, contour clair côté lumière ;
                à affiner ensuite à la main selon les règles du skill (grenouille.py du tableau
                de bord est la référence de ce niveau de détail) ;
    animation   étape 4 : les 24 images du rendu détaillé : gorge × flanc × œil × tête.

Une correction de posture se fait dans SQUELETTE (déplacer un point), une correction de forme
dans formes() : les rendus suivent. Après la validation du croquis, toute correction de forme
ramène à l'étape 2 (voir le skill).
Côtés : _g et _d sont la gauche et la droite de l'animal, jamais de l'image. De profil, tête à
gauche, on voit son côté gauche : les membres _g sont devant, les _d au fond.
Un support (branche, feuille) se fait avec une copie de ce modèle dont formes() rend une seule
forme, aucun trait et None pour l'œil, et dont SQUELETTE et MEMBRES sont vides.
La grenouille de démonstration ci-dessous (une silhouette assise quelconque, en formes simples)
ne sert qu'à vérifier la chaîne : tout remplacer d'après les photos et la fiche morphologique.
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
# ORIGINE : en mode image, coin haut gauche du sprite dans le repère de la scène (en unités) ;
# les formes et le squelette s'écrivent alors directement en unités de la scène
ORIGINE = (0.0, 0.0)
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
X, Y = X + ORIGINE[0], Y + ORIGINE[1]
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
MOTIFS = {"bande": "#4A230E", "barre": "#7E3F17"}  # noms distincts de ceux des matières
# les traits du croquis prennent, à l'étape 3a, la couleur qu'ils traversent, assombrie ; ceux
# nommés ici gardent une matière (une lèvre blanche, par exemple : {"bouche": "levre"})
TRAITS_MATIERE = {}
# œil : iris, pupille, cercle autour du globe, matière de la paupière fermée ; la forme de la
# pupille (et sa taille, dans rendu_oeil) vient des photos
OEIL = {"iris": "#C9A05A", "pupille": "#0B0705", "cercle": "#24120A", "paupiere": "dos"}
PUPILLE = "ronde"  # ronde | horizontale | verticale (fente)
CONTOUR, CONTOUR_CLAIR = "#24120A", "#5A2B14"
# CONTOUR_TEINTE : au rendu détaillé, le contour prend la couleur qu'il borde, assombrie, au lieu
# d'un trait sombre uniforme (choix de l'étape 3b ; C. tomopterna l'a pris, A. blanci non)
CONTOUR_TEINTE = False
SILHOUETTE = {
    "fond": "#56625E",
    "corps": "#8E9B96",
    "devant": "#C3CEC9",
    "globe": "#E6ECEA",
    "pupille": "#141918",
}
# le croquis : du papier, les plans en gris légers, des traits sombres
CROQUIS = {
    "fond": "#C9CCC8",
    "corps": "#E4E6E2",
    "devant": "#F6F7F4",
    "trait": "#5E6663",
    "globe": "#FFFFFF",
    "pupille": "#141918",
}

# ------------------------------------------------------------------ squelette
# Les repères anatomiques, en unités, relevés sur la photo (étape 1) : articulations, bouts des
# doigts et traits du visage. Les membres en sont tirés : une correction de posture se fait en
# déplaçant un point. La planche les trace et les nomme (bouton « Squelette ») ; outils.py
# lecture les pose sur la photo.
SQUELETTE = {
    # tête
    "museau": (2.4, 9.6),
    "narine": (4.6, 8.2),
    "oeil": (11.2, 8.4),
    "commissure": (13.6, 12.2),
    "tympan": (16.6, 10.8),
    # tronc
    "nuque": (18.0, 6.4),
    "sacrum": (31.0, 11.0),
    "cloaque": (44.0, 25.0),
    # bras gauche (devant) et droit (au fond)
    "epaule_g": (21.0, 21.0),
    "coude_g": (20.0, 30.0),
    "poignet_g": (15.5, 36.0),
    "main_g": (13.5, 38.0),
    "epaule_d": (18.0, 22.0),
    "coude_d": (16.0, 29.0),
    "poignet_d": (12.0, 33.5),
    "main_d": (10.5, 34.5),
    # patte arrière gauche (devant) ; la droite est cachée par le corps
    "hanche_g": (40.0, 24.0),
    "genou_g": (28.0, 30.0),
    "talon_g": (42.0, 34.0),
    "tarse_g": (32.0, 38.5),
    "orteils_g": (24.0, 38.8),
    # bouts des doigts de la main gauche (4 doigts, I à IV)
    "doigt1_g": (12.5, 40.6),
    "doigt2_g": (10.0, 40.0),
    "doigt3_g": (9.0, 38.2),
    "doigt4_g": (10.2, 36.6),
}
# chaque membre : ses repères de la racine au bout, son plan, et les bouts des doigts ou des
# orteils, rattachés au dernier repère
MEMBRES = {
    "bras_d": (["epaule_d", "coude_d", "poignet_d", "main_d"], "fond", []),
    "bras_g": (
        ["epaule_g", "coude_g", "poignet_g", "main_g"],
        "devant",
        ["doigt1_g", "doigt2_g", "doigt3_g", "doigt4_g"],
    ),
    "patte_g": (["hanche_g", "genou_g", "talon_g", "tarse_g", "orteils_g"], "devant", []),
}
# l'axe du corps, et la ligne de la bouche
AXES = [["museau", "commissure"], ["nuque", "sacrum", "cloaque"]]


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


def os_(noms, rayons, x=None, y=None):
    """Un membre tiré du squelette : membre() sur les repères nommés."""
    return membre(
        [SQUELETTE[n] for n in noms], rayons, X if x is None else x, Y if y is None else y
    )


def forme(nom, masque, matiere, plan="corps", cernee=False, volume=None, zones=None):
    """plan : fond (membres de l'autre côté), corps, devant (membres proches) ; cernee : un
    contour la sépare de ce qu'elle recouvre ; volume : la forme dont elle prend le relief
    (une matière dessinée dans le corps, comme le flanc, prend celui du corps) ; zones : les
    autres matières et les motifs propres à la forme, {nom: masque}, posés dans l'ordre et
    seulement sur elle : des barres de tibia ne débordent pas sur le bras qui le couvre."""
    return {
        "nom": nom,
        "masque": masque,
        "matiere": matiere,
        "plan": plan,
        "cernee": cernee,
        "volume": volume or nom,
        "zones": zones or {},
    }


def bande(m, dx, dy, n=2):
    """La bande de n pixels de la forme m tournée vers la direction (dx, dy) : les pixels dont
    un des n voisins dans cette direction sort de la forme (la face verte d'un membre)."""
    out = np.zeros(m.shape, bool)
    for k in range(1, n + 1):
        di, dj = round(k * dx), round(k * dy)
        v = np.zeros(m.shape, bool)
        src = m[max(dj, 0) : H + min(dj, 0), max(di, 0) : W + min(di, 0)]
        v[max(-dj, 0) : H + min(-dj, 0), max(-di, 0) : W + min(-di, 0)] = src
        out |= m & ~v
    return out


def barres(chemin, centres, largeur=1.0, portee=3.0):
    """Des barres en travers d'un membre ou d'un doigt : chemin, ses points (os ou tracé du
    doigt) ; centres, le milieu de chaque barre, relevé sur la photo (outils.py profil le long
    de l'os : chaque creux est une barre) ; largeur, le long du membre, et portée, de part et
    d'autre de son axe. Tout en unités. Chaque barre suit le segment le plus proche."""
    m = np.zeros((H, W), bool)
    for cx, cy in centres:
        meilleur = None
        for (xa, ya), (xb, yb) in zip(chemin, chemin[1:], strict=False):
            dx, dy = xb - xa, yb - ya
            t = np.clip(((cx - xa) * dx + (cy - ya) * dy) / (dx * dx + dy * dy), 0, 1)
            d = math.hypot(xa + t * dx - cx, ya + t * dy - cy)
            if meilleur is None or d < meilleur[0]:
                meilleur = (d, dx / math.hypot(dx, dy), dy / math.hypot(dx, dy))
        _, ux, uy = meilleur
        long = (X - cx) * ux + (Y - cy) * uy
        trav = -(X - cx) * uy + (Y - cy) * ux
        m |= (np.abs(long) < largeur / 2) & (np.abs(trav) < portee)
    return m


def pixels(*pts):
    """Un motif posé au pixel près : les pixels (i, j) du sprite, relevés sur la photo."""
    m = np.zeros((H, W), bool)
    for i, j in pts:
        m[j, i] = True
    return m


def tete_tournee(tete):
    """Coordonnées vues par la tête, tournée de tete × PAS autour du cou ; l'effet s'estompe
    vers le corps (rien ne bouge en arrière du pivot). Une tête décrite tournée vers la droite
    (en mode image, sans MIROIR : museau à droite de la nuque) tourne dans l'autre sens."""
    sens = -1 if SQUELETTE and SQUELETTE["museau"][0] > SQUELETTE["nuque"][0] else 1
    poids = np.clip((sens * (PIVOT[0] - X) + 3) / 6, 0, 1)
    a = -sens * tete * PAS * poids
    dx, dy = X - PIVOT[0], Y - PIVOT[1]
    return PIVOT[0] + np.cos(a) * dx - np.sin(a) * dy, PIVOT[1] + np.sin(a) * dx + np.cos(a) * dy


# ------------------------------------------------------------------ la grenouille
def formes(gorge=0.0, souffle=0.0, cligne=False, tete=0):
    """Les formes, du plus loin au plus près, les traits, les motifs et l'œil. À REMPLACER
    d'après les photos : ici, une grenouille assise quelconque, pour vérifier la chaîne."""
    S = SQUELETTE
    xt, yt = tete_tournee(tete)
    corps = dans(
        catmull(
            [
                (2.2, 9.8),
                (4.5, 6.2),
                (10, 4.4),
                S["nuque"],
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
    tibia = os_(["genou_g", "talon_g"], [2.4, 2.8])
    main = np.zeros(X.shape, bool)
    # doigts fins, comme chez A. blanci : chez une autre espèce, la largeur des doigts et le
    # diamètre des disques se mesurent sur la photo
    for d in MEMBRES["bras_g"][2]:
        main |= membre([S["main_g"], S[d]], [0.5, 0.35], X, Y)
        main |= ellipse(*S[d], 0.55, 0.55, X, Y)
    f = [
        forme(
            "bras_fond",
            os_(MEMBRES["bras_d"][0], [2, 1.5, 1, 1]),
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
        forme("cuisse", ellipse(35, 27, 8.5, 6, X, Y, -0.3), "membre", cernee=True),
        forme(
            "pied",
            os_(["talon_g", "tarse_g", "orteils_g"], [1.6, 1.2, 0.8]),
            "membre",
            "devant",
            cernee=True,
        ),
        # les barres du tibia : une zone de la forme, pour qu'elles s'arrêtent sous le bras
        forme(
            "tibia",
            tibia,
            "membre",
            "devant",
            cernee=True,
            zones={"barre": np.sin(X * 1.6) > 0.55},
        ),
        forme(
            "bras",
            os_(MEMBRES["bras_g"][0], [2.6, 1.8, 1.2, 1]) | main,
            "membre",
            "devant",
            cernee=True,
        ),
    ]
    # traits du visage : la bouche, du museau à la commissure, la narine, le tympan
    (mx, my), (cx, cy) = S["museau"], S["commissure"]
    t = np.clip((xt - mx) / (cx - mx), 0, 1)
    tx, ty = S["tympan"]
    rt = np.hypot(xt - tx, yt - ty)
    traits = {
        "bouche": corps & (xt > mx + 0.8) & (xt < cx) & (np.abs(yt - (my + t * (cy - my))) < 0.4),
        "narine": (np.abs(xt - S["narine"][0]) < 0.5) & (np.abs(yt - S["narine"][1]) < 0.5),
        "tympan": (rt > 1.1) & (rt < 1.6),
    }
    motifs = {
        "bande": corps & (np.abs(yt - (9.0 + 0.32 * (xt - 6))) < 1.2) & (xt > 6) & (xt < 36),
    }
    oeil = {"x": S["oeil"][0], "y": S["oeil"][1], "r": 2.6, "ferme": cligne}
    oeil["masque"] = (xt - oeil["x"]) ** 2 + (yt - oeil["y"]) ** 2 <= oeil["r"] ** 2
    oeil["u"], oeil["v"] = (xt - oeil["x"]) / oeil["r"], (yt - oeil["y"]) / oeil["r"]
    if MIROIR:
        oeil["u"] = -oeil["u"]
    return f, traits, motifs, oeil


# ------------------------------------------------------------------ rendus
def etiqueter(f):
    """Matière, plan et numéro de forme de chaque pixel (le plus proche l'emporte)."""
    mat, plan, num = np.full((H, W), "", object), np.full((H, W), "", object), np.full((H, W), -1)
    for k, fo in enumerate(f):
        m = fo["masque"]
        mat[m], plan[m], num[m] = fo["matiere"], fo["plan"], k
        for nom, zone in fo["zones"].items():  # matières et motifs de la forme, dans l'ordre
            mat[m & zone] = nom
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


def miettes(img, n=4, bord=CONTOUR):
    """Les groupes de moins de n pixels isolés par les contours ou les traits de couleur bord
    (voisins par un côté) prennent cette couleur : sinon un pixel flotte, seul, entre deux
    traits (entre la lèvre et le bord du menton, par exemple)."""
    plein = (img != "") & (img != CONTOUR) & (img != bord)
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
                img[p] = bord
    return img


def rendu_oeil(img, oeil, couleurs, ferme_couleur):
    if oeil is None:  # un support (branche, feuille) n'a pas d'œil
        return img
    m, u, v = oeil["masque"], oeil["u"], oeil["v"]
    if oeil.get("autre") is not None and not oeil["ferme"]:
        # l'autre œil est une bosse de peau ; on n'en voit au plus qu'un mince croissant de globe
        img[oeil["autre"] & (img != "")] = couleurs["iris"]
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


def tracer(img, traits, num, oeil, couleur=None):
    """Les traits (bouche, narine, tympan, plis) : d'une couleur donnée (ébauche, croquis) ou,
    sans couleur donnée, de la couleur du dessous assombrie (aplats, détails)."""
    hors_oeil = ~oeil["masque"] if oeil else True
    for nom, m in traits.items():
        ici = m & (num >= 0) & hors_oeil & (img != CONTOUR)
        if couleur is None and nom in TRAITS_MATIERE:
            img[ici] = MATIERES[TRAITS_MATIERE[nom]]
            continue
        for j, i in zip(*np.nonzero(ici), strict=False):
            img[j, i] = couleur or assombrir(img[j, i])
    return miettes(img, bord=couleur) if couleur else img


def ebauche(f, traits, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom, coul in SILHOUETTE.items():
        img[plan == nom] = coul
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil, CONTOUR)
    return rendu_oeil(
        img,
        oeil,
        {"iris": SILHOUETTE["globe"], "pupille": SILHOUETTE["pupille"]},
        SILHOUETTE["corps"],
    )


def croquis(f, traits, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom in ("fond", "corps", "devant"):
        img[plan == nom] = CROQUIS[nom]
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil, CROQUIS["trait"])
    return rendu_oeil(
        img,
        oeil,
        {"iris": CROQUIS["globe"], "pupille": CROQUIS["pupille"], "cercle": CROQUIS["trait"]},
        CROQUIS["corps"],
    )


def aplats(f, traits, motifs, oeil):
    mat, plan, num = etiqueter(f)
    img = np.full((H, W), "", object)
    for nom, coul in (MATIERES | MOTIFS).items():  # matières, et motifs posés en zones
        img[mat == nom] = coul
    for nom, m in motifs.items():  # motifs communs à toutes les formes visibles
        img[m & (num >= 0)] = MOTIFS[nom]
    img = contour(img, f, num)
    img = tracer(img, traits, num, oeil)
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


def details(f, traits, motifs, oeil):
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
        # le gradient du masque flouté pointe vers l'intérieur : un bord tourné vers la lumière
        # (LUMIERE, d'où elle vient) a un gradient opposé à elle
        face = -(gx * lx + gy * ly)  # bord tourné vers la lumière : positif
        ici = num == k
        lum[ici] = (0.55 + 4.0 * face - 0.18 * ((Y - ORIGINE[1]) / UNITES[1] - 0.4))[ici]
    plaques = np.clip(np.floor(_flou(np.random.default_rng(3).random((H, W)), 4) * 6 - 2), -1, 1)
    grain = (np.random.default_rng(5).random((H, W)) - 0.5) * 0.08
    img = np.full((H, W), "", object)
    couleurs = MATIERES | MOTIFS
    zone = mat.copy()
    for nom, m in motifs.items():
        zone[m & (num >= 0)] = nom
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
    img = tracer(img, traits, num, oeil)
    if CONTOUR_TEINTE:
        img = teinter_contour(img)
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


def assombrir(hexa, f=0.55):
    """La même teinte, plus sombre et un peu plus saturée : traits et contours teintés."""
    r, g, b = (int(hexa[i : i + 2], 16) / 255 for i in (1, 3, 5))
    h, clarte, s = colorsys.rgb_to_hls(r, g, b)
    return hexa_de(*colorsys.hls_to_rgb(h, clarte * f, min(s * 1.15, 1)))


TEINTES = set()  # couleurs des contours teintés : coder() les range avec le contour


def teinter_contour(img):
    """Chaque pixel de contour prend la couleur, assombrie, du premier voisin qu'il borde."""
    out = img.copy()
    for j, i in zip(*np.nonzero(img == CONTOUR), strict=False):
        for v in ((j, i + 1), (j, i - 1), (j + 1, i), (j - 1, i)):
            if 0 <= v[0] < H and 0 <= v[1] < W and img[v] not in ("", CONTOUR, CONTOUR_CLAIR):
                out[j, i] = assombrir(img[v], 0.4)
                TEINTES.add(out[j, i])
                break
    return out


RENDUS = {"ebauche": ebauche, "croquis": croquis, "aplats": aplats, "details": details}


# ------------------------------------------------------------------ sortie
def coder(images):
    """Images de couleurs hexa → palette de lettres (K et k réservés au contour) et plages."""
    couleurs = sorted(
        {c for im in images.values() for c in im.ravel() if c} - {CONTOUR, CONTOUR_CLAIR}
    )
    lettres = list("ABCDEFGHIJLMNOPQRSTUVWXYZabcdefghijlmnopqrstuvwxyz")
    lettres += [chr(0x100 + i) for i in range(max(0, len(couleurs) - len(lettres)))]
    palette = {"K": CONTOUR, "k": CONTOUR_CLAIR, **dict(zip(lettres, couleurs, strict=False))}
    # lettres du contour (K, k et les contours teintés) : outils.py controle les ignore
    contour_lettres = "Kk" + "".join(lt for lt, c in palette.items() if c in TEINTES)
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
    return {
        "palette": palette,
        "contour": contour_lettres,
        "largeur": W,
        "hauteur": H,
        "images": out,
    }


def squelette():
    """Les repères et les os en pixels de la toile, pour la planche et outils.py lecture. Avec
    MIROIR, l'image est retournée : les côtés de l'animal s'échangent (_g ↔ _d)."""

    def px(nom):
        x, y = SQUELETTE[nom][0] - ORIGINE[0], SQUELETTE[nom][1] - ORIGINE[1]
        return [round((UNITES[0] - x if MIROIR else x) * ECHELLE, 1), round(y * ECHELLE, 1)]

    def cote(nom):
        if MIROIR and nom[-2:] in ("_g", "_d"):
            return nom[:-2] + ("_d" if nom.endswith("_g") else "_g")
        return nom

    plans = {n: "corps" for n in SQUELETTE}
    os_liste = [[a, b, "corps"] for axe in AXES for a, b in zip(axe, axe[1:], strict=False)]
    for chaine, plan, bouts in MEMBRES.values():
        for n in chaine + bouts:
            plans[n] = plan
        os_liste += [[a, b, plan] for a, b in zip(chaine, chaine[1:], strict=False)]
        os_liste += [[chaine[-1], b, plan] for b in bouts]
    return {
        "reperes": {cote(n): px(n) + [plans[n]] for n in SQUELETTE},
        "os": [[cote(a), cote(b), p] for a, b, p in os_liste],
    }


def produire(etape):
    if etape == "animation":
        images = {}
        for g in (0, 1):
            for s in (0, 1):
                for c in (0, 1):
                    for t in (-1, 0, 1):
                        images[f"{g}{s}{c}{t + 1}"] = details(*formes(g * 0.9, s * 0.7, bool(c), t))
        return coder(images)
    sprite = coder({"0001": RENDUS[etape](*formes())})
    if etape in ("ebauche", "croquis") and SQUELETTE:
        # la planche trace le squelette ; visible d'emblée à l'ébauche, sur demande au croquis
        sprite |= squelette() | {"squelette_visible": etape == "ebauche"}
    return sprite


if __name__ == "__main__":
    etape = sys.argv[1] if len(sys.argv) > 1 else "ebauche"
    sortie = Path(sys.argv[2]) if len(sys.argv) > 2 else ICI / f"{NOM}_{etape}.json"
    sprite = produire(etape)
    sortie.write_text(json.dumps(sprite, separators=(",", ":")), encoding="utf-8")
    print(f"{sortie.name} : {len(sprite['images'])} image(s), {len(sprite['palette'])} couleurs")
