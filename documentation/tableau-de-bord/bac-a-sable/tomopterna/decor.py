"""Le décor de la scène : la photo entière, repeinte au pixel, en un calque.

    uv run python documentation/tableau-de-bord/bac-a-sable/tomopterna/decor.py [apercu.png]

D'après la photo d'Olivier Louguet. De haut en bas : l'herbe floue de l'arrière-plan, la berge
sombre derrière la branche, l'eau lointaine aux reflets étirés, puis l'eau proche, olive et
brune, où la page dessine le reflet des grenouilles et de la branche (calculé depuis les
sprites : miroir assombri, teinté, qui ondule).

Rien n'est copié de la photo. On en a relevé une grille de couleurs grossière (cases de 16
pixels de toile, grenouilles, branche et reflets masqués) et, à la main sur la photo quadrillée,
les brins d'herbe qu'on y distingue : leur tracé (penchés à gauche ou à droite, courbés, cassés,
un long brin presque couché sous la branche), leur largeur et leur éclat, vert ou paille. Le
décor est peint d'après ces relevés : un champ de couleur adouci, les brins en traits flous,
et un grain qui suit, en chaque point, la direction des brins voisins (des traînées intégrées
le long de ce champ de directions), si bien que l'herbe floue part dans tous les sens comme
sur la photo, et non plus seulement à la verticale ; un grain horizontal sur l'eau. Puis une
palette courte et un tramage Bayer 4 × 4, comme les grenouilles. Le hasard est tiré d'une
graine fixe : le décor est le même à chaque fois.
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ICI = Path(__file__).resolve().parent
W, H = 448, 303
CASE = 16  # côté d'une case de la grille relevée, en pixels de toile
X, Y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
_B = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], float)
BAYER = (_B[Y.astype(int) % 4, X.astype(int) % 4] + 0.5) / 16
HERBE_BAS = 118  # sous cette rangée, plus d'herbe : la berge, puis l'eau
EAU_HAUT = 128  # l'eau lointaine commence (reflets étirés)
EAU_PROCHE = 160  # l'eau proche, où tombent les reflets
NUANCES = 30  # couleurs de chacune des deux palettes du décor (l'herbe, l'eau)

# ------------------------------------------------------------------ relevés sur la photo
# GRILLE : une rangée par case de 16 pixels (de haut en bas), 29 couleurs par rangée (RRGGBB).
GRILLE = [
    "606E4A607149666F4A6E7A575A704456674259684562704B68785055633F53623C4A59355C6D48647153585E45"
    "606A486B764C606C435E6B416E7E50707F555462445B69483841294850345B68425565414B58364B5836",
    "4E5B3C5A6643525A3B595E4052643C4F5D3A4B5738535C3E62734B535F3D4957363F4C2D47583650643B4B5237"
    "5D69455A623F505A37485A32607346656E49394A2E4751343A432C4249304955344958394954334A5634",
    "404B304D54384B503944462F424F2D475533525C3B4A4B3056613D4C5B384452333E4B2C4A5A3643552E495836"
    "4F593752613D4B57343D4C284E633C56623E4E613D3E4B2C3E472E3E482D4855344A58364F5A384C5836",
    "434C31404A314D513E4447313E4A2C49533467674A46442E3D4628343E2439472C354027666A485865424E5C37"
    "39402644502F414F2C44552D51603B5564414858384552303F4A2D3D4B2D4352304855314952324A5534",
    "3B40293C422C3C412C3E3F2A3840264D53384F503233321E3C41222A2E19313A223C4128504A2F545A3C3A4427"
    "3942263E492B3F4C2B4554304D5C383C48303A422C414B31343B233A40283F472B3B44254C4C334B5033",
    "3135213B3D28282A1B2B2E1B32371F353A243F3F2830331E363A2030341C31371F40442B55553C3A3F27343721"
    "46472C42482B414A2B434F2D4855323E472C38382443432D31311D3538223A3F263A412543462C474B30",
    "37392437362237362329281635351F302D1A2E2A182F2E1B32341D31341D31351E3D40294547313D3F2A3B3D26"
    "40422941452A41472A424B2C45502F414B2E3C41293B3C2534311A3636212F2C1B2F301C3431203D3E28",
    "393723302A1B36322134311E383822332D1A302B192F2D1A31301B31321C31341D373A233E402A3D3F2A3C3E28"
    "3C3B2438321D37311E3C3E253A3C222E271437311C33301B251F112924142821132F2B19463E28413E28",
    "383220352C1E37311F433F2A3D3C2738342034301C322E1B312F1B36311B41331B341A10423E2837321E322C18"
    "312916302A18332D1B423C25382E192B251436301A2F2A182A23132822132923142D2917423C28413D28",
    "383220362F1F36301F3B37243A3823302F1A342E1B423A2338301B352715322413372C183C37223A341E342E1A"
    "36321E3D392239341F454027312B19342E1E312918322A192921132D24142B231438301C3B32203E3724",
    "38322037301F37301F393321393522302E1B28251539341E252012211B0F241F113028152E26152A25142C2916"
    "352D193F382244422646452A3839212F301C3C3924352E1D2C2314352B1C35291A342E1B3C36213D3722",
    "38322037311F37301F38322039342134311E2E2B19332F1B2C27162621122520113027153D2D1A302916372F19"
    "3F341F483A21413B224C4C30403F2747442B3C34213D331F392D1A392E1C352C1B363322342E1E383220",
    "4944324C4632413B283D36243B3523373320332F1D332F1C302B192B26162823142C2514342917322916342C17"
    "39301B40351E403820464228443F26483C23332B1A372C1A372D1C3833213230213333223A3B26393623",
    "58513B554D384D4631453D2A463C293F342139311F36301D332E1B2F2A192B26162C2615302716312816332A17"
    "362D193B311B3E341D423B22433D24463C233C331E392F1C382E1C3E36244A432F3B3623403E283C3A25",
    "4E4834514833473C294238264C402D4137264236233C332037301E332D1B2F2A192D28172F2717302816312917"
    "342B18372E193B311B3E361F413921433B224037203C331E3A311D3C3320433B273F38253F3B263E3A26",
    "443B27413926393321443A284139273A31213D3321483D263F362239321E342E1B312B193228173627174A3620"
    "3F301C3B2F1B3B301B3D331D3F361F4138214038213E351F3C331E3C331F3F37233F38243F39253E3A25",
    "36301E3A35223D3825423B284B412D504530594D3449402A4E422C433A253C3420362F1D392F1D45352235281A"
    "3A2C1B3A2D1B3B2F1B3C311C3D331E3C331D372D1A3F311E3C301F302819251D132C261A2C251A352F1F",
    "4137244C442E4A412D4D442F4E4632534D384D47344B44324A44313D3528393124292017362C233F34263B3024"
    "3A2E1F3A2D1D3D301D43341F4B3F2A433B29322B1B3D3320473E2A46402B30281C312C1F2F281C322B1D",
    "423A28504834494230453F2E403C2C3935273833263832234137263931204736234232203329194136234F412B"
    "43311F3A2E1D40311F34291A50432B584E32423E283D382531291A433A283E38263E3A293D3625373121",
]
# BRINS : relevés à la main sur la photo quadrillée (pixels de toile). Chaque brin : ses points,
# de haut en bas, sa largeur, son éclat (écart de clarté, négatif pour un brin sombre) et sa
# teinte (vert tendre ou paille, l'herbe sèche)
BRINS = [
    ([(6, 8), (8, 40), (12, 66), (22, 94), (30, 118)], 5, 12, "vert"),
    ([(26, 0), (24, 30), (28, 52)], 4, 9, "vert"),
    ([(18, 62), (26, 84), (38, 104), (46, 120)], 4, -10, "vert"),
    ([(40, 40), (42, 56), (44, 72)], 9, 24, "paille"),
    ([(56, 22), (48, 40), (50, 58), (58, 76)], 4, -11, "vert"),
    ([(66, 0), (68, 30), (66, 60), (62, 92)], 6, 11, "vert"),
    ([(84, 10), (80, 40), (74, 70)], 4, 7, "vert"),
    ([(104, 36), (98, 52), (90, 70), (84, 94)], 7, 22, "paille"),
    ([(104, 56), (116, 58), (130, 65)], 5, 18, "paille"),
    ([(126, 0), (128, 30), (124, 60)], 6, 10, "vert"),
    ([(146, 24), (146, 50), (143, 80)], 5, -12, "vert"),
    ([(162, 0), (164, 30), (168, 56)], 6, 9, "vert"),
    ([(182, 10), (178, 40), (176, 60)], 4, -9, "vert"),
    ([(198, 34), (206, 52), (212, 70), (214, 88)], 6, 24, "paille"),
    ([(236, 8), (230, 36), (228, 62)], 5, 12, "vert"),
    ([(252, 0), (246, 26), (240, 52)], 4, 10, "vert"),
    ([(268, 4), (274, 30), (278, 56)], 5, 9, "vert"),
    ([(288, 0), (290, 20), (296, 44)], 4, -9, "vert"),
    ([(304, 0), (302, 30), (304, 62)], 6, 8, "vert"),
    ([(320, 0), (322, 36), (324, 72)], 5, 15, "vert"),
    ([(344, 6), (350, 36), (356, 64)], 4, -9, "vert"),
    ([(372, 0), (368, 30), (362, 58)], 5, 9, "vert"),
    ([(392, 10), (396, 40), (394, 70)], 4, 7, "vert"),
    ([(412, 0), (406, 24), (400, 50)], 4, -9, "vert"),
    ([(434, 30), (432, 60), (430, 96)], 7, 24, "paille"),
    ([(442, 0), (444, 30)], 4, 8, "vert"),
    # le long brin presque couché sous la branche, à droite, qui monte vers la droite
    ([(334, 128), (360, 121), (388, 114), (416, 108)], 4, 14, "vert"),
]
TEINTES = {"vert": (0.8, 1.0, 0.62), "paille": (0.96, 1.0, 0.8)}


# ------------------------------------------------------------------ peinture
def lisse(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def flou(a, r):
    for axe in (0, 1):
        p = np.pad(a, [(r, r) if k == axe else (0, 0) for k in range(a.ndim)], mode="edge")
        a = sum(np.take(p, range(k, k + a.shape[axe]), axis=axe) for k in range(2 * r + 1))
        a = a / (2 * r + 1)
    return a


def bruit(ex, ey, graine):
    """Bruit de valeur lissé, 0..1, aux échelles ex (en largeur) et ey (en hauteur)."""
    g = np.random.default_rng(graine).random((int(H / ey) + 3, int(W / ex) + 3))
    gx, gy = X / ex, Y / ey
    i, j = gx.astype(int), gy.astype(int)
    tx, ty = gx - i, gy - j
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    haut = g[j, i] * (1 - tx) + g[j, i + 1] * tx
    bas = g[j + 1, i] * (1 - tx) + g[j + 1, i + 1] * tx
    return haut * (1 - ty) + bas * ty


def champ():
    """La grille relevée, étendue à la toile (bilinéaire entre les centres des cases), adoucie."""
    g = np.array(
        [
            [[int(r[k + c : k + c + 2], 16) for c in (0, 2, 4)] for k in range(0, len(r), 6)]
            for r in GRILLE
        ],
        float,
    )
    gh, gw = g.shape[:2]
    gx = np.clip((X - CASE / 2) / CASE, 0, gw - 1.001)
    gy = np.clip((Y - CASE / 2) / CASE, 0, gh - 1.001)
    i, j = gx.astype(int), gy.astype(int)
    tx, ty = (gx - i)[..., None], (gy - j)[..., None]
    c = (g[j, i] * (1 - tx) + g[j, i + 1] * tx) * (1 - ty) + (
        g[j + 1, i] * (1 - tx) + g[j + 1, i + 1] * tx
    ) * ty
    return flou(c, 4)


def courbe(pts, n=6):
    """Les points d'un brin, reliés par une courbe lisse (Catmull-Rom), n pas par segment."""
    p = np.array([pts[0]] + list(pts) + [pts[-1]], float)
    out = []
    for k in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[k - 1], p[k], p[k + 1], p[k + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(
                0.5
                * (
                    2 * p1
                    + (p2 - p0) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                    + (3 * p1 - p0 - 3 * p2 + p3) * t**3
                )
            )
    out.append(p[-2])
    return np.array(out)


def brins():
    """Les brins relevés, en traits flous : clarté (positive ou négative) par teinte, et le
    champ de directions qu'ils donnent à l'herbe autour d'eux (cos 2θ, sin 2θ et poids)."""
    eclat = {t: np.zeros((H, W)) for t in TEINTES}
    champ_d = np.zeros((3, H, W))
    for pts, larg, e, teinte in BRINS:
        c = courbe(pts)
        d = np.diff(c, axis=0)
        long = np.concatenate([[0], np.cumsum(np.hypot(*d.T))])
        x0, x1 = int(c[:, 0].min() - larg - 3), int(c[:, 0].max() + larg + 4)
        y0, y1 = int(c[:, 1].min() - larg - 3), int(c[:, 1].max() + larg + 4)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        xx, yy = X[y0:y1, x0:x1], Y[y0:y1, x0:x1]
        # distance au tracé et abscisse le long du brin, par le plus proche des points
        dist = np.full(xx.shape, 1e9)
        abs_ = np.zeros(xx.shape)
        for k, (cx, cy) in enumerate(c):
            dd = np.hypot(xx - cx, yy - cy)
            plus = dd < dist
            dist[plus], abs_[plus] = dd[plus], long[k]
        fondu = np.clip(np.minimum(abs_, long[-1] - abs_) / 6 + 0.25, 0, 1)
        prof = np.exp(-((dist / (0.5 * larg + 1.0)) ** 2) * 1.6) * fondu
        zone = eclat[teinte][y0:y1, x0:x1]
        if e > 0:
            zone[:] = np.maximum(zone, e * prof)
        else:
            zone[:] = np.minimum(zone, e * prof)
        # la direction de chaque bout de brin, en angle doublé (un brin n'a pas de sens)
        for k in range(len(d)):
            a = np.arctan2(d[k, 0], d[k, 1])  # depuis la verticale
            cx, cy = c[k] + d[k] / 2
            w = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 16.0**2))
            champ_d += [w * np.cos(2 * a), w * np.sin(2 * a), w]
    return {t: flou(v, 1) for t, v in eclat.items()}, champ_d


def directions(champ_d):
    """En chaque point, l'angle des brins (radians, depuis la verticale) : la moyenne de ceux
    des brins voisins, la verticale où il n'y en a pas, et un peu de flottement."""
    cos2, sin2, poids = champ_d
    w0 = 0.12  # la verticale, là où aucun brin ne compte
    a = 0.5 * np.arctan2(sin2, cos2 + w0)
    return a + (bruit(48, 40, 21) - 0.5) * 0.5


def trainees(angle, ex, ey, longueur, graine):
    """Un grain étiré le long des brins : un bruit fin, moyenné le long de la ligne de champ
    qui passe par chaque pixel (intégrale sur ± longueur pixels)."""
    n = bruit(ex, ey, graine)
    somme, poids = np.zeros((H, W)), 0.0
    for sens in (1, -1):
        x, y = X.copy(), Y.copy()
        for k in range(longueur):
            i = np.clip(np.round(x).astype(int), 0, W - 1)
            j = np.clip(np.round(y).astype(int), 0, H - 1)
            if k or sens == 1:
                w = 1 - k / longueur
                somme += w * n[j, i]
                poids += w
            a = angle[j, i]
            x += sens * np.sin(a)
            y += sens * np.cos(a)
    return somme / poids


# le fond de l'eau, vu à travers elle (relevé sur la photo : des taches claires, sable, feuilles
# et galets, et des creux sombres entre elles, aplaties par la perspective ; petites, serrées et
# nettes au loin, larges et floues de près) : par sorte, la probabilité, l'écart de couleur
LIT = {
    "sable": (0.36, (34, 28, 14)),  # feuilles et sable ocre, éclairés
    "creux": (0.44, (-20, -18, -12)),  # les creux et l'ombre entre eux
    "algue": (0.13, (-4, 10, -1)),  # un peu de vert
    "galet": (0.07, (18, 20, 20)),  # galets gris
}


def lit():
    """L'écart de couleur (RGB) que le fond de l'eau ajoute, rangée par rangée : des taches
    elliptiques couchées, d'autant plus grandes, plus hautes et plus floues qu'elles sont
    proches (p : 0 au loin, 1 au bas de la toile)."""
    out = np.zeros((H, W, 3))
    rng = np.random.default_rng(31)
    sortes = list(LIT)
    proba = np.array([LIT[k][0] for k in sortes])
    y = float(EAU_HAUT - 2)
    while y < H + 4:
        p = np.clip((y - EAU_HAUT) / (H - EAU_HAUT), 0, 1)
        larg = 1.8 + 12 * p**1.4  # demi-largeur d'une tache
        haut = max(0.5, larg * (0.13 + 0.15 * p))  # demi-hauteur : la perspective l'écrase
        flou_ = 0.7 + 3.2 * p  # bord adouci (de près, hors de la mise au point)
        for _ in range(int(W / (larg * 1.9))):
            cx = rng.uniform(-larg, W + larg)
            cy = y + rng.uniform(-0.5, 0.5) * haut
            lx, hy = larg * rng.uniform(0.55, 1.4), haut * rng.uniform(0.7, 1.3)
            sorte = sortes[rng.choice(len(sortes), p=proba)]
            force = rng.uniform(0.25, 0.7)
            x0, x1 = int(max(cx - lx - 4, 0)), int(min(cx + lx + 5, W))
            y0, y1 = int(max(cy - hy - 3, 0)), int(min(cy + hy + 4, H))
            if x1 <= x0 or y1 <= y0:
                continue
            xx, yy = X[y0:y1, x0:x1], Y[y0:y1, x0:x1]
            r = np.hypot((xx - cx) / lx, (yy - cy) / hy)
            m = np.clip((1 - r) * lx / flou_ + 0.5, 0, 1)  # 1 dedans, fondu sur le bord
            out[y0:y1, x0:x1] += m[..., None] * force * np.array(LIT[sorte][1], float)
        y += max(0.8, haut * 1.1)
    return out


def palette(c, n, graine=7):
    """Une palette courte tirée du décor lui-même (k-moyennes sur un échantillon)."""
    px = c.reshape(-1, 3)
    ech = px[np.random.default_rng(graine).choice(len(px), 24000, replace=False)]
    lum = ech @ [0.3, 0.59, 0.11]
    centres = ech[np.argsort(lum)[np.linspace(0, len(ech) - 1, n).astype(int)]]
    for _ in range(18):
        lab = np.argmin(((ech[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        for k in range(n):
            if (lab == k).any():
                centres[k] = ech[lab == k].mean(0)
    return np.round(centres[np.argsort(centres @ [0.3, 0.59, 0.11])])


def fond():
    c = champ()
    lum = c @ [0.3, 0.59, 0.11]
    herbe = 1 - lisse(HERBE_BAS - 14, HERBE_BAS, Y)
    eau = lisse(EAU_HAUT - 6, EAU_HAUT + 4, Y)
    # les brins relevés, clairs ou sombres, vert tendre ou paille
    eclats, champ_d = brins()
    for teinte, e in eclats.items():
        c += (1.5 * e)[..., None] * np.array(TEINTES[teinte])
    # le grain de l'herbe floue : des traînées qui suivent la direction des brins voisins ;
    # sur l'eau : reflets étirés au loin, taches plus courtes de près (le fond, les rides)
    angle = directions(champ_d)
    fin, large = trainees(angle, 5.0, 5.0, 18, 11), trainees(angle, 12.0, 12.0, 26, 12)
    v = (fin - 0.5) * 24 + (large - 0.5) * 44
    v = flou(v, 1)
    # sur l'eau, au loin, les reflets étirés de la berge, plus discrets que le fond qu'on voit
    # à travers l'eau
    loin = (bruit(26, 1.6, 13) - 0.5) * 18 * (1 - lisse(EAU_PROCHE, EAU_PROCHE + 30, Y))
    d = v * herbe + loin * eau
    c += d[..., None] * np.array([0.9, 1.0, 0.75])
    # l'eau, transparente : on y voit le fond, plus net au milieu, voilé au loin par les
    # reflets et de près par la profondeur ; l'eau elle-même, un peu verte
    p = np.clip((Y - EAU_HAUT) / (H - EAU_HAUT), 0, 1)
    voile = (0.5 + 0.5 * lisse(0.0, 0.2, p)) * (1 - 0.4 * lisse(0.55, 1.0, p))
    c += (lit() * (eau * voile)[..., None]).clip(-60, 70)
    c += (eau * (0.6 + 0.4 * p))[..., None] * np.array([-3.0, 3.0, 1.0])
    # l'herbe reste olive : là où la grille tire sur le roux (des brins secs flous), le rouge
    # ne dépasse pas le vert
    c[..., 0] = np.where(herbe > 0, np.minimum(c[..., 0], 0.94 * c[..., 1]), c[..., 0])
    # un peu plus lumineux et franc que la photo, pour aller avec les grenouilles
    moy = c.mean(-1, keepdims=True)
    c = moy + (c - moy) * 1.18
    c = c * 1.06 + 4
    c = np.clip(c, 0, 255)
    # tramage ordonné : on décale chaque pixel d'un seuil Bayer avant de prendre la nuance la
    # plus proche ; les dégradés se font en damier fin, sans bruit. Une palette courte pour
    # l'herbe et la berge, une pour l'eau
    seuil = (BAYER - 0.5)[..., None] * 14
    out = np.zeros((H, W, 3), np.uint8)
    for zone in (Y < EAU_HAUT - 4, Y >= EAU_HAUT - 4):
        pal = palette(c[zone][None], NUANCES)
        k = np.argmin((((c[zone] + seuil[zone])[:, None, :] - pal) ** 2).sum(-1), axis=-1)
        out[zone] = pal[k]
    return out, lum


def png(rgb):
    tampon = io.BytesIO()
    Image.fromarray(rgb).save(tampon, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(tampon.getvalue()).decode()


def calques():
    rgb, _ = fond()
    return {"fond": png(rgb)}, rgb


if __name__ == "__main__":
    _, rgb = calques()
    nuances = len(np.unique(rgb.reshape(-1, 3), axis=0))
    print(f"fond : {W} × {H}, {nuances} couleurs")
    if len(sys.argv) > 1:
        Image.fromarray(rgb).resize((W * 3, H * 3), Image.NEAREST).save(sys.argv[1])
