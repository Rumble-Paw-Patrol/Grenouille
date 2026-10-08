"""Le décor de la scène : la photo entière, repeinte au pixel, en un calque.

    uv run python documentation/tableau-de-bord/bac-a-sable/tomopterna/decor.py [apercu.png]

D'après la photo d'Olivier Louguet. De haut en bas : les roseaux flous de l'arrière-plan, la
berge sombre derrière la branche, l'eau lointaine aux reflets étirés, puis l'eau proche, olive
et brune, où la page dessine le reflet des grenouilles et de la branche (calculé depuis les
sprites : miroir assombri, teinté, qui ondule).

Rien n'est copié de la photo. On en a relevé une grille de couleurs grossière (cases de 16
pixels de toile, grenouilles, branche et reflets masqués) et la place des roseaux clairs
(colonnes plus claires que leur voisinage, par bandes de 12 pixels, chaînées en tiges), et le
décor est peint d'après ces relevés : un champ de couleur adouci, les tiges en bandes claires
aux bords fondus, un grain vertical dans les roseaux et horizontal sur l'eau, puis une palette
courte et un tramage Bayer 4 × 4, comme les grenouilles. Le hasard est tiré d'une graine fixe :
le décor est le même à chaque fois.
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
ROSEAUX_BAS = 118  # sous cette rangée, plus de roseaux : la berge, puis l'eau
EAU_HAUT = 128  # l'eau lointaine commence (reflets étirés)
EAU_PROCHE = 160  # l'eau proche, où tombent les reflets
NUANCES = 30  # couleurs de la palette du décor

# ------------------------------------------------------------------ relevés sur la photo
# GRILLE : une rangée par case de 16 pixels (de haut en bas), 29 couleurs par rangée (RRGGBB).
# ROSEAUX : une tige par liste, (haut de la bande de 12 px, centre, largeur, éclat), en pixels
# de toile ; l'éclat est l'écart de luminance (0-255) à la moyenne locale.
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
ROSEAUX = [
    [(0, 65.0, 10, 9), (12, 69.0, 8, 10), (24, 73.5, 5, 7)],
    [
        (0, 131.5, 11, 14),
        (12, 133.0, 12, 16),
        (24, 134.0, 12, 19),
        (36, 135.5, 9, 13),
        (48, 136.0, 8, 12),
        (60, 136.5, 7, 9),
        (72, 135.5, 5, 7),
    ],
    [(0, 208.5, 13, 13), (12, 210.5, 17, 10)],
    [(0, 270.0, 12, 8), (12, 274.0, 10, 12), (24, 273.5, 7, 6)],
    [(0, 312.5, 5, 6)],
    [(0, 333.0, 8, 9)],
    [(0, 362.0, 14, 13), (12, 363.5, 13, 19), (24, 363.5, 7, 9)],
    [(0, 413.5, 13, 9)],
    [(12, 25.5, 7, 8), (24, 22.0, 10, 13), (36, 21.5, 7, 12)],
    [
        (12, 160.0, 10, 9),
        (24, 157.5, 7, 11),
        (36, 158.5, 3, 6),
        (48, 158.0, 6, 9),
        (60, 158.0, 2, 6),
    ],
    [(12, 325.0, 18, 16)],
    [(12, 399.0, 4, 7)],
    [(24, 2.0, 4, 7)],
    [
        (24, 107.5, 5, 8),
        (36, 103.5, 11, 18),
        (48, 100.5, 13, 32),
        (60, 97.0, 16, 16),
        (72, 98.0, 20, 9),
        (84, 93.0, 4, 6),
    ],
    [(24, 218.5, 5, 7)],
    [(24, 242.0, 12, 19), (36, 239.0, 10, 20), (48, 235.5, 9, 13), (60, 234.5, 3, 9)],
    [(24, 319.5, 13, 27), (36, 318.5, 11, 29), (48, 319.5, 9, 18), (60, 319.0, 10, 13)],
    [(36, 45.5, 7, 8), (48, 45.5, 11, 13), (60, 42.0, 10, 12)],
    [(36, 204.5, 11, 19), (48, 206.0, 16, 26)],
    [(36, 266.0, 12, 11)],
    [(36, 343.5, 9, 8)],
    [(48, 256.5, 5, 20)],
    [(48, 302.0, 4, 8), (60, 302.0, 4, 15)],
    [(48, 441.0, 4, 8), (60, 438.5, 9, 31)],
    [(60, 212.0, 12, 31), (72, 210.5, 11, 24), (84, 206.0, 14, 16), (96, 205.5, 9, 10)],
    [(60, 354.5, 9, 8), (72, 355.5, 13, 12), (84, 352.0, 4, 10)],
    [(72, 23.0, 6, 9), (84, 26.0, 10, 14), (96, 29.0, 8, 8)],
    [(72, 244.0, 10, 13), (84, 247.5, 9, 13), (96, 247.0, 2, 15)],
    [(72, 432.5, 5, 10)],
    [(84, 75.0, 6, 8), (96, 74.0, 10, 12), (108, 76.0, 6, 7)],
    [(84, 110.5, 9, 10)],
    [(96, 404.5, 7, 6)],
    [(96, 432.5, 9, 32), (108, 433.5, 11, 16), (120, 435.5, 13, 19)],
    [(108, 373.0, 8, 12)],
    [(120, 10.0, 6, 8)],
    [(120, 94.0, 4, 7)],
    [(120, 234.5, 3, 8)],
    [(120, 255.5, 3, 8)],
    [(120, 297.0, 10, 11)],
    [(120, 339.5, 13, 9)],
]


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


def tiges():
    """Les roseaux clairs : des bandes verticales aux bords fondus, qui s'éteignent en haut et
    en bas, d'éclat et de largeur relevés bande par bande."""
    t = np.zeros((H, W))
    for tige in ROSEAUX:
        ys = np.array([y + 6 for y, *_ in tige], float)
        cs, ls, es = (np.array([v[k] for v in tige], float) for k in (1, 2, 3))
        y0, y1 = ys[0] - 10, ys[-1] + 10
        for y in range(max(0, int(y0)), min(ROSEAUX_BAS + 6, int(y1) + 1)):
            c, larg, e = (np.interp(y, ys, v) for v in (cs, ls, es))
            fondu = min(1, (y - y0) / 8, (y1 - y) / 8)
            prof = np.exp(-(((X[y] - c) / (0.5 * larg + 1.2)) ** 2) * 1.6)
            t[y] = np.maximum(t[y], e * fondu * prof)
    return flou(t, 1)


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
    roseaux = 1 - lisse(ROSEAUX_BAS - 14, ROSEAUX_BAS, Y)
    eau = lisse(EAU_HAUT - 6, EAU_HAUT + 4, Y)
    # tiges claires, vert tendre
    t = tiges() * 0.8
    c += t[..., None] * np.array([0.8, 1.0, 0.62])
    # grain des roseaux flous : vertical ; sur l'eau : reflets étirés au loin, taches plus
    # courtes de près (le fond de l'eau, les rides)
    v = (bruit(5, 38, 11) - 0.5) * 16 + (bruit(17, 70, 12) - 0.5) * 14
    loin = (bruit(26, 1.6, 13) - 0.5) * 30 * (1 - lisse(EAU_PROCHE, EAU_PROCHE + 30, Y))
    pres = (bruit(13, 3.2, 14) - 0.5) * 20 + (bruit(40, 9, 15) - 0.5) * 12
    pres *= lisse(EAU_PROCHE - 10, EAU_PROCHE + 20, Y)
    d = v * roseaux + (loin + pres) * eau
    c += d[..., None] * np.array([0.9, 1.0, 0.75])
    # un peu plus lumineux et franc que la photo, pour aller avec les grenouilles
    moy = c.mean(-1, keepdims=True)
    c = moy + (c - moy) * 1.18
    c = c * 1.06 + 4
    c = np.clip(c, 0, 255)
    pal = palette(c, NUANCES)
    # tramage ordonné : on décale chaque pixel d'un seuil Bayer avant de prendre la nuance la
    # plus proche ; les dégradés se font en damier fin, sans bruit
    seuil = (BAYER - 0.5)[..., None] * 14
    k = np.argmin((((c + seuil)[..., None, :] - pal) ** 2).sum(-1), axis=-1)
    return pal[k].astype(np.uint8), lum


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
