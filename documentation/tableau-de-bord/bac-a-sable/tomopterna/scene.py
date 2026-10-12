"""La scène animée : les deux C. tomopterna au-dessus de l'eau, avec leur reflet.

    python3 documentation/tableau-de-bord/bac-a-sable/tomopterna/scene.py [SORTIE.html]

D'après la photo d'Olivier Louguet. Les deux grenouilles sont celles dessinées pas à pas et
validées (../tomopterna-gauche/gauche.py et ../tomopterna-droite/droite.py) : la page en reprend
les images d'animation (48 chacune : gorge, flanc, œil ouvert, mi-clos ou fermé, tête, et le
regard qui se détourne), avec l'œil en demi-pixels. Puis la branche entière et son ménisque
(branche.py) et le décor (decor.py). Écrit la page : scene.html avec les données intégrées
(index.html à côté par défaut, non suivi).

Le reflet est calculé dans la page, depuis les sprites. Relevés sur la photo, en pixels de
toile : l'eau sous la branche est à la hauteur AXE (un peu plus bas vers la droite, la branche
s'éloigne) ; le reflet d'un point est son miroir autour de cette ligne, décalé de DX vers la
droite. Les grenouilles, vues d'en dessous dans l'eau, paraissent plus hautes : leur reflet,
accroché à la branche (CENTRE), est étiré de ETIRE (l'œil reflété tombe où il est sur la
photo).
"""

import json
import subprocess
import sys
from pathlib import Path

import branche
import decor
import numpy as np

ICI = Path(__file__).resolve().parent
GRENOUILLES = {  # script, sortie, coin du cadre sur la toile, côté où est l'autre
    "gauche": (ICI.parent / "tomopterna-gauche/gauche.py", 96, 60, 1),
    "droite": (ICI.parent / "tomopterna-droite/droite.py", 236, 46, -1),
}
AXE = branche.AXE  # hauteur de l'eau sous la branche : a + b·x
# décalage du reflet vers la droite : 2 pixels, sauf à gauche où la branche touche l'eau (son
# reflet y est collé à elle, sous le ménisque) ; il passe de 0 à 2 entre ces deux abscisses
DX = 2
DX_COLLE = (62, 100)
ETIRE = {"gauche": 1.22, "droite": 1.17}
CENTRE = {"gauche": 150, "droite": 290}  # où leur reflet s'accroche à celui de la branche
# d'où tombent les gouttes : sous la branche (pas à gauche, où elle touche l'eau)
GOUTTES = [80, 156, 214, 248, 352, 404]


def bord_branche(x, cote):
    """Le dessus (cote 0) ou le dessous (1) de la branche à l'abscisse x (pixels de toile)."""
    return branche.ECHELLE * branche.bords(x / branche.ECHELLE)[cote]


def animation(nom):
    """Refait les images d'animation de la grenouille et les lit."""
    script = GRENOUILLES[nom][0]
    subprocess.run([sys.executable, script.name, "animation"], cwd=script.parent, check=True)
    return json.loads((script.parent / f"{nom}_animation.json").read_text(encoding="utf-8"))


def oeil(sprite, x, y):
    """Le centre de l'œil sur la toile : celui de la pièce en demi-pixels de la pose de repos."""
    f = sprite["fins"]["0001"]
    return round(x + f["x"] + len(f["rangs"]) / 4, 1), round(y + f["y"] + len(f["rangs"]) / 4, 1)


def pied(sprite, x, y):
    """Le bout du pied qui pend sous la branche : le plus bas des pixels de la grenouille."""
    rangs = sprite["images"]["0001"]
    for j in range(len(rangs) - 1, -1, -1):
        cols, i = [], 0
        for p in branche._plages(rangs[j]):
            c, n = p[0], int(p[1:])
            if c != ".":
                cols += range(i, i + n)
            i += n
        if cols:
            return [float(x + round(np.mean(cols))), float(y + j + 1)]
    return None


def donnees():
    sprites, yeux = {}, {}
    for nom, (_, x, y, sens) in GRENOUILLES.items():
        s = animation(nom)
        sprites[nom] = {"x": x, "y": y, **s}
        yeux[nom] = (*oeil(s, x, y), sens)
        print(f"{nom} : {len(s['images'])} images, {len(s['palette'])} couleurs")
    calques, _ = decor.calques()
    supports, eclats, eclats_branche = branche.produire()
    sprites |= supports
    return {
        "fond": calques["fond"],
        "sprites": sprites,
        "geo": {
            "axe": AXE,
            "dx": DX,
            "dx_colle": DX_COLLE,
            "etire": ETIRE,
            "contact": {nom: round(bord_branche(x, 0), 1) for nom, x in CENTRE.items()},
            "centre": CENTRE,
            "yeux": yeux,
            "gouttes": [[x, round(bord_branche(x, 1), 1)] for x in GOUTTES]
            + [pied(sprites["droite"], *GRENOUILLES["droite"][1:3])],
            "menisque": eclats,
            "eclats": eclats_branche,
        },
    }


if __name__ == "__main__":
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else ICI / "index.html"
    texte = json.dumps(donnees(), separators=(",", ":"), ensure_ascii=False)
    page = (ICI / "scene.html").read_text(encoding="utf-8")
    sortie.write_text(page.replace("/*DONNEES*/null", texte, 1), encoding="utf-8")
    print(f"{sortie} : {len(texte) // 1024} Ko de données")
