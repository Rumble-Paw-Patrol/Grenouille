"""La scène animée : deux C. tomopterna au-dessus de l'eau, avec leur reflet.

    uv run python documentation/tableau-de-bord/bac-a-sable/tomopterna/scene.py [SORTIE.html]

D'après la photo d'Olivier Louguet. Refait les images d'animation de tomopterna.py (36 par
grenouille : gorge, flanc, paupière, tête, regard) et le calque de decor.py, puis écrit la
page : scene.html avec les données intégrées (index.html à côté par défaut, non suivi).

Le reflet est calculé dans la page, depuis les sprites. Relevés sur la photo, en pixels de
toile : l'eau sous la branche est à la hauteur AXE (un peu plus bas vers la droite, la branche
s'éloigne) ; le reflet d'un point est son miroir autour de cette ligne, décalé de DX vers la
droite. Les grenouilles, vues d'en dessous dans l'eau, paraissent plus hautes : leur reflet,
accroché à la branche (CONTACT), est étiré de ETIRE (l'œil reflété tombe où il est sur la
photo).
"""

import json
import sys
from pathlib import Path

import decor
import tomopterna as t

ICI = Path(__file__).resolve().parent
AXE = (161.8, 0.0083)  # hauteur de l'eau sous la branche : a + b·x
DX = 2
ETIRE = {"gauche": 1.22, "droite": 1.17}
CENTRE = {"gauche": 150, "droite": 290}  # où leur reflet s'accroche à celui de la branche
# l'œil (pixels de toile) : vers où chacune regarde ; « en face » : le côté de l'autre
YEUX = {"gauche": (174.4, 75.0, 1), "droite": (262.6, 63.0, -1)}
# d'où tombent les gouttes : sous la branche, et le bout du pied qui pend
GOUTTES = [12, 58, 132, 214, 248, 352, 404]
PIED = (307.0, 124.0)


def dessous(x):
    """Le bas de la branche à l'abscisse x (pixels de toile)."""
    X = x / t.ECHELLE
    return t.ECHELLE * (t.suivre(t.BRANCHE, X) + t.suivre(t.EPAISSEUR, X))


def dessus(x):
    X = x / t.ECHELLE
    return t.ECHELLE * (t.suivre(t.BRANCHE, X) - t.suivre(t.EPAISSEUR, X))


def donnees():
    sprites = {}
    for nom in t.SPRITES:
        c, sprite = t.produire("animation", nom)
        sprites[nom] = {"x": c.x0, "y": c.y0, **sprite}
        print(f"{nom} : {len(sprite['images'])} image(s), {len(sprite['palette'])} couleurs")
    calques, _ = decor.calques()
    return {
        "fond": calques["fond"],
        "sprites": sprites,
        "geo": {
            "axe": AXE,
            "dx": DX,
            "etire": ETIRE,
            "contact": {nom: round(dessus(x), 1) for nom, x in CENTRE.items()},
            "centre": CENTRE,
            "yeux": YEUX,
            "gouttes": [[x, round(dessous(x), 1)] for x in GOUTTES] + [list(PIED)],
        },
    }


if __name__ == "__main__":
    sortie = Path(sys.argv[1]) if len(sys.argv) > 1 else ICI / "index.html"
    texte = json.dumps(donnees(), separators=(",", ":"), ensure_ascii=False)
    page = (ICI / "scene.html").read_text(encoding="utf-8")
    sortie.write_text(page.replace("/*DONNEES*/null", texte, 1), encoding="utf-8")
    print(f"{sortie} : {len(texte) // 1024} Ko de données")
