"""Planche de relecture de la scène : celle du skill sprite-grenouille, ajustée à une toile large.

    uv run python documentation/tableau-de-bord/bac-a-sable/tomopterna/planche.py \
        scene_details.json SORTIE.html "Étape 3 · détails" "TEXTE"

Reprend planche.html du skill (outils.py planche) et y change deux choses pour cette scène de
448 × 303 : la toile se réduit à la largeur disponible au lieu de déborder (téléphone), et la
mosaïque de pixels ambrés, au lieu d'un grand ovale centré, suit la branche et les deux
grenouilles : une ellipse allongée, inclinée comme la branche.
"""

import subprocess
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
OUTILS = ICI.parents[3] / ".claude" / "skills" / "sprite-grenouille" / "outils.py"
NOM = "Deux C. tomopterna"
# centre de la mosaïque (pixels de la toile), demi-axes le long de la branche et en travers,
# inclinaison de la branche (radians)
MOSAIQUE = {"cx": 220, "cy": 100, "long": 165, "travers": 50, "angle": -0.148}


def main(source, sortie, etape, texte):
    subprocess.run(
        [sys.executable, str(OUTILS), "planche", source, sortie, NOM, texte, etape], check=True
    )
    page = Path(sortie).read_text(encoding="utf-8")
    remplacements = {
        "</style>": (
            "/* toile large : la scène se réduit à la largeur disponible au lieu de déborder */\n"
            ".scene { grid-template-columns: minmax(0, 1fr); }\n"
            ".scene canvas { max-width: 100%; height: auto; justify-self: center; }\n"
            "</style>"
        ),
        "const CX = (DX + C.largeur * 0.54) / TUILE, CY = (DY + C.hauteur / 2) / TUILE;": (
            f"const CX = (DX + {MOSAIQUE['cx']}) / TUILE, CY = (DY + {MOSAIQUE['cy']}) / TUILE;"
        ),
        "const RX = C.largeur * 0.625 / TUILE, RY = C.hauteur * 0.525 / TUILE;": (
            f"const RX = {MOSAIQUE['long']} / TUILE, RY = {MOSAIQUE['travers']} / TUILE;\n"
            f"const CA = Math.cos({MOSAIQUE['angle']}), SA = Math.sin({MOSAIQUE['angle']});"
        ),
        "const d = Math.hypot((x - CX) / RX, (y - CY) / RY);": (
            "const d = Math.hypot(((x - CX) * CA + (y - CY) * SA) / RX, "
            "(-(x - CX) * SA + (y - CY) * CA) / RY);"
        ),
    }
    for avant, apres in remplacements.items():
        assert page.count(avant) == 1, avant
        page = page.replace(avant, apres)
    Path(sortie).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
