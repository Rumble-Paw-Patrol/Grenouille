"""Bac à sable « A. blanci à la chasse » : les images de la grenouille, puis la page.

    uv run python documentation/tableau-de-bord/bac-a-sable/chasse/chasse.py
    (avec --page : garde les images de chasse.json, ne refait que le décor et la page)

Reprend la grenouille du tableau de bord (../grenouille.py, avec ses réglages « bouche » et
« rentres ») et produit les 60 images dont la scène a besoin : 5 angles de tête (de baissée à
relevée, 0,06 rad par cran) × gorge × flanc × œil ouvert ou fermé ; la bouche entrouverte et
ouverte pour la langue, à chaque angle ; les yeux rentrés pour avaler, gorge au repos ou
gonflée, à chaque angle. Les données géométriques dont la page a besoin (bout de la bouche,
œil, pivot de la tête) sont tirées des mêmes constantes. Le décor vient de decor.py, en
trois calques PNG. Écrit index.html : page.html avec les données intégrées (chasse.json n'est
pas suivi).
"""

import importlib.util
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("grenouille", ICI.parent / "grenouille.py")
grenouille = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grenouille)
sys.path.insert(0, str(ICI))
import decor  # noqa: E402

OUVERTURES = {1: 1.0, 2: 1.9}  # bouche entrouverte, ouverte (unités, au bout du museau)


def images():
    """Palette et images codées ; la palette se remplit au fil des images calculées."""
    if "--page" in sys.argv and (ICI / "chasse.json").exists():  # ne refaire que décor et page
        g = json.loads((ICI / "chasse.json").read_text(encoding="utf-8"))["grenouille"]
        return g["palette"], g["images"]
    out = {}
    for t in range(5):
        tete = t - 2
        for g in (0, 1):
            for s in (0, 1):
                for c in (0, 1):
                    out[f"{g}{s}{c}{t}"] = grenouille.composer(g * 0.9, s * 0.7, bool(c), tete)
            out[f"R{g}{t}"] = grenouille.composer(g * 0.9, 0.0, False, tete, 0.0, True)
        for b, ouverture in OUVERTURES.items():
            out[f"B{b}{t}"] = grenouille.composer(0.0, 0.0, False, tete, ouverture)
    return grenouille.PALETTE_NUANCES, {
        cle: [grenouille.plages(r) for r in rangs] for cle, rangs in out.items()
    }


if __name__ == "__main__":
    palette, codees = images()
    donnees = {
        "grenouille": {
            "palette": palette,
            "largeur": grenouille.W,
            "hauteur": grenouille.H,
            "images": codees,
        },
        "decor": decor.calques()[0],
        # en unités du repère de la grenouille (2 pixels par unité)
        "geometrie": {
            "echelle": grenouille.ECHELLE,
            "pivot": [18.0, 9.5],
            "pas": 0.06,
            "bouche": [2.4, 9.8],
            "oeil": [11.2, 8.4],
        },
    }
    texte = json.dumps(donnees, separators=(",", ":"), ensure_ascii=False)
    (ICI / "chasse.json").write_text(texte, encoding="utf-8")
    page = (ICI / "page.html").read_text(encoding="utf-8")
    (ICI / "index.html").write_text(page.replace("/*DONNEES*/null", texte, 1), encoding="utf-8")
    n = len(donnees["grenouille"]["images"])
    print(f"{n} images, index.html : {len(texte) // 1024} Ko de données")
