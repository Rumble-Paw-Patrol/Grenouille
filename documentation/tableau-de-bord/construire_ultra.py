"""Construit index-ultra.html : la version « nuit guyanaise » du tableau de bord.

    uv run python documentation/tableau-de-bord/construire.py
    uv run python documentation/tableau-de-bord/construire_ultra.py

Même page, mêmes données, autre habillage : reprend les données intégrées par construire.py
dans index.html et les place dans modele-ultra.html. Les images à publier sont celles de
fichiers.json. Page de démonstration : https://claude.ai/artifact/Dr7Z8TekTmCM8cUa8mBGVr
"""

from pathlib import Path

ICI = Path(__file__).resolve().parent
DEBUT, FIN = "const D = ", ";\nconst C = D.contenu;"

index = (ICI / "index.html").read_text(encoding="utf-8")
donnees = index[index.index(DEBUT) + len(DEBUT) : index.index(FIN)]
modele = (ICI / "modele-ultra.html").read_text(encoding="utf-8")
page = modele.replace("/*DONNEES*/null", donnees)
(ICI / "index-ultra.html").write_text(page, encoding="utf-8")
print(f"index-ultra.html : {len(page) / 1e6:.2f} Mo")
