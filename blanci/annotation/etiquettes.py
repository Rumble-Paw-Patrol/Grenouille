"""Étiquettes d'intervalle ajoutées par l'annotateur, en plus de celles du schéma (A. blanci,
plusieurs, ?, faux ami) : un nom et une couleur, dans `annotations/etiquettes.json`.

Leur code est `custom_<nom simplifié>` : stocké tel quel dans `intervals.label`. Le jeu de
labels des fenêtres (`labels.LABELS`) ne les connaît pas, l'entraînement les ignore.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from blanci.core.config import project_path
from blanci.service import CUSTOM_PREFIX as PREFIX

LABELS_FILE = project_path("annotations/etiquettes.json")
DEFAULT_COLOR = "#B08CF2"


def is_custom(code: str) -> bool:
    return str(code).startswith(PREFIX)


def slug(name: str) -> str:
    """« Crapaud ? » → « custom_crapaud » (sans accents, minuscules, tirets bas)."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return PREFIX + re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def read_custom(path: Path | None = None) -> dict[str, dict[str, str]]:
    """{code : {"name", "color"}} dans l'ordre du fichier ; vide s'il n'existe pas ou est abîmé."""
    path = path or LABELS_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        str(code): {"name": str(v["name"]), "color": str(v.get("color") or DEFAULT_COLOR)}
        for code, v in data.items()
        if isinstance(v, dict) and v.get("name") and is_custom(code)
    }


def _write(labels: dict[str, dict[str, str]], path: Path | None) -> None:
    path = path or LABELS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add_custom(name: str, color: str, path: Path | None = None) -> str:
    """Ajoute (ou recolore, si le nom existe déjà) une étiquette ; renvoie son code."""
    name = name.strip()
    if not name:
        raise ValueError("nom d'étiquette vide")
    code = slug(name)
    if code == PREFIX:
        raise ValueError(f"nom inutilisable : {name!r} (lettres ou chiffres attendus)")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"couleur inconnue : {color!r} (attendue : #RRGGBB)")
    labels = read_custom(path)
    labels[code] = {"name": name, "color": color.upper()}
    _write(labels, path)
    return code


def remove_custom(code: str, path: Path | None = None) -> None:
    """Retire l'étiquette de la liste proposée ; les intervalles déjà posés la gardent."""
    labels = read_custom(path)
    if labels.pop(code, None) is not None:
        _write(labels, path)
