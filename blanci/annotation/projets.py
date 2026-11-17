"""Projets du poste d'annotation : un projet est un préréglage (en-tête, étiquettes et qualités,
files d'écoute, style), choisi dans le panneau de gauche à côté de l'annotateur.

Le projet de départ (`DEFAULT_PROJECT`) est celui d'origine : ses fichiers sont ceux de
`annotations/` (`entete/`, `etiquettes.json`, `files_ecoute/`). Un autre projet vit dans
`annotations/projets/<nom>/` (mêmes noms de fichiers). Le style de chaque projet, lui, est
gardé avec les préférences de l'utilisateur (`habillage.py`). Les annotations validées restent
dans la base commune : elles portent l'espèce et l'étiquette, pas le projet.

`activate` redirige les fichiers d'en-tête et d'étiquettes vers ceux du projet pour la
séance en cours ; les modules les lisent à chaque appel.
"""

from __future__ import annotations

import re
from pathlib import Path

from blanci.annotation import bandeau, etiquettes
from blanci.annotation.habillage import DEFAULT_PROJECT
from blanci.core.config import project_path

PROJECTS_DIR = project_path("annotations/projets")
DEFAULT_LABEL = "Anomaloglossus blanci"
QUEUES_NAME = "files_ecoute"

# Fichiers du projet de départ, à remettre quand on y revient.
_default_files: dict[str, Path] = {}


def project_dir(name: str) -> Path | None:
    """Dossier du projet, ou None pour le projet de départ."""
    return None if name == DEFAULT_PROJECT else PROJECTS_DIR / name


def list_projects() -> list[str]:
    """Le projet de départ, puis les projets de `annotations/projets/` par ordre alphabétique."""
    found = (
        sorted(p.name for p in PROJECTS_DIR.iterdir() if p.is_dir())
        if PROJECTS_DIR.is_dir()
        else []
    )
    return [DEFAULT_PROJECT, *found]


def label(name: str) -> str:
    return DEFAULT_LABEL if name == DEFAULT_PROJECT else name


def create_project(name: str) -> str:
    """Crée le projet (dossiers vides : en-tête et étiquettes d'origine tant qu'on n'y touche
    pas) et renvoie son nom. ValueError si le nom est vide, déjà pris ou impropre à un dossier."""
    name = re.sub(r"\s+", " ", name or "").strip()
    if not name:
        raise ValueError("Écrire le nom du projet.")
    if re.search(r'[\\/:*?"<>|]', name) or name.startswith(".") or name == DEFAULT_PROJECT:
        raise ValueError(
            'Nom de projet impossible : éviter / \\ : * ? " < > | et le point initial.'
        )
    if name.casefold() in (p.casefold() for p in list_projects()) or name == DEFAULT_LABEL:
        raise ValueError(f"Le projet « {name} » existe déjà.")
    (PROJECTS_DIR / name / QUEUES_NAME).mkdir(parents=True)
    return name


def _ours(path: Path) -> bool:
    return PROJECTS_DIR in path.parents


def activate(name: str) -> Path | None:
    """Met en service le projet `name` : en-tête et étiquettes redirigés vers ses fichiers.
    Renvoie le dossier de ses files d'écoute, ou None pour le projet de départ (celui de
    `paths.queues`). Un projet disparu ramène au projet de départ."""
    # Les fichiers du projet de départ : ceux que les modules ont quand nous n'y avons pas
    # touché (un test peut avoir posé les siens).
    if not _ours(bandeau.HEADER_DIR):
        _default_files["header"] = bandeau.HEADER_DIR
    if not _ours(etiquettes.LABELS_FILE):
        _default_files["labels"] = etiquettes.LABELS_FILE
    folder = project_dir(name)
    if folder is None or not folder.is_dir():
        bandeau.HEADER_DIR = _default_files["header"]
        etiquettes.LABELS_FILE = _default_files["labels"]
        return None
    bandeau.HEADER_DIR = folder / "entete"
    etiquettes.LABELS_FILE = folder / "etiquettes.json"
    return folder / QUEUES_NAME
