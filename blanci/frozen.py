"""Jeu gelé (§6) : enregistrements écoutés en entier, mis de côté pour toujours.

Un jeu gelé est une liste d'enregistrements figée à une date, versionnée
(`paths.frozen_test/jeu_gele_<version>.csv`, en lecture seule). Ses enregistrements :
- ne servent **jamais** à entraîner ni à régler quoi que ce soit : ni tête, ni seuil, ni
  benchmark, ni gabarit de baseline, ni requête de similarité, ni fusion ;
- servent à mesurer, toujours sur les mêmes enregistrements : comparer deux versions de la
  tête ou deux encodeurs avant de basculer, tracer la courbe d'apprentissage, décider l'arrêt
  des tours d'annotation (gain d'AP < 0,02 sur deux tours, §5).

Sans jeu gelé, chaque mesure se fait sur des données qui ont aidé à construire le modèle
(directement, ou en guidant nos choix) : elle est optimiste, et de plus en plus à mesure que
les tours s'enchaînent. Un nouveau gel crée une nouvelle version ; une version existante ne
change jamais.
"""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pandas as pd

from blanci.config import config_path
from blanci.dataset import current_labels, recordings_table
from blanci.db import utc_now
from blanci.labels import POSITIVE_LABELS

PREFIX = "jeu_gele_"


def frozen_directory(cfg: dict) -> Path:
    return config_path(cfg, "frozen_test")


def frozen_versions(cfg: dict) -> dict[str, pd.DataFrame]:
    """{version: liste des enregistrements} de tous les gels."""
    directory = frozen_directory(cfg)
    if not directory.is_dir():
        return {}
    return {
        path.stem.removeprefix(PREFIX): pd.read_csv(path)
        for path in sorted(directory.glob(f"{PREFIX}*.csv"))
    }


def frozen_recordings(cfg: dict, versions: list[str] | None = None) -> set[str]:
    """Identifiants des enregistrements gelés (toutes versions par défaut)."""
    frozen = frozen_versions(cfg)
    wanted = versions or list(frozen)
    unknown = set(wanted) - set(frozen)
    if unknown:
        raise ValueError(f"jeu gelé inconnu : {sorted(unknown)} (existants : {sorted(frozen)})")
    return {rid for v in wanted for rid in frozen[v]["recording_id"]}


def freeze(con: sqlite3.Connection, cfg: dict, source: Path, version: str) -> tuple[Path, dict]:
    """Gèle les enregistrements d'une file CSV (colonne recording_id) sous `version`.

    Refuse d'écraser une version existante. Renvoie le chemin et un bilan : nombre
    d'enregistrements, déjà gelés ailleurs, et labels existants qui sortent de l'entraînement.
    """
    target = frozen_directory(cfg) / f"{PREFIX}{version}.csv"
    if target.exists():
        raise FileExistsError(f"le jeu gelé {version} existe déjà ({target}) : jamais réécrit")
    ids = pd.read_csv(source)["recording_id"].drop_duplicates()
    recordings = recordings_table(con).set_index("recording_id")
    unknown = set(ids) - set(recordings.index)
    if unknown:
        raise ValueError(f"{len(unknown)} enregistrements inconnus de l'inventaire dans {source}")
    table = recordings.loc[ids, ["path", "site", "mic_id", "start_utc"]].reset_index()
    table["frozen_at"] = utc_now()

    labels = current_labels(con)
    withdrawn = labels[labels["recording_id"].isin(ids)]
    report = {
        "n_recordings": len(table),
        "already_frozen": len(set(ids) & frozen_recordings(cfg)),
        "labels_withdrawn": len(withdrawn),
        "positive_labels_withdrawn": int(withdrawn["label"].isin(POSITIVE_LABELS).sum()),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    os.chmod(target, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)  # lecture seule
    return target, report
