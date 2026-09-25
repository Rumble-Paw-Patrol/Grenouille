"""Scores hors-pli enregistrés (DECISIONS n° 91) : le format commun de tous les benchmarks.

Chaque benchmark (encodeur × tête, baseline, fusion, ensemble, détecteur, Blancinet, logits de
congénères) écrit ses scores hors-pli dans `paths.reports/oof/<source>.parquet`, une ligne par
fenêtre évaluée :

    source, kind, window_id, recording_id, offset_s, dur_s, point, site, y, presumed, score,
    fold, fingerprint

- `fold` : pli où la fenêtre a été testée (plis communs à tous, `dataset.folds_for`) ;
- `fingerprint` : empreinte des labels, du jeu gelé et des réglages d'évaluation (négatifs
  appariés, plis, graine) au moment du calcul. Deux sources d'empreintes différentes n'ont pas
  vu les mêmes données : le benchmark complet le signale au lieu de les comparer en silence.

Le niveau enregistrement est commun à toutes les sources : une fenêtre de 3 s et une de 5 s ne
se comparent pas, deux scores maximaux par enregistrement si. Le stock sert aussi de matière
première aux ensembles et à la fusion à plusieurs entrées : des scores déjà hors-pli, sur les
mêmes plis.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from blanci.config import config_path

OOF_COLUMNS = [
    "source",
    "kind",
    "window_id",
    "recording_id",
    "offset_s",
    "dur_s",
    "point",
    "site",
    "y",
    "presumed",
    "score",
    "fold",
    "fingerprint",
]
KINDS = ("encoder_head", "baseline", "fusion", "ensemble", "detector", "external")


def labels_fingerprint(con: sqlite3.Connection, cfg: dict) -> str:
    """Empreinte courte de ce qui définit un benchmark : labels courants, jeux gelés, négatifs
    appariés, plis, graine. Change dès qu'un label est ajouté."""
    from blanci.dataset import current_labels, pairing_options
    from blanci.frozen import frozen_versions

    labels = current_labels(con)[["window_id", "label"]].sort_values("window_id")
    payload = {
        "labels": labels.to_numpy().tolist(),
        "frozen": sorted(frozen_versions(cfg)),
        "pairing": pairing_options(cfg),
        "negatives_per_positive": cfg["benchmark"]["negatives_per_positive"],
        "n_splits": cfg["head"]["n_splits"],
        "seed": cfg["head"]["seed"],
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def source_path(cfg: dict, source: str) -> Path:
    """Fichier d'une source : caractères interdits dans un nom de fichier remplacés, et une
    empreinte courte du nom exact pour que deux sources ne tombent jamais sur le même fichier."""
    safe = re.sub(r"[^A-Za-z0-9_.@+=-]", "_", source)
    tag = hashlib.sha256(source.encode("utf-8")).hexdigest()[:6]
    return config_path(cfg, "reports") / "oof" / f"{safe}-{tag}.parquet"


def oof_frame(
    source: str,
    kind: str,
    data: pd.DataFrame,
    scores: np.ndarray,
    assignment: dict[str, int] | None,
    fingerprint: str,
    dur_s: float | None = None,
) -> pd.DataFrame:
    """Lignes d'une source à partir d'un jeu évalué (`embedded_training_set`, fenêtres des
    baselines…) : window_id, recording_id, offset_s, y, point, site, presumed[, dur_s]."""
    if kind not in KINDS:
        raise ValueError(f"sorte de source inconnue : {kind!r} (attendues : {KINDS})")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(data):
        raise ValueError("autant de scores que de fenêtres évaluées")
    points = data["point"].astype(str)
    out = pd.DataFrame(
        {
            "source": source,
            "kind": kind,
            "window_id": data["window_id"].to_numpy(),
            "recording_id": data["recording_id"].to_numpy(),
            "offset_s": data["offset_s"].to_numpy(dtype=float),
            "dur_s": data["dur_s"].to_numpy(dtype=float) if "dur_s" in data else dur_s,
            "point": points.to_numpy(),
            "site": data["site"].to_numpy() if "site" in data else None,
            "y": data["y"].to_numpy(dtype=int),
            "presumed": data["presumed"].to_numpy(dtype=bool) if "presumed" in data else False,
            "score": scores,
            "fold": points.map(assignment).fillna(-1).astype(int).to_numpy() if assignment else -1,
            "fingerprint": fingerprint,
        }
    )
    return out[OOF_COLUMNS]


def save_oof(cfg: dict, frame: pd.DataFrame) -> Path:
    """Écrit (remplace) une source. Une seule source par appel."""
    sources = frame["source"].unique()
    if len(sources) != 1:
        raise ValueError(f"une source par fichier, reçu {list(sources)}")
    path = source_path(cfg, str(sources[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[OOF_COLUMNS].to_parquet(path, index=False)
    return path


def load_oof(cfg: dict, sources: list[str] | None = None) -> pd.DataFrame:
    """Toutes les sources enregistrées (ou celles demandées)."""
    directory = config_path(cfg, "reports") / "oof"
    if sources is not None:
        paths = [source_path(cfg, s) for s in sources]
        missing = [s for s, p in zip(sources, paths, strict=True) if not p.exists()]
        if missing:
            raise ValueError(f"sources sans scores hors-pli : {missing}")
    else:
        paths = sorted(directory.glob("*.parquet")) if directory.exists() else []
    if not paths:
        return pd.DataFrame(columns=OOF_COLUMNS)
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def list_sources(cfg: dict) -> pd.DataFrame:
    """Une ligne par source : sorte, fenêtres, enregistrements, positifs, empreinte."""
    table = load_oof(cfg)
    if table.empty:
        return pd.DataFrame(
            columns=["source", "kind", "n_windows", "n_recordings", "n_pos", "fingerprint"]
        )
    return (
        table.groupby("source")
        .agg(
            kind=("kind", "first"),
            n_windows=("window_id", "size"),
            n_recordings=("recording_id", "nunique"),
            n_pos=("y", "sum"),
            fingerprint=("fingerprint", "first"),
        )
        .reset_index()
    )


def recording_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Score par (source, enregistrement) = maximum de ses fenêtres ; label = maximum."""
    return (
        table.groupby(["source", "recording_id"])
        .agg(score=("score", "max"), y=("y", "max"), point=("point", "first"))
        .reset_index()
    )
