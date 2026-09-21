"""Couche de service (§4) : le seul point d'entrée de la CLI et de la future GUI.

Les écrans de l'outil (Analyser, Vérifier, Résultats, Modèle) appellent ces fonctions, jamais
les modules de calcul directement : le jour où la GUI remplace la CLI, rien d'autre ne bouge.

Toute décision est estampillée (`encoder_id`, `head_version`, `threshold_id`) : on doit pouvoir
dire de quel modèle vient un « à vérifier » remonté six mois plus tôt (§4).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.active import build_queue
from blanci.aggregate import aggregate_recording, rank_points
from blanci.config import config_path
from blanci.dataset import current_labels, embedded_training_set, local_minutes, recordings_table
from blanci.db import encoder_params, model_params, next_version, register_model, utc_now
from blanci.evaluate import evaluate, recall_at_precision
from blanci.head import Head, oof_scores, train_head
from blanci.index import search
from blanci.labels import LABELS, QUALITIES, SOURCES
from blanci.store import EmbeddingStore

SCORE_CHUNK = 200_000


def head_id(encoder_id: str, version: str) -> str:
    return f"{encoder_id}:head:{version}"


def threshold_id(encoder_id: str, version: str, min_precision: float) -> str:
    return f"{encoder_id}:head:{version}:p{min_precision}"


def head_directory(cfg: dict, encoder_id: str, version: str) -> Path:
    return config_path(cfg, "models") / "head" / f"{encoder_id}-{version}"


def store_for(cfg: dict, encoder_id: str) -> EmbeddingStore:
    return EmbeddingStore(config_path(cfg, "embeddings"), encoder_id)


# --- Entraînement -----------------------------------------------------------------------------


@dataclass
class TrainResult:
    encoder_id: str
    version: str
    model_id: str
    threshold: float
    threshold_id: str
    min_precision: float
    recall_at_threshold: float
    directory: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"tête {self.encoder_id} {self.version} : "
            f"{self.metrics.get('n_pos', 0)} positifs, {self.metrics.get('n_neg', 0)} négatifs, "
            f"AP {self.metrics.get('ap', float('nan')):.3f} (enregistrements) ; "
            f"seuil {self.threshold:.3f} à précision ≥ {self.min_precision} "
            f"(rappel {self.recall_at_threshold:.3f})"
        )


def train_and_register(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    filters: dict | None = None,
    min_precision: float | None = None,
) -> TrainResult:
    """Entraîne la tête sur tous les labels, calibre le seuil, enregistre le tout.

    Le seuil est calculé sur les scores **hors-pli** : un seuil calibré sur des scores en-pli
    serait optimiste, la tête ayant vu chacun de ses exemples (§6).
    """
    params = encoder_params(con, encoder_id)
    bench, head_cfg = cfg["benchmark"], cfg["head"]
    min_precision = bench["precisions"][0] if min_precision is None else min_precision

    data, X = embedded_training_set(
        con,
        store_for(cfg, encoder_id),
        params["window_s"],
        per_positive=bench["negatives_per_positive"],
        slot_tolerance_min=bench["slot_tolerance_min"],
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
        seed=head_cfg["seed"],
        filters=filters,
    )
    y = data["y"].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(f"{encoder_id} : une seule classe dans le jeu étiqueté")
    groups = data["point"].to_numpy()
    recordings = data["recording_id"].to_numpy()

    oof = oof_scores(
        X,
        y,
        groups,
        n_splits=head_cfg["n_splits"],
        method="logistic",
        C_grid=head_cfg["C_grid"],
        seed=head_cfg["seed"],
    )
    recall, threshold = recall_at_precision(y, oof.values, min_precision)
    metrics = evaluate(
        oof.values,
        y,
        recordings,
        level="recording",
        precisions=tuple(bench["precisions"]),
        n_boot=bench["n_boot"],
        seed=head_cfg["seed"],
    )

    head = train_head(X, y, groups, head_cfg["C_grid"], seed=head_cfg["seed"])
    version = next_version(con, "head", encoder_id)
    tid = threshold_id(encoder_id, version, min_precision)
    head.meta |= {
        "encoder_id": encoder_id,
        "version": version,
        "threshold": threshold,
        "threshold_id": tid,
        "min_precision": min_precision,
        "recall_at_threshold": recall,
        "trained_at": utc_now(),
    }
    directory = head_directory(cfg, encoder_id, version)
    head.save(directory)

    register_model(
        con,
        head_id(encoder_id, version),
        "head",
        encoder_id,
        version,
        {"path": str(directory), "metrics": metrics, **head.meta},
    )
    register_model(
        con,
        tid,
        "threshold",
        encoder_id,
        version,
        {
            "head_model_id": head_id(encoder_id, version),
            "threshold": threshold,
            "min_precision": min_precision,
            "recall_at_threshold": recall,
        },
    )
    return TrainResult(
        encoder_id,
        version,
        head_id(encoder_id, version),
        threshold,
        tid,
        min_precision,
        recall,
        directory,
        metrics,
    )


def load_head(con: sqlite3.Connection, encoder_id: str, version: str) -> tuple[Head, dict]:
    """Tête sauvegardée + sa ligne de registre. `version` = 'latest' prend la plus récente."""
    if version == "latest":
        row = con.execute(
            "SELECT version FROM models WHERE kind = 'head' AND name = ? "
            "ORDER BY created_at DESC, model_id DESC LIMIT 1",
            (encoder_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"aucune tête entraînée pour {encoder_id} (lancer `blanci train`)")
        version = row["version"]
    params = model_params(con, head_id(encoder_id, version), kind="head")
    return Head.load(Path(params["path"])), params


# --- Score et décisions -----------------------------------------------------------------------


@dataclass
class ScoreResult:
    encoder_id: str
    version: str
    threshold: float
    windows: int
    decisions: pd.DataFrame

    def summary(self) -> str:
        counts = self.decisions["status"].value_counts().to_dict()
        detail = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        return f"{self.windows} fenêtres scorées, {len(self.decisions)} enregistrements : {detail}"


def score_and_decide(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    version: str = "latest",
    filters: dict | None = None,
) -> ScoreResult:
    """Score toutes les fenêtres du stock, puis décide par enregistrement (§1).

    Les décisions du même triplet (encodeur, tête, seuil) sont remplacées : elles sont
    reproductibles, contrairement aux labels qui ne s'écrasent jamais (§13.7).
    """
    head, params = load_head(con, encoder_id, version)
    version = params["version"]
    threshold, tid = params["threshold"], params["threshold_id"]
    store = store_for(cfg, encoder_id)
    model_id = head_id(encoder_id, version)
    agg = cfg["aggregate"]

    frames, total = [], 0
    for path in store.fragments(filters):
        meta, emb = store.read(path)
        scores = head.decision(emb)
        con.executemany(
            "INSERT INTO scores (window_id, model_id, score) VALUES (?, ?, ?) "
            "ON CONFLICT(window_id, model_id) DO UPDATE SET score = excluded.score",
            [(w, model_id, float(s)) for w, s in zip(meta["window_id"], scores, strict=True)],
        )
        total += len(scores)
        frames.append(meta.assign(score=scores))
    con.commit()
    if not frames:
        raise ValueError(f"aucun embedding pour {encoder_id} (filtres : {filters})")

    scored = pd.concat(frames, ignore_index=True)
    rows = []
    for recording_id, group in scored.groupby("recording_id"):
        group = group.sort_values("offset_s")
        fraction, status = aggregate_recording(
            group["score"].to_numpy(),
            threshold,
            group["offset_s"].to_numpy(),
            min_positive_windows=agg["min_positive_windows"],
            min_span_s=agg["min_span_s"],
        )
        rows.append({"recording_id": recording_id, "fraction": fraction, "status": status})
    decisions = pd.DataFrame(rows)

    con.execute(
        "DELETE FROM decisions WHERE encoder_id = ? AND head_version = ? AND threshold_id = ?",
        (encoder_id, version, tid),
    )
    con.executemany(
        "INSERT INTO decisions (recording_id, encoder_id, head_version, threshold_id, "
        "fraction, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (r.recording_id, encoder_id, version, tid, r.fraction, r.status, utc_now())
            for r in decisions.itertuples()
        ],
    )
    con.commit()
    return ScoreResult(encoder_id, version, threshold, total, decisions)


def ranked_points(
    con: sqlite3.Connection, encoder_id: str, version: str = "latest"
) -> pd.DataFrame:
    """Classement des points à partir des décisions enregistrées (§1)."""
    if version == "latest":
        row = con.execute(
            "SELECT head_version FROM decisions WHERE encoder_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (encoder_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"aucune décision pour {encoder_id} (lancer `blanci score`)")
        version = row["head_version"]
    decisions = pd.read_sql_query(
        "SELECT d.recording_id, d.fraction, d.status, r.dataset, r.site, r.mic_id, r.start_utc "
        "FROM decisions d JOIN recordings r USING (recording_id) "
        "WHERE d.encoder_id = ? AND d.head_version = ?",
        con,
        params=(encoder_id, version),
    )
    if decisions.empty:
        raise ValueError(f"aucune décision pour {encoder_id} {version}")
    return rank_points(decisions)


# --- File de vérification ---------------------------------------------------------------------


def make_queue(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    n: int | None = None,
    version: str = "latest",
    mix: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """File d'apprentissage actif : 60 % incertains, 20 % meilleurs, 20 % aléatoire (§5).

    Unité : l'enregistrement. Son score est le maximum de ses fenêtres, comme pour l'AP.
    """
    head, params = load_head(con, encoder_id, version)
    model_id = head_id(encoder_id, params["version"])
    n = n or cfg["active"]["batch_recordings"]
    mix = tuple(mix or cfg["active"]["mix"])

    scores = pd.read_sql_query(
        "SELECT w.recording_id, MAX(s.score) AS score FROM scores s "
        "JOIN windows w USING (window_id) WHERE s.model_id = ? GROUP BY w.recording_id",
        con,
        params=(model_id,),
    )
    if scores.empty:
        raise ValueError(f"aucun score pour {model_id} (lancer `blanci score`)")

    recordings = recordings_table(con)
    scores = scores.merge(
        recordings[["recording_id", "mic_id", "start_utc"]], on="recording_id", how="left"
    )
    offset = cfg["recorder"]["filename_utc_offset_h"]
    scores["hour"] = (local_minutes(scores["start_utc"], offset) // 60).astype("Int64")

    labelled = current_labels(con)[["recording_id"]].drop_duplicates()
    queue = build_queue(
        scores,
        labelled,
        n=n,
        mix=mix,
        threshold=params["threshold"],
        seed=cfg["head"]["seed"],
    )
    return queue.merge(recordings[["recording_id", "path", "site"]], on="recording_id", how="left")


# --- Annotation -------------------------------------------------------------------------------


def append_label(
    con: sqlite3.Connection,
    window_id: str,
    label: str,
    source: str,
    quality: str | None = None,
    species: str | None = None,
    conditions: dict | None = None,
    annotator: str | None = None,
) -> int:
    """Ajoute un label. Une correction est une nouvelle ligne, jamais une mise à jour (§13.7)."""
    if label not in LABELS:
        raise ValueError(f"label inconnu : {label!r} (attendus : {', '.join(LABELS)})")
    if quality is not None and quality not in QUALITIES:
        raise ValueError(f"qualité inconnue : {quality!r} (attendues : {', '.join(QUALITIES)})")
    if source not in SOURCES:
        raise ValueError(f"source inconnue : {source!r} (attendues : {', '.join(SOURCES)})")
    if con.execute("SELECT 1 FROM windows WHERE window_id = ?", (window_id,)).fetchone() is None:
        raise ValueError(f"fenêtre inconnue : {window_id}")
    cursor = con.execute(
        "INSERT INTO labels (window_id, label, quality, species, conditions, annotator, "
        "source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            window_id,
            label,
            quality,
            species,
            json.dumps(conditions, ensure_ascii=False) if conditions else None,
            annotator,
            source,
            utc_now(),
        ),
    )
    con.commit()
    return int(cursor.lastrowid)


# --- Recherche par similarité -----------------------------------------------------------------


def similarity_search(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    k: int = 300,
    filters: dict | None = None,
    use_paired_negatives: bool = True,
) -> pd.DataFrame:
    """Fenêtres les plus proches des positifs annotés (§5, récolte hors Mataroni).

    Avec `use_paired_negatives`, la similarité aux négatifs est retranchée : c'est le
    prototype différentiel appliqué à la recherche, il retire le fond sonore partagé (§3).
    """
    params = encoder_params(con, encoder_id)
    store = store_for(cfg, encoder_id)
    bench = cfg["benchmark"]
    data, X = embedded_training_set(
        con,
        store,
        params["window_s"],
        per_positive=bench["negatives_per_positive"] if use_paired_negatives else 0,
        slot_tolerance_min=bench["slot_tolerance_min"],
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
        seed=cfg["head"]["seed"],
    )
    y = data["y"].to_numpy()
    if not (y == 1).any():
        raise ValueError("aucun positif annoté : la recherche par similarité a besoin de requêtes")
    negatives = X[y == 0] if use_paired_negatives and (y == 0).any() else None
    found = search(X[y == 1], store, k=k, filters=filters, negatives=negatives)
    recordings = recordings_table(con)
    return found.merge(
        recordings[["recording_id", "path", "site", "mic_id", "start_utc"]],
        on="recording_id",
        how="left",
    )


# --- Évaluation sur sites tenus à l'écart -----------------------------------------------------


def evaluate_holdout(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    holdout_sites: list[str] | None = None,
    level: str = "recording",
) -> dict[str, Any]:
    """Entraîne hors des sites tenus à l'écart, évalue dessus (§6, niveau 2).

    Sans `holdout_sites`, retombe sur la validation groupée par micro (§6, niveau 1).
    """
    params = encoder_params(con, encoder_id)
    bench, head_cfg = cfg["benchmark"], cfg["head"]
    data, X = embedded_training_set(
        con,
        store_for(cfg, encoder_id),
        params["window_s"],
        per_positive=bench["negatives_per_positive"],
        slot_tolerance_min=bench["slot_tolerance_min"],
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
        seed=head_cfg["seed"],
    )
    y = data["y"].to_numpy()
    recordings = data["recording_id"].to_numpy()

    wanted = {site.lower() for site in holdout_sites or []}
    if not wanted:
        oof = oof_scores(
            X,
            y,
            data["point"].to_numpy(),
            n_splits=head_cfg["n_splits"],
            method="logistic",
            C_grid=head_cfg["C_grid"],
            seed=head_cfg["seed"],
        )
        scores, mask = oof.values, np.ones(len(y), dtype=bool)
        protocol = "plis groupés par micro"
    else:
        mask = data["site"].str.lower().isin(wanted).to_numpy()
        if not mask.any():
            raise ValueError(f"aucune fenêtre sur les sites tenus à l'écart : {holdout_sites}")
        if len(np.unique(y[~mask])) < 2:
            raise ValueError("une seule classe hors des sites tenus à l'écart : rien à apprendre")
        head = train_head(
            X[~mask],
            y[~mask],
            data.loc[~mask, "point"].to_numpy(),
            head_cfg["C_grid"],
            seed=head_cfg["seed"],
        )
        scores = head.decision(X)
        protocol = f"entraînement hors {', '.join(sorted(wanted))}"

    if len(np.unique(y[mask])) < 2:
        raise ValueError("une seule classe sur les sites tenus à l'écart : AP indéfinie")
    metrics = evaluate(
        scores[mask],
        y[mask],
        recordings[mask],
        level=level,
        precisions=tuple(bench["precisions"]),
        n_boot=bench["n_boot"],
        seed=head_cfg["seed"],
    )
    return {
        "encoder_id": encoder_id,
        "protocol": protocol,
        "holdout": sorted(wanted),  # normalisés en minuscules, comme la comparaison
        "n_windows": int(mask.sum()),
        **metrics,
    }
