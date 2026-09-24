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
from blanci.activity import (
    curve_correlation,
    daily_probability,
    detections_table,
    diel_index,
    reference_checks,
)
from blanci.aggregate import aggregate_recording, rank_points
from blanci.config import config_path
from blanci.dataset import (
    current_labels,
    embedded_training_set,
    local_minutes,
    recordings_table,
    training_set,
)
from blanci.db import encoder_params, model_params, next_version, register_model, utc_now
from blanci.evaluate import (
    evaluate,
    false_alarms_per_hour,
    grouped_folds,
    paired_bootstrap,
    recall_at_precision,
    recall_by_group,
    to_recordings,
)
from blanci.frozen import frozen_recordings, frozen_versions
from blanci.fusion import FusionWeights, fit_fusion, fusion_oof
from blanci.head import Head, OOFScores, fit_logistic, oof_scores, select_C, train_head
from blanci.index import search
from blanci.labels import LABELS, POSITIVE_LABELS, QUALITIES, SOURCES
from blanci.qc import apply_annotation_flags, is_excluded
from blanci.sequential import load_onsets, recording_persistence, window_rhythm
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
        exclude_recordings=frozen_recordings(cfg),
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
        # Jeux gelés exclus à l'entraînement : seuls eux peuvent juger cette tête (§6).
        "frozen_excluded": sorted(frozen_versions(cfg)),
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
    """Tête sauvegardée + sa ligne de registre. `version` : v1, v2…, « latest » (la plus
    récente), « adopted » (celle qu'a retenue `blanci retrain`), « default » (l'adoptée s'il
    y en a une, sinon la plus récente)."""
    if version in ("adopted", "default"):
        adopted = adopted_version(con, encoder_id)
        if adopted is None and version == "adopted":
            raise ValueError(f"aucune tête adoptée pour {encoder_id} (lancer `blanci retrain`)")
        version = adopted or "latest"
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
    threshold_id: str = ""

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
    fusion: bool = False,
) -> ScoreResult:
    """Score toutes les fenêtres du stock, puis décide par enregistrement (§1).

    Avec `fusion`, le score de décision est celui de la fusion enregistrée pour cette tête
    (`blanci fusion`) : tête + rythme + persistance, à son propre seuil.
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
    if fusion:
        fusion_params = model_params(con, fusion_id(encoder_id, version), kind="fusion")
        window_s = encoder_params(con, encoder_id)["window_s"]
        scored["score"] = fused_scores(con, cfg, fusion_params, scored, window_s)
        threshold, tid = fusion_params["threshold"], fusion_params["threshold_id"]
        con.executemany(
            "INSERT INTO scores (window_id, model_id, score) VALUES (?, ?, ?) "
            "ON CONFLICT(window_id, model_id) DO UPDATE SET score = excluded.score",
            [
                (w, fusion_id(encoder_id, version), float(s))
                for w, s in zip(scored["window_id"], scored["score"], strict=True)
            ],
        )
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
    return ScoreResult(encoder_id, version, threshold, total, decisions, tid)


def ranked_points(
    con: sqlite3.Connection,
    encoder_id: str,
    version: str = "latest",
    threshold_id: str | None = None,
) -> pd.DataFrame:
    """Classement des points à partir des décisions enregistrées (§1).

    `threshold_id` sépare les décisions de la tête seule de celles de la fusion.
    """
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
        "WHERE d.encoder_id = ? AND d.head_version = ? AND (? IS NULL OR d.threshold_id = ?)",
        con,
        params=(encoder_id, version, threshold_id, threshold_id),
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
    row = con.execute(
        "SELECT recording_id FROM windows WHERE window_id = ?", (window_id,)
    ).fetchone()
    if row is None:
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
    apply_annotation_flags(con, [row[0]])  # micro dans sac, pluie : drapeaux posés à l'écoute
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
        exclude_recordings=frozen_recordings(cfg),
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
        exclude_recordings=frozen_recordings(cfg),
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
    # Rappel par qualité A/B/C et par site, fenêtre par fenêtre, au seuil de précision
    # plancher (§6) : un rappel global cache souvent des chants lointains tous manqués.
    floor = min(bench["precisions"])
    _, threshold = recall_at_precision(y[mask], scores[mask], floor)
    frame = data.loc[mask]
    strata = {
        "qualité": frame["quality"].to_numpy() if "quality" in frame else np.full(len(frame), None),
        "site": frame["site"].to_numpy(),
    }
    by_stratum = pd.concat(
        [
            recall_by_group(scores[mask], y[mask], values, threshold).assign(by=name)
            for name, values in strata.items()
        ],
        ignore_index=True,
    )
    return {
        "encoder_id": encoder_id,
        "protocol": protocol,
        "holdout": sorted(wanted),  # normalisés en minuscules, comme la comparaison
        "n_windows": int(mask.sum()),
        **metrics,
        "stratum_threshold": threshold,
        "stratum_precision": floor,
        "by_stratum": by_stratum,
    }


# --- Clustering (§5 bis) ----------------------------------------------------------------------


def _qc_flagged(qc: Any, keys: tuple[str, ...] = ("saturation", "rain", "in_bag")) -> bool:
    return is_excluded(qc, keys)


def run_clustering(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    mode: str = "c0",
    n: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """C0 : échantillon global du stock. C1 : positifs du site de `cluster.c1_site` + fenêtres
    des mêmes micros aux mêmes heures (négatifs appariés présumés).

    Renvoie (résumé, tableau par groupe, groupe de chaque fenêtre).
    """
    from blanci.cluster import c0_summary, c1_verdict, cluster_embeddings, cluster_table

    ccfg = cfg["cluster"]
    store = store_for(cfg, encoder_id)
    recordings = recordings_table(con).set_index("recording_id")
    if mode == "c0":
        meta, X = store.sample(n or ccfg["c0_sample"], seed=ccfg["seed"])
        if not len(meta):
            raise ValueError(f"aucun embedding pour {encoder_id}")
        meta = meta.assign(y=np.nan)
    elif mode == "c1":
        params = encoder_params(con, encoder_id)
        n_neg = n or ccfg["c1_negatives"]
        labels = current_labels(con)
        n_pos_rec = labels[labels["label"].isin(POSITIVE_LABELS)]["recording_id"].nunique()
        meta, X = embedded_training_set(
            con,
            store,
            params["window_s"],
            per_positive=max(1, -(-n_neg // max(1, n_pos_rec))),
            slot_tolerance_min=cfg["benchmark"]["slot_tolerance_min"],
            utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
            seed=ccfg["seed"],
            filters={"site": ccfg["c1_site"]},
            exclude_recordings=frozen_recordings(cfg),
        )
    else:
        raise ValueError(f"mode inconnu : {mode!r} (c0 ou c1)")

    rec = recordings.loc[meta["recording_id"]]
    mics = rec["point"].to_numpy()
    flagged = rec["qc_flags"].map(_qc_flagged).to_numpy()
    assignments = cluster_embeddings(
        X,
        n_components=ccfg["pca_components"],
        min_cluster_size=ccfg["min_cluster_size"],
        min_samples=ccfg["min_samples"],
        seed=ccfg["seed"],
    )
    y = meta["y"].to_numpy() if mode == "c1" else None
    table = cluster_table(assignments, mics, y, flagged)
    if mode == "c1":
        summary = c1_verdict(
            assignments,
            y,
            mics,
            ccfg["c1_min_recall"],
            ccfg["c1_min_enrichment"],
            ccfg["c1_max_ami_mic"],
        )
    else:
        summary = c0_summary(assignments, mics)
    summary = {"encoder_id": encoder_id, "mode": mode} | summary
    windows = meta[["window_id", "recording_id", "offset_s", "y"]].assign(
        point=mics, cluster=assignments
    )
    return summary, table, windows


# --- Jeu gelé (§6) --------------------------------------------------------------------------


def evaluate_frozen(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    version: str = "latest",
    frozen_version: str | None = None,
) -> dict[str, Any]:
    """Juge une tête enregistrée sur un jeu gelé qu'elle n'a jamais vu.

    Refuse une tête entraînée avant le gel (le jeu gelé aurait servi à l'entraîner). Le seuil
    est celui de la tête : on mesure l'outil tel qu'il serait livré. Les enregistrements gelés
    sont écoutés en entier : les fausses alarmes par heure y ont un sens.
    """
    head, params = load_head(con, encoder_id, version)
    frozen = frozen_versions(cfg)
    if not frozen:
        raise ValueError("aucun jeu gelé (blanci freeze)")
    frozen_version = frozen_version or sorted(frozen)[-1]
    if frozen_version not in frozen:
        raise ValueError(f"jeu gelé inconnu : {frozen_version} (existants : {sorted(frozen)})")
    if frozen_version not in params.get("frozen_excluded", []):
        raise ValueError(
            f"la tête {params['version']} a été entraînée avant le gel de {frozen_version} : "
            "réentraîner (blanci train) avant de l'évaluer dessus"
        )
    ids = set(frozen[frozen_version]["recording_id"])
    store = store_for(cfg, encoder_id)
    meta, emb = store.load()
    keep = meta["recording_id"].isin(ids).to_numpy()
    if not keep.any():
        raise ValueError(
            f"aucun enregistrement du jeu gelé {frozen_version} encodé par {encoder_id}"
        )
    grid = (
        meta[keep].reset_index(drop=True).assign(dur_s=encoder_params(con, encoder_id)["window_s"])
    )
    data = training_set(con, grid, per_positive=0)
    if data.empty or data["y"].nunique() < 2:
        raise ValueError(f"jeu gelé {frozen_version} : labels absents ou d'une seule classe")
    scores = head.decision(emb[keep][data["row"].to_numpy()].astype(np.float32))
    y = data["y"].to_numpy()
    recordings = data["recording_id"].to_numpy()
    bench = cfg["benchmark"]
    out: dict[str, Any] = {
        "encoder_id": encoder_id,
        "head_version": params["version"],
        "frozen_version": frozen_version,
        "n_recordings": int(len(set(recordings))),
    }
    for level in ("recording", "window"):
        out[level] = evaluate(
            scores,
            y,
            recordings,
            level=level,
            precisions=tuple(bench["precisions"]),
            n_boot=bench["n_boot"],
            seed=cfg["head"]["seed"],
        )
    threshold = params["threshold"]
    hours = recordings_table(con).set_index("recording_id").loc[list(set(recordings)), "duration_s"]
    out["threshold"] = threshold
    out["recall_at_threshold"] = float((scores[y == 1] >= threshold).mean())
    out["false_alarms_per_hour"] = false_alarms_per_hour(scores, y, threshold, hours.sum() / 3600)
    strata = {
        "qualité": data["quality"].to_numpy() if "quality" in data else np.full(len(data), None),
        "site": data["site"].to_numpy(),
    }
    out["by_stratum"] = pd.concat(
        [recall_by_group(scores, y, v, threshold).assign(by=k) for k, v in strata.items()],
        ignore_index=True,
    )
    return out


# --- Module séquentiel et fusion (§3) -----------------------------------------------------------

# Frontière de décision de la tête logistique (classes équilibrées) : sert à compter les
# fenêtres « positives » des descripteurs de persistance, à l'entraînement comme au score.
PERSISTENCE_THRESHOLD = 0.0


def fusion_id(encoder_id: str, head_version: str) -> str:
    return f"{encoder_id}:fusion:{head_version}"


def _store_rows(store: EmbeddingStore, recording_ids: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Toutes les fenêtres encodées de ces enregistrements (persistance = tout l'enregistrement)."""
    metas, embs = [], []
    for path in store.fragments():
        meta, emb = store.read(path)
        keep = meta["recording_id"].isin(recording_ids).to_numpy()
        if keep.any():
            metas.append(meta[keep])
            embs.append(emb[keep])
    if not metas:
        return pd.DataFrame(columns=["window_id", "recording_id", "offset_s"]), np.zeros((0, 0))
    return pd.concat(metas, ignore_index=True), np.concatenate(embs).astype(np.float32)


def _sequential_features(
    windows: pd.DataFrame, persistence: pd.DataFrame, onsets: dict, cfg: dict
) -> pd.DataFrame:
    rhythm = window_rhythm(windows, onsets, tuple(cfg["signal"]["ioi_range_s"]))
    persist = persistence.reindex(windows["recording_id"]).set_index(windows.index)
    return pd.concat([rhythm, persist], axis=1)


def train_fusion(
    con: sqlite3.Connection, encoder_id: str, cfg: dict, version: str = "latest"
) -> dict[str, Any]:
    """Évalue puis enregistre la fusion (stacking, §3) au-dessus d'une tête enregistrée.

    1. Plis groupés par micro : dans chaque pli, une tête entraînée sans le micro donne le
       score hors-pli des fenêtres étiquetées **et** de toutes les fenêtres de leurs
       enregistrements (persistance) ;
    2. descripteurs : rythme dans la fenêtre (débuts de notes) + persistance de
       l'enregistrement ; colonnes retenues : `fusion.columns` ;
    3. fusion évaluée hors-pli sur les mêmes plis, comparée à la tête seule (bootstrap apparié
       par enregistrement) ; seuil de la fusion pris sur ses scores hors-pli ;
    4. fusion finale ajustée sur tout le jeu de développement, enregistrée en JSON.
    """
    _, params = load_head(con, encoder_id, version)
    version = params["version"]
    enc = encoder_params(con, encoder_id)
    bench, head_cfg = cfg["benchmark"], cfg["head"]
    columns = list(cfg["fusion"]["columns"])
    data, X = embedded_training_set(
        con,
        store_for(cfg, encoder_id),
        enc["window_s"],
        per_positive=bench["negatives_per_positive"],
        slot_tolerance_min=bench["slot_tolerance_min"],
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
        seed=head_cfg["seed"],
        exclude_recordings=frozen_recordings(cfg),
    )
    y, groups = data["y"].to_numpy(), data["point"].to_numpy()
    recordings = data["recording_id"].to_numpy()
    onsets = load_onsets(con, set(recordings))
    if not onsets:
        raise ValueError(
            "aucun début de note calculé pour ces enregistrements : ils se calculent pendant "
            "`blanci embed`, ou avec `blanci onsets`"
        )
    meta_all, emb_all = _store_rows(store_for(cfg, encoder_id), set(recordings))

    folds = grouped_folds(y, groups, head_cfg["n_splits"], head_cfg["seed"])
    head_oof = np.full(len(y), np.nan)
    persistence_parts = []
    for train, test in folds:
        if len(np.unique(groups[train])) >= 2:
            C, _ = select_C(
                X[train],
                y[train],
                groups[train],
                head_cfg["C_grid"],
                head_cfg["n_splits"],
                head_cfg["seed"],
            )
        else:
            C = head_cfg["C_grid"][0]
        fold_head = fit_logistic(X[train], y[train], C, head_cfg["seed"])
        head_oof[test] = fold_head.decision(X[test])
        mask = meta_all["recording_id"].isin(set(recordings[test])).to_numpy()
        scored = meta_all[mask].assign(score=fold_head.decision(emb_all[mask]))
        persistence_parts.append(recording_persistence(scored, PERSISTENCE_THRESHOLD))
    persistence = pd.concat(persistence_parts)
    windows = data.assign(dur_s=enc["window_s"])
    features = _sequential_features(windows, persistence, onsets, cfg)

    head_scores = OOFScores(head_oof, tuple(folds), "logistic")
    fused = fusion_oof(
        head_scores, features, y, groups, columns, head_cfg["n_splits"], head_cfg["seed"]
    )
    floor = min(bench["precisions"])
    recall, threshold = recall_at_precision(y, fused.values, floor)
    comparison = {}
    for level in ("recording", "window"):
        for name, values in (("head", head_oof), ("fusion", fused.values)):
            comparison[f"{name}_{level}"] = evaluate(
                values,
                y,
                recordings,
                level=level,
                precisions=tuple(bench["precisions"]),
                n_boot=bench["n_boot"],
                seed=head_cfg["seed"],
            )
    rec_head = to_recordings(head_oof, y, recordings).set_index("recording_id")
    rec_fused = to_recordings(fused.values, y, recordings)
    paired = paired_bootstrap(
        rec_fused["y"].to_numpy(),
        rec_fused["score"].to_numpy(),
        rec_head.loc[rec_fused["recording_id"], "score"].to_numpy(),
        rec_fused["recording_id"].to_numpy(),
        n_boot=bench["n_boot"],
        seed=head_cfg["seed"],
    )

    final = FusionWeights.from_fusion(fit_fusion(head_scores, features, y, columns))
    model_id = fusion_id(encoder_id, version)
    register_model(
        con,
        model_id,
        "fusion",
        encoder_id,
        version,
        {
            "weights": final.to_dict(),
            "threshold": threshold,
            "threshold_id": f"{model_id}:p{floor}",
            "min_precision": floor,
            "recall_at_threshold": recall,
            "persistence_threshold": PERSISTENCE_THRESHOLD,
            "head_model_id": head_id(encoder_id, version),
            "frozen_excluded": sorted(frozen_versions(cfg)),
            "ap_recording": {
                "head": comparison["head_recording"]["ap"],
                "fusion": comparison["fusion_recording"]["ap"],
            },
            "trained_at": utc_now(),
        },
    )
    return {
        "model_id": model_id,
        "columns": columns,
        "coefficients": dict(zip(["head", *columns], final.coef.tolist(), strict=True)),
        "threshold": threshold,
        "recall_at_threshold": recall,
        "comparison": comparison,
        "paired": paired,
        "n_with_onsets": int(pd.Series(recordings).isin(list(onsets)).sum()),
        "n_windows": len(y),
    }


def fused_scores(
    con: sqlite3.Connection,
    cfg: dict,
    fusion_params: dict,
    scored: pd.DataFrame,
    window_s: float,
) -> np.ndarray:
    """Scores de fusion de fenêtres déjà scorées par la tête (recording_id, offset_s, score)."""
    weights = FusionWeights.from_dict(fusion_params["weights"])
    persistence = recording_persistence(scored, fusion_params["persistence_threshold"])
    onsets = load_onsets(con, set(scored["recording_id"]))
    windows = scored.assign(dur_s=window_s)
    features = _sequential_features(windows, persistence, onsets, cfg)
    return weights.decision(scored["score"].to_numpy(), features)


# --- Jetons pour l'attentive probing (§3) --------------------------------------------------------


def compute_tokens(con: sqlite3.Connection, encoder: Any, cfg: dict) -> dict[str, int]:
    """Jetons des fenêtres du benchmark (étiquetées + négatifs appariés, jeu gelé exclu) pour
    un encodeur déjà passé par `embed` : seules ces fenêtres servent à la sonde attentive.

    Relit l'audio des enregistrements concernés (lecture seule) et repasse ces fenêtres dans
    l'encodeur : c'est un encodage, limité à ~1 500 fenêtres.
    """
    from blanci.attentive import TokenStore
    from blanci.audio import cut_windows, load_audio
    from blanci.encoders import encoder_id as make_encoder_id

    if not encoder.has_tokens:
        raise ValueError(f"{encoder.name} n'expose pas de jetons (seul perch_v2 aujourd'hui)")
    eid = make_encoder_id(encoder)
    bench = cfg["benchmark"]
    data, _ = embedded_training_set(
        con,
        store_for(cfg, eid),
        encoder.window_s,
        per_positive=bench["negatives_per_positive"],
        slot_tolerance_min=bench["slot_tolerance_min"],
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
        seed=cfg["head"]["seed"],
        exclude_recordings=frozen_recordings(cfg),
    )
    store = TokenStore(config_path(cfg, "tokens"), eid)
    done = set(store.read()[0]) if store.path.exists() else set()
    todo = data[~data["window_id"].isin(done)]
    paths = recordings_table(con).set_index("recording_id")["path"]
    report = {"windows": 0, "recordings": 0, "skipped": len(data) - len(todo)}
    for rid, group in todo.groupby("recording_id"):
        wav, sr = load_audio(config_path(cfg, "raw") / paths[rid], cfg["audio"]["channel"])
        windows = [(o, encoder.window_s) for o in group["offset_s"]]
        tokens = encoder.embed_tokens(cut_windows(wav, sr, windows), sr)
        store.write(group["window_id"].tolist(), tokens)
        report["windows"] += len(group)
        report["recordings"] += 1
    return report


# --- Réentraînement sans intervention et adoption (§4, jalon M5) --------------------------------

# Tête adoptée = celle que `blanci score` utilise par défaut. Une tête entraînée n'est adoptée
# qu'après avoir été jugée sur le jeu gelé : une tête plus récente n'est pas forcément
# meilleure (labels ajoutés bruités, tour d'annotation déséquilibré).


def adopted_version(con: sqlite3.Connection, encoder_id: str) -> str | None:
    """Version de la tête adoptée en dernier pour cet encodeur, ou None."""
    row = con.execute(
        "SELECT version FROM models WHERE kind = 'adoption' AND name = ? "
        "ORDER BY rowid DESC LIMIT 1",
        (encoder_id,),
    ).fetchone()
    return row[0] if row else None


def adopt_head(
    con: sqlite3.Connection, encoder_id: str, version: str, reason: str, evidence: dict
) -> str:
    """Inscrit l'adoption d'une tête (historique gardé : une ligne par adoption)."""
    load_head(con, encoder_id, version)  # la tête doit exister
    n = con.execute(
        "SELECT COUNT(*) FROM models WHERE kind = 'adoption' AND name = ?", (encoder_id,)
    ).fetchone()[0]
    model_id = f"{encoder_id}:adoption:{n + 1}"
    register_model(
        con,
        model_id,
        "adoption",
        encoder_id,
        version,
        {"head_version": version, "reason": reason, "adopted_at": utc_now(), **evidence},
    )
    return model_id


def _frozen_summary(result: dict[str, Any]) -> dict[str, Any]:
    rec = result["recording"]
    return {
        "head_version": result["head_version"],
        "ap_recording": rec["ap"],
        "ap_lo": rec["ap_lo"],
        "ap_hi": rec["ap_hi"],
        "recall_at_threshold": result["recall_at_threshold"],
        "false_alarms_per_hour": result["false_alarms_per_hour"],
        "n_recordings": result["n_recordings"],
    }


def retrain(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    frozen_version: str | None = None,
    tolerance: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Réentraîne la tête et ne l'adopte que si elle n'est pas moins bonne (ONF, §4).

    1. drapeaux posés à l'écoute recalculés (labels ajoutés depuis le dernier passage) ;
    2. nouvelle tête sur tous les labels hors jeu gelé, seuil calibré hors-pli
       (`train_and_register`) ;
    3. nouvelle tête et tête adoptée jugées sur le même jeu gelé, à leur seuil ;
    4. adoption si l'AP (niveau enregistrement) et le rappel au seuil de la nouvelle tête ne
       sont pas inférieurs de plus de `tolerance` à ceux de la tête adoptée. Sinon la tête
       adoptée reste en service ; la nouvelle reste en base, jamais effacée.

    Sans jeu gelé, rien ne permet de juger : pas d'adoption, sauf `force`. Une tête adoptée
    entraînée avant le gel ne peut pas être jugée dessus : la nouvelle est alors adoptée.
    """
    tolerance = cfg["retrain"]["tolerance"] if tolerance is None else tolerance
    apply_annotation_flags(con)
    trained = train_and_register(con, encoder_id, cfg)
    new_version, previous = trained.version, adopted_version(con, encoder_id)
    out: dict[str, Any] = {
        "encoder_id": encoder_id,
        "new_version": new_version,
        "previous_version": previous,
        "tolerance": tolerance,
        "train_metrics": trained.metrics,
    }
    if not frozen_versions(cfg):
        out["reason"] = "aucun jeu gelé : la nouvelle tête n'a pas pu être jugée"
        out["adopted"] = force
        if force:
            adopt_head(con, encoder_id, new_version, "forcée, sans jeu gelé", {})
        return out

    new = _frozen_summary(evaluate_frozen(con, encoder_id, cfg, new_version, frozen_version))
    out["new"] = new
    old = None
    if previous is not None:
        try:
            old = _frozen_summary(evaluate_frozen(con, encoder_id, cfg, previous, frozen_version))
        except ValueError as exc:  # tête adoptée entraînée avant le gel : pas jugeable dessus
            out["previous_not_judged"] = str(exc)
    out["previous"] = old
    if old is None:
        adopted, reason = True, "aucune tête adoptée jugeable sur ce jeu gelé"
    else:
        worse = [
            key
            for key in ("ap_recording", "recall_at_threshold")
            if new[key] < old[key] - tolerance
        ]
        adopted = not worse or force
        reason = (
            f"pas moins bonne que {previous} (tolérance {tolerance})"
            if not worse
            else f"moins bonne que {previous} sur {', '.join(worse)}"
            + (" ; adoption forcée" if force else "")
        )
    out["adopted"], out["reason"] = adopted, reason
    if adopted:
        out["adoption_id"] = adopt_head(
            con, encoder_id, new_version, reason, {"frozen_new": new, "frozen_previous": old}
        )
    return out


# --- Courbes d'activité (§6 niveau 3, jalon M4) ---------------------------------------------------


def activity_curves(
    con: sqlite3.Connection,
    encoder_id: str,
    cfg: dict,
    version: str = "default",
    fusion: bool = False,
    dataset: str | None = None,
    by: str = "site",
    reference: dict[str, pd.DataFrame] | None = None,
    include_suspect: bool = False,
) -> dict[str, Any]:
    """Courbes journalières et saisonnières des décisions d'une tête (ou de sa fusion),
    confrontées aux patrons publiés et, si fournies, à des courbes de référence numérisées
    (`reference` : {"hour": ..., "month": ...})."""
    _, params = load_head(con, encoder_id, version)
    version = params["version"]
    tid = params["threshold_id"]
    if fusion:
        tid = model_params(con, fusion_id(encoder_id, version), kind="fusion")["threshold_id"]
    detections = detections_table(
        con,
        encoder_id,
        version,
        tid,
        cfg["recorder"]["filename_utc_offset_h"],
        dataset=dataset,
        include_suspect=include_suspect,
    )
    diel = diel_index(detections, by)
    seasonal = daily_probability(detections, by)
    checks = reference_checks(diel, seasonal, cfg["activity"]["reference"], by)
    correlations = None
    if reference:
        parts = [
            curve_correlation(diel, reference["hour"], "hour", "index", by)
            if "hour" in reference
            else None,
            curve_correlation(seasonal, reference["month"], "month", "probability", by)
            if "month" in reference
            else None,
        ]
        correlations = pd.concat([p for p in parts if p is not None], ignore_index=True)
    return {
        "encoder_id": encoder_id,
        "head_version": version,
        "threshold_id": tid,
        "detections": detections,
        "diel": diel,
        "seasonal": seasonal,
        "checks": checks,
        "correlations": correlations,
    }
