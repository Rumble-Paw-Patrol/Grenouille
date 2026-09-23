"""Jeux d'apprentissage et d'évaluation.

Labels courants, transfert vers la grille, négatifs appariés.

- Transfert (DECISIONS n° 4) : une fenêtre de grille hérite du label d'une annotation si elle la
  contient (annotation courte, 3 s) ou si elle y est contenue (annotation d'enregistrement
  entier, §5). Les fenêtres qui ne font que chevaucher l'annotation sont écartées.
- Négatifs appariés (§2) : fenêtres des mêmes micros, au même créneau horaire (± tolérance),
  d'autres jours, dans des enregistrements sans label positif. Ce sont des négatifs *présumés*
  (colonne `presumed`) : jamais écrits dans la table labels. En saison, à l'heure de pic, une
  partie peut contenir A. blanci : bruit d'étiquette identique pour tous les encodeurs.
  Aucun n'est tiré à moins de `DETECTION_MARGIN_S` d'une détection Blancinet non écoutée
  (DECISIONS n° 80, 85) : ce serait prendre pour négatif une fenêtre où un détecteur
  indépendant entend A. blanci.
- Négatifs annotés : négatifs quel que soit le contexte ajouté par les encodeurs à fenêtre
  de 5–6 s (DECISIONS n° 85). Si A. blanci ne chante pas pendant les 3 s écoutées, qu'elle
  commence juste après est peu probable ; biais possible, gardé en tête.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from blanci.labels import POSITIVE_LABELS
from blanci.qc import EXCLUDING_FLAGS, is_excluded
from blanci.store import EmbeddingStore

EXCLUDED_LABELS = ("blanci_uncertain", "uncertain")

# « A. blanci détecté » : détection non écoutée d'un détecteur indépendant (Blancinet,
# `blanci import-detections`) à ce score au moins, ou positif annoté. Jamais les scores de
# nos propres têtes : ils serviraient à choisir les labels qui les jugent.
INDEPENDENT_DETECTORS = ("blancinet",)
DETECTION_MIN_SCORE = 0.5
# Voisinage d'une détection : la fenêtre de 3 s de chaque côté.
DETECTION_MARGIN_S = 3.0


def _comment(conditions: str | None) -> str | None:
    if not conditions:
        return None
    return json.loads(conditions).get("comment") or None


def current_labels(con: sqlite3.Connection) -> pd.DataFrame:
    """Dernier label de chaque fenêtre annotée (une correction est une ligne plus récente),
    avec le commentaire de l'annotateur (`comment`, texte brut, vide s'il n'y en a pas)."""
    df = pd.read_sql_query(
        """SELECT l.label_id, l.window_id, l.label, l.quality, l.species, l.source,
                  l.conditions, w.recording_id, w.offset_s, w.dur_s
           FROM labels l JOIN windows w USING (window_id)
           WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)""",
        con,
    )
    df["comment"] = df["conditions"].map(_comment)
    return df.drop(columns="conditions")


def detected_blanci(
    con: sqlite3.Connection,
    labels: pd.DataFrame,
    detectors: tuple[str, ...] = INDEPENDENT_DETECTORS,
    min_score: float = DETECTION_MIN_SCORE,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Où A. blanci est détecté : positifs annotés, et détections ≥ `min_score` que personne
    n'a écoutées (aucune fenêtre annotée ne les contient). window_id, recording_id,
    offset_s, dur_s, score (vide pour un positif annoté)."""
    marks = ", ".join("?" * len(detectors))
    found = pd.read_sql_query(
        f"""SELECT s.window_id, w.recording_id, w.offset_s, w.dur_s, s.score
            FROM scores s JOIN windows w USING (window_id)
            WHERE s.model_id IN ({marks}) AND s.score >= ?""",
        con,
        params=(*detectors, min_score),
    )
    if len(found) and len(labels):
        pairs = found.reset_index(names="_row").merge(
            labels[["recording_id", "offset_s", "dur_s"]], on="recording_id", suffixes=("", "_a")
        )
        heard = (pairs["offset_s_a"] <= pairs["offset_s"] + eps) & (
            pairs["offset_s_a"] + pairs["dur_s_a"] >= pairs["offset_s"] + pairs["dur_s"] - eps
        )
        found = found.drop(index=pairs.loc[heard, "_row"].unique())
    positives = labels[labels["label"].isin(POSITIVE_LABELS)].assign(score=np.nan)
    columns = ["window_id", "recording_id", "offset_s", "dur_s", "score"]
    parts = [part[columns] for part in (found, positives) if len(part)]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def near_detection(
    windows: pd.DataFrame, detected: pd.DataFrame, margin_s: float = DETECTION_MARGIN_S
) -> pd.Series:
    """Fenêtres (recording_id, offset_s, dur_s) à moins de `margin_s` d'une détection."""
    near = pd.Series(False, index=windows.index)
    candidates = windows[windows["recording_id"].isin(set(detected["recording_id"]))]
    if candidates.empty:
        return near
    pairs = candidates[["recording_id", "offset_s", "dur_s"]].reset_index(names="_row")
    pairs = pairs.merge(detected, on="recording_id", suffixes=("", "_d"))
    close = (pairs["offset_s_d"] < pairs["offset_s"] + pairs["dur_s"] + margin_s) & (
        pairs["offset_s_d"] + pairs["dur_s_d"] > pairs["offset_s"] - margin_s
    )
    near[pairs.loc[close, "_row"].unique()] = True
    return near


def recordings_table(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, qc_flags "
        "FROM recordings",
        con,
    )
    df["point"] = df["site"].fillna("?") + "/" + df["mic_id"].fillna("?")  # groupe « micro »
    return df


def local_minutes(start_utc: pd.Series, utc_offset_h: float) -> pd.Series:
    """Minute du jour en heure locale."""
    t = pd.to_datetime(start_utc, utc=True) + pd.Timedelta(hours=utc_offset_h)
    return t.dt.hour * 60 + t.dt.minute


def transfer_labels(
    annotations: pd.DataFrame, grid: pd.DataFrame, eps: float = 1e-6
) -> pd.DataFrame:
    """Labels des fenêtres de grille. annotations : recording_id, offset_s, dur_s, label ;
    grid : window_id, recording_id, offset_s, dur_s (dur_s = fenêtre de l'encodeur)."""
    merged = grid.merge(annotations, on="recording_id", suffixes=("", "_ann"))
    g0, g1 = merged["offset_s"], merged["offset_s"] + merged["dur_s"]
    a0, a1 = merged["offset_s_ann"], merged["offset_s_ann"] + merged["dur_s_ann"]
    contains = (g0 <= a0 + eps) & (g1 >= a1 - eps)
    inside = (g0 >= a0 - eps) & (g1 <= a1 + eps)
    kept = merged[(contains | inside) & ~merged["label"].isin(EXCLUDED_LABELS)].copy()
    kept["y"] = kept["label"].isin(POSITIVE_LABELS).astype(int)
    conflicts = kept.groupby("window_id")["y"].nunique()
    kept = kept[~kept["window_id"].isin(conflicts[conflicts > 1].index)]
    kept = kept.sort_values("y", ascending=False).drop_duplicates("window_id")
    columns = ["window_id", "recording_id", "offset_s", "label", "y"]
    if "quality" in kept:  # qualité A/B/C de l'annotation : rappel par qualité (§6)
        columns.append("quality")
    if "comment" in kept:  # le commentaire de l'annotateur suit la fenêtre
        columns.append("comment")
    return kept[columns].reset_index(drop=True)


def paired_negatives(
    grid: pd.DataFrame,
    recordings: pd.DataFrame,
    positive_recording_ids: set[str],
    per_positive: int,
    slot_tolerance_min: float = 30,
    utc_offset_h: float = -3,
    seed: int = 0,
) -> pd.DataFrame:
    """`per_positive` fenêtres présumées négatives par enregistrement positif (même micro,
    même créneau horaire local, autre enregistrement sans label positif)."""
    rng = np.random.default_rng(seed)
    rec = recordings.set_index("recording_id")
    minutes = local_minutes(rec["start_utc"], utc_offset_h)
    candidates = grid[~grid["recording_id"].isin(positive_recording_ids)]
    by_recording = {rid: g.index.to_numpy() for rid, g in candidates.groupby("recording_id")}
    chosen: set[int] = set()
    for rid in sorted(positive_recording_ids):
        if rid not in rec.index:
            continue
        same_point = rec.index[(rec["point"] == rec.at[rid, "point"])]
        gap = (minutes[same_point] - minutes[rid]).abs()
        gap = np.minimum(gap, 24 * 60 - gap)
        pool = [
            i
            for r in same_point[gap <= slot_tolerance_min]
            if r in by_recording
            for i in by_recording[r]
            if i not in chosen
        ]
        if pool:
            chosen.update(
                rng.choice(pool, size=min(per_positive, len(pool)), replace=False).tolist()
            )
    out = grid.loc[sorted(chosen), ["window_id", "recording_id", "offset_s"]].copy()
    out["label"], out["y"] = "background_presumed", 0
    return out.reset_index(drop=True)


def training_set(
    con: sqlite3.Connection,
    grid: pd.DataFrame,
    per_positive: int = 0,
    slot_tolerance_min: float = 30,
    utc_offset_h: float = -3,
    seed: int = 0,
    exclude_recordings: set[str] | None = None,
) -> pd.DataFrame:
    """Fenêtres étiquetées de la grille (+ négatifs appariés présumés si per_positive > 0).

    `grid` = métadonnées du stock d'embeddings (window_id, recording_id, offset_s), avec dur_s.
    Renvoie aussi `row` (indice dans `grid`), `point` (groupe des plis) et `site`.
    `exclude_recordings` (le jeu gelé, §6) : ni leurs labels, ni leurs fenêtres comme
    négatifs appariés. `row` reste l'indice dans la grille complète.
    """
    grid = grid.reset_index(drop=True)
    if exclude_recordings:
        grid = grid[~grid["recording_id"].isin(exclude_recordings)]
    labels = current_labels(con)
    labeled = transfer_labels(labels, grid)
    parts = [labeled.assign(presumed=False)]
    recordings = recordings_table(con)
    if per_positive > 0:
        positives = set(labeled.loc[labeled["y"] == 1, "recording_id"])
        # Pas de négatif présumé à côté d'une détection non écoutée (DECISIONS n° 80, 85).
        pool = grid[~grid["window_id"].isin(labeled["window_id"])]
        pool = pool[~near_detection(pool, detected_blanci(con, labels))]
        negatives = paired_negatives(
            pool,
            recordings,
            positives,
            per_positive,
            slot_tolerance_min,
            utc_offset_h,
            seed,
        )
        parts.append(negatives.assign(presumed=True))
    data = pd.concat(parts, ignore_index=True)
    row_of = pd.Series(grid.index.to_numpy(), index=grid["window_id"])
    data["row"] = row_of.loc[data["window_id"]].to_numpy()
    return data.merge(recordings[["recording_id", "point", "site"]], on="recording_id", how="left")


def embedded_training_set(
    con: sqlite3.Connection,
    store: EmbeddingStore,
    window_s: float,
    per_positive: int = 0,
    slot_tolerance_min: float = 30,
    utc_offset_h: float = -3,
    seed: int = 0,
    filters: dict | None = None,
    exclude_recordings: set[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fenêtres étiquetées du stock d'un encodeur, avec leurs embeddings alignés.

    Point d'entrée commun du benchmark (§2) et de l'entraînement de la tête (§4) : les deux
    doivent voir exactement les mêmes labels et les mêmes négatifs appariés.
    """
    meta, emb = store.load(filters)
    if not len(meta):
        raise ValueError(f"aucun embedding pour {store.encoder_id} (filtres : {filters})")
    data = training_set(
        con,
        meta.assign(dur_s=window_s),
        per_positive=per_positive,
        slot_tolerance_min=slot_tolerance_min,
        utc_offset_h=utc_offset_h,
        seed=seed,
        exclude_recordings=exclude_recordings,
    )
    if data.empty:
        raise ValueError(f"aucune fenêtre étiquetée dans le stock de {store.encoder_id}")
    return data, emb[data["row"].to_numpy()].astype(np.float32)


def benchmark_recordings(
    con: sqlite3.Connection,
    slot_tolerance_min: float = 30,
    utc_offset_h: float = -3,
    exclude_flags: tuple[str, ...] = EXCLUDING_FLAGS,
) -> pd.DataFrame:
    """Enregistrements dont le benchmark a besoin : les annotés, plus les candidats aux
    négatifs appariés (même micro, créneau horaire à ± `slot_tolerance_min`, sans positif).

    C'est tout ce que `paired_negatives` peut tirer : encoder ce sous-ensemble suffit au §2,
    sans passer les 29 000 enregistrements dans chaque encodeur. Colonne `role` ∈
    {labelled, paired_candidate}. Un candidat signalé (drapeaux QC) est écarté.
    """
    recordings = recordings_table(con)
    labels = current_labels(con)
    labels = labels[~labels["label"].isin(EXCLUDED_LABELS)]
    labelled = set(labels["recording_id"])
    positives = set(labels.loc[labels["label"].isin(POSITIVE_LABELS), "recording_id"])

    flagged = recordings["qc_flags"].map(lambda q: is_excluded(q, exclude_flags))
    minutes = local_minutes(recordings["start_utc"], utc_offset_h)
    candidates: set[str] = set()
    for rid in positives:
        this = recordings["recording_id"] == rid
        if not this.any():
            continue
        gap = (minutes - minutes[this].iloc[0]).abs()
        gap = np.minimum(gap, 24 * 60 - gap)
        same_slot = (recordings["point"] == recordings.loc[this, "point"].iloc[0]) & (
            gap <= slot_tolerance_min
        )
        candidates.update(recordings.loc[same_slot & ~flagged, "recording_id"])
    candidates -= positives

    out = recordings[recordings["recording_id"].isin(labelled | candidates)].copy()
    out["role"] = np.where(out["recording_id"].isin(labelled), "labelled", "paired_candidate")
    return out.sort_values("path").reset_index(drop=True)
