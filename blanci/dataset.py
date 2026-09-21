"""Jeux d'apprentissage et d'évaluation.

Labels courants, transfert vers la grille, négatifs appariés.

- Transfert (DECISIONS n° 4) : une fenêtre de grille hérite du label d'une annotation si elle la
  contient (annotation courte, 3 s) ou si elle y est contenue (annotation d'enregistrement
  entier, §5). Les fenêtres qui ne font que chevaucher l'annotation sont écartées.
- Négatifs appariés (§2) : fenêtres des mêmes micros, au même créneau horaire (± tolérance),
  d'autres jours, dans des enregistrements sans label positif. Ce sont des négatifs *présumés*
  (colonne `presumed`) : jamais écrits dans la table labels. En saison, à l'heure de pic, une
  partie peut contenir A. blanci : bruit d'étiquette identique pour tous les encodeurs.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from blanci.labels import POSITIVE_LABELS

EXCLUDED_LABELS = ("blanci_uncertain", "uncertain")


def current_labels(con: sqlite3.Connection) -> pd.DataFrame:
    """Dernier label de chaque fenêtre annotée (une correction est une ligne plus récente)."""
    return pd.read_sql_query(
        """SELECT l.label_id, l.window_id, l.label, l.quality, l.species, l.source,
                  w.recording_id, w.offset_s, w.dur_s
           FROM labels l JOIN windows w USING (window_id)
           WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)""",
        con,
    )


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
    return kept[["window_id", "recording_id", "offset_s", "label", "y"]].reset_index(drop=True)


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
) -> pd.DataFrame:
    """Fenêtres étiquetées de la grille (+ négatifs appariés présumés si per_positive > 0).

    `grid` = métadonnées du stock d'embeddings (window_id, recording_id, offset_s), avec dur_s.
    Renvoie aussi `row` (indice dans `grid`), `point` (groupe des plis) et `site`.
    """
    grid = grid.reset_index(drop=True)
    labeled = transfer_labels(current_labels(con), grid)
    parts = [labeled.assign(presumed=False)]
    recordings = recordings_table(con)
    if per_positive > 0:
        positives = set(labeled.loc[labeled["y"] == 1, "recording_id"])
        negatives = paired_negatives(
            grid[~grid["window_id"].isin(labeled["window_id"])],
            recordings,
            positives,
            per_positive,
            slot_tolerance_min,
            utc_offset_h,
            seed,
        )
        parts.append(negatives.assign(presumed=True))
    data = pd.concat(parts, ignore_index=True)
    row_of = pd.Series(np.arange(len(grid)), index=grid["window_id"])
    data["row"] = row_of.loc[data["window_id"]].to_numpy()
    return data.merge(recordings[["recording_id", "point", "site"]], on="recording_id", how="left")
