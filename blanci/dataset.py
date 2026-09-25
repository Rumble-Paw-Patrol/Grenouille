"""Jeux d'apprentissage et d'évaluation.

Labels courants, transfert vers la grille, négatifs appariés.

- Transfert (DECISIONS n° 4) : une fenêtre de grille hérite du label d'une annotation si elle la
  contient (annotation courte, 3 s) ou si elle y est contenue (annotation d'enregistrement
  entier, §5). Les fenêtres qui ne font que chevaucher l'annotation sont écartées.
- Négatifs appariés (§2) : fenêtres du même micro, dans des enregistrements sans label positif.
  Ce sont des négatifs *présumés* (colonne `presumed`) : jamais écrits dans la table labels.
  En saison, à l'heure de pic, une partie peut contenir A. blanci : bruit d'étiquette identique
  pour tous les encodeurs. Quatre stratégies (DECISIONS n° 88, 101, `benchmark.pairing`) :
  - `nearest` (défaut) : les fenêtres les plus proches dans le temps, même enregistrement
    compris — hors fenêtres qui chevauchent une annotation positive ou qui sont encadrées de
    positifs (faux négatifs suspects, n° 102) ; puis le même jour, puis un autre jour. Risque :
    si A. blanci chante tout l'enregistrement (H20), les voisines d'un positif en contiennent ;
  - `other_day` : même créneau horaire (± tolérance), un **autre** jour ;
  - `same_day` : même jour, les enregistrements les plus proches à au moins `min_gap_min`
    (jusqu'à `min_gap_min` + tolérance). Meilleur témoin du fond (même météo, même chœur),
    mais A. blanci chante par épisodes : le plus proche est aussi le plus exposé ;
  - `mixed` : moitié l'un, moitié l'autre (l'un complète l'autre s'il manque de fenêtres).
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
from blanci.sequential import GAP_RADIUS_S, surrounded_by_positives
from blanci.store import EmbeddingStore, gated_mask

EXCLUDED_LABELS = ("blanci_uncertain", "uncertain")
PAIRING_STRATEGIES = ("nearest", "other_day", "same_day", "mixed")


def pairing_options(cfg: dict) -> dict:
    """Réglages des négatifs appariés lus dans la config, à passer tels quels (**) à
    `paired_negatives`, `training_set`, `embedded_training_set` et `benchmark_recordings`."""
    bench = cfg["benchmark"]
    return {
        "slot_tolerance_min": bench["slot_tolerance_min"],
        "utc_offset_h": cfg["recorder"]["filename_utc_offset_h"],
        "strategy": bench.get("pairing", "nearest"),
        "min_gap_min": bench.get("same_day_min_gap_min", 30),
        "gap_radius_s": cfg.get("sequential", {}).get("gap_radius_s", GAP_RADIUS_S),
    }


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


def local_days(start_utc: pd.Series, utc_offset_h: float) -> pd.Series:
    """Jour calendaire en heure locale (« même jour » des négatifs appariés)."""
    t = pd.to_datetime(start_utc, utc=True) + pd.Timedelta(hours=utc_offset_h)
    return t.dt.strftime("%Y-%m-%d")


def _pairing_pools(
    rec: pd.DataFrame,
    rid: str,
    strategy: str,
    slot_tolerance_min: float,
    min_gap_min: float,
    utc_offset_h: float,
) -> tuple[list[str], list[list[str]]]:
    """Enregistrements du même micro éligibles pour un positif : (autre jour, même jour).

    Le second est ordonné par proximité : une liste de rangs, chacun groupant les
    enregistrements à la même distance du positif (avant et après), du plus proche au plus loin.
    `rec` : indexé par recording_id, colonnes point, _minute, _day.
    """
    if strategy not in PAIRING_STRATEGIES:
        raise ValueError(f"stratégie inconnue : {strategy!r} (attendues : {PAIRING_STRATEGIES})")
    same_point = rec[(rec["point"] == rec.at[rid, "point"]) & (rec.index != rid)]
    minute, day = rec.at[rid, "_minute"], rec.at[rid, "_day"]
    other_day: list[str] = []
    same_day: list[list[str]] = []
    if strategy in ("other_day", "mixed", "nearest"):
        gap = (same_point["_minute"] - minute).abs()
        gap = np.minimum(gap, 24 * 60 - gap)
        other_day = same_point.index[(gap <= slot_tolerance_min) & (same_point["_day"] != day)]
        other_day = list(other_day)
    if strategy in ("same_day", "mixed", "nearest"):
        today = same_point[same_point["_day"] == day]
        gap = (today["_minute"] - minute).abs()
        kept = gap[(gap >= min_gap_min) & (gap <= min_gap_min + slot_tolerance_min)]
        same_day = [list(g.index) for _, g in kept.groupby(kept)]  # groupby trie les écarts
    return other_day, same_day


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
    strategy: str = "nearest",
    min_gap_min: float = 30,
    positive_windows: pd.DataFrame | None = None,
    gap_radius_s: float = GAP_RADIUS_S,
) -> pd.DataFrame:
    """`per_positive` fenêtres présumées négatives par enregistrement positif, du même micro,
    selon `strategy` (voir le module). Colonne `pairing` : same_recording, same_day ou
    other_day, pour comparer les sortes de négatifs.

    `nearest` a besoin de `positive_windows` (recording_id, offset_s, dur_s : les annotations
    positives) : dans l'enregistrement positif, une fenêtre qui chevauche une annotation
    positive n'est jamais tirée, ni une fenêtre encadrée d'annotations positives à moins de
    `gap_radius_s` (faux négatif suspect, DECISIONS n° 102). `grid` doit alors contenir les
    fenêtres des enregistrements positifs, avec `dur_s`.
    """
    rng = np.random.default_rng(seed)
    rec = recordings.set_index("recording_id").assign(
        _minute=lambda r: local_minutes(r["start_utc"], utc_offset_h),
        _day=lambda r: local_days(r["start_utc"], utc_offset_h),
    )
    rec = rec[~rec.index.duplicated()]
    candidates = grid[~grid["recording_id"].isin(positive_recording_ids)]
    by_recording = {rid: g.index.to_numpy() for rid, g in candidates.groupby("recording_id")}
    chosen: dict[int, str] = {}
    if strategy == "nearest":
        if positive_windows is None:
            raise ValueError("stratégie nearest : les annotations positives sont nécessaires")
        own = {
            rid: g
            for rid, g in grid[grid["recording_id"].isin(positive_recording_ids)].groupby(
                "recording_id"
            )
        }
        annotations = {rid: g for rid, g in positive_windows.groupby("recording_id")}

    def free(recording_ids: list[str]) -> list[int]:
        return [
            i
            for r in recording_ids
            if r in by_recording
            for i in by_recording[r]
            if i not in chosen
        ]

    def draw(pool: list[int], n: int, kind: str) -> int:
        if n <= 0 or not pool:
            return 0
        picked = rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()
        chosen.update(dict.fromkeys(picked, kind))
        return len(picked)

    def draw_nearest(ranks: list[list[str]], n: int) -> int:
        taken = 0
        for rank in ranks:
            if taken >= n:
                break
            taken += draw(free(rank), n - taken, "same_day")
        return taken

    def draw_same_recording(rid: str, n: int) -> int:
        """Les fenêtres de l'enregistrement positif les plus proches d'une annotation positive,
        sans la chevaucher et sans être encadrées de positifs."""
        windows, positive = own.get(rid), annotations.get(rid)
        if windows is None or positive is None or n <= 0:
            return 0
        w0 = windows["offset_s"].to_numpy(dtype=float)
        w1 = w0 + windows["dur_s"].to_numpy(dtype=float)
        p0 = positive["offset_s"].to_numpy(dtype=float)
        p1 = p0 + positive["dur_s"].to_numpy(dtype=float)
        overlap = (w0[:, None] < p1[None, :] - 1e-6) & (w1[:, None] > p0[None, :] + 1e-6)
        gap = np.maximum(p0[None, :] - w1[:, None], w0[:, None] - p1[None, :]).clip(min=0)
        distance = gap.min(axis=1)
        suspect = surrounded_by_positives((w0 + w1) / 2, (p0 + p1) / 2, gap_radius_s)
        order = np.lexsort((w0, distance))
        picked = [
            windows.index[j]
            for j in order
            if not overlap[j].any() and not suspect[j] and windows.index[j] not in chosen
        ][:n]
        chosen.update(dict.fromkeys(picked, "same_recording"))
        return len(picked)

    for rid in sorted(positive_recording_ids):
        if rid not in rec.index:
            continue
        other_day, same_day = _pairing_pools(
            rec, rid, strategy, slot_tolerance_min, min_gap_min, utc_offset_h
        )
        if strategy == "nearest":  # même enregistrement, puis même jour, puis un autre jour
            taken = draw_same_recording(rid, per_positive)
            taken += draw_nearest(same_day, per_positive - taken)
            draw(free(other_day), per_positive - taken, "other_day")
        elif strategy == "other_day":
            draw(free(other_day), per_positive, "other_day")
        elif strategy == "same_day":
            draw_nearest(same_day, per_positive)
        else:  # mixed : moitié autre jour ; le même jour complète, puis l'autre jour
            taken = draw(free(other_day), (per_positive + 1) // 2, "other_day")
            taken += draw_nearest(same_day, per_positive - taken)
            draw(free(other_day), per_positive - taken, "other_day")
    rows = sorted(chosen)
    out = grid.loc[rows, ["window_id", "recording_id", "offset_s"]].copy()
    out["label"], out["y"] = "background_presumed", 0
    out["pairing"] = [chosen[i] for i in rows]
    return out.reset_index(drop=True)


def positive_annotations(
    labels: pd.DataFrame, exclude_recordings: set[str] | None = None
) -> pd.DataFrame:
    """Annotations positives courantes (recording_id, offset_s, dur_s), jeu gelé exclu."""
    positive = labels[labels["label"].isin(POSITIVE_LABELS)]
    if exclude_recordings:
        positive = positive[~positive["recording_id"].isin(exclude_recordings)]
    return positive[["recording_id", "offset_s", "dur_s"]].reset_index(drop=True)


def suspect_false_negatives(
    windows: pd.DataFrame,
    grid: pd.DataFrame,
    positive_windows: pd.DataFrame,
    radius: float = GAP_RADIUS_S,
) -> np.ndarray:
    """Pour chaque fenêtre négative de `windows` (window_id, recording_id, offset_s, y) : vrai
    si des annotations positives l'encadrent à moins de `radius` (même enregistrement)."""
    out = np.zeros(len(windows), dtype=bool)
    if windows.empty or positive_windows.empty:
        return out
    dur = grid.drop_duplicates("window_id").set_index("window_id")["dur_s"]
    half = dur.reindex(windows["window_id"]).fillna(0.0).to_numpy(dtype=float) / 2
    centers = windows["offset_s"].to_numpy(dtype=float) + half
    by_recording = {
        rid: (g["offset_s"] + g["dur_s"] / 2).to_numpy(dtype=float)
        for rid, g in positive_windows.groupby("recording_id")
    }
    negative = windows["y"].to_numpy() == 0
    rids = windows["recording_id"].to_numpy()
    for rid in set(rids[negative]) & set(by_recording):
        rows = np.flatnonzero(negative & (rids == rid))
        out[rows] = surrounded_by_positives(centers[rows], by_recording[rid], radius)
    return out


def training_set(
    con: sqlite3.Connection,
    grid: pd.DataFrame,
    per_positive: int = 0,
    slot_tolerance_min: float = 30,
    utc_offset_h: float = -3,
    seed: int = 0,
    exclude_recordings: set[str] | None = None,
    strategy: str = "nearest",
    min_gap_min: float = 30,
    gap_radius_s: float = GAP_RADIUS_S,
) -> pd.DataFrame:
    """Fenêtres étiquetées de la grille (+ négatifs appariés présumés si per_positive > 0).

    `grid` = métadonnées du stock d'embeddings (window_id, recording_id, offset_s), avec dur_s.
    Renvoie aussi `row` (indice dans `grid`), `point` (groupe des plis) et `site`.
    `exclude_recordings` (le jeu gelé, §6) : ni leurs labels, ni leurs fenêtres comme
    négatifs appariés. `row` reste l'indice dans la grille complète.
    Colonne `suspect_fn` : négatif annoté encadré d'annotations positives à moins de
    `gap_radius_s` — faux négatif suspect, à réécouter ; il reste négatif (n° 85, 102).
    """
    grid = grid.reset_index(drop=True)
    if exclude_recordings:
        grid = grid[~grid["recording_id"].isin(exclude_recordings)]
    annotations = current_labels(con)
    labeled = transfer_labels(annotations, grid)
    positive_windows = positive_annotations(annotations, exclude_recordings)
    labeled["suspect_fn"] = suspect_false_negatives(labeled, grid, positive_windows, gap_radius_s)
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
            strategy,
            min_gap_min,
            positive_windows,
            gap_radius_s,
        )
        parts.append(negatives.assign(presumed=True, suspect_fn=False))
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
    strategy: str = "nearest",
    min_gap_min: float = 30,
    gap_radius_s: float = GAP_RADIUS_S,
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
        strategy=strategy,
        min_gap_min=min_gap_min,
        gap_radius_s=gap_radius_s,
    )
    if data.empty:
        raise ValueError(f"aucune fenêtre étiquetée dans le stock de {store.encoder_id}")
    data["gated"] = gated_mask(meta)[data["row"].to_numpy()]  # seuillage en amont
    return data, emb[data["row"].to_numpy()].astype(np.float32)


def benchmark_recordings(
    con: sqlite3.Connection,
    slot_tolerance_min: float = 30,
    utc_offset_h: float = -3,
    exclude_flags: tuple[str, ...] = EXCLUDING_FLAGS,
    strategy: str = "nearest",
    min_gap_min: float = 30,
    gap_radius_s: float = GAP_RADIUS_S,
) -> pd.DataFrame:
    """Enregistrements dont le benchmark a besoin : les annotés, plus les candidats aux
    négatifs appariés de la stratégie `strategy` (même micro, sans positif).

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
    rec = recordings.set_index("recording_id").assign(
        _minute=lambda r: local_minutes(r["start_utc"], utc_offset_h),
        _day=lambda r: local_days(r["start_utc"], utc_offset_h),
    )
    rec = rec[~rec.index.duplicated()]
    usable = set(recordings.loc[~flagged, "recording_id"])
    candidates: set[str] = set()
    for rid in positives:
        if rid not in rec.index:
            continue
        other_day, same_day = _pairing_pools(
            rec, rid, strategy, slot_tolerance_min, min_gap_min, utc_offset_h
        )
        candidates.update(r for r in other_day if r in usable)
        candidates.update(r for rank in same_day for r in rank if r in usable)
    candidates -= positives

    out = recordings[recordings["recording_id"].isin(labelled | candidates)].copy()
    out["role"] = np.where(out["recording_id"].isin(labelled), "labelled", "paired_candidate")
    return out.sort_values("path").reset_index(drop=True)


def benchmark_subset(con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Enregistrements à encoder pour le benchmark : annotés + candidats des **deux**
    stratégies de négatifs appariés (autre jour et même jour). Changer `benchmark.pairing`
    ne demande alors pas de nouvel encodage."""
    return benchmark_recordings(con, **(pairing_options(cfg) | {"strategy": "mixed"}))


def benchmark_folds(
    con: sqlite3.Connection,
    n_splits: int = 5,
    seed: int = 0,
    exclude_recordings: set[str] | None = None,
) -> dict[str, int]:
    """Pli de chaque point (micro), commun à tous les modèles (DECISIONS n° 91).

    Calculé sur les enregistrements annotés (hors jeu gelé `exclude_recordings`), jamais sur
    les fenêtres d'un encodeur : deux encodeurs, une baseline ou une fusion voient exactement
    les mêmes micros tenus à l'écart dans chaque pli.
    """
    from blanci.evaluate import fold_assignment

    labels = current_labels(con)
    labels = labels[~labels["label"].isin(EXCLUDED_LABELS)]
    if exclude_recordings:
        labels = labels[~labels["recording_id"].isin(exclude_recordings)]
    per_recording = (
        labels.assign(pos=labels["label"].isin(POSITIVE_LABELS))
        .groupby("recording_id")["pos"]
        .max()
    )
    points = recordings_table(con).set_index("recording_id")["point"]
    points = points[~points.index.duplicated()]
    known = per_recording.index.intersection(points.index)
    return fold_assignment(
        points.loc[known].to_numpy(), per_recording.loc[known].to_numpy(), n_splits, seed
    )


def folds_for(con: sqlite3.Connection, cfg: dict) -> dict[str, int]:
    """`benchmark_folds` avec les réglages de la config (plis, graine, jeu gelé exclu)."""
    from blanci.frozen import frozen_recordings

    return benchmark_folds(
        con, cfg["head"]["n_splits"], cfg["head"]["seed"], frozen_recordings(cfg)
    )
