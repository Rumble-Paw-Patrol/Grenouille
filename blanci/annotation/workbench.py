"""Poste d'annotation (§5, M2) : candidats à écouter, extraits, écoute, labels.

Logique pure, sans interface : l'application Streamlit (`blanci/annotation/app.py`) n'en est qu'un
affichage, et la future GUI du livrable appellera les mêmes fonctions (§4).

Candidats = une file CSV (recording_id, offset_s, dur_s, reason, source, score…) :
- `random_candidates` : fenêtres tirées au hasard (site, micro, heure), qui mesurent ce
  qu'aucun détecteur n'a remonté ;
- les files de l'outil de sélection (`selection.py`, `blanci select`) et du tirage par plan
  (`plan.py`, `blanci candidates --plan`).

Chaque réponse est un label en ajout seul (`service.append_label`) ; la fenêtre est créée si
elle n'existe pas. L'audio n'est que lu.
"""

from __future__ import annotations

import io
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.core.db import window_id_for
from blanci.embedding.embed import select_recordings
from blanci.embedding.grid import window_grid
from blanci.inputs.dataset import local_minutes, recordings_table
from blanci.inputs.labels import (
    comment_fields,
)
from blanci.service import INTERVAL_LABELS, append_label, append_span

CANDIDATE_COLUMNS = [
    "recording_id",
    "path",
    "site",
    "mic_id",
    "start_utc",
    "offset_s",
    "dur_s",
    "score",
    "reason",
    "source",
]

# Réponses proposées à l'écoute : labels du schéma (§5), dans l'ordre des boutons.
ANSWERS = (
    ("blanci", "A. blanci"),
    ("blanci_chorus", "A. blanci, plusieurs"),
    ("blanci_uncertain", "A. blanci ?"),
    ("false_friend", "faux ami"),
    ("amphibian", "autre amphibien"),
    ("amphibian_contact_call", "cri de contact"),
    ("bird", "oiseau"),
    ("orthoptera", "insecte"),
    ("rain", "pluie"),
    ("background", "rien"),
    ("other", "autre"),
)


# --- Constitution des files ---------------------------------------------------------------------


def _round_robin(
    frame: pd.DataFrame,
    by: str,
    n: int,
    rng: np.random.Generator,
    used: Counter | None = None,
) -> pd.DataFrame:
    """Jusqu'à `n` lignes, à parts égales entre les valeurs de `by`, au hasard dans chacune.

    `used` compte les tirages déjà faits par valeur (d'une tranche de score à l'autre) : les
    valeurs les moins servies passent d'abord. Il est mis à jour.
    """
    used = used if used is not None else Counter()
    if n <= 0 or frame.empty:
        return frame.iloc[:0]
    groups = {
        key: list(g.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).index)
        for key, g in frame.groupby(by)
    }
    picked = []
    while len(picked) < n and any(groups.values()):
        keys = [k for k, rows in groups.items() if rows]
        rng.shuffle(keys)
        key = min(keys, key=lambda k: used[k])  # à égalité, l'ordre mélangé départage
        picked.append(groups[key].pop(0))
        used[key] += 1
    return frame.loc[picked]


def random_candidates(
    con: sqlite3.Connection,
    cfg: dict,
    n: int = 20,
    sites: list[str] | None = None,
    peak_hours: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """`n` fenêtres au hasard, à parts égales entre sites puis micros, aux heures de pic.

    La strate aléatoire mesure les faux négatifs : ce qu'aucun modèle n'a proposé (§5).
    """
    rng = np.random.default_rng(seed)
    recordings = select_recordings(
        con,
        peak_hours=cfg["peak_hours_local"] if peak_hours else None,
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
    )
    if sites:
        wanted = {s.lower() for s in sites}
        recordings = recordings[recordings["site"].str.lower().isin(wanted)]
    if recordings.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    picked = []
    for quota, (_, part) in zip(
        _split_quota(n, recordings["site"].nunique()), recordings.groupby("site"), strict=True
    ):
        picked.append(_round_robin(part, "mic_id", quota, rng))
    chosen = pd.concat(picked)
    w3 = cfg["grids"]["w3"]
    offsets = []
    for r in chosen.itertuples():
        grid = window_grid(r.duration_s or w3["window_s"], w3["window_s"], w3["hop_s"])
        offsets.append(grid[int(rng.integers(len(grid)))][0])
    out = chosen.assign(
        offset_s=offsets, dur_s=w3["window_s"], score=np.nan, reason="random", source="random"
    )
    return _finish(_drop_labelled(con, out), seed)


def recording_candidates(
    con: sqlite3.Connection,
    cfg: dict,
    n: int,
    sites: list[str] | None = None,
    peak_hours: bool = False,
    reason: str = "audit",
    seed: int = 0,
) -> pd.DataFrame:
    """`n` enregistrements à écouter en entier, à parts égales entre micros puis heures locales.

    Sert à l'audit aléatoire (§6 : 300 enregistrements de Mataroni, seule mesure du rappel qui
    ne dépend d'aucun détecteur) et au jeu gelé (§6 : 60 enregistrements stratifiés
    par micro et heure). La fenêtre couvre tout l'enregistrement : son label vaut pour toutes
    les fenêtres de la grille (annotation par enregistrement, §5). Source « audit ».
    Les enregistrements qui portent déjà un label d'enregistrement entier sont écartés.
    """
    rng = np.random.default_rng(seed)
    recordings = select_recordings(
        con,
        peak_hours=cfg["peak_hours_local"] if peak_hours else None,
        utc_offset_h=cfg["recorder"]["filename_utc_offset_h"],
    )
    if sites:
        wanted = {s.lower() for s in sites}
        recordings = recordings[recordings["site"].str.lower().isin(wanted)]
    recordings = recordings.assign(
        offset_s=0.0, dur_s=recordings["duration_s"].astype(float).round(2)
    )
    recordings = _drop_labelled(con, recordings)
    if recordings.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    hours = local_minutes(recordings["start_utc"], cfg["recorder"]["filename_utc_offset_h"]) // 60
    recordings = recordings.assign(
        stratum=recordings["mic_id"].astype(str) + "@" + hours.astype(str)
    )
    # Parts égales entre micros ; dans chaque micro, les heures les moins servies d'abord.
    picked = []
    mics = sorted(recordings["mic_id"].dropna().unique())
    for mic, quota in zip(mics, _split_quota(n, len(mics)), strict=True):
        part = recordings[recordings["mic_id"] == mic]
        picked.append(_round_robin(part, "stratum", quota, rng))
    out = pd.concat(picked).assign(score=np.nan, reason=reason, source="audit")
    return _finish(out, seed)


def _split_quota(n: int, parts: int) -> list[int]:
    """n réparti en `parts` entiers qui diffèrent d'au plus 1."""
    if parts <= 0:
        return []
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def flagged_candidates(
    con: sqlite3.Connection, flag: str, sites: list[str] | None = None, seed: int = 0
) -> pd.DataFrame:
    """Enregistrements écartés par le drapeau `flag` (ex. `clock_off`), à écouter en entier
    pour juger ce qu'ils valent (DECISIONS n° 154). Écartés, ils ne sont jamais encodés : seule
    l'écoute les rend utiles. Source « flag », motif = le drapeau. Sautés : ceux qu'un autre
    drapeau écarte aussi (un test de quelques secondes resterait écarté quoi qu'on entende), et
    ceux qui portent déjà un label d'enregistrement entier."""
    from blanci.inputs.qc import EXCLUDING_FLAGS, flag_raised, is_excluded, parse_flags

    recordings = recordings_table(con)
    others = tuple(k for k in EXCLUDING_FLAGS if k != flag)
    raised = recordings["qc_flags"].map(
        lambda q: flag_raised(parse_flags(q), flag) and not is_excluded(q, others)
    )
    recordings = recordings[raised]
    if sites:
        wanted = {s.lower() for s in sites}
        recordings = recordings[recordings["site"].str.lower().isin(wanted)]
    recordings = recordings.assign(
        offset_s=0.0, dur_s=recordings["duration_s"].astype(float).round(2)
    )
    recordings = _drop_labelled(con, recordings)
    out = recordings.assign(score=np.nan, reason=flag, source="flag")
    return _finish(out, seed)


def annotation_status(
    con: sqlite3.Connection, frame: pd.DataFrame, ignore_sources: tuple[str, ...] = ()
) -> pd.DataFrame:
    """État d'écoute des fenêtres de `frame` (recording_id, offset_s, dur_s), sur son index :
    window_id ; `label`, celui que lui donnent les intervalles (`interval_labels`), sinon son
    dernier label de fenêtre ; `heard`, l'un ou l'autre existe (fenêtre étiquetée, ou couverte
    par un extrait annoté par intervalles, n° 182) ; `positive`, dernier label positif ou
    intervalle positif courant. Comme `_merge_labelled`, un label de fenêtre et un intervalle
    en désaccord (positif / négatif) écartent la fenêtre : elle reste écoutée (jamais
    reproposée), de label `blanci_uncertain` et jamais positive ferme.
    `ignore_sources` : labels de fenêtre de ces sources non comptés
    (« bulk » : ce qui a été entendu, pas ce qui a été propagé)."""
    from blanci.inputs.dataset import (
        EXCLUDED_LABELS,
        current_labels,
        interval_labels,
        load_spans,
    )
    from blanci.inputs.labels import POSITIVE_LABELS

    ids = _window_ids(frame) if len(frame) else []
    labels = current_labels(con)
    labels = labels[~labels["source"].isin(ignore_sources)]
    own = dict(zip(labels["window_id"], labels["label"], strict=True))
    grid = pd.DataFrame(
        {
            "window_id": ids,
            "recording_id": frame["recording_id"].to_numpy(),
            "offset_s": frame["offset_s"].to_numpy(dtype=float),
            "dur_s": frame["dur_s"].to_numpy(dtype=float),
        }
    ).drop_duplicates("window_id")
    derived = interval_labels(*load_spans(con), grid)
    spans = dict(zip(derived["window_id"], derived["label"], strict=True))
    y = dict(zip(derived["window_id"], derived["y"], strict=True))
    label = [spans.get(i, own.get(i)) for i in ids]

    def clash(i: str) -> bool:
        mine, theirs = own.get(i), spans.get(i)
        if mine is None or theirs is None:
            return False
        if mine in EXCLUDED_LABELS or theirs in EXCLUDED_LABELS:
            return False
        return (mine in POSITIVE_LABELS) != (y.get(i) == 1)

    conflict = [clash(i) for i in ids]
    label = ["blanci_uncertain" if c else lab for c, lab in zip(conflict, label, strict=True)]
    return pd.DataFrame(
        {
            "window_id": ids,
            "label": pd.Series(label, dtype=object).to_numpy(),
            "heard": [lab is not None for lab in label],
            "positive": [
                not c and (own.get(i) in POSITIVE_LABELS or y.get(i) == 1)
                for c, i in zip(conflict, ids, strict=True)
            ],
        },
        index=frame.index,
    )


def annotated_recordings(con: sqlite3.Connection) -> tuple[set[str], set[str]]:
    """(enregistrements écoutés, enregistrements où A. blanci a été entendu), mêmes règles que
    `annotation_status` à l'échelle de l'enregistrement : un label de fenêtre ou un extrait ;
    un dernier label positif ou un intervalle positif courant."""
    from blanci.inputs.dataset import current_intervals, current_labels, load_spans
    from blanci.inputs.labels import POSITIVE_LABELS

    labels = current_labels(con)
    spans, intervals = load_spans(con)
    valid = current_intervals(spans, intervals)
    heard = set(labels["recording_id"]) | set(spans["recording_id"])
    positive = set(labels.loc[labels["label"].isin(POSITIVE_LABELS), "recording_id"]) | set(
        valid.loc[valid["label"].isin(POSITIVE_LABELS), "recording_id"]
    )
    return heard, positive


def _drop_labelled(con: sqlite3.Connection, candidates: pd.DataFrame) -> pd.DataFrame:
    """Candidats jamais écoutés (`annotation_status`)."""
    if candidates.empty:
        return candidates
    return candidates[~annotation_status(con, candidates)["heard"].to_numpy()]


def _window_ids(frame: pd.DataFrame) -> list[str]:
    return [
        window_id_for(r, o, d)
        for r, o, d in zip(frame["recording_id"], frame["offset_s"], frame["dur_s"], strict=True)
    ]


def _finish(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Colonnes de la file, ordre mélangé : l'annotateur ne doit pas deviner la strate."""
    out = candidates.reindex(columns=CANDIDATE_COLUMNS)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_candidates(path: Path, con: sqlite3.Connection) -> pd.DataFrame:
    """Relit une file CSV (celles-ci, ou `blanci select`) et la complète depuis la
    base : chemin, site, micro. Une file sans `offset_s` (unité : l'enregistrement) démarre
    à 0 s ; sans `dur_s`, la fenêtre fait 3 s."""
    queue = pd.read_csv(path)
    if "recording_id" not in queue.columns:
        raise ValueError(f"{Path(path).name} : colonne recording_id absente")
    queue = queue.drop(columns=[c for c in ("path", "site", "mic_id", "start_utc") if c in queue])
    recordings = recordings_table(con)[["recording_id", "path", "site", "mic_id", "start_utc"]]
    missing = ~queue["recording_id"].isin(recordings["recording_id"])
    if missing.any():
        raise ValueError(
            f"{Path(path).name} : {int(missing.sum())} enregistrement(s) sur {len(queue)} absents "
            f"de la base ({len(recordings)} enregistrements) — mauvaise base, ou `blanci ingest` "
            "pas encore lancé ?"
        )
    queue = queue.merge(recordings, on="recording_id", how="left")
    for column, default in (("offset_s", 0.0), ("dur_s", 3.0), ("score", np.nan)):
        if column not in queue:
            queue[column] = default
    if "reason" not in queue:
        queue["reason"] = ""
    if "source" not in queue:
        queue["source"] = "active"
    return queue


def latest_labels(
    con: sqlite3.Connection, window_ids: list[str], annotator: str | None = None
) -> dict[str, str]:
    """Dernier label de chacune de ces fenêtres déjà écoutées (les autres sont absentes).

    Avec `annotator` (calibration entre annotateurs, §5), seules ses propres réponses
    comptent : chacun écoute la même file sans voir ce que l'autre a répondu.
    """
    if annotator is None:
        rows = con.execute(
            "SELECT window_id, label FROM labels WHERE label_id IN "
            "(SELECT MAX(label_id) FROM labels GROUP BY window_id)"
        )
    else:
        rows = con.execute(
            "SELECT window_id, label FROM labels WHERE label_id IN "
            "(SELECT MAX(label_id) FROM labels WHERE annotator = ? GROUP BY window_id)",
            (annotator,),
        )
    wanted = set(window_ids)
    return {wid: label for wid, label in rows if wid in wanted}


def progress(
    con: sqlite3.Connection, queue: pd.DataFrame, annotator: str | None = None
) -> pd.Series:
    """Pour chaque candidat : son label s'il a déjà été écouté, sinon None (`annotator` : voir
    `latest_labels`). Un candidat entièrement dans un extrait annoté par intervalles prend le
    label qui s'en déduit (A. blanci s'il touche un intervalle), avant son label de fenêtre."""
    from blanci.inputs.dataset import interval_labels, load_spans

    ids = _window_ids(queue)
    latest = latest_labels(con, ids, annotator)
    grid = pd.DataFrame(
        {
            "window_id": ids,
            "recording_id": queue["recording_id"].to_numpy(),
            "offset_s": queue["offset_s"].to_numpy(dtype=float),
            "dur_s": queue["dur_s"].to_numpy(dtype=float),
        }
    ).drop_duplicates("window_id")
    derived = interval_labels(*load_spans(con, annotator), grid)
    latest |= dict(zip(derived["window_id"], derived["label"], strict=True))
    return pd.Series([latest.get(i) for i in ids], index=queue.index, dtype=object)


def next_position(
    done: pd.Series, pos: int, unheard_only: bool, step: int = 1, wrap: bool = True
) -> int:
    """Candidat qui suit `pos` dans la file (`step` -1 : qui le précède), en position dans la
    file entière, jamais dans les seuls candidats restants : « 2 / 1521 » suit « 1 / 1521 ».
    `unheard_only` : le prochain jamais écouté, en faisant le tour de la file si `wrap`.
    Renvoie `pos` s'il n'y en a pas (bout de la file, ou tout est écouté)."""
    n = len(done)
    if not unheard_only:
        return min(max(pos + step, 0), n - 1)
    heard = done.notna().to_numpy()
    ahead = range(pos + 1, n) if step > 0 else range(pos - 1, -1, -1)
    behind = (range(pos + 1) if step > 0 else range(n - 1, pos - 1, -1)) if wrap else ()
    for i in [*ahead, *behind]:
        if not heard[i]:
            return i
    return pos


def split_windows(start_s: float, stop_s: float, anchor_s: float, length_s: float) -> list[float]:
    """Débuts des fenêtres de `length_s` qui découpent [start_s, stop_s], calées sur `anchor_s`
    (le début du candidat, qui reste donc une des fenêtres) ; les bouts plus courts sont
    laissés de côté."""
    if length_s <= 0:
        raise ValueError("longueur de fenêtre nulle")
    k = int(np.ceil((start_s - anchor_s) / length_s - 1e-6))
    offsets = []
    while anchor_s + k * length_s + length_s <= stop_s + 1e-6:
        offsets.append(round(anchor_s + k * length_s, 2))
        k += 1
    return offsets


def ordered_classes(classes: list[str]) -> tuple[str, list[str]]:
    """(label, autres classes) d'une réponse à plusieurs classes : le label rangé dans
    `labels.label` est la première dans l'ordre de `ANSWERS` (A. blanci d'abord), les autres
    vont dans `conditions.extra_labels`. Rien de coché : « rien » ; « rien » coché avec une
    autre classe est ignoré."""
    order = [label for label, _ in ANSWERS]
    chosen = sorted(dict.fromkeys(classes), key=lambda c: order.index(c) if c in order else 99)
    if len(chosen) > 1:
        chosen = [c for c in chosen if c != "background"]
    if not chosen:
        return "background", []
    return chosen[0], chosen[1:]


# --- Écoute ---------------------------------------------------------------------------------------


def read_clip(
    raw_root: Path,
    path: str,
    offset_s: float,
    dur_s: float,
    context_s: float = 2.0,
    channel: int = 0,
) -> tuple[np.ndarray, int, float]:
    """(forme d'onde, f_e, début de l'extrait en s) : la fenêtre avec `context_s` de chaque côté.

    Lecture partielle, jamais d'écriture. `channel` : 0 = micro 1 (gain 6 dB), 1 = micro 2
    (gain 18 dB, plus fort, utile pour un chant lointain, DECISIONS n° 58).
    """
    with sf.SoundFile(Path(raw_root) / path) as f:
        start = max(0.0, offset_s - context_s)
        stop = min(f.frames / f.samplerate, offset_s + dur_s + context_s)
        f.seek(round(start * f.samplerate))
        wav = f.read(round((stop - start) * f.samplerate), dtype="float32", always_2d=True)
        sr = f.samplerate
    return wav[:, min(channel, wav.shape[1] - 1)], sr, start


def wav_bytes(
    wav: np.ndarray,
    sr: int,
    gain_db: float = 0.0,
    band_hz: tuple[float, float] | None = None,
) -> bytes:
    """WAV 16 bits en mémoire pour le lecteur ; `gain_db` et `band_hz` (passe-bande : on
    n'entend que cette bande) n'agissent que sur l'écoute."""
    if band_hz is not None:
        from blanci.heads.signal_processing import bandpass

        wav = bandpass(wav, sr, band_hz)
    x = np.clip(wav * 10 ** (gain_db / 20), -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, x, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# --- Réponses -------------------------------------------------------------------------------------


def ensure_window(con: sqlite3.Connection, recording_id: str, offset_s: float, dur_s: float) -> str:
    """Identifiant de la fenêtre, créée si besoin (même règle que l'import)."""
    wid = window_id_for(recording_id, offset_s, dur_s)
    con.execute(
        "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
        "VALUES (?, ?, ?, ?)",
        (wid, recording_id, round(float(offset_s), 2), float(dur_s)),
    )
    return wid


def save_answer(
    con: sqlite3.Connection,
    candidate: dict[str, Any],
    label: str,
    annotator: str,
    quality: str | None = None,
    comment: str | None = None,
    channel: int | None = None,
    species: str | None = None,
    extra_labels: list[str] | None = None,
) -> int:
    """Enregistre la réponse de l'annotateur (label en ajout seul) ; renvoie label_id.

    `species` : espèce entendue (faux ami, congénère), rangée dans `labels.species`.
    `extra_labels` : autres classes entendues dans la même fenêtre (`ordered_classes`),
    rangées dans `conditions.extra_labels`."""
    from blanci.inputs.labels import LABELS

    unknown = [c for c in extra_labels or () if c not in LABELS]
    if unknown:
        raise ValueError(f"classe inconnue : {', '.join(unknown)}")
    wid = ensure_window(con, candidate["recording_id"], candidate["offset_s"], candidate["dur_s"])
    conditions: dict[str, Any] = {"candidate_reason": candidate.get("reason") or None}
    if comment:
        # Le commentaire est gardé tel quel, et lu comme à l'import : « pluie » écrit ici
        # pose le drapeau pluie de l'enregistrement, une espèce citée est retrouvée.
        conditions |= comment_fields(comment) | {"comment": comment}
    if extra_labels:
        conditions["extra_labels"] = list(extra_labels)
    if channel is not None:
        conditions["channel_listened"] = channel
    score = candidate.get("score")
    if score is not None and not pd.isna(score):
        conditions["previous_model_score"] = float(score)
    conditions = {k: v for k, v in conditions.items() if v is not None}
    return append_label(
        con,
        wid,
        label,
        source=candidate.get("source") or "active",
        quality=quality,
        species=species or None,
        conditions=conditions,
        annotator=annotator,
    )


INTERVAL_ANSWERS = tuple((label, name) for label, name in ANSWERS if label in INTERVAL_LABELS)


def save_span(
    con: sqlite3.Connection,
    candidate: dict[str, Any],
    start_s: float,
    end_s: float,
    intervals: list[tuple],
    classes: list[str],
    annotator: str,
    quality: str | None = None,
    comment: str | None = None,
    channel: int | None = None,
    species: str | None = None,
    multiclass: bool = True,
) -> int:
    """Enregistre un extrait écouté et ses intervalles d'A. blanci ; renvoie span_id.

    `classes` : classes entendues dans l'extrait (cases cochées) ; celles des intervalles
    (A. blanci, faux ami) s'ajoutent d'elles-mêmes. A. blanci coché sans aucun intervalle
    est refusé : toutes les fenêtres de l'extrait deviendraient négatives. Les fenêtres hors
    intervalles prennent la première des autres classes (`ordered_classes`), « rien » sinon.
    `multiclass` faux : seule A. blanci a été notée, les autres classes n'ont pas été
    cherchées (`conditions.multiclass` = false) ; « rien » veut alors dire « pas d'A. blanci ».
    """
    blanci = {"blanci", "blanci_chorus", "blanci_uncertain"}
    if blanci & set(classes) and not intervals:
        raise ValueError("A. blanci coché sans intervalle : tracer où il chante")
    heard = list(dict.fromkeys([*classes, *(i[2] for i in intervals)]))
    # Les faux amis ont leurs intervalles : ils ne donnent pas leur label au reste de l'extrait.
    other, _ = ordered_classes([c for c in classes if c not in INTERVAL_LABELS])
    conditions: dict[str, Any] = {"candidate_reason": candidate.get("reason") or None}
    if comment:
        conditions |= comment_fields(comment) | {"comment": comment}
    if channel is not None:
        conditions["channel_listened"] = channel
    if not multiclass:
        conditions["multiclass"] = False
    conditions = {k: v for k, v in conditions.items() if v is not None}
    return append_span(
        con,
        candidate["recording_id"],
        start_s,
        end_s,
        intervals,
        other,
        source=candidate.get("source") or "active",
        classes=heard or ["background"],
        quality=quality,
        species=species or None,
        conditions=conditions,
        annotator=annotator,
    )


def span_intervals(
    con: sqlite3.Connection,
    recording_id: str,
    start_s: float,
    end_s: float,
    annotator: str | None = None,
) -> list[tuple[float, float, str, str | None]] | None:
    """Intervalles (début, fin, label, qualité) du dernier extrait [start_s, end_s] déjà annoté
    (à réafficher quand on y revient), None s'il ne l'a jamais été."""
    sql = (
        "SELECT span_id FROM spans WHERE recording_id = ? AND ABS(start_s - ?) < 0.006 "
        "AND ABS(end_s - ?) < 0.006"
    )
    params: tuple = (recording_id, start_s, end_s)
    if annotator is not None:
        sql, params = sql + " AND annotator = ?", (*params, annotator)
    row = con.execute(sql + " ORDER BY span_id DESC LIMIT 1", params).fetchone()
    if row is None:
        return None
    return [
        (float(a), float(b), str(c), q)
        for a, b, c, q in con.execute(
            "SELECT start_s, end_s, label, quality FROM intervals WHERE span_id = ? "
            "ORDER BY start_s",
            (row[0],),
        )
    ]


def local_time(start_utc: str | None, utc_offset_h: float) -> str:
    """« 2026-01-10 07:00 » en heure locale, pour l'affichage."""
    if not start_utc:
        return "?"
    minutes = local_minutes(pd.Series([start_utc]), utc_offset_h).iloc[0]
    day = (pd.to_datetime(start_utc, utc=True) + pd.Timedelta(hours=utc_offset_h)).date()
    return f"{day} {int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


# --- Accord entre annotateurs ---------------------------------------------------------------------


def agreement(
    con: sqlite3.Connection, first: str, second: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Accord de deux annotateurs sur les fenêtres qu'ils ont tous deux écoutées (§5).

    Dernière réponse de chacun. Rapporte l'accord brut sur le label, l'accord sur la question
    « A. blanci ou non » (un « A. blanci ? » compte comme non) et l'accord sur les positifs,
    2a / (2a + b + c), qui ne se laisse pas gonfler par les nombreux négatifs faciles. Renvoie
    aussi le tableau croisé des labels.
    """
    from blanci.inputs.labels import POSITIVE_LABELS

    def latest(name: str) -> pd.Series:
        rows = con.execute(
            "SELECT window_id, label FROM labels WHERE label_id IN "
            "(SELECT MAX(label_id) FROM labels WHERE annotator = ? GROUP BY window_id)",
            (name,),
        ).fetchall()
        return pd.Series(dict(rows), dtype=object)

    a, b = latest(first), latest(second)
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
    pos_a, pos_b = a.isin(POSITIVE_LABELS), b.isin(POSITIVE_LABELS)
    both, only = int((pos_a & pos_b).sum()), int((pos_a ^ pos_b).sum())
    n = len(shared)
    summary = {
        "first": first,
        "second": second,
        "n_windows": n,
        "label_agreement": float((a == b).mean()) if n else float("nan"),
        "blanci_agreement": float((pos_a == pos_b).mean()) if n else float("nan"),
        "positive_agreement": 2 * both / (2 * both + only) if (both + only) else float("nan"),
        "n_blanci_first": int(pos_a.sum()),
        "n_blanci_second": int(pos_b.sum()),
    }
    table = pd.crosstab(a.rename(first), b.rename(second), dropna=False) if n else pd.DataFrame()
    return summary, table
