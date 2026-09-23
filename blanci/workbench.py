"""Poste d'annotation (§5, M2) : candidats à écouter, extraits, spectrogrammes, labels.

Logique pure, sans interface : l'application Streamlit (`blanci/app.py`) n'en est qu'un
affichage, et la future GUI du livrable appellera les mêmes fonctions (§4).

Candidats = une file CSV (recording_id, offset_s, dur_s, reason, source, score…) :
- `blancinet_candidates` : détections jamais écoutées de l'export Blancinet, réparties entre
  sites, micros et tranches de score (hors Mataroni surtout : tous les positifs actuels en
  viennent, l'évaluation « nouveaux sites » du §6 en manque) ;
- `random_candidates` : fenêtres tirées au hasard (site, micro, heure), qui mesurent ce que
  Blancinet n'a jamais remonté ;
- plus tard, les files `blanci queue` et `blanci search`, dès qu'un encodeur sera choisi.

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
from scipy.signal import spectrogram

from blanci.dataset import local_minutes, recordings_table
from blanci.db import window_id_for
from blanci.embed import select_recordings
from blanci.grid import window_grid
from blanci.labels import (
    _is_blank,
    detect_columns,
    file_key,
    parse_offset,
    read_annotation_table,
)
from blanci.service import append_label

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

# Tranches de score de Blancinet : les scores bas sont les plus nombreux et les plus
# incertains, les hauts disent si le modèle se trompe quand il est sûr de lui.
SCORE_BINS = (0.0, 0.3, 0.7, 1.01)

# Réponses proposées à l'écoute : labels du schéma (§5), dans l'ordre des boutons.
ANSWERS = (
    ("blanci", "A. blanci"),
    ("blanci_chorus", "A. blanci, plusieurs"),
    ("blanci_uncertain", "A. blanci ?"),
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


def blancinet_candidates(
    con: sqlite3.Connection,
    table: Path,
    cfg: dict,
    per_site: int = 30,
    sites: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Détections Blancinet jamais vérifiées, `per_site` par site.

    Par site : parts égales entre tranches de score (`SCORE_BINS`), puis entre micros ; une
    seule fenêtre par enregistrement, pour entendre le plus de situations possibles. Les
    fenêtres déjà étiquetées et les enregistrements signalés sont écartés.
    """
    rng = np.random.default_rng(seed)
    icfg = cfg["labels"]["import"]
    df = read_annotation_table(Path(table))
    columns = detect_columns(df, icfg["columns"])
    for needed in ("file", "offset_s", "verdict"):
        if needed not in columns:
            raise ValueError(f"colonne {needed!r} introuvable dans {Path(table).name}")
    df = df[df[columns["verdict"]].map(_is_blank)]

    recordings = select_recordings(con)  # sans les enregistrements signalés
    by_key = {file_key(p): i for i, p in recordings["path"].items()}
    rows = []
    for record in df.to_dict("records"):
        i = by_key.get(file_key(str(record[columns["file"]])))
        if i is None:
            continue
        try:
            offset = parse_offset(
                record[columns["offset_s"]], icfg["offset_unit"], float(icfg["window_s"])
            )
        except ValueError:
            continue
        score = record.get(columns.get("score", ""), np.nan)
        rows.append((i, round(offset, 2), float(score) if not pd.isna(score) else np.nan))
    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    found = pd.DataFrame(rows, columns=["row", "offset_s", "score"])
    found = found.join(recordings, on="row").drop(columns="row")
    found = _drop_labelled(con, found)
    if sites:
        wanted = {s.lower() for s in sites}
        found = found[found["site"].str.lower().isin(wanted)]
    found["bin"] = pd.cut(found["score"].fillna(0.0), SCORE_BINS, right=False, labels=False)

    picked = []
    for _, part in found.groupby("site"):
        part = part.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        part = part.drop_duplicates("recording_id")
        bins = sorted(part["bin"].dropna().unique())
        used: Counter = Counter()  # micros servis, d'une tranche à l'autre
        for b, q in zip(bins, _split_quota(per_site, len(bins)), strict=True):
            picked.append(_round_robin(part[part["bin"] == b], "mic_id", q, rng, used))
    out = pd.concat(picked) if picked else found.iloc[:0]
    out = out.assign(
        dur_s=float(icfg["window_s"]),
        reason="blancinet_" + out["bin"].map(lambda b: _bin_name(int(b))).astype(str),
        source="active",
    )
    return _finish(out, seed)


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


def _split_quota(n: int, parts: int) -> list[int]:
    """n réparti en `parts` entiers qui diffèrent d'au plus 1."""
    if parts <= 0:
        return []
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def _bin_name(b: int) -> str:
    lo, hi = SCORE_BINS[b], min(SCORE_BINS[b + 1], 1.0)
    return f"{lo:.1f}-{hi:.1f}"


def _drop_labelled(con: sqlite3.Connection, candidates: pd.DataFrame) -> pd.DataFrame:
    done = {row[0] for row in con.execute("SELECT DISTINCT window_id FROM labels")}
    ids = [
        window_id_for(r, o)
        for r, o in zip(candidates["recording_id"], candidates["offset_s"], strict=True)
    ]
    return candidates[[i not in done for i in ids]]


def _finish(candidates: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Colonnes de la file, ordre mélangé : l'annotateur ne doit pas deviner la strate."""
    out = candidates.reindex(columns=CANDIDATE_COLUMNS)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_candidates(path: Path, con: sqlite3.Connection) -> pd.DataFrame:
    """Relit une file CSV (celles-ci, ou `blanci queue` / `search`) et la complète depuis la
    base : chemin, site, micro. Une file sans `offset_s` (unité : l'enregistrement) démarre
    à 0 s ; sans `dur_s`, la fenêtre fait 3 s."""
    queue = pd.read_csv(path)
    if "recording_id" not in queue.columns:
        raise ValueError(f"{Path(path).name} : colonne recording_id absente")
    queue = queue.drop(columns=[c for c in ("path", "site", "mic_id", "start_utc") if c in queue])
    recordings = recordings_table(con)[["recording_id", "path", "site", "mic_id", "start_utc"]]
    queue = queue.merge(recordings, on="recording_id", how="left")
    for column, default in (("offset_s", 0.0), ("dur_s", 3.0), ("score", np.nan)):
        if column not in queue:
            queue[column] = default
    if "reason" not in queue:
        queue["reason"] = ""
    if "source" not in queue:
        queue["source"] = "active"
    return queue


def progress(con: sqlite3.Connection, queue: pd.DataFrame) -> pd.Series:
    """Pour chaque candidat : son dernier label s'il a déjà été écouté, sinon None."""
    latest = dict(
        con.execute(
            "SELECT window_id, label FROM labels WHERE label_id IN "
            "(SELECT MAX(label_id) FROM labels GROUP BY window_id)"
        ).fetchall()
    )
    ids = [
        window_id_for(r, o) for r, o in zip(queue["recording_id"], queue["offset_s"], strict=True)
    ]
    return pd.Series([latest.get(i) for i in ids], index=queue.index, dtype=object)


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


def clip_spectrogram(
    wav: np.ndarray, sr: int, fmax_hz: float = 10_000.0, nperseg: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fréquences Hz, temps s, puissance dB) jusqu'à `fmax_hz`, pour l'affichage."""
    nperseg = min(nperseg, len(wav)) or 1
    freqs, times, power = spectrogram(
        wav.astype(np.float64), fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4
    )
    keep = freqs <= fmax_hz
    return freqs[keep], times, 10 * np.log10(power[keep] + 1e-12)


def wav_bytes(wav: np.ndarray, sr: int, gain_db: float = 0.0) -> bytes:
    """WAV 16 bits en mémoire pour le lecteur ; `gain_db` n'agit que sur l'écoute."""
    x = np.clip(wav * 10 ** (gain_db / 20), -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, x, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# --- Réponses -------------------------------------------------------------------------------------


def ensure_window(con: sqlite3.Connection, recording_id: str, offset_s: float, dur_s: float) -> str:
    """Identifiant de la fenêtre, créée si besoin (même règle que l'import)."""
    wid = window_id_for(recording_id, offset_s)
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
) -> int:
    """Enregistre la réponse de l'annotateur (label en ajout seul) ; renvoie label_id."""
    wid = ensure_window(con, candidate["recording_id"], candidate["offset_s"], candidate["dur_s"])
    conditions: dict[str, Any] = {"candidate_reason": candidate.get("reason") or None}
    if comment:
        conditions["comment"] = comment
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
        conditions=conditions,
        annotator=annotator,
    )


def local_time(start_utc: str | None, utc_offset_h: float) -> str:
    """« 2026-01-10 07:00 » en heure locale, pour l'affichage."""
    if not start_utc:
        return "?"
    minutes = local_minutes(pd.Series([start_utc]), utc_offset_h).iloc[0]
    day = (pd.to_datetime(start_utc, utc=True) + pd.Timedelta(hours=utc_offset_h)).date()
    return f"{day} {int(minutes) // 60:02d}:{int(minutes) % 60:02d}"
