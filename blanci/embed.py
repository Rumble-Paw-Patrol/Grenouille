"""Extraction des embeddings : enregistrements → grille → encodeur → stock Parquet (§4).

Reprenable : un enregistrement déjà présent dans sa partition est sauté ; le stock est écrit
tous les `flush_every` enregistrements. L'audio lu sert aussi au contrôle qualité (drapeaux
audio) des enregistrements qui ne l'ont pas encore eu : un enregistrement silencieux ou
micro dans sac est alors écarté avant d'être encodé. Le débit mesuré (fenêtres/s, facteur
temps réel) est enregistré dans la table models : c'est la colonne « vitesse » du benchmark
(§2).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from blanci.audio import cut_windows, load_audio
from blanci.db import utc_now, window_id_for
from blanci.encoders.base import Encoder, encoder_id
from blanci.grid import window_grid
from blanci.qc import (
    EXCLUDING_FLAGS,
    is_excluded,
    merge_audio_flags,
    parse_flags,
    positive_recordings,
    qc_flags,
    qc_indices,
)
from blanci.sequential import recording_onsets, store_onsets
from blanci.store import EmbeddingStore


@dataclass
class EmbedReport:
    encoder_id: str
    recordings: int = 0
    skipped: int = 0
    windows: int = 0
    audio_s: float = 0.0
    encode_s: float = 0.0
    errors: int = 0
    qc_checked: int = 0  # contrôle audio fait pendant ce passage
    qc_excluded: int = 0  # écartés par ce contrôle (silencieux, micro dans sac)

    @property
    def windows_per_s(self) -> float:
        return self.windows / self.encode_s if self.encode_s else float("nan")

    @property
    def realtime_factor(self) -> float:
        return self.audio_s / self.encode_s if self.encode_s else float("nan")


def select_recordings(
    con: sqlite3.Connection,
    dataset: str | None = None,
    site: str | None = None,
    peak_hours: list[list[int]] | None = None,
    utc_offset_h: float = -3,
    exclude_flags: tuple[str, ...] = EXCLUDING_FLAGS,
) -> pd.DataFrame:
    """Enregistrements à encoder, filtrés par jeu, site, heures de pic locales et drapeaux QC.

    Un enregistrement signalé (`EXCLUDING_FLAGS`) n'est jamais encodé : il ne peut donc
    devenir ni négatif apparié, ni candidat de file de vérification, ni score. Le fichier,
    lui, n'est pas touché. Exception : un enregistrement où A. blanci a été entendu est
    toujours gardé.
    """
    df = pd.read_sql_query("SELECT * FROM recordings ORDER BY path", con)
    if dataset:
        df = df[df["dataset"] == dataset]
    if site:
        df = df[df["site"].str.lower() == site.lower()]
    if peak_hours:
        hours = (
            pd.to_datetime(df["start_utc"], utc=True) + pd.Timedelta(hours=utc_offset_h)
        ).dt.hour
        df = df[np.logical_or.reduce([(hours >= a) & (hours < b) for a, b in peak_hours])]
    bad = df["qc_flags"].map(lambda q: is_excluded(q, exclude_flags))
    bad &= ~df["recording_id"].isin(positive_recordings(con))
    return df[~bad].reset_index(drop=True)


def month_of(start_utc: str | None) -> str:
    return start_utc[:7].replace("-", "") if start_utc else "unknown"


def _flush(
    store: EmbeddingStore,
    con: sqlite3.Connection,
    metas: list[pd.DataFrame],
    embs: list[np.ndarray],
    dataset: str,
    site: str,
    month: str,
) -> None:
    """Écrit le tampon dans sa partition puis le vide. Sans effet si le tampon est vide."""
    if not metas:
        return
    store.write(pd.concat(metas, ignore_index=True), np.concatenate(embs), dataset, site, month)
    con.commit()
    metas.clear()
    embs.clear()


def _store_logits(
    con: sqlite3.Connection, encoder: Encoder, encoder_id: str, window_ids: list[str]
) -> None:
    """Logits de classes gardés par l'encodeur pendant `embed` (perch_v2 : congénères, §2),
    rangés dans `scores` sous `<encodeur>:logit:<classe>`. Sans effet pour les autres."""
    pop = getattr(encoder, "pop_logits", None)
    logits = pop() if pop else None
    if logits is None or len(logits) != len(window_ids):
        return
    rows = [
        (wid, f"{encoder_id}:logit:{name}", float(value))
        for j, name in enumerate(encoder.logit_names)
        for wid, value in zip(window_ids, logits[:, j], strict=True)
    ]
    con.executemany(
        "INSERT OR REPLACE INTO scores (window_id, model_id, score) VALUES (?, ?, ?)", rows
    )


def embed_recordings(
    con: sqlite3.Connection,
    encoder: Encoder,
    recordings: pd.DataFrame,
    raw_root: Path,
    store_root: Path,
    hop_ratio: float = 0.5,
    flush_every: int = 50,
    progress_every: int = 100,
    channel: int | str = "mean",
    signal_cfg: dict | None = None,
    qc_thresholds: dict | None = None,
) -> EmbedReport:
    """Encode les enregistrements. L'audio est déjà en mémoire, une seule lecture sert aussi :
    - avec `signal_cfg`, aux débuts de notes de chaque enregistrement qui n'en a pas encore
      (module séquentiel, §3) ;
    - avec `qc_thresholds`, au contrôle audio de chaque enregistrement qui ne l'a pas encore
      eu ; s'il lève un drapeau d'exclusion (silencieux, micro dans sac), l'enregistrement
      n'est pas encodé, sauf si A. blanci y a été entendu."""
    eid = encoder_id(encoder)
    protected = positive_recordings(con) if qc_thresholds is not None else set()
    with_onsets = {row[0] for row in con.execute("SELECT recording_id FROM onsets")}
    store = EmbeddingStore(store_root, eid)
    window_s = round(encoder.window_s, 2)
    hop_s = round(window_s * hop_ratio, 2)
    report = EmbedReport(eid)
    recordings = recordings.assign(month=recordings["start_utc"].map(month_of))

    for (dataset, site, month), group in recordings.groupby(["dataset", "site", "month"]):
        path = store.partition_path(dataset, site, month)
        done = set(store.read_meta(path)["recording_id"]) if path.exists() else set()
        metas: list[pd.DataFrame] = []
        embs: list[np.ndarray] = []

        for rec in group.itertuples():
            if rec.recording_id in done:
                report.skipped += 1
                continue
            try:
                wav, sr = load_audio(Path(raw_root) / rec.path, channel)
            except Exception:  # fichier illisible : déjà signalé à l'inventaire
                report.errors += 1
                continue
            if qc_thresholds is not None and "indices" not in parse_flags(rec.qc_flags):
                audio = qc_flags(qc_indices(wav, sr), qc_thresholds)
                merged = merge_audio_flags(con, rec.recording_id, audio)
                report.qc_checked += 1
                if is_excluded(merged) and rec.recording_id not in protected:
                    report.qc_excluded += 1
                    continue
            if signal_cfg is not None and rec.recording_id not in with_onsets:
                store_onsets(con, rec.recording_id, recording_onsets(wav, sr, signal_cfg), channel)
                with_onsets.add(rec.recording_id)
            windows = window_grid(len(wav) / sr, window_s, hop_s)
            start = perf_counter()
            emb = encoder.embed(cut_windows(wav, sr, windows), sr)
            report.encode_s += perf_counter() - start
            ids = [window_id_for(rec.recording_id, offset, dur) for offset, dur in windows]
            con.executemany(
                "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
                "VALUES (?, ?, ?, ?)",
                [
                    (wid, rec.recording_id, offset, dur)
                    for wid, (offset, dur) in zip(ids, windows, strict=True)
                ],
            )
            _store_logits(con, encoder, report.encoder_id, ids)
            metas.append(
                pd.DataFrame(
                    {
                        "window_id": ids,
                        "recording_id": rec.recording_id,
                        "offset_s": [o for o, _ in windows],
                    }
                )
            )
            embs.append(emb)
            report.recordings += 1
            report.windows += len(windows)
            report.audio_s += len(wav) / sr
            if report.recordings % flush_every == 0:
                _flush(store, con, metas, embs, dataset, site, month)
            if report.recordings % progress_every == 0:
                print(
                    f"  {report.recordings} enregistrements, {report.windows_per_s:.1f} fenêtres/s",
                    flush=True,
                )
        _flush(store, con, metas, embs, dataset, site, month)

    register_encoder(con, encoder, hop_s, report, channel)
    return report


def register_encoder(
    con: sqlite3.Connection,
    encoder: Encoder,
    hop_s: float,
    report: EmbedReport,
    channel: int | str = "mean",
) -> None:
    params: dict[str, Any] = {
        "sample_rate": encoder.sample_rate,
        "window_s": encoder.window_s,
        "hop_s": hop_s,
        "channel": channel,  # micro lu : des embeddings de micros différents ne se comparent pas
        "dim": encoder.dim,
        "has_tokens": encoder.has_tokens,
        "last_run": asdict(report)
        | {"windows_per_s": report.windows_per_s, "realtime_factor": report.realtime_factor},
    }
    con.execute(
        "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
        "VALUES (?, 'encoder', ?, ?, ?, ?) "
        "ON CONFLICT(model_id) DO UPDATE SET params_json = excluded.params_json",
        (report.encoder_id, encoder.name, encoder.version, json.dumps(params), utc_now()),
    )
    con.commit()
