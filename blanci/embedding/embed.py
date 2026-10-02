"""Extraction des embeddings : enregistrements → grille → encodeur → stock Parquet (§4).

Reprenable : un enregistrement déjà présent dans sa partition est sauté ; le stock est écrit
tous les `flush_every` enregistrements. L'audio lu sert aussi au contrôle qualité (drapeaux
audio) des enregistrements qui ne l'ont pas encore eu : un enregistrement silencieux ou
micro dans sac est alors écarté avant d'être encodé. Le débit mesuré (fenêtres/s, facteur
temps réel) est enregistré dans la table models : c'est la colonne « vitesse » du benchmark
(§2).

La lecture, le contrôle audio et les débuts de notes de l'enregistrement suivant se font dans
un fil à part pendant que l'encodeur traite le courant (`core.ahead`) : faits à leur tour, ils
prenaient un quart du temps d'un passage, encodeur à l'arrêt. `blanci qc` fait ce contrôle et
ces débuts de notes d'avance, pour tout le corpus (`inputs.qc.check_recordings`) : il ne reste
alors ici que la lecture et l'encodeur.
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

from blanci.core.ahead import ahead as read_ahead
from blanci.core.audio import cut_windows, load_audio, resample
from blanci.core.db import utc_now, window_id_for
from blanci.embedding.encoders.base import Encoder, stock_id
from blanci.embedding.grid import hop_for_overlap, overlap_of, window_grid
from blanci.embedding.store import GATED, EmbeddingStore
from blanci.heads.signal_processing import (
    Upstream,
    gate_values,
    load_onsets,
    recording_onsets,
    store_onsets,
)
from blanci.inputs.qc import (
    EXCLUDING_FLAGS,
    is_excluded,
    merge_audio_flags,
    parse_flags,
    positive_recordings,
    qc_flags,
    qc_indices,
)


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
    gated: int = 0  # fenêtres arrêtées par les portes du seuillage en amont (non encodées)

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


@dataclass
class _Audio:
    """Ce qu'un enregistrement donne avant l'encodeur, calculé dans le fil de lecture."""

    wav: np.ndarray | None = None
    sr: int = 0
    failed: bool = False  # fichier illisible
    flags: dict[str, Any] | None = None  # drapeaux audio, si le contrôle restait à faire
    onsets: np.ndarray | None = None  # débuts de notes, s'ils restaient à calculer
    windows: list | None = None  # grille de l'enregistrement
    cut: np.ndarray | None = None  # ses fenêtres, prêtes pour l'encodeur...
    rate: int = 0  # ...et leur fréquence d'échantillonnage


RESAMPLE_MODES = ("recording", "window")


def resample_mode(encoder: Encoder, mode: str = "recording") -> str:
    """Où se fait le rééchantillonnage vers la fréquence de l'encodeur (`encoders.resample`) :
    « recording », sur l'enregistrement entier avant la découpe (chaque seconde une seule fois,
    et le filtre voit le vrai son de part et d'autre de chaque fenêtre) ; « window », fenêtre
    par fenêtre dans l'encodeur (stocks d'avant le n° 176). Un encodeur enveloppé (passe-bas,
    transformations en amont) transforme le son à sa fréquence d'origine : il reste à « window »."""
    if mode not in RESAMPLE_MODES:
        raise ValueError(f"encoders.resample : {mode!r} inconnu (attendu : {RESAMPLE_MODES})")
    return "window" if hasattr(encoder, "inner") else mode


def encoder_windows(
    encoder: Encoder, wav: np.ndarray, sr: int, windows: list, mode: str
) -> tuple[np.ndarray, int]:
    """Fenêtres à donner à `encoder.embed`, et leur fréquence d'échantillonnage."""
    target = int(encoder.sample_rate)
    if mode == "recording" and sr != target:
        return cut_windows(resample(wav, sr, target), target, windows), target
    return cut_windows(wav, sr, windows), sr


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
    store.append(pd.concat(metas, ignore_index=True), np.concatenate(embs), dataset, site, month)
    con.commit()
    metas.clear()
    embs.clear()


def _store_logits(
    con: sqlite3.Connection, encoder: Encoder, encoder_id: str, window_ids: list[str]
) -> None:
    """Logits de classes gardés par l'encodeur pendant `embed` (`logit_classes` : birdnet_v3
    sur AnuraSet), rangés dans `scores` sous `<encodeur>:logit:<classe>`. Sans effet pour les
    autres."""
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
    overlap: float = 0.5,
    flush_every: int = 50,
    progress_every: int = 100,
    channel: int | str = "mean",
    signal_cfg: dict | None = None,
    qc_thresholds: dict | None = None,
    gates: Upstream | None = None,
    ahead: int = 2,
    resample: str = "recording",
) -> EmbedReport:
    """Encode les enregistrements. L'audio est déjà en mémoire, une seule lecture sert aussi :
    - avec `signal_cfg`, aux débuts de notes de chaque enregistrement qui n'en a pas encore
      (module de traitement du signal, §3) ;
    - avec `qc_thresholds`, au contrôle audio de chaque enregistrement qui ne l'a pas encore
      eu ; s'il lève un drapeau d'exclusion (silencieux, micro dans sac), l'enregistrement
      n'est pas encodé, sauf si A. blanci y a été entendu.

    `overlap` : chevauchement des fenêtres (0 à 0,99) ; hors 50 %, le stock porte le
    chevauchement dans son nom (`stock_id`).

    `gates` (traitement du signal en amont, portes actives seulement) : une fenêtre arrêtée n'est
    pas encodée ; elle est rangée avec un embedding nul et `gated` vrai, pour que l'agrégation
    la compte (score le plus bas) et que l'entraînement l'ignore. Stock `<id>+g-…`.

    Un stock déjà encodé avec un autre canal, un autre checkpoint ou d'autres transformations
    en amont est refusé (`check_stock_identity`) : la reprise y ajouterait des embeddings qui
    ne se comparent pas aux siens (DECISIONS n° 143).

    `ahead` : enregistrements lus et contrôlés d'avance par le fil de lecture (0 : aucun).
    `resample` : voir `resample_mode` ; rangé avec le stock, qui refuse d'en changer."""
    eid = stock_id(encoder, overlap)
    if gates is not None and gates.gates:
        eid = f"{eid}+{gates.gate_tag()}"
    else:
        gates = None
    mode = resample_mode(encoder, resample)
    check_stock_identity(con, eid, stock_identity(encoder, channel, mode))
    protected = positive_recordings(con) if qc_thresholds is not None else set()
    with_onsets = {row[0] for row in con.execute("SELECT recording_id FROM onsets")}
    store = EmbeddingStore(store_root, eid)
    window_s = round(encoder.window_s, 2)
    hop_s = hop_for_overlap(window_s, overlap)
    report = EmbedReport(eid)
    recordings = recordings.assign(month=recordings["start_utc"].map(month_of))

    def prepare(rec: Any) -> _Audio:
        """Fil de lecture : ce qui se calcule sur le signal sans la base ni l'encodeur."""
        try:
            wav, sr = load_audio(Path(raw_root) / rec.path, channel)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            return _Audio(failed=True)
        audio = _Audio(wav, sr)
        known = parse_flags(rec.qc_flags)
        if qc_thresholds is not None and "indices" not in known:
            audio.flags = qc_flags(qc_indices(wav, sr), qc_thresholds)
            if is_excluded(known | audio.flags) and rec.recording_id not in protected:
                return audio  # sera écarté : pas de débuts de notes
        if signal_cfg is not None and rec.recording_id not in with_onsets:
            audio.onsets = recording_onsets(wav, sr, signal_cfg)
        audio.windows = window_grid(len(wav) / sr, window_s, hop_s)
        audio.cut, audio.rate = encoder_windows(encoder, wav, sr, audio.windows, mode)
        return audio

    for (dataset, site, month), group in recordings.groupby(["dataset", "site", "month"]):
        path = store.consolidate(dataset, site, month)  # morceaux d'un passage interrompu
        done = set(store.read_meta(path)["recording_id"]) if path.exists() else set()
        metas: list[pd.DataFrame] = []
        embs: list[np.ndarray] = []
        todo = [rec for rec in group.itertuples() if rec.recording_id not in done]
        report.skipped += len(group) - len(todo)

        for rec, audio in read_ahead(todo, prepare, ahead):
            if audio.failed:
                report.errors += 1
                continue
            wav, sr = audio.wav, audio.sr
            if audio.flags is not None:
                merged = merge_audio_flags(con, rec.recording_id, audio.flags)
                report.qc_checked += 1
                if is_excluded(merged) and rec.recording_id not in protected:
                    report.qc_excluded += 1
                    continue
            found = None  # débuts de notes de l'enregistrement, s'ils viennent d'être calculés
            if signal_cfg is not None and rec.recording_id not in with_onsets:
                found = audio.onsets
                if found is None:
                    found = recording_onsets(wav, sr, signal_cfg)
                store_onsets(con, rec.recording_id, found, channel)
                with_onsets.add(rec.recording_id)
            if audio.cut is None:  # contrôle audio prévu écartant, démenti par la base
                audio.windows = window_grid(len(wav) / sr, window_s, hop_s)
                audio.cut, audio.rate = encoder_windows(encoder, wav, sr, audio.windows, mode)
            windows, cut, rate = audio.windows, audio.cut, audio.rate
            passed = np.ones(len(windows), dtype=bool)
            if gates is not None:  # notes et rythme : les débuts de notes de l'enregistrement
                if found is None:
                    found = load_onsets(con, {rec.recording_id}).get(rec.recording_id)
                if found is None:
                    found = recording_onsets(wav, sr, gates.signal_cfg)
                values = gate_values(
                    cut if rate == sr else cut_windows(wav, sr, windows),  # f_e d'origine
                    sr,
                    gates.signal_cfg,
                    offsets_s=np.array([o for o, _ in windows]),
                    onsets=found,
                    dur_s=window_s,
                )
                passed = gates.passes(values)
                report.gated += int((~passed).sum())
            start = perf_counter()
            emb = np.zeros((len(windows), encoder.dim), dtype=np.float32)
            if passed.any():
                emb[passed] = encoder.embed(cut[passed], rate)
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
            _store_logits(
                con, encoder, report.encoder_id, [i for i, p in zip(ids, passed, strict=True) if p]
            )
            meta = pd.DataFrame(
                {
                    "window_id": ids,
                    "recording_id": rec.recording_id,
                    "offset_s": [o for o, _ in windows],
                }
            )
            if gates is not None:
                meta[GATED] = ~passed
            metas.append(meta)
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
        store.consolidate(dataset, site, month)

    register_encoder(con, encoder, hop_s, report, channel, gates, mode)
    return report


# Réglage d'un stock rangé avant que ce réglage n'existe.
LEGACY_IDENTITY = {"resample": "window"}


def stock_identity(
    encoder: Encoder, channel: int | str, resample: str = "window"
) -> dict[str, Any]:
    """Réglages qui font qu'un embedding se compare aux autres de son stock sans être dans son
    nom : le canal lu, le checkpoint (bacpipe `birdmae_base`), les transformations en amont
    avec tous leurs réglages, le rééchantillonnage (`resample_mode`). Rangés avec l'encodeur
    (`register_encoder`)."""
    identity: dict[str, Any] = {"channel": channel, "resample": resample}
    inner = getattr(encoder, "inner", encoder)  # UpstreamEncoder, LowpassEncoder
    checkpoint = getattr(encoder, "checkpoint", None) or getattr(inner, "checkpoint", None)
    if checkpoint:
        identity["checkpoint"] = checkpoint
    upstream = getattr(encoder, "upstream", None)
    if upstream is not None and upstream.transforms:
        identity["transforms"] = upstream.transforms
    return json.loads(json.dumps(identity))  # comme relu de la base (listes, pas tuples)


def check_stock_identity(con: sqlite3.Connection, eid: str, identity: dict[str, Any]) -> None:
    """ValueError si le stock `eid` a été encodé avec d'autres réglages (`stock_identity`). Un
    réglage absent de la base (stock d'avant ce contrôle) vaut `LEGACY_IDENTITY`, sinon n'est
    pas comparé."""
    row = con.execute(
        "SELECT params_json FROM models WHERE model_id = ? AND kind = 'encoder'", (eid,)
    ).fetchone()
    if row is None:
        return
    stored = json.loads(row[0] or "{}")
    for key, value in identity.items():
        known = stored.get(key, LEGACY_IDENTITY.get(key))
        if known is not None and known != value:
            raise ValueError(
                f"le stock {eid} a été encodé avec {key} = {known!r}, pas {value!r} : "
                "les embeddings ne se comparent pas ; reprendre avec le même réglage, ou "
                "encoder sous un autre nom (config encoders)"
            )


def register_encoder(
    con: sqlite3.Connection,
    encoder: Encoder,
    hop_s: float,
    report: EmbedReport,
    channel: int | str = "mean",
    gates: Upstream | None = None,
    resample: str = "window",
) -> None:
    params: dict[str, Any] = {
        "sample_rate": encoder.sample_rate,
        "window_s": encoder.window_s,
        "hop_s": hop_s,
        "overlap": round(overlap_of(encoder.window_s, hop_s), 4),
        **stock_identity(encoder, channel, resample),  # canal, checkpoint… (n° 143)
        "dim": encoder.dim,
        "has_tokens": encoder.has_tokens,
        "gates": {"thresholds": gates.gates, "combine": gates.combine} if gates else None,
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
