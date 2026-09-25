"""Exploration en notebook (`notebooks/`) : un enregistrement, ses fenêtres, le module
séquentiel, les négatifs appariés, les embeddings et le prototype différentiel.

Tout est en lecture seule : la base est ouverte en lecture (`open_readonly`), l'audio est lu,
jamais écrit (disques de l'ONF et de Biophonia). Les graphiques sont dans
`blanci/explore_plots.py` (matplotlib, groupe `notebook`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import istft, spectrogram, stft

from blanci.audio import load_audio
from blanci.config import config_path
from blanci.dataset import (
    EXCLUDED_LABELS,
    _pairing_pools,
    current_labels,
    embedded_training_set,
    local_days,
    local_minutes,
    paired_negatives,
    pairing_options,
    positive_annotations,
    recordings_table,
    transfer_labels,
)
from blanci.db import encoder_params, window_id_for
from blanci.embed import month_of
from blanci.frozen import frozen_recordings
from blanci.grid import hop_for_overlap, window_grid
from blanci.head import differential_prototype
from blanci.index import l2_normalize
from blanci.labels import POSITIVE_LABELS
from blanci.qc import is_excluded
from blanci.sequential import (
    GAP_RADIUS_S,
    GATES,
    Upstream,
    detect_onsets,
    gate_threshold,
    gate_values,
    surrounded_by_positives,
)
from blanci.store import EmbeddingStore

PAIRING_ORDER = {"same_recording": 0, "same_day": 1, "other_day": 2}
# Colonnes de `recording_windows` ; les autres sont les scores des détecteurs importés.
WINDOW_COLUMNS = (
    "window_id",
    "recording_id",
    "offset_s",
    "dur_s",
    "center_s",
    "label",
    "y",
    "overlaps_positive",
    "distance_to_positive_s",
    "suspect_fn",
)


def open_readonly(path: Path) -> sqlite3.Connection:
    """Base ouverte en lecture seule : un notebook ne peut rien y écrire."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# --- Enregistrements ----------------------------------------------------------------------------


def recordings_overview(con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Un enregistrement par ligne, avec de quoi choisir : heure locale, fenêtres annotées
    (positives, négatives), détections des détecteurs importés (BlanciNet : nombre, score
    max), exclusion par les drapeaux QC, jeu gelé. Les positifs d'abord."""
    rec = recordings_table(con)
    rec = rec[~rec["recording_id"].duplicated()].copy()
    offset_h = cfg["recorder"]["filename_utc_offset_h"]
    local = pd.to_datetime(rec["start_utc"], utc=True) + pd.Timedelta(hours=offset_h)
    rec["local"] = local.dt.tz_localize(None)
    labels = current_labels(con)
    labels = labels[~labels["label"].isin(EXCLUDED_LABELS)]
    counts = (
        labels.assign(positive=labels["label"].isin(POSITIVE_LABELS))
        .groupby("recording_id")["positive"]
        .agg(n_positive="sum", n_labelled="size")
    )
    rec = rec.join(counts, on="recording_id")
    rec[["n_positive", "n_labelled"]] = rec[["n_positive", "n_labelled"]].fillna(0).astype(int)
    rec["n_negative"] = rec["n_labelled"] - rec["n_positive"]
    detected = pd.read_sql_query(
        """SELECT w.recording_id, s.model_id, COUNT(*) AS n, MAX(s.score) AS best
           FROM scores s JOIN windows w USING (window_id) JOIN models m USING (model_id)
           WHERE m.kind = 'detector' GROUP BY w.recording_id, s.model_id""",
        con,
    )
    for model, group in detected.groupby("model_id"):
        per = group.set_index("recording_id")
        rec[f"n_{model}"] = rec["recording_id"].map(per["n"]).fillna(0).astype(int)
        rec[f"max_{model}"] = rec["recording_id"].map(per["best"])
    rec["excluded"] = rec["qc_flags"].map(is_excluded)
    rec["frozen"] = rec["recording_id"].isin(frozen_recordings(cfg))
    return rec.sort_values(["n_positive", "path"], ascending=[False, True]).reset_index(drop=True)


def read_recording(cfg: dict, path: str, channel: int | str | None = None):
    """Enregistrement entier, mono (canal de la config par défaut) : (forme d'onde, f_e)."""
    channel = cfg["audio"]["channel"] if channel is None else channel
    return load_audio(config_path(cfg, "raw") / path, channel)


def read_segment(
    cfg: dict, path: str, offset_s: float, dur_s: float, channel: int | str | None = None
) -> tuple[np.ndarray, int]:
    """Lecture partielle (seek) d'un segment, complété par des zéros après la fin du fichier."""
    channel = cfg["audio"]["channel"] if channel is None else channel
    with sf.SoundFile(config_path(cfg, "raw") / path) as f:
        sr = f.samplerate
        f.seek(min(max(0, round(offset_s * sr)), f.frames))
        data = f.read(round(dur_s * sr), dtype="float32", always_2d=True)
    if channel == "mean":
        wav = data.mean(axis=1)
    else:
        wav = data[:, min(int(channel), data.shape[1] - 1)]
    return _fit(wav, round(dur_s * sr)), sr


def cut(wav: np.ndarray, sr: int, offset_s: float, dur_s: float) -> np.ndarray:
    """Segment [offset_s, offset_s + dur_s[ d'une forme d'onde, complété par des zéros."""
    start = max(0, round(offset_s * sr))
    return _fit(wav[start : start + round(dur_s * sr)], round(dur_s * sr))


def _fit(wav: np.ndarray, n: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    return np.pad(wav, (0, n - len(wav))) if len(wav) < n else wav[:n]


def spectrogram_db(
    wav: np.ndarray, sr: int, nperseg: int = 512, overlap: float = 0.75
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fréquences Hz, temps s, puissance dB) ; fenêtre de Hann de `nperseg` échantillons."""
    x = np.asarray(wav, dtype=np.float64)
    nperseg = min(nperseg, len(x))
    freqs, times, power = spectrogram(
        x, fs=sr, nperseg=nperseg, noverlap=int(nperseg * overlap), scaling="spectrum"
    )
    return freqs, times, 10 * np.log10(power + 1e-12)


def mean_spectrum_db(wav: np.ndarray, sr: int, nperseg: int = 1024) -> tuple[np.ndarray, ...]:
    """Spectre moyen (dB) d'un segment : (fréquences Hz, niveau dB)."""
    freqs, _, power_db = spectrogram_db(wav, sr, nperseg)
    return freqs, 10 * np.log10(np.mean(10 ** (power_db / 10), axis=1))


# --- Fenêtres ---------------------------------------------------------------------------------


def recording_windows(
    con: sqlite3.Connection,
    cfg: dict,
    recording_id: str,
    grid: str = "w3",
    overlap: float | None = None,
) -> pd.DataFrame:
    """Fenêtres de la grille d'un enregistrement, avec ce qu'on en sait.

    Colonnes : window_id, recording_id, offset_s, dur_s, center_s ; label et y (annotation
    transférée à la grille, vide sinon) ; overlaps_positive, distance_to_positive_s (écart à
    l'annotation positive la plus proche, 0 si chevauchement) ; suspect_fn (encadrée de
    positives, DECISIONS n° 102) ; une colonne par détecteur importé (BlanciNet : score max des
    détections qui couvrent au moins la moitié de la fenêtre ; vide = pas de détection).
    `overlap` (0–0,99) remplace le pas de la grille de la config.
    """
    rec = recordings_table(con).drop_duplicates("recording_id").set_index("recording_id")
    window_s = float(cfg["grids"][grid]["window_s"])
    hop = cfg["grids"][grid]["hop_s"] if overlap is None else hop_for_overlap(window_s, overlap)
    offsets = [o for o, _ in window_grid(rec.at[recording_id, "duration_s"] or 0.0, window_s, hop)]
    out = pd.DataFrame(
        {
            "window_id": [window_id_for(recording_id, o, window_s) for o in offsets],
            "recording_id": recording_id,
            "offset_s": offsets,
            "dur_s": window_s,
        }
    )
    out["center_s"] = out["offset_s"] + window_s / 2
    labels = current_labels(con)
    labels = labels[labels["recording_id"] == recording_id]
    labelled = transfer_labels(labels, out)
    out = out.merge(labelled[["window_id", "label", "y"]], on="window_id", how="left")

    positives = positive_annotations(labels)
    p0 = positives["offset_s"].to_numpy(dtype=float)
    p1 = p0 + positives["dur_s"].to_numpy(dtype=float)
    w0 = out["offset_s"].to_numpy(dtype=float)
    w1 = w0 + window_s
    if len(p0):
        overlaps = ((w0[:, None] < p1[None, :] - 1e-6) & (w1[:, None] > p0[None, :] + 1e-6)).any(1)
        gap = np.maximum(p0[None, :] - w1[:, None], w0[:, None] - p1[None, :]).clip(min=0)
        distance = gap.min(axis=1)
    else:
        overlaps, distance = np.zeros(len(out), dtype=bool), np.full(len(out), np.nan)
    radius = (cfg.get("sequential") or {}).get("gap_radius_s", GAP_RADIUS_S)
    out["overlaps_positive"] = overlaps
    out["distance_to_positive_s"] = distance
    out["suspect_fn"] = ~overlaps & surrounded_by_positives(out["center_s"], (p0 + p1) / 2, radius)
    return out.join(detector_scores(con, out))


def detections(con: sqlite3.Connection, recording_ids: list[str]) -> pd.DataFrame:
    """Détections des détecteurs importés (modèles de type detector : BlanciNet, qui n'exporte
    que les scores ≥ 0,1) : recording_id, model_id, offset_s, dur_s, score."""
    ids = list(dict.fromkeys(recording_ids))
    parts = [
        pd.read_sql_query(
            f"""SELECT w.recording_id, s.model_id, w.offset_s, w.dur_s, s.score
                FROM scores s JOIN windows w USING (window_id) JOIN models m USING (model_id)
                WHERE m.kind = 'detector' AND w.recording_id IN ({", ".join("?" * len(chunk))})
                ORDER BY w.recording_id, w.offset_s""",
            con,
            params=chunk,
        )
        for chunk in (ids[i : i + 500] for i in range(0, len(ids), 500))
    ]
    columns = ["recording_id", "model_id", "offset_s", "dur_s", "score"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def detector_models(con: sqlite3.Connection) -> list[str]:
    """Détecteurs importés (modèles de type detector : BlanciNet)."""
    rows = con.execute("SELECT model_id FROM models WHERE kind = 'detector' ORDER BY model_id")
    return [row[0] for row in rows]


def detector_scores(con: sqlite3.Connection, windows: pd.DataFrame) -> pd.DataFrame:
    """Score de chaque détecteur importé sur des fenêtres (recording_id, offset_s, dur_s) : le
    plus haut des détections qui couvrent au moins la moitié de la fenêtre, vide sans
    détection. Une colonne par détecteur, alignée sur `windows`."""
    found = detections(con, windows["recording_id"].tolist())
    models = sorted(found["model_id"].unique())
    out = pd.DataFrame(np.nan, index=windows.index, columns=models)
    by_recording = dict(tuple(found.groupby("recording_id")))
    for rid, rows in windows.groupby("recording_id").groups.items():
        if rid not in by_recording:
            continue
        w0 = windows.loc[rows, "offset_s"].to_numpy(dtype=float)
        dur = windows.loc[rows, "dur_s"].to_numpy(dtype=float)
        for model, group in by_recording[rid].groupby("model_id"):
            d0 = group["offset_s"].to_numpy(dtype=float)
            d1 = d0 + group["dur_s"].to_numpy(dtype=float)
            shared = np.minimum((w0 + dur)[:, None], d1[None, :]) - np.maximum(w0[:, None], d0)
            covered = shared >= dur[:, None] / 2 - 1e-6
            best = np.where(covered, group["score"].to_numpy()[None, :], -np.inf).max(axis=1)
            out.loc[rows, model] = np.where(np.isfinite(best), best, np.nan)
    return out


def annotation_coverage(con: sqlite3.Connection) -> pd.DataFrame:
    """Pour chaque enregistrement à annotation positive : fenêtres annotées positives,
    secondes couvertes (union des annotations), détections des détecteurs importés hors de
    toute annotation positive (nombre, score médian) — du chant probable, non annoté."""
    positives = positive_annotations(current_labels(con))
    found = detections(con, positives["recording_id"].tolist())
    durations = recordings_table(con).drop_duplicates("recording_id").set_index("recording_id")
    rows = []
    for rid, group in positives.sort_values("offset_s").groupby("recording_id"):
        p0 = group["offset_s"].to_numpy(dtype=float)
        p1 = p0 + group["dur_s"].to_numpy(dtype=float)
        covered, end = 0.0, -np.inf
        for a, b in zip(p0, p1, strict=True):
            covered += max(0.0, b - max(a, end))
            end = max(end, b)
        mine = found[found["recording_id"] == rid]
        d0 = mine["offset_s"].to_numpy(dtype=float)
        d1 = d0 + mine["dur_s"].to_numpy(dtype=float)
        inside = (d0[:, None] < p1[None, :] - 1e-6) & (d1[:, None] > p0[None, :] + 1e-6)
        outside = ~inside.any(axis=1)
        scores = mine["score"].to_numpy(dtype=float)[outside]
        rows.append(
            {
                "recording_id": rid,
                "n_positive": len(group),
                "covered_s": covered,
                "duration_s": durations.at[rid, "duration_s"],
                "detections_outside": int(outside.sum()),
                "median_score_outside": float(np.median(scores)) if len(scores) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n_positive", ascending=False).reset_index(drop=True)


def window_indices(
    wav: np.ndarray,
    sr: int,
    windows: pd.DataFrame,
    cfg: dict,
    onsets: np.ndarray | None = None,
) -> pd.DataFrame:
    """Valeurs des portes (énergie, contraste, notes, rythme) de chaque fenêtre, sur la forme
    d'onde de l'enregistrement déjà lue ; notes et rythme comptés sur `onsets` (débuts de notes
    de l'enregistrement) s'ils sont donnés, comme à l'encodage."""
    segments = np.stack(
        [cut(wav, sr, o, d) for o, d in zip(windows["offset_s"], windows["dur_s"], strict=True)]
    )
    values = gate_values(
        segments,
        sr,
        cfg["signal"],
        offsets_s=windows["offset_s"].to_numpy(dtype=float),
        onsets=onsets,
        dur_s=float(windows["dur_s"].iloc[0]),
    )
    return values.set_axis(windows.index)


# --- Module séquentiel en amont ------------------------------------------------------------------


def upstream_stages(wav: np.ndarray, sr: int, upstream: Upstream) -> list[tuple[str, np.ndarray]]:
    """Le son à chaque étape des transformations actives, dans leur ordre d'application :
    brut, puis après chacune (cumulé) — ce que l'encodeur recevrait."""
    stages = [("brut", np.asarray(wav, dtype=np.float32))]
    for name, opts in upstream.transforms.items():
        step = Upstream({name: opts})
        stages.append((step.transform_tag() or name, step.transform(stages[-1][1], sr)))
    return stages


def gate_report(values: pd.Series, cfg: dict, upstream: Upstream) -> pd.DataFrame:
    """Une ligne par porte : valeur de la fenêtre, seuil de la config, porte activée dans
    `upstream`, passe (une valeur manquante laisse passer, comme `gate_mask`)."""
    rows = []
    for gate in GATES:
        value, threshold = float(values[gate]), gate_threshold(cfg, gate)
        rows.append(
            {
                "gate": gate,
                "value": value,
                "threshold": threshold,
                "enabled": gate in upstream.gates,
                "passes": bool(np.isnan(value) or value >= threshold),
            }
        )
    return pd.DataFrame(rows)


def onset_sweep(
    segments: list[np.ndarray],
    sr: int,
    cfg: dict,
    k_values: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0),
    smooth_values: tuple[float, ...] = (0.005, 0.01, 0.02),
) -> pd.DataFrame:
    """Notes et intervalles d'A. blanci détectés dans chaque segment (fenêtre seule) pour
    chaque réglage (k × MAD, lissage de l'enveloppe) : de quoi régler `signal.onset_k_mad`.
    Une ligne par (segment, k, lissage) : segment, k_mad, smooth_s, notes, rhythm."""
    signal = cfg["signal"]
    band, note_dur = tuple(signal["band_hz"]), tuple(signal["note_dur_s"])
    lo, hi = signal["ioi_range_s"]
    rows = []
    for i, wav in enumerate(segments):
        for k in k_values:
            for smooth in smooth_values:
                onsets = detect_onsets(wav, sr, band, note_dur, k, smooth)
                iois = np.diff(onsets)
                rows.append((i, k, smooth, len(onsets), int(((iois >= lo) & (iois <= hi)).sum())))
    return pd.DataFrame(rows, columns=["segment", "k_mad", "smooth_s", "notes", "rhythm"])


def subtract_background(
    wav: np.ndarray,
    background: np.ndarray,
    sr: int,
    strength: float = 1.0,
    floor: float = 0.1,
    frame_s: float = 0.02,
) -> np.ndarray:
    """Soustraction spectrale du fond d'un autre segment (le négatif apparié) : |X| −
    strength × médiane de |B| par fréquence, jamais sous `floor` × |X| ; la phase de X est
    gardée. L'analogue sonore du prototype différentiel, qui retranche aux positifs le
    centroïde de leurs négatifs appariés (`sequential.denoise` prend le fond dans la fenêtre
    elle-même)."""
    nperseg = int(2 ** round(np.log2(max(16, frame_s * sr))))
    noverlap = nperseg * 3 // 4
    x = np.asarray(wav, dtype=np.float64)
    b = np.asarray(background, dtype=np.float64)
    _, _, spec = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    _, _, noise = stft(b, fs=sr, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(spec)
    profile = np.median(np.abs(noise), axis=-1, keepdims=True)
    cleaned = np.maximum(magnitude - strength * profile, floor * magnitude)
    _, y = istft(cleaned * np.exp(1j * np.angle(spec)), fs=sr, nperseg=nperseg, noverlap=noverlap)
    return _fit(y, len(x))


# --- Négatifs appariés ---------------------------------------------------------------------------


def paired_for_recording(
    con: sqlite3.Connection,
    cfg: dict,
    recording_id: str,
    positive_windows: pd.DataFrame | None = None,
    strategy: str | None = None,
    per_positive: int | None = None,
    grid: str = "w3",
) -> pd.DataFrame:
    """Négatifs appariés d'un enregistrement positif, tirés par `paired_negatives` comme au
    benchmark : même grille, mêmes exclusions (drapeaux QC, jeu gelé, enregistrements positifs,
    fenêtres déjà annotées).

    `positive_windows` (recording_id, offset_s, dur_s) : les annotations positives à prendre ;
    par défaut celles de la base. Pour un enregistrement sans annotation positive, passer la
    fenêtre choisie : on voit les négatifs qu'elle aurait si elle était annotée positive.
    La partie « même enregistrement » de `nearest` est déterministe, identique au benchmark ;
    les tirages au hasard (même jour, autre jour) peuvent différer, le benchmark tirant tous
    les enregistrements positifs à la suite.

    Colonnes en plus : path, local (heure locale), dur_s, minutes_from_positive (écart entre
    les débuts d'enregistrement), distance_to_positive_s (même enregistrement : écart à
    l'annotation positive la plus proche).
    """
    options = pairing_options(cfg) | ({"strategy": strategy} if strategy else {})
    per_positive = per_positive or cfg["benchmark"]["negatives_per_positive"]
    recordings = recordings_table(con).drop_duplicates("recording_id")
    labels = current_labels(con)
    labels = labels[~labels["label"].isin(EXCLUDED_LABELS)]
    positive_recordings = set(labels.loc[labels["label"].isin(POSITIVE_LABELS), "recording_id"])
    if positive_windows is None:
        positive_windows = positive_annotations(labels)
    positive_windows = positive_windows[positive_windows["recording_id"] == recording_id]
    if positive_windows.empty:
        raise ValueError(
            f"{recording_id} n'a pas d'annotation positive : passer `positive_windows` "
            "(la fenêtre choisie) pour voir ses négatifs appariés"
        )
    utc_offset_h = options["utc_offset_h"]
    rec = recordings.set_index("recording_id").assign(
        _minute=lambda r: local_minutes(r["start_utc"], utc_offset_h),
        _day=lambda r: local_days(r["start_utc"], utc_offset_h),
    )
    other_day, same_day = _pairing_pools(
        rec,
        recording_id,
        options["strategy"],
        options["slot_tolerance_min"],
        options["min_gap_min"],
        utc_offset_h,
    )
    flagged = set(recordings.loc[recordings["qc_flags"].map(is_excluded), "recording_id"])
    barred = flagged | frozen_recordings(cfg) | positive_recordings
    pool = [r for r in other_day + [r for rank in same_day for r in rank] if r not in barred]

    window_s = float(cfg["grids"][grid]["window_s"])
    hop = float(cfg["grids"][grid]["hop_s"])
    candidates = pd.DataFrame(
        [
            (window_id_for(rid, o, window_s), rid, o, window_s)
            for rid in [recording_id, *pool]
            for o, _ in window_grid(rec.at[rid, "duration_s"] or 0.0, window_s, hop)
        ],
        columns=["window_id", "recording_id", "offset_s", "dur_s"],
    )
    candidates = candidates[~candidates["window_id"].isin(labels["window_id"])]
    negatives = paired_negatives(
        candidates.reset_index(drop=True),
        recordings,
        {recording_id},
        per_positive,
        seed=cfg["head"]["seed"],
        positive_windows=positive_windows,
        **options,
    ).assign(dur_s=window_s)

    start = pd.to_datetime(rec["start_utc"], utc=True)
    negatives["path"] = negatives["recording_id"].map(rec["path"])
    negatives["local"] = (
        negatives["recording_id"].map(start) + pd.Timedelta(hours=utc_offset_h)
    ).dt.tz_localize(None)
    delta = negatives["recording_id"].map(start) - start[recording_id]
    negatives["minutes_from_positive"] = delta.dt.total_seconds() / 60
    p0 = positive_windows["offset_s"].to_numpy(dtype=float)
    p1 = p0 + positive_windows["dur_s"].to_numpy(dtype=float)
    w0 = negatives["offset_s"].to_numpy(dtype=float)
    gap = np.maximum(p0[None, :] - (w0 + window_s)[:, None], w0[:, None] - p1[None, :]).clip(min=0)
    negatives["distance_to_positive_s"] = np.where(
        negatives["recording_id"] == recording_id, gap.min(axis=1), np.nan
    )
    return negatives


def nearest_first(negatives: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    """Négatifs appariés rangés pour une fenêtre : ceux du même enregistrement d'abord, du
    plus proche au plus loin de `offset_s`, puis ceux du même jour, puis d'un autre jour."""
    rank = negatives["pairing"].map(PAIRING_ORDER).fillna(len(PAIRING_ORDER))
    closeness = np.where(
        negatives["pairing"] == "same_recording",
        (negatives["offset_s"] - offset_s).abs(),
        negatives["minutes_from_positive"].abs() * 60,
    )
    order = np.lexsort((closeness, rank.to_numpy()))
    return negatives.iloc[order].reset_index(drop=True)


# --- Embeddings et prototypes ------------------------------------------------------------------


def embedding_stocks(cfg: dict) -> list[str]:
    """Stocks d'embeddings présents (dossiers de `paths.embeddings`)."""
    root = config_path(cfg, "embeddings")
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def recording_embeddings(
    cfg: dict, encoder_id: str, recording: pd.Series
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fenêtres encodées d'un enregistrement (`recording` : ligne de `recordings_overview`)
    dans le stock `encoder_id`. Seules ses lignes sont lues dans la partition (filtre Parquet) :
    une partition entière ne tiendrait pas toujours en mémoire."""
    store = EmbeddingStore(config_path(cfg, "embeddings"), encoder_id)
    filters = {
        "dataset": str(recording["dataset"]),
        "site": str(recording["site"]),
        "month": month_of(recording["start_utc"]),
    }
    metas, embs = [], []
    for path in store.fragments(filters):
        table = pq.read_table(path, filters=[("recording_id", "=", recording["recording_id"])])
        if not table.num_rows:
            continue
        column = table.column("emb").combine_chunks()
        dim = column.type.list_size
        embs.append(column.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim))
        metas.append(table.drop(["emb"]).to_pandas())
    if not metas:
        return pd.DataFrame(columns=["window_id", "recording_id", "offset_s"]), np.zeros((0, 0))
    meta = pd.concat(metas, ignore_index=True)
    order = np.argsort(meta["offset_s"].to_numpy(), kind="stable")
    return meta.iloc[order].reset_index(drop=True), np.concatenate(embs)[order].astype(np.float32)


def stock_window_s(con: sqlite3.Connection, encoder_id: str, default: float = 3.0) -> float:
    """Durée des fenêtres d'un stock, telle qu'`embed` l'a enregistrée (défaut sinon)."""
    try:
        return float(encoder_params(con, encoder_id.split("@")[0])["window_s"])
    except (ValueError, KeyError):
        return default


def nearest_embedding(
    meta: pd.DataFrame, emb: np.ndarray, center_s: float, window_s: float
) -> tuple[pd.Series, np.ndarray]:
    """La fenêtre du stock dont le centre est le plus proche de `center_s`, et son embedding."""
    if not len(meta):
        raise ValueError("aucune fenêtre encodée pour cet enregistrement")
    i = int(np.argmin(np.abs(meta["offset_s"].to_numpy(dtype=float) + window_s / 2 - center_s)))
    return meta.iloc[i], emb[i]


def compare_embeddings(e_pos: np.ndarray, e_neg: np.ndarray) -> dict:
    """Une fenêtre et son négatif apparié : cosinus, normes, différence des embeddings
    normalisés — le prototype différentiel d'une seule paire (w = ê₊ − ê₋)."""
    pos, neg = l2_normalize(e_pos)[0], l2_normalize(e_neg)[0]
    return {
        "cosine": float(pos @ neg),
        "norm_pos": float(np.linalg.norm(e_pos)),
        "norm_neg": float(np.linalg.norm(e_neg)),
        "difference": pos - neg,
    }


def corpus_prototypes(con: sqlite3.Connection, cfg: dict, encoder_id: str) -> dict:
    """Prototypes appris sur tout le stock, comme la tête `prototype` : différentiel
    (w = μ₊ − μ₋ sur embeddings normalisés, b au milieu) et simple (μ₊), sur les labels et les
    négatifs appariés du benchmark (jeu gelé exclu)."""
    data, X = embedded_training_set(
        con,
        EmbeddingStore(config_path(cfg, "embeddings"), encoder_id),
        stock_window_s(con, encoder_id),
        per_positive=cfg["benchmark"]["negatives_per_positive"],
        seed=cfg["head"]["seed"],
        exclude_recordings=frozen_recordings(cfg),
        **pairing_options(cfg),
    )
    y = data["y"].to_numpy()
    X = np.asarray(X, dtype=np.float32)
    w, b = differential_prototype(X[y == 1], X[y == 0])
    mu_pos = l2_normalize(X[y == 1]).mean(axis=0)
    mu_neg = l2_normalize(X[y == 0]).mean(axis=0)
    return {
        "w": w,
        "b": b,
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "cosine_centroids": float(l2_normalize(mu_pos)[0] @ l2_normalize(mu_neg)[0]),
    }


def prototype_scores(X: np.ndarray, prototypes: dict) -> pd.DataFrame:
    """Scores des prototypes : différentiel (ê·w + b, seuil 0) et simple (cosinus à μ₊)."""
    E = l2_normalize(np.asarray(X, dtype=np.float32))
    return pd.DataFrame(
        {
            "differential": E @ prototypes["w"] + prototypes["b"],
            "simple": E @ l2_normalize(prototypes["mu_pos"])[0],
        }
    )
