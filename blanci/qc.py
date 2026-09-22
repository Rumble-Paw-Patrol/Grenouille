"""Contrôle qualité par enregistrement : pluie, saturation, micro dans sac, silence.

Indices simples (numpy/scipy) ; les seuils de config/default.yaml sont provisoires et se
calibrent sur les enregistrements étiquetés `rain` et `artefact_in_bag`.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch


def qc_indices(wav: np.ndarray, sr: int) -> dict[str, float]:
    x = wav - wav.mean()
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    freqs, psd = welch(x, fs=sr, nperseg=min(2048, len(x)))
    total = psd[freqs >= 100].sum()
    hf = psd[freqs >= 2000].sum()
    mid = psd[(freqs >= 1000) & (freqs <= min(10000, sr / 2))]
    flatness = (
        float(np.exp(np.mean(np.log(mid + 1e-20))) / (mid.mean() + 1e-20)) if len(mid) else 0.0
    )
    return {
        "rms_dbfs": 20 * float(np.log10(max(rms, 1e-10))),
        "peak": float(np.abs(wav).max()) if len(wav) else 0.0,
        "clip_fraction": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
        "hf_ratio": float(hf / total) if total > 0 else 0.0,
        "flatness_1_10k": flatness,
    }


def qc_flags(indices: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    silent = indices["rms_dbfs"] < thresholds["silent_dbfs"]
    return {
        "silent": silent,
        "saturation": indices["clip_fraction"] > thresholds["clip_fraction"],
        "in_bag": not silent and indices["hf_ratio"] < thresholds["in_bag_hf_ratio"],
        "rain": indices["flatness_1_10k"] > thresholds["rain_flatness"]
        and indices["rms_dbfs"] > thresholds["rain_min_dbfs"],
        "indices": {k: round(v, 6) for k, v in indices.items()},
    }


# --- Drapeaux d'inventaire : calculés sur les métadonnées, sans lire l'audio -----------------

METADATA_FLAGS = ("duration_off", "off_campaign")


def metadata_flags(recordings: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """`duration_off` et `off_campaign` par enregistrement.

    - duration_off : durée hors du programme (2 min ± tolérance) — tests, déclenchements
      manuels, fichiers coupés.
    - off_campaign : enregistrement isolé de la série de son micro. Pour chaque couple
      (jeu, site, micro), les dates sont découpées en blocs là où un trou dépasse
      `campaign_gap_days` ; le plus gros bloc est la campagne, les autres en sortent (tests
      d'avant pose, restes de carte SD d'un autre lieu). Un micro posé en continu sur
      plusieurs relevés d'un même site reste un seul bloc.

    `recordings` : recording_id, dataset, site, mic_id, start_utc, duration_s.
    """
    expected = float(thresholds["expected_duration_s"])
    tolerance = float(thresholds["duration_tolerance_s"])
    gap = pd.Timedelta(days=float(thresholds["campaign_gap_days"]))

    df = recordings[["recording_id", "dataset", "site", "mic_id", "start_utc", "duration_s"]]
    out = pd.DataFrame({"recording_id": df["recording_id"]})
    out["duration_off"] = (df["duration_s"] - expected).abs() > tolerance
    out["off_campaign"] = False

    dated = df.assign(t=pd.to_datetime(df["start_utc"], utc=True, errors="coerce"))
    dated = dated.dropna(subset=["t"])
    keys = [dated[c].fillna("?") for c in ("dataset", "site", "mic_id")]
    for _, group in dated.sort_values("t").groupby(keys, sort=False):
        block = (group["t"].diff() > gap).cumsum()
        main = block.value_counts().idxmax()
        outside = group.index[block != main]
        out.loc[outside, "off_campaign"] = True
    return out


def apply_metadata_flags(con: sqlite3.Connection, thresholds: dict[str, Any]) -> dict[str, int]:
    """Recalcule les drapeaux d'inventaire et les fusionne dans `recordings.qc_flags`.

    Seules les clés `duration_off` et `off_campaign` sont réécrites : les indices audio déjà
    calculés (pluie, saturation…) sont conservés. Rien n'est supprimé.
    """
    recordings = pd.read_sql_query(
        "SELECT recording_id, dataset, site, mic_id, start_utc, duration_s, qc_flags "
        "FROM recordings",
        con,
    )
    if recordings.empty:
        return dict.fromkeys(METADATA_FLAGS, 0)
    flags = metadata_flags(recordings, thresholds).set_index("recording_id")
    updates = []
    for rid, current in zip(recordings["recording_id"], recordings["qc_flags"], strict=True):
        merged = json.loads(current) if isinstance(current, str) and current else {}
        merged |= {k: bool(flags.at[rid, k]) for k in METADATA_FLAGS}
        updates.append((json.dumps(merged), rid))
    with con:
        con.executemany("UPDATE recordings SET qc_flags = ? WHERE recording_id = ?", updates)
    return {k: int(flags[k].sum()) for k in METADATA_FLAGS}
