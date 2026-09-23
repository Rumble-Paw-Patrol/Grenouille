"""Calibration du contrôle qualité sur les fenêtres étiquetées (§3, indices acoustiques).

Les seuils de `qc` dans la config sont provisoires. Un drapeau audio a une conséquence :
un enregistrement « micro dans sac » ou « silencieux » n'est jamais encodé
(`embed.select_recordings`), donc jamais scoré. Un seuil trop large ferait disparaître des
enregistrements où A. blanci chante, sans que personne le voie.

La calibration calcule les indices de chaque enregistrement étiqueté (et de chaque fenêtre
étiquetée), les range par groupe — cible du drapeau (`artefact_in_bag`, `rain` ou mention
de pluie) ; à protéger (A. blanci) ; autres — et propose pour chaque drapeau le seuil qui
signale le plus de cibles sans signaler aucun enregistrement à A. blanci. Elle n'écrit
rien dans la config : le seuil proposé se reporte à la main dans `config/local.yaml`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from blanci.labels import POSITIVE_LABELS
from blanci.qc import qc_flags, qc_indices

# (drapeau, indice, sens) : le drapeau se lève quand l'indice est sous (« below ») ou
# au-dessus (« above ») du seuil. Clé de config du seuil.
FLAGS = {
    "in_bag": ("hf_ratio", "below", "in_bag_hf_ratio"),
    "rain": ("flatness_1_10k", "above", "rain_flatness"),
    "silent": ("rms_dbfs", "below", "silent_dbfs"),
    "saturation": ("clip_fraction", "above", "clip_fraction"),
}


def labelled_windows(con: sqlite3.Connection) -> pd.DataFrame:
    """Dernier label de chaque fenêtre, avec ses conditions et le chemin de l'enregistrement."""
    df = pd.read_sql_query(
        """SELECT l.window_id, l.label, l.conditions, w.recording_id, w.offset_s, w.dur_s,
                  r.path
           FROM labels l JOIN windows w USING (window_id) JOIN recordings r USING (recording_id)
           WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)""",
        con,
    )
    tags = df["conditions"].map(lambda c: set(json.loads(c).get("tags", [])) if c else set())
    df["group"] = np.select(
        [
            df["label"] == "artefact_in_bag",
            (df["label"] == "rain") | tags.map(lambda t: "rain" in t),
            df["label"].isin(POSITIVE_LABELS),
        ],
        ["in_bag", "rain", "blanci"],
        default="other",
    )
    return df.drop(columns="conditions")


def calibration_indices(
    windows: pd.DataFrame, raw_root: Path, channel: int | str = 0
) -> pd.DataFrame:
    """Indices QC de chaque fenêtre (`w_*`) et de son enregistrement entier (`r_*`).

    Chaque enregistrement est lu une fois, en entier (lecture seule).
    """
    rows = []
    for path, group in windows.groupby("path"):
        try:
            wav, sr = sf.read(Path(raw_root) / path, dtype="float32", always_2d=True)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            continue
        if channel == "mean":
            x = wav.mean(axis=1)
        else:
            x = wav[:, min(int(channel), wav.shape[1] - 1)]
        whole = {f"r_{k}": v for k, v in qc_indices(x, sr).items()}
        for _, row in group.iterrows():
            start = round(row["offset_s"] * sr)
            cut = x[start : start + round(row["dur_s"] * sr)]
            part = {f"w_{k}": v for k, v in qc_indices(cut, sr).items()} if len(cut) else {}
            rows.append({**row.to_dict(), **whole, **part})
    return pd.DataFrame(rows)


def _suggest(targets: np.ndarray, protected: np.ndarray, direction: str) -> tuple[float, str]:
    """Seuil qui lève le drapeau sur le plus de cibles sans toucher aucun protégé."""
    if not len(targets):
        return float("nan"), "aucune cible étiquetée"
    if not len(protected):
        return float("nan"), "aucun enregistrement à protéger"
    if direction == "below":
        lo, hi = float(np.max(targets)), float(np.min(protected))
        if lo < hi:
            return (lo + hi) / 2, "séparation nette : milieu entre cibles et positifs"
        return hi, "chevauchement : juste sous le positif le plus bas"
    lo, hi = float(np.max(protected)), float(np.min(targets))
    if lo < hi:
        return (lo + hi) / 2, "séparation nette : milieu entre positifs et cibles"
    return lo, "chevauchement : juste au-dessus du positif le plus haut"


def _raised(values: pd.Series, threshold: float, direction: str) -> int:
    """Nombre de valeurs qui lèvent le drapeau à ce seuil (0 si le seuil est indéfini)."""
    if not np.isfinite(threshold):
        return 0
    return int((values < threshold).sum() if direction == "below" else (values > threshold).sum())


def suggest_thresholds(indices: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """Par drapeau : effet du seuil actuel et seuil proposé, au niveau de l'enregistrement
    (celui où le drapeau décide de l'encodage). Un enregistrement est « à protéger » s'il
    contient au moins une fenêtre A. blanci ; « cible » s'il contient une fenêtre du drapeau."""
    recordings = indices.groupby("recording_id").agg(
        groups=("group", lambda g: set(g)), **{c: (c, "first") for c in indices if c[:2] == "r_"}
    )
    rows = []
    has_blanci = recordings["groups"].map(lambda g: "blanci" in g)
    for flag, (index, direction, key) in FLAGS.items():
        column = f"r_{index}"
        protected = recordings.loc[has_blanci, column]
        if flag in ("in_bag", "rain"):
            is_target = recordings["groups"].map(lambda g, f=flag: f in g)
            # Une cible qui contient aussi du chant (pluie + A. blanci) reste à protéger.
            targets = recordings.loc[is_target & ~has_blanci, column]
        else:
            targets = pd.Series(dtype=float)
        current = float(thresholds[key])
        suggestion, why = _suggest(targets.to_numpy(), protected.to_numpy(), direction)
        rows.append(
            {
                "flag": flag,
                "index": index,
                "direction": "sous" if direction == "below" else "au-dessus",
                "config_key": key,
                "current": current,
                "targets": len(targets),
                "targets_flagged_now": _raised(targets, current, direction),
                "blanci_recordings": len(protected),
                "blanci_flagged_now": _raised(protected, current, direction),
                "suggested": suggestion,
                "targets_flagged_suggested": _raised(targets, suggestion, direction),
                "reason": why,
            }
        )
    return pd.DataFrame(rows)


def current_flags(indices: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """Drapeaux que lèveraient les seuils actuels, enregistrement par enregistrement."""
    rows = []
    for rid, group in indices.groupby("recording_id"):
        values = {k[2:]: v for k, v in group.iloc[0].items() if k.startswith("r_")}
        flags = qc_flags(values, thresholds)
        rows.append(
            {
                "recording_id": rid,
                "groups": ",".join(sorted(set(group["group"]))),
                **{k: flags[k] for k in FLAGS},
            }
        )
    return pd.DataFrame(rows)
