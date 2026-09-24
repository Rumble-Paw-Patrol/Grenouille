"""Contrôle qualité par enregistrement : les drapeaux.

Un drapeau est une remarque sur un enregistrement, rangée dans `recordings.qc_flags` (le
fichier n'est jamais touché). Trois origines :
- inventaire, sans lire l'audio : durée anormale (`duration_off`), hors relevé
  (`off_campaign`) ;
- audio, calculé sur le son (à l'inventaire avec contrôle, ou pendant `embed`) : silencieux,
  saturation, micro dans sac, pluie ; indices simples (numpy/scipy), seuils de
  config/default.yaml calibrés par `blanci qc-calibrate` ;
- écoute : clé `annotated`, présente sur tout enregistrement annoté à la main, avec la liste
  des drapeaux que l'annotateur y a posés (label `artefact_in_bag`, label ou mention de pluie).

Seuls `EXCLUDING_FLAGS` écartent un enregistrement du corpus (jamais encodé) ; pluie et
saturation sont des remarques : un micro sous la pluie enregistre son milieu, et ces
enregistrements font partie du jeu de données (DECISIONS n° 79). Un enregistrement où
A. blanci a été entendu n'est jamais écarté : l'écoute prime sur le calcul.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch

from blanci.labels import POSITIVE_LABELS

# Drapeaux qui écartent un enregistrement du corpus : jamais encodé, donc ni négatif apparié,
# ni candidat à écouter, ni score. Les autres (pluie, saturation) sont des remarques.
EXCLUDING_FLAGS = ("in_bag", "silent", "duration_off", "off_campaign")
AUDIO_FLAGS = ("silent", "saturation", "in_bag", "rain")


def parse_flags(qc: Any) -> dict[str, Any]:
    """Contenu de `recordings.qc_flags` (JSON, dict déjà lu, ou vide)."""
    if isinstance(qc, dict):
        return qc
    return json.loads(qc) if isinstance(qc, str) and qc else {}


def flag_raised(flags: dict[str, Any], key: str) -> bool:
    """Drapeau levé par le calcul (inventaire, audio) ou posé à l'écoute."""
    return bool(flags.get(key)) or key in flags.get("annotated", ())


def is_excluded(qc: Any, keys: tuple[str, ...] = EXCLUDING_FLAGS) -> bool:
    flags = parse_flags(qc)
    return any(flag_raised(flags, k) for k in keys)


def positive_recordings(con: sqlite3.Connection) -> set[str]:
    """Enregistrements dont au moins une fenêtre a pour dernier label un positif A. blanci."""
    marks = ", ".join("?" * len(POSITIVE_LABELS))
    rows = con.execute(
        f"""SELECT DISTINCT w.recording_id FROM labels l JOIN windows w USING (window_id)
            WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)
              AND l.label IN ({marks})""",
        POSITIVE_LABELS,
    )
    return {row[0] for row in rows}


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
    """Drapeaux audio d'après les indices. Les indices sont gardés avec eux : un seuil changé
    se réapplique sans relire l'audio (`apply_audio_flags`)."""
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


def _write_flags(con: sqlite3.Connection, updates: list[tuple[str, str]]) -> None:
    if updates:
        with con:
            con.executemany("UPDATE recordings SET qc_flags = ? WHERE recording_id = ?", updates)


def merge_audio_flags(
    con: sqlite3.Connection, recording_id: str, audio: dict[str, Any]
) -> dict[str, Any]:
    """Range les drapeaux audio d'un enregistrement sans effacer les autres ; renvoie le tout."""
    row = con.execute(
        "SELECT qc_flags FROM recordings WHERE recording_id = ?", (recording_id,)
    ).fetchone()
    merged = parse_flags(row[0] if row else None) | audio
    _write_flags(con, [(json.dumps(merged), recording_id)])
    return merged


def apply_audio_flags(con: sqlite3.Connection, thresholds: dict[str, Any]) -> dict[str, int]:
    """Recalcule les drapeaux audio depuis les indices déjà rangés, aux seuils actuels.

    Sans relire l'audio : sert après un changement de seuil. Les enregistrements sans indices
    (contrôle audio jamais fait) ne changent pas.
    """
    updates, counts = [], Counter()
    for rid, qc in con.execute("SELECT recording_id, qc_flags FROM recordings"):
        flags = parse_flags(qc)
        if "indices" not in flags:
            continue
        audio = qc_flags(flags["indices"], thresholds)
        counts.update(k for k in AUDIO_FLAGS if audio[k])
        if any(flags.get(k) != audio[k] for k in AUDIO_FLAGS):
            updates.append((json.dumps(flags | audio), rid))
    _write_flags(con, updates)
    return {k: counts[k] for k in AUDIO_FLAGS}


# --- Drapeaux posés à l'écoute --------------------------------------------------------------

# Dernier label d'une fenêtre écoutée → drapeau de son enregistrement. Une mention de pluie
# dans le commentaire (étiquette de condition « rain ») le pose aussi, même sur un positif.
ANNOTATION_FLAGS = {"artefact_in_bag": "in_bag", "rain": "rain"}
CONDITION_FLAGS = {"rain": "rain"}


def annotation_flags(
    con: sqlite3.Connection, recording_ids: Iterable[str] | None = None
) -> dict[str, list[str]]:
    """{enregistrement annoté : drapeaux posés à l'écoute} (liste vide : rien de signalé)."""
    wanted = None if recording_ids is None else set(recording_ids)
    found: dict[str, set[str]] = defaultdict(set)
    rows = con.execute(
        """SELECT w.recording_id, l.label, l.conditions
           FROM labels l JOIN windows w USING (window_id)
           WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)"""
    )
    for rid, label, conditions in rows:
        if wanted is not None and rid not in wanted:
            continue
        flags = found[rid]  # annoté, même sans drapeau
        if label in ANNOTATION_FLAGS:
            flags.add(ANNOTATION_FLAGS[label])
        tags = json.loads(conditions).get("tags", []) if conditions else []
        flags.update(CONDITION_FLAGS[t] for t in tags if t in CONDITION_FLAGS)
    return {rid: sorted(flags) for rid, flags in found.items()}


def apply_annotation_flags(
    con: sqlite3.Connection, recording_ids: Iterable[str] | None = None
) -> dict[str, int]:
    """Réécrit la clé `annotated` d'après les derniers labels (tous les enregistrements, ou
    ceux donnés). Un label corrigé retire le drapeau qu'il avait posé. Renvoie le nombre
    d'enregistrements annotés et, par drapeau, le nombre d'enregistrements signalés."""
    wanted = None if recording_ids is None else set(recording_ids)
    found = annotation_flags(con, wanted)
    updates = []
    for rid, qc in con.execute("SELECT recording_id, qc_flags FROM recordings"):
        if wanted is not None and rid not in wanted:
            continue
        flags = parse_flags(qc)
        if rid in found:
            new = flags | {"annotated": found[rid]}
        else:
            new = {k: v for k, v in flags.items() if k != "annotated"}
        if new != flags:
            updates.append((json.dumps(new), rid))
    _write_flags(con, updates)
    counts = Counter(flag for flags in found.values() for flag in flags)
    return {"annotated": len(found), **{k: counts[k] for k in ("in_bag", "rain")}}
