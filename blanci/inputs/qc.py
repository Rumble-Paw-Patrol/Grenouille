"""Contrôle qualité par enregistrement : les drapeaux.

Un drapeau est une remarque sur un enregistrement, rangée dans `recordings.qc_flags` (le
fichier n'est jamais touché). Trois origines :
- inventaire, sans lire l'audio : durée anormale (`duration_off`), hors relevé
  (`off_campaign`), horloge douteuse (`clock_off`) ;
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
from scipy import fft
from scipy.signal import get_window, welch

from blanci.inputs.labels import POSITIVE_LABELS

# Drapeaux qui écartent un enregistrement du corpus : jamais encodé, donc ni négatif apparié,
# ni candidat à écouter, ni score. Les autres (pluie, saturation) sont des remarques.
EXCLUDING_FLAGS = ("in_bag", "silent", "duration_off", "off_campaign", "clock_off")
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


def _welch(x: np.ndarray, sr: int, nperseg: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Densité spectrale de `scipy.signal.welch(x, fs=sr, nperseg=nperseg)` (Hann, recouvrement
    de moitié, moyenne de chaque segment retirée), par paquets de segments : trois à quatre fois
    plus vite sur un enregistrement de 2 min, écart relatif de 1e-7 (somme en double précision).
    Un signal de moins de deux segments passe par scipy."""
    if len(x) < 2 * nperseg:
        return welch(x, fs=sr, nperseg=min(nperseg, len(x)))
    step = nperseg // 2
    window = get_window("hann", nperseg).astype(x.dtype)
    segments = np.lib.stride_tricks.sliding_window_view(x, nperseg)[::step]
    power = np.zeros(nperseg // 2 + 1)
    for i in range(0, len(segments), 256):
        chunk = segments[i : i + 256]
        spectrum = fft.rfft((chunk - chunk.mean(axis=-1, keepdims=True)) * window, axis=-1)
        power += (spectrum.real**2 + spectrum.imag**2).sum(axis=0, dtype=np.float64)
    psd = power / (len(segments) * sr * float((window * window).sum()))
    psd[1:-1] *= 2  # spectre d'un seul côté : tout sauf le continu et Nyquist compte double
    return fft.rfftfreq(nperseg, 1 / sr), psd


def qc_indices(wav: np.ndarray, sr: int) -> dict[str, float]:
    x = wav - wav.mean()
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    freqs, psd = _welch(x, sr)
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

METADATA_FLAGS = ("duration_off", "off_campaign", "clock_off")


def metadata_flags(
    recordings: pd.DataFrame, thresholds: dict[str, Any], utc_offset_h: float = -3.0
) -> pd.DataFrame:
    """`duration_off`, `off_campaign` et `clock_off` par enregistrement.

    - duration_off : durée hors du programme (2 min ± tolérance) — tests, déclenchements
      manuels, fichiers coupés.
    - off_campaign : enregistrement isolé de la série de son micro. Pour chaque couple
      (jeu, site, micro), les dates sont découpées en blocs là où un trou dépasse
      `campaign_gap_days` ; le plus gros bloc est la campagne, les autres en sortent (tests
      d'avant pose, restes de carte SD d'un autre lieu). Un micro posé en continu sur
      plusieurs relevés d'un même site reste un seul bloc.
    - clock_off : l'heure du nom de fichier (heure locale de l'enregistreur, à
      `utc_offset_h`) et celle de l'en-tête (`start_utc`, GUANO) diffèrent de plus de
      `clock_tolerance_min` : on ne sait pas laquelle croire (DECISIONS n° 154 : Molokoi
      SMA14636 en avril 2024, en-tête en avance d'une heure). Sans `path`, ou sans
      horodatage dans le nom, rien n'est signalé.

    `recordings` : recording_id, dataset, site, mic_id, start_utc, duration_s ; path.
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

    out["clock_off"] = False
    if "path" in recordings:
        stamp = recordings["path"].astype(str).str.extract(r"_(\d{8}_\d{6})\.\w+$")[0]
        named = pd.to_datetime(stamp, format="%Y%m%d_%H%M%S", errors="coerce")
        header = pd.to_datetime(recordings["start_utc"], utc=True, errors="coerce")
        local = (header + pd.Timedelta(hours=utc_offset_h)).dt.tz_localize(None)
        minutes = (local - named).dt.total_seconds().abs() / 60
        tolerance_min = float(thresholds.get("clock_tolerance_min", 5.0))
        out["clock_off"] = (minutes > tolerance_min).fillna(False).to_numpy()
    return out


def apply_metadata_flags(
    con: sqlite3.Connection, thresholds: dict[str, Any], utc_offset_h: float = -3.0
) -> dict[str, int]:
    """Recalcule les drapeaux d'inventaire et les fusionne dans `recordings.qc_flags`.

    Seules les clés de `METADATA_FLAGS` sont réécrites : les indices audio déjà
    calculés (pluie, saturation…) sont conservés. Rien n'est supprimé.
    """
    recordings = pd.read_sql_query(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, qc_flags "
        "FROM recordings",
        con,
    )
    if recordings.empty:
        return dict.fromkeys(METADATA_FLAGS, 0)
    flags = metadata_flags(recordings, thresholds, utc_offset_h).set_index("recording_id")
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


def check_recordings(
    con: sqlite3.Connection,
    recordings: pd.DataFrame,
    raw_root: Any,
    thresholds: dict[str, Any],
    channel: int | str = 0,
    signal_cfg: dict | None = None,
    workers: int = 4,
    commit_every: int = 100,
    progress_every: int = 1000,
) -> dict[str, int]:
    """Contrôle audio des enregistrements qui ne l'ont pas encore eu, sans encodeur : une
    lecture de chaque fichier, dans `workers` fils. Avec `signal_cfg`, la même lecture donne
    les débuts de notes (module séquentiel) de ceux qui n'en ont pas, sauf s'ils sont écartés.

    À lancer avant le premier encodage (`blanci qc`) : `embed` ne refait ni l'un ni l'autre, et
    n'encode aucun enregistrement écarté. Reprenable : ce qui est fait est sauté.
    """
    from pathlib import Path
    from time import perf_counter

    from blanci.core.ahead import ahead
    from blanci.core.audio import load_audio

    with_onsets: set[str] = set()
    if signal_cfg is not None:
        from blanci.heads.sequential import recording_onsets, store_onsets

        with_onsets = {row[0] for row in con.execute("SELECT recording_id FROM onsets")}
    protected = positive_recordings(con)
    report = {"checked": 0, "excluded": 0, "onsets": 0, "skipped": 0, "errors": 0}
    todo = []
    for rec in recordings.itertuples():
        known = parse_flags(rec.qc_flags)
        need_qc = "indices" not in known
        need_onsets = (
            signal_cfg is not None
            and rec.recording_id not in with_onsets
            and (need_qc or not is_excluded(known) or rec.recording_id in protected)
        )
        if need_qc or need_onsets:
            todo.append((rec, need_qc, need_onsets))
        else:
            report["skipped"] += 1

    def prepare(job: tuple) -> tuple | None:
        rec, need_qc, need_onsets = job
        try:
            wav, sr = load_audio(Path(raw_root) / rec.path, channel)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            return None
        known = parse_flags(rec.qc_flags)
        flags = qc_flags(qc_indices(wav, sr), thresholds) if need_qc else None
        excluded = is_excluded(known | (flags or {})) and rec.recording_id not in protected
        onsets = recording_onsets(wav, sr, signal_cfg) if need_onsets and not excluded else None
        return flags, onsets

    start = perf_counter()
    for n, ((rec, _, _), out) in enumerate(ahead(todo, prepare, 2 * workers, workers), start=1):
        if out is None:
            if not Path(raw_root).exists():  # disque débranché : ne pas tout compter illisible
                con.commit()
                raise RuntimeError(f"{raw_root} n'est plus accessible ; relancer (reprenable)")
            report["errors"] += 1
            continue
        flags, onsets = out
        if flags is not None:
            merged = merge_audio_flags(con, rec.recording_id, flags)
            report["checked"] += 1
            report["excluded"] += is_excluded(merged) and rec.recording_id not in protected
        if onsets is not None:
            store_onsets(con, rec.recording_id, onsets, channel)
            report["onsets"] += 1
        if n % commit_every == 0:
            con.commit()
        if n % progress_every == 0:
            rate = n / (perf_counter() - start)
            left_h = (len(todo) - n) / rate / 3600
            print(
                f"  {n}/{len(todo)} enregistrements, {rate:.1f} par seconde, "
                f"reste ≈ {left_h:.1f} h",
                flush=True,
            )
    con.commit()
    return report


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
