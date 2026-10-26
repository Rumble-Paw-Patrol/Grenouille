"""Contrôle qualité par enregistrement : les drapeaux.

Un drapeau est une remarque sur un enregistrement, rangée dans `recordings.qc_flags` (le
fichier n'est jamais touché). Trois origines :
- inventaire, sans lire l'audio : durée anormale (`duration_off`), hors relevé
  (`off_campaign`), horloge douteuse (`clock_off`, une remarque : n'écarte plus rien,
  DECISIONS n° 191) ;
- audio, calculé sur le son (`blanci qc`, avant tout encodage) : silencieux, son étouffé
  (`muffled`) ; indices simples (numpy/scipy) des deux pistes, seuils de config/default.yaml
  calibrés par `blanci qc-calibrate` ;
- écoute : clé `annotated`, présente sur tout enregistrement annoté à la main, avec la liste
  des drapeaux que l'annotateur y a posés (label `artefact_muffled`, label ou mention de pluie).

Son étouffé (micro dans un sac, bonnette contre le sol…) : un enregistrement n'est écarté que si
**les deux pistes** le sont (`muffled_0` et `muffled_1`, chacune par la règle des suites) ; si
une piste est valable il reste dans le corpus avec son drapeau (DECISIONS n° 203).

Le contrôle audio ne sert qu'à sortir du projet les fichiers cassés ou sans rapport avec lui
(DECISIONS n° 174) : il ne signale ni la pluie ni la saturation, qui font partie du milieu
enregistré (n° 79). Seuls `EXCLUDING_FLAGS` écartent un enregistrement du corpus (jamais
encodé). Un enregistrement où A. blanci a été entendu n'est jamais écarté : l'écoute prime sur
le calcul.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import fft
from scipy.signal import get_window, welch

from blanci.inputs.labels import POSITIVE_LABELS

# Drapeaux qui écartent un enregistrement du corpus : jamais encodé, donc ni négatif apparié,
# ni candidat à écouter, ni score. La pluie notée à l'écoute est une remarque.
EXCLUDING_FLAGS = ("muffled", "silent", "duration_off", "off_campaign")
AUDIO_FLAGS = ("silent", "muffled")
# Ancien nom du drapeau (« micro dans sac », piste 0 seule, avant le n° 203) : lu comme `muffled`
# tant que `blanci qc` n'a pas recalculé la base, donc l'enregistrement reste écarté.
LEGACY_FLAGS = {"in_bag": "muffled"}


def parse_flags(qc: Any) -> dict[str, Any]:
    """Contenu de `recordings.qc_flags` (JSON, dict déjà lu, ou vide). Les anciens noms de
    drapeaux (`LEGACY_FLAGS`) sont lus sous le nouveau, dans les clés et dans `annotated`."""
    if isinstance(qc, dict):
        flags = qc
    else:
        flags = json.loads(qc) if isinstance(qc, str) and qc else {}
    if any(k in LEGACY_FLAGS for k in flags) or any(
        k in LEGACY_FLAGS for k in flags.get("annotated", ())
    ):
        flags = {LEGACY_FLAGS.get(k, k): v for k, v in flags.items()} | {
            k: v for k, v in flags.items() if k not in LEGACY_FLAGS
        }
        if "annotated" in flags:
            flags["annotated"] = [LEGACY_FLAGS.get(k, k) for k in flags["annotated"]]
    return flags


def flag_raised(flags: dict[str, Any], key: str) -> bool:
    """Drapeau levé par le calcul (inventaire, audio) ou posé à l'écoute."""
    return bool(flags.get(key)) or key in flags.get("annotated", ())


def is_excluded(qc: Any, keys: tuple[str, ...] = EXCLUDING_FLAGS) -> bool:
    flags = parse_flags(qc)
    return any(flag_raised(flags, k) for k in keys)


def positive_recordings(con: sqlite3.Connection) -> set[str]:
    """Enregistrements où A. blanci a été entendu : une fenêtre a pour dernier label un
    positif, ou un intervalle positif est encore valable (annotation par intervalles, n° 182)."""
    from blanci.inputs.dataset import _positive_intervals, load_spans

    marks = ", ".join("?" * len(POSITIVE_LABELS))
    rows = con.execute(
        f"""SELECT DISTINCT w.recording_id FROM labels l JOIN windows w USING (window_id)
            WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)
              AND l.label IN ({marks})""",
        POSITIVE_LABELS,
    )
    heard = set(_positive_intervals(*load_spans(con))["recording_id"])
    return {row[0] for row in rows} | heard


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


def muffled_candidate(indices: dict[str, float], thresholds: dict[str, float]) -> bool:
    """Une piste d'un enregistrement ressemble à un son étouffé : pas silencieuse, et peu
    d'énergie au-dessus de 2 kHz. Un candidat, pas un drapeau (règle des suites)."""
    return (
        indices["rms_dbfs"] >= thresholds["silent_dbfs"]
        and indices["hf_ratio"] < thresholds["muffled_hf_ratio"]
    )


def qc_flags(
    indices: dict[str, float],
    thresholds: dict[str, float],
    indices_1: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Drapeaux audio d'après les indices de la piste 0 (micro 1) et, pour un fichier stéréo,
    de la piste 1 (micro 2). Les indices sont gardés avec eux : un seuil changé se réapplique
    sans relire l'audio (`apply_audio_flags`). Tous les indices mesurés sont rangés, y compris
    ceux qu'aucun drapeau ne lit (saturation, platitude du spectre).

    `muffled_0` et `muffled_1` ne sont ici que des candidats, jugés sur un seul enregistrement :
    `apply_audio_flags` ne les garde que dans une suite d'enregistrements consécutifs du même
    micro (n° 194), et ne lève `muffled` que si les deux pistes le sont (n° 203)."""
    out: dict[str, Any] = {
        "silent": bool(indices["rms_dbfs"] < thresholds["silent_dbfs"]),
        "muffled_0": muffled_candidate(indices, thresholds),
        "indices": {k: round(v, 6) for k, v in indices.items()},
    }
    if indices_1 is not None:
        out["muffled_1"] = muffled_candidate(indices_1, thresholds)
        out["indices_1"] = {k: round(v, 6) for k, v in indices_1.items()}
    return out


def audio_qc(source: Any, thresholds: dict[str, float]) -> dict[str, Any]:
    """`qc_flags` d'un fichier audio (chemin ou tampon), les deux pistes lues en une fois."""
    wav, sr = sf.read(source, dtype="float32", always_2d=True)
    second = qc_indices(wav[:, 1], sr) if wav.shape[1] > 1 else None
    return qc_flags(qc_indices(wav[:, 0], sr), thresholds, second)


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
      `clock_tolerance_min` (DECISIONS n° 154 : Molokoi SMA14636 en avril 2024, en-tête en
      avance d'une heure). Remarque seulement : ces enregistrements restent dans le corpus
      (n° 191). Sans `path`, ou sans horodatage dans le nom, rien n'est signalé.

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
    workers: int = 4,
    commit_every: int = 100,
    progress_every: int = 1000,
    renamed: dict[str, str] | None = None,
) -> dict[str, int]:
    """Contrôle audio des enregistrements qui ne l'ont pas encore eu, sans encodeur : une
    lecture de chaque fichier, dans `workers` fils.

    À lancer avant le premier encodage (`blanci qc`) : `embed` ne le refait pas, et n'encode
    aucun enregistrement écarté. Reprenable : ce qui est fait est sauté.

    `renamed` (`audio.renamed`) : l'audio est cherché sous son chemin d'origine, puis sous le nom
    que lui a donné la copie sur un disque du stage.
    """
    from pathlib import Path

    from blanci.core.locate import variants

    protected = positive_recordings(con)
    report = {"checked": 0, "excluded": 0, "skipped": 0, "errors": 0}

    def measured(rec: Any) -> bool:
        """Indices rangés. La piste 1 n'est lue que si la piste 0 est candidate : sans elle,
        l'enregistrement ne peut pas être `muffled` (les deux pistes le sont ou aucune)."""
        flags = parse_flags(rec.qc_flags)
        if "indices" not in flags:
            return False
        return (
            "indices_1" in flags
            or getattr(rec, "channels", 2) == 1
            or not muffled_candidate(flags["indices"], thresholds)
        )

    todo = [rec for rec in recordings.itertuples() if not measured(rec)]
    report["skipped"] = len(recordings) - len(todo)

    def prepare(rec: Any) -> dict[str, Any] | None:
        try:
            names = variants(rec.path, renamed)
            found = next((n for n in names if (Path(raw_root) / n).is_file()), rec.path)
            return audio_qc(Path(raw_root) / found, thresholds)
        except Exception:  # fichier illisible : déjà signalé à l'inventaire
            return None

    by_id = {rec.recording_id: rec for rec in recordings.itertuples()}
    todo_ids: set[str] = set()
    todo = todo or [by_id[rid] for rid in _unread_neighbours(con, by_id, thresholds)]
    while todo:
        todo_ids |= {rec.recording_id for rec in todo}
        _read_tracks(
            con, todo, prepare, workers, report, protected, raw_root, commit_every, progress_every
        )
        # La suite d'une piste 1 peut déborder la suite de la piste 0 : les voisins d'un
        # candidat dont la piste 1 n'a pas été lue le sont aussi, de proche en proche.
        todo = [
            by_id[rid] for rid in _unread_neighbours(con, by_id, thresholds) if rid not in todo_ids
        ]
    con.commit()
    return report


def _unread_neighbours(
    con: sqlite3.Connection, known: dict[str, Any], thresholds: dict[str, Any]
) -> list[str]:
    """Enregistrements stéréo de `known` dont la piste 1 n'est pas lue et qui suivent ou
    précèdent (sans trou de plus de `muffled_max_gap_min`) un candidat piste 1 du même micro."""
    gap = pd.Timedelta(minutes=float(thresholds.get("muffled_max_gap_min", 60.0)))
    rows = [
        (rid, dataset, site, mic, start, channels, parse_flags(qc))
        for rid, dataset, site, mic, start, channels, qc in con.execute(
            "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags "
            "FROM recordings"
        )
        if rid in known
    ]
    frame = pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "mic": [(r[1], r[2], r[3]) for r in rows],
            "t": pd.to_datetime([r[4] for r in rows], utc=True, errors="coerce"),
            "unread": ["indices" in r[6] and "indices_1" not in r[6] and r[5] != 1 for r in rows],
            "cand": [
                "indices_1" in r[6] and muffled_candidate(r[6]["indices_1"], thresholds)
                for r in rows
            ],
        }
    ).dropna(subset=["t"])
    wanted: set[str] = set()
    for _, mic in frame.sort_values("t").groupby("mic", sort=False):
        t, unread, cand, ids = mic["t"], mic["unread"], mic["cand"], mic["id"]
        near_before = (t.diff() <= gap) & cand.shift(fill_value=False)
        near_after = (t.diff(-1).abs() <= gap) & cand.shift(-1, fill_value=False)
        wanted.update(ids[unread & (near_before | near_after)])
    return sorted(wanted)


def _read_tracks(
    con: sqlite3.Connection,
    todo: list[Any],
    prepare: Any,
    workers: int,
    report: dict[str, int],
    protected: set[str],
    raw_root: Any,
    commit_every: int,
    progress_every: int,
) -> None:
    from pathlib import Path
    from time import perf_counter

    from blanci.core.ahead import ahead

    start = perf_counter()
    for n, (rec, flags) in enumerate(ahead(todo, prepare, 2 * workers, workers), start=1):
        if flags is None:
            if not Path(raw_root).exists():  # disque débranché : ne pas tout compter illisible
                con.commit()
                raise RuntimeError(f"{raw_root} n'est plus accessible ; relancer (reprenable)")
            report["errors"] += 1
            continue
        merged = merge_audio_flags(con, rec.recording_id, flags)
        report["checked"] += 1
        # `muffled_*` ne sont ici que des candidats : seule la règle des suites
        # (`apply_audio_flags`, n° 194 et 203) les confirme. Pas comptés parmi les écartés.
        report["excluded"] += (
            is_excluded(merged | {"muffled": False}) and rec.recording_id not in protected
        )
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


def muffled_runs(series: pd.DataFrame, min_run: int = 4, max_gap_min: float = 60.0) -> set[str]:
    """Enregistrements `muffled` retenus : candidats d'une suite d'au moins `min_run`
    enregistrements consécutifs du même micro (jeu, site, micro), sans trou de plus de
    `max_gap_min` minutes entre deux voisins. Un micro dans un sac donne une série continue ;
    un enregistrement grave isolé (singes hurleurs, chœur, orage) n'en est pas une.

    `series` : recording_id, dataset, site, mic_id, start_utc, candidate (bool).
    """
    t = pd.to_datetime(series["start_utc"], utc=True, errors="coerce")
    ordered = series.assign(t=t).dropna(subset=["t"]).sort_values("t")
    keys = [ordered[c].fillna("?") for c in ("dataset", "site", "mic_id")]
    gap = pd.Timedelta(minutes=max_gap_min)
    kept: set[str] = set()
    for _, mic in ordered.groupby(keys, sort=False):
        broken = ~mic["candidate"] | (mic["t"].diff() > gap)
        run = broken.cumsum()
        runs = mic[mic["candidate"]].groupby(run[mic["candidate"]])["recording_id"]
        for _, ids in runs:
            if len(ids) >= min_run:
                kept.update(ids)
    return kept


def apply_audio_flags(con: sqlite3.Connection, thresholds: dict[str, Any]) -> dict[str, int]:
    """Recalcule les drapeaux audio depuis les indices déjà rangés, aux seuils actuels.

    Sans relire l'audio : sert après un changement de seuil. Les enregistrements sans indices
    (contrôle audio jamais fait) ne changent pas. `muffled_0` et `muffled_1` : ratio de la piste
    sous le seuil **et** suite d'au moins `muffled_min_run` enregistrements consécutifs du même
    micro (`muffled_runs`). `muffled` : les deux pistes à la fois (un fichier mono n'a que la
    piste 0). Un fichier stéréo dont la piste 1 n'a pas été lue et dont la piste 0 est
    candidate garde son `muffled` d'avant (relancer `blanci qc` pour lire la piste 1).
    """
    rows = con.execute(
        "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags FROM recordings"
    ).fetchall()
    audio_by_id, series = {}, {0: [], 1: []}
    for rid, dataset, site, mic, start, channels, qc in rows:
        flags = parse_flags(qc)
        if "indices" not in flags:
            continue
        audio = qc_flags(flags["indices"], thresholds, flags.get("indices_1"))
        if channels == 1:  # une seule piste : la piste 1 est la piste 0
            audio["muffled_1"] = audio["muffled_0"]
        audio_by_id[rid] = audio
        for track in (0, 1):
            series[track].append(
                (rid, dataset, site, mic, start, bool(audio.get(f"muffled_{track}")))
            )
    columns = ["recording_id", "dataset", "site", "mic_id", "start_utc", "candidate"]
    min_run = int(thresholds.get("muffled_min_run", 4))
    max_gap = float(thresholds.get("muffled_max_gap_min", 60.0))
    kept = {}
    for track, items in series.items():
        kept[track] = muffled_runs(pd.DataFrame(items, columns=columns), min_run, max_gap)
    updates, counts = [], Counter()
    for rid, qc in ((r[0], r[6]) for r in rows):
        if rid not in audio_by_id:
            continue
        flags = parse_flags(qc)
        audio = audio_by_id[rid]
        unread = "muffled_1" not in audio and audio["muffled_0"]  # piste 1 à lire
        new = {
            "silent": audio["silent"],
            "muffled_0": rid in kept[0],
            "muffled_1": rid in kept[1] if "muffled_1" in audio else False,
        }
        new["muffled"] = (
            bool(flags.get("muffled")) if unread else new["muffled_0"] and new["muffled_1"]
        )
        if unread:
            new.pop("muffled_1")
        counts.update(k for k in AUDIO_FLAGS if new[k])
        counts.update(k for k in ("muffled_0", "muffled_1") if new.get(k))
        if any(flags.get(k) != new.get(k) for k in (*AUDIO_FLAGS, "muffled_0", "muffled_1")):
            updates.append((json.dumps(flags | new), rid))
    _write_flags(con, updates)
    return {k: counts[k] for k in (*AUDIO_FLAGS, "muffled_0", "muffled_1")}


# --- Drapeaux posés à l'écoute --------------------------------------------------------------

# Dernier label d'une fenêtre écoutée → drapeau de son enregistrement. Une mention de pluie
# dans le commentaire (étiquette de condition « rain ») le pose aussi, même sur un positif.
ANNOTATION_FLAGS = {"artefact_muffled": "muffled", "artefact_in_bag": "muffled", "rain": "rain"}
CONDITION_FLAGS = {"rain": "rain"}


def _current_spans(con: sqlite3.Connection) -> list[tuple[str, list[str], str | None]]:
    """(enregistrement, classes entendues avec le label hors intervalles, conditions) de chaque
    extrait encore valable : un extrait réécouté plus tard, entièrement couvert par le nouvel
    extrait, est remplacé (comme `dataset.current_intervals`)."""
    rows = con.execute(
        "SELECT span_id, recording_id, start_s, end_s, other_label, classes, conditions "
        "FROM spans ORDER BY span_id"
    ).fetchall()
    by_recording: dict[str, list[tuple]] = defaultdict(list)
    for row in rows:
        by_recording[row[1]].append(tuple(row))
    out = []
    for span_id, rid, start, end, other, classes, conditions in (tuple(r) for r in rows):
        if any(
            later[0] > span_id and later[2] <= start + 1e-6 and later[3] >= end - 1e-6
            for later in by_recording[rid]
        ):
            continue
        heard = [other, *(json.loads(classes) if classes else [])]
        out.append((rid, heard, conditions))
    return out


def annotation_flags(
    con: sqlite3.Connection, recording_ids: Iterable[str] | None = None
) -> dict[str, list[str]]:
    """{enregistrement annoté : drapeaux posés à l'écoute} (liste vide : rien de signalé).
    Lit les derniers labels de fenêtres et les extraits encore valables (classes cochées,
    label hors intervalles, étiquettes du commentaire, n° 182)."""
    wanted = None if recording_ids is None else set(recording_ids)
    found: dict[str, set[str]] = defaultdict(set)
    rows = con.execute(
        """SELECT w.recording_id, l.label, l.conditions
           FROM labels l JOIN windows w USING (window_id)
           WHERE l.label_id IN (SELECT MAX(label_id) FROM labels GROUP BY window_id)"""
    ).fetchall()
    heard = [(rid, [label], conditions) for rid, label, conditions in rows]
    for rid, labels, conditions in heard + _current_spans(con):
        if wanted is not None and rid not in wanted:
            continue
        flags = found[rid]  # annoté, même sans drapeau
        flags.update(ANNOTATION_FLAGS[label] for label in labels if label in ANNOTATION_FLAGS)
        tags = json.loads(conditions).get("tags", []) if conditions else []
        flags.update(CONDITION_FLAGS[t] for t in tags if t in CONDITION_FLAGS)
    return {rid: sorted(flags) for rid, flags in found.items()}


def apply_annotation_flags(
    con: sqlite3.Connection, recording_ids: Iterable[str] | None = None
) -> dict[str, int]:
    """Réécrit la clé `annotated` d'après les derniers labels et les extraits écoutés (tous
    les enregistrements, ou ceux donnés). Un label corrigé, ou un extrait réécouté, retire le
    drapeau qu'il avait posé. Renvoie le nombre
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
    return {"annotated": len(found), **{k: counts[k] for k in ("muffled", "rain")}}
