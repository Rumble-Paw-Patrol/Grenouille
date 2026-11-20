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
**les deux pistes** le sont (`muffled_0` et `muffled_1`, chacune par la carte CUSUM,
`muffled_cusum`) ; si une piste est valable il reste dans le corpus avec son drapeau
(DECISIONS n° 203).

Le contrôle audio ne sert qu'à sortir du projet les fichiers cassés ou sans rapport avec lui
(DECISIONS n° 174) : il ne signale ni la pluie ni la saturation, qui font partie du milieu
enregistré (n° 79). Seuls `EXCLUDING_FLAGS` écartent un enregistrement du corpus (jamais
encodé). Un enregistrement où A. blanci a été entendu n'est jamais écarté : l'écoute prime sur
le calcul.
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
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
# Drapeaux confirmés à l'écoute (`confirm_flags`) : clés propres, que ni l'inventaire ni le
# contrôle audio ne réécrivent. Ils écartent comme le drapeau calculé, qu'il soit levé ou non.
CONFIRMED_FLAGS = ("confirmed_muffled", "confirmed_duration_off", "confirmed_off_campaign")
# Écouté et gardé malgré un drapeau (calculé ou posé à l'écoute) : l'emporte sur tous les autres.
KEEP_FLAG = "keep"
EXCLUDING_FLAGS = ("muffled", "silent", "duration_off", "off_campaign", *CONFIRMED_FLAGS)
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
    """Écarté par l'un des drapeaux `keys`, sauf `keep` (écouté et jugé bon, `keep_flags`)."""
    flags = parse_flags(qc)
    if flags.get(KEEP_FLAG):
        return False
    return any(flag_raised(flags, k) for k in keys)


def confirm_flags(
    con: sqlite3.Connection, recording_ids: Iterable[str], flags: Iterable[str]
) -> int:
    """Pose `confirmed_<drapeau>` sur ces enregistrements, après écoute. Renvoie le nombre
    d'enregistrements modifiés. Les autres clés de `qc_flags` ne bougent pas."""
    keys = [f"confirmed_{f}" for f in flags]
    unknown = [k for k in keys if k not in CONFIRMED_FLAGS]
    if unknown:
        raise ValueError(f"drapeau inconnu : {unknown} (attendus : {CONFIRMED_FLAGS})")
    updates = []
    for rid in dict.fromkeys(recording_ids):
        row = con.execute(
            "SELECT qc_flags FROM recordings WHERE recording_id = ?", (rid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"enregistrement absent de la base : {rid}")
        current = parse_flags(row[0])
        new = current | dict.fromkeys(keys, True)
        if new != current:
            updates.append((json.dumps(new), rid))
    _write_flags(con, updates)
    return len(updates)


def keep_flags(con: sqlite3.Connection, recording_ids: Iterable[str]) -> int:
    """Pose `keep` sur ces enregistrements : plus écartés, quels que soient leurs drapeaux.
    Renvoie le nombre d'enregistrements modifiés."""
    updates = []
    for rid in dict.fromkeys(recording_ids):
        row = con.execute(
            "SELECT qc_flags FROM recordings WHERE recording_id = ?", (rid,)
        ).fetchone()
        if row is None:
            raise KeyError(f"enregistrement absent de la base : {rid}")
        current = parse_flags(row[0])
        if not current.get(KEEP_FLAG):
            updates.append((json.dumps(current | {KEEP_FLAG: True}), rid))
    _write_flags(con, updates)
    return len(updates)


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
    d'énergie au-dessus de 2 kHz. Un candidat, pas un drapeau (carte CUSUM, `muffled_cusum`) ;
    sert à décider de lire la piste 1."""
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
    `apply_audio_flags` les remplace par la carte CUSUM de la série du micro (`muffled_cusum`),
    et ne lève `muffled` que si les deux pistes sont retenues (n° 203)."""
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

METADATA_FLAGS = ("duration_off", "off_campaign", "clock_off", "survey_edge")


def survey_of(path: Any) -> str | None:
    """Le relevé d'origine d'un enregistrement : les deux premiers dossiers de son chemin en
    base (`Projet blanci 2025/RELEVE 3 Mataroni - 06-13 janv 2026`), en NFC. `None` sans chemin."""
    if not isinstance(path, str) or not path:
        return None
    parts = path.replace("\\", "/").split("/")
    return unicodedata.normalize("NFC", "/".join(parts[:2]))


def declared_surveys(thresholds: dict[str, Any]) -> dict[tuple[str, str | None], tuple[Any, Any]]:
    """Les relevés aux dates déclarées (`qc.surveys`) : `{(relevé, site ou None): (pose, relève)}`.

    Chaque entrée de la configuration a `survey` (`survey_of`), `start` (jour de la pose) et `end`
    (jour de la relève), jours locaux inclus ; `site` est facultatif (relevé dont les sites n'ont
    pas été relevés le même jour : phénologie 2023-24). Une entrée avec `site` l'emporte sur
    celle du relevé entier. Lève `ValueError` sur une entrée incomplète, une relève avant la
    pose ou un relevé déclaré deux fois."""
    out: dict[tuple[str, str | None], tuple[Any, Any]] = {}
    for entry in thresholds.get("surveys") or ():
        try:
            survey, start, end = entry["survey"], entry["start"], entry["end"]
        except KeyError as err:
            raise ValueError(f"qc.surveys : clé {err} manquante dans {entry!r}") from err
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if end < start:
            raise ValueError(f"qc.surveys : relève avant la pose dans {entry!r}")
        key = (unicodedata.normalize("NFC", str(survey)), entry.get("site"))
        if key in out:
            raise ValueError(f"qc.surveys : relevé déclaré deux fois : {key}")
        out[key] = (start, end)
    return out


def survey_windows(
    recordings: pd.DataFrame, thresholds: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Pose et relève déclarées de chaque enregistrement (`NaT` : relevé sans dates).

    Entrée : `recordings` avec `path` et `site`. Sortie : deux séries `start`, `end` (jours, sans
    fuseau), de même index. Sans colonne `path`, tout est `NaT`."""
    start = pd.Series(pd.NaT, index=recordings.index, dtype="datetime64[ns]")
    end = start.copy()
    declared = declared_surveys(thresholds)
    if not declared or "path" not in recordings:
        return start, end
    survey = recordings["path"].map(survey_of)
    # Les entrées du relevé entier d'abord, celles d'un site ensuite : elles les recouvrent.
    entries = sorted(declared.items(), key=lambda kv: kv[0][1] is not None)
    for (name, site), (first, last) in entries:
        rows = survey == name
        if site is not None:
            rows &= recordings["site"] == site
        start[rows], end[rows] = first, last
    return start, end


def position_in_survey(
    local: pd.Series, first: pd.Series, last: pd.Series, buffer_h: float = 0.0
) -> pd.Series:
    """Où tombe chaque heure locale par rapport à son relevé déclaré (`survey_windows`).

    `dans` : du début du jour de pose à la fin du jour de relève. `bordure avant` / `bordure
    après` : hors de ces jours, mais à moins de `buffer_h` heures (zone tampon : un micro
    récupéré le lendemain de la relève, posé la veille de la pose). `avant la pose` / `après la
    relève` : plus loin, hors relevé. `NaN` : relevé sans dates, ou fichier sans date."""
    buffer = pd.Timedelta(hours=float(buffer_h))
    begin, stop = first, last + pd.Timedelta(days=1)
    pos = pd.Series(np.nan, index=local.index, dtype=object)
    known = first.notna() & local.notna()
    pos[known & (local >= begin) & (local < stop)] = "dans"
    pos[known & (local < begin)] = "avant la pose"
    pos[known & (local < begin) & (local >= begin - buffer)] = "bordure avant"
    pos[known & (local >= stop)] = "après la relève"
    pos[known & (local >= stop) & (local < stop + buffer)] = "bordure après"
    return pos


def undeclared_surveys(recordings: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    """Les relevés sans dates déclarées, jugés par la règle de repli (trou de plus de
    `campaign_gap_days`) : une ligne par relevé d'origine (`survey`), avec son nombre
    d'enregistrements (`recordings`)."""
    start, _ = survey_windows(recordings, thresholds)
    rest = recordings.loc[start.isna()]
    if "path" not in rest or rest.empty:
        return pd.DataFrame(columns=["survey", "recordings"])
    counts = rest["path"].map(survey_of).fillna("(sans chemin)").value_counts().sort_index()
    return counts.rename_axis("survey").rename("recordings").reset_index()


def metadata_flags(
    recordings: pd.DataFrame, thresholds: dict[str, Any], utc_offset_h: float = -3.0
) -> pd.DataFrame:
    """`duration_off`, `off_campaign`, `clock_off` et `survey_edge` par enregistrement.

    - duration_off : durée hors du programme (2 min ± tolérance) — tests, déclenchements
      manuels, fichiers coupés.
    - off_campaign : enregistrement hors des dates de son relevé. Un relevé déclaré dans
      `qc.surveys` (dossier d'origine, jour de pose, jour de relève, en jours locaux à
      `utc_offset_h`) écarte tout fichier daté hors de ces jours et de la zone tampon
      `survey_buffer_h` : essais de réglage d'avant la pose, restes de carte SD, enregistreur
      oublié après la relève.
    - survey_edge : fichier d'un relevé déclaré, hors des jours de pose à relève mais dans la
      zone tampon (`survey_buffer_h`, 24 h : micro posé la veille ou récupéré le lendemain).
      Remarque seulement : il reste dans le corpus. Un relevé sans dates
      déclarées (ou sans `path`) garde l'ancienne règle, de repli : pour chaque couple (jeu,
      site, micro), les dates sont découpées en blocs là où un trou dépasse `campaign_gap_days`,
      le plus gros bloc est la campagne, les autres en sortent. Elle devine les dates de pose et
      se trompe dans les deux sens (essais à 5,8-6,9 jours de la pose laissés passer, micro
      reposé tard écarté) ; `undeclared_surveys` liste les relevés qui y passent encore. Un
      fichier sans date n'est jamais signalé.
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

    t = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    first, last = survey_windows(recordings, thresholds)
    local = (t + pd.Timedelta(hours=utc_offset_h)).dt.tz_localize(None)
    where = position_in_survey(local, first, last, float(thresholds.get("survey_buffer_h", 0.0)))
    out["off_campaign"] = where.isin(("avant la pose", "après la relève")).to_numpy()
    out["survey_edge"] = where.isin(("bordure avant", "bordure après")).to_numpy()

    # Repli : ce que les dates déclarées ne couvrent pas, jugé sur la série de son micro.
    dated = df.assign(t=t)[first.isna()].dropna(subset=["t"])
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
        # L'épisode d'une piste 1 peut déborder celui de la piste 0, et la carte retient des
        # pistes 0 au-dessus du seuil : ces pistes 1 sont lues aussi, de proche en proche.
        todo = [
            by_id[rid] for rid in _unread_neighbours(con, by_id, thresholds) if rid not in todo_ids
        ]
    con.commit()
    return report


def _unread_neighbours(
    con: sqlite3.Connection, known: dict[str, Any], thresholds: dict[str, Any]
) -> list[str]:
    """Enregistrements stéréo de `known` dont la piste 1 n'est pas lue et qu'il faut lire : la
    piste 0 est retenue par la carte CUSUM sans être candidate (dans un épisode, au-dessus de la
    référence), ou l'enregistrement suit ou précède au programme (créneau voisin du même micro,
    nuit sautée, `_programme_ranks`) un candidat piste 1. La série du micro est lue dans toute la
    base, pour que la carte et le programme soient ceux de `apply_audio_flags`."""
    settings = cusum_settings(thresholds)
    rows = []
    for rid, dataset, site, mic, start, channels, qc in con.execute(
        "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags "
        "FROM recordings"
    ):
        flags = parse_flags(qc)
        if "indices" not in flags:
            continue
        rows.append(
            (
                rid,
                dataset,
                site,
                mic,
                start,
                _track_hf(flags["indices"], thresholds),
                "indices_1" not in flags and channels != 1,
                "indices_1" in flags and muffled_candidate(flags["indices_1"], thresholds),
            )
        )
    columns = [*SERIES_COLUMNS, "unread", "cand"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["start_utc"]).sort_values("start_utc", kind="stable")
    retained = muffled_cusum(frame, **settings)
    wanted = set(frame.loc[frame["unread"] & frame["recording_id"].isin(retained), "recording_id"])
    keys = [frame[c].fillna("?") for c in ("dataset", "site", "mic_id")]
    for _, mic in frame.groupby(keys, sort=False):
        seg = _segments(_programme_ranks(mic["start_utc"]), settings["max_missed"])
        same = np.diff(seg) == 0  # deux voisins au programme, sans trou
        cand = mic["cand"].to_numpy()
        near = np.zeros(len(mic), dtype=bool)
        near[1:] |= cand[:-1] & same
        near[:-1] |= cand[1:] & same
        wanted.update(mic["recording_id"][mic["unread"].to_numpy() & near])
    return sorted(wanted & set(known))


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
        # `muffled_*` ne sont ici que des candidats : seule la carte CUSUM
        # (`apply_audio_flags`, n° 203) les confirme. Pas comptés parmi les écartés.
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


# --- Son étouffé : carte CUSUM sur la série de chaque piste de chaque micro ------------------

# Colonnes d'une série pour `muffled_cusum` : `hf_ratio` vide = piste non mesurée (non lue,
# muette) à ce créneau.
SERIES_COLUMNS = ["recording_id", "dataset", "site", "mic_id", "start_utc", "hf_ratio"]
# Plancher de `hf_ratio` avant le logarithme : sous toutes les valeurs mesurées (la plus basse
# est vers 4e-4) ; borne l'apport d'un seul enregistrement à log10(référence / plancher).
HF_FLOOR = 1e-4


def cusum_settings(thresholds: dict[str, Any]) -> dict[str, float]:
    """Réglage de la carte du son étouffé, lu dans `qc` de la configuration : référence
    (`muffled_hf_ratio`), seuil (`muffled_cusum_h`) et créneaux manquants tolérés
    (`muffled_max_missed_slots`)."""
    return {
        "ref_ratio": float(thresholds["muffled_hf_ratio"]),
        "h": float(thresholds.get("muffled_cusum_h", 4.0)),
        "max_missed": int(thresholds.get("muffled_max_missed_slots", 1)),
    }


def _track_hf(indices: dict[str, float] | None, thresholds: dict[str, Any]) -> float:
    """`hf_ratio` d'une piste pour la carte ; NaN si elle n'est pas lue ou si elle est muette
    (une piste muette n'est pas étouffée : elle sort de la série)."""
    if indices is None or indices["rms_dbfs"] < thresholds["silent_dbfs"]:
        return np.nan
    return float(indices["hf_ratio"])


def _programme_ranks(t: pd.Series) -> np.ndarray:
    """Rang de chaque enregistrement d'un micro dans son programme : deux créneaux qui se
    suivent au programme ont des rangs consécutifs, **nuit comprise** (le dernier créneau du
    soir et le premier du lendemain se suivent). `t` : heures de début d'un seul micro.

    Le programme est lu dans les données : les minutes du jour (UTC) présentes au moins un jour
    sur deux, complétées en grille régulière au pas médian entre elles (un créneau manqué reste
    un créneau). Le jour du programme commence après le plus grand écart entre deux créneaux,
    la nuit, ce qui tient aussi pour un programme qui passe minuit UTC. Un enregistrement hors
    grille (essai, fichier coupé) prend le rang du créneau le plus proche. Sans programme
    lisible (moins de deux créneaux), un rang par pas de 30 min.
    """
    t = pd.to_datetime(pd.Series(t), utc=True)
    minutes = ((t - pd.Timestamp("2000-01-01", tz="UTC")).dt.total_seconds() / 60).to_numpy()
    seen = pd.DataFrame({"m": np.floor(minutes % 1440), "d": minutes // 1440}).drop_duplicates()
    share = seen.groupby("m").size() / seen["d"].nunique()
    slots = np.sort(share.index[share >= 0.5].to_numpy())
    if len(slots) < 2:
        return np.round(minutes / 30).astype(np.int64)
    night = int(np.argmax(np.diff(np.append(slots, slots[0] + 1440))))
    start = slots[(night + 1) % len(slots)]
    offsets = np.sort((slots - start) % 1440)
    step = float(np.median(np.diff(offsets)))
    n = int(round(offsets[-1] / step)) + 1
    shifted = minutes - start
    day = np.floor(shifted / 1440)
    position = np.clip(np.round((shifted - day * 1440) / step), 0, n - 1)
    return (day * n + position).astype(np.int64)


def _segments(ranks: np.ndarray, max_missed: int) -> np.ndarray:
    """Numéro de tronçon de chaque enregistrement (rangs dans l'ordre du temps) : un nouveau
    tronçon commence après plus de `max_missed` créneaux manqués. La carte repart de zéro."""
    missed = np.diff(ranks) - 1
    return np.concatenate([[0], np.cumsum(missed > max_missed)]).astype(np.int64)


def _cusum(increments: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Statistique de Page (1954) : S = max(0, S précédent + apport), remise à zéro à chaque
    tronçon."""
    out = np.zeros(len(increments))
    s = 0.0
    for i, inc in enumerate(increments):
        if i and segments[i] != segments[i - 1]:
            s = 0.0
        s = max(0.0, s + inc)
        out[i] = s
    return out


def _alarmed(stat: np.ndarray, h: float, segments: np.ndarray) -> np.ndarray:
    """Points d'une excursion (S > 0 sans interruption, dans un tronçon) dont le maximum
    atteint `h` : l'excursion commence au début estimé de l'épisode (dernier zéro de S)."""
    positive = stat > 0
    follows = np.concatenate([[False], positive[:-1] & (np.diff(segments) == 0)])
    excursion = np.cumsum(positive & ~follows)
    peak = pd.Series(stat).groupby(excursion).transform("max").to_numpy()
    return positive & (peak >= h)


def _cusum_episodes(increments: np.ndarray, segments: np.ndarray, h: float) -> np.ndarray:
    """Points retenus : le cœur de l'épisode (dans une excursion qui alarme en lisant la série
    dans le sens du temps **et** à l'envers : du début estimé à la fin estimée) en entier ; la
    traîne (excursion d'un seul sens, la statistique n'est pas encore retombée) seulement sous
    la référence (apport positif)."""
    forward = _alarmed(_cusum(increments, segments), h, segments)
    backward = _alarmed(_cusum(increments[::-1], segments[::-1]), h, segments[::-1])[::-1]
    return (forward & backward) | ((forward | backward) & (increments > 0))


def muffled_cusum(
    series: pd.DataFrame,
    ref_ratio: float = 0.05,
    h: float = 4.0,
    max_missed: int = 1,
) -> set[str]:
    """Pistes retenues étouffées : carte CUSUM de Page (1954) sur la série de `hf_ratio` d'une
    piste, micro par micro (jeu, site, micro), dans l'ordre du programme.

    Apport de chaque enregistrement : log10(`ref_ratio`) − log10(`hf_ratio`), en décades ;
    positif sous la référence. La référence est le milieu, en log, entre une piste saine
    (médiane 0,75) et une piste dans un sac (médiane 0,0035) : 0,05, la valeur de référence
    de Page pour ce saut. Une alarme quand la somme atteint `h` : à h = 4, un sac typique
    (+1,16 par enregistrement) est retenu au 4e enregistrement, un enregistrement seul jamais
    (apport au plus log10(0,05 / 1e-4) = 2,7). La série suit le programme
    (`_programme_ranks`) : la nuit ne la coupe pas ; plus de `max_missed` créneaux sans mesure
    la coupent. Épisodes retenus : `_cusum_episodes`. Réglage calibré le 19/11/2026
    (`data/reports/son_etouffe/cusum.md`).

    `series` : `SERIES_COLUMNS` ; toutes les lignes du micro servent à lire son programme,
    seules celles qui ont un `hf_ratio` entrent dans la carte.
    """
    t = pd.to_datetime(series["start_utc"], utc=True, errors="coerce")
    ordered = series.assign(t=t).dropna(subset=["t"]).sort_values("t", kind="stable")
    keys = [ordered[c].fillna("?") for c in ("dataset", "site", "mic_id")]
    ref = np.log10(ref_ratio)
    kept: set[str] = set()
    for _, mic in ordered.groupby(keys, sort=False):
        hf = mic["hf_ratio"].to_numpy(dtype=float)
        measured = ~np.isnan(hf)
        if not measured.any():
            continue
        segments = _segments(_programme_ranks(mic["t"])[measured], max_missed)
        increments = ref - np.log10(np.maximum(hf[measured], HF_FLOOR))
        retained = _cusum_episodes(increments, segments, h)
        kept.update(mic["recording_id"].to_numpy()[measured][retained])
    return kept


def apply_audio_flags(con: sqlite3.Connection, thresholds: dict[str, Any]) -> dict[str, int]:
    """Recalcule les drapeaux audio depuis les indices déjà rangés, aux seuils actuels.

    Sans relire l'audio : sert après un changement de réglage. Les enregistrements sans indices
    (contrôle audio jamais fait) ne changent pas. `muffled_0` et `muffled_1` : piste retenue
    par la carte CUSUM de la série de son micro (`muffled_cusum`, réglage `cusum_settings`).
    `muffled` : les deux pistes à la fois (un fichier mono n'a que la piste 0, qui compte pour
    les deux ; n° 203). Un fichier stéréo dont la piste 1 n'a pas été lue et dont la piste 0
    est candidate ou retenue garde son `muffled` d'avant (relancer `blanci qc` pour la lire).
    """
    rows = con.execute(
        "SELECT recording_id, dataset, site, mic_id, start_utc, channels, qc_flags FROM recordings"
    ).fetchall()
    audio_by_id, items = {}, []
    for rid, dataset, site, mic, start, channels, qc in rows:
        flags = parse_flags(qc)
        if "indices" not in flags:
            continue
        audio = qc_flags(flags["indices"], thresholds, flags.get("indices_1"))
        hf_0 = _track_hf(flags["indices"], thresholds)
        hf_1 = _track_hf(flags.get("indices_1"), thresholds)
        if channels == 1:  # une seule piste : la piste 1 est la piste 0
            audio["muffled_1"] = audio["muffled_0"]
            hf_1 = hf_0
        audio_by_id[rid] = audio
        items.append((rid, dataset, site, mic, start, hf_0, hf_1))
    frame = pd.DataFrame(items, columns=[*SERIES_COLUMNS[:-1], "hf_0", "hf_1"])
    settings = cusum_settings(thresholds)
    kept = {
        track: muffled_cusum(frame.rename(columns={f"hf_{track}": "hf_ratio"}), **settings)
        for track in (0, 1)
    }
    updates, counts = [], Counter()
    for rid, qc in ((r[0], r[6]) for r in rows):
        if rid not in audio_by_id:
            continue
        flags = parse_flags(qc)
        audio = audio_by_id[rid]
        # piste 1 à lire : sans elle, on ne sait pas si les deux pistes sont étouffées
        unread = "muffled_1" not in audio and (audio["muffled_0"] or rid in kept[0])
        new = {
            "silent": audio["silent"],
            "muffled_0": rid in kept[0],
            "muffled_1": rid in kept[1] if "muffled_1" in audio else False,
        }
        new["muffled"] = (
            bool(flags.get("muffled")) if unread else new["muffled_0"] and new["muffled_1"]
        )
        merged = flags | new
        if unread:  # piste 1 inconnue : pas de `muffled_1`, même rangé avant (False périmé)
            new.pop("muffled_1")
            merged.pop("muffled_1")
        counts.update(k for k in AUDIO_FLAGS if new[k])
        counts.update(k for k in ("muffled_0", "muffled_1") if new.get(k))
        if any(flags.get(k) != new.get(k) for k in (*AUDIO_FLAGS, "muffled_0", "muffled_1")):
            updates.append((json.dumps(merged), rid))
    _write_flags(con, updates)
    return {k: counts[k] for k in (*AUDIO_FLAGS, "muffled_0", "muffled_1")}


# --- Drapeaux posés à l'écoute --------------------------------------------------------------

# Dernier label d'une fenêtre écoutée → drapeau de son enregistrement. Une mention de pluie
# dans le commentaire (étiquette de condition « rain ») le pose aussi, même sur un positif ; un
# enregistrement dit inutilisable (« poubelle », « dans sac », « deux pistes muffled », « en
# train d'être posé »… : étiquette « discard ») est écarté comme un son étouffé.
ANNOTATION_FLAGS = {"artefact_muffled": "muffled", "artefact_in_bag": "muffled", "rain": "rain"}
CONDITION_FLAGS = {"rain": "rain", "discard": "muffled"}


def _condition_tags(conditions: str | None) -> list[str]:
    """Étiquettes de condition d'un label ou d'un extrait : celles rangées à l'envoi, et celles
    que le commentaire dit aujourd'hui (`labels.comment_fields`, relu à chaque fois : une
    étiquette ajoutée après l'envoi, comme « discard », vaut pour les anciens commentaires)."""
    from blanci.inputs.labels import comment_fields

    if not conditions:
        return []
    stored = json.loads(conditions)
    return [*stored.get("tags", []), *comment_fields(stored.get("comment")).get("tags", [])]


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
        tags = _condition_tags(conditions)
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
