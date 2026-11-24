"""Instruments de lecture de l'inventaire : chaque chiffre sur le corpus sort d'une fonction.

Ce module ne fait pas partie de la chaîne : `blanci/inputs/ingest.py` et `qc.py` restent seuls
à écrire dans la base. Ici, tout est en lecture :

- la base s'ouvre en lecture seule (`open_database`, `mode=ro`) ; aucune fonction n'écrit dans
  une table, ni dans `qc_flags` ;
- l'audio et ses en-têtes sont lus, jamais écrits (disques de l'ONF et de Biophonia) ;
- chaque fonction reçoit ses entrées (connexion, DataFrame, configuration déjà chargée par
  `blanci.core.config.load_config`) et rend un DataFrame ou un dict ; rien n'est affiché ni écrit.

Le DataFrame de départ est `load_recordings` : une ligne par enregistrement, avec l'heure locale,
le jour, le créneau et les drapeaux lus. Les fonctions qui lisent le disque (`read_headers`,
`file_size_check`, `disk_inventory`) prennent un échantillon explicite et lèvent
`FileNotFoundError` quand le disque n'est pas branché. `database_fingerprint` date un chiffre :
la base change tous les jours, un chiffre sans l'état de la base ne se rejoue pas.

`presentation_4_numbers` rejoue les chiffres de la présentation n° 4 et dit, pour chacun, s'il
tient encore. La table des matières, une fiche par fonction : `inventaire.md`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

from blanci.core.config import config_path
from blanci.core.locate import locate, variants
from blanci.inputs.ingest import (
    AUDIO_SUFFIXES,
    SONGMETER_NAME,
    iter_audio_files,
    parse_guano_timestamp,
    read_guano,
)
from blanci.inputs.qc import (
    EXCLUDING_FLAGS,
    flag_raised,
    is_excluded,
    metadata_flags,
    muffled_candidate,
    parse_flags,
    position_in_survey,
    survey_windows,
)

# Un point d'écoute : le micro à un site, pour un jeu (en 2023, la station garde ses micros
# successifs : `station_successions`, `empty_days(level="station")`).
POINT = ["dataset", "site", "mic_id"]
# Jeux du projet ; la base contient aussi `2018` (tests d'un ancien enregistreur) et `reference`
# (chants dirigés), que la présentation n° 4 ne compte pas (`dataset_counts` les montre).
PROJECT_DATASETS = ("2023", "2026")


# =============================================================================================
# 0. Socle et provenance
# =============================================================================================


def open_database(cfg: dict[str, Any]) -> sqlite3.Connection:
    """Ouvre la base de `paths.db` en lecture seule.

    Question : sur quelle base calcule-t-on ? Entrée : la configuration chargée. Sortie : une
    connexion `mode=ro` (toute écriture lève `sqlite3.OperationalError`). Règle : reprend
    `blanci.exploration.explore.open_readonly`. Limite : la base doit exister ; sinon
    `FileNotFoundError`.
    """
    from blanci.exploration.explore import open_readonly

    return open_readonly(config_path(cfg, "db"))


def database_fingerprint(con: sqlite3.Connection) -> dict[str, Any]:
    """État de la base au moment du calcul : de quoi dire « ce chiffre vaut pour cette base ».

    Question : la base sur laquelle je rejoue un chiffre est-elle celle d'où il vient ?
    Entrée : une connexion. Sortie : dict avec `user_version` (version du schéma), `size_bytes`
    (pages × taille de page), `rows` (lignes par table), `inventory_sha256` (empreinte de
    `recordings` sans `qc_flags` : chemins, sites, micros, heures, durées), `flags_sha256`
    (empreinte des seuls `qc_flags`), `last_span` et `last_label` (dernière annotation).
    Règle : SHA-256 des lignes triées par `recording_id`, champs séparés par une tabulation ;
    deux bases de même empreinte ont le même inventaire, au bit près.
    Limite : compte toutes les lignes de toutes les tables (quelques secondes sur la table
    `scores`) ; l'empreinte ne dit pas ce qui a changé, seulement que quelque chose a changé.
    """
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    rows = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    inventory, flags = hashlib.sha256(), hashlib.sha256()
    for r in con.execute(
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, sample_rate, "
        "channels, qc_flags FROM recordings ORDER BY recording_id"
    ):
        inventory.update(("\t".join(str(v) for v in tuple(r)[:9]) + "\n").encode())
        flags.update((f"{r[0]}\t{r[9]}\n").encode())

    def last(table: str) -> str | None:
        return con.execute(f"SELECT MAX(created_at) FROM {table}").fetchone()[0] \
            if table in tables else None

    page_size = con.execute("PRAGMA page_size").fetchone()[0]
    pages = con.execute("PRAGMA page_count").fetchone()[0]
    return {
        "user_version": con.execute("PRAGMA user_version").fetchone()[0],
        "size_bytes": int(page_size * pages),
        "rows": rows,
        "inventory_sha256": inventory.hexdigest(),
        "flags_sha256": flags.hexdigest(),
        "last_span": last("spans"),
        "last_label": last("labels"),
    }


def load_recordings(
    con: sqlite3.Connection,
    cfg: dict[str, Any],
    datasets: Sequence[str] | None = PROJECT_DATASETS,
) -> pd.DataFrame:
    """La table `recordings`, une ligne par enregistrement, avec ce que toutes les autres
    fonctions lisent.

    Question : quels enregistrements a-t-on, à quelle heure locale, avec quels drapeaux ?
    Entrée : connexion, configuration, jeux gardés (`None` = tous, y compris `2018` et
    `reference`). Sortie : les colonnes de la table, plus `fl` (dict lu par
    `qc.parse_flags`), `t` (début en UTC, sans fuseau), `local` (heure locale à
    `recorder.filename_utc_offset_h`), `day`, `slot` (« HH:MM » local), `excluded`
    (`qc.is_excluded`), `campaign` (sans `off_campaign`), `normal` (durée à
    `qc.duration_tolerance_s` de `qc.expected_duration_s`), `survey` (les deux premiers dossiers
    du chemin : le relevé d'origine), `family` (trois premiers caractères du micro : SMA, 2LA,
    2MA).
    Règle : reprise de `charger()` (`documentation/prez/scripts/presentation_4_figures.py`) et
    des cellules d'ouverture des notebooks 05 et 06 ; `campaign` lit `off_campaign` comme
    elles (`flags.get`).
    Limite : l'heure locale suppose un décalage fixe (Guyane, UTC−3, sans heure d'été) ; les
    drapeaux sont ceux rangés, à jour seulement si `blanci flag` a tourné après le dernier
    changement de configuration (`stale_flags`).
    """
    query = (
        "SELECT recording_id, path, dataset, site, mic_id, start_utc, duration_s, sample_rate, "
        "channels, sha256, qc_flags FROM recordings"
    )
    params: list[str] = []
    if datasets is not None:
        query += f" WHERE dataset IN ({', '.join('?' * len(datasets))})"
        params = [str(d) for d in datasets]
    rec = pd.read_sql_query(query, con, params=params)
    rec["dataset"] = rec["dataset"].astype(str)
    rec["fl"] = rec["qc_flags"].map(parse_flags)
    offset = pd.Timedelta(hours=float(cfg["recorder"]["filename_utc_offset_h"]))
    rec["t"] = pd.to_datetime(rec["start_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    rec["local"] = rec["t"] + offset
    rec["day"] = rec["local"].dt.normalize()
    rec["slot"] = rec["local"].dt.strftime("%H:%M")
    rec["excluded"] = rec["qc_flags"].map(is_excluded)
    rec["campaign"] = ~rec["fl"].map(lambda f: bool(f.get("off_campaign")))
    qc = cfg["qc"]
    rec["normal"] = (rec["duration_s"] - float(qc["expected_duration_s"])).abs() <= float(
        qc["duration_tolerance_s"])
    rec["survey"] = rec["path"].str.split("/").str[:2].str.join("/")
    rec["family"] = rec["mic_id"].str[:3]
    return rec


def dataset_counts(con: sqlite3.Connection) -> pd.DataFrame:
    """Tous les jeux de la base, y compris ceux que les chiffres du projet ne comptent pas.

    Question : qu'y a-t-il dans la base hors des jeux 2023 et 2026 ? Entrée : connexion.
    Sortie : une ligne par jeu (`dataset`, `recordings`, `hours`, `recorders`, `in_project`).
    Règle : `in_project` = jeu de `PROJECT_DATASETS`. Limite : aucune.
    """
    out = pd.read_sql_query(
        "SELECT dataset, COUNT(*) AS recordings, SUM(duration_s) / 3600.0 AS hours, "
        "COUNT(DISTINCT mic_id) AS recorders FROM recordings GROUP BY dataset ORDER BY dataset",
        con,
    )
    out["dataset"] = out["dataset"].astype(str)
    out["in_project"] = out["dataset"].isin(PROJECT_DATASETS)
    return out


# =============================================================================================
# 1. Structure du corpus
# =============================================================================================


def corpus_totals(rec: pd.DataFrame) -> dict[str, Any]:
    """Les totaux de l'inventaire.

    Question : combien d'enregistrements, d'heures, d'enregistreurs, de points ? Entrée :
    `load_recordings`. Sortie : dict `recordings`, `hours`, `recorders` (numéros de série
    distincts), `points` (couples jeu, site, micro distincts), `sites`, `excluded`, et
    `by_dataset` (les mêmes par jeu).
    Règle : un enregistreur posé à deux sites compte une fois en `recorders`, deux fois en
    `points`. Limite : les heures somment les durées réelles (un fichier coupé compte pour ce
    qu'il dure), y compris les écartés.
    """

    def totals(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "recordings": int(len(df)),
            "hours": float(df["duration_s"].sum() / 3600),
            "recorders": int(df["mic_id"].nunique()),
            "points": int(df.groupby(POINT).ngroups),
            "sites": int(df["site"].nunique()),
            "excluded": int(df["excluded"].sum()),
        }

    out = totals(rec)
    out["by_dataset"] = {ds: totals(g) for ds, g in rec.groupby("dataset")}
    return out


def counts_by(rec: pd.DataFrame, keys: str | Sequence[str]) -> pd.DataFrame:
    """Effectifs selon n'importe quel regroupement (jeu, site, micro, jour, heure, relevé…).

    Question : combien d'enregistrements, d'heures, de micros, sur quelles dates, par groupe ?
    Entrée : `load_recordings` (ou un sous-ensemble), clés de regroupement (colonnes ; pour
    l'heure, ajouter `rec.local.dt.hour` sous un nom). Sortie : une ligne par groupe :
    `recordings`, `hours`, `recorders`, `first_day`, `last_day`, `days` (jours distincts),
    `excluded`. Règle : reprise du tableau du § 2 du notebook 05 et de `fig_sites`.
    Limite : les groupes vides n'apparaissent pas (un jour sans enregistrement : `empty_days`).
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    return rec.groupby(keys).agg(
        recordings=("recording_id", "size"),
        hours=("duration_s", lambda s: s.sum() / 3600),
        recorders=("mic_id", "nunique"),
        first_day=("day", "min"),
        last_day=("day", "max"),
        days=("day", "nunique"),
        excluded=("excluded", "sum"),
    ).reset_index()


def audio_formats(rec: pd.DataFrame) -> pd.DataFrame:
    """Fréquences d'échantillonnage et nombres de pistes.

    Question : tout est-il en 48 kHz stéréo ? Entrée : `load_recordings`. Sortie : une ligne
    par (jeu, fréquence, pistes) : `recordings`, `recorders`, `median_duration_s`,
    `normal_duration` (combien ont la durée du programme). Règle : lu dans la base (en-tête
    libsndfile à l'inventaire). Limite : ne dit rien du codage (16 ou 24 bits).
    """
    return rec.groupby(["dataset", "sample_rate", "channels"], dropna=False).agg(
        recordings=("recording_id", "size"),
        recorders=("mic_id", "nunique"),
        median_duration_s=("duration_s", "median"),
        normal_duration=("normal", "sum"),
    ).reset_index()


def survey_folders(rec: pd.DataFrame) -> pd.DataFrame:
    """Les relevés tels qu'ils sont rangés sur le disque d'origine.

    Question : combien d'enregistrements vient de chaque relevé (dossier), sur quels sites et
    quelles dates ? Entrée : `load_recordings`. Sortie : une ligne par (jeu, relevé) :
    `recordings`, `sites`, `recorders`, `first_day`, `last_day`, `excluded`.
    Règle : le relevé est le couple des deux premiers dossiers du chemin rangé (« Projet
    blanci 2025/RELEVE 1 CDR - 21-27 décembre 2025 »). L'ordre des relevés compte : à
    l'inventaire, le premier fichier d'un nom l'emporte (`ingest`). Limite : un fichier
    déplacé garde le relevé de son nouveau chemin.
    """
    out = rec.groupby(["dataset", "survey"]).agg(
        recordings=("recording_id", "size"),
        sites=("site", lambda s: ", ".join(sorted(s.dropna().unique()))),
        recorders=("mic_id", "nunique"),
        first_day=("day", "min"),
        last_day=("day", "max"),
        excluded=("excluded", "sum"),
    ).reset_index()
    return out.sort_values(["dataset", "first_day"]).reset_index(drop=True)


def station_successions(
    rec: pd.DataFrame, stations: dict[str, dict[str, list[str]]]
) -> pd.DataFrame:
    """Les micros successifs d'une même station 2023, et ce qui se passe au changement.

    Question : quand chaque micro d'une station a-t-il pris le relais, avec quel trou ou quel
    recouvrement ? Entrée : `load_recordings`, `plan.stations_2023` de la configuration
    (site → station → micros). Sortie : une ligne par (station, micro) : `first_day`,
    `last_day`, `recordings`, `gap_to_next_days` (jours sans micro avant le suivant ; négatif =
    les deux enregistrent en même temps).
    Règle : enregistrements de la campagne seulement (`campaign`), jeu 2023.
    Limite : la liste des micros d'une station vient de la configuration, pas des données ; un
    micro absent de la liste n'apparaît pas.
    """
    rows = []
    r23 = rec[(rec["dataset"] == "2023") & rec["campaign"]]
    for site, sts in stations.items():
        for station, mics in sts.items():
            spans = []
            for mic in mics:
                g = r23[(r23["site"] == site) & (r23["mic_id"] == mic)]
                if g.empty:
                    continue
                spans.append({"site": site, "station": station, "mic_id": mic,
                              "first_day": g["day"].min(), "last_day": g["day"].max(),
                              "recordings": len(g)})
            spans.sort(key=lambda s: s["first_day"])
            for a, b in zip(spans, spans[1:] + [None], strict=True):
                a["gap_to_next_days"] = (
                    (b["first_day"] - a["last_day"]).days - 1 if b is not None else np.nan)
            rows.extend(spans)
    return pd.DataFrame(rows, columns=["site", "station", "mic_id", "first_day", "last_day",
                                       "recordings", "gap_to_next_days"])


# =============================================================================================
# 2. Programme et complétude
# =============================================================================================


def _minutes(slot: str) -> int:
    h, m = slot.split(":")
    return int(h) * 60 + int(m)


def recording_programme(rec: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """Le programme d'enregistrement de chaque point, lu dans les données.

    Question : à quelles heures chaque micro enregistre-t-il ? Entrée : `load_recordings`,
    `min_share` (part des jours où un créneau doit être présent pour être au programme).
    Sortie : une ligne par point : `slots` (créneaux « HH:MM »), `first`, `last`, `n_slots`,
    `step_min` (écart médian entre créneaux), `label` (« 05:00-19:30 »), `inner_days`.
    Règle : enregistrements de la campagne et de durée normale ; jours intérieurs seulement
    (ni le premier ni le dernier jour du point, pose et relève) ; un créneau est au programme
    s'il est présent au moins `min_share` des jours. Reprise de `programmes()`
    (`presentation_4_figures.py`) et du § 4 du notebook 05. `min_share = 0,5` n'est pas dans
    `config/default.yaml` : c'est un choix de lecture, pas un seuil du projet.
    Limite : un micro qui n'a jamais tenu son programme (batterie faible dès le premier jour)
    a pour programme ce qu'il a fait ; un point de moins de trois jours n'a pas de jour
    intérieur, donc pas de programme.
    """
    ok = rec[rec["campaign"] & rec["normal"]]
    days = ok.groupby([*POINT, "day"])["slot"].agg(frozenset).reset_index()
    edge = days.groupby(POINT)["day"].transform
    inner = days[(days["day"] != edge("min")) & (days["day"] != edge("max"))]
    rows = []
    for key, g in inner.groupby(POINT):
        freq = pd.Series([s for x in g["slot"] for s in x]).value_counts() / len(g)
        slots = sorted(freq[freq >= min_share].index)
        if not slots:
            continue
        minutes = [_minutes(s) for s in slots]
        rows.append({
            **dict(zip(POINT, key, strict=True)),
            "slots": tuple(slots), "first": slots[0], "last": slots[-1],
            "n_slots": len(slots),
            "step_min": float(np.median(np.diff(minutes))) if len(minutes) > 1 else np.nan,
            "label": f"{slots[0]}-{slots[-1]}", "inner_days": len(g),
        })
    return pd.DataFrame(rows, columns=[*POINT, "slots", "first", "last", "n_slots", "step_min",
                                       "label", "inner_days"])


def programme_of(rec: pd.DataFrame, programmes: pd.DataFrame) -> pd.Series:
    """Le programme (`label`) du point de chaque enregistrement, « ? » sans programme lu.

    Question : sous quel programme chaque enregistrement a-t-il été fait ? Entrée :
    `load_recordings`, `recording_programme`. Sortie : une série alignée sur `rec`.
    Règle : jointure sur (jeu, site, micro). Limite : celle de `recording_programme`.
    """
    labels = programmes.set_index(POINT)["label"].to_dict()
    keys = zip(rec["dataset"], rec["site"], rec["mic_id"], strict=True)
    return pd.Series([labels.get(k, "?") for k in keys], index=rec.index, name="programme")


def programme_counts(rec: pd.DataFrame, programmes: pd.DataFrame) -> pd.DataFrame:
    """Enregistrements selon le programme de leur micro.

    Question : combien d'enregistrements viennent de micros programmés 5 h → 19 h 30, combien
    de 6 h → 18 h 30 ? Entrée : `load_recordings` (tous, écartés compris), `recording_programme`.
    Sortie : une ligne par (programme, jeu, site) : `recordings`, `recorders`.
    Règle : reprise de `fig_programmes` (`presentation_4_figures.py`). Limite : un
    enregistrement hors programme compte sous le programme de son micro.
    """
    df = rec.assign(programme=programme_of(rec, programmes))
    return df.groupby(["programme", "dataset", "site"]).agg(
        recordings=("recording_id", "size"), recorders=("mic_id", "nunique")).reset_index()


def off_programme_recordings(rec: pd.DataFrame, programmes: pd.DataFrame) -> pd.DataFrame:
    """Enregistrements faits hors des créneaux du programme de leur micro.

    Question : quels enregistrements ne sont pas sur un créneau du programme (déclenchement
    manuel, test, horloge décalée) ? Entrée : `load_recordings`, `recording_programme`.
    Sortie : les lignes de `rec` concernées, avec `reason` (« créneau hors programme » ou
    « point sans programme »). Règle : le créneau (« HH:MM » local, à la minute) n'est pas
    dans `slots` du point. Limite : un enregistrement qui commence quelques secondes après
    le créneau (« 05:00:12 ») est sur le créneau ; un décalage d'horloge d'une heure exacte
    tombe sur un autre créneau et ne se voit pas ici (`name_start_offsets`).
    """
    slots = programmes.set_index(POINT)["slots"].to_dict()
    keys = list(zip(rec["dataset"], rec["site"], rec["mic_id"], strict=True))
    reason = [
        "point sans programme" if k not in slots else
        ("créneau hors programme" if s not in slots[k] else None)
        for k, s in zip(keys, rec["slot"], strict=True)
    ]
    out = rec.assign(reason=reason)
    return out[out["reason"].notna()]


def mic_days(rec: pd.DataFrame, programmes: pd.DataFrame) -> pd.DataFrame:
    """Un jour-micro : ce qui a été reçu face à ce que le programme prévoyait.

    Question : chaque jour, chaque micro a-t-il fait tout son programme ? Entrée :
    `load_recordings`, `recording_programme`. Sortie : une ligne par (jeu, site, micro, jour) :
    `n` (enregistrements de durée normale reçus), `slots`, `edge` (premier ou dernier jour du
    point), `expected` (créneaux du programme), `completeness` (= n / expected), `n_missing`
    (créneaux du programme absents ce jour-là).
    Règle : campagne et durée normale. Reprise de `jours_micros()` (`presentation_4_figures.py`)
    et du § 4 du notebook 05. Limite : un jour sans aucun enregistrement n'a pas de ligne
    (`empty_days`) ; un point sans programme a `expected` vide.
    """
    ok = rec[rec["campaign"] & rec["normal"]]
    days = ok.groupby([*POINT, "day"]).agg(
        n=("recording_id", "size"), slots=("slot", frozenset)).reset_index()
    edge = days.groupby(POINT)["day"].transform
    days["edge"] = (days["day"] == edge("min")) | (days["day"] == edge("max"))
    prog = programmes.set_index(POINT)["slots"].to_dict()
    keys = list(zip(days["dataset"], days["site"], days["mic_id"], strict=True))
    days["expected"] = [len(prog[k]) if k in prog else np.nan for k in keys]
    days["completeness"] = days["n"] / days["expected"]
    days["n_missing"] = [
        len(set(prog[k]) - s) if k in prog else np.nan
        for k, s in zip(keys, days["slots"], strict=True)
    ]
    return days


def completeness_summary(days: pd.DataFrame) -> dict[str, Any]:
    """La complétude du programme, en chiffres.

    Question : sur quelle part des jours entre pose et relève le programme est-il tenu ?
    Entrée : `mic_days`. Sortie : dict `mic_days` (jours intérieurs), `complete` (complétude
    ≥ 1), `share_complete`, `partial` (strictement entre 0 et 1), et `by_dataset`.
    Règle : jours intérieurs seulement (`edge` faux). Limite : un jour sans aucun enregistrement
    n'est pas un jour-micro, il n'entre pas dans la part (`empty_days`) ; la complétude est
    relative au programme lu, pas à un programme réglé connu.
    """

    def summary(d: pd.DataFrame) -> dict[str, Any]:
        complete = int((d["completeness"] >= 1).sum())
        return {
            "mic_days": int(len(d)),
            "complete": complete,
            "share_complete": complete / len(d) if len(d) else np.nan,
            "partial": int(((d["completeness"] < 1) & (d["completeness"] > 0)).sum()),
        }

    inner = days[~days["edge"]]
    out = summary(inner)
    out["by_dataset"] = {ds: summary(g) for ds, g in inner.groupby("dataset")}
    return out


def _runs(dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Suites de jours consécutifs : (début, fin, longueur)."""
    if len(dates) == 0:
        return []
    s = dates.sort_values().to_series()
    run = (s.diff() != pd.Timedelta(days=1)).cumsum()
    return [(g.min(), g.max(), len(g)) for _, g in s.groupby(run.to_numpy())]


def empty_days(
    rec: pd.DataFrame,
    level: str = "mic",
    stations: dict[str, dict[str, list[str]]] | None = None,
) -> pd.DataFrame:
    """Les jours sans aucun enregistrement entre la pose et la relève : les trous.

    Question : où le suivi s'est-il interrompu ? Entrée : `load_recordings` ; `level` = « mic »
    (un point = jeu, site, micro) ou « station » (2023 : la station et ses micros successifs,
    d'après `stations` = `plan.stations_2023`). Sortie : une ligne par point qui a au moins un
    jour vide : `empty_days`, `followed_days` (du premier au dernier jour), `first_empty`,
    `last_empty`, `longest_start`, `longest_end`, `longest_days`, `empty_dates`.
    Règle : enregistrements de la campagne et de durée normale, comme le § 4 du notebook 05 et
    `fig_completude`. Limite : au niveau du micro, le trou entre deux micros d'une même station
    ne se voit pas (le premier s'arrête, le second commence) ; au niveau de la station, si.
    """
    ok = rec[rec["campaign"] & rec["normal"]]
    if level == "station":
        if stations is None:
            raise ValueError("level='station' demande `stations` (plan.stations_2023)")
        mapping = {(site, mic): f"{site}/{st}" for site, sts in stations.items()
                   for st, mics in sts.items() for mic in mics}
        ok = ok[ok["dataset"] == "2023"]
        ok = ok.assign(station=[mapping.get((a, b), f"{a}/?")
                                for a, b in zip(ok["site"], ok["mic_id"], strict=True)])
        keys = ["dataset", "station"]
    elif level == "mic":
        keys = POINT
    else:
        raise ValueError(f"level inconnu : {level}")
    rows = []
    for key, g in ok.groupby(keys):
        span = pd.date_range(g["day"].min(), g["day"].max(), freq="D")
        missing = span.difference(pd.DatetimeIndex(g["day"].unique()))
        if len(missing) == 0:
            continue
        runs = _runs(missing)
        start, end, length = max(runs, key=lambda r: r[2])
        rows.append({
            **dict(zip(keys, key, strict=True)),
            "empty_days": len(missing), "followed_days": len(span),
            "first_empty": missing.min(), "last_empty": missing.max(),
            "longest_start": start, "longest_end": end, "longest_days": length,
            "empty_dates": list(missing),
        })
    cols = [*keys, "empty_days", "followed_days", "first_empty", "last_empty", "longest_start",
            "longest_end", "longest_days", "empty_dates"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("empty_days", ascending=False).reset_index(drop=True)


# =============================================================================================
# 3. Noms de fichiers et arborescence
# =============================================================================================


def filename_conformity(rec: pd.DataFrame) -> pd.DataFrame:
    """Le nom de chaque fichier suit-il la convention Song Meter ?

    Question : quels fichiers ont un nom hors du motif `<série>_<AAAAMMJJ>_<HHMMSS>`, un préfixe
    qui n'est pas leur micro, une extension inattendue ? Entrée : `load_recordings`. Sortie :
    une ligne par enregistrement : `stem`, `extension`, `name_ok` (motif et date valide),
    `prefix`, `prefix_is_mic`, `suffix` (texte après l'heure, vide d'ordinaire),
    `extension_ok` (`.wav` ou `.flac` en minuscules).
    Règle : motif `ingest.SONGMETER_NAME`, date lue par `datetime.strptime` comme
    `ingest.parse_songmeter_name`. Limite : un nom conforme peut porter une fausse heure
    (horloge) ; le préfixe d'un enregistreur renommé (« TEST-BLANCI ») n'est pas une erreur.
    """
    paths = rec["path"].astype(str)
    stem = paths.str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    ext = paths.str.extract(r"(\.[^./]+)$")[0]
    m = stem.str.extract(SONGMETER_NAME.pattern)
    stamp = pd.to_datetime(m["date"] + m["time"], format="%Y%m%d%H%M%S", errors="coerce")
    return pd.DataFrame({
        "recording_id": rec["recording_id"], "path": paths, "stem": stem, "extension": ext,
        "name_ok": stamp.notna(), "prefix": m["prefix"],
        "prefix_is_mic": m["prefix"] == rec["mic_id"],
        "suffix": stem.str.extract(r"_\d{8}_\d{6}(_.*)?$")[0].fillna(""),
        "extension_ok": ext.isin(AUDIO_SUFFIXES),
    })


def homonyms(rec: pd.DataFrame) -> pd.DataFrame:
    """Les noms de fichiers portés par plusieurs lignes de la base.

    Question : un même nom (sans l'extension, sans la casse) est-il rangé deux fois ?
    Entrée : `load_recordings`. Sortie : une ligne par nom répété : `stem`, `n`, `sites`,
    `paths`, `recording_ids`. Règle : comparaison en minuscules, comme l'index des noms de
    `ingest`. Limite : l'inventaire écarte déjà les doublons de nom à l'entrée ; un homonyme
    en base veut dire deux contenus différents admis sous deux identifiants, ou un jeu ancien.
    """
    stem = rec["path"].astype(str).str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    df = rec.assign(stem=stem.str.lower())
    rep = df[df["stem"].duplicated(keep=False)]
    out = rep.groupby("stem").agg(
        n=("recording_id", "size"),
        sites=("site", lambda s: ", ".join(sorted(s.dropna().astype(str).unique()))),
        paths=("path", list), recording_ids=("recording_id", list)).reset_index()
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def mic_folders(rec: pd.DataFrame) -> pd.DataFrame:
    """Les dossiers sous lesquels chaque micro est rangé.

    Question : le même enregistreur est-il rangé sous plusieurs noms de dossier, et le dossier
    porte-t-il son numéro de série ? Entrée : `load_recordings`. Sortie : une ligne par point :
    `folders` (noms distincts du dossier du micro), `n_folders`, `mic_in_folder` (tous les
    dossiers contiennent le numéro de série).
    Règle : le dossier du micro est le parent du fichier, ou son grand-parent si le parent est
    `Data`. Le micro, lui, vient du GUANO (`ingest.describe_recording`). Limite : en 2023 les
    dossiers s'appellent « SM4 C », « SM E_SMA14636 » : un dossier sans numéro n'est pas une
    erreur, seulement un nom qui ne vérifie rien.
    """
    parts = rec["path"].astype(str).str.split("/")
    folder = [p[-3] if len(p) >= 3 and p[-2] == "Data" else (p[-2] if len(p) >= 2 else "")
              for p in parts]
    df = rec.assign(folder=folder)
    out = df.groupby(POINT).agg(folders=("folder", lambda s: sorted(set(s)))).reset_index()
    out["n_folders"] = out["folders"].map(len)
    out["mic_in_folder"] = [all(str(m) in f for f in fs)
                            for m, fs in zip(out["mic_id"], out["folders"], strict=True)]
    return out


# =============================================================================================
# 4. En-têtes (GUANO)
# =============================================================================================

_TZ = re.compile(r"([+-]\d{1,2}:\d{2})$")


def parse_guano_fields(guano: dict[str, str]) -> dict[str, Any]:
    """Les champs d'un en-tête GUANO, typés.

    Question : que dit l'en-tête d'un fichier ? Entrée : le dict de `ingest.read_guano`.
    Sortie : dict `make`, `model`, `serial`, `firmware`, `prefix`, `original_filename`,
    `timestamp` (texte), `tz` (fuseau écrit, « -3:00 »), `tz_hours`, `header_start`
    (horodatage avec fuseau, `ingest.parse_guano_timestamp`), `length_s`, `samplerate`,
    `gain_0`, `gain_1`, `rate_0`, `rate_1` (réglages audio par piste), `lat`, `lon`,
    `has_fix` (position non nulle), `temperature_c`, `n_fields`.
    Règle : « Loc Position = 0.00000 0.00000 » veut dire pas de position (vu sur un test de CDR
    le 18/11/2026) ; les gains sont dans `WA|Song Meter|Audio settings`, une liste JSON, une
    entrée par piste. Les Song Meter Mini lus le 18/11/2026 n'écrivent aucun champ de batterie.
    Limite : un champ absent vaut `None` ; un FLAC n'a pas d'en-tête GUANO (dict vide).
    """
    try:
        settings = json.loads(guano.get("WA|Song Meter|Audio settings", "[]"))
    except ValueError:
        settings = []
    settings = settings if isinstance(settings, list) else []

    def num(key: str) -> float | None:
        try:
            return float(guano[key])
        except (KeyError, ValueError):
            return None

    lat = lon = None
    pos = guano.get("Loc Position", "").split()
    if len(pos) == 2:
        try:
            lat, lon = float(pos[0]), float(pos[1])
        except ValueError:
            lat = lon = None
    stamp = guano.get("Timestamp")
    start = parse_guano_timestamp(stamp) if stamp else None
    tz = _TZ.search(stamp.strip()) if stamp else None
    offset = start.utcoffset() if start is not None and start.tzinfo is not None else None
    rate = num("Samplerate")
    return {
        "make": guano.get("Make"), "model": guano.get("Model"), "serial": guano.get("Serial"),
        "firmware": guano.get("Firmware Version"), "prefix": guano.get("WA|Song Meter|Prefix"),
        "original_filename": guano.get("Original Filename"), "timestamp": stamp,
        "tz": tz.group(1) if tz else None,
        "tz_hours": offset.total_seconds() / 3600 if offset is not None else None,
        "header_start": start, "length_s": num("Length"),
        "samplerate": int(rate) if rate is not None else None,
        "gain_0": settings[0].get("gain") if len(settings) > 0 else None,
        "gain_1": settings[1].get("gain") if len(settings) > 1 else None,
        "rate_0": settings[0].get("rate") if len(settings) > 0 else None,
        "rate_1": settings[1].get("rate") if len(settings) > 1 else None,
        "lat": lat, "lon": lon,
        "has_fix": lat is not None and lon is not None and not (lat == 0 and lon == 0),
        "temperature_c": num("Temperature Int"), "n_fields": len(guano),
    }


def sample_recordings(
    rec: pd.DataFrame,
    per: Sequence[str] = (*POINT, "day"),
    hour: float = 10.0,
    n: int = 1,
    channels: int | None = None,
) -> pd.DataFrame:
    """Un échantillon explicite d'enregistrements à lire sur le disque.

    Question : quels fichiers lire pour connaître les en-têtes sans lire les 96 000 ?
    Entrée : `load_recordings`, groupes (`per`, par défaut un par micro et par jour), heure
    locale visée, nombre par groupe, nombre de pistes exigé. Sortie : les lignes tirées.
    Règle : parmi les enregistrements de durée normale, les `n` plus proches de `hour` ; même
    règle que le motif « jour » de `scripts/controle_qualite/echantillon_pistes.py` (10 h,
    stéréo), qui a produit `paths.reports/pistes/echantillon_pistes.csv`. Déterministe : à
    égalité, l'ordre de `start_utc` départage.
    Limite : un réglage changé en cours de journée n'est vu que le lendemain ; un enregistrement
    écarté peut être tiré (l'en-tête ne dépend pas de la qualité du son).
    """
    ok = rec[rec["normal"]]
    if channels is not None:
        ok = ok[ok["channels"] == channels]
    gap = (ok["local"].dt.hour + ok["local"].dt.minute / 60 - hour).abs()
    ok = ok.assign(_gap=gap).sort_values(["_gap", "start_utc"], kind="stable")
    out = ok.groupby(list(per)).head(n).drop(columns="_gap")
    return out.sort_values([*per]).reset_index(drop=True)


def read_headers(
    sample: pd.DataFrame,
    raw: Path | None,
    renamed: dict[str, str] | None = None,
    extra_roots: Sequence[Path] = (),
) -> pd.DataFrame:
    """Les en-têtes GUANO d'un échantillon de fichiers, lus sur le disque.

    Question : que disent les en-têtes des fichiers tirés ? Entrée : un échantillon
    (`sample_recordings`, ou toute partie de `load_recordings` avec `path`), `paths.raw`,
    `audio.renamed`, racines supplémentaires. Sortie : les colonnes de l'échantillon, plus
    `found` et les champs de `parse_guano_fields`.
    Règle : le fichier est cherché par `core.locate.locate` (racine, dossiers renommés à la
    copie, disques montés) ; seul le bloc `guan` est lu, pas l'audio.
    Limite : exige le disque. Si aucun fichier de l'échantillon n'est trouvé, lève
    `FileNotFoundError` (disque débranché) ; un fichier absent seul a `found` faux. `locate`
    retient les racines où il a déjà trouvé un fichier dans la session : il cherche aussi là.
    """
    rows = []
    for rec in sample.to_dict("records"):
        path = locate(rec["path"], raw, list(extra_roots), renamed)
        fields = parse_guano_fields(read_guano(path)) if path is not None else {}
        rows.append({**rec, "found": path is not None, **fields})
    out = pd.DataFrame(rows)
    if len(out) and not out["found"].any():
        raise FileNotFoundError(
            f"aucun des {len(out)} fichiers de l'échantillon n'est sur un disque branché "
            f"(racine {raw})")
    return out


def header_field_summary(
    headers: pd.DataFrame, fields: Sequence[str] | None = None
) -> pd.DataFrame:
    """Les champs d'en-tête : présents ou absents, combien de valeurs.

    Question : quels champs manquent, lesquels varient ? Entrée : `read_headers` (fichiers
    trouvés), ou toute table au même nom de colonnes (l'échantillon `echantillon_pistes.csv`
    a `model`, `firmware`, `gain_0`, `gain_1`, `lat`, `lon`, `temperature_c`). Sortie : une
    ligne par champ : `present`, `share_present`, `distinct`, `most_common`, `values` (les cinq
    plus fréquentes, avec effectifs). Règle : compte sur les lignes trouvées (`found`).
    Limite : ne vaut que pour l'échantillon lu.
    """
    df = headers[headers["found"]] if "found" in headers else headers
    fields = list(fields) if fields else [c for c in (
        "make", "model", "serial", "firmware", "tz", "samplerate", "gain_0", "gain_1",
        "rate_0", "rate_1", "has_fix", "temperature_c", "length_s") if c in df]
    rows = []
    for f in fields:
        s = df[f]
        counts = s.dropna().astype(str).value_counts()
        rows.append({"field": f, "present": int(s.notna().sum()),
                     "share_present": float(s.notna().mean()) if len(s) else np.nan,
                     "distinct": int(len(counts)),
                     "most_common": counts.index[0] if len(counts) else None,
                     "values": counts.head(5).to_dict()})
    return pd.DataFrame(rows)


def header_outliers(
    headers: pd.DataFrame, field: str, by: Sequence[str] = ("dataset", "model")
) -> pd.DataFrame:
    """Les micros dont un champ d'en-tête sort du lot.

    Question : quel micro a un firmware, une fréquence, un fuseau différent des autres du même
    groupe ? Entrée : `read_headers` (ou l'échantillon des pistes), le champ, les colonnes du
    groupe. Sortie : une ligne par (point, valeur) différente de la valeur la plus fréquente
    du groupe : `value`, `group_mode`, `files`. Règle : mode du groupe, sans seuil.
    Limite : un groupe partagé à parts égales désigne arbitrairement un mode.
    """
    df = headers[headers["found"]] if "found" in headers else headers
    df = df.dropna(subset=[field])
    by = [b for b in by if b in df]
    mode = df.groupby(by)[field].agg(lambda s: s.astype(str).value_counts().index[0])
    df = df.assign(value=df[field].astype(str), group_mode=[
        mode.loc[tuple(k) if len(by) > 1 else k[0]]
        for k in df[by].itertuples(index=False, name=None)])
    odd = df[df["value"] != df["group_mode"]]
    return odd.groupby([*POINT, "value", "group_mode"]).size().rename("files").reset_index()


def gain_settings(headers: pd.DataFrame) -> pd.DataFrame:
    """Le gain réglé de chaque piste, par point.

    Question : quel micro a la piste 0 au gain faible, lequel au gain fort, lequel a changé de
    réglage ? Entrée : `read_headers`, ou l'échantillon `echantillon_pistes.csv` (mêmes noms de
    colonnes : `gain_0`, `gain_1`, `start_utc`, `model`, `firmware`). Sortie : une ligne par
    point : `config` (premier réglage lu, « 6 / 18 dB », « 6 dB, mono »), `sequence` (réglages
    successifs, « … puis … »), `model`, `firmware`, `files`.
    Règle : reprise du § 3 du notebook 05 et de `fig_gains` (`presentation_4_figures.py`),
    réglage par point = le premier dans l'ordre du temps. Lecture de l'en-tête, pas mesure.
    Limite : un réglage changé entre deux fichiers lus n'est daté qu'au jour près.
    """
    df = headers[headers["found"]] if "found" in headers else headers
    df = df.dropna(subset=["gain_0"]).sort_values("start_utc")
    config = [f"{int(a)} / {int(b)} dB" if pd.notna(b) else f"{int(a)} dB, mono"
              for a, b in zip(df["gain_0"], df["gain_1"], strict=True)]
    df = df.assign(config=config, dataset=df["dataset"].astype(str))
    agg = {"config": ("config", "first"),
           "sequence": ("config", lambda c: " puis ".join(dict.fromkeys(c))),
           "files": ("config", "size")}
    for c in ("model", "firmware"):
        if c in df:
            agg[c] = (c, "first")
    return df.groupby(POINT).agg(**agg).reset_index()


# =============================================================================================
# 5. Horloge
# =============================================================================================


def _name_stamp(paths: pd.Series) -> pd.Series:
    stamp = paths.astype(str).str.extract(r"_(\d{8}_\d{6})(?:_[^/]*)?\.\w+$")[0]
    return pd.to_datetime(stamp, format="%Y%m%d_%H%M%S", errors="coerce")


def name_start_offsets(rec: pd.DataFrame) -> pd.DataFrame:
    """L'heure du nom de fichier contre l'heure de début rangée en base, sans lire le disque.

    Question : pour quels enregistrements l'heure rangée (lue dans le GUANO à l'inventaire) ne
    tombe-t-elle pas sur l'heure du nom ? Entrée : `load_recordings`. Sortie : une ligne par
    enregistrement : `name_local` (heure du nom), `local` (heure rangée, en heure locale),
    `offset_min` (rangée − nom), `clock_corrected`.
    Règle : même comparaison que `clock_off` (`qc.metadata_flags`), signée. Les enregistrements
    `clock_corrected` ont pour heure celle du nom : leur écart est nul par construction, et
    l'heure de leur en-tête n'est plus en base (`clock_offsets` la relit sur le disque).
    Limite : un fichier sans horodatage dans le nom a `offset_min` vide ; une horloge fausse
    qui écrit la même fausse heure dans le nom et l'en-tête ne se voit pas.
    """
    named = _name_stamp(rec["path"])
    return pd.DataFrame({
        **{c: rec[c] for c in ("recording_id", *POINT, "path")},
        "name_local": named, "local": rec["local"],
        "offset_min": (rec["local"] - named).dt.total_seconds() / 60,
        "clock_corrected": rec["fl"].map(lambda f: bool(f.get("clock_corrected"))),
    })


def clock_offsets(headers: pd.DataFrame, utc_offset_h: float) -> pd.DataFrame:
    """L'heure du nom contre l'heure de l'en-tête GUANO, fichier par fichier, lue sur le disque.

    Question : de combien l'en-tête d'un fichier s'écarte-t-il de son nom, et sous quel
    fuseau ? Entrée : `read_headers`, `recorder.filename_utc_offset_h`. Sortie : une ligne par
    fichier trouvé : `name_local`, `header_local` (heure de l'en-tête ramenée à l'heure locale
    de la Guyane), `offset_min` (en-tête − nom), `tz`.
    Règle : le nom est en heure locale de l'enregistreur ; l'en-tête porte son fuseau.
    Limite : un en-tête sans fuseau est lu à `utc_offset_h` ; ne voit que les fichiers lus.
    """
    df = headers[headers["found"]].copy() if "found" in headers else headers.copy()
    named = _name_stamp(df["path"])
    tz = pd.Timedelta(hours=utc_offset_h)
    header = [
        (pd.Timestamp(h).tz_convert("UTC").tz_localize(None) + tz) if h is not None
        and pd.notna(h) and pd.Timestamp(h).tzinfo is not None
        else (pd.Timestamp(h) if h is not None and pd.notna(h) else pd.NaT)
        for h in df["header_start"]
    ]
    df["name_local"] = named.to_numpy()
    df["header_local"] = pd.to_datetime(header)
    df["offset_min"] = (df["header_local"] - df["name_local"]).dt.total_seconds() / 60
    return df[["recording_id", *POINT, "path", "name_local", "header_local", "offset_min", "tz"]]


def clock_summary(offsets: pd.DataFrame, tolerance_min: float) -> pd.DataFrame:
    """Décalages d'horloge par micro et par période : constants, en dérive, changement de fuseau.

    Question : quand un micro est décalé, l'est-il d'un bloc ou en dérive ? Entrée :
    `clock_offsets` ou `name_start_offsets` (colonnes `offset_min`, `name_local`), la tolérance
    (`qc.clock_tolerance_min`). Sortie : une ligne par suite de fichiers consécutifs d'un même
    point au même décalage (arrondi à la minute) : `offset_min`, `tz` (si connu), `first`,
    `last`, `files`, `off` (|décalage| > tolérance), et `drift_min_per_day` (pente du décalage
    sur toute la série du point, estimateur de Theil-Sen, insensible à quelques fichiers de test
    à l'heure UTC ; vide sous trois fichiers).
    Règle : une suite change quand le décalage arrondi change. Limite : une dérive de moins
    d'une minute par fichier se lit dans la pente, pas dans les suites ; un saut d'une heure
    sur une partie de la série (Molokoï) donne une pente sans signification : lire les suites.
    """
    df = offsets.dropna(subset=["offset_min", "name_local"]).sort_values([*POINT, "name_local"])
    rows = []
    for key, g in df.groupby(POINT):
        rounded = g["offset_min"].round()
        tz = g["tz"] if "tz" in g else pd.Series([None] * len(g), index=g.index)
        block = ((rounded != rounded.shift()) | (tz.astype(str) != tz.astype(str).shift())).cumsum()
        days = (g["name_local"] - g["name_local"].min()).dt.total_seconds() / 86400
        drift = float(theilslopes(g["offset_min"], days)[0]) \
            if len(g) >= 3 and days.nunique() > 1 else np.nan
        for _, b in g.groupby(block.to_numpy()):
            off = float(b["offset_min"].round().iloc[0])
            rows.append({**dict(zip(POINT, key, strict=True)), "offset_min": off,
                         "tz": tz.loc[b.index[0]], "first": b["name_local"].min(),
                         "last": b["name_local"].max(), "files": len(b),
                         "off": abs(off) > tolerance_min, "drift_min_per_day": drift})
    return pd.DataFrame(rows, columns=[*POINT, "offset_min", "tz", "first", "last", "files",
                                       "off", "drift_min_per_day"])


def clock_corrected_counts(rec: pd.DataFrame) -> pd.DataFrame:
    """Les enregistrements dont l'heure a été relue dans le nom (`clock_corrected`).

    Question : combien d'enregistrements la correction d'horloge touche-t-elle, sur quelle
    période ? Entrée : `load_recordings`. Sortie : une ligne par point : `recordings`, `first`,
    `last` (heure locale). Règle : drapeau `clock_corrected`, posé par
    `ingest.apply_clock_corrections` d'après `recorder.clock_corrections`. Limite : compte ce
    qui est rangé ; à jour seulement après `blanci flag`.
    """
    df = rec[rec["fl"].map(lambda f: bool(f.get("clock_corrected")))]
    return df.groupby(POINT).agg(recordings=("recording_id", "size"), first=("local", "min"),
                                 last=("local", "max")).reset_index()


def clock_lag_correlation(
    rec: pd.DataFrame,
    target: str,
    reference: str,
    site: str | None = None,
    lags_h: Iterable[float] = tuple(np.arange(-3, 3.25, 0.5)),
    header_shift_h: float = 1.0,
) -> pd.DataFrame:
    """Le son d'un micro voisin dit laquelle des deux heures croire.

    Question : à quel décalage le niveau sonore d'un micro à l'horloge douteuse se superpose-t-il
    à celui d'un voisin sain ? Entrée : `load_recordings` (RMS de la piste 0 rangé par
    `blanci qc` dans `qc_flags.indices`), le micro douteux, le micro de référence, le site (les
    médianes horaires se calculent sur ce site seulement), les décalages essayés, et
    `header_shift_h` : l'avance de l'en-tête sur le nom pour les enregistrements
    `clock_corrected` (leur heure d'en-tête n'est plus en base ; + 1 h pour SMA14636, d'après
    `recorder.clock_corrections` et l'en-tête « -4:00 » lu le 18/11/2026).
    Sortie : une ligne par décalage : `lag_h`, `r` (Pearson), `n` (créneaux communs).
    Règle : reprise de `molokoi_niveaux()` (`presentation_4_figures.py`) et du § 6 du notebook
    05 : résidu = RMS − médiane du même micro à la même heure locale (le cycle jour-nuit est
    retiré) ; enregistrements `clock_corrected` du micro douteux placés à l'heure de leur
    en-tête, décalés de `lag_h`, joints à la référence à la seconde près.
    Limite : une seule paire, une seule période ; une horloge fausse et cohérente partout ne se
    voit pas ; Pearson est sensible aux événements forts (orage) qui font l'essentiel du signal.
    """
    df = rec if site is None else rec[rec["site"] == site]
    df = df.assign(rms=df["fl"].map(lambda f: f.get("indices", {}).get("rms_dbfs")))
    df = df.dropna(subset=["rms"])
    df = df.assign(resid=df["rms"] - df.groupby(["mic_id", df["local"].dt.hour])["rms"]
                   .transform("median"))
    corrected = df["fl"].map(lambda f: bool(f.get("clock_corrected")))
    a = df[corrected & (df["mic_id"] == target)]
    a = a.set_index(a["t"] + pd.Timedelta(hours=header_shift_h))["resid"]
    b = df[df["mic_id"] == reference].set_index("t")["resid"]
    rows = []
    for lag in lags_h:
        shifted = a.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=float(lag))
        j = pd.concat([shifted.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
        rows.append({"lag_h": float(lag), "r": float(j["a"].corr(j["b"])) if len(j) > 1
                     else np.nan, "n": len(j)})
    return pd.DataFrame(rows)


# =============================================================================================
# 6. Position
# =============================================================================================


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in
                              (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        (lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(h))


def mic_positions(headers: pd.DataFrame) -> pd.DataFrame:
    """La position GPS de chaque point, et sa stabilité.

    Question : où est chaque micro, combien de fichiers sans position, a-t-il bougé ? Entrée :
    `read_headers`, ou l'échantillon des pistes (colonnes `lat`, `lon`). Sortie : une ligne par
    point : `files`, `no_fix` (position absente ou 0, 0), `lat`, `lon` (médianes des positions
    valides), `max_move_m` (plus grande distance d'un fichier à la médiane).
    Règle : distance de haversine (sphère de 6 371 km). Limite : la précision du GPS d'un Song
    Meter Mini n'est pas connue ici ; quelques dizaines de mètres de dispersion ne disent pas
    qu'il a bougé.
    """
    df = headers[headers["found"]] if "found" in headers else headers
    valid = df["lat"].notna() & df["lon"].notna() & ~((df["lat"] == 0) & (df["lon"] == 0))
    df = df.assign(valid=valid)
    rows = []
    for key, g in df.groupby(POINT):
        v = g[g["valid"]]
        lat, lon = (v["lat"].median(), v["lon"].median()) if len(v) else (np.nan, np.nan)
        move = float(_haversine_m(v["lat"], v["lon"], lat, lon).max()) if len(v) else np.nan
        rows.append({**dict(zip(POINT, key, strict=True)), "files": len(g),
                     "no_fix": int((~g["valid"]).sum()), "lat": lat, "lon": lon,
                     "max_move_m": move})
    return pd.DataFrame(rows, columns=[*POINT, "files", "no_fix", "lat", "lon", "max_move_m"])


def position_outliers(positions: pd.DataFrame, max_km: float) -> pd.DataFrame:
    """Les points loin des autres micros de leur site : le site annoncé est-il le bon ?

    Question : un micro rangé sous un site est-il vraiment à ce site ? Entrée : `mic_positions`,
    `max_km`. Sortie : une ligne par point positionné : `site_lat`, `site_lon` (médiane des
    autres points du même jeu et site), `distance_km`, `outlier` (> `max_km`).
    Règle : la configuration n'a pas de coordonnées de site ; le centre du site est celui des
    autres micros. `max_km` n'est dans aucun fichier de configuration : à choisir et à dire.
    Limite : un site d'un seul micro n'a pas de référence ; si la moitié des micros d'un site
    sont mal rangés, le centre est faux.
    """
    pos = positions.dropna(subset=["lat", "lon"])
    rows = []
    for _, g in pos.groupby(["dataset", "site"]):
        for i, r in g.iterrows():
            others = g.drop(index=i)
            if others.empty:
                lat = lon = dist = np.nan
            else:
                lat, lon = others["lat"].median(), others["lon"].median()
                dist = float(_haversine_m(r["lat"], r["lon"], lat, lon)) / 1000
            rows.append({**{c: r[c] for c in POINT}, "site_lat": lat, "site_lon": lon,
                         "distance_km": dist, "outlier": bool(dist > max_km)})
    return pd.DataFrame(rows, columns=[*POINT, "site_lat", "site_lon", "distance_km", "outlier"])


def gps_places(headers: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Les lieux où les enregistreurs ont pris leur position : terrain ou lieu de réglage ?

    Question : la position de l'en-tête est-elle celle de la pose ? Entrée : `read_headers`, ou
    l'échantillon des pistes (colonnes `lat`, `lon`). Sortie : une ligne par point et par lieu :
    `place_lat`, `place_lon` (lieu arrondi), `lat`, `lon` (médianes des fichiers du lieu),
    `files`, `first`, `last` (`start_utc`), `sites` (nombre de sites qui ont un point à ce
    lieu), `setup` (lieu de réglage).
    Règle : longitude forcée négative (les 2MA l'écrivent sans signe) ; positions arrondies à
    `decimals` décimales (3 ≈ 110 m) ; un lieu où des points de deux sites ou plus ont pris leur
    position ne peut pas être une pose (les sites sont à des dizaines de km) : c'est un lieu de
    réglage (bureau, maison de réserve), où l'enregistreur a gardé la dernière position lue.
    Limite : un lieu propre à un seul site n'est pas pour autant une pose prouvée ; deux poses à
    moins de 110 m l'une de l'autre et à des sites différents sont impossibles ici, mais un lieu
    de réglage utilisé par un seul site passe pour du terrain.
    """
    df = headers[headers["found"]] if "found" in headers else headers
    df = df[df["lat"].notna() & df["lon"].notna() & ~((df["lat"] == 0) & (df["lon"] == 0))]
    df = df.assign(lon=-df["lon"].abs(), place_lat=df["lat"].round(decimals),
                   place_lon=(-df["lon"].abs()).round(decimals))
    cols = [*POINT, "place_lat", "place_lon", "lat", "lon", "files", "first", "last", "sites",
            "setup"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    places = df.groupby([*POINT, "place_lat", "place_lon"]).agg(
        lat=("lat", "median"), lon=("lon", "median"), files=("lat", "size"),
        first=("start_utc", "min"), last=("start_utc", "max")).reset_index()
    sites = df.groupby(["place_lat", "place_lon"])["site"].nunique().rename("sites")
    places = places.join(sites, on=["place_lat", "place_lon"])
    places["setup"] = places["sites"] >= 2
    return places[cols]


# =============================================================================================
# 7. Température et fin de série
# =============================================================================================


def temperature_series(headers: pd.DataFrame, by: Sequence[str] = (*POINT, "day")) -> pd.DataFrame:
    """La température interne de l'enregistreur (en-tête), par micro et par jour.

    Question : quelles températures l'enregistreur a-t-il vues ? Entrée : `read_headers` ou
    l'échantillon des pistes (`temperature_c`, et `day` ou `start_utc`). Sortie : une ligne par
    groupe : `files`, `min_c`, `median_c`, `max_c`.
    Règle : `Temperature Int` du GUANO, lu au début de chaque fichier. Limite : c'est la
    température du boîtier (soleil direct compris), pas celle de l'air ; un fichier par jour
    ne montre pas l'amplitude de la journée.
    """
    df = headers[headers["found"]] if "found" in headers else headers
    if "day" not in df and "day" in by:
        df = df.assign(day=pd.to_datetime(df["start_utc"], utc=True).dt.tz_convert(None)
                       .dt.normalize())
    return df.dropna(subset=["temperature_c"]).groupby(list(by)).agg(
        files=("temperature_c", "size"), min_c=("temperature_c", "min"),
        median_c=("temperature_c", "median"), max_c=("temperature_c", "max")).reset_index()


def series_ends(rec: pd.DataFrame, days: pd.DataFrame, cfg_qc: dict[str, Any]) -> pd.DataFrame:
    """La fin de la série de chaque point : arrêt à la relève, ou arrêt avant (batterie, carte) ?

    Question : quels micros se sont arrêtés avant les autres de leur site, sur un dernier jour
    incomplet ou un dernier fichier coupé ? Entrée : `load_recordings`, `mic_days`, `qc` de la
    configuration. Sortie : une ligne par point : `last_local` (dernier enregistrement de la
    campagne), `last_day_completeness`, `last_duration_s`, `cut_at_end` (dernier fichier plus
    court que `expected_duration_s` − `duration_tolerance_s`), `site_last_day` (dernier jour du
    site), `stops_early` (s'arrête plus d'un jour avant le site).
    Règle : sans champ de batterie dans le GUANO (`parse_guano_fields`), l'arrêt en fin de
    batterie ne se lit qu'indirectement : fin anticipée, dernier jour partiel, dernier fichier
    coupé. Limite : un indice, pas une preuve ; une carte pleine ou un retrait anticipé donnent
    la même signature ; la relève d'un site peut s'étaler sur deux jours.
    """
    camp = rec[rec["campaign"]].sort_values("local")
    last = camp.groupby(POINT).tail(1).set_index(POINT)
    short = float(cfg_qc["expected_duration_s"]) - float(cfg_qc["duration_tolerance_s"])
    by_day = days.set_index([*POINT, "day"])["completeness"]
    out = pd.DataFrame({
        "last_local": last["local"],
        "last_duration_s": last["duration_s"],
        "cut_at_end": last["duration_s"] < short,
    }).reset_index()
    out["last_day_completeness"] = [
        by_day.get((d, s, m, ll.normalize()), np.nan)
        for d, s, m, ll in zip(out["dataset"], out["site"], out["mic_id"], out["last_local"],
                               strict=True)]
    site_last = out.groupby(["dataset", "site"])["last_local"].transform("max").dt.normalize()
    out["site_last_day"] = site_last
    out["stops_early"] = (site_last - out["last_local"].dt.normalize()) > pd.Timedelta(days=1)
    return out


# =============================================================================================
# 8. Durées et tailles
# =============================================================================================


def duration_classes(
    rec: pd.DataFrame, cfg_qc: dict[str, Any], test_max_s: float = 30.0
) -> pd.DataFrame:
    """Les fichiers de durée anormale, rangés en tests, enregistrements coupés et longs.

    Question : parmi les fichiers qui ne durent pas 2 min, lesquels sont des tests, lesquels du
    vrai terrain coupé ? Entrée : `load_recordings`, `qc` de la configuration, `test_max_s`.
    Sortie : les lignes de `rec` de durée anormale, avec `duration_class` (« test » sous
    `test_max_s`, « coupé » entre `test_max_s` et la durée normale, « long » au-delà) et
    `on_slot` (début à moins d'une minute d'une minute 0 ou 30 : un créneau du programme).
    Règle : durée anormale = `duration_off` de `qc.metadata_flags` (à `duration_tolerance_s` de
    `expected_duration_s`), recalculée ici. Reprise du § 8 du notebook 05. `test_max_s` = 30 s
    n'est pas dans la configuration : choix de lecture.
    Limite : un test lancé pile sur un créneau passe pour du programme ; la cadence de 30 min
    est supposée (`recording_programme` la lit).
    """
    expected, tol = float(cfg_qc["expected_duration_s"]), float(cfg_qc["duration_tolerance_s"])
    off = rec[(rec["duration_s"] - expected).abs() > tol].copy()
    off["duration_class"] = np.select(
        [off["duration_s"] < test_max_s, off["duration_s"] < expected], ["test", "coupé"], "long")
    minute = (off["local"].dt.minute % 30) + off["local"].dt.second / 60
    off["on_slot"] = (minute < 1) | (minute > 29)
    return off


def expected_size(duration_s: float, sample_rate: int, channels: int, bits: int = 16,
                  header_bytes: int = 44) -> int:
    """La taille d'un WAV PCM pour une durée donnée.

    Question : combien d'octets un fichier de cette durée devrait-il peser ? Entrée : durée,
    fréquence, pistes, bits par échantillon, taille de l'en-tête. Sortie : un entier (octets).
    Règle : en-tête + durée × fréquence × pistes × bits / 8 (2 min, 48 kHz, stéréo, 16 bits :
    23 040 044 octets). Limite : le bloc GUANO et d'éventuels blocs de bourrage s'ajoutent
    (quelques centaines d'octets) ; un FLAC est compressé, la formule ne s'y applique pas.
    """
    return int(header_bytes + round(duration_s * sample_rate) * channels * bits // 8)


def file_size_check(
    sample: pd.DataFrame,
    raw: Path | None,
    renamed: dict[str, str] | None = None,
    bits: int = 16,
    tolerance_bytes: int = 65_536,
) -> pd.DataFrame:
    """Taille réelle contre taille attendue : fichiers vides, tronqués, ou plus gros.

    Question : un fichier pèse-t-il ce que sa durée annonce ? Entrée : un échantillon de
    `load_recordings` (`path`, `duration_s`, `sample_rate`, `channels`), `paths.raw`,
    `audio.renamed`. Sortie : `found`, `size`, `expected`, `diff`, `is_empty`, `size_off`
    (|écart| > `tolerance_bytes`). Règle : `expected_size` ; seule la taille est lue (`stat`),
    pas le contenu. `tolerance_bytes` = 64 Kio (≈ 0,3 s de stéréo 48 kHz), choix de lecture.
    Limite : exige le disque (`FileNotFoundError` si aucun fichier n'est trouvé) ; la durée en
    base vient de l'en-tête libsndfile, donc un fichier tronqué après l'écriture de l'en-tête
    se voit, un en-tête faux qui annonce la durée tronquée, non.
    """
    rows = []
    for r in sample.to_dict("records"):
        path = locate(r["path"], raw, renamed=renamed)
        size = path.stat().st_size if path is not None else np.nan
        exp = expected_size(r["duration_s"], r["sample_rate"], r["channels"], bits) \
            if str(r["path"]).lower().endswith(".wav") else np.nan
        rows.append({"recording_id": r["recording_id"], "path": r["path"],
                     "found": path is not None, "size": size, "expected": exp})
    out = pd.DataFrame(rows)
    if len(out) and not out["found"].any():
        raise FileNotFoundError(f"aucun des {len(out)} fichiers n'est sur un disque branché")
    out["diff"] = out["size"] - out["expected"]
    out["is_empty"] = out["size"] == 0
    out["size_off"] = out["diff"].abs() > tolerance_bytes
    return out


# =============================================================================================
# 9. Doublons et conflits
# =============================================================================================


def _survey(paths: pd.Series) -> pd.Series:
    return paths.astype(str).str.replace("\\", "/").str.split("/").str[:2].str.join("/")


def duplicate_report_summary(duplicates: pd.DataFrame) -> pd.DataFrame:
    """Les doublons écartés à l'inventaire, par relevé.

    Question : combien de fichiers l'inventaire a-t-il écartés comme copies, et de quel relevé
    viennent l'original et la copie ? Entrée : les rapports `paths.reports/ingest_duplicates_*.csv`
    concaténés (colonnes `ecarte`, `conserve`). Sortie : une ligne par (relevé écarté, relevé
    conservé) : `files`. Règle : relevé = deux premiers dossiers du chemin. Reprise du § 5 du
    notebook 05. Limite : le rapport dit ce que l'inventaire a décidé, pas que le contenu
    concorde (`duplicate_verification_summary`) ; les doublons ne sont pas dans la base.
    """
    df = duplicates.assign(survey_dropped=_survey(duplicates["ecarte"]),
                           survey_kept=_survey(duplicates["conserve"]))
    return df.groupby(["survey_dropped", "survey_kept"]).size().rename("files").reset_index()


def duplicate_verification_summary(verification: pd.DataFrame) -> pd.DataFrame:
    """Les verdicts de la vérification des doublons par le contenu.

    Question : les paires de même nom ont-elles le même contenu ? Entrée : le rapport de
    `scripts/controle_qualite/verifier_doublons.py` (`paths.reports/doublons/
    verification_doublons*.csv`, colonnes `verdict`, `raison`). Sortie : une ligne par verdict
    (« identiques », « conflit », « introuvable ») : `pairs`. Règle : la comparaison est
    `ingest.same_recording` (longueur, fréquence, pistes, série et heure GUANO, première seconde
    décodée), lancée par le script. Limite : une paire « introuvable » n'est pas vérifiée
    (disque d'origine absent).
    """
    return verification["verdict"].value_counts().rename("pairs").rename_axis("verdict") \
        .reset_index()


def content_duplicates(rec: pd.DataFrame) -> dict[str, Any]:
    """Les enregistrements de même empreinte SHA-256 sous deux identifiants.

    Question : la base contient-elle deux fois le même contenu sous deux noms ? Entrée :
    `load_recordings`. Sortie : dict `hashed` (lignes avec une empreinte), `share_hashed`,
    `groups` (DataFrame : empreinte, `n`, `recording_ids`, `paths`).
    Règle : empreinte de tout le fichier, calculée à l'inventaire quand elle est demandée.
    Limite : la plupart des lignes n'ont pas d'empreinte (inventaires lancés sans) : le
    résultat ne vaut que pour `share_hashed` de la base ; un WAV et son FLAC n'ont pas la même
    empreinte (`ingest.same_recording` les compare décodés).
    """
    hashed = rec.dropna(subset=["sha256"])
    rep = hashed[hashed["sha256"].duplicated(keep=False)]
    groups = rep.groupby("sha256").agg(n=("recording_id", "size"),
                                       recording_ids=("recording_id", list),
                                       paths=("path", list)).reset_index()
    return {"hashed": int(len(hashed)),
            "share_hashed": len(hashed) / len(rec) if len(rec) else np.nan, "groups": groups}


def ingest_errors_summary(errors: pd.DataFrame) -> pd.DataFrame:
    """Les fichiers que l'inventaire n'a pas pu lire.

    Question : combien de fichiers sont restés hors de la base parce qu'illisibles, et
    pourquoi ? Entrée : les rapports `paths.reports/ingest_errors_*.csv` concaténés (`path`,
    `error`). Sortie : une ligne par (relevé, type d'erreur) : `files`. Règle : le type est le
    texte avant le premier « : » (« LibsndfileError »). Limite : seulement les inventaires dont
    le rapport a été gardé.
    """
    df = errors.assign(survey=_survey(errors["path"]),
                       kind=errors["error"].astype(str).str.split(":").str[0])
    return df.groupby(["survey", "kind"]).size().rename("files").reset_index()


# =============================================================================================
# 10. Pistes
# =============================================================================================


def track_levels(rec: pd.DataFrame) -> pd.DataFrame:
    """Le niveau et l'indice d'aigus de chaque piste, et leur écart, pour tout le corpus.

    Question : de combien la piste 0 est-elle plus forte ou plus faible que la piste 1 ?
    Entrée : `load_recordings`. Sortie : une ligne par enregistrement contrôlé : `rms0`,
    `rms1` (dBFS), `gap_db` (= rms0 − rms1), `hf0`, `hf1`.
    Règle : indices rangés par `blanci qc` dans `qc_flags` (`indices`, `indices_1`) : RMS sur
    les 2 min après retrait de la moyenne, en dBFS (`qc.qc_indices`). Aucune lecture du disque :
    c'est la même mesure que `level_diff_db` de l'échantillon des pistes.
    Limite : `blanci qc` ne lit la piste 1 que là où elle sert (candidats au son étouffé et
    leurs voisins) : `rms1` et `gap_db` n'existent que pour une petite partie du corpus (1 404
    enregistrements le 18/11/2026), choisie par l'indice, donc pas représentative ; le gain
    réglé n'est pas en base (`gain_settings`) ; un fichier mono n'a pas de piste 1.
    """
    ind = rec["fl"].map(lambda f: f.get("indices", {}))
    ind1 = rec["fl"].map(lambda f: f.get("indices_1", {}))
    out = pd.DataFrame({
        **{c: rec[c] for c in ("recording_id", *POINT, "t", "local", "excluded")},
        "rms0": ind.map(lambda d: d.get("rms_dbfs")), "rms1": ind1.map(lambda d: d.get("rms_dbfs")),
        "hf0": ind.map(lambda d: d.get("hf_ratio")), "hf1": ind1.map(lambda d: d.get("hf_ratio")),
    })
    out = out.dropna(subset=["rms0"])
    out["gap_db"] = out["rms0"] - out["rms1"]
    return out


def track_gap_by_point(levels: pd.DataFrame, settings: pd.DataFrame,
                       gap_col: str = "gap_db") -> pd.DataFrame:
    """L'écart mesuré entre pistes face à l'écart réglé, par point.

    Question : quel point a un micro qui n'entend plus comme l'autre ? Entrée : `track_levels`
    (ou l'échantillon des pistes, `gap_col="level_diff_db"`), `gain_settings`. Sortie : une
    ligne par point : `config`, `set_gap_db` (gain 0 − gain 1), `median_gap_db`, `deviation_db`
    (médiane − réglé), `files`.
    Règle : médiane, pour qu'un orage ou un chœur proche d'un micro ne déplace pas l'écart.
    Limite : une dégradation de quelques semaines dans une série d'un an ne déplace pas la
    médiane du point : la série se lit dans le temps (`levels` par jour).
    """
    st = settings.assign(set_gap_db=[
        float(c.split(" / ")[0]) - float(c.split(" / ")[1].split()[0]) if " / " in c else np.nan
        for c in settings["config"]])
    med = levels.dropna(subset=[gap_col]).assign(dataset=lambda d: d["dataset"].astype(str)) \
        .groupby(POINT)[gap_col].agg(median_gap_db="median", files="size").reset_index()
    out = med.merge(st[[*POINT, "config", "set_gap_db"]], on=POINT, how="left")
    out["deviation_db"] = out["median_gap_db"] - out["set_gap_db"]
    return out


def track_flags_summary(rec: pd.DataFrame, cfg_qc: dict[str, Any]) -> pd.DataFrame:
    """Ce que deviennent les pistes sous le seuil du son étouffé.

    Question : combien d'enregistrements ont les deux pistes retenues, une seule, ou seulement
    des candidats isolés, et lesquels sont écartés ? Entrée : `load_recordings`, `qc` de la
    configuration. Sortie : une ligne par (catégorie, écarté) : `recordings`.
    Règle : « retenue » = `muffled_0` / `muffled_1` rangés (piste dans un épisode de la carte
    CUSUM, `qc.apply_audio_flags`) ; « candidat » = `qc.muffled_candidate` sur les
    indices rangés, aux seuils actuels (`silent_dbfs`, `muffled_hf_ratio`). Catégories :
    deux pistes retenues ; piste 0 retenue seule ; piste 1 retenue seule ; candidat isolé sur
    la piste 0 (aucune piste retenue) ; candidat isolé sur la piste 1 seule ; aucun candidat ;
    sans indices (contrôle audio jamais fait).
    Limite : la piste 1 d'un fichier mono est sa piste 0 ; « écarté » mêle le calcul et les
    confirmations à l'écoute.
    """
    th = {"silent_dbfs": float(cfg_qc["silent_dbfs"]),
          "muffled_hf_ratio": float(cfg_qc["muffled_hf_ratio"])}

    def category(f: dict[str, Any]) -> str:
        if "indices" not in f:
            return "sans indices"
        m0, m1 = bool(f.get("muffled_0")), bool(f.get("muffled_1"))
        if m0 and m1:
            return "deux pistes retenues"
        if m0:
            return "piste 0 retenue seule"
        if m1:
            return "piste 1 retenue seule"
        if muffled_candidate(f["indices"], th):
            return "candidat isolé, piste 0"
        if "indices_1" in f and muffled_candidate(f["indices_1"], th):
            return "candidat isolé, piste 1 seule"
        return "aucun candidat"

    cat = [category(f) for f in rec["fl"]]
    return rec.assign(category=cat).groupby(["category", "excluded"]).size() \
        .rename("recordings").reset_index()


# Tiers d'octave en base 2 (IEC 61260-1:2014) : centres exacts 1000 · 2^(k/3), bornes à
# centre · 2^(±1/6) ; noms nominaux 100, 125, 160… 20 000 Hz.
_NOMINAL = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800)


def third_octave_bands(fmin: float = 100.0, fmax: float = 20000.0) -> pd.DataFrame:
    """Les bandes de tiers d'octave entre deux fréquences.

    Question : dans quelles bandes ranger un spectre pour comparer deux pistes ? Entrée : bornes
    nominales (Hz). Sortie : une ligne par bande : `name` (nominal, entier, 100 → 20 000),
    `centre`, `low`, `high` (Hz).
    Règle : IEC 61260-1:2014, base 2 : centres 1000 · 2^(k/3), bornes à centre · 2^(±1/6) ; une
    bande par tiers d'octave, la largeur croît avec la fréquence comme la résolution de
    l'oreille et des encodeurs (mel). 100 Hz – 20 kHz : 24 bandes.
    Limite : avec la densité de Welch de `qc` (2 048 points à 48 kHz, 23,4 Hz par raie), les
    bandes sous 200 Hz ne contiennent qu'une à trois raies.
    """
    k0, k1 = round(3 * np.log2(fmin / 1000)), round(3 * np.log2(fmax / 1000))
    k = np.arange(k0, k1 + 1)
    centre = 1000 * 2.0 ** (k / 3)
    name = [_NOMINAL[(i + 10) % 10] * 10 ** ((i + 10) // 10) for i in k]
    return pd.DataFrame({"name": name, "centre": centre, "low": centre * 2 ** (-1 / 6),
                         "high": centre * 2 ** (1 / 6)})


def band_levels(freqs: np.ndarray, psd: np.ndarray, bands: pd.DataFrame) -> np.ndarray:
    """Le niveau d'un spectre dans chaque bande, en dB (puissance, dBFS² si `psd` est en FS²/Hz).

    Question : combien d'énergie dans chaque tiers d'octave ? Entrée : densité spectrale
    (`qc._welch`), `third_octave_bands`. Sortie : un niveau par bande, 10 log10(Σ densité ×
    pas), NaN pour une bande sans raie (au-delà de Nyquist) ou sans énergie.
    Règle : somme des raies dont la fréquence tombe dans [low, high[ : la puissance de la bande.
    Limite : une raie au bord compte entière dans une seule bande (pas de partage).
    """
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    out = np.full(len(bands), np.nan)
    for i, (lo, hi) in enumerate(zip(bands["low"], bands["high"], strict=True)):
        p = psd[(freqs >= lo) & (freqs < hi)].sum() * df
        if p > 0:
            out[i] = 10 * np.log10(p)
    return out


def track_band_levels(wav: np.ndarray, sr: int, bands: pd.DataFrame) -> dict[str, float]:
    """Le spectre moyen de chaque piste d'un enregistrement, en tiers d'octave.

    Question : comment l'énergie se répartit-elle selon la fréquence sur chaque piste ? Entrée :
    l'audio lu (`(n, pistes)`, flottant en pleine échelle), sa fréquence, `third_octave_bands`.
    Sortie : `L<piste>_<bande>` en dB, pour les pistes 0 et 1.
    Règle : comme `qc.qc_indices` : moyenne retirée, densité de Welch (Hann, 2 048 points,
    recouvrement de moitié) sur les 2 min entières, puis `band_levels`.
    Limite : un spectre moyen sur 2 min : un défaut de quelques secondes s'y dilue.
    """
    from blanci.inputs.qc import _welch

    out: dict[str, float] = {}
    for track in range(min(2, wav.shape[1])):
        x = wav[:, track] - wav[:, track].mean()
        freqs, psd = _welch(x, sr)
        for name, level in zip(bands["name"], band_levels(freqs, psd, bands), strict=True):
            out[f"L{track}_{name}"] = float(level)
    return out


def spectral_gaps(levels: pd.DataFrame, sample: pd.DataFrame, utc_offset_h: float) -> pd.DataFrame:
    """L'écart de spectre entre les deux pistes d'un même enregistrement, corrigé du réglage.

    Question : bande par bande, la piste 0 entend-elle comme la piste 1, au gain près ? Entrée :
    les niveaux par bande (`track_band_levels`, une ligne par `recording_id`), l'échantillon
    des pistes (`echantillon_pistes.csv` : point, `start_utc`, `motif`, `gain_0`, `gain_1`),
    le décalage des noms (`recorder.filename_utc_offset_h`). Sortie : une ligne par
    enregistrement stéréo : point, `day` (local), `motif`, `set_gap_db` (gain 0 − gain 1), et
    `d_<bande>` = L0 − L1 − (gain 0 − gain 1), en dB.
    Règle : les deux micros sont à quelques centimètres et entendent la même scène : à chaque
    fréquence, leur écart ne dépend que du gain réglé. Le paysage sonore (chœur, pluie,
    insectes) s'annule dans la différence ; un micro qui faiblit, s'étouffe ou crépite d'un
    seul côté la fait sortir de 0, et la bande dit à quelle fréquence.
    Limite : deux pistes étouffées ensemble ont d ≈ 0 (la règle du son étouffé reste en
    parallèle) ; une bande où la scène est sous le bruit propre de l'enregistreur compare deux
    bruits de fond (sa forme dépend du gain), pas deux micros ; un bruit tout près d'un seul
    micro (insecte posé sur la bonnette, goutte) fait sortir d'une seule bande un seul jour.
    """
    cols = [c for c in levels if c.startswith("L0_")]
    names = [c[3:] for c in cols]
    keep = [c for c in ("recording_id", *POINT, "start_utc", "motif", "gain_0", "gain_1")
            if c in sample]
    df = sample[keep].merge(levels, on="recording_id", how="inner")
    df = df.dropna(subset=["gain_0", "gain_1", *[f"L1_{n}" for n in names]], how="any")
    set_gap = df["gain_0"].astype(float) - df["gain_1"].astype(float)
    local = pd.to_datetime(df["start_utc"], utc=True).dt.tz_convert(None) \
        + pd.Timedelta(hours=utc_offset_h)
    out = df[[c for c in ("recording_id", *POINT, "motif") if c in df]].assign(
        dataset=df["dataset"].astype(str), day=local.dt.normalize(), set_gap_db=set_gap)
    d = {f"d_{n}": df[f"L0_{n}"] - df[f"L1_{n}"] - set_gap for n in names}
    return pd.concat([out, pd.DataFrame(d, index=df.index)], axis=1).reset_index(drop=True)


def _signed_extreme(values: np.ndarray) -> np.ndarray:
    """Par ligne, la valeur de plus grand module, avec son signe (NaN ignorés)."""
    filled = np.where(np.isnan(values), 0.0, values)
    pick = np.abs(filled).argmax(axis=1)
    out = values[np.arange(len(values)), pick]
    return np.where(np.isnan(values).all(axis=1), np.nan, out)


def spectral_gap_by_point(gaps: pd.DataFrame, fmin: float = 100.0, fmax: float = 12000.0,
                          window_days: int = 14, min_days: int = 7) -> pd.DataFrame:
    """Le pire épisode d'écart de spectre entre pistes, par point.

    Question : quel point a eu, pendant au moins une semaine, un micro qui n'entendait plus
    comme l'autre, et à quelle fréquence ? Entrée : `spectral_gaps`, une ligne par jour (motif
    « jour » de l'échantillon des pistes). Sortie : une ligne par point : `extreme_db` (signé :
    négatif = piste 0 plus faible que réglé), `episode_start`, `episode_end`, `episode_days`,
    `days`, `median_extreme_db` (sur toute la série), `set_gap_db`, et `d_<bande>` = médiane
    de chaque bande sur l'épisode ; trié de `extreme_db` le plus faible au plus fort.
    Règle : chaque jour, l'écart de plus grand module parmi les bandes de `fmin` à `fmax`
    (100 Hz – 12 kHz : sous 6 kHz l'écart d'un point sain est aussi stable à 100 Hz qu'à 1 kHz,
    et un micro peut ne perdre que les graves (SMA14636 : plus de 20 dB sous 1 kHz) ; au-dessus
    de 12 kHz la scène tombe sous le bruit propre de l'enregistreur et les deux pistes ne se
    comparent plus : Kaw D, sourd de 25 dB, y a d ≈ 0). Puis médiane glissante sur
    `window_days` jours (au moins `min_days` jours lus, ou tous ceux du point s'il en a moins) :
    un orage ou un chœur proche d'un seul micro un jour ne la déplace pas, il faut une moitié
    de fenêtre de mauvais jours. L'épisode retenu est la fenêtre de plus grand module.
    Limite : le maximum de 21 bandes bruitées n'est jamais 0 (plus qu'avec 11 bandes de 1 à
    12 kHz : le seuil se resserre) ; une panne de moins d'une semaine passe ; deux micros qui
    faiblissent ensemble ne se voient pas ; un point de moins de `min_days` jours n'a qu'une
    fenêtre, sa médiane entière.
    """
    names = [c for c in gaps if c.startswith("d_")]
    centre = np.array([float(c[2:]) for c in names])
    inside = [c for c, f in zip(names, centre, strict=True) if fmin <= f <= fmax]
    if "motif" in gaps:  # les bascules d'état ajoutées à l'échantillon ne sont pas des jours
        gaps = gaps[gaps["motif"] == "jour"]
    rows = []
    for key, g in gaps.sort_values("day").groupby(POINT, sort=False):
        g = g.drop_duplicates("day", keep="first")
        daily = pd.Series(_signed_extreme(g[inside].to_numpy(dtype=float)),
                          index=pd.DatetimeIndex(g["day"]))
        need = min(min_days, len(daily))
        rolled = daily.rolling(f"{window_days}D", min_periods=need).median()
        if rolled.notna().any():
            end = rolled.abs().idxmax()
        else:
            end = daily.index[-1]
        start = end - pd.Timedelta(days=window_days - 1)
        episode = g[(g["day"] >= start) & (g["day"] <= end)]
        rows.append({**dict(zip(POINT, key, strict=True)),
                     "extreme_db": float(rolled.get(end, np.nan)),
                     "episode_start": episode["day"].min(), "episode_end": episode["day"].max(),
                     "episode_days": len(episode), "days": len(daily),
                     "median_extreme_db": float(daily.median()),
                     "set_gap_db": float(g["set_gap_db"].median()),
                     **episode[names].median().to_dict()})
    return pd.DataFrame(rows).sort_values("extreme_db").reset_index(drop=True)


def spectral_gap_profile(gaps: pd.DataFrame, periods: dict[str, tuple[str, str]],
                         **point: str) -> pd.DataFrame:
    """L'écart de spectre d'un point, bande par bande, sur des périodes nommées.

    Question : à quelle fréquence un micro a-t-il perdu, et l'a-t-il retrouvé ensuite ? Entrée :
    `spectral_gaps`, des périodes {nom : (premier jour, dernier jour)} en heure locale, le
    point (`mic_id=…`, et `site=`, `dataset=` si besoin). Sortie : une ligne par période :
    `days` et la médiane de chaque `d_<bande>`.
    Règle : médiane par bande sur les jours de la période (bornes comprises).
    Limite : les bornes viennent d'une lecture de la série (`spectral_gap_by_point` ou
    `track_gap_by_point`), pas d'une date de panne connue.
    """
    sel = gaps
    for k, v in point.items():
        sel = sel[sel[k].astype(str) == str(v)]
    names = [c for c in gaps if c.startswith("d_")]
    rows = []
    for label, (a, b) in periods.items():
        part = sel[(sel["day"] >= pd.Timestamp(a)) & (sel["day"] <= pd.Timestamp(b))]
        rows.append({"period": label, "days": len(part), **part[names].median().to_dict()})
    return pd.DataFrame(rows)


# =============================================================================================
# 11. Drapeaux
# =============================================================================================


def exclusion_cause(flags: dict[str, Any]) -> str:
    """La cause d'exclusion principale d'un enregistrement, une seule, dans un ordre fixe.

    Question : pourquoi cet enregistrement est-il écarté ? Entrée : un dict `qc_flags` lu.
    Sortie : « étouffé (calcul) », « étouffé (confirmé) », « étouffé (noté à l'écoute) »,
    « durée et hors relevé », « durée », « hors relevé », « silencieux », ou « autre ».
    Règle : ordre du notebook 06 (§ 1) : l'étouffé l'emporte ; la durée et le hors relevé
    comptent aussi leurs confirmations (`confirmed_…`). Limite : un enregistrement à plusieurs
    causes n'en montre qu'une ; ne regarde pas `keep` (`is_excluded` le fait).
    """
    muffled = bool(flags.get("muffled"))
    if muffled:
        return "étouffé (calcul)"
    if flags.get("confirmed_muffled"):
        return "étouffé (confirmé)"
    if "muffled" in flags.get("annotated", ()):
        return "étouffé (noté à l'écoute)"
    dur = bool(flags.get("duration_off") or flags.get("confirmed_duration_off"))
    off = bool(flags.get("off_campaign") or flags.get("confirmed_off_campaign"))
    if dur and off:
        return "durée et hors relevé"
    if dur:
        return "durée"
    if off:
        return "hors relevé"
    if flags.get("silent"):
        return "silencieux"
    return "autre"


def flag_counts(rec: pd.DataFrame) -> dict[str, Any]:
    """Les drapeaux de la base, comptés : levés, écartants, causes d'exclusion.

    Question : combien d'enregistrements sont écartés, pour quelle cause, combien portent
    chaque drapeau ? Entrée : `load_recordings`. Sortie : dict `recordings`, `excluded`,
    `kept` (encodables), `kept_by_keep` (un drapeau écartant levé, mais `keep`), `raised`
    (par drapeau, `qc.flag_raised`), `causes` (écartés par `exclusion_cause`).
    Règle : `qc.is_excluded` décide, comme partout dans la chaîne. Limite : compte ce qui est
    rangé (`stale_flags` dit si c'est à jour).
    """
    keys = (*EXCLUDING_FLAGS, "clock_off", "survey_edge", "clock_corrected", "rain", "muffled_0",
            "muffled_1", "keep")
    raised = {k: int(rec["fl"].map(lambda f, k=k: flag_raised(f, k)).sum()) for k in keys}
    no_keep = rec["fl"].map(lambda f: is_excluded({k: v for k, v in f.items() if k != "keep"}))
    causes = rec.loc[rec["excluded"], "fl"].map(exclusion_cause).value_counts()
    return {"recordings": int(len(rec)), "excluded": int(rec["excluded"].sum()),
            "kept": int((~rec["excluded"]).sum()),
            "kept_by_keep": int((no_keep & ~rec["excluded"]).sum()),
            "raised": raised, "causes": causes.to_dict()}


def _declared_side(rec: pd.DataFrame, cfg_qc: dict[str, Any]) -> pd.Series:
    """Où tombe chaque enregistrement d'un relevé déclaré : `dans`, `bordure avant`, `bordure
    après` (zone tampon, gardés), `avant la pose`, `après la relève` (hors relevé) ; `NaN` pour
    un relevé sans dates. Même calcul que `qc.metadata_flags`, sur l'heure locale de `rec`."""
    first, last = survey_windows(rec, cfg_qc)
    return position_in_survey(rec["local"], first, last, float(cfg_qc.get("survey_buffer_h", 0)))


_OFF_SIDES = ("avant la pose", "après la relève")


def campaign_blocks(rec: pd.DataFrame, cfg_qc: dict[str, Any]) -> pd.DataFrame:
    """Les blocs de la série de chaque micro : la campagne et ce qui est hors relevé.

    Question : comment la règle du hors relevé découpe-t-elle chaque série ? Entrée :
    `load_recordings`, `qc` de la configuration. Sortie : une ligne par bloc : `block`,
    `campaign` (pas hors relevé), `n`, `n_normal` (de durée normale), `first`, `last` (heure
    locale), `excluded`, `survey` (le relevé déclaré, vide en repli), `where` (`dans`, `bordure
    avant` ou `bordure après` : zone tampon, gardée, `avant la pose`, `après la relève` : hors
    relevé, ou `repli`).
    Règle : celle de `off_campaign` (`qc.metadata_flags`). Un relevé déclaré (`qc.surveys`) donne
    pour chaque micro un bloc dans les dates de pose et de relève, et au plus un bloc par côté
    hors des dates (zone tampon `survey_buffer_h` comprise : ces fichiers restent, signalés
    `survey_edge`). Un relevé sans dates garde l'ancienne règle : dates triées par (jeu, site,
    micro), coupure à chaque trou de plus de `campaign_gap_days`, le plus gros bloc est la
    campagne (§ 7 du notebook 05).
    Limite : un micro qui passe plusieurs relevés a un bloc dans chacun, donc plusieurs blocs de
    campagne ; les « petits blocs » ne sont plus comparables à ceux d'avant les dates déclarées
    (un reste de carte étalé sur des mois fait un seul bloc avant la pose). En repli, le seuil
    n'est pas estimé (pas de dates de pose du terrain) et un micro reposé au même site après
    une pause de plus de `campaign_gap_days` fait deux vrais blocs.
    """
    side = _declared_side(rec, cfg_qc)
    declared = survey_windows(rec, cfg_qc)[0].notna()
    gap = pd.Timedelta(days=float(cfg_qc["campaign_gap_days"]))
    dated = rec.dropna(subset=["t"])

    def row(key: tuple, part: pd.DataFrame, campaign: bool, survey: str, where: str) -> dict:
        return {**dict(zip(POINT, key, strict=True)), "campaign": campaign, "n": len(part),
                "n_normal": int(part["normal"].sum()), "first": part["local"].min(),
                "last": part["local"].max(), "excluded": int(part["excluded"].sum()),
                "survey": survey, "where": where}

    rows = []
    for key, g in dated[~declared[dated.index]].sort_values("t").groupby(POINT):
        block = (g["t"].diff() > gap).cumsum()
        main = block.value_counts().idxmax()
        for k, part in g.groupby(block):
            rows.append(row(key, part, k == main, "", "repli"))
    decl = dated[declared[dated.index]]
    for (*key, survey, where), part in decl.groupby([*POINT, "survey", side[decl.index]]):
        rows.append(row(tuple(key), part, where not in _OFF_SIDES, survey, where))
    cols = [*POINT, "block", "campaign", "n", "n_normal", "first", "last", "excluded",
            "survey", "where"]
    out = pd.DataFrame(rows, columns=[c for c in cols if c != "block"])
    out = out.sort_values([*POINT, "first"], kind="stable").reset_index(drop=True)
    out.insert(len(POINT), "block", out.groupby(POINT).cumcount())
    return out[cols]


def near_campaign_orphans(
    rec: pd.DataFrame, cfg_qc: dict[str, Any], min_gap_days: float = 3.0
) -> pd.DataFrame:
    """Les petits groupes isolés de leur série à l'intérieur du relevé.

    Question : quels enregistrements sont dans les dates de leur relevé, mais séparés du gros de
    la série de leur micro par au moins `min_gap_days` ? Entrée : `load_recordings`, `qc` de la
    configuration, `min_gap_days`. Sortie : les blocs concernés (colonnes de `campaign_blocks`
    sauf `where`, plus `gap_days` jusqu'au bloc voisin le plus proche).
    Règle : découpage refait avec `min_gap_days`, par micro et par relevé déclaré (seulement les
    jours de pose à relève) ; en repli, par micro (ancienne règle de `off_campaign`, où ces
    blocs étaient ceux séparés par moins de `campaign_gap_days`). On garde les blocs non
    principaux dont aucun enregistrement n'est hors relevé. `min_gap_days` = 3 n'est pas dans la
    configuration : choix de lecture.
    Limite : ne dit pas ce que sont ces fichiers (tests, restes de carte) : il faut écouter. Avec
    les dates déclarées, les essais de réglage d'avant la pose n'y sont plus : ils sont hors
    relevé.
    """
    side = _declared_side(rec, cfg_qc)
    declared = survey_windows(rec, cfg_qc)[0].notna()
    gap = pd.Timedelta(days=float(min_gap_days))
    dated = rec.dropna(subset=["t"])
    repli = dated[~declared[dated.index]]
    inside = dated[declared[dated.index] & ~side[dated.index].isin(_OFF_SIDES)]
    series = [(k, g) for k, g in repli.groupby(POINT)]
    series += [(k[:-1], g) for k, g in inside.groupby([*POINT, "survey"])]
    rows = []
    for key, g in series:
        g = g.sort_values("t")
        diff = g["t"].diff()
        block = (diff > gap).cumsum()
        main = block.value_counts().idxmax()
        for k, part in g.groupby(block):
            if k == main or not part["campaign"].all():
                continue
            before = diff.loc[part.index[0]]
            nxt = g["t"][g["t"] > part["t"].max()]
            after = nxt.min() - part["t"].max() if len(nxt) else pd.NaT
            near = min(x for x in (before, after) if pd.notna(x))
            rows.append({**dict(zip(POINT, key, strict=True)), "block": int(k), "n": len(part),
                         "n_normal": int(part["normal"].sum()), "first": part["local"].min(),
                         "last": part["local"].max(), "excluded": int(part["excluded"].sum()),
                         "gap_days": near.total_seconds() / 86400})
    return pd.DataFrame(rows, columns=[*POINT, "block", "n", "n_normal", "first", "last",
                                       "excluded", "gap_days"])


def stale_flags(rec: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Les drapeaux d'inventaire rangés sont-ils ceux que la configuration actuelle donnerait ?

    Question : un chiffre lu dans `qc_flags` est-il à jour ? Entrée : `load_recordings`
    (de préférence `datasets=None` : le hors relevé se calcule par micro, tous jeux confondus),
    la configuration. Sortie : une ligne par drapeau d'inventaire (`duration_off`,
    `off_campaign`, `clock_off`, `survey_edge`) : `stored`, `recomputed`, `differ`
    (enregistrements dont la valeur change). Règle : `qc.metadata_flags`, la fonction même de
    `blanci flag`, appelée sur une copie : rien n'est écrit. Limite : les drapeaux audio
    (`muffled`…) ne se recalculent pas ici (ils demandent la carte CUSUM,
    `qc.apply_audio_flags`, qui écrit en base).
    """
    flags = metadata_flags(rec, cfg["qc"], float(cfg["recorder"]["filename_utc_offset_h"]))
    rows = []
    for k in ("duration_off", "off_campaign", "clock_off", "survey_edge"):
        stored = rec["fl"].map(lambda f, k=k: bool(f.get(k))).to_numpy()
        new = flags[k].to_numpy(dtype=bool)
        rows.append({"flag": k, "stored": int(stored.sum()), "recomputed": int(new.sum()),
                     "differ": int((stored != new).sum())})
    return pd.DataFrame(rows)


# =============================================================================================
# 12. Base et disque
# =============================================================================================


def disk_inventory(root: Path, suffixes: Sequence[str] = AUDIO_SUFFIXES) -> pd.DataFrame:
    """Les fichiers audio présents sous une racine.

    Question : qu'y a-t-il sur ce disque ? Entrée : un dossier (par exemple `paths.raw`), les
    extensions (`audio.suffixes`). Sortie : une ligne par fichier : `path` (relatif à `root`,
    séparateurs POSIX), `stem`, `size`. Règle : parcours de `ingest.iter_audio_files` (ordre
    naturel, fichiers cachés ignorés). Limite : exige le disque (`FileNotFoundError`) ; ne lit
    que les noms et tailles.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"racine absente : {root}")
    rows = [{"path": p.relative_to(root).as_posix(), "stem": p.stem, "size": p.stat().st_size}
            for p in iter_audio_files(root, tuple(suffixes))]
    return pd.DataFrame(rows, columns=["path", "stem", "size"])


def base_vs_disk(rec: pd.DataFrame, disk: pd.DataFrame,
                 renamed: dict[str, str] | None = None) -> pd.DataFrame:
    """Lignes de la base sans fichier, fichiers du disque sans ligne.

    Question : la base et le disque disent-ils la même chose ? Entrée : `load_recordings`,
    `disk_inventory` du disque, `audio.renamed`. Sortie : une ligne par chemin : `source`
    (« base » ou « disque »), `path`, `status` : « sur le disque » (même chemin), « sur le
    disque, dossier renommé », « absent du disque » (son relevé y est, pas lui), « relevé pas
    sur ce disque », « fichier sans ligne ».
    Règle : un chemin de base se cherche sous son nom et sous chaque nom de dossier renommé à
    la copie (`core.locate.variants`). Limite : un « fichier sans ligne » peut être un doublon
    écarté, une erreur de lecture (`ingest_errors_summary`) ou un relevé jamais inventorié ;
    seul son chemin le dit.
    """
    on_disk = set(disk["path"])
    surveys = {"/".join(p.split("/")[:2]) for p in on_disk}
    used: set[str] = set()
    rows = []
    for rid, path in zip(rec["recording_id"], rec["path"], strict=True):
        names = variants(path, renamed)
        hit = next((n for n in names if n in on_disk), None)
        if hit is not None:
            used.add(hit)
            status = "sur le disque" if hit == path else "sur le disque, dossier renommé"
        elif any("/".join(n.split("/")[:2]) in surveys for n in names):
            status = "absent du disque"
        else:
            status = "relevé pas sur ce disque"
        rows.append({"source": "base", "recording_id": rid, "path": path, "status": status})
    rows += [{"source": "disque", "recording_id": None, "path": p, "status": "fichier sans ligne"}
             for p in disk["path"] if p not in used]
    return pd.DataFrame(rows, columns=["source", "recording_id", "path", "status"])


# =============================================================================================
# 13. Annotations et scores
# =============================================================================================


def listened_recordings(con: sqlite3.Connection) -> pd.DataFrame:
    """Les enregistrements écoutés et ce qu'on y a entendu.

    Question : quels enregistrements ont été écoutés, combien de secondes, A. blanci y a-t-il
    été entendu ? Entrée : connexion. Sortie : une ligne par enregistrement écouté :
    `listened_s` (union des extraits écoutés, table `spans`), `labelled_windows` (fenêtres
    étiquetées, table `labels`, ancienne annotation), `positive` (`qc.positive_recordings`).
    Règle : un extrait réécouté ne compte qu'une fois (union des intervalles). Limite : une
    fenêtre étiquetée compte comme écoutée sans durée ; `positive` suit les derniers labels.
    """
    from blanci.inputs.dataset import load_spans
    from blanci.inputs.qc import positive_recordings

    spans, _ = load_spans(con)
    rows = []
    for rid, g in spans.sort_values("start_s").groupby("recording_id"):
        total, end = 0.0, -np.inf
        for a, b in zip(g["start_s"], g["end_s"], strict=True):
            total += max(0.0, b - max(a, end))
            end = max(end, b)
        rows.append({"recording_id": rid, "listened_s": total})
    out = pd.DataFrame(rows, columns=["recording_id", "listened_s"])
    lab = pd.read_sql_query(
        "SELECT w.recording_id, COUNT(DISTINCT l.window_id) AS labelled_windows "
        "FROM labels l JOIN windows w USING (window_id) GROUP BY w.recording_id", con)
    out = out.merge(lab, on="recording_id", how="outer")
    out["listened_s"] = pd.to_numeric(out["listened_s"]).fillna(0.0)
    out["labelled_windows"] = pd.to_numeric(out["labelled_windows"]).fillna(0).astype(int)
    out["positive"] = out["recording_id"].isin(positive_recordings(con))
    return out


def annotation_coverage_by(rec: pd.DataFrame, listened: pd.DataFrame,
                           by: str | Sequence[str] = ("dataset", "site")) -> pd.DataFrame:
    """La part de l'inventaire qui a été écoutée, par groupe.

    Question : quels sites, micros, tranches ont été écoutés, lesquels sont tenus à l'écart ?
    Entrée : `load_recordings`, `listened_recordings`, clés de regroupement. Sortie : une ligne
    par groupe : `recordings`, `listened`, `share_listened`, `listened_h`, `positive`,
    `excluded_listened`. Règle : jointure sur `recording_id`. Limite : écouté ne veut pas dire
    écouté en entier (`listened_h`) ; un site à zéro peut être tenu à l'écart exprès
    (`plan.level2_site`, réservé au test).
    """
    by = [by] if isinstance(by, str) else list(by)
    df = rec.merge(listened, on="recording_id", how="left")
    df["was_listened"] = df["listened_s"].fillna(0).gt(0) | df["labelled_windows"].fillna(0).gt(0)
    df["positive"] = df["positive"].astype("boolean").fillna(False).astype(bool)
    out = df.groupby(by).agg(
        recordings=("recording_id", "size"), listened=("was_listened", "sum"),
        listened_h=("listened_s", lambda s: s.fillna(0).sum() / 3600),
        positive=("positive", "sum"),
        excluded_listened=("excluded", lambda s: int((s & df.loc[s.index, "was_listened"]).sum())),
    ).reset_index()
    out["share_listened"] = out["listened"] / out["recordings"]
    return out


def score_coverage(con: sqlite3.Connection, model_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Les scores en base, par modèle : combien de fenêtres et d'enregistrements.

    Question : quels enregistrements ont été passés par quel modèle ? Entrée : connexion, les
    modèles à compter (`None` = tous : la jointure parcourt toute la table `scores`, plusieurs
    dizaines de millions de lignes, compter une minute). Sortie : une ligne par modèle :
    `windows`, `recordings`. Règle : jointure `scores` → `windows`. Limite : un score ne dit pas
    que l'enregistrement a été encodé en entier.
    """
    where, params = "", []
    if model_ids is not None:
        where = f"WHERE s.model_id IN ({', '.join('?' * len(model_ids))})"
        params = list(model_ids)
    return pd.read_sql_query(
        "SELECT s.model_id, COUNT(*) AS windows, COUNT(DISTINCT w.recording_id) AS recordings "
        f"FROM scores s JOIN windows w USING (window_id) {where} GROUP BY s.model_id",
        con, params=params)


def recording_detections(con: sqlite3.Connection, model_id: str) -> pd.DataFrame:
    """Les détections d'un modèle, par enregistrement.

    Question : combien de fenêtres un modèle (BlanciNet de Biophonia : `blancinet`) a-t-il
    retenues par enregistrement, avec quel score maximal ? Entrée : connexion, modèle. Sortie :
    une ligne par enregistrement scoré : `detections` (fenêtres scorées), `score_max`.
    Règle : requête du notebook 06 (§ 1) et du § 8 du notebook 05. Limite : `blancinet` n'est
    pas un modèle du projet ; ses scores sont un témoin indépendant, pas une vérité.
    """
    return pd.read_sql_query(
        "SELECT w.recording_id, COUNT(*) AS detections, MAX(s.score) AS score_max "
        "FROM scores s JOIN windows w USING (window_id) WHERE s.model_id = ? "
        "GROUP BY w.recording_id", con, params=[model_id])


def excluded_with_detections(rec: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    """Les écartés qu'un détecteur signale.

    Question : parmi les enregistrements écartés, lesquels ont des détections, et pour quelle
    cause sont-ils écartés ? Entrée : `load_recordings`, `recording_detections`. Sortie : les
    écartés détectés, avec `detections`, `score_max`, `cause` (`exclusion_cause`).
    Règle : reprise du notebook 06 (§ 1) ; le seuil de lecture (0,5 dans le notebook) se pose
    après, sur `score_max`. Limite : biais de vérification : on ne voit que ce que le détecteur
    a scoré ; un écarté sans score n'est pas un écarté sans chant.
    """
    df = rec[rec["excluded"]].merge(detections, on="recording_id", how="inner")
    return df.assign(cause=df["fl"].map(exclusion_cause))


def protected_positives(rec: pd.DataFrame, positives: Iterable[str]) -> dict[str, int]:
    """Les enregistrements où A. blanci a été entendu face aux drapeaux.

    Question : un enregistrement positif est-il écarté ? Entrée : `load_recordings`,
    `qc.positive_recordings(con)`. Sortie : dict `positive` (positifs dans `rec`),
    `flag_raised` (portent un drapeau écartant, `keep` ignoré), `excluded` (écartés malgré
    tout ; attendu : 0). Règle : `qc.is_excluded`. Limite : ne protège que ce qui a été écouté.
    """
    pos = rec[rec["recording_id"].isin(set(positives))]
    raised = pos["fl"].map(lambda f: is_excluded({k: v for k, v in f.items() if k != "keep"}))
    return {"positive": int(len(pos)), "flag_raised": int(raised.sum()),
            "excluded": int(pos["excluded"].sum())}


def weighted_prevalence(drawn: pd.DataFrame, by: str = "niveau", n_boot: int = 4000,
                        seed: int = 0) -> pd.DataFrame:
    """La prévalence d'A. blanci par enregistrement tiré, brute et corrigée du tirage.

    Question : quelle part des enregistrements de 2 min contient A. blanci, une fois corrigé le
    surtirage des strates (heures de pic doublées) ? Entrée : `drawn`, une ligne par extrait
    écouté (ou par enregistrement) avec `recording_id`, `pos` (A. blanci entendu), `prob_tirage`
    (probabilité d'inclusion, colonne du `candidats.csv` du plan) et la colonne `by` (le niveau
    du test) ; `n_boot`, `seed` (le tirage du bootstrap, fixé pour que le chiffre se rejoue).
    Sortie : une ligne par valeur de `by` : `n` (enregistrements), `pos` (positifs), `brute`
    (`pos / n`), `ponderee`, `bas`, `haut` (intervalle à 95 %, percentiles du bootstrap).
    Règle : un enregistrement vaut positif si un de ses extraits l'est ; estimateur de
    Horvitz-Thompson normalisé, chaque enregistrement pèse 1 / `prob_tirage` ; le bootstrap
    rééchantillonne les enregistrements d'un niveau, poids compris, à l'intérieur de chaque
    niveau (un seul générateur, qui passe les niveaux dans l'ordre trié). Limite : le tirage de
    l'enregistrement est supposé indépendant d'un enregistrement à l'autre ; un niveau de
    quelques enregistrements donne un intervalle large, et un niveau sans positif un intervalle
    nul. Lève `ValueError` si une probabilité manque ou n'est pas dans ]0, 1].
    """
    per_recording = drawn.groupby("recording_id").agg(
        pos=("pos", "max"), p=("prob_tirage", "first"), group=(by, "first"))
    if not per_recording["p"].between(0, 1, inclusive="right").all():
        raise ValueError("prob_tirage manque ou n'est pas dans ]0, 1] : prévalence non pondérable")
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in per_recording.groupby("group"):
        w, y = 1 / g["p"].to_numpy(), g["pos"].to_numpy(float)
        est = (w * y).sum() / w.sum()
        k = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = (w[k] * y[k]).sum(1) / w[k].sum(1)
        rows.append((key, len(g), int(y.sum()), y.mean(), est, *np.percentile(boot, [2.5, 97.5])))
    return pd.DataFrame(rows, columns=[by, "n", "pos", "brute", "ponderee", "bas", "haut"])


# =============================================================================================
# 14. Comparaisons
# =============================================================================================


def compare_groups(rec: pd.DataFrame, by: str | Sequence[str] = "dataset",
                   programmes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Deux jeux, ou plusieurs sites, sur les mêmes indicateurs, côte à côte.

    Question : en quoi 2023-2024 et 2025-2026 (ou deux sites) diffèrent-ils ? Entrée :
    `load_recordings`, regroupement, `recording_programme` (facultatif). Sortie : une ligne par
    groupe : `recordings`, `hours`, `points`, `recorders`, `days`, `recordings_per_point_day`,
    `share_excluded`, `share_normal`, `share_stereo_48k`, `families`, et `programmes` (si
    donné). Règle : effectifs bruts, aucun test statistique. Limite : les deux jeux ne sont pas
    comparables en saison (un an contre des semaines de saison des pluies).
    """
    by = [by] if isinstance(by, str) else list(by)
    df = rec.assign(stereo48=(rec["sample_rate"] == 48000) & (rec["channels"] == 2),
                    point=rec["dataset"] + "/" + rec["site"].astype(str) + "/"
                    + rec["mic_id"].astype(str))
    out = df.groupby(by).agg(
        recordings=("recording_id", "size"), hours=("duration_s", lambda s: s.sum() / 3600),
        points=("point", "nunique"), recorders=("mic_id", "nunique"), days=("day", "nunique"),
        share_excluded=("excluded", "mean"), share_normal=("normal", "mean"),
        share_stereo_48k=("stereo48", "mean"),
        families=("family", lambda s: ", ".join(sorted(s.dropna().unique()))),
    ).reset_index()
    point_days = df.groupby([*by, "point", "day"]).size().groupby(by).mean()
    out["recordings_per_point_day"] = out.set_index(by).index.map(point_days)
    if programmes is not None:
        p = df.assign(programme=programme_of(rec, programmes)).groupby(by)["programme"].agg(
            lambda s: ", ".join(f"{k} ({v})" for k, v in s.value_counts().items()))
        out["programmes"] = out.set_index(by).index.map(p)
    return out


def _tranche(hours: pd.Series, tranches: dict[str, Sequence[float]]) -> pd.Series:
    out = pd.Series(None, index=hours.index, dtype=object)
    for name, (a, b) in tranches.items():
        out[(hours >= a) & (hours < b)] = name
    return out


def stratum_effort(rec: pd.DataFrame, tranches: dict[str, Sequence[float]],
                   periods: dict[str, Sequence[int]] | None = None,
                   kept_only: bool = True) -> pd.DataFrame:
    """L'effort d'enregistrement par strate du tirage : tranche horaire × période × site.

    Question : combien d'enregistrements le tirage stratifié a-t-il à sa disposition dans
    chaque strate, et quelles strates sont vides ? Entrée : `load_recordings`,
    `plan.tranches` (heure locale, [début, fin[), `plan.periods_2023` (mois locaux,
    facultatif), `kept_only` (seulement les encodables). Sortie : une ligne par (jeu, site,
    tranche[, période]) : `recordings`, `points`. Règle : les tranches et périodes de la
    configuration, sans changement. Limite : une heure hors des tranches (la nuit) n'apparaît
    pas ; les périodes ne valent que pour 2023.
    """
    df = rec[~rec["excluded"]] if kept_only else rec
    hour = df["local"].dt.hour + df["local"].dt.minute / 60
    df = df.assign(tranche=_tranche(hour, tranches))
    keys = ["dataset", "site", "tranche"]
    if periods is not None:
        month = df["local"].dt.month
        period = pd.Series(None, index=df.index, dtype=object)
        for name, months in periods.items():
            period[month.isin(months)] = name
        df = df.assign(period=period)
        keys.append("period")
    return df.dropna(subset=["tranche"]).groupby(keys).agg(
        recordings=("recording_id", "size"), points=("mic_id", "nunique")).reset_index()


# =============================================================================================
# 15. Rejeu des chiffres de la présentation n° 4
# =============================================================================================


def _number(diapo: str, what: str, announced: float | None, value: Any, function: str,
            digits: int = 0, status: str | None = None) -> dict[str, Any]:
    if status is None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            status = "non recalculé"
        elif announced is None:
            status = "recalculé (pas de chiffre annoncé)"
        else:
            status = "ok" if round(float(value), digits) == round(float(announced), digits) \
                else "différent"
    return {"diapo": diapo, "chiffre": what, "annonce": announced, "recalcule": value,
            "fonction": function, "statut": status}


def presentation_4_numbers(
    con: sqlite3.Connection,
    cfg: dict[str, Any],
    lieux: dict[str, str] | None = None,
    pistes: pd.DataFrame | None = None,
    duplicates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Les chiffres de la présentation n° 4 (inventaire et contrôle qualité), recalculés.

    Question : chaque chiffre annoncé tient-il encore sur la base d'aujourd'hui, et d'où
    vient-il ? Entrée : connexion, configuration, `lieux` (site → lieu, pour les relevés
    empilés sur leur site ; `documentation/tableau-de-bord/structure.yaml`, clés au nom de la
    base), `pistes` (`paths.reports/pistes/echantillon_pistes.csv`, sur le disque des rapports)
    et `duplicates` (`paths.reports/ingest_duplicates_*.csv`) quand ils sont lisibles.
    Sortie : une ligne par chiffre : `diapo`, `chiffre`, `annonce` (valeur des notes orateur
    `presentation_4_inventaire_qc_v2_script.md`), `recalcule`, `fonction` (l'appel qui le
    produit), `statut` (« ok », « différent », « non recalculé » faute de fichier).
    Règle : chaque valeur vient d'une fonction de ce module ; aucune n'est recopiée.
    Limite : les chiffres qui demandent l'audio (spectrogrammes de Molokoï, doublons par le
    contenu) ou les files d'écoute (diapos 18, 20 à 23) ne sont pas rejoués ici ; ils sont
    listés à part dans `inventaire.md`.
    """
    qc = cfg["qc"]
    out: list[dict[str, Any]] = []
    add = out.append
    rec = load_recordings(con, cfg)
    tot = corpus_totals(rec)
    fp = database_fingerprint(con)
    add(_number("3", "enregistrements (jeux 2023 et 2026)", 96310, tot["recordings"],
                "corpus_totals(load_recordings(con, cfg))['recordings']"))
    add(_number("3", "heures", 3207, tot["hours"], "corpus_totals(rec)['hours']"))
    add(_number("3", "points d'écoute (jeu, site, micro)", 164, tot["points"],
                "corpus_totals(rec)['points']"))
    add(_number("3", "enregistreurs", 95, tot["recorders"], "corpus_totals(rec)['recorders']"))
    if lieux is not None:
        add(_number("3", "lieux (relevés empilés sur leur site)", 4,
                    rec["site"].map(lambda s: lieux.get(s, s)).nunique(),
                    "rec.site.map(lieux).nunique()"))
    outside = dataset_counts(con)
    add(_number("—", "lignes hors des jeux 2023 et 2026 (non comptées)", None,
                int(outside.loc[~outside["in_project"], "recordings"].sum()),
                "dataset_counts(con)"))
    add(_number("5 bis", "taille de la base (Go)", 6.7, fp["size_bytes"] / 1e9,
                "database_fingerprint(con)['size_bytes']", 1))
    prog = recording_programme(rec)
    pc = programme_counts(rec, prog).groupby("programme")["recordings"].sum()
    add(_number("3", "enregistrements, programme 5 h → 19 h 30", 74007,
                int(pc.get("05:00-19:30", 0)), "programme_counts(rec, recording_programme(rec))"))
    add(_number("3", "enregistrements, programme 6 h → 18 h 30", 22289,
                int(pc.get("06:00-18:30", 0)), "programme_counts(rec, recording_programme(rec))"))
    add(_number("3", "enregistrements de points sans programme lu", None, int(pc.get("?", 0)),
                "programme_counts(rec, recording_programme(rec))"))

    orphans = near_campaign_orphans(rec, qc)
    mat = orphans[(orphans["site"] == "Mataroni") & (orphans["first"] < "2026-01-01")]
    dec = rec[(rec["site"] == "Mataroni")
              & rec["day"].isin(pd.to_datetime(["2025-12-30", "2025-12-31"]))]
    add(_number("10", "Mataroni, 30-31/12/2025 : enregistrements", 26, len(dec),
                "counts_by(rec[Mataroni, 30-31/12], 'site')"))
    add(_number("10", "Mataroni, 30-31/12/2025 : micros", 20, int(dec["mic_id"].nunique()),
                "counts_by(...)"))
    add(_number("4", "dont dans la campagne (trou de moins de 7 j) : enregistrements", 25,
                int(mat["n"].sum()), "near_campaign_orphans(rec, cfg['qc'])"))
    add(_number("4", "dont dans la campagne : micros", 12, int(mat["mic_id"].nunique()),
                "near_campaign_orphans(rec, cfg['qc'])"))
    off_dec = dec[~dec["campaign"]]
    add(_number("10", "Mataroni, 30-31/12/2025, hors relevé (dates déclarées) : enregistrements",
                26, len(off_dec), "rec[Mataroni, 30-31/12].campaign"))
    add(_number("10", "Mataroni, 30-31/12/2025, hors relevé (dates déclarées) : micros", 20,
                int(off_dec["mic_id"].nunique()), "rec[Mataroni, 30-31/12].campaign"))
    add(_number("10", "en bordure de relevé (zone tampon de 24 h, gardés)", None,
                int(rec["fl"].map(lambda f: bool(f.get("survey_edge"))).sum()),
                "flag_counts(rec)['raised']['survey_edge']"))

    if pistes is not None:
        S = pistes[pistes["erreur"].isna()] if "erreur" in pistes else pistes
        S = S.assign(dataset=S["dataset"].astype(str))
        g = gain_settings(S.assign(found=True))
        vc = g["config"].value_counts()
        add(_number("6", "points réglés 6 / 18 dB", 143, int(vc.get("6 / 18 dB", 0)),
                    "gain_settings(echantillon_pistes)"))
        add(_number("6", "points réglés 18 / 6 dB", 21, int(vc.get("18 / 6 dB", 0)),
                    "gain_settings(echantillon_pistes)"))
        inv = g[g["config"] == "18 / 6 dB"]
        fam = inv["mic_id"].str[:3].value_counts()
        add(_number("6", "dont enregistreurs 2MA", 18, int(fam.get("2MA", 0)),
                    "gain_settings(echantillon_pistes)"))
        add(_number("6", "dont enregistreurs SMA", 3, int(fam.get("SMA", 0)),
                    "gain_settings(echantillon_pistes)"))
        jour = S[S["motif"] == "jour"] if "motif" in S else S
        add(_number("6", "enregistrements lus (toutes lignes de l'échantillon)", 3607, len(S),
                    "len(echantillon_pistes.csv)"))
        add(_number("6", "dont un par micro et par jour (le reste : bascules d'état)", None,
                    len(jour), "sample_recordings(rec, channels=2) ≈ motif « jour »"))
        add(_number("6", "points de l'échantillon", 164, int(jour.groupby(POINT).ngroups),
                    "echantillon_pistes.csv"))
        gaps = track_gap_by_point(S, g, gap_col="level_diff_db")
        add(_number("20", "points à plus de 6 dB de l'écart réglé", 5,
                    int((gaps["deviation_db"].abs() > 6).sum()),
                    "track_gap_by_point(echantillon_pistes, gain_settings(...), 'level_diff_db')"))
    else:
        for what, v in (("points réglés 6 / 18 dB", 143), ("points réglés 18 / 6 dB", 21),
                        ("enregistrements lus (un par micro et par jour)", 3607)):
            add(_number("6", what, v, None, "gain_settings(echantillon_pistes)",
                        status="non recalculé : echantillon_pistes.csv absent"))

    days = mic_days(rec, prog)
    cs = completeness_summary(days)
    add(_number("7", "jours-micros intérieurs", 3137, cs["mic_days"],
                "completeness_summary(mic_days(rec, prog))"))
    add(_number("7", "jours-micros complets", 3128, cs["complete"], "completeness_summary(...)"))
    add(_number("7", "part complète (%)", 99.7, 100 * cs["share_complete"],
                "completeness_summary(...)", 1))
    add(_number("7", "jours partiels", 9, cs["partial"], "completeness_summary(...)"))
    for ds, n, share in (("2023", 2217, 99.8), ("2026", 920, 99.5)):
        b = cs["by_dataset"].get(ds, {"mic_days": None, "share_complete": np.nan})
        add(_number("7", f"jours-micros {ds}", n, b["mic_days"], "completeness_summary(...)"))
        add(_number("7", f"part complète {ds} (%)", share, 100 * b["share_complete"],
                    "completeness_summary(...)", 1))
    st = empty_days(rec, "station", cfg["plan"]["stations_2023"])
    longest = st.set_index("station")["longest_days"]
    add(_number("7", "Molokoï E : plus long trou (jours)", 54,
                int(longest.get("Molokoi/E", 0)), "empty_days(rec, 'station', stations)"))
    add(_number("7", "Kaw D : plus long trou (jours)", 5, int(longest.get("Kaw/D", 0)),
                "empty_days(rec, 'station', stations)"))
    em = empty_days(rec, "mic")
    rnrt = em[(em["site"] == "RNRT")]
    n_rnrt = int(sum(pd.Timestamp("2026-02-10") in e for e in rnrt["empty_dates"]))
    add(_number("7", "RNRT : micros sans rien le 10/02/2026", 5, n_rnrt, "empty_days(rec, 'mic')"))

    cc = clock_corrected_counts(rec)
    add(_number("9", "enregistrements à l'horloge corrigée", 1521, int(cc["recordings"].sum()),
                "clock_corrected_counts(rec)"))
    lag = clock_lag_correlation(rec, "SMA14636", "SMA14163", site="Molokoi").set_index("lag_h")
    add(_number("9", "corrélation à −1 h", 0.57, lag.loc[-1.0, "r"],
                "clock_lag_correlation(rec, 'SMA14636', 'SMA14163', site='Molokoi')", 2))
    add(_number("9", "corrélation sans décalage", 0.34, lag.loc[0.0, "r"],
                "clock_lag_correlation(...)", 2))
    best = lag["r"].idxmax() if lag["r"].notna().any() else None
    add(_number("9", "décalage de corrélation maximale (h)", -1, best,
                "clock_lag_correlation(...)", 1))
    add(_number("9", "créneaux communs à −1 h", 1521, int(lag.loc[-1.0, "n"]),
                "clock_lag_correlation(...)"))

    fc = flag_counts(rec)
    causes = fc["causes"]
    add(_number("9 ter", "écartés", 681, fc["excluded"], "flag_counts(rec)['excluded']"))
    add(_number("24", "encodables (diapo : 95 611 sur 96 292)", 95611, fc["kept"],
                "flag_counts(rec)['kept']"))
    add(_number("24", "inventaire du bilan (diapo : 96 292)", 96292, fc["recordings"],
                "flag_counts(rec)['recordings']"))
    muff = sum(v for k, v in causes.items() if k.startswith("étouffé"))
    add(_number("9 ter", "écartés pour son étouffé", 506, muff, "flag_counts(rec)['causes']"))
    add(_number("9 ter", "dont étouffé par le calcul (deux pistes)", 399,
                causes.get("étouffé (calcul)", 0), "flag_counts(rec)['causes']"))
    dur = causes.get("durée", 0) + causes.get("durée et hors relevé", 0)
    add(_number("9 ter", "écartés pour la durée", 173, dur, "flag_counts(rec)['causes']"))
    add(_number("9 ter", "hors relevé seul", 2, causes.get("hors relevé", 0),
                "flag_counts(rec)['causes']"))
    add(_number("9 ter", "hors relevé et durée", 86, causes.get("durée et hors relevé", 0),
                "flag_counts(rec)['causes']"))

    blocks = campaign_blocks(rec, qc)
    small = blocks[~blocks["campaign"]]
    add(_number("10", "blocs", 238, len(blocks), "campaign_blocks(rec, cfg['qc'])"))
    add(_number("10", "petits blocs (hors relevé)", 74, len(small), "campaign_blocks(...)"))
    add(_number("10", "enregistrements hors relevé (levés)", 106, int(small["n"].sum()),
                "campaign_blocks(...)"))
    add(_number("10", "dont de 2 min", 20, int(small["n_normal"].sum()), "campaign_blocks(...)"))
    add(_number("10", "hors relevé écartés", 88, int(small["excluded"].sum()),
                "campaign_blocks(...)"))
    dc = duration_classes(rec, qc)
    add(_number("10", "fichiers de durée anormale", 173, len(dc),
                "duration_classes(rec, cfg['qc'])"))
    vc = dc["duration_class"].value_counts()
    add(_number("10", "dont tests (< 30 s)", 125, int(vc.get("test", 0)), "duration_classes(...)"))
    add(_number("10", "dont coupés (30 à 119 s)", 34, int(vc.get("coupé", 0)),
                "duration_classes(...)"))
    add(_number("10", "dont longs (> 121 s)", 14, int(vc.get("long", 0)), "duration_classes(...)"))
    det = recording_detections(con, "blancinet")
    scored26 = rec[(rec["dataset"] == "2026") & rec["recording_id"].isin(det["recording_id"])]
    add(_number("10", "enregistrements 2026 scorés par BlanciNet", 8349, len(scored26),
                "recording_detections(con, 'blancinet')"))
    dd = dc.merge(det, on="recording_id")
    cut = dd[dd["duration_class"] == "coupé"]
    add(_number("10", "coupés (30 à 119 s) à score BlanciNet ≥ 0,5", 3,
                int((cut["score_max"] >= 0.5).sum()),
                "duration_classes(...) ⋈ recording_detections(con, 'blancinet')"))
    ex = excluded_with_detections(rec, det)
    add(_number("nb 06", "écartés avec une détection BlanciNet", 108, len(ex),
                "excluded_with_detections(rec, recording_detections(con, 'blancinet'))"))
    add(_number("nb 06", "dont score ≥ 0,5", 19, int((ex["score_max"] >= 0.5).sum()),
                "excluded_with_detections(...)"))

    if duplicates is not None:
        add(_number("8", "doublons écartés à l'inventaire (rapport)", 3673, len(duplicates),
                    "duplicate_report_summary(ingest_duplicates_*.csv)"))
    else:
        add(_number("8", "doublons écartés à l'inventaire (rapport)", 3673, None,
                    "duplicate_report_summary(ingest_duplicates_*.csv)",
                    status="non recalculé : rapport absent"))

    tf = track_flags_summary(rec, qc).set_index(["category", "excluded"])["recordings"]
    add(_number("17", "deux pistes retenues, écartés", 399,
                int(tf.get(("deux pistes retenues", True), 0)), "track_flags_summary(rec, qc)"))
    add(_number("17", "piste 0 retenue seule, gardés (encodés sur la piste 1)", 407,
                int(tf.get(("piste 0 retenue seule", False), 0)), "track_flags_summary(...)"))
    add(_number("17", "piste 0 retenue seule, écartés (piste 1 douteuse)", 85,
                int(tf.get(("piste 0 retenue seule", True), 0)), "track_flags_summary(...)"))
    cats = tf.groupby(level=0).sum()
    add(_number("17", "piste 1 retenue seule (notes : « une quarantaine »)", 40,
                int(cats.get("piste 1 retenue seule", 0)), "track_flags_summary(...)"))
    add(_number("17", "candidats isolés de la piste 0 (aucune piste retenue)", 268,
                int(cats.get("candidat isolé, piste 0", 0)), "track_flags_summary(...)"))
    excl = rec[rec["excluded"]].groupby(["site", "mic_id"]).size().sort_values(ascending=False)
    add(_number("19", "écartés portés par les six points les plus touchés (tous à CDR)", 452,
                int(excl.head(6).sum()), "counts_by(rec[rec.excluded], ['site', 'mic_id'])"))
    stale = stale_flags(load_recordings(con, cfg, datasets=None), cfg).set_index("flag")
    add(_number("—", "drapeaux d'inventaire périmés en base (à recalculer par blanci flag)", 0,
                int(stale["differ"].sum()), "stale_flags(load_recordings(con, cfg, None), cfg)"))
    from blanci.inputs.qc import positive_recordings

    prot = protected_positives(rec, positive_recordings(con))
    add(_number("11", "enregistrements où A. blanci a été entendu, écartés (règle : aucun)", 0,
                prot["excluded"], "protected_positives(rec, positive_recordings(con))"))
    return pd.DataFrame(out)


# =============================================================================================
# 18. Tableau de bord : ce que le contrôle qualité a gardé, écarté, et ce qui change de piste
# =============================================================================================


def qc_outcome(rec: pd.DataFrame, kept: Iterable[str]) -> dict[str, Any]:
    """Ce que le contrôle qualité a fait de l'inventaire : retenus, écartés par cause, sauvés.

    Question : combien d'enregistrements le corpus garde-t-il, combien sont écartés, et pour
    quelle cause ? Entrée : `load_recordings(con, cfg, datasets=None)` ; `kept` : les
    identifiants que garde `embed.select_recordings` (il ne retire jamais un enregistrement
    où A. blanci a été entendu). Sortie : dict `recordings`, `kept`, `excluded` (= recordings -
    kept), `hours_total`, `hours_kept`, `causes` (écartés par `exclusion_cause`), `flagged`
    (drapeaux écartants levés, avant le positif), `saved_by_positive` (flagués mais gardés).
    Règle : une seule cause par enregistrement écarté (`exclusion_cause`). Limite : compte la
    base telle qu'elle est ; `database_fingerprint` la date.
    """
    gardes = rec["recording_id"].isin(set(kept))
    ecartes = rec.loc[~gardes]
    causes = ecartes["fl"].map(exclusion_cause).value_counts()
    return {
        "recordings": int(len(rec)),
        "kept": int(gardes.sum()),
        "excluded": int((~gardes).sum()),
        "hours_total": float(rec["duration_s"].sum() / 3600),
        "hours_kept": float(rec.loc[gardes, "duration_s"].sum() / 3600),
        "causes": {str(k): int(v) for k, v in causes.items()},
        "flagged": int(rec["excluded"].sum()),
        "saved_by_positive": int((rec["excluded"] & gardes).sum()),
    }


def piste_outcome(corpus: pd.DataFrame) -> dict[str, Any]:
    """Sur quelle piste s'encode chaque enregistrement retenu, et pourquoi on a changé de piste.

    Question : combien d'enregistrements passent sur la piste 1 (le « switch »), et pour quelle
    raison ? Entrée : le manifeste du corpus, `scripts/encodage/manifeste_corpus.corpus(con,
    cfg)` : colonnes `piste` (« 0 » ou « 1 ») et `origine`. Sortie : dict `kept` (lignes du
    manifeste), `piste0_defaut` (origine « défaut »), `piste0_regle` (piste 0 choisie par une
    règle ou une écoute : pas un switch), `switches` (piste 1), `causes` (origine → nombre, pour
    la piste 1). Règle : la piste 0 est le défaut, la piste 1 est un switch. Limite : l'origine
    dit ce qui a décidé, pas si la décision était juste ; voir `channels.expected_channels`.
    """
    piste = corpus["piste"].astype(str)
    origine = corpus["origine"].astype(str)
    s1 = piste == "1"
    return {
        "kept": int(len(corpus)),
        "piste0_defaut": int(((piste == "0") & (origine == "défaut")).sum()),
        "piste0_regle": int(((piste == "0") & (origine != "défaut")).sum()),
        "switches": int(s1.sum()),
        "causes": {str(k): int(v) for k, v in origine[s1].value_counts().items()},
    }


def window_counts(rec: pd.DataFrame, windows: Sequence[float] = (3, 5, 6),
                  overlap: float = 0.5) -> dict[float, dict[str, int]]:
    """Fenêtres potentielles par longueur, par enregistrement de 2 min et au total.

    Question : combien de fenêtres d'encodage le corpus représente-t-il ? Entrée : `load_recordings`
    (colonne `duration_s`). Sortie : pour chaque longueur w, `per_120s` (fenêtres d'un
    enregistrement de 120 s), `hop_s` (pas, w × (1 − chevauchement)) et `total` (somme sur
    tous les enregistrements). Règle : une fenêtre par pas tant qu'elle tient dans le fichier.
    Limite : un fichier plus court que w n'a aucune fenêtre.
    """
    out: dict[float, dict[str, int]] = {}
    dur = rec["duration_s"].astype(float)
    for w in windows:
        hop = w * (1 - overlap)
        n = np.where(dur >= w, np.floor((dur - w) / hop) + 1, 0).astype(np.int64)
        out[w] = {"per_120s": int(np.floor((120 - w) / hop) + 1), "hop_s": hop,
                  "total": int(n.sum())}
    return out
