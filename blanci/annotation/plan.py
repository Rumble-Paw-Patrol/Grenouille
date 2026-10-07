"""Tirage par plan du lot 1 (DECISIONS §5.1–5.5, n° 157, n° 196) : sans détecteur ni score,
graine fixée, tout avant écoute.

1. **Partition des points** (`draw_partition`) : niveau 2, un site 2026 entier ; niveau 1,
   20 % des points de chacun des autres sites 2026 ; niveau 3, une station 2023 sur deux par
   site. Tout le reste va à l'entraînement. Un point 2026 est le couple site/micro
   (`recordings_table`, n° 38) ; un point 2023 est la station (Kaw C, Kaw D…), qui garde ses
   micros successifs (`plan.stations_2023`). La partition est versionnée
   (`plan.partition_file`) et relue ensuite telle quelle : on ne la tire qu'une fois.
2. **Lot d'entraînement 1** (`training_lot`) : extraits de 30 s, début tiré au hasard dans
   l'enregistrement. 3 par point 2026 (pic du matin, pic du soir, une autre tranche au
   hasard) ; 50 par station 2023 (périodes haute 50 % dont deux tiers en février–mars,
   transition 20 %, basse 30 % ; pics 60 %, autres tranches 40 %).
3. **Jeu de test** (`evaluation_set`) : enregistrements de 2 min entiers sur les points tenus à
   l'écart. Strate = niveau × site × tranche (× période en 2023) ; dans une strate, des
   couples (point, jour) distincts tirés au hasard, puis un enregistrement au hasard dans
   chacun. La probabilité d'inclusion est donc exacte : n_h / M_h × 1 / R, avec M_h les
   couples (point, jour) de la strate et R les enregistrements du couple tiré. Le tirage est
   réparti entre strates en proportion de M_h × poids (pics × 2, période haute × 2).

Partout : au plus un enregistrement par point, jour et tranche ; enregistrements écartés par
un drapeau et enregistrements déjà labellisés (dont les labels du détecteur externe, §5.2)
exclus ; pluie et saturation restent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blanci.annotation.workbench import CANDIDATE_COLUMNS
from blanci.inputs.dataset import local_days, local_minutes, recordings_table
from blanci.inputs.qc import is_excluded

TRAIN = "entrainement"
LEVELS = ("niveau1", "niveau2", "niveau3")
PEAKS = ("pic_matin", "pic_soir")
# Colonnes ajoutées à la file du poste (relues telles quelles par `load_candidates`).
PLAN_COLUMNS = ["point", "niveau", "strate", "prob_tirage"]


# --- Enregistrements éligibles ------------------------------------------------------------------


def _tranche(minutes: pd.Series, tranches: dict[str, list[int]]) -> pd.Series:
    """Tranche horaire (heure locale, [début, fin[) ; hors tranche (nuit) : None."""
    hours = minutes / 60
    out = pd.Series(None, index=minutes.index, dtype=object)
    for name, (start, stop) in tranches.items():
        out[(hours >= start) & (hours < stop)] = name
    return out


def _period(months: pd.Series, periods: dict[str, list[int]]) -> pd.Series:
    """Période 2023 (§5.3) par mois local : haute_fevmars, haute (reste de décembre–avril),
    transition, basse."""
    out = pd.Series(None, index=months.index, dtype=object)
    for name, wanted in periods.items():
        out[months.isin(wanted)] = name
    return out


def station_of(cfg: dict) -> dict[tuple[str, str], str]:
    """(site, micro) 2023 → point « site/station » (`plan.stations_2023`)."""
    return {
        (site, mic): f"{site}/{station}"
        for site, stations in cfg["plan"]["stations_2023"].items()
        for station, mics in stations.items()
        for mic in mics
    }


def eligible_recordings(con: sqlite3.Connection, cfg: dict) -> pd.DataFrame:
    """Enregistrements tirables, avec point, jour, tranche et période locaux.

    Écartés : drapeau d'exclusion, hors tranche horaire, déjà labellisés (une fenêtre ou un
    intervalle, quelle qu'en soit la source)."""
    plan = cfg["plan"]
    offset = cfg["recorder"]["filename_utc_offset_h"]
    rec = recordings_table(con)
    rec = rec[~rec["qc_flags"].map(is_excluded)].copy()
    labelled = {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT w.recording_id FROM labels l JOIN windows w USING (window_id) "
            "UNION SELECT DISTINCT recording_id FROM spans"
        )
    }
    rec = rec[~rec["recording_id"].isin(labelled)]
    stations = station_of(cfg)
    is_2023 = rec["dataset"].astype(str) == "2023"
    rec.loc[is_2023, "point"] = [
        stations.get((s, m), f"{s}/{m}")
        for s, m in zip(rec.loc[is_2023, "site"], rec.loc[is_2023, "mic_id"], strict=True)
    ]
    rec["day"] = local_days(rec["start_utc"], offset)
    rec["tranche"] = _tranche(local_minutes(rec["start_utc"], offset), plan["tranches"])
    months = pd.to_datetime(rec["day"]).dt.month
    rec["period"] = np.where(is_2023, _period(months, plan["periods_2023"]), None)
    return rec[rec["tranche"].notna()].reset_index(drop=True)


# --- 1. Partition des points --------------------------------------------------------------------


def draw_partition(rec: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Une ligne par point : dataset, site, point, micros, niveau (niveau1/2/3 ou
    entrainement). Graine `plan.seed`."""
    plan = cfg["plan"]
    rng = np.random.default_rng(plan["seed"])
    points = (
        rec.groupby(["dataset", "site", "point"])["mic_id"]
        .agg(lambda s: " ".join(sorted(set(s))))
        .rename("micros")
        .reset_index()
        .sort_values(["dataset", "site", "point"])
        .reset_index(drop=True)
    )
    points["niveau"] = TRAIN
    level2 = plan["level2_site"]
    for (dataset, site), part in points.groupby(["dataset", "site"], sort=True):
        if str(dataset) == "2026" and site == level2:
            points.loc[part.index, "niveau"] = "niveau2"
        elif str(dataset) == "2026":
            k = max(1, round(plan["level1_fraction"] * len(part)))
            points.loc[rng.choice(part.index, size=k, replace=False), "niveau"] = "niveau1"
        else:  # 2023 : une station sur deux (la moitié, arrondie à l'inférieur, au moins 1)
            k = max(1, len(part) // 2)
            points.loc[rng.choice(part.index, size=k, replace=False), "niveau"] = "niveau3"
    if level2 not in set(points.loc[points["dataset"].astype(str) == "2026", "site"]):
        raise ValueError(f"site du niveau 2 introuvable en 2026 : {level2}")
    return points


def read_partition(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"dataset": str})


# --- Tirage commun -----------------------------------------------------------------------------


def _pick_days(pool: pd.DataFrame, k: int, rng: np.random.Generator) -> tuple[pd.DataFrame, int]:
    """k jours distincts du pool (un point, une tranche…), puis un enregistrement au hasard
    dans chacun. Renvoie les enregistrements tirés (colonne `n_day` : enregistrements du jour)
    et le nombre de jours disponibles."""
    days = sorted(pool["day"].unique())
    if not days or k <= 0:
        return pool.iloc[:0], len(days)
    chosen = rng.choice(days, size=min(k, len(days)), replace=False)
    rows = []
    for day in chosen:
        same = pool[pool["day"] == day]
        row = same.iloc[[int(rng.integers(len(same)))]].assign(n_day=len(same))
        rows.append(row)
    return pd.concat(rows), len(days)


def _largest_remainder(weights: pd.Series, n: int, caps: pd.Series) -> pd.Series:
    """n réparti en proportion de `weights` (plus forts restes), sans dépasser `caps`."""
    out = pd.Series(0, index=weights.index, dtype="int64")
    left = n
    active = weights[(weights > 0) & (caps > 0)].index
    while left > 0 and len(active):
        share = weights[active] / weights[active].sum() * left
        base = np.minimum(
            np.floor(share).astype("int64"), caps[active].astype("int64") - out[active]
        )
        out[active] += base
        left -= int(base.sum())
        room = active[out[active] < caps[active]]
        if left > 0 and len(room):
            rest = (share - np.floor(share))[room].sort_values(ascending=False, kind="stable")
            for key in rest.index[:left]:
                out[key] += 1
                left -= 1
        active = active[out[active] < caps[active]]
        if base.sum() == 0 and left > 0 and not len(room):
            break
    return out


def _as_queue(picked: pd.DataFrame, offsets, dur_s, seed: int) -> pd.DataFrame:
    out = picked.assign(offset_s=offsets, dur_s=dur_s, score=np.nan, source="plan")
    out = out.reindex(columns=CANDIDATE_COLUMNS + PLAN_COLUMNS)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --- 2. Lot d'entraînement 1 ---------------------------------------------------------------------


def _quota(n: int, shares: dict[str, float]) -> dict[str, int]:
    weights = pd.Series(shares, dtype=float)
    return _largest_remainder(weights, n, pd.Series(n, index=weights.index)).to_dict()


def training_lot(rec: pd.DataFrame, partition: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Extraits de 30 s des points d'entraînement (§5.4). Colonnes du poste + point, niveau,
    strate (prob_tirage vide : l'entraînement n'est pas repondéré)."""
    plan, lot = cfg["plan"], cfg["plan"]["lot1"]
    rng = np.random.default_rng(plan["seed"] + 1)
    train = set(partition.loc[partition["niveau"] == TRAIN, "point"])
    rec = rec[rec["point"].isin(train)]
    others = [t for t in plan["tranches"] if t not in PEAKS]
    picked = []
    for _point, part in rec[rec["dataset"].astype(str) == "2026"].groupby("point", sort=True):
        tranches = [*PEAKS, str(rng.choice(others))][: lot["per_point_2026"]]
        for tranche in tranches:
            got, _ = _pick_days(part[part["tranche"] == tranche], 1, rng)
            picked.append(got.assign(strate=f"2026/{tranche}"))
    for _point, part in rec[rec["dataset"].astype(str) == "2023"].groupby("point", sort=True):
        per_period = _quota(lot["per_station_2023"], lot["periods_2023"])
        for period, n in per_period.items():
            n_peak = round(lot["peak_share"] * n)
            wanted = [PEAKS[i % 2] for i in range(n_peak)]
            rng.shuffle(wanted)  # le pic en plus (n impair) tombe au hasard
            wanted += [str(rng.choice(others)) for _ in range(n - n_peak)]
            for tranche in sorted(set(wanted)):
                k = wanted.count(tranche)
                pool = part[(part["period"] == period) & (part["tranche"] == tranche)]
                got, _ = _pick_days(pool, k, rng)
                picked.append(got.assign(strate=f"2023/{period}/{tranche}"))
    out = pd.concat(picked, ignore_index=True)
    extract = float(lot["extract_s"])
    room = (out["duration_s"].astype(float) - extract).clip(lower=0)
    offsets = np.round(rng.uniform(0, 1, len(out)) * room, 1)
    out = out.assign(niveau=TRAIN, reason="lot1/" + out["strate"], prob_tirage=np.nan)
    return _as_queue(out, offsets, extract, plan["seed"])


# --- 3. Jeu de test ----------------------------------------------------------------------------


def evaluation_set(rec: pd.DataFrame, partition: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Enregistrements entiers des points tenus à l'écart (§5.5), avec leur probabilité
    d'inclusion (`prob_tirage`)."""
    plan, test = cfg["plan"], cfg["plan"]["test"]
    rng = np.random.default_rng(plan["seed"] + 2)
    level = partition.set_index("point")["niveau"]
    rec = rec[rec["point"].map(level).isin(LEVELS)].copy()
    rec["niveau"] = rec["point"].map(level)
    rec["strate"] = (
        rec["niveau"]
        + "/"
        + rec["site"]
        + "/"
        + np.where(rec["period"].notna(), rec["period"].astype(str) + "/", "")
        + rec["tranche"]
    )
    rec["psu"] = rec["point"] + "|" + rec["day"]
    strata = rec.groupby("strate").agg(
        niveau=("niveau", "first"),
        tranche=("tranche", "first"),
        period=("period", "first"),
        m=("psu", "nunique"),
    )
    weight = strata["m"].astype(float)
    weight[strata["tranche"].isin(PEAKS)] *= test["peak_weight"]
    weight[strata["period"].isin(test["high_periods"])] *= test["high_weight"]
    per_level = _quota(test["n"], {lv: 1.0 for lv in sorted(strata["niveau"].unique())})
    alloc = pd.concat(
        [
            _largest_remainder(
                weight[strata["niveau"] == lv], n, strata.loc[strata["niveau"] == lv, "m"]
            )
            for lv, n in per_level.items()
        ]
    )
    picked = []
    for strate, n in alloc.items():
        if n <= 0:
            continue
        units = rec[rec["strate"] == strate]
        psus = sorted(units["psu"].unique())
        chosen = rng.choice(psus, size=int(n), replace=False)
        for psu in chosen:
            same = units[units["psu"] == psu]
            row = same.iloc[[int(rng.integers(len(same)))]]
            picked.append(row.assign(prob_tirage=n / len(psus) / len(same)))
    out = pd.concat(picked, ignore_index=True)
    out = out.assign(reason="test_v1/" + out["strate"])
    return _as_queue(out, 0.0, out["duration_s"].astype(float).round(2).to_numpy(), plan["seed"])


def draw_plan(con: sqlite3.Connection, cfg: dict, partition_path: Path) -> dict[str, Any]:
    """Partition (relue si `partition_path` existe, tirée et écrite sinon), lot 1, test v1."""
    rec = eligible_recordings(con, cfg)
    if partition_path.exists():
        partition, drawn = read_partition(partition_path), False
    else:
        partition, drawn = draw_partition(rec, cfg), True
        partition_path.parent.mkdir(parents=True, exist_ok=True)
        partition.to_csv(partition_path, index=False)
    return {
        "partition": partition,
        "partition_drawn": drawn,
        "lot1": training_lot(rec, partition, cfg),
        "test": evaluation_set(rec, partition, cfg),
    }
