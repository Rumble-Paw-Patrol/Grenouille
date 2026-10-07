"""Tirage par plan du lot 1 (DECISIONS §5.2–5.5, n° 196), sur une base synthétique."""

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from blanci.annotation.plan import (
    PEAKS,
    TRAIN,
    draw_partition,
    draw_plan,
    eligible_recordings,
    evaluation_set,
    training_lot,
)
from blanci.annotation.workbench import CANDIDATE_COLUMNS, load_candidates
from blanci.core.config import load_config
from blanci.core.db import connect, recording_id_for, window_id_for
from blanci.service import append_label

STATIONS = {"Kaw": {"C": ["SMA1"], "D": ["SMA2", "SMA3"]}}


def _cfg():
    cfg = load_config()
    cfg["plan"]["stations_2023"] = STATIONS
    cfg["plan"]["level2_site"] = "RNRT"
    cfg["plan"]["test"]["n"] = 60
    return cfg


def _add_days(rows, dataset, site, mic, days):
    """Un enregistrement de 120 s toutes les 30 min, de 5 h à 20 h locales (UTC−3)."""
    for day in days:
        for k in range(30):
            local = day + timedelta(hours=5, minutes=30 * k)
            rel = f"{dataset}/{site}/{mic}/{mic}_{local:%Y%m%d_%H%M%S}.wav"
            utc = local + timedelta(hours=3)
            rows.append(
                (recording_id_for(rel), rel, dataset, site, mic, f"{utc:%Y-%m-%dT%H:%M:%SZ}")
            )


@pytest.fixture
def con(tmp_path):
    con = connect(tmp_path / "blanci.sqlite")
    rows = []
    days_2026 = [datetime(2026, 2, 9) + timedelta(days=i) for i in range(4)]
    for i in range(3):
        _add_days(rows, "2026", "RNRT", f"R{i}", days_2026)
    for i in range(10):
        _add_days(rows, "2026", "Mataroni", f"M{i}", days_2026)
    # 2023 : quatre jours par mois sur un an ; la station D change de micro en septembre.
    days_2023 = [datetime(2024, m, d) for m in range(1, 13) for d in (3, 10, 17, 24)]
    _add_days(rows, "2023", "Kaw", "SMA1", days_2023)
    _add_days(rows, "2023", "Kaw", "SMA2", [d for d in days_2023 if d.month < 9])
    _add_days(rows, "2023", "Kaw", "SMA3", [d for d in days_2023 if d.month >= 9])
    con.executemany(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) VALUES (?, ?, ?, ?, ?, ?, 120.0, 48000, 1, "
        "'{}')",
        rows,
    )
    con.commit()
    return con


def test_partition_holds_out_a_site_a_fifth_of_points_and_one_station_of_two(con):
    cfg = _cfg()
    rec = eligible_recordings(con, cfg)
    part = draw_partition(rec, cfg)
    by = part.groupby(["site", "niveau"]).size()
    assert by[("RNRT", "niveau2")] == 3 and ("RNRT", TRAIN) not in by
    assert by[("Mataroni", "niveau1")] == 2 and by[("Mataroni", TRAIN)] == 8
    kaw = part[part["site"] == "Kaw"].set_index("point")
    assert set(kaw.index) == {"Kaw/C", "Kaw/D"}  # la station garde ses deux micros
    assert kaw.loc["Kaw/D", "micros"] == "SMA2 SMA3"
    assert sorted(kaw["niveau"]) == ["entrainement", "niveau3"]
    assert draw_partition(rec, cfg).equals(part)  # graine fixée


def test_training_lot_follows_the_quotas_and_never_touches_held_out_points(con):
    cfg = _cfg()
    rec = eligible_recordings(con, cfg)
    part = draw_partition(rec, cfg)
    lot = training_lot(rec, part, cfg)
    assert list(lot.columns[: len(CANDIDATE_COLUMNS)]) == CANDIDATE_COLUMNS
    assert set(lot["source"]) == {"plan"} and lot["score"].isna().all()
    assert set(lot["point"]) <= set(part.loc[part["niveau"] == TRAIN, "point"])
    assert (lot["dur_s"] == 30).all()
    assert ((lot["offset_s"] >= 0) & (lot["offset_s"] <= 90)).all()
    lot26 = lot[lot["strate"].str.startswith("2026")]
    per_point = lot26.groupby("point")["strate"].apply(lambda s: sorted(s.str.split("/").str[-1]))
    assert len(per_point) == 8
    for tranches in per_point:
        assert len(tranches) == 3 and set(PEAKS) <= set(tranches)
    lot23 = lot[lot["strate"].str.startswith("2023")]
    assert len(lot23) == 50
    period = lot23["strate"].str.split("/").str[1].value_counts()
    assert period.to_dict() == {"haute_fevmars": 17, "basse": 15, "transition": 10, "haute": 8}
    peaks = lot23["strate"].str.split("/").str[-1].isin(PEAKS).sum()
    assert 28 <= peaks <= 32  # 60 % par période, arrondis
    # Au plus un enregistrement par point, jour et tranche.
    rec_day = rec.set_index("recording_id")[["day", "tranche"]]
    keys = lot.join(rec_day, on="recording_id")[["point", "day", "tranche"]]
    assert not keys.duplicated().any()


def test_test_set_draws_whole_held_out_recordings_with_their_probability(con):
    cfg = _cfg()
    rec = eligible_recordings(con, cfg)
    part = draw_partition(rec, cfg)
    test = evaluation_set(rec, part, cfg)
    assert len(test) == 60 and test.groupby("niveau").size().to_dict() == {
        "niveau1": 20,
        "niveau2": 20,
        "niveau3": 20,
    }
    held = set(part.loc[part["niveau"] != TRAIN, "point"])
    assert set(test["point"]) <= held
    assert (test["offset_s"] == 0).all() and (test["dur_s"] == 120).all()
    # Probabilité exacte : n_h / M_h (couples point-jour de la strate) × 1 / R (enregistrements
    # du couple). Ici chaque couple point-jour-tranche a le même nombre d'enregistrements.
    rec_day = rec.set_index("recording_id")[["day", "tranche"]]
    drawn = test.join(rec_day, on="recording_id")
    for _strate, part_h in drawn.groupby("strate"):
        units = rec[rec["point"].isin(part[part["niveau"] == part_h["niveau"].iat[0]]["point"])]
        units = units[(units["site"] == part_h["site"].iat[0])]
        units = units[units["tranche"] == part_h["tranche"].iat[0]]
        if part_h["strate"].iat[0].count("/") == 3:  # 2023 : la période en plus
            units = units[units["period"] == part_h["strate"].iat[0].split("/")[2]]
        m = units.groupby(["point", "day"]).ngroups
        r = units.groupby(["point", "day"]).size().iat[0]
        assert part_h["prob_tirage"].tolist() == pytest.approx([len(part_h) / m / r] * len(part_h))
    # Pics surreprésentés : à stock égal, une strate de pic tire plus qu'une autre.
    tranche = drawn[drawn["niveau"] == "niveau2"]["tranche"].value_counts()
    assert tranche.get("pic_matin", 0) > tranche.get("aube", 0)


def test_flagged_and_labelled_recordings_are_never_drawn(con):
    cfg = _cfg()
    rnrt = con.execute("SELECT recording_id FROM recordings WHERE site = 'RNRT'").fetchall()
    flagged, labelled = rnrt[0][0], rnrt[1][0]
    con.execute(
        "UPDATE recordings SET qc_flags = ? WHERE recording_id = ?",
        (json.dumps({"in_bag": True}), flagged),
    )
    con.execute(
        "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) VALUES (?, ?, 0, 3)",
        (window_id_for(labelled, 0.0), labelled),
    )
    append_label(con, window_id_for(labelled, 0.0), "blanci", "flag")
    rec = eligible_recordings(con, cfg)
    assert not {flagged, labelled} & set(rec["recording_id"])


def test_draw_plan_writes_the_partition_once_then_reads_it_back(con, tmp_path):
    cfg = _cfg()
    path = tmp_path / "plan" / "partition_v1.csv"
    first = draw_plan(con, cfg, path)
    assert first["partition_drawn"] and path.exists()
    saved = pd.read_csv(path, dtype={"dataset": str})
    assert set(saved.columns) == {"dataset", "site", "point", "micros", "niveau"}
    cfg["plan"]["seed"] = 7  # une autre graine ne retire pas la partition versionnée
    again = draw_plan(con, cfg, path)
    assert not again["partition_drawn"]
    assert again["partition"].equals(saved)


def test_the_workbench_reads_a_plan_queue(con, tmp_path):
    cfg = _cfg()
    plan = draw_plan(con, cfg, tmp_path / "partition.csv")
    path = tmp_path / "candidats.csv"
    plan["lot1"].to_csv(path, index=False)
    queue = load_candidates(path, con)
    assert len(queue) == len(plan["lot1"]) and (queue["dur_s"] == 30).all()
    assert queue["path"].notna().all() and (queue["source"] == "plan").all()
