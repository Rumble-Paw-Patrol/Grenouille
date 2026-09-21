import numpy as np
import pandas as pd
import pytest

from blanci.dataset import (
    current_labels,
    local_minutes,
    paired_negatives,
    recordings_table,
    training_set,
    transfer_labels,
)
from blanci.db import connect, recording_id_for, utc_now, window_id_for

WINDOW_S = 3.0
HOP_S = 1.5


@pytest.fixture
def con(tmp_path):
    return connect(tmp_path / "blanci.sqlite")


def add_recording(con, rel_path, site="mataroni", mic="M1", start_utc="2026-02-10T13:00:00Z"):
    """Un enregistrement de 2 min. Heure locale = UTC−3, donc 13 h UTC = 10 h sur le terrain."""
    rid = recording_id_for(rel_path)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels) VALUES (?, ?, '2026', ?, ?, ?, 120.0, 32000, 1)",
        (rid, rel_path, site, mic, start_utc),
    )
    return rid


def add_window(con, recording_id, offset_s, dur_s=WINDOW_S):
    wid = window_id_for(recording_id, offset_s)
    con.execute(
        "INSERT OR IGNORE INTO windows (window_id, recording_id, offset_s, dur_s) "
        "VALUES (?, ?, ?, ?)",
        (wid, recording_id, offset_s, dur_s),
    )
    return wid


def add_label(con, window_id, label, source="import", quality=None):
    con.execute(
        "INSERT INTO labels (window_id, label, quality, source, created_at) VALUES (?, ?, ?, ?, ?)",
        (window_id, label, quality, source, utc_now()),
    )


def grid_frame(recording_ids, duration_s=120.0, window_s=WINDOW_S, hop_s=HOP_S):
    rows = []
    for rid in recording_ids:
        offset = 0.0
        while offset + window_s <= duration_s + 1e-9:
            rows.append(
                {
                    "window_id": window_id_for(rid, offset),
                    "recording_id": rid,
                    "offset_s": offset,
                    "dur_s": window_s,
                }
            )
            offset = round(offset + hop_s, 2)
    return pd.DataFrame(rows)


# --- Dernier label de chaque fenêtre ----------------------------------------------------------


def test_current_labels_keeps_the_latest_correction(con):
    """Les labels sont en ajout seul : une correction est une ligne plus récente (§13.7)."""
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    wid = add_window(con, rid, 30.0)
    add_label(con, wid, "blanci_solo")
    add_label(con, wid, "bird", source="active")
    labels = current_labels(con)
    assert len(labels) == 1
    assert labels.loc[0, "label"] == "bird" and labels.loc[0, "source"] == "active"


def test_current_labels_joins_window_geometry(con):
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    add_label(con, add_window(con, rid, 42.0), "blanci_solo", quality="A")
    labels = current_labels(con)
    assert labels.loc[0, "offset_s"] == 42.0 and labels.loc[0, "dur_s"] == WINDOW_S
    assert labels.loc[0, "quality"] == "A"


def test_recordings_table_builds_the_point_key(con):
    add_recording(con, "2026/tresor/T3/a.wav", site="tresor", mic="T3")
    assert recordings_table(con).loc[0, "point"] == "tresor/T3"


# --- Transfert des annotations vers la grille (DECISIONS n° 4) --------------------------------


def annotation(recording_id, offset_s, dur_s, label="blanci_solo"):
    return pd.DataFrame(
        [{"recording_id": recording_id, "offset_s": offset_s, "dur_s": dur_s, "label": label}]
    )


def test_window_containing_a_short_annotation_inherits_it():
    """Annotation Biophonia de 3 s dans une fenêtre de 5 s : la fenêtre la contient."""
    grid = pd.DataFrame([{"window_id": "w", "recording_id": "r", "offset_s": 10.0, "dur_s": 5.0}])
    out = transfer_labels(annotation("r", 11.0, 3.0), grid)
    assert list(out["window_id"]) == ["w"] and out.loc[0, "y"] == 1


def test_window_inside_a_recording_annotation_inherits_it():
    """Annotation par enregistrement (§5) : toutes ses fenêtres héritent du label."""
    grid = grid_frame(["r"])
    out = transfer_labels(annotation("r", 0.0, 120.0), grid)
    assert len(out) == len(grid) and (out["y"] == 1).all()


def test_merely_overlapping_window_is_discarded():
    """Ni contenante ni contenue : la note pourrait être coupée, on écarte (DECISIONS n° 4)."""
    grid = pd.DataFrame([{"window_id": "w", "recording_id": "r", "offset_s": 10.0, "dur_s": 3.0}])
    assert transfer_labels(annotation("r", 12.0, 3.0), grid).empty


def test_negative_annotation_gives_y_zero():
    grid = grid_frame(["r"])
    out = transfer_labels(annotation("r", 0.0, 120.0, label="bird"), grid)
    assert (out["y"] == 0).all() and (out["label"] == "bird").all()


def test_uncertain_labels_are_excluded():
    grid = grid_frame(["r"])
    for label in ("blanci_uncertain", "uncertain"):
        assert transfer_labels(annotation("r", 0.0, 120.0, label=label), grid).empty


def test_conflicting_annotations_drop_the_window():
    """Une fenêtre annotée positive et négative est écartée, pas arbitrée."""
    grid = pd.DataFrame([{"window_id": "w", "recording_id": "r", "offset_s": 0.0, "dur_s": 3.0}])
    both = pd.concat([annotation("r", 0.0, 3.0), annotation("r", 0.0, 3.0, label="bird")])
    assert transfer_labels(both, grid).empty


def test_agreeing_annotations_are_deduplicated():
    grid = pd.DataFrame([{"window_id": "w", "recording_id": "r", "offset_s": 0.0, "dur_s": 3.0}])
    both = pd.concat([annotation("r", 0.0, 3.0), annotation("r", 0.5, 2.0)])
    out = transfer_labels(both, grid)
    assert len(out) == 1 and out.loc[0, "y"] == 1


def test_annotations_never_cross_recordings():
    grid = grid_frame(["r1"])
    assert transfer_labels(annotation("r2", 0.0, 120.0), grid).empty


# --- Négatifs appariés ------------------------------------------------------------------------


def test_local_minutes_applies_the_guyane_offset():
    """UTC−3 : 13 h UTC = 10 h locales = 600 min."""
    utc = pd.Series(["2026-02-10T13:00:00Z"])
    assert local_minutes(utc, -3).iloc[0] == 600


def test_paired_negatives_share_the_mic_and_the_time_slot(con):
    """Mêmes micro et créneau horaire, autre jour : le fond sonore est le même (§2)."""
    positive = add_recording(con, "2026/mataroni/M1/d10.wav", start_utc="2026-02-10T13:00:00Z")
    same_slot = add_recording(con, "2026/mataroni/M1/d11.wav", start_utc="2026-02-11T13:00:00Z")
    other_hour = add_recording(con, "2026/mataroni/M1/d11b.wav", start_utc="2026-02-11T20:00:00Z")
    other_mic = add_recording(
        con, "2026/mataroni/M9/d11.wav", mic="M9", start_utc="2026-02-11T13:00:00Z"
    )
    grid = grid_frame([positive, same_slot, other_hour, other_mic])
    out = paired_negatives(grid, recordings_table(con), {positive}, per_positive=5)
    assert set(out["recording_id"]) == {same_slot}
    assert len(out) == 5
    assert (out["y"] == 0).all() and (out["label"] == "background_presumed").all()


def test_paired_negatives_tolerance_widens_the_slot(con):
    positive = add_recording(con, "2026/mataroni/M1/a.wav", start_utc="2026-02-10T13:00:00Z")
    near = add_recording(con, "2026/mataroni/M1/b.wav", start_utc="2026-02-11T13:20:00Z")
    grid = grid_frame([positive, near])
    assert paired_negatives(grid, recordings_table(con), {positive}, 3, slot_tolerance_min=5).empty
    assert len(paired_negatives(grid, recordings_table(con), {positive}, 3, 30)) == 3


def test_paired_negatives_never_come_from_a_positive_recording(con):
    positive = add_recording(con, "2026/mataroni/M1/a.wav", start_utc="2026-02-10T13:00:00Z")
    grid = grid_frame([positive])
    assert paired_negatives(grid, recordings_table(con), {positive}, per_positive=5).empty


def test_paired_negatives_are_reproducible(con):
    positive = add_recording(con, "2026/mataroni/M1/a.wav", start_utc="2026-02-10T13:00:00Z")
    other = add_recording(con, "2026/mataroni/M1/b.wav", start_utc="2026-02-11T13:00:00Z")
    grid = grid_frame([positive, other])
    recordings = recordings_table(con)
    first = paired_negatives(grid, recordings, {positive}, 5, seed=3)
    second = paired_negatives(grid, recordings, {positive}, 5, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_paired_negatives_are_not_reused_across_positives(con):
    """Deux positifs au même créneau ne doivent pas se partager les mêmes fenêtres négatives."""
    p1 = add_recording(con, "2026/mataroni/M1/a.wav", start_utc="2026-02-10T13:00:00Z")
    p2 = add_recording(con, "2026/mataroni/M1/b.wav", start_utc="2026-02-11T13:00:00Z")
    neg = add_recording(con, "2026/mataroni/M1/c.wav", start_utc="2026-02-12T13:00:00Z")
    grid = grid_frame([p1, p2, neg])
    out = paired_negatives(grid, recordings_table(con), {p1, p2}, per_positive=4)
    assert out["window_id"].is_unique and len(out) == 8


# --- Jeu d'apprentissage ----------------------------------------------------------------------


def test_training_set_marks_presumed_negatives(con):
    """Les négatifs appariés sont présumés : jamais écrits dans la table labels (§ dataset)."""
    positive = add_recording(con, "2026/mataroni/M1/a.wav", start_utc="2026-02-10T13:00:00Z")
    add_recording(con, "2026/mataroni/M1/b.wav", start_utc="2026-02-11T13:00:00Z")
    add_label(con, add_window(con, positive, 0.0), "blanci_solo")
    grid = grid_frame([r["recording_id"] for _, r in recordings_table(con).iterrows()])
    data = training_set(con, grid, per_positive=5)
    assert data.loc[data["presumed"], "y"].eq(0).all()
    assert data.loc[~data["presumed"], "y"].eq(1).any()
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1


def test_training_set_exposes_rows_points_and_sites(con):
    rid = add_recording(con, "2026/tresor/T1/a.wav", site="tresor", mic="T1")
    add_label(con, add_window(con, rid, 6.0), "blanci_solo")
    grid = grid_frame([rid])
    data = training_set(con, grid)
    assert data.loc[0, "point"] == "tresor/T1" and data.loc[0, "site"] == "tresor"
    assert grid.loc[data.loc[0, "row"], "window_id"] == data.loc[0, "window_id"]


def test_training_set_rows_index_the_embedding_store(con):
    """`row` sert à retrouver l'embedding : il doit pointer la bonne ligne de la grille."""
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    for offset in (0.0, 30.0, 60.0):
        add_label(con, add_window(con, rid, offset), "blanci_solo")
    grid = grid_frame([rid])
    data = training_set(con, grid)
    assert np.array_equal(
        grid.loc[data["row"], "window_id"].to_numpy(), data["window_id"].to_numpy()
    )


def test_training_set_without_paired_negatives_has_only_real_labels(con):
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    add_label(con, add_window(con, rid, 0.0), "blanci_solo")
    data = training_set(con, grid_frame([rid]), per_positive=0)
    assert not data["presumed"].any() and len(data) == 1
