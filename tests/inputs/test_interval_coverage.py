"""L'annotation par intervalles (n° 182) compte partout où comptent les labels de fenêtres :
plis, jeu d'entraînement sans le jeu gelé, négatifs présumés, drapeaux d'écoute, gel."""

import json

import pandas as pd
import pytest

from blanci.core.db import connect
from blanci.inputs.dataset import benchmark_folds, current_intervals, load_spans, training_set
from tests.inputs.test_dataset import _span, add_label, add_recording, add_window, grid_frame


@pytest.fixture
def con(tmp_path):
    return connect(tmp_path / "blanci.sqlite")


def _flags(con, rid):
    row = con.execute("SELECT qc_flags FROM recordings WHERE recording_id = ?", (rid,)).fetchone()
    return json.loads(row[0] or "{}")


# --- Plis (n° 91) ---------------------------------------------------------------------------


def _mic(con, mic):
    return add_recording(con, f"2026/mataroni/{mic}/{mic}.wav", mic=mic)


def test_folds_include_recordings_annotated_only_by_intervals(con):
    """Quatre micros annotés seulement par intervalles : chacun a son pli."""
    for mic, sung in (("M1", True), ("M2", True), ("M3", False), ("M4", False)):
        _span(con, _mic(con, mic), 0.0, 30.0, [(10.0, 10.4, "blanci")] if sung else [])
    folds = benchmark_folds(con, n_splits=2)
    assert set(folds) == {f"mataroni/M{i}" for i in range(1, 5)}


def test_folds_mix_window_labels_and_intervals_and_skip_the_frozen_set(con):
    add_label(con, add_window(con, _mic(con, "M1"), 0.0), "blanci")
    _span(con, _mic(con, "M2"), 0.0, 30.0, [(10.0, 10.4, "blanci")])
    _span(con, _mic(con, "M3"), 0.0, 30.0, [])
    add_label(con, add_window(con, _mic(con, "M4"), 0.0), "bird")
    frozen = _mic(con, "M5")
    _span(con, frozen, 0.0, 30.0, [])
    folds = benchmark_folds(con, n_splits=2, exclude_recordings={frozen})
    assert set(folds) == {f"mataroni/M{i}" for i in range(1, 5)}


# --- Jeu gelé et intervalles ------------------------------------------------------------------


def test_training_set_drops_the_intervals_of_frozen_recordings(con):
    """Un enregistrement gelé annoté par intervalles : ni KeyError, ni fenêtre."""
    kept = add_recording(con, "2026/mataroni/M1/a.wav")
    frozen = add_recording(con, "2026/mataroni/M1/b.wav", start_utc="2026-02-11T13:00:00Z")
    _span(con, kept, 0.0, 30.0, [(10.0, 10.4, "blanci")])
    _span(con, frozen, 0.0, 30.0, [(10.0, 10.4, "blanci")])
    data = training_set(con, grid_frame([kept, frozen]), exclude_recordings={frozen})
    assert set(data["recording_id"]) == {kept} and data["y"].sum() == 2


def test_current_intervals_keeps_intervals_without_their_spans(con):
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    _span(con, rid, 0.0, 30.0, [(10.0, 10.4, "blanci")])
    spans, intervals = load_spans(con)
    assert len(current_intervals(spans.iloc[0:0], intervals)) == 1


# --- Fenêtres « A. blanci ? » et bords : jamais négatifs présumés (n° 182) --------------------


def test_uncertain_and_edge_windows_are_never_presumed_negatives(con):
    """« blanci » 10–11 s et « A. blanci ? » 20–21 s sur l'extrait 0–30 s : les fenêtres
    18,0 et 19,5 s (« A. blanci ? »), les plus proches du chant, ne sont pas tirées."""
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    _span(con, rid, 0.0, 30.0, [(10.0, 11.0, "blanci"), (20.0, 21.0, "blanci_uncertain")])
    grid = grid_frame([rid])
    data = training_set(con, grid, per_positive=200, strategy="nearest")
    presumed = data[data["presumed"]]
    assert not presumed.empty
    assert not {18.0, 19.5} & set(presumed["offset_s"])
    assert 18.0 not in set(data["offset_s"]) and 19.5 not in set(data["offset_s"])


def test_uncertain_window_labels_are_never_presumed_negatives(con):
    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    add_label(con, add_window(con, rid, 30.0), "blanci")
    add_label(con, add_window(con, rid, 60.0), "uncertain")
    data = training_set(con, grid_frame([rid]), per_positive=200, strategy="nearest")
    assert 60.0 not in set(data.loc[data["presumed"], "offset_s"])


# --- Drapeaux posés à l'écoute, lus aussi sur les extraits -------------------------------------


def test_positive_recordings_read_the_intervals(con):
    from blanci.inputs.qc import positive_recordings

    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    other = add_recording(con, "2026/mataroni/M1/b.wav")
    _span(con, rid, 0.0, 30.0, [(10.0, 10.4, "blanci")])
    _span(con, other, 0.0, 30.0, [(10.0, 10.4, "blanci_uncertain")])
    assert positive_recordings(con) == {rid}


def test_a_corrected_span_no_longer_protects_its_recording(con):
    from blanci.inputs.qc import positive_recordings

    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    _span(con, rid, 0.0, 30.0, [(10.0, 10.4, "blanci")])
    _span(con, rid, 0.0, 30.0, [])  # réécoute : pas d'A. blanci
    assert positive_recordings(con) == set()


def test_span_classes_and_comment_tags_raise_the_listening_flags(con):
    from blanci.inputs.qc import annotation_flags, apply_annotation_flags
    from blanci.service import append_span

    bag = add_recording(con, "2026/mataroni/M1/a.wav")
    rain = add_recording(con, "2026/mataroni/M1/b.wav")
    plain = add_recording(con, "2026/mataroni/M1/c.wav")
    append_span(
        con, bag, 0.0, 30.0, [], "artefact_in_bag", source="random",
        classes=["artefact_in_bag"],
    )  # fmt: skip
    append_span(
        con, rain, 0.0, 30.0, [(10.0, 10.4, "blanci")], "background", source="random",
        conditions={"tags": ["rain"], "comment": "pluie"},
    )  # fmt: skip
    append_span(con, plain, 0.0, 30.0, [], "background", source="random")
    assert annotation_flags(con) == {bag: ["in_bag"], rain: ["rain"], plain: []}
    counts = apply_annotation_flags(con, [bag, rain])
    assert counts == {"annotated": 2, "in_bag": 1, "rain": 1}
    assert _flags(con, bag)["annotated"] == ["in_bag"]
    assert _flags(con, plain).get("annotated", []) == []


def test_a_relistened_span_withdraws_its_flag(con):
    from blanci.inputs.qc import annotation_flags
    from blanci.service import append_span

    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    append_span(
        con, rid, 0.0, 30.0, [], "artefact_in_bag", source="random", classes=["artefact_in_bag"]
    )
    append_span(con, rid, 0.0, 30.0, [], "background", source="random")
    assert annotation_flags(con) == {rid: []}


def test_in_bag_heard_on_a_span_excludes_unless_blanci_is_traced(con):
    from blanci.embedding.embed import select_recordings
    from blanci.inputs.qc import apply_annotation_flags
    from blanci.service import append_span

    bag = add_recording(con, "2026/mataroni/M1/a.wav")
    sung = add_recording(con, "2026/mataroni/M1/b.wav")
    for rid in (bag, sung):
        append_span(
            con, rid, 0.0, 30.0, [], "artefact_in_bag", source="random",
            classes=["artefact_in_bag"],
        )  # fmt: skip
    _span(con, sung, 40.0, 60.0, [(50.0, 50.4, "blanci")])
    apply_annotation_flags(con)
    selected = set(select_recordings(con)["recording_id"])
    assert bag not in selected and sung in selected


# --- Gel (§6) : les extraits qui sortent de l'entraînement sont comptés ----------------------


def test_freeze_counts_the_spans_and_intervals_withdrawn(con, tmp_path):
    from blanci.inputs.frozen import freeze

    rid = add_recording(con, "2026/mataroni/M1/a.wav")
    _span(con, rid, 0.0, 30.0, [(10.0, 10.4, "blanci"), (20.0, 20.4, "false_friend")])
    source = tmp_path / "file.csv"
    pd.DataFrame({"recording_id": [rid]}).to_csv(source, index=False)
    cfg = {"paths": {"frozen_test": str(tmp_path / "gele")}}
    _, report = freeze(con, cfg, source, "v1")
    assert report["labels_withdrawn"] == 0
    assert report["spans_withdrawn"] == 1
    assert report["intervals_withdrawn"] == 2 and report["positive_intervals_withdrawn"] == 1
