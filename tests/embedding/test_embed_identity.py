"""Identité d'un stock rangée dès le premier passage (n° 143) ; micro dans sac jugé par suites
(n° 194) pendant `embed`, `ingest --qc` et `blanci qc`."""

import json

import pytest

from blanci.core.config import load_config
from blanci.core.db import connect
from blanci.embedding.embed import (
    check_stock_identity,
    embed_recordings,
    select_recordings,
    stock_identity,
)
from blanci.embedding.store import EmbeddingStore
from blanci.inputs.qc import check_recordings, parse_flags
from tests.embedding.test_embed import FakeEncoder, add_recording, recordings_of

QC = load_config()["qc"]
ALL_CANDIDATES = QC | {"in_bag_hf_ratio": 2.0}  # tout enregistrement est un candidat


@pytest.fixture
def workspace(tmp_path):
    return connect(tmp_path / "blanci.sqlite"), tmp_path / "raw", tmp_path / "embeddings"


def _flags(con, rid):
    row = con.execute("SELECT qc_flags FROM recordings WHERE recording_id = ?", (rid,)).fetchone()
    return parse_flags(row[0])


# --- Identité du stock ------------------------------------------------------------------------


def test_an_interrupted_first_run_already_holds_the_identity(workspace):
    """Premier passage coupé après un lot : la reprise avec un autre canal est refusée."""
    con, raw, store_root = workspace
    for i in range(3):
        add_recording(con, raw, f"r{i}.wav")

    class Failing(FakeEncoder):
        def _forward(self, batch):
            if self.calls == 1:
                raise RuntimeError("coupure")
            return super()._forward(batch)

    with pytest.raises(RuntimeError, match="coupure"):
        embed_recordings(
            con, Failing(), recordings_of(con), raw, store_root, flush_every=1, channel=0
        )
    assert EmbeddingStore(store_root, "fake-1").load()[0]["recording_id"].nunique() == 1
    with pytest.raises(ValueError, match="channel"):
        embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root, channel=1)
    report = embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root, channel=0)
    assert report.recordings == 2
    params = json.loads(con.execute("SELECT params_json FROM models").fetchone()[0])
    assert params["totals"]["runs"] == 1 and params["last_run"]["recordings"] == 2


def test_the_identity_holds_the_window_length(workspace):
    """Une fenêtre avex changée dans la config : même nom de stock, grille différente."""
    con, raw, store_root = workspace
    add_recording(con, raw, "a.wav")
    embed_recordings(con, FakeEncoder(), recordings_of(con), raw, store_root)

    class Longer(FakeEncoder):
        window_s = 5.0

    with pytest.raises(ValueError, match="window_s"):
        embed_recordings(con, Longer(), recordings_of(con), raw, store_root)


def test_openvino_precision_and_gate_settings_belong_to_the_identity():
    from blanci.heads.signal_processing import Upstream

    encoder = FakeEncoder()
    assert "openvino_precision" not in stock_identity(encoder, 0)
    encoder.openvino = {"device": "GPU", "precision": "f16"}
    assert stock_identity(encoder, 0)["openvino_precision"] == "f16"
    gates = Upstream(gates={"notes": 1.0}, signal_cfg={"band_hz": (4400, 5500)})
    assert stock_identity(FakeEncoder(), 0, gates=gates)["gate_signal"] == {"band_hz": [4400, 5500]}
    assert "gate_signal" not in stock_identity(FakeEncoder(), 0, gates=Upstream())


def test_an_old_identity_without_the_new_keys_is_still_accepted(workspace):
    """Stock rangé avant ces réglages : précision absente = f32 (valeur d'alors), portes
    absentes = non comparées ; une précision f16 est refusée."""
    con, *_ = workspace
    con.execute(
        "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
        "VALUES ('fake-1', 'encoder', 'fake', '1', ?, 'x')",
        (json.dumps({"channel": 0, "window_s": 3.0, "resample": "recording"}),),
    )
    encoder = FakeEncoder()
    encoder.openvino = {"precision": "f32"}
    check_stock_identity(con, "fake-1", stock_identity(encoder, 0, "recording"))
    encoder.openvino = {"precision": "f16"}
    with pytest.raises(ValueError, match="openvino_precision"):
        check_stock_identity(con, "fake-1", stock_identity(encoder, 0, "recording"))


# --- Micro dans sac : règle des suites (n° 194) ----------------------------------------------


def test_embed_encodes_an_isolated_in_bag_candidate(workspace):
    """Un enregistrement grave isolé n'est qu'un candidat : encodé, jamais écarté."""
    con, raw, store_root = workspace
    rid = add_recording(con, raw, "grave.wav")
    report = embed_recordings(
        con, FakeEncoder(), select_recordings(con), raw, store_root, qc_thresholds=ALL_CANDIDATES
    )
    assert (report.recordings, report.qc_checked, report.qc_excluded) == (1, 1, 0)
    assert _flags(con, rid)["in_bag"] is False
    assert rid in set(select_recordings(con)["recording_id"])


def test_embed_flags_a_run_of_in_bag_candidates_at_the_end_of_the_pass(workspace):
    con, raw, store_root = workspace
    ids = [
        add_recording(con, raw, f"r{i}.wav", start=f"2026-02-10T{13 + i:02d}:00:00Z")
        for i in range(4)
    ]
    embed_recordings(
        con, FakeEncoder(), select_recordings(con), raw, store_root,
        qc_thresholds=ALL_CANDIDATES | {"in_bag_max_gap_min": 90},
    )  # fmt: skip
    assert all(_flags(con, rid)["in_bag"] for rid in ids)
    assert select_recordings(con).empty


def test_qc_pass_does_not_count_in_bag_candidates_as_excluded(workspace):
    con, raw, _ = workspace
    add_recording(con, raw, "grave.wav")
    report = check_recordings(con, recordings_of(con), raw, ALL_CANDIDATES, workers=1)
    assert (report["checked"], report["excluded"]) == (1, 0)


def test_ingest_qc_applies_the_run_rule(tmp_path, cfg):
    from blanci.core.db import connect
    from blanci.inputs.ingest import ingest
    from tests.conftest import write_wav

    root = tmp_path / "raw"
    write_wav(root / "2026" / "Mataroni" / "M01" / "SMM01_20260212_070000.wav", duration_s=5.0)
    cfg["qc"] = cfg["qc"] | {"in_bag_hf_ratio": 2.0, "expected_duration_s": 5.0}
    con = connect(tmp_path / "db.sqlite")
    ingest(con, root, "2026", cfg, run_qc=True, hash_file=False)
    (qc,) = con.execute("SELECT qc_flags FROM recordings").fetchone()
    flags = json.loads(qc)
    assert "indices" in flags and flags["in_bag"] is False
    assert len(select_recordings(con)) == 1
