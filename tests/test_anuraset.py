"""Pré-benchmark AnuraSet (§2) sur une mini-archive au format de Zenodo."""

import io
import zipfile

import numpy as np
import pytest
import soundfile as sf

from blanci.anuraset import (
    dominant_frequencies,
    prepare,
    read_strong_labels,
    run_anuraset_benchmark,
    species_profile,
    suggest_species,
    window_labels,
    write_anuraset_report,
)
from blanci.db import connect

SR = 22_050
DURATION_S = 12.0


def _pip(hz, dur=0.12):
    t = np.arange(int(dur * SR)) / SR
    return 0.3 * np.hanning(len(t)) * np.sin(2 * np.pi * hz * t)


@pytest.fixture
def archive(tmp_path):
    """2 sites × 4 minutes : ADEMAR (notes brèves à 4,5 kHz) dans la moitié des fichiers,
    BOAFAB (notes à 1 kHz) ailleurs, un chœur PHYSAU annoté d'un seul tenant."""
    rng = np.random.default_rng(0)
    raw = io.BytesIO()
    labels = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as zr, zipfile.ZipFile(labels, "w") as zl:
        for site, prefix in (("INCT04", "INCT4"), ("INCT17", "INCT17")):
            for k in range(4):
                name = f"{prefix}_2019100{k + 1}_0{k}0000"
                x = rng.normal(0, 0.01, int(SR * DURATION_S))
                lines = []
                species, hz = ("ADEMAR", 4500) if k % 2 == 0 else ("BOAFAB", 1000)
                for start in np.arange(0.5, DURATION_S - 0.5, 1.0):
                    i = int(start * SR)
                    pip = _pip(hz)
                    x[i : i + len(pip)] += pip
                    lines.append(f"{start:.6f}\t{start + 0.12:.6f}\t{species}_M")
                if k == 3:
                    lines.append(f"0.000000\t{DURATION_S:.6f}\tPHYSAU_L")
                buffer = io.BytesIO()
                sf.write(buffer, x.astype(np.float32), SR, format="WAV", subtype="PCM_16")
                zr.writestr(f"raw_data/{site}/{name}.wav", buffer.getvalue())
                zl.writestr(f"strong_labels/{site}/{name}.txt", "\n".join(lines) + "\n")
    base = tmp_path / "anuraset"
    base.mkdir()
    (base / "raw_data.zip").write_bytes(raw.getvalue())
    (base / "strong_labels.zip").write_bytes(labels.getvalue())
    return base


@pytest.fixture
def acfg(cfg, archive, tmp_path):
    cfg["paths"] |= {
        "raw": str(archive),
        "db": str(tmp_path / "anuraset.sqlite"),
        "embeddings": str(tmp_path / "emb"),
        "reports": str(tmp_path / "reports"),
    }
    cfg["qc"] |= {"expected_duration_s": DURATION_S, "campaign_gap_days": 100000}
    cfg["anuraset"] = {
        "archive": str(archive / "raw_data.zip"),
        "labels": str(archive / "strong_labels.zip"),
        "species": ["ADEMAR"],
        "negatives_per_positive": 3,
        "max_call_s": 5.0,
        "profile_per_species": 5,
    }
    cfg["benchmark"]["n_boot"] = 10
    cfg["head"] |= {"n_splits": 2, "C_grid": [1.0]}
    return cfg


def test_prepare_extracts_by_site_and_inventories(acfg, archive):
    con = connect(acfg["paths"]["db"])
    report = prepare(con, acfg)
    assert report == {"extracted": 8, "added": 8, "errors": 0}
    assert (archive / "anuraset" / "INCT04" / "INCT4_20191001_000000.wav").exists()
    sites = {r[0] for r in con.execute("SELECT DISTINCT site FROM recordings")}
    assert sites == {"INCT04", "INCT17"}
    assert prepare(con, acfg)["extracted"] == 0  # reprise : rien à refaire


def test_profile_measures_calls_and_suggests_the_brief_mid_frequency_species(acfg):
    con = connect(acfg["paths"]["db"])
    prepare(con, acfg)
    calls = read_strong_labels(acfg["anuraset"]["labels"])
    assert set(calls["quality"]) == {"medium", "low"}
    profile = species_profile(calls).set_index("species")
    assert profile.loc["ADEMAR", "n_sites"] == 2
    assert profile.loc["ADEMAR", "duration_median_s"] == pytest.approx(0.12)
    assert profile.loc["PHYSAU", "n_long"] == 2 and np.isnan(
        profile.loc["PHYSAU", "duration_median_s"]
    )
    import pandas as pd

    recordings = pd.read_sql_query("SELECT path FROM recordings", con)
    freqs = dominant_frequencies(calls, acfg["paths"]["raw"], recordings, per_species=5)
    assert freqs["ADEMAR"] == pytest.approx(4500, abs=150)
    profile = profile.reset_index().merge(freqs, left_on="species", right_index=True, how="left")
    chosen = suggest_species(profile, min_calls=10)
    assert chosen["species"].tolist() == ["ADEMAR"]


def test_window_labels_keep_whole_calls_and_drop_cut_ones():
    import pandas as pd

    calls = pd.DataFrame(
        {
            "file_key": ["f", "f", "g"],
            "site": "s",
            "start_s": [1.0, 5.8, 0.0],
            "end_s": [1.1, 6.2, 60.0],
            "species": ["A", "A", "A"],
            "quality": "medium",
        }
    )
    windows = pd.DataFrame(
        {
            "file_key": ["f", "f", "f", "h", "g"],
            "offset_s": [0.0, 3.0, 9.0, 0.0, 10.0],
            "dur_s": 3.0,
        }
    )
    y = window_labels(windows, calls, "A")
    assert y[0] == 1 and y[2] == 0 and y[3] == 0  # chant entier ; rien ; autre fichier
    assert np.isnan(y[1])  # chant coupé en 6 s : écarté
    assert y[4] == 1  # dans un chœur annoté d'un seul tenant


def test_benchmark_ranks_an_encoder_by_site_folds(acfg, tmp_path):
    from blanci.embed import embed_recordings, select_recordings
    from tests.test_cli_pipeline import ToyEncoder

    con = connect(acfg["paths"]["db"])
    prepare(con, acfg)
    embed_recordings(
        con, ToyEncoder(), select_recordings(con), acfg["paths"]["raw"], tmp_path / "emb"
    )
    calls = read_strong_labels(acfg["anuraset"]["labels"])
    results, comparisons = run_anuraset_benchmark(con, acfg, ["toy-1"], ["ADEMAR"], calls)
    window = results[(results["level"] == "window") & (results["probe"] == "logistic")]
    assert window["ap"].iloc[0] > 0.9 and window["n_sites"].iloc[0] == 2
    assert comparisons.empty  # un seul encodeur : rien à comparer
    path = write_anuraset_report(results, comparisons, tmp_path / "reports")
    assert "ADEMAR" in path.read_text(encoding="utf-8")
