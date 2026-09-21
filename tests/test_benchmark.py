import numpy as np
import pandas as pd
import pytest

from blanci.benchmark import (
    benchmark_encoder,
    compare_encoders,
    encoder_params,
    knn_top1,
    probe_table,
    run_benchmark,
    to_markdown,
    write_report,
)
from blanci.config import load_config
from blanci.db import connect, recording_id_for, utc_now, window_id_for
from blanci.store import EmbeddingStore

WINDOW_S = 3.0
HOP_S = 1.5
N_WINDOWS = 8  # 8 fenêtres par enregistrement : assez pour agréger, assez court pour les tests
DIM = 16


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / key) for key in cfg["paths"]}
    cfg["benchmark"]["n_boot"] = 50  # tests rapides
    cfg["benchmark"]["negatives_per_positive"] = 4
    cfg["head"]["n_splits"] = 3
    cfg["head"]["C_grid"] = [1.0]
    return cfg


@pytest.fixture
def con(tmp_path):
    return connect(tmp_path / "blanci.sqlite")


def add_recording(con, rel_path, site, mic, start_utc):
    rid = recording_id_for(rel_path)
    con.execute(
        "INSERT INTO recordings (recording_id, path, dataset, site, mic_id, start_utc, "
        "duration_s, sample_rate, channels, qc_flags) "
        "VALUES (?, ?, '2026', ?, ?, ?, 120.0, 32000, 1, '{}')",
        (rid, rel_path, site, mic, start_utc),
    )
    return rid


def register_encoder(con, encoder_id, dim=DIM, window_s=WINDOW_S, windows_per_s=12.5):
    import json

    params = {
        "sample_rate": 32000,
        "window_s": window_s,
        "hop_s": window_s / 2,
        "dim": dim,
        "has_tokens": False,
        "last_run": {"windows_per_s": windows_per_s, "realtime_factor": windows_per_s * window_s},
    }
    con.execute(
        "INSERT INTO models (model_id, kind, name, version, params_json, created_at) "
        "VALUES (?, 'encoder', ?, '1', ?, ?)",
        (encoder_id, encoder_id.split("-")[0], json.dumps(params), utc_now()),
    )
    con.commit()


def build_recordings(con, n_mics=4, days=4):
    """Corpus : 4 micros × 4 jours, tous au même créneau. Le premier jour de chaque micro
    est positif, les autres fournissent les négatifs appariés.

    Un enregistrement est entièrement positif ou entièrement négatif, comme sur le terrain :
    un fichier mixte rendrait l'AP au niveau enregistrement indéfinie.
    """
    layout = []
    for mic in range(n_mics):
        for day in range(days):
            rel = f"2026/mataroni/M{mic}/d{day}.wav"
            rid = add_recording(
                con, rel, "mataroni", f"M{mic}", f"2026-02-{10 + day:02d}T13:00:00Z"
            )
            offsets = [round(i * HOP_S, 2) for i in range(N_WINDOWS)]
            ids = [window_id_for(rid, o) for o in offsets]
            con.executemany(
                "INSERT INTO windows (window_id, recording_id, offset_s, dur_s) "
                "VALUES (?, ?, ?, ?)",
                [(w, rid, o, WINDOW_S) for w, o in zip(ids, offsets, strict=True)],
            )
            positive = day == 0
            if positive:
                con.executemany(
                    "INSERT INTO labels (window_id, label, source, created_at) "
                    "VALUES (?, 'blanci_solo', 'import', ?)",
                    [(w, utc_now()) for w in ids],
                )
            layout.append((rid, ids, offsets, positive))
    con.commit()
    return layout


def write_embeddings(con, tmp_path, encoder_id, layout, separation, seed=0):
    """Embeddings d'un encodeur pour le corpus. `separation` règle la distance entre positifs
    et négatifs : c'est elle qui distingue un bon encodeur d'un mauvais."""
    rng = np.random.default_rng(seed)
    register_encoder(con, encoder_id)
    store = EmbeddingStore(tmp_path / "embeddings", encoder_id)
    direction = np.zeros(DIM, dtype=np.float32)
    direction[0] = separation

    metas, embs = [], []
    for rid, ids, offsets, positive in layout:
        emb = rng.normal(0, 0.3, (N_WINDOWS, DIM)).astype(np.float32)
        emb += rng.normal(0, 0.2, DIM).astype(np.float32)  # fond propre au micro et au jour
        if positive:
            emb += direction
        metas.append(pd.DataFrame({"window_id": ids, "recording_id": rid, "offset_s": offsets}))
        embs.append(emb)
    store.write(
        pd.concat(metas, ignore_index=True),
        np.concatenate(embs),
        "2026",
        "mataroni",
        "202602",
    )
    return store


# --- Métadonnées d'encodeur -------------------------------------------------------------------


def test_encoder_params_are_read_from_models(con):
    register_encoder(con, "birdmae-1", dim=768, windows_per_s=7.5)
    params = encoder_params(con, "birdmae-1")
    assert params["dim"] == 768
    assert params["last_run"]["windows_per_s"] == 7.5


def test_unknown_encoder_is_rejected(con):
    with pytest.raises(ValueError, match="encodeur inconnu"):
        encoder_params(con, "jamais-encode")


# --- kNN top-1 --------------------------------------------------------------------------------


def test_knn_top1_is_perfect_on_separated_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(5, 0.1, (20, DIM)), rng.normal(-5, 0.1, (20, DIM))])
    y = np.r_[np.ones(20, int), np.zeros(20, int)]
    recordings = np.array([f"r{i}" for i in range(40)])
    assert knn_top1(X, y, recordings) == 1.0


def test_knn_top1_excludes_the_same_recording():
    """Deux fenêtres voisines du même fichier partagent le fond : ce serait une fuite."""
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 0.01, (2, DIM)) + 10  # deux fenêtres identiques d'un même positif
    neg = rng.normal(0, 0.01, (4, DIM))
    X = np.vstack([pos, neg])
    y = np.array([1, 1, 0, 0, 0, 0])
    same_file = np.array(["r0", "r0", "r1", "r2", "r3", "r4"])
    assert knn_top1(X, y, same_file) == 0.0  # le seul autre positif est dans le même fichier
    apart = np.array(["r0", "r9", "r1", "r2", "r3", "r4"])
    assert knn_top1(X, y, apart) == 1.0


def test_knn_top1_is_undefined_without_positives():
    X = np.zeros((4, DIM))
    assert np.isnan(knn_top1(X, np.zeros(4, int), np.array(["a", "b", "c", "d"])))


def test_knn_top1_handles_chunking():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(5, 0.1, (20, DIM)), rng.normal(-5, 0.1, (20, DIM))])
    y = np.r_[np.ones(20, int), np.zeros(20, int)]
    recordings = np.array([f"r{i}" for i in range(40)])
    assert knn_top1(X, y, recordings, chunk_rows=3) == knn_top1(X, y, recordings)


# --- Tableau des sondes -----------------------------------------------------------------------


def probe_case(separation=2.0, n_recordings=60, per_recording=4, seed=0):
    """Enregistrements homogènes : chacun est entièrement positif ou entièrement négatif."""
    rng = np.random.default_rng(seed)
    per_recording_labels = np.tile([1, 0], n_recordings // 2)
    y = np.repeat(per_recording_labels, per_recording)
    recordings = np.repeat([f"rec{i}" for i in range(n_recordings)], per_recording)
    groups = np.repeat([f"mic{i % 6}" for i in range(n_recordings)], per_recording)
    X = rng.normal(0, 0.5, (len(y), DIM)).astype(np.float32)
    X[:, 0] += y * separation
    return X, y, groups, recordings


def test_probe_table_covers_every_probe_and_level():
    X, y, groups, recordings = probe_case()
    table, scores = probe_table(X, y, groups, recordings, n_splits=3, n_boot=20)
    assert len(table) == 6  # 3 sondes × 2 niveaux
    assert set(table["probe"]) == {"knn", "prototype", "logistic"}
    assert set(table["level"]) == {"window", "recording"}
    assert set(scores) == {"knn", "prototype", "logistic"}


def test_probe_table_reports_the_roadmap_metrics():
    X, y, groups, recordings = probe_case()
    table, _ = probe_table(X, y, groups, recordings, n_splits=3, n_boot=20)
    for column in ("ap", "ap_lo", "ap_hi", "recall@p0.1", "recall@p0.5", "n_pos", "n_neg"):
        assert column in table.columns
    assert (table["ap_lo"] <= table["ap"]).all() and (table["ap"] <= table["ap_hi"]).all()


def test_probe_table_separates_an_easy_case():
    X, y, groups, recordings = probe_case(separation=4.0)
    table, _ = probe_table(X, y, groups, recordings, n_splits=3, n_boot=20)
    assert (table["ap"] > 0.9).all()


def test_probe_scores_are_out_of_fold():
    """Les scores servent aux comparaisons appariées : ils doivent couvrir toutes les fenêtres."""
    X, y, groups, recordings = probe_case()
    _, scores = probe_table(X, y, groups, recordings, n_splits=3, n_boot=20)
    for values in scores.values():
        assert values.shape == y.shape and not np.isnan(values).any()


# --- Benchmark de bout en bout ----------------------------------------------------------------


def test_benchmark_encoder_builds_the_full_row(con, tmp_path, cfg):
    write_embeddings(con, tmp_path, "good-1", build_recordings(con), separation=3.0)
    table, scores, y, recordings = benchmark_encoder(con, "good-1", tmp_path / "embeddings", cfg)
    assert set(table["encoder_id"]) == {"good-1"}
    assert len(table) == 6
    assert table["dim"].eq(DIM).all()
    assert table["window_s"].eq(WINDOW_S).all()
    assert table["windows_per_s"].eq(12.5).all()
    assert table["n_mics"].eq(4).all()
    assert not table["knn_top1"].isna().any()
    assert scores.shape == y.shape == recordings.shape


def test_benchmark_uses_paired_negatives(con, tmp_path, cfg):
    """Sans labels négatifs réels, ce sont les négatifs appariés qui font la classe 0 (§2)."""
    write_embeddings(con, tmp_path, "good-1", build_recordings(con), separation=3.0)
    table, _, y, _ = benchmark_encoder(con, "good-1", tmp_path / "embeddings", cfg)
    assert (y == 0).sum() > 0 and (y == 1).sum() > 0
    window_row = table[table["level"] == "window"].iloc[0]
    assert window_row["n_pos"] == int((y == 1).sum())


def test_benchmark_groups_folds_by_mic(con, tmp_path, cfg):
    """Quatre micros, trois plis : la validation est groupée par point (§6)."""
    write_embeddings(con, tmp_path, "good-1", build_recordings(con), separation=3.0)
    table, _, _, _ = benchmark_encoder(con, "good-1", tmp_path / "embeddings", cfg)
    assert table["n_mics"].eq(4).all()


def test_benchmark_rejects_an_encoder_without_embeddings(con, tmp_path, cfg):
    register_encoder(con, "vide-1")
    with pytest.raises(ValueError, match="aucun embedding"):
        benchmark_encoder(con, "vide-1", tmp_path / "embeddings", cfg)


def test_benchmark_rejects_a_single_class_corpus(con, tmp_path, cfg):
    cfg["benchmark"]["negatives_per_positive"] = 0
    write_embeddings(con, tmp_path, "good-1", build_recordings(con), separation=3.0)
    with pytest.raises(ValueError, match="une seule classe"):
        benchmark_encoder(con, "good-1", tmp_path / "embeddings", cfg)


# --- Classement et comparaisons ---------------------------------------------------------------


def test_run_benchmark_ranks_the_better_encoder_first(con, tmp_path, cfg):
    """Deux encodeurs sur les mêmes enregistrements : le mieux séparant doit sortir en tête."""
    layout = build_recordings(con)
    write_embeddings(con, tmp_path, "good-1", layout, separation=4.0, seed=0)
    write_embeddings(con, tmp_path, "weak-1", layout, separation=0.2, seed=0)
    results, comparisons = run_benchmark(con, ["weak-1", "good-1"], tmp_path / "embeddings", cfg)
    window = results[(results["level"] == "window") & (results["probe"] == "logistic")]
    best = window.sort_values("ap", ascending=False).iloc[0]
    assert best["encoder_id"] == "good-1"
    assert len(comparisons) == 1


def test_comparison_detects_a_real_gap(con, tmp_path, cfg):
    layout = build_recordings(con)
    write_embeddings(con, tmp_path, "good-1", layout, separation=4.0, seed=0)
    write_embeddings(con, tmp_path, "weak-1", layout, separation=0.2, seed=0)
    _, comparisons = run_benchmark(con, ["good-1", "weak-1"], tmp_path / "embeddings", cfg)
    row = comparisons.iloc[0]
    assert row["a"] == "good-1" and row["b"] == "weak-1"
    assert row["ap_a"] > row["ap_b"]
    assert row["diff"] > 0


def paired_case(n_recordings=60, per_recording=2, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.tile([1, 0], n_recordings // 2), per_recording)
    recordings = np.repeat([f"rec{i}" for i in range(n_recordings)], per_recording)
    return y + rng.normal(0, 0.5, len(y)), y, recordings


def test_comparison_of_an_encoder_with_itself_is_a_tie():
    scores, y, recordings = paired_case()
    per_encoder = {"a-1": (scores, y, recordings), "b-1": (scores.copy(), y, recordings)}
    cfg = {"benchmark": {"n_boot": 50}, "head": {"seed": 0}}
    row = compare_encoders(per_encoder, cfg).iloc[0]
    assert row["diff"] == 0.0 and not row["significant"]


def test_comparison_is_empty_for_a_single_encoder():
    scores, y, recordings = paired_case()
    cfg = {"benchmark": {"n_boot": 20}, "head": {"seed": 0}}
    assert compare_encoders({"a-1": (scores, y, recordings)}, cfg).empty


def test_comparison_survives_a_single_class_corpus():
    """Tous les enregistrements positifs : AP indéfinie, la comparaison ne doit pas planter."""
    y = np.ones(40, dtype=int)
    recordings = np.repeat([f"rec{i}" for i in range(20)], 2)
    scores = np.arange(40.0)
    per_encoder = {"a-1": (scores, y, recordings), "b-1": (scores[::-1], y, recordings)}
    cfg = {"benchmark": {"n_boot": 20}, "head": {"seed": 0}}
    row = compare_encoders(per_encoder, cfg).iloc[0]
    assert np.isnan(row["diff"]) and not row["significant"]


# --- Rapports ---------------------------------------------------------------------------------


def test_markdown_table_renders_without_tabulate():
    df = pd.DataFrame({"encoder_id": ["a-1"], "ap": [0.8123], "knn_top1": [float("nan")]})
    text = to_markdown(df)
    assert "| encoder_id | ap | knn_top1 |" in text
    assert "0.812" in text
    assert "—" in text  # les NaN ne s'affichent pas en « nan »


def test_markdown_table_handles_an_empty_frame():
    assert to_markdown(pd.DataFrame()) == "_(vide)_\n"


def test_write_report_produces_csv_and_markdown(con, tmp_path, cfg):
    layout = build_recordings(con)
    write_embeddings(con, tmp_path, "good-1", layout, separation=3.0, seed=0)
    write_embeddings(con, tmp_path, "weak-1", layout, separation=0.3, seed=0)
    results, comparisons = run_benchmark(con, ["good-1", "weak-1"], tmp_path / "embeddings", cfg)
    paths = write_report(results, comparisons, tmp_path / "reports")
    assert paths["csv"].exists() and paths["markdown"].exists()
    assert paths["comparisons_csv"].exists()

    reloaded = pd.read_csv(paths["csv"])
    assert len(reloaded) == len(results)

    text = paths["markdown"].read_text(encoding="utf-8")
    assert "## Niveau window" in text and "## Niveau recording" in text
    assert "good-1" in text and "weak-1" in text
    assert "Comparaisons appariées" in text
    assert "licence" in text  # colonnes à renseigner à la main (§2)
