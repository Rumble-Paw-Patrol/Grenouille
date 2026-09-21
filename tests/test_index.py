import numpy as np
import pandas as pd
import pytest

from blanci.index import search
from blanci.store import EmbeddingStore

DIM = 16


def meta_for(prefix: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": [f"{prefix}:{i * 1.5:.2f}" for i in range(n)],
            "recording_id": prefix,
            "offset_s": np.arange(n) * 1.5,
        }
    )


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(0)
    store = EmbeddingStore(tmp_path / "embeddings", "fake")
    for site in ("mataroni", "tresor"):
        store.write(meta_for(site, 50), rng.normal(size=(50, DIM)), "2026", site, "202602")
    return store


def test_roundtrip_keeps_float16_and_metadata(tmp_path):
    store = EmbeddingStore(tmp_path, "fake")
    emb = np.random.default_rng(1).normal(size=(10, DIM)).astype(np.float32)
    path = store.write(meta_for("r", 10), emb, "2026", "kaw", "202601")
    assert path == tmp_path / "fake" / "2026" / "kaw" / "202601.parquet"
    meta, back = store.read(path)
    assert back.dtype == np.float16 and back.shape == (10, DIM)
    np.testing.assert_allclose(back, emb, atol=1e-2)
    pd.testing.assert_frame_equal(meta, meta_for("r", 10))


def test_rewriting_a_window_replaces_it(tmp_path):
    store = EmbeddingStore(tmp_path, "fake")
    store.write(meta_for("r", 3), np.zeros((3, DIM)), "2026", "kaw", "202601")
    store.write(meta_for("r", 1), np.ones((1, DIM)), "2026", "kaw", "202601")
    meta, emb = store.load()
    assert len(meta) == 3 and meta["window_id"].is_unique
    assert emb[meta["window_id"] == "r:0.00"].sum() == DIM


def test_search_finds_planted_neighbours(store):
    _, emb = store.load({"site": "tresor"})
    queries = emb[[3, 17]].astype(np.float32) + 0.01
    result = search(queries, store, k=2)
    assert set(result["window_id"]) == {"tresor:4.50", "tresor:25.50"}
    assert list(result["rank"]) == [1, 2]
    assert (result["score"] > 0.99).all()


def test_search_respects_partition_filters(store):
    queries = np.random.default_rng(2).normal(size=(3, DIM))
    result = search(queries, store, k=20, filters={"site": "mataroni"})
    assert (result["recording_id"] == "mataroni").all()


def test_negative_queries_demote_their_neighbours(store):
    _, emb = store.load({"site": "mataroni"})
    positive, negative = emb[[0]].astype(np.float32), emb[[1]].astype(np.float32)
    query = positive + negative  # proche des deux fenêtres
    plain = search(query, store, k=100)
    contrasted = search(query, store, k=100, negatives=negative)
    rank = lambda df, wid: int(df.loc[df["window_id"] == wid, "rank"].iloc[0])  # noqa: E731
    assert rank(contrasted, "mataroni:1.50") > rank(plain, "mataroni:1.50")
    assert contrasted.iloc[0]["window_id"] == "mataroni:0.00"


def test_chunking_does_not_change_results(store):
    queries = np.random.default_rng(3).normal(size=(4, DIM))
    a = search(queries, store, k=15)
    b = search(queries, store, k=15, chunk_rows=7)
    pd.testing.assert_frame_equal(a, b)


def test_k_larger_than_store_returns_everything(store):
    result = search(np.ones((1, DIM)), store, k=1000)
    assert len(result) == 100 and list(result["rank"]) == list(range(1, 101))
    assert result["score"].is_monotonic_decreasing


def test_empty_store_and_unknown_filter(tmp_path):
    empty = EmbeddingStore(tmp_path, "none")
    assert search(np.ones((1, DIM)), empty, k=5).empty
    with pytest.raises(ValueError):
        list(empty.fragments({"mic": "M01"}))
