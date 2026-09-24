"""Voie non supervisée (§5 bis) : HDBSCAN sur ACP, AMI avec les micros, verdict C1."""

import numpy as np
import pandas as pd

from blanci.cluster import NOISE, ami, c0_summary, c1_verdict, cluster_embeddings, cluster_table
from blanci.store import EmbeddingStore

DIM = 32


def blobs(sizes, spread=0.05, seed=0):
    """Amas bien séparés sur des axes orthogonaux (les embeddings sont normalisés)."""
    rng = np.random.default_rng(seed)
    X, labels = [], []
    for k, size in enumerate(sizes):
        center = np.zeros(DIM)
        center[k] = 1.0
        X.append(center + rng.normal(0, spread, (size, DIM)))
        labels += [k] * size
    return np.vstack(X).astype(np.float32), np.array(labels)


def test_hdbscan_recovers_separated_groups():
    X, truth = blobs([60, 60, 60])
    found = cluster_embeddings(X, n_components=10, min_cluster_size=10)
    assert len(set(found) - {NOISE}) == 3
    assert ami(found, truth) > 0.9


def test_ami_says_when_groups_are_just_microphones():
    X, mics = blobs([50, 50, 50])
    found = cluster_embeddings(X, n_components=10, min_cluster_size=10)
    assert c0_summary(found, mics)["ami_mic"] > 0.9  # les groupes = les micros : paysage sonore


def test_c1_passes_when_the_song_forms_its_own_group():
    """Positifs groupés à part, négatifs répartis sur trois micros : C1 réussi."""
    X, groups = blobs([30, 300, 300, 300])
    y = (groups == 0).astype(int)
    mics = np.where(groups == 0, np.arange(len(groups)) % 3 + 1, groups)  # positifs sur 3 micros
    found = cluster_embeddings(X, n_components=10, min_cluster_size=10)
    # Ici les négatifs de chaque micro forment leur groupe : AMI élevé, toléré pour ce test.
    verdict = c1_verdict(found, y, mics, max_ami_mic=1.0)
    assert verdict["best_recall"] > 0.9 and verdict["best_enrichment"] > 20
    assert verdict["passed"]
    assert not c1_verdict(found, y, mics)["passed"]  # seuil par défaut : groupes = micros


def test_c1_fails_when_positives_are_scattered_across_soundscapes():
    """Positifs mêlés aux paysages de leurs micros : aucun groupe ne les rassemble."""
    X, mics = blobs([150, 150, 150])
    y = np.zeros(len(mics), dtype=int)
    y[::15] = 1
    found = cluster_embeddings(X, n_components=10, min_cluster_size=10)
    verdict = c1_verdict(found, y, mics)
    assert verdict["best_recall"] < 0.5 and not verdict["passed"]


def test_cluster_table_reports_recall_enrichment_and_flags():
    labels = np.array([0, 0, 0, 1, 1, 1, NOISE, NOISE])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    mics = np.array(["a", "a", "b", "c", "c", "c", "a", "b"])
    flagged = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    table = cluster_table(labels, mics, y, flagged).set_index("cluster")
    assert table.loc[0, "recall"] == 1.0
    assert table.loc[0, "enrichment"] == (2 / 3) / (2 / 8)
    assert table.loc[1, "top_mic"] == "c" and table.loc[1, "top_mic_share"] == 1.0
    assert table.loc[1, "flagged_share"] == 2 / 3


def test_store_sample_is_proportional_and_reproducible(tmp_path):
    store = EmbeddingStore(tmp_path, "toy-1")
    for site, n in (("a", 300), ("b", 100)):
        meta = pd.DataFrame(
            {"window_id": [f"{site}{i}" for i in range(n)], "recording_id": site, "offset_s": 0.0}
        )
        store.write(meta, np.ones((n, 4)), "2026", site, "202602")
    meta, emb = store.sample(40, seed=1)
    assert len(meta) == 40 and emb.shape == (40, 4)
    assert (meta["recording_id"] == "a").sum() == 30
    again, _ = store.sample(40, seed=1)
    assert again["window_id"].tolist() == meta["window_id"].tolist()
    everything, _ = store.sample(10_000)
    assert len(everything) == 400
