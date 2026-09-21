"""Recherche par similarité cosinus, exhaustive, partition par partition (§4).

Score d'une fenêtre = similarité maximale aux requêtes positives, moins la similarité
maximale aux requêtes négatives si elles sont fournies (requêtes empilées).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from blanci.store import EmbeddingStore


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.arange(len(scores))
    return np.argpartition(-scores, k - 1)[:k]


def search(
    queries: np.ndarray,
    store: EmbeddingStore,
    k: int,
    filters: dict | None = None,
    negatives: np.ndarray | None = None,
    chunk_rows: int = 200_000,
) -> pd.DataFrame:
    """Les k fenêtres les mieux classées de la partie du stock retenue par `filters`.

    Colonnes : rank, window_id, recording_id, offset_s, score, sim_pos, sim_neg, query.
    """
    q = l2_normalize(queries)
    n = l2_normalize(negatives) if negatives is not None and len(negatives) else None
    candidates = []
    for path in store.fragments(filters):
        meta, emb = store.read(path)
        for start in range(0, len(emb), chunk_rows):
            x = l2_normalize(emb[start : start + chunk_rows])
            sims = x @ q.T
            best_query = sims.argmax(axis=1)
            sim_pos = sims[np.arange(len(x)), best_query]
            sim_neg = (x @ n.T).max(axis=1) if n is not None else np.zeros(len(x), np.float32)
            score = sim_pos - sim_neg
            keep = _top_k(score, k)
            part = meta.iloc[start + keep].reset_index(drop=True)
            part["score"] = score[keep]
            part["sim_pos"] = sim_pos[keep]
            part["sim_neg"] = sim_neg[keep]
            part["query"] = best_query[keep]
            candidates.append(part)

    columns = ["rank", "window_id", "recording_id", "offset_s", "score", "sim_pos", "sim_neg", "query"]
    if not candidates:
        return pd.DataFrame(columns=columns)
    out = pd.concat(candidates, ignore_index=True)
    out = out.sort_values("score", ascending=False, kind="stable").head(k).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[columns]
