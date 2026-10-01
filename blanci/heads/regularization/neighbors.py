"""R39 : similarité aux k plus proches références."""

from __future__ import annotations

import numpy as np

# --- R39 : k plus proches voisins ----------------------------------------------------------------


def nearest_similarity(sims: np.ndarray, k: int = 1, weighted: bool = False) -> np.ndarray:
    """Similarité de chaque ligne à ses k références les plus proches (R39, DECISIONS n° 115),
    appelée par les têtes `knn:k=…` et `exemplar:k=…` (`head.py`).

    k = 1 : le plus proche seul. k > 1 : moyenne des k plus proches, plus lisse, moins sensible
    à une référence bizarre. `weighted` : moyenne pondérée par 1 / distance (distance
    euclidienne entre vecteurs de norme 1, √(2 − 2·cos), comme `weights="distance"` de
    scikit-learn) : les voisins très proches comptent davantage, entre k = 1 et la moyenne."""
    k = max(1, min(int(k), sims.shape[1]))
    top = -np.partition(-sims, k - 1, axis=1)[:, :k] if k > 1 else sims.max(axis=1)[:, None]
    if not weighted:
        return top.mean(axis=1)
    w = 1.0 / np.maximum(np.sqrt(np.clip(2.0 - 2.0 * top, 0.0, None)), 1e-6)
    return (w * top).sum(axis=1) / w.sum(axis=1)
