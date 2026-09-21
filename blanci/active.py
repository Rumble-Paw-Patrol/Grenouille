"""Files de vérification pour l'apprentissage actif (§5).

File = 60 % incertains, 20 % scores maximaux, 20 % aléatoire stratifié (micro, heure). La strate
aléatoire mesure les faux négatifs confiants, dont ceux que Biophonia n'avait jamais remontés.
Unité : l'enregistrement (lots de 30–50).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_queue(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    n: int,
    mix: tuple[float, float, float] = (0.6, 0.2, 0.2),
    threshold: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """File de `n` enregistrements non encore étiquetés.

    scores : recording_id, score, mic_id, hour (heure locale) ; labels : recording_id déjà vus.
    `threshold` : seuil de décision courant (0 pour un logit), les incertains en sont les plus
    proches. Colonne `reason` ∈ {uncertain, top, random}.
    """
    if abs(sum(mix) - 1) > 1e-9:
        raise ValueError("les proportions de la file doivent sommer à 1")
    rng = np.random.default_rng(seed)
    pool = scores[~scores["recording_id"].isin(labels["recording_id"])].copy()
    n = min(n, len(pool))
    n_unc, n_top = round(n * mix[0]), round(n * mix[1])
    n_rand = n - n_unc - n_top

    pool["_dist"] = (pool["score"] - threshold).abs()
    uncertain = pool.nsmallest(n_unc, "_dist").assign(reason="uncertain")
    pool = pool.drop(uncertain.index)
    top = pool.nlargest(n_top, "score").assign(reason="top")
    pool = pool.drop(top.index)
    random = _stratified_sample(pool, n_rand, rng).assign(reason="random")

    queue = pd.concat([uncertain, top, random]).drop(columns="_dist")
    return queue.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # ordre mélangé


def _stratified_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Tirage à parts égales entre strates (micro, heure), puis au hasard dans chaque strate."""
    if n <= 0 or pool.empty:
        return pool.iloc[:0]
    strata = [idx.to_numpy() for _, idx in pool.groupby(["mic_id", "hour"]).groups.items()]
    for s in strata:
        rng.shuffle(s)
    chosen: list = []
    depth = 0
    while len(chosen) < n:
        layer = [s[depth] for s in strata if depth < len(s)]
        if not layer:
            break
        rng.shuffle(layer)
        chosen.extend(layer[: n - len(chosen)])
        depth += 1
    return pool.loc[chosen]
