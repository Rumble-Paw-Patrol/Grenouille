"""Poolings des jetons d'un encodeur, pour le linear probe (DECISIONS n° 92).

L'embedding par défaut d'un encodeur résume ses jetons (morceaux temps × fréquence de la
fenêtre), le plus souvent par une moyenne. Une note de 0,09 s dans une fenêtre de 5 s ne tient
que dans un ou deux jetons : la moyenne la dilue, le maximum la garde. Poolings comparés par le
benchmark des têtes (`blanci/head_benchmark.py`) :

- `mean`, `max` : sur tous les jetons ;
- `meanmax` : les deux, concaténés (dimension × 2) ;
- `meanstd` : moyenne et écart-type concaténés (statistics pooling) ;
- `topk` : moyenne des k plus fortes valeurs de chaque dimension (k = 2 : la note occupe un ou
  deux jetons), entre la moyenne et le maximum ;
- `mean_f_max_t`, `max_f_mean_t` : sur une grille temps × fréquence seulement (perch_v2 :
  16 temps × 4 fréquences) — moyenne en fréquence puis maximum en temps, ou l'inverse.

Jetons en (fenêtres, jetons, dim) ou (fenêtres, temps, fréquence, dim). Pour un transformer
dont les jetons arrivent à plat, `as_grid` les remet en grille si l'ordre est connu
(`encoders.models.<nom>.token_grid`, à vérifier sur le code du modèle, notes.md).
"""

from __future__ import annotations

import numpy as np

POOLINGS = ("mean", "max", "meanmax", "meanstd", "topk", "mean_f_max_t", "max_f_mean_t")
GRID_POOLINGS = ("mean_f_max_t", "max_f_mean_t")


def as_grid(tokens: np.ndarray, time: int, freq: int, order: str = "time_major") -> np.ndarray:
    """(n, temps × fréquence, dim) → (n, temps, fréquence, dim).

    `time_major` : les jetons d'un même instant se suivent (t0f0, t0f1, …) ; `freq_major` :
    ceux d'une même bande (t0f0, t1f0, …).
    """
    tokens = np.asarray(tokens)
    n, count, dim = tokens.shape
    if count != time * freq:
        raise ValueError(f"{count} jetons ≠ {time} temps × {freq} fréquences")
    if order == "time_major":
        return tokens.reshape(n, time, freq, dim)
    if order == "freq_major":
        return tokens.reshape(n, freq, time, dim).transpose(0, 2, 1, 3)
    raise ValueError(f"ordre inconnu : {order!r} (time_major ou freq_major)")


def available_poolings(tokens: np.ndarray | None) -> list[str]:
    """Poolings possibles pour ces jetons (aucun sans jetons, sans grille pas de temps × fréq.)."""
    if tokens is None:
        return []
    return (
        list(POOLINGS) if np.ndim(tokens) == 4 else [p for p in POOLINGS if p not in GRID_POOLINGS]
    )


def pool(tokens: np.ndarray, how: str, k: int = 2) -> np.ndarray:
    """Un vecteur par fenêtre, selon `how` (voir le module)."""
    x = np.asarray(tokens, dtype=np.float32)
    if x.ndim == 4:
        if how == "mean_f_max_t":
            return x.mean(axis=2).max(axis=1)
        if how == "max_f_mean_t":
            return x.max(axis=2).mean(axis=1)
        x = x.reshape(len(x), -1, x.shape[-1])
    elif x.ndim != 3:
        raise ValueError(f"jetons attendus en 3 ou 4 dimensions, reçu {x.shape}")
    elif how in GRID_POOLINGS:
        raise ValueError(f"{how} demande une grille temps × fréquence (jetons 4-D)")
    if how == "mean":
        return x.mean(axis=1)
    if how == "max":
        return x.max(axis=1)
    if how == "meanmax":
        return np.concatenate([x.mean(axis=1), x.max(axis=1)], axis=1)
    if how == "meanstd":
        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=1)
    if how == "topk":
        k = max(1, min(k, x.shape[1]))
        return np.sort(x, axis=1)[:, -k:, :].mean(axis=1)
    raise ValueError(f"pooling inconnu : {how!r} (connus : {POOLINGS})")


def flat_tokens(tokens: np.ndarray) -> np.ndarray:
    """(n, temps, fréquence, dim) → (n, temps × fréquence, dim) ; 3-D inchangé (attentive)."""
    tokens = np.asarray(tokens)
    return tokens.reshape(len(tokens), -1, tokens.shape[-1]) if tokens.ndim == 4 else tokens
