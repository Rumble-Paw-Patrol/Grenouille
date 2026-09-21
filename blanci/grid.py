"""Grille de fenêtres, indépendante de l'encodeur (§4).

Une fenêtre est un couple (offset_s, dur_s). Les décalages sont arrondis au centième,
la précision de `window_id`.
"""

from __future__ import annotations

import math

Window = tuple[float, float]


def window_grid(
    duration_s: float, window_s: float, hop_s: float, tol_s: float = 0.05
) -> list[Window]:
    """Fenêtres pleines de `window_s` tous les `hop_s`, couvrant tout l'enregistrement.

    - Une fenêtre qui déborde de moins de `tol_s` est gardée (l'encodeur complète par des zéros).
    - Si la fin n'est pas couverte, une dernière fenêtre est alignée sur la fin.
    - Un enregistrement plus court qu'une fenêtre donne une seule fenêtre nominale.
    """
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s et hop_s doivent être positifs")
    if hop_s > window_s:
        raise ValueError(f"pas {hop_s} s > fenêtre {window_s} s : la grille aurait des trous")
    if abs(hop_s * 100 - round(hop_s * 100)) > 1e-6:
        raise ValueError("le pas doit être un multiple de 0,01 s (précision de window_id)")
    if duration_s <= window_s + tol_s:
        return [(0.0, window_s)]

    n = math.floor((duration_s - window_s + tol_s) / hop_s + 1e-9) + 1
    offsets = [round(i * hop_s, 2) for i in range(n)]
    if offsets[-1] + window_s < duration_s - tol_s:
        offsets.append(math.floor((duration_s - window_s) * 100) / 100)
    return [(o, window_s) for o in offsets]


def containing_windows(
    start_s: float, end_s: float, windows: list[Window], eps: float = 1e-6
) -> list[int]:
    """Indices des fenêtres qui contiennent entièrement l'intervalle [start_s, end_s]."""
    return [
        i
        for i, (offset, dur) in enumerate(windows)
        if offset <= start_s + eps and offset + dur >= end_s - eps
    ]


def max_hop_without_cut(window_s: float, note_s: float) -> float:
    """Pas maximal garantissant qu'une note de durée `note_s` tient entière dans une fenêtre."""
    return window_s - note_s
