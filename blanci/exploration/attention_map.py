"""Carte d'attention : où, dans la fenêtre, la tête attentive a-t-elle écouté ?

La tête attentive (`blanci/heads/attentive.py`) donne un poids à chaque jeton d'une fenêtre. Pour un
encodeur à jetons en grille (perch_v2 : 16 temps × 4 fréquences), chaque jeton couvre un
rectangle du spectrogramme : on peut donc superposer les poids au spectrogramme et juger à
l'œil si la tête écoute la note d'A. blanci (4,4–5,5 kHz) ou le fond.

Une mesure en découle, sans annotation de plus : la **part d'attention dans la bande de la
note**, comparée à ce que donnerait une attention uniforme (la part de la grille que couvre la
bande). Une tête qui a appris la note met sur les positives nettement plus que cette part.

Hypothèses à vérifier par encodeur (notebook 04, cellule « contrôle par un son pur ») :
- l'ordre des jetons à plat est `time_major` (t0f0, t0f1, …), celui de `pooling.as_grid` ;
- l'indice de fréquence 0 est la bande la plus **basse** ;
- les bandes découpent l'axe de l'encodeur en parts égales sur une échelle mel, entre `fmin`
  et `fmax` (perch_v2 : 60 Hz–16 kHz, `[À VÉRIFIER]` dans le code du modèle).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blanci.heads.pooling import as_grid


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


@dataclass(frozen=True)
class TokenGrid:
    """Géométrie des jetons d'un encodeur : combien de pas de temps et de bandes, sur quelle
    plage de fréquences, dans quel ordre."""

    time: int
    freq: int
    fmin_hz: float
    fmax_hz: float
    scale: str = "mel"  # "mel" ou "linear"
    order: str = "time_major"
    low_first: bool = True  # indice de fréquence 0 = bande la plus basse

    def band_edges_hz(self) -> np.ndarray:
        """Bords des `freq` bandes, du bas vers le haut (freq + 1 valeurs)."""
        if self.scale == "mel":
            return mel_to_hz(
                np.linspace(hz_to_mel(self.fmin_hz), hz_to_mel(self.fmax_hz), self.freq + 1)
            )
        if self.scale == "linear":
            return np.linspace(self.fmin_hz, self.fmax_hz, self.freq + 1)
        raise ValueError(f"échelle inconnue : {self.scale!r} (mel ou linear)")

    def time_edges_s(self, window_s: float) -> np.ndarray:
        return np.linspace(0.0, window_s, self.time + 1)


# Géométries connues (`[À VÉRIFIER]` : contrôle par un son pur dans le notebook 04).
GRIDS = {
    "perch_v2": TokenGrid(time=16, freq=4, fmin_hz=60.0, fmax_hz=16_000.0),
}


def attention_grid(weights: np.ndarray, grid: TokenGrid) -> np.ndarray:
    """Poids (fenêtres, jetons) → (fenêtres, temps, fréquence), fréquence du bas vers le haut."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[None, :]
    out = as_grid(w[..., None], grid.time, grid.freq, grid.order)[..., 0]
    return out if grid.low_first else out[:, :, ::-1]


def band_share(
    grid_weights: np.ndarray, grid: TokenGrid, band_hz: tuple[float, float]
) -> np.ndarray:
    """Part d'attention de chaque fenêtre sur les jetons qui recouvrent `band_hz`.

    Un jeton qui ne recouvre la bande qu'en partie compte au prorata du recouvrement (en Hz) :
    avec 4 bandes larges, la note n'occupe qu'une fraction de la bande qui la contient.
    """
    edges = grid.band_edges_hz()
    lo, hi = band_hz
    overlap = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
    frac = overlap / np.diff(edges)  # (freq,)
    w = np.asarray(grid_weights, dtype=float)
    return (w.sum(axis=1) * frac).sum(axis=1)


def uniform_band_share(grid: TokenGrid, band_hz: tuple[float, float]) -> float:
    """Part qu'aurait une attention uniforme (le point de comparaison de `band_share`)."""
    uniform = np.full((1, grid.time, grid.freq), 1.0 / (grid.time * grid.freq))
    return float(band_share(uniform, grid, band_hz)[0])


def peak_ratio(weights: np.ndarray) -> np.ndarray:
    """Poids du jeton le plus lourd × nombre de jetons : 1 = attention uniforme, N = un seul
    jeton. Dit si la tête est piquée (note brève) ou diffuse (≈ moyenne)."""
    w = np.asarray(weights, dtype=float).reshape(len(weights), -1)
    return w.max(axis=1) * w.shape[1]


def plot_attention(
    ax,
    grid_weights: np.ndarray,
    grid: TokenGrid,
    window_s: float,
    color: str = "#E9B44C",
    max_alpha: float = 0.85,
    fmax_hz: float | None = None,
) -> None:
    """Rectangles des jetons sur un spectrogramme en kHz et en s (`explore_plots.show_spectrogram`),
    opacité proportionnelle au poids (le plus lourd à `max_alpha`)."""
    from matplotlib.patches import Rectangle

    w = np.asarray(grid_weights, dtype=float)
    if w.ndim == 3:
        w = w[0]
    top = w.max() or 1.0
    t_edges, f_edges = grid.time_edges_s(window_s), grid.band_edges_hz() / 1000
    if fmax_hz is not None:
        f_edges = np.minimum(f_edges, fmax_hz / 1000)
    for i in range(grid.time):
        for j in range(grid.freq):
            if f_edges[j + 1] <= f_edges[j]:
                continue
            ax.add_patch(
                Rectangle(
                    (t_edges[i], f_edges[j]),
                    t_edges[i + 1] - t_edges[i],
                    f_edges[j + 1] - f_edges[j],
                    color=color,
                    alpha=float(w[i, j] / top) * max_alpha,
                    lw=0,
                )
            )
    for f in f_edges[1:-1]:
        ax.axhline(f, color="white", lw=0.4, alpha=0.5)
