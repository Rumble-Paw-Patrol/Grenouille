"""Contrat des encodeurs (§13.4) et base commune : rééchantillonnage et lots.

Convention : `embed(wav, sr)` reçoit un lot de fenêtres déjà découpées sur la grille, à la
f_e native de l'enregistrement, de forme (n_windows, n_samples) — ou une seule fenêtre 1-D.
Le rééchantillonnage vers `sample_rate` se fait ici, jamais chez l'appelant (§13.7).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from blanci.audio import resample


@runtime_checkable
class Encoder(Protocol):
    name: str
    version: str
    sample_rate: int
    window_s: float
    dim: int
    has_tokens: bool

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:  # (n_windows, dim)
        ...

    def embed_tokens(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        ...  # (n_windows, n_tokens, dim)


def encoder_id(encoder: Encoder) -> str:
    return f"{encoder.name}-{encoder.version}"


class BaseEncoder:
    """Implémente `embed` à partir de `_forward` (lot à la f_e du modèle → embeddings)."""

    name: str
    version: str
    sample_rate: int
    window_s: float
    dim: int
    has_tokens: bool = False
    batch_size: int = 64

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _forward_tokens(self, batch: np.ndarray) -> np.ndarray | None:
        return None

    def _prepare(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.atleast_2d(np.asarray(wav, dtype=np.float32))
        if sr != self.sample_rate:
            wav = np.stack([resample(w, sr, self.sample_rate) for w in wav])
        n = round(self.window_s * self.sample_rate)
        if wav.shape[1] != n:  # écart d'arrondi du rééchantillonnage ou fenêtre plus courte
            fixed = np.zeros((len(wav), n), dtype=np.float32)
            fixed[:, : min(n, wav.shape[1])] = wav[:, :n]
            wav = fixed
        return wav

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        x = self._prepare(wav, sr)
        out = [self._forward(x[i : i + self.batch_size]) for i in range(0, len(x), self.batch_size)]
        emb = np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)
        return emb.astype(np.float32, copy=False)

    def embed_tokens(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        if not self.has_tokens:
            return None
        x = self._prepare(wav, sr)
        return np.concatenate(
            [self._forward_tokens(x[i : i + self.batch_size]) for i in range(0, len(x), self.batch_size)]
        )
