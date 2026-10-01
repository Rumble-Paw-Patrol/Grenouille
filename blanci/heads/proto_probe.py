"""Sonde à prototypes sur les jetons (n° 151) : la tête que proposent les auteurs de Bird-MAE
(« prototypical probing »), proche de celle d'AudioProtoPNet et de Perch 2.0.

K prototypes appris, de la taille d'un jeton ; chaque jeton de la fenêtre est comparé à chaque
prototype (cosinus, après la standardisation des dimensions) ; le maximum sur les jetons donne
K activations, qu'une couche linéaire transforme en score. Une note brève dans un seul jeton
n'est pas diluée par la moyenne de la fenêtre (≠ notre `prototype`, qui est un seul vecteur
sur l'embedding moyen). Version simplifiée : pas de perte d'orthogonalité entre prototypes,
départ sur des jetons de fenêtres positives tirés au hasard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blanci.heads.regularization.torch_training import (
    balanced_bce,
    feature_scaling,
    import_torch,
    optimise,
)


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


@dataclass
class ProtoProbeHead:
    mean: np.ndarray  # (d,)
    scale: np.ndarray  # (d,)
    prototypes: np.ndarray  # (K, d), normés
    weight: np.ndarray  # (K,)
    bias: float
    meta: dict[str, Any] = field(default_factory=dict)

    def activations(self, tokens: np.ndarray) -> np.ndarray:
        """(fenêtres, K) : ressemblance maximale de la fenêtre à chaque prototype."""
        x = _normalise((np.asarray(tokens, dtype=np.float32) - self.mean) / self.scale)
        return np.einsum("ntd,kd->ntk", x, self.prototypes).max(axis=1)

    def decision(self, tokens: np.ndarray) -> np.ndarray:
        return self.activations(tokens) @ self.weight + self.bias


def fit_proto_probe(
    tokens: np.ndarray,
    y: np.ndarray,
    n_prototypes: int = 8,
    weight_decay: float = 1e-3,
    epochs: int = 200,
    lr: float = 5e-2,
    seed: int = 0,
    patience: int | None = 20,
) -> ProtoProbeHead:
    """Entraîne la sonde (lot entier, Adam, entropie croisée à classes équilibrées)."""
    torch = import_torch("la sonde à prototypes")
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 3:
        raise ValueError(f"jetons attendus en (fenêtres, jetons, dim), reçu {tokens.shape}")
    y = np.asarray(y).astype(np.float32)
    mean, scale = feature_scaling(tokens.reshape(-1, tokens.shape[-1]))
    x = torch.from_numpy(_normalise((tokens - mean) / scale))
    target = torch.from_numpy(y)

    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    source = positives if len(positives) else np.arange(len(y))
    start = x[rng.choice(source, n_prototypes), rng.integers(0, x.shape[1], n_prototypes)]
    torch.manual_seed(seed)
    prototypes = start.clone().requires_grad_(True)
    weight = torch.zeros(n_prototypes, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    loss_fn = balanced_bce(torch, y)

    def loss_of():
        p = torch.nn.functional.normalize(prototypes, dim=1)
        activation = torch.einsum("ntd,kd->ntk", x, p).amax(dim=1)
        return loss_fn(activation @ weight + bias, target)

    optimise(
        [prototypes, weight, bias],
        [False, True, True],
        loss_of,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )
    final = torch.nn.functional.normalize(prototypes.detach(), dim=1).numpy()
    return ProtoProbeHead(
        mean.astype(np.float32),
        scale,
        final.astype(np.float32),
        weight.detach().numpy().astype(np.float32),
        float(bias.detach().numpy()[0]),
        {"n_prototypes": n_prototypes, "epochs": epochs, "lr": lr, "n_tokens": tokens.shape[1]},
    )
