"""R85 : sonde à portes (« attentive sur l'embedding », idée de Léonard, DECISIONS n° 110).

La logistique donne un poids fixe w_j à chaque dimension de l'embedding. Ici le poids dépend du
son entendu : une porte g(x) ∈ ]0, 1[ par dimension, calculée depuis la fenêtre elle-même,
laisse passer ou éteint chaque dimension avant le classement :

    x̃ = (x − moyenne) / écart-type,  g = σ(B·(A·x̃) + c),  score = w · (x̃ ⊙ g) + b

A (r × d) et B (d × r) : rang r faible (8), sinon d² paramètres pour ~50 enregistrements
positifs. Départ : B = 0, c = 0 → g = ½ partout, la tête part d'une logistique ; la porte ne
s'écarte de ½ que si les données le demandent. Paramètres : 2·r·d + 2·d + 1 (≈ 28 000 pour
perch_v2 à r = 8), contre d + 1 pour la logistique : weight decay et peu d'époques.
Entraînement avec torch (groupe `research`), application en numpy, comme l'attentive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@dataclass
class GatedHead:
    mean: np.ndarray  # (d,)
    scale: np.ndarray  # (d,)
    down: np.ndarray  # A : (d, r)
    up: np.ndarray  # B : (r, d)
    gate_bias: np.ndarray  # c : (d,)
    weight: np.ndarray  # w : (d,)
    bias: float
    meta: dict[str, Any] = field(default_factory=dict)

    def gates(self, X: np.ndarray) -> np.ndarray:
        """Porte de chaque dimension, fenêtre par fenêtre (n, d) : ce que la tête écoute."""
        x = (np.asarray(X, dtype=np.float32) - self.mean) / self.scale
        return _sigmoid((x @ self.down) @ self.up + self.gate_bias)

    def decision(self, X: np.ndarray) -> np.ndarray:
        x = (np.asarray(X, dtype=np.float32) - self.mean) / self.scale
        return (x * _sigmoid((x @ self.down) @ self.up + self.gate_bias)) @ self.weight + self.bias


def fit_gated(
    X: np.ndarray,
    y: np.ndarray,
    rank: int = 8,
    weight_decay: float = 1e-2,
    epochs: int = 300,
    lr: float = 5e-2,
    seed: int = 0,
    *,
    dim_dropout: float = 0.0,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int | None = 20,
    monitor: str = "loss",
) -> GatedHead:
    """Entraîne la sonde (lot entier, AdamW, entropie croisée à classes équilibrées).

    `dim_dropout` : R46, dropout de x̃ ⊙ g ; `validation`, `patience` et `monitor` : arrêt
    précoce (R42, `attentive.fit_with_options`)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise RuntimeError("R85 s'entraîne avec torch (uv sync --group research)") from exc

    from blanci.attentive import optimise, validation_criterion

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    scale = np.where(std > 0, std, 1.0).astype(np.float32)
    x = torch.from_numpy((X - mean) / scale)
    target = torch.from_numpy(y)
    d = x.shape[1]
    r = max(1, min(int(rank), d))

    torch.manual_seed(seed)
    down = torch.nn.Parameter(torch.randn(d, r) / d**0.5)
    up = torch.nn.Parameter(torch.zeros(r, d))  # départ : porte ½ partout (logistique)
    gate_bias = torch.nn.Parameter(torch.zeros(d))
    weight = torch.nn.Parameter(torch.zeros(d))
    bias = torch.nn.Parameter(torch.zeros(1))
    n_pos = max(float(y.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - n_pos) / n_pos))

    def logits(inputs, train: bool):
        gated = inputs * torch.sigmoid((inputs @ down) @ up + gate_bias)
        if train and dim_dropout > 0:
            gated = torch.nn.functional.dropout(gated, dim_dropout, training=True)
        return gated @ weight + bias

    validation_loss = None
    if validation is not None:
        x_val = torch.from_numpy((np.asarray(validation[0], dtype=np.float32) - mean) / scale)
        validation_loss = validation_criterion(
            torch, loss_fn, lambda: logits(x_val, False), validation[1], monitor
        )

    parameters = [down, up, gate_bias, weight, bias]
    curve: list[float] = []
    best_epoch = optimise(
        parameters,
        [True] * len(parameters),
        lambda: loss_fn(logits(x, True), target),
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        optimizer="adamw",
        validation_loss=validation_loss,
        patience=patience,
        curve=curve,
    )

    def numpy(t) -> np.ndarray:
        return t.detach().numpy().astype(np.float32)

    meta: dict[str, Any] = {"rank": r, "weight_decay": weight_decay, "epochs": epochs, "lr": lr}
    if dim_dropout:
        meta["dim_dropout"] = dim_dropout
    if validation is not None:
        meta |= {"best_epoch": best_epoch, "validation_curve": curve}
    return GatedHead(
        mean.astype(np.float32),
        scale,
        numpy(down),
        numpy(up),
        numpy(gate_bias),
        numpy(weight),
        float(bias.detach().numpy()[0]),
        meta,
    )
