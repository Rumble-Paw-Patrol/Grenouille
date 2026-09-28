"""R66 : tête DANN, apprentissage adverse contre le micro (Ganin et al. 2016, DECISIONS n° 124).

La version apprise de R21. R21 retire, en une fois et en ligne droite, les directions de
l'embedding qui trahissent le micro. Ici, un petit réseau apprend une représentation
h = tanh(A·x̃ + a) de l'embedding standardisé x̃, avec deux sorties :

- le score A. blanci, w·h + b, qu'on garde ;
- un classifieur de micro, branché sur h à travers une couche d'inversion du gradient
  (`regularization.grad_reverse`) : il apprend à reconnaître le micro, et la représentation,
  qui reçoit son gradient inversé, apprend à l'en empêcher.

Perte = entropie croisée A. blanci (classes équilibrées) + entropie croisée du micro. La force
de l'inversion monte de 0 à `strength` au fil de l'entraînement (`dann_strength`) : la tête
apprend d'abord le chant.

Par défaut, le micro n'est appris que sur les **négatifs** (`domain_on`), comme R21 : la
représentation doit rendre les fonds indiscernables d'un micro à l'autre, sans être poussée à
cacher le chant. Si A. blanci chante surtout sur certains micros, effacer le micro sur toutes
les fenêtres effacerait aussi une partie du chant (le mécanisme du n° 109).

Entraînement : `regularization.optimise` (AdamW), avec R40, R42, R46, R59, R64 par le nom de
la tête (`dann+R42`). Réglages : section `regularization.R66` de la config. Torch (groupe
`research`) ; application en numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DannHead:
    mean: np.ndarray  # (d,)
    scale: np.ndarray  # (d,)
    down: np.ndarray  # A : (d, k)
    down_bias: np.ndarray  # a : (k,)
    weight: np.ndarray  # w : (k,)
    bias: float
    meta: dict[str, Any] = field(default_factory=dict)

    def representation(self, X: np.ndarray) -> np.ndarray:
        """h (n, k) : ce que la tête garde de l'embedding, le micro en moins."""
        x = (np.asarray(X, dtype=np.float32) - self.mean) / self.scale
        return np.tanh(x @ self.down + self.down_bias)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.representation(X) @ self.weight + self.bias


def fit_dann(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    *,
    groups: np.ndarray,
    hidden: int = 32,
    strength: float = 1.0,
    domain_on: str = "negatives",
    weight_decay: float = 1e-2,
    epochs: int = 300,
    lr: float = 1e-2,
    dim_dropout: float = 0.0,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int | None = 20,
    monitor: str = "loss",
    warmup: int = 0,
    clip_norm: float | None = None,
    average: str | None = None,
    ema_decay: float = 0.99,
    swa_start: float = 0.75,
) -> DannHead:
    """Entraîne la tête DANN. `groups` : le micro de chaque fenêtre (le « domaine ») ;
    `domain_on` : "negatives" (défaut) ou "all", les fenêtres sur lesquelles le micro est
    appris ; `strength` : force maximale de l'inversion (0 : pas d'adversaire, un petit réseau
    ordinaire)."""
    from blanci.regularization import (
        balanced_bce,
        dann_strength,
        dropout,
        feature_scaling,
        grad_reverse,
        import_torch,
        optimise,
        validation_criterion,
    )

    torch = import_torch("R66")

    if domain_on not in ("negatives", "all"):
        raise ValueError(f"R66 : domain_on {domain_on!r} (negatives ou all)")
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(np.float32)
    mean, scale = feature_scaling(X)
    x = torch.from_numpy((X - mean) / scale)
    target = torch.from_numpy(y)
    names, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    mask = np.ones(len(y), dtype=bool) if domain_on == "all" else y == 0
    adversary = len(np.unique(codes[mask])) >= 2 and strength > 0
    domain = torch.from_numpy(codes[mask].astype(np.int64))
    rows = torch.from_numpy(np.flatnonzero(mask))
    d, k = x.shape[1], max(1, int(hidden))

    torch.manual_seed(seed)
    down = torch.nn.Parameter(torch.randn(d, k) / d**0.5)
    down_bias = torch.nn.Parameter(torch.zeros(k))
    weight = torch.nn.Parameter(torch.zeros(k))
    bias = torch.nn.Parameter(torch.zeros(1))
    dom_weight = torch.nn.Parameter(torch.randn(k, len(names)) * 0.01)
    dom_bias = torch.nn.Parameter(torch.zeros(len(names)))
    loss_fn = balanced_bce(torch, y)
    domain_loss = torch.nn.CrossEntropyLoss()

    def represent(inputs):
        return torch.tanh(inputs @ down + down_bias)

    def logits(inputs, train: bool):
        return dropout(torch, represent(inputs), dim_dropout, train) @ weight + bias

    step = {"epoch": 0}

    def loss_of():
        step["epoch"] += 1
        h = represent(x)
        loss = loss_fn(dropout(torch, h, dim_dropout, True) @ weight + bias, target)
        if adversary:
            lam = dann_strength(step["epoch"] / max(int(epochs), 1), strength)
            reversed_h = grad_reverse(torch, h[rows], lam)
            loss = loss + domain_loss(reversed_h @ dom_weight + dom_bias, domain)
        return loss

    validation_loss = None
    if validation is not None:
        x_val = torch.from_numpy((np.asarray(validation[0], dtype=np.float32) - mean) / scale)
        validation_loss = validation_criterion(
            torch, loss_fn, lambda: logits(x_val, False), validation[1], monitor
        )
    parameters = [down, down_bias, weight, bias, dom_weight, dom_bias]
    curve: list[float] = []
    best_epoch = optimise(
        parameters,
        [True] * len(parameters),
        loss_of,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        optimizer="adamw",
        validation_loss=validation_loss,
        patience=patience,
        curve=curve,
        warmup=warmup,
        clip_norm=clip_norm,
        average=average,
        ema_decay=ema_decay,
        swa_start=swa_start,
    )

    def numpy(t) -> np.ndarray:
        return t.detach().numpy().astype(np.float32)

    meta: dict[str, Any] = {
        "hidden": k,
        "strength": strength,
        "domain_on": domain_on,
        "n_domains": int(len(names)),
        "adversary": bool(adversary),
        "weight_decay": weight_decay,
        "epochs": epochs,
        "lr": lr,
    }
    if validation is not None:
        meta |= {"best_epoch": best_epoch, "validation_curve": curve}
    return DannHead(
        mean.astype(np.float32),
        scale,
        numpy(down),
        numpy(down_bias),
        numpy(weight),
        float(bias.detach().numpy()[0]),
        meta,
    )


fit_dann.needs_groups = True  # `regularization.fit_with_options` lui passe les micros
