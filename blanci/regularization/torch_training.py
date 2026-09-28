"""Têtes et réseaux entraînés avec torch : boucle `optimise` (R41, R42, R59, R64), dropout (R45,
R46), R43, R47, R61–R63, inversion du gradient (R66), `fit_with_options` (R40, R42)."""

from __future__ import annotations

from typing import Any

import numpy as np

from blanci.regularization.selection import grouped_search, usable_folds

# --- Socle commun des têtes torch (attentive, R85, R66) ----------------------------------------


def import_torch(what: str):
    """torch, ou RuntimeError qui dit quoi installer : `what` s'entraîne avec torch."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise RuntimeError(f"{what} s'entraîne avec torch (uv sync --group research)") from exc
    return torch


def feature_scaling(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(moyenne, écart-type) de chaque dimension sur les lignes de `X`, écart nul → 1, en
    float32 : la standardisation des têtes torch, rangée avec elles pour l'inférence numpy."""
    X = np.asarray(X, dtype=np.float32)
    std = X.std(axis=0)
    return X.mean(axis=0), np.where(std > 0, std, 1.0).astype(np.float32)


def balanced_bce(torch, y: np.ndarray):
    """Entropie croisée binaire où les positifs pèsent n_négatifs / n_positifs : les deux
    classes comptent autant, comme `class_weight="balanced"` de la logistique."""
    n_pos = max(float(np.sum(y)), 1.0)
    return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - n_pos) / n_pos))


# --- R66 : inversion du gradient (DANN) -------------------------------------------------------


def grad_reverse(torch, x, strength: float):
    """R66 : couche d'inversion du gradient (Ganin et al. 2016). À l'aller, l'identité ; au
    retour, le gradient multiplié par −`strength`. Placée entre la représentation et le
    classifieur de micro, elle fait apprendre au classifieur à reconnaître le micro, et à la
    représentation à l'en empêcher."""

    class _Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs):
            return inputs.view_as(inputs)

        @staticmethod
        def backward(ctx, grad):
            return -strength * grad

    return _Reverse.apply(x)


def dann_strength(progress: float, strength: float = 1.0) -> float:
    """R66 : force de l'inversion selon l'avancement p ∈ [0, 1] de l'entraînement,
    λ(p) = strength · (2 / (1 + e^(−10 p)) − 1) : nulle au départ (la tête apprend d'abord le
    chant), pleine ensuite (Ganin et al. 2016)."""
    return float(strength) * (2.0 / (1.0 + np.exp(-10.0 * float(progress))) - 1.0)


# --- Têtes et réseaux entraînés avec torch : R40–R42, R45–R47, R59, R62, R63 ----------------------


def optimise(
    parameters: list,
    decayed: list[bool],
    loss_of,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer: str = "adam",
    validation_loss=None,
    patience: int | None = 20,
    curve: list[float] | None = None,
    warmup: int = 0,
    clip_norm: float | None = None,
    average: str | None = None,
    ema_decay: float = 0.99,
    swa_start: float = 0.75,
) -> int:
    """Descente en lot entier, partagée par l'attentive et la sonde à portes (R85), et par les
    réseaux à venir.

    `loss_of()` : perte d'entraînement (dropout compris) ; weight decay sur les seuls paramètres
    `decayed`. `optimizer` : "adam" (L2 couplée, défaut historique) ou "adamw" (R41).
    `validation_loss()` : critère à minimiser sur des micros tenus à l'écart (R42), relevé à
    chaque époque dans `curve` (liste remplie en place) ; avec `patience`, l'entraînement
    s'arrête `patience` époques après la meilleure, dont les paramètres sont restaurés.
    R59 : `warmup` époques de montée linéaire du pas d'apprentissage, `clip_norm` : norme
    maximale du gradient (écrêtage). R64, `average` : les poids gardés sont une moyenne des
    poids successifs — "ema" : moyenne glissante (poids × `ema_decay` à chaque pas), "swa" :
    moyenne simple à partir de la fraction `swa_start` de l'entraînement ; la validation (R42)
    juge alors ces poids moyens. Renvoie l'époque retenue (la dernière sans validation)."""
    import torch

    kinds = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if optimizer not in kinds:
        raise ValueError(f"optimiseur inconnu : {optimizer!r} (adam, adamw)")
    groups = [
        {
            "params": [p for p, d in zip(parameters, decayed, strict=True) if d],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p, d in zip(parameters, decayed, strict=True) if not d],
            "weight_decay": 0.0,
        },
    ]
    if average not in (None, "ema", "swa"):
        raise ValueError(f"R64 : moyenne inconnue {average!r} (ema, swa)")
    opt = kinds[optimizer]([g for g in groups if g["params"]], lr=lr)
    averager = WeightAverage(parameters, average, ema_decay, swa_start, int(epochs))
    best, best_epoch, best_state = float("inf"), int(epochs), None
    for epoch in range(1, int(epochs) + 1):
        if warmup:
            for group in opt.param_groups:
                group["lr"] = lr * min(1.0, epoch / warmup)
        opt.zero_grad()
        loss = loss_of()
        loss.backward()
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        opt.step()
        averager.update(epoch)
        if validation_loss is None:
            continue
        with torch.no_grad(), averager.swapped():
            current = float(validation_loss())
            if current < best - 1e-7:
                best_state = [p.detach().clone() for p in parameters]
        if curve is not None:
            curve.append(current)
        if current < best - 1e-7:
            best, best_epoch = current, epoch
        elif patience is not None and epoch - best_epoch >= patience:
            break
    if best_state is None:
        averager.apply()
    else:
        with torch.no_grad():
            for p, value in zip(parameters, best_state, strict=True):
                p.copy_(value)
    return best_epoch


class WeightAverage:
    """R64 : moyenne des poids pendant l'entraînement (Izmailov et al. 2018 pour SWA ; EMA).

    À la fin d'un entraînement, les poids oscillent autour d'un minimum (petits lots, pas
    d'apprentissage grand) ; la dernière époque est un point pris au hasard dans ces
    oscillations, la moyenne tombe plus près du centre du creux. Sans `method`, rien ne
    change."""

    def __init__(self, parameters, method, decay: float, start: float, epochs: int):
        self.parameters, self.method, self.decay = parameters, method, float(decay)
        self.start = max(1, int(np.ceil(float(start) * epochs)))
        self.mean = None
        self.count = 0

    def update(self, epoch: int) -> None:
        if self.method is None:
            return
        current = [p.detach().clone() for p in self.parameters]
        if self.method == "ema":
            self.mean = (
                current
                if self.mean is None
                else [
                    self.decay * m + (1.0 - self.decay) * c
                    for m, c in zip(self.mean, current, strict=True)
                ]
            )
        elif epoch >= self.start:  # swa
            self.count += 1
            if self.mean is None:
                self.mean = current
            else:
                self.mean = [
                    m + (c - m) / self.count for m, c in zip(self.mean, current, strict=True)
                ]

    def apply(self) -> None:
        """Remplace les poids par leur moyenne."""
        if self.mean is None:
            return
        for p, m in zip(self.parameters, self.mean, strict=True):
            p.data.copy_(m)

    def swapped(self):
        """Contexte : les poids moyens le temps d'une évaluation, puis les poids courants."""
        from contextlib import contextmanager

        @contextmanager
        def swap():
            if self.mean is None:
                yield
                return
            saved = [p.detach().clone() for p in self.parameters]
            self.apply()
            try:
                yield
            finally:
                for p, v in zip(self.parameters, saved, strict=True):
                    p.data.copy_(v)

        return swap()


# --- R61, R62 : fine-tuning d'un réseau pré-entraîné (`finetune.py`, à écrire) ------------------


def layerwise_lr_groups(
    layers: list, lr: float, decay: float = 0.8, weight_decay: float = 0.0
) -> list[dict]:
    """R61, pas d'apprentissage par couche (LLRD) : la dernière couche de `layers` (de bas en
    haut) reçoit `lr`, celle du dessous lr × decay, puis lr × decay²… Les couches basses,
    génériques, bougent peu ; les hautes, spécialisées, s'adaptent. Groupes pour un optimiseur
    torch."""
    n = len(layers)
    return [
        {
            "params": list(layer.parameters()),
            "lr": lr * decay ** (n - 1 - i),
            "weight_decay": weight_decay,
        }
        for i, layer in enumerate(layers)
    ]


def unfreezing_schedule(n_layers: int, epochs: int, every: int) -> list[int]:
    """R61, dégel progressif (ULMFiT, Howard et Ruder 2018) : nombre de couches du haut qui
    apprennent à chaque époque — 1, puis une de plus toutes les `every` époques."""
    return [min(n_layers, 1 + epoch // max(1, every)) for epoch in range(epochs)]


def unfreeze_top(layers: list, n_top: int) -> None:
    """R61 : seules les `n_top` dernières couches de `layers` apprennent, les autres sont
    gelées."""
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad_(i >= len(layers) - n_top)


def snapshot(model) -> dict:
    """R62 : copie des poids pré-entraînés d'un réseau torch, avant l'adaptation."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def l2_sp_model_penalty(torch, model, reference: dict, alpha: float, beta: float = 0.0):
    """R62, L2-SP complet (Li, Grandvalet et Davoine 2018, « SP » = starting point) :
    α/2 Σ ‖θ − θ⁰‖² sur les poids qui existaient avant l'adaptation (`reference`, de
    `snapshot`, clés = noms des poids dans `model`), β/2 Σ ‖θ‖² sur les poids nouveaux (tête,
    adaptateurs LoRA) ; les poids gelés ne comptent pas."""
    total = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in reference:
            total = total + 0.5 * alpha * ((p - reference[name]) ** 2).sum()
        elif beta:
            total = total + 0.5 * beta * (p**2).sum()
    return total


def validation_criterion(torch, loss_fn, scores_of, y_val: np.ndarray, monitor: str = "loss"):
    """Critère de R42, à minimiser : la perte d'entraînement sur les fenêtres de validation
    ("loss"), ou 1 − AP ("ap"). La perte monte dès que la tête devient trop sûre d'elle, même
    quand son classement s'améliore encore (vu sur données simulées, DECISIONS n° 117 ; le
    choix entre les deux se fera sur la base complète, n° 119)."""
    from blanci.evaluate import average_precision

    y_np = np.asarray(y_val).astype(int)
    target = torch.from_numpy(y_np.astype(np.float32))
    if monitor == "loss":
        return lambda: loss_fn(scores_of(), target)
    if monitor == "ap":
        return lambda: 1.0 - average_precision(y_np, scores_of().numpy())
    raise ValueError(f"R42 : critère inconnu {monitor!r} (loss ou ap)")


def attention_entropy(torch, weights):
    """R43 : entropie moyenne des poids d'attention (fenêtres, jetons), −Σ a log a par
    fenêtre. Maximale (log du nombre de jetons) pour une attention uniforme, la moyenne des
    jetons ; nulle quand tout le poids est sur un jeton, le maximum. Ajoutée à la perte avec un
    coefficient β : β > 0 pousse vers une attention piquée (a priori « la note est brève, un ou
    deux jetons »), β < 0 vers une attention diffuse (chœur, bruit : ne pas s'accrocher à un
    jeton)."""
    return -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1).mean()


def keep_mask(torch, n: int, t: int, p: float):
    """R45 : jetons gardés (n, t), chacun masqué avec la probabilité p, au moins un par
    fenêtre."""
    keep = torch.rand(n, t) >= p
    keep[torch.arange(n), torch.randint(t, (n,))] = True
    return keep


def dropout(torch, x, p: float, train: bool):
    """R46 : dropout (sorties éteintes avec la probabilité p, les autres × 1/(1 − p)), à
    l'entraînement seulement."""
    if train and p > 0:
        return torch.nn.functional.dropout(x, p, training=True)
    return x


def logistic_start(
    Z: np.ndarray, y: np.ndarray, C: float, seed: int = 0
) -> tuple[np.ndarray, float]:
    """R47 : (w₀, b₀) d'une logistique apprise sur Z et réécrite dans l'espace de Z (sans sa
    standardisation) : le point de départ, et la cible du rétrécissement, de l'attentive."""
    from blanci.head import fit_logistic

    head = fit_logistic(Z, np.asarray(y).astype(int), C, seed)
    w0 = head.coef / head.scale
    return w0, float(head.intercept - float(head.coef @ (head.mean / head.scale)))


def l2_sp_penalty(torch, parameters: list, references: list, strength: float):
    """R62 (L2-SP) : ½λ Σ ‖θ − θ_réf‖², l'écart aux poids de référence plutôt qu'à zéro. Les
    poids pré-entraînés pour un réseau adapté (fine-tuning, `finetune.py`) ; ceux de la
    logistique pour l'attentive (R47)."""
    total = 0.0
    for p, ref in zip(parameters, references, strict=True):
        total = total + ((p - ref) ** 2).sum()
    return 0.5 * strength * total


def distillation_loss(torch, student_logits, teacher_logits, temperature: float = 2.0):
    """R63 : perte de la distillation (détecteur distillé, `detectors/distilled.py`). L'élève
    apprend les probabilités de l'enseignant (la chaîne gelée), adoucies par la température T :
    entropie croisée binaire entre σ(élève / T) et σ(enseignant / T), × T² pour garder
    l'échelle des gradients (Hinton et al. 2015). Les labels souples de l'enseignant disent
    aussi « à peu près » et « pas sûr », ce qui régularise l'élève."""
    t = float(temperature)
    target = torch.sigmoid(teacher_logits / t)
    return torch.nn.functional.binary_cross_entropy_with_logits(student_logits / t, target) * t**2


def fit_with_options(
    fit,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    *,
    weight_decays: list[float] | None = None,
    early_stopping: bool = False,
    monitor: str = "ap",
    n_splits: int = 3,
    **options,
):
    """Tête torch `fit` (fit_attentive, fit_gated) entourée de R40 et R42, sur plis groupés.

    R40 (`weight_decays`) : chaque valeur est jugée par l'AP moyenne en validation groupée
    interne (`n_splits` plis, micros entiers, `grouped_search`), la meilleure est retenue. R42
    (`early_stopping`) : dans chaque pli interne, la courbe du critère de validation
    (`monitor` : "loss" ou "ap") est relevée à chaque époque ; l'époque retenue minimise la
    courbe moyenne des plis, et la tête est réentraînée sur tout `X` pour ce nombre d'époques
    (en lot entier, une époque = un pas, quel que soit l'effectif). Un pli unique de 1 à 3
    micros donnait une époque au hasard (données simulées, DECISIONS n° 117). Faute de deux
    micros, ou d'une classe dans un pli, R40/R42 sont sautées.
    """
    from blanci.evaluate import average_precision

    X, y, groups = np.asarray(X), np.asarray(y).astype(int), np.asarray(groups)
    extra: dict[str, Any] = {}
    if weight_decays and len(weight_decays) > 1:

        def score(wd, train, test):
            head = fit_with_options(
                fit,
                X[train],
                y[train],
                groups[train],
                seed,
                early_stopping=early_stopping,
                monitor=monitor,
                n_splits=n_splits,
                **options | {"weight_decay": wd},
            )
            return average_precision(y[test], head.decision(X[test]))

        best, results, _ = grouped_search(
            score, weight_decays, y, groups, n_splits, seed, more_regularized="high"
        )
        if best is not None:
            options["weight_decay"] = best
            extra["weight_decay_cv"] = {str(k): v for k, v in results.items()}

    def with_groups(rows) -> dict:
        """Les têtes qui apprennent aussi le micro (R66, `needs_groups`) reçoivent ses groupes."""
        return {"groups": groups[rows]} if getattr(fit, "needs_groups", False) else {}

    everything = np.arange(len(y))
    folds = usable_folds(y, groups, n_splits, seed) if early_stopping else []
    if folds:
        curves = [
            fit(
                X[train],
                y[train],
                seed=seed,
                validation=(X[test], y[test]),
                patience=None,
                monitor=monitor,
                **options,
                **with_groups(train),
            ).meta["validation_curve"]
            for train, test in folds
        ]
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        best_epoch = int(np.nanargmin(mean_curve)) + 1
        head = fit(X, y, seed=seed, **(options | {"epochs": best_epoch}), **with_groups(everything))
        head.meta |= extra | {
            "early_stopping": {"best_epoch": best_epoch, "monitor": monitor, "folds": len(curves)}
        }
        return head
    head = fit(X, y, seed=seed, **options, **with_groups(everything))
    head.meta |= extra
    return head
