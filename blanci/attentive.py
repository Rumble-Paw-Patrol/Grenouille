"""Attentive probing (§3) : tête d'attention sur les jetons d'un encodeur, avant agrégation.

L'embedding d'une fenêtre est la moyenne de ses jetons (morceaux de temps × fréquence). Une
note de 0,1 s dans une fenêtre de 5 s ne tient que dans un ou deux jetons : la moyenne la
dilue dans le fond. La tête d'attention apprend une requête q qui pondère les jetons :

    a_t = softmax_t(x̃_t · q / √d),  z = Σ_t a_t x̃_t,  score = w · z + b

(x̃ : jetons standardisés). Une requête, un vecteur de classement : d·2 + 1 paramètres, peu
pour ~150–200 positifs (§3 : « ≥ 150–200 annotations »). Entraînement avec torch (groupe
`research`), application en numpy ; sauvegarde JSON + npz, sans pickle (§7).

Régularisations (tri de Léonard du 26/09, DECISIONS n° 117), coupées par défaut, activées dans
le nom de la tête (`attentive+R41+R42`) :

| R | quoi |
|---|---|
| R40 | weight decay choisi par validation groupée sur une grille (`fit_with_options`) |
| R41 | AdamW (weight decay découplé) au lieu de la L2 d'Adam |
| R42 | arrêt précoce : époques choisies sur la courbe de validation moyenne des plis groupés |
| R45 | dropout des jetons : chaque jeton masqué avec la probabilité p avant l'attention |
| R46 | dropout des dimensions de z, le vecteur agrégé |
| R47 | départ et rétrécissement vers la logistique sur la moyenne des jetons : ½λ‖w − w₀‖² |

R40, R42 et R46 valent aussi pour la sonde à portes (R85, `blanci/gated.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


@dataclass
class AttentiveHead:
    mean: np.ndarray  # (d,)
    scale: np.ndarray  # (d,)
    query: np.ndarray  # (d,)
    weight: np.ndarray  # (d,)
    bias: float
    meta: dict[str, Any] = field(default_factory=dict)

    def attention(self, tokens: np.ndarray) -> np.ndarray:
        """Poids d'attention (fenêtres, jetons) : quels morceaux de la fenêtre comptent."""
        x = (np.asarray(tokens, dtype=np.float32) - self.mean) / self.scale
        return _softmax(x @ self.query / np.sqrt(x.shape[-1]), axis=1)

    def decision(self, tokens: np.ndarray) -> np.ndarray:
        x = (np.asarray(tokens, dtype=np.float32) - self.mean) / self.scale
        a = _softmax(x @ self.query / np.sqrt(x.shape[-1]), axis=1)
        return np.einsum("nt,ntd->nd", a, x) @ self.weight + self.bias

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "weights.npz",
            mean=self.mean,
            scale=self.scale,
            query=self.query,
            weight=self.weight,
        )
        manifest = {"kind": "attentive", "bias": self.bias, "meta": self.meta}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> AttentiveHead:
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        w = np.load(directory / "weights.npz")
        return cls(
            w["mean"], w["scale"], w["query"], w["weight"], manifest["bias"], manifest["meta"]
        )


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
) -> int:
    """Descente en lot entier, partagée par l'attentive et la sonde à portes (R85).

    `loss_of()` : perte d'entraînement (dropout compris) ; weight decay sur les seuls paramètres
    `decayed`. `optimizer` : "adam" (L2 couplée, défaut historique) ou "adamw" (R41).
    `validation_loss()` : critère à minimiser sur des micros tenus à l'écart (R42), relevé à
    chaque époque dans `curve` (liste remplie en place) ; avec `patience`, l'entraînement
    s'arrête `patience` époques après la meilleure, dont les paramètres sont restaurés.
    Renvoie l'époque retenue (la dernière sans validation)."""
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
    opt = kinds[optimizer]([g for g in groups if g["params"]], lr=lr)
    best, best_epoch, best_state = float("inf"), int(epochs), None
    for epoch in range(1, int(epochs) + 1):
        opt.zero_grad()
        loss = loss_of()
        loss.backward()
        opt.step()
        if validation_loss is None:
            continue
        with torch.no_grad():
            current = float(validation_loss())
        if curve is not None:
            curve.append(current)
        if current < best - 1e-7:
            best, best_epoch = current, epoch
            best_state = [p.detach().clone() for p in parameters]
        elif patience is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        with torch.no_grad():
            for p, value in zip(parameters, best_state, strict=True):
                p.copy_(value)
    return best_epoch


def validation_criterion(torch, loss_fn, scores_of, y_val: np.ndarray, monitor: str = "loss"):
    """Critère de R42, à minimiser : la perte d'entraînement sur les fenêtres de validation
    ("loss"), ou 1 − AP ("ap"). La perte monte dès que la tête devient trop sûre d'elle, même
    quand son classement s'améliore encore (mesuré, DECISIONS n° 117)."""
    from blanci.evaluate import average_precision

    y_np = np.asarray(y_val).astype(int)
    target = torch.from_numpy(y_np.astype(np.float32))
    if monitor == "loss":
        return lambda: loss_fn(scores_of(), target)
    if monitor == "ap":
        return lambda: 1.0 - average_precision(y_np, scores_of().numpy())
    raise ValueError(f"R42 : critère inconnu {monitor!r} (loss ou ap)")


def keep_mask(torch, n: int, t: int, p: float):
    """Jetons gardés (n, t) : chacun masqué avec la probabilité p, au moins un par fenêtre."""
    keep = torch.rand(n, t) >= p
    keep[torch.arange(n), torch.randint(t, (n,))] = True
    return keep


def fit_attentive(
    tokens: np.ndarray,
    y: np.ndarray,
    weight_decay: float = 1e-3,
    epochs: int = 300,
    lr: float = 5e-2,
    seed: int = 0,
    *,
    optimizer: str = "adam",
    token_dropout: float = 0.0,
    dim_dropout: float = 0.0,
    shrink: float = 0.0,
    shrink_C: float = 1.0,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    patience: int | None = 20,
    monitor: str = "loss",
) -> AttentiveHead:
    """Entraîne la tête (lot entier, Adam, entropie croisée à classes équilibrées).

    Options (toutes coupées par défaut) : `optimizer="adamw"` (R41), `token_dropout` (R45),
    `dim_dropout` (R46), `shrink` λ > 0 (R47 : w part de w₀, logistique de C `shrink_C` sur la
    moyenne des jetons, et ½λ‖w − w₀‖² remplace le weight decay sur w), `validation`
    (jetons, labels) de micros tenus à l'écart, `patience` (None : pas d'arrêt, courbe
    complète) et `monitor` ("loss" ou "ap") pour R42 (`fit_with_options`) ; la courbe de
    validation est rangée dans `meta["validation_curve"]`."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'installation
        raise RuntimeError(
            "l'attentive probing s'entraîne avec torch (uv sync --group research)"
        ) from exc

    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 3:
        raise ValueError(f"jetons attendus en (fenêtres, jetons, dim), reçu {tokens.shape}")
    y = np.asarray(y).astype(np.float32)
    flat = tokens.reshape(-1, tokens.shape[-1])
    mean = flat.mean(axis=0)
    scale = np.where(flat.std(axis=0) > 0, flat.std(axis=0), 1.0).astype(np.float32)
    x = torch.from_numpy((tokens - mean) / scale)
    target = torch.from_numpy(y)
    n, t, d = x.shape

    torch.manual_seed(seed)
    query = torch.zeros(d, requires_grad=True)  # départ : moyenne simple des jetons
    weight = torch.zeros(d, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    start = None
    if shrink > 0:  # R47 : départ depuis la logistique sur la moyenne des jetons
        from blanci.head import fit_logistic

        logistic = fit_logistic(x.mean(dim=1).numpy(), y.astype(int), shrink_C, seed)
        start = torch.from_numpy(logistic.coef / logistic.scale)
        with torch.no_grad():
            weight.copy_(start)
            bias.fill_(logistic.intercept - float(logistic.coef @ (logistic.mean / logistic.scale)))
    n_pos = max(float(y.sum()), 1.0)
    pos_weight = torch.tensor((len(y) - n_pos) / n_pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def logits(inputs, train: bool):
        scores = inputs @ query / d**0.5
        if train and token_dropout > 0:
            scores = scores.masked_fill(~keep_mask(torch, *scores.shape, token_dropout), -1e9)
        z = torch.einsum("nt,ntd->nd", torch.softmax(scores, dim=1), inputs)
        if train and dim_dropout > 0:
            z = torch.nn.functional.dropout(z, dim_dropout, training=True)
        return z @ weight + bias

    def loss_of():
        loss = loss_fn(logits(x, True), target)
        if start is not None:
            loss = loss + 0.5 * shrink * ((weight - start) ** 2).sum()
        return loss

    validation_loss = None
    if validation is not None:
        x_val = torch.from_numpy((np.asarray(validation[0], dtype=np.float32) - mean) / scale)
        validation_loss = validation_criterion(
            torch, loss_fn, lambda: logits(x_val, False), validation[1], monitor
        )
    curve: list[float] = []
    best_epoch = optimise(
        [query, weight, bias],
        [True, start is None, True],
        loss_of,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        optimizer=optimizer,
        validation_loss=validation_loss,
        patience=patience,
        curve=curve,
    )
    meta: dict[str, Any] = {
        "weight_decay": weight_decay,
        "epochs": epochs,
        "lr": lr,
        "n_tokens": tokens.shape[1],
    }
    options = {
        "optimizer": optimizer if optimizer != "adam" else None,
        "token_dropout": token_dropout or None,
        "dim_dropout": dim_dropout or None,
        "shrink": shrink or None,
        "shrink_C": shrink_C if shrink else None,
        "best_epoch": best_epoch if validation is not None else None,
        "validation_curve": curve or None,
    }
    meta |= {k: v for k, v in options.items() if v is not None}
    return AttentiveHead(
        mean.astype(np.float32),
        scale,
        query.detach().numpy().astype(np.float32),
        weight.detach().numpy().astype(np.float32),
        float(bias.detach().numpy()[0]),
        meta,
    )


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
    interne (`n_splits` plis, micros entiers), la meilleure est retenue. R42
    (`early_stopping`) : dans chaque pli interne, la courbe du critère de validation
    (`monitor` : "loss" ou "ap") est relevée à chaque époque ; l'époque retenue minimise la
    courbe moyenne des plis, et la tête est réentraînée sur tout `X` pour ce nombre d'époques
    (en lot entier, une époque = un pas, quel que soit l'effectif). Un pli unique de 1 à 3
    micros donnait une époque au hasard (mesuré, DECISIONS n° 117). Faute de deux micros, ou
    d'une classe dans un pli, R40/R42 sont sautées.
    """
    from blanci.evaluate import average_precision, grouped_folds

    X, y, groups = np.asarray(X), np.asarray(y).astype(int), np.asarray(groups)
    extra: dict[str, Any] = {}
    several = len(np.unique(groups)) >= 2
    folds = []
    if several:
        folds = [
            (train, test)
            for train, test in grouped_folds(y, groups, n_splits, seed)
            if len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2
        ]
    if weight_decays and len(weight_decays) > 1 and folds:
        results = {}
        for wd in weight_decays:
            aps = [
                average_precision(
                    y[test],
                    fit_with_options(
                        fit,
                        X[train],
                        y[train],
                        groups[train],
                        seed,
                        early_stopping=early_stopping,
                        monitor=monitor,
                        n_splits=n_splits,
                        **options | {"weight_decay": wd},
                    ).decision(X[test]),
                )
                for train, test in folds
            ]
            results[float(wd)] = float(np.nanmean(aps))
        valid = {k: v for k, v in results.items() if np.isfinite(v)}
        if valid:
            options["weight_decay"] = max(valid, key=valid.get)
            extra["weight_decay_cv"] = {str(k): v for k, v in results.items()}
    if early_stopping and folds:
        curves = [
            fit(
                X[train],
                y[train],
                seed=seed,
                validation=(X[test], y[test]),
                patience=None,
                monitor=monitor,
                **options,
            ).meta["validation_curve"]
            for train, test in folds
        ]
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        best = int(np.nanargmin(mean_curve)) + 1
        head = fit(X, y, seed=seed, **(options | {"epochs": best}))
        head.meta |= extra | {
            "early_stopping": {"best_epoch": best, "monitor": monitor, "folds": len(curves)}
        }
        return head
    head = fit(X, y, seed=seed, **options)
    head.meta |= extra
    return head


# --- Stock de jetons -------------------------------------------------------------------------


@dataclass
class TokenStore:
    """Jetons des seules fenêtres du benchmark (étiquetées + négatifs appariés) : un stock
    complet serait trop lourd (perch_v2 : 16 × 1 536 valeurs par fenêtre)."""

    root: Path  # data/tokens
    encoder_id: str

    @property
    def path(self) -> Path:
        return Path(self.root) / f"{self.encoder_id}.npz"

    def write(self, window_ids: list[str], tokens: np.ndarray) -> Path:
        """Ajoute des jetons ; une fenêtre déjà présente est remplacée."""
        ids, values = list(window_ids), np.asarray(tokens, dtype=np.float16)
        if self.path.exists():
            old_ids, old = self.read()
            new = set(ids)
            keep = [i for i, w in enumerate(old_ids) if w not in new]
            ids = [old_ids[i] for i in keep] + ids
            values = np.concatenate([old[keep], values])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, window_ids=np.array(ids), tokens=values)
        tmp.replace(self.path)
        return self.path

    def read(self) -> tuple[list[str], np.ndarray]:
        data = np.load(self.path)
        return data["window_ids"].tolist(), data["tokens"]

    def load(self, window_ids: list[str]) -> np.ndarray | None:
        """Jetons alignés sur `window_ids`, ou None s'il en manque (stock absent ou partiel)."""
        if not self.path.exists():
            return None
        ids, tokens = self.read()
        index = {w: i for i, w in enumerate(ids)}
        if any(w not in index for w in window_ids):
            return None
        return tokens[[index[w] for w in window_ids]].astype(np.float32)
