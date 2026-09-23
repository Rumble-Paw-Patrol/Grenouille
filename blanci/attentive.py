"""Attentive probing (§3) : tête d'attention sur les jetons d'un encodeur, avant agrégation.

L'embedding d'une fenêtre est la moyenne de ses jetons (morceaux de temps × fréquence). Une
note de 0,1 s dans une fenêtre de 5 s ne tient que dans un ou deux jetons : la moyenne la
dilue dans le fond. La tête d'attention apprend une requête q qui pondère les jetons :

    a_t = softmax_t(x̃_t · q / √d),  z = Σ_t a_t x̃_t,  score = w · z + b

(x̃ : jetons standardisés). Une requête, un vecteur de classement : d·2 + 1 paramètres, peu
pour ~150–200 positifs (§3 : « ≥ 150–200 annotations »). Entraînement avec torch (groupe
`research`), application en numpy ; sauvegarde JSON + npz, sans pickle (§7).
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


def fit_attentive(
    tokens: np.ndarray,
    y: np.ndarray,
    weight_decay: float = 1e-3,
    epochs: int = 300,
    lr: float = 5e-2,
    seed: int = 0,
) -> AttentiveHead:
    """Entraîne la tête (lot entier, Adam, entropie croisée à classes équilibrées)."""
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
    d = x.shape[-1]

    torch.manual_seed(seed)
    query = torch.zeros(d, requires_grad=True)  # départ : moyenne simple des jetons
    weight = torch.zeros(d, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    n_pos = max(float(y.sum()), 1.0)
    pos_weight = torch.tensor((len(y) - n_pos) / n_pos)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam([query, weight, bias], lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        a = torch.softmax(x @ query / d**0.5, dim=1)
        z = torch.einsum("nt,ntd->nd", a, x)
        loss = loss_fn(z @ weight + bias, target)
        loss.backward()
        optimizer.step()
    return AttentiveHead(
        mean.astype(np.float32),
        scale,
        query.detach().numpy().astype(np.float32),
        weight.detach().numpy().astype(np.float32),
        float(bias.detach().numpy()[0]),
        {"weight_decay": weight_decay, "epochs": epochs, "lr": lr, "n_tokens": tokens.shape[1]},
    )


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
