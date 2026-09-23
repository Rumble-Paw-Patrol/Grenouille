"""Encodeurs de recherche via bacpipe (§2) : birdmae, beats, naturebeats, perch_v2, birdnet...

bacpipe n'est pas une dépendance du livrable : installé par `uv sync --group research`.
Validé contre bacpipe 1.3.5 (torch 2.6, TensorFlow 2.15, Windows) le 23/09/2026
(DECISIONS n° 64) :

- un module par modèle, `bacpipe.model_pipelines.feature_extractors.<nom>`, avec les
  constantes SAMPLE_RATE et LENGTH_IN_SAMPLES et une classe `Model` ;
- `Model(model_name=..., **réglages)` lit ses réglages (device, dossier des poids, classifieur…)
  dans `bacpipe.settings` seulement si `device` n'est pas donné : on les passe donc tous ;
- `model.preprocess(lot torch)` puis `model(lot prétraité)` → tenseur torch ou TensorFlow ;
- certains modèles rendent la séquence de jetons (lot × jetons × dim, ex. birdmae) : elle est
  moyennée ici, et l'encodeur est déclaré `has_tokens` ;
- les poids sont téléchargés par `bacpipe.ensure_models_exist` dans `paths.models/bacpipe`
  (pas dans le dossier courant, défaut de bacpipe) ; birdmae passe par le cache Hugging Face.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from blanci.encoders.base import BaseEncoder


def _load_bacpipe_model(model_name: str, device: str, model_base_path: Path):
    """(module, modèle prêt pour l'inférence) ; télécharge les poids s'il le faut."""
    import bacpipe
    from bacpipe.core.constants import NEEDS_CHECKPOINT

    path = f"bacpipe.model_pipelines.feature_extractors.{model_name}"
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError as exc:
        if exc.name != path:  # le modèle existe, une de ses dépendances manque
            raise
        raise RuntimeError(
            f"modèle bacpipe {model_name!r} introuvable ; bacpipe est-il installé "
            "(uv sync --group research) ?"
        ) from exc
    model_base_path = Path(model_base_path)
    if model_name in NEEDS_CHECKPOINT:
        bacpipe.ensure_models_exist(model_base_path=model_base_path, model_names=[model_name])
    settings = vars(bacpipe.settings) | {
        "device": device,
        "model_base_path": str(model_base_path),
        "run_pretrained_classifier": False,  # embeddings seulement
    }
    model = module.Model(model_name=model_name, **settings)
    model.prepare_inference()
    return module, model


def _default_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_numpy(out) -> np.ndarray:
    """Tenseur torch ou TensorFlow → numpy float32."""
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    elif hasattr(out, "numpy"):
        out = out.numpy()
    return np.asarray(out, dtype=np.float32)


class BacpipeEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
        model_base_path: Path | str = "data/models/bacpipe",
    ):
        from importlib.metadata import version

        import bacpipe  # noqa: F401  (message clair si absent)

        self.name = model_name
        self.version = f"bacpipe{version('bacpipe')}"
        self.device = device or _default_device()
        self.batch_size = batch_size
        module, self._model = _load_bacpipe_model(model_name, self.device, Path(model_base_path))
        self.sample_rate = int(module.SAMPLE_RATE)
        self.window_s = module.LENGTH_IN_SAMPLES / module.SAMPLE_RATE
        probe = self._raw(np.zeros((1, module.LENGTH_IN_SAMPLES), np.float32))
        self.has_tokens = probe.ndim == 3
        self.dim = int(probe.shape[-1])

    def _raw(self, batch: np.ndarray) -> np.ndarray:
        """Sortie du modèle telle quelle : (lot, dim) ou (lot, jetons, dim)."""
        import torch

        with torch.no_grad():
            x = self._model.preprocess(torch.from_numpy(np.ascontiguousarray(batch)))
            out = _to_numpy(self._model(x))
        if out.ndim == 1:
            out = out[None, :]
        if out.ndim > 3:
            out = out.reshape(len(batch), -1, out.shape[-1])
        return out

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        out = self._raw(batch)
        if out.ndim == 3:  # jetons : moyenne (agrégation E du §1 à comparer plus tard)
            out = out.mean(axis=1)
        return out.reshape(len(batch), -1)

    def _forward_tokens(self, batch: np.ndarray) -> np.ndarray | None:
        out = self._raw(batch)
        return out if out.ndim == 3 else None
