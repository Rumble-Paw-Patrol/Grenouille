"""Encodeurs de recherche via bacpipe (§2) : birdmae, beats, naturebeats, perch_v2, birdnet...

bacpipe n'est pas une dépendance du livrable : installé par `uv sync --group research`.
Tout ce qui dépend de l'API interne de bacpipe est isolé dans `_load_bacpipe_model` et
`_forward` ; ces deux fonctions sont à valider à l'installation (M1) contre la documentation
de bacpipe 1.3.x. [À VÉRIFIER]
"""

from __future__ import annotations

import importlib

import numpy as np

from blanci.encoders.base import BaseEncoder


def _load_bacpipe_model(model_name: str, device: str):
    """(module, modèle) bacpipe : un module par modèle, constantes SAMPLE_RATE et LENGTH_IN_SAMPLES.

    [À VÉRIFIER] chemin du module et signature du constructeur dans bacpipe 1.3.x.
    """
    try:
        module = importlib.import_module(
            f"bacpipe.embedding_generation_pipelines.feature_extractors.{model_name}"
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"modèle bacpipe {model_name!r} introuvable ; bacpipe est-il installé "
            "(uv sync --group research) ?"
        ) from exc
    return module, module.Model(model_name=model_name, device=device)


def _default_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BacpipeEncoder(BaseEncoder):
    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        import bacpipe  # noqa: F401  (message clair si absent)

        self.name = model_name
        self.version = f"bacpipe{getattr(bacpipe, '__version__', '')}"
        self.device = device or _default_device()
        self.batch_size = batch_size
        module, self._model = _load_bacpipe_model(model_name, self.device)
        self.sample_rate = int(module.SAMPLE_RATE)
        self.window_s = module.LENGTH_IN_SAMPLES / module.SAMPLE_RATE
        self.dim = int(self._forward(np.zeros((1, module.LENGTH_IN_SAMPLES), np.float32)).shape[1])

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        """[À VÉRIFIER] prétraitement et inférence d'un lot de fenêtres dans bacpipe."""
        import torch

        with torch.no_grad():
            x = self._model.preprocess(torch.from_numpy(batch))
            out = self._model(x)
        if hasattr(out, "detach"):
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float32).reshape(len(batch), -1)
