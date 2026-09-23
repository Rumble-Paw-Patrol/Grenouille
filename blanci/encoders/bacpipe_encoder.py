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
  (pas dans le dossier courant, défaut de bacpipe) ; birdmae passe par le cache Hugging Face ;
- perch_v2 (ONNX, sans TensorFlow) garde les logits de ses 14 795 classes après chaque appel
  (`model.results["logits"]`, noms dans `model.classes`) : `logit_classes` en retient
  quelques-unes (les trois *Anomaloglossus* congénères, §2), relues par `pop_logits` après
  `embed`, sans seconde inférence.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from blanci.encoders.base import BaseEncoder


def _birdmae_with(module, checkpoint: str):
    """Classe `Model` de birdmae chargeant `checkpoint` au lieu de Bird-MAE-Huge.

    bacpipe 1.3.5 impose la version Huge (632 M paramètres : 0,5 fenêtre/s sur l'i5,
    DECISIONS n° 72) ; le §7 prévoit Bird-MAE-Base. Même extracteur de spectrogramme et même
    appel que bacpipe : seul le réseau change.
    """
    from bacpipe.model_pipelines.model_utils import ModelBaseClass
    from transformers import AutoFeatureExtractor, AutoModel

    class Model(module.Model):
        def __init__(self, **kwargs):
            ModelBaseClass.__init__(
                self, sr=module.SAMPLE_RATE, segment_length=module.LENGTH_IN_SAMPLES, **kwargs
            )
            self.audio_processor = AutoFeatureExtractor.from_pretrained(
                "DBD-research-group/Bird-MAE-Base", trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            self.preproc_batch_size = 511

    return Model


def _load_bacpipe_model(
    model_name: str, device: str, model_base_path: Path, checkpoint: str | None = None
):
    """(module, modèle prêt pour l'inférence) ; télécharge les poids s'il le faut.

    `checkpoint` : autre jeu de poids Hugging Face, pour birdmae seulement.
    """
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
    if checkpoint is not None:
        if model_name != "birdmae":
            raise ValueError(f"checkpoint n'est pris en charge que pour birdmae, pas {model_name}")
        model_class = _birdmae_with(module, checkpoint)
    else:
        model_class = module.Model
    model = model_class(model_name=model_name, **settings)
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
    logit_names: tuple[str, ...] | list[str] = ()
    _logit_index: tuple[int, ...] | list[int] = ()  # aucun logit gardé par défaut

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: str | None = None,
        model_base_path: Path | str = "data/models/bacpipe",
        logit_classes: list[str] | None = None,
        checkpoint: str | None = None,
        name: str | None = None,
    ):
        from importlib.metadata import version

        import bacpipe  # noqa: F401  (message clair si absent)

        self.name = name or model_name  # nom de la config : birdmae_base ≠ birdmae
        self.version = f"bacpipe{version('bacpipe')}"
        self.device = device or _default_device()
        self.batch_size = batch_size
        module, self._model = _load_bacpipe_model(
            model_name, self.device, Path(model_base_path), checkpoint
        )
        self.sample_rate = int(module.SAMPLE_RATE)
        self.window_s = module.LENGTH_IN_SAMPLES / module.SAMPLE_RATE
        self.logit_names = list(logit_classes or [])
        self._logit_index = self._resolve_classes(self.logit_names)
        self._logits: list[np.ndarray] = []
        probe = self._raw(np.zeros((1, module.LENGTH_IN_SAMPLES), np.float32))
        self.has_tokens = probe.ndim == 3
        self.dim = int(probe.shape[-1])
        self._logits.clear()

    def _resolve_classes(self, names: list[str]) -> list[int]:
        if not names:
            return []
        classes = list(getattr(self._model, "classes", []) or [])
        missing = [n for n in names if n not in classes]
        if missing:
            raise ValueError(f"{self.name} n'a pas de classe {missing} (logit_classes)")
        return [classes.index(n) for n in names]

    def pop_logits(self) -> np.ndarray | None:
        """Logits des `logit_classes` depuis le dernier appel : (fenêtres, classes) ; vidé."""
        if not self._logit_index:
            return None
        out = (
            np.concatenate(self._logits)
            if self._logits
            else np.zeros((0, len(self._logit_index)), np.float32)
        )
        self._logits.clear()
        return out

    def _raw(self, batch: np.ndarray) -> np.ndarray:
        """Sortie du modèle telle quelle : (lot, dim) ou (lot, jetons, dim)."""
        import torch

        with torch.no_grad():
            x = self._model.preprocess(torch.from_numpy(np.ascontiguousarray(batch)))
            out = _to_numpy(self._model(x))
        if self._logit_index:
            logits = _to_numpy(self._model.results["logits"]).reshape(len(batch), -1)
            self._logits.append(logits[:, self._logit_index])
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
