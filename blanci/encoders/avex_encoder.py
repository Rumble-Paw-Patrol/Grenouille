"""Encodeurs esp-aves2 (Earth Species Project, ICLR 2026) par la bibliothèque AVEX (§2, n° 151).

Référence de benchmark seulement : les dix points de contrôle sont sous CC-BY-NC-SA-4.0
(non déployables, `documentation/encodeurs-bacpipe.md`). `pip install avex` ;
`avex.load_model(<nom>, return_features_only=True)` rend, pour un lot d'audio à 16 kHz :

- EfficientNet-B0 (`esp_aves2_effnetb0_*`) : la dernière carte (lot, 1 280, fréquence, temps) ;
- BEATs (`esp_aves2_sl_beats_*`, `esp_aves2_naturelm_audio_v1_beats`) : les jetons
  (lot, temps × 8 fréquences, 768), grille temps d'abord ;
- EAT (`esp_aves2_eat_*`, `esp_aves2_sl_eat_*`) : un jeton de classe puis les jetons d'un
  spectrogramme complété à 1 024 trames (10 s, sa durée d'entraînement), temps × 8 fréquences.

Embedding : l'agrégation d'AVEX elle-même (jeton de classe pour EAT, moyenne sinon). Jetons :
grille (fenêtres, temps, fréquence, dim) sur la seule durée de la fenêtre (les lignes de temps
du complément d'EAT sont retirées). Fenêtre de 5 s : la grille de perch_v2, pour comparer
fenêtre à fenêtre (benchmark 07).
"""

from __future__ import annotations

import math

import numpy as np

from blanci.encoders.base import BaseEncoder

HOP_LENGTH = 160  # trames de 10 ms à 16 kHz (configuration audio d'AVEX)
PATCH = 16  # jetons de 16 trames × 16 bandes (BEATs, EAT)
FREQ_BANDS = 8  # 128 bandes mel / 16


def remap_fairseq_eat(state: dict, model_keys) -> dict | None:
    """Clés fairseq d'un point de contrôle EAT → clés du modèle d'AVEX, ou None s'il n'est pas
    au format fairseq.

    `eat_bio` et `eat_all` (EAT auto-supervisé seul) sont publiés au format fairseq
    (`blocks.*`, `modality_encoders.IMAGE.*`, `context_encoder.norm`) ; AVEX 1.3.0 n'ajoute que
    le préfixe `backbone.` et n'en charge aucun tenseur (0/150) : le modèle garde alors les
    poids de l'EAT générique d'AudioSet. Le décodeur de reconstruction est laissé de côté.
    """
    if not any(k.startswith("modality_encoders.IMAGE.") for k in state):
        return None
    out = {}
    for key, value in state.items():
        key = key.replace("modality_encoders.IMAGE.", "")
        key = key.replace("context_encoder.norm.", "pre_norm.")
        out["backbone.model." + key] = value
    missing = sorted(set(model_keys) - set(out))
    if missing:
        raise ValueError(f"point de contrôle EAT fairseq : clés absentes {missing[:5]}")
    return {k: out[k] for k in model_keys}


def _load_fairseq_eat(model, model_name: str) -> None:
    from avex.models.utils.load import get_checkpoint_path
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = get_checkpoint_path(model_name)
    if not path or not path.startswith("hf://"):
        return
    org, repo, filename = path[len("hf://") :].split("/", 2)
    state = load_file(hf_hub_download(f"{org}/{repo}", filename))
    remapped = remap_fairseq_eat(state, list(model.state_dict()))
    if remapped is not None:
        model.load_state_dict(remapped, strict=True)


class AvexEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        window_s: float = 5.0,
        device: str = "cpu",
        name: str | None = None,
    ):
        from importlib.metadata import version

        import avex

        self.name = name or model_name
        self.version = f"avex{version('avex')}"
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        spec = avex.get_model_spec(model_name)
        self.family = spec.name  # efficientnet, beats, eat_hf
        self.sample_rate = int(spec.audio_config.sample_rate)
        self.window_s = float(window_s)
        self._model = avex.load_model(model_name, device=device, return_features_only=True)
        if self.family.startswith("eat"):
            _load_fairseq_eat(self._model, model_name)
        self._model.eval()
        probe = self._grid(self._features(np.zeros((1, self._samples()), np.float32)))
        self.dim = int(probe[1].shape[-1])
        self.has_tokens = True

    def _samples(self) -> int:
        return round(self.window_s * self.sample_rate)

    def _features(self, batch: np.ndarray):
        import torch

        with torch.no_grad():
            out = self._model(torch.from_numpy(np.ascontiguousarray(batch)).to(self.device))
        return out.detach().cpu().numpy().astype(np.float32)

    def _grid(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(embedding, jetons en grille (lot, temps, fréquence, dim))."""
        if feats.ndim == 4:  # EfficientNet : (lot, canaux, fréquence, temps)
            return feats.mean(axis=(2, 3)), np.ascontiguousarray(feats.transpose(0, 3, 2, 1))
        if self.family.startswith("eat"):
            cls, patches = feats[:, 0], feats[:, 1:]
            n, count, dim = patches.shape
            grid = patches.reshape(n, count // FREQ_BANDS, FREQ_BANDS, dim)
            rows = math.ceil(self._samples() / HOP_LENGTH / PATCH)  # sans le complément à 10 s
            return cls, grid[:, :rows]
        n, count, dim = feats.shape  # BEATs
        if count % FREQ_BANDS:
            raise ValueError(f"{self.name} : {count} jetons, pas un multiple de {FREQ_BANDS}")
        return feats.mean(axis=1), feats.reshape(n, count // FREQ_BANDS, FREQ_BANDS, dim)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        return self._grid(self._features(batch))[0]

    def _forward_tokens(self, batch: np.ndarray) -> np.ndarray:
        return self._grid(self._features(batch))[1]
