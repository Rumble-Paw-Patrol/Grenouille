"""Modèle calculé par OpenVINO, sur le processeur graphique intégré d'Intel (Iris Xe).

Remplace la session ONNX Runtime d'un encodeur (`encoders.models.<nom>.openvino`) : même
appel, mêmes sorties à 5e-7 près en float32 (DECISIONS n° 179). Pour un modèle torch
(AudioProtoPNet, ConvNeXt BirdSet), remplace son réseau, converti une fois (`OpenVinoBackbone`).
OpenVINO n'est pas une dépendance obligatoire : absent, ou sans le périphérique demandé (Mac,
autre poste), l'encodeur garde ONNX Runtime ou torch sur le CPU.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


class OpenVinoSession:
    """`run(None, {entrée: lot})` comme `onnxruntime.InferenceSession`, sorties dans l'ordre
    du modèle. Le graphe est compilé pour un lot de taille fixe (plus rapide sur le processeur
    graphique) : un lot plus court est complété par des zéros, un plus long découpé."""

    def __init__(
        self,
        model_path: Path,
        device: str = "GPU",
        batch: int = 4,
        precision: str = "f32",
        cache_dir: Path | None = None,
    ):
        import openvino as ov

        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f"périphérique {device} absent ({core.available_devices})")
        model = core.read_model(str(model_path))
        self.batch = int(batch)
        self._samples = model.inputs[0].get_partial_shape()[1].get_length()
        model.reshape({model.inputs[0].get_any_name(): [self.batch, self._samples]})
        properties: dict[str, Any] = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
        }
        if cache_dir is not None:  # graphe compilé gardé : 23 s au premier lancement seulement
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            properties["CACHE_DIR"] = str(cache_dir)
        self._compiled = core.compile_model(model, device, properties)
        self._request = self._compiled.create_infer_request()
        self.output_names = [out.get_any_name() for out in self._compiled.outputs]
        self.device = core.get_property(device, "FULL_DEVICE_NAME")

    def get_providers(self) -> list[str]:
        return [f"OpenVINO ({self.device})"]

    def _infer(self, x: np.ndarray) -> list[np.ndarray]:
        n = len(x)
        if n < self.batch:
            x = np.concatenate([x, np.zeros((self.batch - n, x.shape[1]), dtype=x.dtype)])
        result = self._request.infer({0: np.ascontiguousarray(x, dtype=np.float32)})
        return [np.array(result[out][:n]) for out in self._compiled.outputs]

    def run(self, output_names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        (x,) = feeds.values()
        parts = [self._infer(x[i : i + self.batch]) for i in range(0, len(x), self.batch)]
        return [np.concatenate(column) for column in zip(*parts, strict=True)]


def _convert(backbone, example):
    """Réseau converti en OpenVINO ; il rend le tuple (carte, embedding) au lieu de sa sortie
    Hugging Face. torch n'est importé qu'ici (comme dans `export.py`)."""
    import openvino as ov
    import torch

    class Outputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            out = self.backbone(x)
            return out.last_hidden_state, out.pooler_output

    return ov.convert_model(Outputs().eval(), example_input=example[:1])


def weights_digest(module) -> str:
    """Empreinte des poids d'un module torch : le graphe converti n'est repris que pour ces
    poids-là (un checkpoint mis à jour sur Hugging Face donne un autre fichier)."""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


class OpenVinoBackbone:
    """Réseau torch d'un modèle bacpipe (le ConvNeXt d'AudioProtoPNet et de ConvNeXt BirdSet)
    calculé par OpenVINO : s'appelle comme lui sur un lot de spectrogrammes prétraités et rend
    `last_hidden_state` et `pooler_output`, en tenseurs torch. Converti une fois, rangé dans
    `cache_dir` sous l'empreinte de ses poids ; compilé pour un lot fixe, comme
    `OpenVinoSession`."""

    def __init__(
        self,
        backbone,
        example,
        device: str = "GPU",
        batch: int = 4,
        precision: str = "f32",
        cache_dir: Path | None = None,
    ):
        import openvino as ov

        core = ov.Core()
        if device not in core.available_devices:
            raise RuntimeError(f"périphérique {device} absent ({core.available_devices})")
        properties: dict[str, Any] = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": precision,
        }
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            xml = cache_dir / f"torch_{weights_digest(backbone)}.xml"
            if not xml.exists():  # f32 sur le disque : la précision se choisit à la compilation
                cache_dir.mkdir(parents=True, exist_ok=True)
                ov.save_model(_convert(backbone, example), str(xml), compress_to_fp16=False)
            model = core.read_model(str(xml))
            properties["CACHE_DIR"] = str(cache_dir)
        else:
            model = _convert(backbone, example)
        self.batch = int(batch)
        self._shape = tuple(example.shape[1:])
        model.reshape({model.inputs[0].get_any_name(): [self.batch, *self._shape]})
        self._compiled = core.compile_model(model, device, properties)
        self._request = self._compiled.create_infer_request()
        self.device = core.get_property(device, "FULL_DEVICE_NAME")

    def __call__(self, x):
        import torch

        x = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        parts = []
        for i in range(0, len(x), self.batch):
            chunk = x[i : i + self.batch]
            n = len(chunk)
            if n < self.batch:
                pad = np.zeros((self.batch - n, *self._shape), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])
            result = self._request.infer({0: chunk})
            parts.append([np.array(result[out][:n]) for out in self._compiled.outputs])
        fmap, pooled = (np.concatenate(column) for column in zip(*parts, strict=True))
        return SimpleNamespace(
            last_hidden_state=torch.from_numpy(fmap), pooler_output=torch.from_numpy(pooled)
        )


def openvino_backbone(
    backbone, example, settings: dict[str, Any], cache_dir: Path | None = None
) -> OpenVinoBackbone | None:
    """`OpenVinoBackbone` selon `settings`, ou None si OpenVINO ou le périphérique manque :
    l'appelant garde alors torch."""
    try:
        return OpenVinoBackbone(
            backbone,
            example,
            device=settings.get("device", "GPU"),
            batch=settings.get("batch_size", 4),
            precision=settings.get("precision", "f32"),
            cache_dir=cache_dir,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"OpenVINO non utilisé ({exc}) : torch sur le CPU", flush=True)
        return None


def openvino_session(
    model_path: Path, settings: dict[str, Any], cache_dir: Path | None = None
) -> OpenVinoSession | None:
    """Session OpenVINO selon `settings` (device, batch_size, precision), ou None si OpenVINO
    ou le périphérique manque : l'appelant garde alors ONNX Runtime."""
    try:
        return OpenVinoSession(
            model_path,
            device=settings.get("device", "GPU"),
            batch=settings.get("batch_size", 4),
            precision=settings.get("precision", "f32"),
            cache_dir=cache_dir,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"OpenVINO non utilisé ({exc}) : ONNX Runtime sur le CPU", flush=True)
        return None
