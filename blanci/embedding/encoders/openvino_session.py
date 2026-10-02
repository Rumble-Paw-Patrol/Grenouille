"""Modèle ONNX calculé par OpenVINO, sur le processeur graphique intégré d'Intel (Iris Xe).

Remplace la session ONNX Runtime d'un encodeur (`encoders.models.<nom>.openvino`) : même
appel, mêmes sorties à 5e-7 près en float32 (DECISIONS n° 179). OpenVINO n'est pas une
dépendance obligatoire : absent, ou sans le périphérique demandé (Mac, autre poste), l'encodeur
garde ONNX Runtime sur le CPU.
"""

from __future__ import annotations

from pathlib import Path
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
