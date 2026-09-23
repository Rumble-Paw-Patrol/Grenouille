"""Encodeurs : toujours obtenus par `get_encoder`, jamais appelés en direct (§13.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blanci.encoders.base import BaseEncoder, Encoder, encoder_id

__all__ = ["BaseEncoder", "Encoder", "encoder_id", "get_encoder"]


def get_encoder(name: str, cfg: dict[str, Any]) -> Encoder:
    """Encodeur de recherche (bacpipe) ou paquet ONNX `data/models/encoder/<name>`.

    Une entrée de `encoders.models` avec `lowpass_hz` enveloppe l'encodeur d'un passe-bas
    (contrôle du §2, `blanci/encoders/lowpass.py`).
    """
    spec = cfg["encoders"]["models"].get(name)
    batch = cfg["encoders"]["batch_size"]
    if spec is not None and spec["backend"] == "bacpipe":
        from blanci.encoders.bacpipe_encoder import BacpipeEncoder

        encoder: Encoder = BacpipeEncoder(
            spec["model"],
            batch_size=batch,
            model_base_path=Path(cfg["paths"]["models"]) / "bacpipe",
            logit_classes=spec.get("logit_classes"),
            checkpoint=spec.get("checkpoint"),
            name=name if spec.get("checkpoint") else None,
        )
        if spec.get("lowpass_hz"):
            from blanci.encoders.lowpass import LowpassEncoder

            encoder = LowpassEncoder(encoder, spec["lowpass_hz"])
        return encoder
    package = Path(cfg["paths"]["models"]) / "encoder" / name
    if (package / "manifest.json").exists():
        from blanci.encoders.onnx_encoder import OnnxEncoder

        return OnnxEncoder(package, batch_size=batch)
    raise ValueError(
        f"encodeur inconnu : {name!r} (ni dans encoders.models, ni paquet ONNX dans {package})"
    )
