"""Encodeurs : toujours obtenus par `get_encoder`, jamais appelés en direct (§13.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blanci.encoders.base import BaseEncoder, Encoder, encoder_id

__all__ = ["BaseEncoder", "Encoder", "encoder_id", "get_encoder"]


def get_encoder(name: str, cfg: dict[str, Any]) -> Encoder:
    """Encodeur de recherche (bacpipe) ou paquet ONNX `data/models/encoder/<name>`."""
    spec = cfg["encoders"]["models"].get(name)
    batch = cfg["encoders"]["batch_size"]
    if spec is not None and spec["backend"] == "bacpipe":
        from blanci.encoders.bacpipe_encoder import BacpipeEncoder

        return BacpipeEncoder(
            spec["model"],
            batch_size=batch,
            model_base_path=Path(cfg["paths"]["models"]) / "bacpipe",
        )
    package = Path(cfg["paths"]["models"]) / "encoder" / name
    if (package / "manifest.json").exists():
        from blanci.encoders.onnx_encoder import OnnxEncoder

        return OnnxEncoder(package, batch_size=batch)
    raise ValueError(
        f"encodeur inconnu : {name!r} (ni dans encoders.models, ni paquet ONNX dans {package})"
    )
