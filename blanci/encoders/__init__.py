"""Encodeurs : toujours obtenus par `get_encoder`, jamais appelés en direct (§13.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blanci.encoders.base import BaseEncoder, Encoder, encoder_id, stock_id

__all__ = ["BaseEncoder", "Encoder", "encoder_id", "get_encoder", "stock_id"]


def get_encoder(name: str, cfg: dict[str, Any], upstream: Any = None) -> Encoder:
    """Encodeur de recherche (bacpipe) ou paquet ONNX `data/models/encoder/<name>`.

    Une entrée de `encoders.models` avec `lowpass_hz` enveloppe l'encodeur d'un passe-bas
    (contrôle du §2, `blanci/encoders/lowpass.py`) ; avec `transforms` (ex. `{bandpass:
    {band_hz: [3000, 7000]}}`), des transformations du module séquentiel en amont. `upstream`
    (`blanci.sequential.Upstream`, option `--upstream`) y ajoute ses transformations actives.
    """
    return _with_transforms(_get_encoder(name, cfg), name, cfg, upstream)


def _with_transforms(encoder: Encoder, name: str, cfg: dict[str, Any], upstream: Any) -> Encoder:
    from blanci.sequential import Upstream

    spec = cfg["encoders"]["models"].get(name) or {}
    transforms = dict(spec.get("transforms") or {})
    if upstream is not None:
        transforms |= upstream.transforms
    if not transforms:
        return encoder
    from blanci.encoders.upstream import UpstreamEncoder
    from blanci.sequential import upstream_from_cfg

    defaults = upstream_from_cfg(cfg, only=list(transforms)).transforms  # réglages manquants
    merged = {k: defaults.get(k, {}) | (v or {}) for k, v in transforms.items()}
    return UpstreamEncoder(encoder, Upstream(merged, signal_cfg=cfg.get("signal", {})))


def _get_encoder(name: str, cfg: dict[str, Any]) -> Encoder:
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
