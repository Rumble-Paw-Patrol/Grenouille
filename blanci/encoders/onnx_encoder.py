"""Encodeur du livrable : ONNX Runtime sur CPU, sans torch ni TensorFlow (§7, §13.1).

Paquet d'encodeur : data/models/encoder/<nom>-<version>/ avec `manifest.json` et `model.onnx`.
Le graphe prend la forme d'onde (lot, échantillons) à `sample_rate` : le spectrogramme est
inclus dans l'export (voir export.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blanci.encoders.base import BaseEncoder

MANIFEST_KEYS = ("name", "version", "sha256", "license", "sample_rate", "window_s", "dim")


@dataclass
class EncoderManifest:
    name: str
    version: str
    sha256: str
    license: str
    sample_rate: int
    window_s: float
    dim: int
    input_name: str = "waveform"
    output_name: str = "embedding"
    tokens_output_name: str | None = None

    @classmethod
    def read(cls, directory: Path) -> EncoderManifest:
        data = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        missing = [k for k in MANIFEST_KEYS if k not in data]
        if missing:
            raise ValueError(f"manifest.json incomplet : {missing}")
        return cls(**data)

    def write(self, directory: Path) -> None:
        (Path(directory) / "manifest.json").write_text(
            json.dumps(self.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxEncoder(BaseEncoder):
    def __init__(self, directory: Path, batch_size: int = 16, threads: int | None = None):
        import onnxruntime as ort

        directory = Path(directory)
        manifest = EncoderManifest.read(directory)
        model_path = directory / "model.onnx"
        if file_sha256(model_path) != manifest.sha256:
            raise ValueError(f"SHA-256 de {model_path} différent du manifeste : paquet corrompu")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._manifest = manifest
        self.name, self.version = manifest.name, manifest.version
        self.sample_rate, self.window_s, self.dim = manifest.sample_rate, manifest.window_s, manifest.dim
        self.has_tokens = manifest.tokens_output_name is not None
        self.batch_size = batch_size

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        (out,) = self._session.run(
            [self._manifest.output_name], {self._manifest.input_name: batch}
        )
        return out

    def _forward_tokens(self, batch: np.ndarray) -> np.ndarray | None:
        (out,) = self._session.run(
            [self._manifest.tokens_output_name], {self._manifest.input_name: batch}
        )
        return out
