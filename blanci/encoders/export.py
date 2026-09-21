"""Export torch → ONNX, quantification int8 dynamique, test d'équivalence (M5, §7).

Critère d'acceptation (§13.6) : cosinus > 0,99 entre embeddings torch et ONNX (quantifié
compris) sur des fenêtres réelles. torch n'est importé qu'ici, jamais par le livrable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blanci.encoders.base import Encoder
from blanci.encoders.onnx_encoder import EncoderManifest, OnnxEncoder, file_sha256


def export_onnx(
    module,
    directory: Path,
    manifest: EncoderManifest,
    opset: int = 17,
) -> Path:
    """Exporte un `torch.nn.Module` forme d'onde (lot, échantillons) → embedding (lot, dim)."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.onnx"
    n = round(manifest.window_s * manifest.sample_rate)
    module.eval()
    torch.onnx.export(
        module,
        torch.zeros(2, n),
        str(path),
        input_names=[manifest.input_name],
        output_names=[manifest.output_name],
        dynamic_axes={manifest.input_name: {0: "batch"}, manifest.output_name: {0: "batch"}},
        opset_version=opset,
    )
    manifest.sha256 = file_sha256(path)
    manifest.write(directory)
    return path


def quantize_int8(directory: Path, out_directory: Path) -> Path:
    """Quantification dynamique int8 (poids) ; copie le manifeste avec la nouvelle empreinte."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_directory = Path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)
    out = out_directory / "model.onnx"
    quantize_dynamic(str(Path(directory) / "model.onnx"), str(out), weight_type=QuantType.QInt8)
    manifest = EncoderManifest.read(directory)
    manifest.version = f"{manifest.version}-int8"
    manifest.sha256 = file_sha256(out)
    manifest.write(out_directory)
    return out


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def check_equivalence(
    reference: Encoder, directory: Path, windows: np.ndarray, sr: int, min_cosine: float = 0.99
) -> dict[str, float | bool]:
    """Compare l'encodeur de référence (torch) au paquet ONNX sur les mêmes fenêtres."""
    cos = cosine_rows(reference.embed(windows, sr), OnnxEncoder(directory).embed(windows, sr))
    return {
        "min_cosine": float(cos.min()),
        "mean_cosine": float(cos.mean()),
        "passed": bool(cos.min() > min_cosine),
    }
