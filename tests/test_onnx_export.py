"""Chemin du livrable (§7, §13.6) : export torch → ONNX, `OnnxEncoder`, critère cosinus > 0,99.

Un petit réseau forme d'onde → embedding tient lieu d'encodeur : ce qui se teste ici est le
paquet (manifeste, empreinte, axes dynamiques, lots), pas un modèle réel (DECISIONS n° 145).
"""

import json

import numpy as np
import pytest

from blanci.encoders.base import BaseEncoder
from blanci.encoders.onnx_encoder import EncoderManifest, OnnxEncoder

torch = pytest.importorskip("torch")  # groupe research : export seulement, jamais le livrable
pytest.importorskip("onnx")  # torch.onnx.export et la quantification en ont besoin

pytestmark = pytest.mark.slow  # pytest -m 'not slow' : suite rapide

SR, WINDOW_S, DIM = 8000, 0.5, 6


class TinyNet(torch.nn.Module):
    """Forme d'onde (lot, échantillons) → embedding (lot, DIM) : trame, projection, moyenne."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.frame = 200
        self.proj = torch.nn.Linear(self.frame, DIM)

    def forward(self, wav):
        frames = wav.reshape(wav.shape[0], -1, self.frame)
        return torch.tanh(self.proj(frames)).mean(dim=1)


class TorchEncoder(BaseEncoder):
    name, version, sample_rate, window_s, dim = "tiny", "1", SR, WINDOW_S, DIM

    def __init__(self, net):
        self.net = net

    def _forward(self, batch):
        with torch.no_grad():
            return self.net(torch.from_numpy(batch)).numpy()


def manifest():
    return EncoderManifest("tiny", "1", "", "test", SR, WINDOW_S, DIM)


@pytest.fixture
def package(tmp_path):
    from blanci.encoders.export import export_onnx

    net = TinyNet()
    export_onnx(net, tmp_path / "tiny-1", manifest())
    return net, tmp_path / "tiny-1"


def windows(n=5, sr=SR, seed=0):
    return np.random.default_rng(seed).normal(0, 0.1, (n, int(WINDOW_S * sr))).astype(np.float32)


def test_exported_package_matches_torch_on_any_batch_size(package):
    from blanci.encoders.export import check_equivalence

    net, directory = package
    result = check_equivalence(TorchEncoder(net), directory, windows(5), SR)
    assert result["passed"] and result["min_cosine"] > 0.999
    onnx = OnnxEncoder(directory, batch_size=2)  # lots de 2, 2 et 1 : axe dynamique
    assert onnx.embed(windows(5), SR).shape == (5, DIM)
    assert onnx.name == "tiny" and onnx.dim == DIM and not onnx.has_tokens


def test_onnx_encoder_resamples_like_every_encoder(package):
    """Fenêtres à 16 kHz pour un modèle à 8 kHz : le rééchantillonnage est fait par l'encodeur."""
    net, directory = package
    ref = TorchEncoder(net).embed(windows(3, sr=16000), 16000)
    out = OnnxEncoder(directory).embed(windows(3, sr=16000), 16000)
    assert np.allclose(ref, out, atol=1e-4)


def test_a_corrupted_package_is_refused(package):
    _, directory = package
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data["sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        OnnxEncoder(directory)


def test_an_incomplete_manifest_is_refused(package):
    _, directory = package
    (directory / "manifest.json").write_text(json.dumps({"name": "tiny"}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplet"):
        OnnxEncoder(directory)


def test_int8_quantization_keeps_the_embeddings(package, tmp_path):
    from blanci.encoders.export import check_equivalence, quantize_int8

    net, directory = package
    quantize_int8(directory, tmp_path / "tiny-1-int8")
    result = check_equivalence(TorchEncoder(net), tmp_path / "tiny-1-int8", windows(5), SR)
    assert result["mean_cosine"] > 0.99
    assert EncoderManifest.read(tmp_path / "tiny-1-int8").version == "1-int8"
