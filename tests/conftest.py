from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from blanci.core.config import load_config
from blanci.embedding.encoders.base import BaseEncoder


@pytest.fixture(autouse=True)
def _no_local_config(monkeypatch, tmp_path):
    """Les tests ne lisent jamais config/local.yaml (vraie base, vrai disque)."""
    monkeypatch.setattr("blanci.core.config.LOCAL_CONFIG", tmp_path / "absent.yaml")


def write_wav(
    path: Path,
    sr: int = 16000,
    duration_s: float = 2.0,
    freq: float = 4750.0,
    channels: int = 1,
    guano: dict[str, str] | None = None,
) -> Path:
    """WAV synthétique (sinus), avec bloc GUANO optionnel comme sur un Song Meter."""
    t = np.arange(int(sr * duration_s)) / sr
    x = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels > 1:
        x = np.stack([x] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, sr, subtype="PCM_16")
    if guano:
        text = "GUANO|Version: 1.0\n" + "".join(f"{k}: {v}\n" for k, v in guano.items())
        payload = text.encode("utf-8")
        chunk = b"guan" + len(payload).to_bytes(4, "little") + payload + b"\0" * (len(payload) % 2)
        data = bytearray(path.read_bytes()) + chunk
        data[4:8] = (len(data) - 8).to_bytes(4, "little")
        path.write_bytes(bytes(data))
    return path


class ToyEncoder(BaseEncoder):
    """Encodeur factice : l'énergie en bande 4–6 kHz suffit à séparer le corpus de test
    (pipeline CLI, AnuraSet)."""

    name = "toy"
    version = "1"
    sample_rate = 16000
    window_s = 3.0
    dim = 8
    has_tokens = False

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(batch, axis=1))
        bands = np.array_split(spectrum, self.dim, axis=1)
        return np.stack([np.log1p(b.mean(axis=1)) for b in bands], axis=1).astype(np.float32)


def quick_cfg(tmp_path: Path, n_boot: int = 30, negatives_per_positive: int = 4) -> dict:
    """Configuration de test rapide : chemins sous `tmp_path`, 3 plis, un seul C, peu de
    tirages bootstrap. Chaque fichier y ajoute ce qui lui est propre."""
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / key) for key in cfg["paths"]}
    cfg["benchmark"] |= {"n_boot": n_boot, "negatives_per_positive": negatives_per_positive}
    cfg["head"] |= {"n_splits": 3, "C_grid": [1.0]}
    return cfg


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / "data" / key) for key in cfg["paths"]}
    cfg["paths"]["db"] = str(tmp_path / "data" / "db" / "blanci.sqlite")
    return cfg
