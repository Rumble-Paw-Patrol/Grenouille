from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from blanci.config import load_config


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


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = {key: str(tmp_path / "data" / key) for key in cfg["paths"]}
    cfg["paths"]["db"] = str(tmp_path / "data" / "db" / "blanci.sqlite")
    return cfg
