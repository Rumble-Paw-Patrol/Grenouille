"""Chargement de la configuration YAML.

config/default.yaml, surchargée par un fichier utilisateur passé à `--config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG = PROJECT_ROOT / "config" / "local.yaml"


def default_user_config() -> Path | None:
    """config/local.yaml s'il existe : la commande et le poste le prennent sans `--config`."""
    return LOCAL_CONFIG if LOCAL_CONFIG.is_file() else None


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` par-dessus `base`, section par section. Une section vide dans le fichier
    utilisateur (`qc:` suivi de commentaires seulement, lue None) garde celle par défaut au
    lieu de la remplacer par None."""
    out = dict(base)
    for key, value in override.items():
        if value is None and isinstance(out.get(key), dict):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "sequential" in user:  # ancien nom de la section signal_processing (n° 175)
            user.setdefault("signal_processing", user.pop("sequential"))
        cfg = _merge(cfg, user)
    return cfg


def project_path(value: str | Path) -> Path:
    """Chemin de la configuration : un chemin relatif (« data/db/blanci.sqlite ») part de la
    racine du projet, pas du dossier d'où la commande est lancée ; lancée depuis D:\, elle
    écrirait sinon sur le disque externe (DECISIONS n° 144)."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_path(cfg: dict[str, Any], key: str) -> Path:
    return project_path(cfg["paths"][key])
