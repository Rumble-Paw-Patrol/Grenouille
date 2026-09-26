"""Emplacement du fine-tuning et du LoRA (§3, DECISIONS n° 97) — à programmer plus tard.

Adapter l'encodeur de fondation lui-même, en tout (fine-tuning) ou en partie (LoRA : de petites
matrices de rang faible ajoutées aux projections d'attention, le reste gelé). §3 : demande des
centaines à des milliers d'annotations (AnuraSet aidera), risque de sur-apprentissage au site,
conditionné et hors chemin critique.

Contrat de sortie : un **encodeur** (audio → embedding), exporté en paquet ONNX
(`encoders/export.py`, test d'équivalence cosinus > 0,99) sous un nouveau nom
(`<encodeur>-ft<version>`). Il se branche alors partout comme les autres : `blanci embed`,
têtes, fusion, benchmark complet ; rien d'autre à écrire.

Contrainte d'évaluation, à ne pas contourner : un encodeur adapté sur tous les labels ne peut
pas être jugé sur ces mêmes labels (fuite : la validation croisée des têtes paraîtrait
excellente). Deux voies honnêtes :
- adapter **pli par pli** (`finetune.per_fold`), un encodeur par pli des plis communs, et
  encoder chaque micro testé avec l'encodeur qui ne l'a pas vu ;
- ou ne juger que sur ce qui n'a jamais servi : jeu gelé, nouveaux sites (Trésor, Kaw).

Réglages réservés : section `finetune` de la config (méthode, rang et alpha du LoRA, modules
ciblés, époques, pas d'apprentissage).

Régularisations (`blanci/regularization.py`, DECISIONS n° 121) : R60, rang du LoRA réglable
(`finetune.lora.rank`) ; R62, L2-SP : `l2_sp_penalty` vers les poids pré-entraînés ; boucle
`optimise` avec R41, R42 (`fit_with_options`), R46 (`dropout`), R59 (warm-up, écrêtage). R65
(accord de Léonard) : ajuster d'abord sur AnuraSet, puis sur nos labels.
"""

from __future__ import annotations

import sqlite3

PLANNED = "fine-tuning / LoRA : emplacement réservé, pas encore programmé (§3, conditionné)"
METHODS = ("lora", "full")


def finetune_encoder(
    con: sqlite3.Connection,
    cfg: dict,
    encoder: str,
    method: str = "lora",
    fold: int | None = None,
):
    """Adapte `encoder` sur les labels (hors micros du pli `fold` s'il est donné) et rend le
    dossier du paquet ONNX de l'encodeur adapté. À écrire."""
    if method not in METHODS:
        raise ValueError(f"méthode inconnue : {method!r} (connues : {METHODS})")
    raise NotImplementedError(PLANNED)
