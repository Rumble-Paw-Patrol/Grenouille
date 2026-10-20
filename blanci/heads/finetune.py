"""Emplacement du fine-tuning et du LoRA (§3, DECISIONS n° 97) — à programmer plus tard.

EN ATTENTE — emplacement gardé pour plus tard, volontairement non branché ; ne pas signaler
comme code mort à l'audit.

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

Régularisations (`blanci/heads/regularization/`, DECISIONS n° 121, 123), toutes décidées :

- R60, LoRA sur les **couches hautes seulement** (`finetune.lora.layers` derniers blocs, rang
  `finetune.lora.rank`). Les couches basses d'un encodeur apprennent des choses génériques
  (débuts de sons, harmoniques, textures), les hautes des concepts propres à ses données
  d'origine (les oiseaux) : notre écart est en haut. Adapter le haut seulement : moins de
  paramètres (régularisation), calcul bien moindre (la sortie des couches gelées se calcule une
  fois), connaissances générales préservées. Nombre de couches et rang : deux réglages de
  capacité à comparer.
- R61, pas d'apprentissage par couche (`layerwise_lr_groups`, `finetune.llrd_decay`) et dégel
  progressif (`unfreezing_schedule`, `unfreeze_top`, `finetune.unfreeze_every`).
- R62, L2-SP (`snapshot` avant l'adaptation, puis `l2_sp_model_penalty`, `finetune.l2_sp`).
- Boucle d'entraînement `optimise` : R41 (AdamW), R42 (arrêt précoce, `fit_with_options`),
  R46 (`dropout`), R59 (warm-up, écrêtage du gradient), R64 (moyenne des poids).
- R65 (accord de Léonard) : ajuster d'abord sur AnuraSet, puis sur nos labels, et mesurer si
  cela aide.
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
