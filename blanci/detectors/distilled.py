"""Emplacement du modèle de distillation (§3, DECISIONS n° 97) — à programmer plus tard.

Principe : un petit réseau « élève » (CNN sur spectrogramme 3–7 kHz) apprend à imiter la chaîne
« professeur » gelée (encodeur de fondation + tête + fusion), pour tourner seul sur l'i5 de l'ONF
sans encodeur de fondation. Livrable léger, hérite des erreurs du professeur.

Plan (à trancher le moment venu, §13.7 : pas avant le jalon M4) :

1. Professeur : la chaîne adoptée (`blanci score`, tête adoptée, fusion) ; ses scores sont
   calculés sur des fenêtres **non étiquetées** en volume (des centaines de milliers : aucun
   coût d'annotation), plus les fenêtres étiquetées.
2. Cibles : score du professeur (logit, ou probabilité tempérée) ; les labels humains, quand ils
   existent, priment sur le professeur.
3. Élève : spectrogramme log-mel restreint à la bande utile, 3 à 5 couches de convolution,
   quelques centaines de milliers de paramètres ; entraînement torch (groupe `research`).
4. Export ONNX (`encoders/export.py`) avec le spectrogramme dans le graphe ; exécution par
   ONNX Runtime, sans torch ni TensorFlow (§13.1).
5. Jugement : banc d'essai des détecteurs (`blanci detector-bench --detector distilled`), mêmes
   plis que tous les modèles ; écart d'AP au professeur sur le jeu gelé ; débit sur l'i5 (1 h
   d'audio en moins de 10 min, §13.6).

Attention au benchmark : l'élève doit être distillé **pli par pli** à partir d'un professeur
lui-même appris sans le micro testé ; sinon il hérite de scores en-pli et paraît meilleur qu'il
n'est.

Régularisations (`blanci/regularization.py`, DECISIONS n° 121) : perte de distillation
`distillation_loss` (R63, température) ; boucle `optimise` avec R41 (AdamW), R42 (arrêt
précoce, `fit_with_options`), R46 (`dropout`), R59 (warm-up, écrêtage du gradient).
"""

from __future__ import annotations

import numpy as np

PLANNED = "distillation : emplacement réservé, pas encore programmé (§3, après le jalon M4)"


class DistilledDetector:
    """Élève distillé : `fit` (distillation depuis le professeur) et `score`, à écrire."""

    name = "distilled"
    version = "0"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or {}).get("detectors", {}).get("distilled", {})

    def fit(self, windows: list[np.ndarray], sr: int, y: np.ndarray) -> DistilledDetector:
        raise NotImplementedError(PLANNED)

    def score(self, windows: list[np.ndarray], sr: int) -> np.ndarray:
        raise NotImplementedError(PLANNED)
