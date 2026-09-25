"""Emplacement du modèle fait maison (§3, DECISIONS n° 97) — à programmer plus tard.

Un réseau conçu et entraîné ici, directement sur les labels (sans professeur, à la différence
de la distillation) : par exemple un petit CNN sur le spectrogramme de la bande de la note, ou
un modèle qui exploite ce que les encodeurs généralistes voient mal — une note de 0,09 s qui
revient toutes les ~1,4 s.

Contrat : celui des détecteurs entraînables (`detectors/base.py`) — `fit(fenêtres, f_e, labels)`
rend un détecteur appris, `score(fenêtres, f_e)` un score par fenêtre. Le banc d'essai
(`blanci detector-bench --detector homemade`) l'entraîne pli par pli sur les plis communs : ses
scores rejoignent le stock hors-pli, le benchmark complet et les ensembles sans autre code.

Contraintes à garder : peu de positifs (51 enregistrements, 13 micros, tous à Mataroni) → risque
fort de sur-apprentissage au site ; augmentation de données (bruit de fond d'autres micros,
gain, décalage temporel) ; exécution finale en ONNX sur l'i5 (§13.1).
"""

from __future__ import annotations

import numpy as np

PLANNED = "modèle fait maison : emplacement réservé, pas encore programmé (§3)"


class HomemadeDetector:
    """Modèle fait maison : `fit` et `score` à écrire."""

    name = "homemade"
    version = "0"

    def __init__(self, cfg: dict | None = None):
        self.cfg = (cfg or {}).get("detectors", {}).get("homemade", {})

    def fit(self, windows: list[np.ndarray], sr: int, y: np.ndarray) -> HomemadeDetector:
        raise NotImplementedError(PLANNED)

    def score(self, windows: list[np.ndarray], sr: int) -> np.ndarray:
        raise NotImplementedError(PLANNED)
