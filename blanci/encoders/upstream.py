"""Encodeur précédé des transformations du module séquentiel en amont (`sequential.Upstream`).

Le son est transformé à la f_e d'origine, avant le rééchantillonnage de l'encodeur (comme le
passe-bas du §2). Nom : `<encodeur>+<étiquette>` (« birdmae+bp3-7k ») : son stock
d'embeddings, ses têtes et ses scores sont ceux d'un autre encodeur, comparé par le benchmark.
"""

from __future__ import annotations

import numpy as np

from blanci.encoders.base import Encoder
from blanci.sequential import Upstream


class UpstreamEncoder:
    def __init__(self, inner: Encoder, upstream: Upstream):
        if not upstream.transforms:
            raise ValueError("aucune transformation active : l'encodeur serait inchangé")
        self.inner = inner
        self.upstream = upstream
        self.name = f"{inner.name}+{upstream.transform_tag()}"
        self.version = inner.version
        self.sample_rate = inner.sample_rate
        self.window_s = inner.window_s
        self.dim = inner.dim
        self.has_tokens = inner.has_tokens
        self.logit_names = list(getattr(inner, "logit_names", []) or [])

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        return self.inner.embed(self.upstream.transform(wav, sr), sr)

    def embed_tokens(self, wav: np.ndarray, sr: int) -> np.ndarray | None:
        return self.inner.embed_tokens(self.upstream.transform(wav, sr), sr)

    def pop_logits(self) -> np.ndarray | None:
        """Logits de l'encodeur enveloppé (perch_v2 : congénères), sur le son transformé."""
        pop = getattr(self.inner, "pop_logits", None)
        return pop() if pop else None
