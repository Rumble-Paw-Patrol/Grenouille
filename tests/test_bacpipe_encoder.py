"""Adaptateur bacpipe : conversion des sorties, sans télécharger de modèle (faux modèle)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from blanci.encoders.bacpipe_encoder import BacpipeEncoder  # noqa: E402


class FakeModel:
    """Imite un extracteur bacpipe : preprocess(lot torch) puis appel → tenseur."""

    def __init__(self, tokens):
        self.tokens = tokens

    def preprocess(self, audio):
        assert isinstance(audio, torch.Tensor)
        return audio

    def __call__(self, x):
        feats = torch.stack([x.mean(dim=1), x.std(dim=1)], dim=1)  # (lot, 2)
        if self.tokens:
            return torch.stack([feats, 3 * feats], dim=1)  # (lot, 2 jetons, 2)
        return feats


def encoder(tokens):
    enc = BacpipeEncoder.__new__(BacpipeEncoder)
    enc._model = FakeModel(tokens)
    enc.sample_rate, enc.window_s, enc.batch_size = 100, 1.0, 4
    probe = enc._raw(np.zeros((1, 100), np.float32))
    enc.has_tokens, enc.dim = probe.ndim == 3, int(probe.shape[-1])
    return enc


def test_pooled_output_is_passed_through():
    enc = encoder(tokens=False)
    x = np.random.default_rng(0).normal(size=(6, 100)).astype(np.float32)
    out = enc.embed(x, 100)
    assert out.shape == (6, 2) and not enc.has_tokens
    assert enc.embed_tokens(x, 100) is None


def test_token_sequences_are_averaged_and_kept_for_later():
    enc = encoder(tokens=True)
    x = np.random.default_rng(0).normal(size=(6, 100)).astype(np.float32)
    out = enc.embed(x, 100)
    assert enc.has_tokens and enc.dim == 2 and out.shape == (6, 2)
    assert np.allclose(out[:, 1], 2 * x.std(axis=1, ddof=1), rtol=1e-4)  # moyenne de 1× et 3×
    assert enc.embed_tokens(x, 100).shape == (6, 2, 2)


class FakePerch(FakeModel):
    """Imite perch_v2 : garde les logits de toutes ses classes après chaque appel."""

    classes = ["Otophryne pyburni", "Anomaloglossus stepheni", "Anomaloglossus surinamensis"]

    def __call__(self, x):
        self.results = {"logits": torch.stack([x.mean(1), x.max(1).values, x.min(1).values], 1)}
        return super().__call__(x)


def test_congener_logits_are_kept_without_a_second_inference():
    enc = BacpipeEncoder.__new__(BacpipeEncoder)
    enc._model = FakePerch(tokens=False)
    enc.name, enc.sample_rate, enc.window_s, enc.batch_size = "perch", 100, 1.0, 4
    enc.logit_names = ["Anomaloglossus stepheni", "Anomaloglossus surinamensis"]
    enc._logit_index = enc._resolve_classes(enc.logit_names)
    enc._logits = []
    x = np.random.default_rng(0).normal(size=(6, 100)).astype(np.float32)
    enc.embed(x, 100)  # deux lots : 4 + 2
    logits = enc.pop_logits()
    assert logits.shape == (6, 2)
    assert np.allclose(logits[:, 0], x.max(axis=1)) and np.allclose(logits[:, 1], x.min(axis=1))
    assert enc.pop_logits().shape == (0, 2)  # vidé après lecture


def test_an_unknown_congener_class_is_refused():
    enc = BacpipeEncoder.__new__(BacpipeEncoder)
    enc._model, enc.name = FakePerch(tokens=False), "perch"
    with pytest.raises(ValueError, match="Anomaloglossus blanci"):
        enc._resolve_classes(["Anomaloglossus blanci"])
