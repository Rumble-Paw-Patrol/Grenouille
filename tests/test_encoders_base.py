import numpy as np
import pytest

from blanci.encoders.base import BaseEncoder, Encoder, encoder_id

MODEL_SR = 16000
WINDOW_S = 3.0


class MeanEncoder(BaseEncoder):
    """Encodeur factice : renvoie des statistiques du lot, pour vérifier ce qu'il a reçu."""

    name = "mean"
    version = "1"
    sample_rate = MODEL_SR
    window_s = WINDOW_S
    dim = 3
    has_tokens = False

    def __init__(self):
        self.seen: list[np.ndarray] = []

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        self.seen.append(batch)
        return np.stack([batch.mean(1), batch.std(1), (batch != 0).sum(1)], axis=1)


class TokenEncoder(MeanEncoder):
    name = "tokens"
    has_tokens = True
    n_tokens = 4

    def _forward_tokens(self, batch: np.ndarray) -> np.ndarray:
        return np.zeros((len(batch), self.n_tokens, self.dim), dtype=np.float32)


def windows(n, sr, duration_s=WINDOW_S, value=0.5):
    return np.full((n, round(sr * duration_s)), value, dtype=np.float32)


# --- Contrat ----------------------------------------------------------------------------------


def test_base_encoder_satisfies_the_protocol():
    assert isinstance(MeanEncoder(), Encoder)


def test_encoder_id_joins_name_and_version():
    assert encoder_id(MeanEncoder()) == "mean-1"


def test_forward_must_be_implemented():
    class Bare(BaseEncoder):
        name, version, sample_rate, window_s, dim = "bare", "1", MODEL_SR, WINDOW_S, 2

    with pytest.raises(NotImplementedError):
        Bare().embed(windows(1, MODEL_SR), MODEL_SR)


# --- Rééchantillonnage : il se fait ici, jamais chez l'appelant (§13.7) -----------------------


def test_native_rate_passes_through_untouched():
    encoder = MeanEncoder()
    batch = windows(4, MODEL_SR)
    encoder.embed(batch, MODEL_SR)
    assert np.allclose(encoder.seen[0], batch)


def test_higher_rate_is_resampled_to_the_model_rate():
    """Song Meter Mini 2 à 32 kHz → modèle à 16 kHz : c'est le wrapper qui convertit."""
    encoder = MeanEncoder()
    encoder.embed(windows(4, 32000), 32000)
    assert encoder.seen[0].shape == (4, round(MODEL_SR * WINDOW_S))


def test_lower_rate_is_resampled_up():
    encoder = MeanEncoder()
    encoder.embed(windows(2, 8000), 8000)
    assert encoder.seen[0].shape == (2, round(MODEL_SR * WINDOW_S))


def test_resampling_preserves_a_constant_signal():
    encoder = MeanEncoder()
    emb = encoder.embed(windows(2, 48000, value=0.5), 48000)
    assert emb[:, 0] == pytest.approx(0.5, abs=0.01)


def test_a_single_window_may_be_one_dimensional():
    encoder = MeanEncoder()
    emb = encoder.embed(np.full(round(MODEL_SR * WINDOW_S), 0.25, dtype=np.float32), MODEL_SR)
    assert emb.shape == (1, 3)


# --- Complément de zéros ----------------------------------------------------------------------


def test_short_window_is_zero_padded():
    """Fenêtre de fin plus courte : complétée par des zéros, jamais rejetée."""
    encoder = MeanEncoder()
    short = np.ones((1, MODEL_SR), dtype=np.float32)  # 1 s au lieu de 3
    emb = encoder.embed(short, MODEL_SR)
    assert encoder.seen[0].shape == (1, round(MODEL_SR * WINDOW_S))
    assert emb[0, 2] == pytest.approx(MODEL_SR)  # seuls les échantillons réels sont non nuls


def test_long_window_is_truncated():
    encoder = MeanEncoder()
    encoder.embed(np.ones((1, MODEL_SR * 5), dtype=np.float32), MODEL_SR)
    assert encoder.seen[0].shape == (1, round(MODEL_SR * WINDOW_S))


def test_resampling_rounding_is_absorbed():
    """44,1 kHz → 16 kHz ne tombe pas juste : l'écart d'arrondi ne doit pas propager."""
    encoder = MeanEncoder()
    encoder.embed(windows(3, 44100), 44100)
    assert encoder.seen[0].shape == (3, round(MODEL_SR * WINDOW_S))


# --- Lots -------------------------------------------------------------------------------------


def test_embedding_is_batched_but_seamless():
    encoder = MeanEncoder()
    encoder.batch_size = 8
    emb = encoder.embed(windows(20, MODEL_SR), MODEL_SR)
    assert emb.shape == (20, 3)
    assert [len(b) for b in encoder.seen] == [8, 8, 4]


def test_empty_input_gives_an_empty_embedding():
    encoder = MeanEncoder()
    emb = encoder.embed(np.zeros((0, round(MODEL_SR * WINDOW_S)), dtype=np.float32), MODEL_SR)
    assert emb.shape == (0, 3)


def test_embedding_is_float32():
    emb = MeanEncoder().embed(windows(2, MODEL_SR).astype(np.float64), MODEL_SR)
    assert emb.dtype == np.float32


# --- Tokens -----------------------------------------------------------------------------------


def test_tokens_are_none_when_unsupported():
    assert MeanEncoder().embed_tokens(windows(2, MODEL_SR), MODEL_SR) is None


def test_tokens_are_returned_when_supported():
    """Nécessaire à l'attentive probing (§3) : les tokens avant agrégation."""
    tokens = TokenEncoder().embed_tokens(windows(5, 32000), 32000)
    assert tokens.shape == (5, TokenEncoder.n_tokens, TokenEncoder.dim)
