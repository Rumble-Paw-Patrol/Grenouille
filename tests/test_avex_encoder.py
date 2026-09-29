import pytest

from blanci.encoders.avex_encoder import remap_fairseq_eat

MODEL_KEYS = [
    "backbone.model.extra_tokens",
    "backbone.model.local_encoder.proj.weight",
    "backbone.model.pre_norm.weight",
    "backbone.model.blocks.0.attn.qkv.weight",
]


def test_remap_fairseq_eat():
    state = {
        "modality_encoders.IMAGE.extra_tokens": 1,
        "modality_encoders.IMAGE.local_encoder.proj.weight": 2,
        "modality_encoders.IMAGE.context_encoder.norm.weight": 3,
        "blocks.0.attn.qkv.weight": 4,
        "modality_encoders.IMAGE.decoder.blocks.0.0.weight": 5,
    }
    assert remap_fairseq_eat(state, MODEL_KEYS) == dict(zip(MODEL_KEYS, [1, 2, 3, 4], strict=True))


def test_remap_fairseq_eat_ignore_format_avex():
    assert remap_fairseq_eat({"backbone.model.blocks.0.attn.qkv.weight": 1}, MODEL_KEYS) is None


def test_remap_fairseq_eat_cle_absente():
    with pytest.raises(ValueError):
        remap_fairseq_eat({"modality_encoders.IMAGE.extra_tokens": 1}, MODEL_KEYS)
