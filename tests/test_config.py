from dataclasses import replace

import pytest

from mini_llm.config import (
    EngineConfig,
    ModelConfig,
    ModelSettings,
    SamplingParams,
)


def qwen2_config(**overrides: object) -> ModelConfig:
    values = {
        "vocab_size": 151936,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_loads_expected_local_qwen2_config() -> None:
    settings = ModelSettings()
    config = ModelConfig.from_pretrained(settings)

    assert config.hidden_size == 896
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 14
    assert config.num_key_value_heads == 2
    assert config.head_dim == 64
    assert config.num_key_value_groups == 7
    assert config.intermediate_size == 4864
    assert config.vocab_size == 151936
    assert config.rope_theta == 1_000_000.0
    assert config.rms_norm_eps == 1e-6
    assert config.tie_word_embeddings


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_size", 895, "hidden_size"),
        ("num_key_value_heads", 3, "num_attention_heads"),
    ],
)
def test_rejects_invalid_model_dimensions(
    field: str,
    value: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        qwen2_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("block_size", 0),
        ("block_size", -1),
        ("gpu_memory_utilization", 0.0),
        ("gpu_memory_utilization", 1.1),
    ],
)
def test_rejects_invalid_engine_fields(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        EngineConfig(**{field: value})


def test_rejects_model_length_above_model_limit() -> None:
    config = qwen2_config()
    engine = EngineConfig(max_model_len=config.max_position_embeddings + 1)

    with pytest.raises(ValueError, match="max_model_len"):
        engine.validate(config)


@pytest.mark.parametrize("world_size", [3, 7])
def test_rejects_world_size_that_cannot_shard_heads(world_size: int) -> None:
    config = qwen2_config()
    engine = EngineConfig(world_size=world_size)

    with pytest.raises(ValueError, match="world_size"):
        engine.validate(config)


def test_accepts_qwen2_single_and_dual_gpu_configs() -> None:
    config = qwen2_config()

    EngineConfig(world_size=1).validate(config)
    EngineConfig(world_size=2).validate(config)


def test_sampling_params_validation() -> None:
    params = SamplingParams()
    assert params.temperature == 0.0
    assert params.max_tokens == 10
    assert not params.ignore_eos
    assert params.seed == 0

    with pytest.raises(ValueError, match="temperature"):
        replace(params, temperature=-0.1)
    with pytest.raises(ValueError, match="max_tokens"):
        replace(params, max_tokens=0)
