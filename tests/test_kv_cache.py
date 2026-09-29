import pytest
import torch

from mini_llm.config import ModelConfig
from mini_llm.engine import KVCache
from mini_llm.models import Qwen2p5ForCausalLM


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )


def allocate_cache(
    config: ModelConfig,
    max_model_len: int,
    dtype: torch.dtype = torch.float32,
) -> KVCache:
    return KVCache.allocate(
        config,
        batch_size=1,
        max_model_len=max_model_len,
        device="cpu",
        dtype=dtype,
    )


def test_allocates_one_contiguous_cache_per_layer(
    small_config: ModelConfig,
) -> None:
    cache = allocate_cache(small_config, 128, torch.bfloat16)
    expected_shape = (1, 1, 128, 4)

    assert cache.length == 0
    assert cache.batch_size == 1
    assert cache.max_model_len == 128
    assert len(cache.keys) == small_config.num_hidden_layers
    assert len(cache.values) == small_config.num_hidden_layers
    for key, value in zip(cache.keys, cache.values):
        assert key.shape == expected_shape
        assert value.shape == expected_shape
        assert key.dtype == torch.bfloat16
        assert value.dtype == torch.bfloat16
        assert key.device.type == "cpu"
        assert value.device.type == "cpu"
        assert key.is_contiguous()
        assert value.is_contiguous()


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 127, 128])
def test_prefill_matches_full_forward(
    small_config: ModelConfig,
    seq_len: int,
) -> None:
    torch.manual_seed(42)
    model = Qwen2p5ForCausalLM(small_config).eval()
    input_ids = torch.randint(0, small_config.vocab_size, (1, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0)
    cache = allocate_cache(small_config, seq_len)

    with torch.inference_mode():
        expected = model.compute_logits(model(input_ids, positions))[:, -1]
        actual = model.prefill(input_ids, cache)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert cache.length == seq_len


@pytest.mark.parametrize("seq_len", [1, 15, 16, 17, 127, 128])
def test_cached_decode_matches_full_recomputation_and_updates_one_slot(
    small_config: ModelConfig,
    seq_len: int,
) -> None:
    torch.manual_seed(42)
    model = Qwen2p5ForCausalLM(small_config).eval()
    input_ids = torch.randint(0, small_config.vocab_size, (1, seq_len))
    cache = allocate_cache(small_config, seq_len)

    with torch.inference_mode():
        actual = model.prefill(input_ids[:, :1], cache)
        expected = model.compute_logits(
            model(input_ids[:, :1], torch.zeros(1, 1, dtype=torch.long))
        )[:, -1]
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

        for index in range(1, seq_len):
            previous_keys = [key.clone() for key in cache.keys]
            previous_values = [value.clone() for value in cache.values]
            actual = model.decode(input_ids[:, index:index + 1], cache)
            positions = torch.arange(index + 1).unsqueeze(0)
            expected = model.compute_logits(
                model(input_ids[:, :index + 1], positions)
            )[:, -1]

            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            assert cache.length == index + 1
            for layer_index in range(small_config.num_hidden_layers):
                key = cache.keys[layer_index]
                value = cache.values[layer_index]
                old_key = previous_keys[layer_index]
                old_value = previous_values[layer_index]
                torch.testing.assert_close(
                    key[:, :, :index], old_key[:, :, :index]
                )
                torch.testing.assert_close(
                    value[:, :, :index], old_value[:, :, :index]
                )
                assert not torch.equal(
                    key[:, :, index:index + 1],
                    old_key[:, :, index:index + 1],
                )
                assert not torch.equal(
                    value[:, :, index:index + 1],
                    old_value[:, :, index:index + 1],
                )
                torch.testing.assert_close(
                    key[:, :, index + 1:], old_key[:, :, index + 1:]
                )
                torch.testing.assert_close(
                    value[:, :, index + 1:], old_value[:, :, index + 1:]
                )


def test_rejects_invalid_prefill_and_decode_boundaries(
    small_config: ModelConfig,
) -> None:
    model = Qwen2p5ForCausalLM(small_config).eval()
    cache = allocate_cache(small_config, 1)

    with torch.inference_mode():
        model.prefill(torch.tensor([[1]]), cache)

    with pytest.raises(ValueError, match="empty KV cache"):
        model.prefill(torch.tensor([[1]]), cache)
    with pytest.raises(ValueError, match="capacity exceeded"):
        model.decode(torch.tensor([[1]]), cache)
    with pytest.raises(ValueError, match=r"shape \(batch, 1\)"):
        model.decode(torch.tensor([[1, 2]]), allocate_cache(small_config, 2))
    with pytest.raises(ValueError, match="batch size"):
        model.prefill(
            torch.tensor([[1], [2]]),
            allocate_cache(small_config, 1),
        )
