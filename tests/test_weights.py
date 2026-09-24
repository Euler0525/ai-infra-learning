from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mini_llm.config import ModelConfig
from mini_llm.models import Qwen2p5ForCausalLM
from mini_llm.utils import load_safetensors_weights


@pytest.fixture
def small_model() -> Qwen2p5ForCausalLM:
    config = ModelConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    return Qwen2p5ForCausalLM(config)


def make_checkpoint(
    model: Qwen2p5ForCausalLM,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.full(
            parameter.shape,
            index + 1,
            dtype=torch.bfloat16,
        )
        for index, (name, parameter) in enumerate(model.named_parameters())
    }


def test_loads_all_weights_and_converts_dtype(
    tmp_path: Path,
    small_model: Qwen2p5ForCausalLM,
) -> None:
    checkpoint = make_checkpoint(small_model)
    path = tmp_path / "model.safetensors"
    save_file(checkpoint, path)

    report = load_safetensors_weights(small_model, path)

    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    for name, parameter in small_model.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, checkpoint[name].float())
    assert small_model.lm_head.weight is small_model.model.embed_tokens.weight
    assert (
        small_model.lm_head.weight.data_ptr()
        == small_model.model.embed_tokens.weight.data_ptr()
    )


@pytest.mark.parametrize("invalid_key", ["missing", "unexpected"])
def test_rejects_incompatible_keys(
    tmp_path: Path,
    small_model: Qwen2p5ForCausalLM,
    invalid_key: str,
) -> None:
    checkpoint = make_checkpoint(small_model)
    if invalid_key == "missing":
        checkpoint.pop("model.embed_tokens.weight")
    else:
        checkpoint["unexpected.weight"] = torch.zeros(1)
    path = tmp_path / f"{invalid_key}.safetensors"
    save_file(checkpoint, path)

    with pytest.raises(ValueError, match=rf"{invalid_key}_keys=.*"):
        load_safetensors_weights(small_model, path)


def test_rejects_shape_mismatch_before_loading(
    tmp_path: Path,
    small_model: Qwen2p5ForCausalLM,
) -> None:
    checkpoint = make_checkpoint(small_model)
    checkpoint["model.norm.weight"] = torch.zeros(1, dtype=torch.bfloat16)
    path = tmp_path / "wrong-shape.safetensors"
    save_file(checkpoint, path)
    original_embedding = small_model.model.embed_tokens.weight.detach().clone()

    with pytest.raises(ValueError, match="shape mismatch for 'model.norm.weight'"):
        load_safetensors_weights(small_model, path)

    torch.testing.assert_close(
        small_model.model.embed_tokens.weight,
        original_embedding,
    )
