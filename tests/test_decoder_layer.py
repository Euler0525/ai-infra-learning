import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RotaryEmbedding,
)

from mini_llm.config import ModelConfig, ModelSettings
from mini_llm.layers import DecoderLayer, SwiGLU


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=56,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.mark.parametrize("shape", [(3, 56), (2, 4, 56)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_matches_reference(
    small_config: ModelConfig, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    torch.manual_seed(42)
    mlp = SwiGLU(small_config).to(dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    gate = F.linear(x, mlp.gate_proj.weight)
    up = F.linear(x, mlp.up_proj.weight)
    expected = F.linear(F.silu(gate) * up, mlp.down_proj.weight)

    tolerance = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(
        mlp(x), expected, rtol=tolerance[0], atol=tolerance[1]
    )


def test_decoder_residual_paths(small_config: ModelConfig) -> None:
    layer = DecoderLayer(small_config)
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    x = torch.randn(2, 4, small_config.hidden_size)
    positions = torch.arange(4).expand(2, -1)

    torch.testing.assert_close(layer(x, positions), x)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.bfloat16)],
)
def test_matches_hugging_face_decoder_layer(device: str, dtype: torch.dtype) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    settings = ModelSettings()
    hf_config = AutoConfig.from_pretrained(
        settings.model_id, revision=settings.revision, local_files_only=True
    )
    hf_config._attn_implementation = "eager"
    config = ModelConfig.from_pretrained(settings)
    torch.manual_seed(42)
    hf_layer = Qwen2DecoderLayer(hf_config, layer_idx=0).to(
        device=device, dtype=dtype
    ).eval()
    layer = DecoderLayer(config).to(device=device, dtype=dtype).eval()
    layer.load_state_dict(hf_layer.state_dict())
    hf_rope = Qwen2RotaryEmbedding(hf_config).to(device)

    x = torch.randn(2, 5, config.hidden_size, device=device, dtype=dtype)
    positions = torch.arange(5, device=device).expand(2, -1)
    cos, sin = hf_rope(x, positions)
    mask = torch.full(
        (2, 1, 5, 5), torch.finfo(dtype).min, device=device, dtype=dtype
    ).triu(1)

    with torch.no_grad():
        actual = layer(x, positions)
        expected = hf_layer(
            x,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=(cos, sin),
        )
    tolerance = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1]
    )
