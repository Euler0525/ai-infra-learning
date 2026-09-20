import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from mini_llm.config import ModelSettings
from mini_llm.layers import RotaryEmbedding


@pytest.fixture(scope="module")
def rope_config():
    settings = ModelSettings()
    return AutoConfig.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        local_files_only=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matches_hugging_face_for_prefill_and_batched_positions(
    rope_config, dtype: torch.dtype, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(42)
    head_dim = rope_config.hidden_size // rope_config.num_attention_heads
    q = torch.randn(2, 5, rope_config.num_attention_heads, head_dim).to(
        device=device, dtype=dtype
    )
    k = torch.randn(2, 5, rope_config.num_key_value_heads, head_dim).to(
        device=device, dtype=dtype
    )
    positions = torch.tensor(
        [[0, 1, 15, 16, 1024], [16, 15, 1, 0, 1024]], device=device
    )

    rope = RotaryEmbedding(head_dim, rope_config.rope_parameters["rope_theta"])
    actual_q, actual_k = rope.to(device)(q, k, positions)
    cos, sin = Qwen2RotaryEmbedding(rope_config).to(device)(q, positions)
    expected_q, expected_k = apply_rotary_pos_emb(
        q.transpose(1, 2), k.transpose(1, 2), cos, sin
    )

    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        actual_q, expected_q.transpose(1, 2), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        actual_k, expected_k.transpose(1, 2), atol=tolerance, rtol=tolerance
    )


def test_decode_position_matches_prefill_and_zero_is_identity(rope_config) -> None:
    head_dim = rope_config.hidden_size // rope_config.num_attention_heads
    rope = RotaryEmbedding(head_dim, rope_config.rope_parameters["rope_theta"])
    q = torch.randn(1, 17, rope_config.num_attention_heads, head_dim)
    k = torch.randn(1, 17, rope_config.num_key_value_heads, head_dim)

    prefill_q, prefill_k = rope(q, k, torch.arange(17).unsqueeze(0))
    decode_q, decode_k = rope(q[:, 16:17], k[:, 16:17], torch.tensor([[16]]))

    torch.testing.assert_close(decode_q, prefill_q[:, 16:17])
    torch.testing.assert_close(decode_k, prefill_k[:, 16:17])
    torch.testing.assert_close(prefill_q[:, :1], q[:, :1])
    torch.testing.assert_close(prefill_k[:, :1], k[:, :1])


def test_packed_positions(rope_config) -> None:
    head_dim = rope_config.hidden_size // rope_config.num_attention_heads
    rope = RotaryEmbedding(head_dim, rope_config.rope_parameters["rope_theta"])
    q = torch.randn(3, rope_config.num_attention_heads, head_dim)
    k = torch.randn(3, rope_config.num_key_value_heads, head_dim)
    positions = torch.tensor([0, 1, 16])

    packed_q, packed_k = rope(q, k, positions)
    batched_q, batched_k = rope(q.unsqueeze(
        0), k.unsqueeze(0), positions.unsqueeze(0))

    torch.testing.assert_close(packed_q, batched_q.squeeze(0))
    torch.testing.assert_close(packed_k, batched_k.squeeze(0))
