import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding

from mini_llm.config import ModelConfig, ModelSettings
from mini_llm.layers import GQAAttention


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


def reference_attention(
    attention: GQAAttention, x: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    num_heads = attention.num_heads
    num_kv_heads = attention.num_kv_heads
    head_dim = attention.head_dim
    q = F.linear(x, attention.q_proj.weight, attention.q_proj.bias).reshape(
        batch_size, seq_len, num_heads, head_dim
    )
    k = F.linear(x, attention.k_proj.weight, attention.k_proj.bias).reshape(
        batch_size, seq_len, num_kv_heads, head_dim
    )
    v = F.linear(x, attention.v_proj.weight, attention.v_proj.bias).reshape(
        batch_size, seq_len, num_kv_heads, head_dim
    )

    angles = positions.float()[..., None] * attention.rotary_emb.inv_freq
    cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        left, right = tensor.chunk(2, dim=-1)
        return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)

    q = rotate(q).transpose(1, 2)
    k = rotate(k)
    kv_head_indices = torch.arange(num_heads, device=x.device) // attention.num_key_value_groups
    k = k[:, :, kv_head_indices].transpose(1, 2)
    v = v[:, :, kv_head_indices].transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / head_dim**0.5
    future = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
    weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    output = (weights @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
    return F.linear(output, attention.o_proj.weight)


@pytest.mark.parametrize("seq_len", [1, 5])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matches_pytorch_reference(
    small_config: ModelConfig, seq_len: int, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(42)
    attention = GQAAttention(small_config).to(device)
    x = torch.randn(2, seq_len, small_config.hidden_size, device=device)
    positions = torch.arange(seq_len, device=device).expand(2, -1)

    actual = attention(x, positions)
    expected = reference_attention(attention, x, positions)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_future_tokens_do_not_change_past_output(small_config: ModelConfig) -> None:
    torch.manual_seed(42)
    attention = GQAAttention(small_config)
    x = torch.randn(1, 5, small_config.hidden_size)
    positions = torch.arange(5).unsqueeze(0)
    before = attention(x, positions)
    x[:, -1] += 10
    after = attention(x, positions)

    torch.testing.assert_close(before[:, :-1], after[:, :-1])


def test_seven_query_heads_share_each_kv_head(small_config: ModelConfig) -> None:
    attention = GQAAttention(small_config)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.zero_()
        attention.v_proj.bias[: small_config.head_dim] = 1
        attention.v_proj.bias[small_config.head_dim :] = 2
        attention.o_proj.weight.copy_(torch.eye(small_config.hidden_size))

    x = torch.zeros(1, 1, small_config.hidden_size)
    output = attention(x, torch.zeros(1, 1, dtype=torch.long))
    expected = torch.cat((torch.ones(28), torch.full((28,), 2.0))).view(1, 1, -1)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.bfloat16)],
)
def test_matches_hugging_face_first_layer_attention(device: str, dtype: torch.dtype) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    settings = ModelSettings()
    hf_config = AutoConfig.from_pretrained(
        settings.model_id, revision=settings.revision, local_files_only=True
    )
    hf_config._attn_implementation = "eager"
    config = ModelConfig.from_pretrained(settings)
    torch.manual_seed(42)
    attention = GQAAttention(config).to(device=device, dtype=dtype).eval()
    hf_attention = Qwen2Attention(hf_config, layer_idx=0).to(device=device, dtype=dtype).eval()
    hf_attention.load_state_dict(attention.state_dict())
    hf_rope = Qwen2RotaryEmbedding(hf_config).to(device)

    x = torch.randn(1, 5, config.hidden_size, device=device, dtype=dtype)
    positions = torch.arange(5, device=device).unsqueeze(0)
    cos, sin = hf_rope(x, positions)
    mask = torch.full(
        (1, 1, 5, 5), torch.finfo(dtype).min, device=device, dtype=dtype
    ).triu(1)

    actual = attention(x, positions)
    expected, _ = hf_attention(x, (cos, sin), mask)
    tolerance = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
