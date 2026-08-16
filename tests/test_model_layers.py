import pytest
import torch

from mini_llm.model import RMSNorm


@pytest.mark.parametrize("hidden_size", [8, 32, 96])
@pytest.mark.parametrize("sequence_length", [1, 4, 11])
def test_rms_norm_matches_formula(
    hidden_size: int,
    sequence_length: int,
) -> None:
    torch.manual_seed(hidden_size + sequence_length)
    hidden_states = torch.randn(1, sequence_length, hidden_size)
    layer = RMSNorm(hidden_size, eps=1e-6)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(hidden_size))

    actual = layer(hidden_states)
    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    expected = (
        hidden_states.float()
        * torch.rsqrt(variance + layer.eps)
        * layer.weight
    ).to(hidden_states.dtype)

    torch.testing.assert_close(actual, expected)
