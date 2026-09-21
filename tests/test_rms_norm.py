import pytest
import torch

from mini_llm.layers import RMSNorm


@pytest.mark.parametrize("shape", [(3, 64), (2, 5, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=device)
    norm = RMSNorm(shape[-1], eps=1e-6).to(device)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)

    x_fp32 = x.float()
    expected = (
        norm.weight
        * (
            x_fp32
            * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + norm.eps)
        ).to(dtype)
    ).to(dtype)
    actual = norm(x)

    assert actual.dtype == dtype
    tolerance = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_and_large_values(dtype: torch.dtype, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    norm = RMSNorm(64).to(device)
    zeros = torch.zeros(2, 64, dtype=dtype, device=device)
    large = torch.full((2, 64), 1e8, dtype=dtype, device=device)

    torch.testing.assert_close(norm(zeros), zeros)
    torch.testing.assert_close(norm(large), torch.ones_like(large))
    assert torch.isfinite(norm(large)).all()
