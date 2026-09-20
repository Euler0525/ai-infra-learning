import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    """Qwen2.5 RoPE for Q/K shaped (..., seq, heads, head_dim)."""

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = (
            positions.to(device=q.device, dtype=torch.float32)[..., None]
            * self.inv_freq
        )
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = angles.cos().to(q.dtype).unsqueeze(-2)
        sin = angles.sin().to(q.dtype).unsqueeze(-2)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
