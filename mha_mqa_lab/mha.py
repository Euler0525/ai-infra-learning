import math
import torch


def mha_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MHA: Q/K/V are [batch, heads, length, head_dim]."""
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    query_length, key_length = scores.shape[-2:]
    if query_length > 1:
        mask = torch.triu(
            torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=key_length - query_length + 1,
        )
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(
        scores, dim=-1, dtype=torch.float32).to(query.dtype)

    return weights @ value, weights
