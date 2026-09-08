import math
import torch


def mqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MQA: Q is multi-head while K/V each contain one shared head."""
    key = key[:, 0]  # [B, 1, K, D] -> [B, K, D]
    value = value[:, 0]
    scores = torch.einsum("bhqd,bkd->bhqk", query, key)
    scores = scores / math.sqrt(query.shape[-1])
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
    output = torch.einsum("bhqk,bkv->bhqv", weights, value)

    return output, weights
