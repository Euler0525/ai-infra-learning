from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

from mini_llm.model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    """Pure PyTorch RMSNorm with float32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        float_states = hidden_states.to(torch.float32)
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class SwiGLUMLP(nn.Module):
    """Qwen feed-forward network using a SiLU-gated linear unit."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gated * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    """Pre-norm Qwen Transformer block with two residual connections."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
