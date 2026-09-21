import torch
from torch import nn

from mini_llm.config.model import ModelConfig
from mini_llm.layers.attention import GQAAttention
from mini_llm.layers.mlp import SwiGLU
from mini_llm.layers.rms_norm import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GQAAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = SwiGLU(config)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


"""
Qwen2.5-0.5B-Instruct DecoderLayer (one of 24)
┌────────────────────────────────────────────────────────────────┐
│ x [B, S, 896] ───────────────────────────┐                     │
│       │                                  │                     │
│       ▼                                  │                     │
│ input_layernorm: RMSNorm (eps=1e-6)      │                     │
│       │                                  │                     │
│       ▼                                  │                     │
│ self_attn: causal GQA Self-Attention     │                     │
│   Q: 14 heads × 64, K/V: 2 heads × 64    │                     │
│   RoPE(Q, K); o_proj: 896 → 896          │                     │
│       │                                  │                     │
│       └──────────────────► + ◄───────────┘                     │
│                            │                                   │
│                            ▼                                   │
│ y [B, S, 896] ───────────────────────────┐                     │
│       │                                  │                     │
│       ▼                                  │                     │
│ post_attention_layernorm: RMSNorm        │                     │
│       │                                  │                     │
│       ▼                                  │                     │
│ mlp: SwiGLU                              │                     │
│   gate/up: 896 → 4864                    │                     │
│   SiLU(gate) × up; down: 4864 → 896      │                     │
│       │                                  │                     │
│       └──────────────────► + ◄───────────┘                     │
│                            │                                   │
│                            ▼                                   │
│                     output [B, S, 896]                         │
└────────────────────────────────────────────────────────────────┘
"""
