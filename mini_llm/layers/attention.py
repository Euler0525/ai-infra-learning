import torch
from torch import nn

from mini_llm.config.model import ModelConfig
from mini_llm.layers.rotary_embedding import RotaryEmbedding


class GQAAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        q, k = self.rotary_emb(q, k, positions)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(
            self.num_key_value_groups, dim=1)
        v = v.transpose(1, 2).repeat_interleave(
            self.num_key_value_groups, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.scaling
        future = torch.ones(
            seq_len, seq_len, device=hidden_states.device, dtype=torch.bool
        ).triu(1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = torch.softmax(
            scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)
        return self.o_proj(output)


"""
hidden_states
[B, S, hidden_size]
        │
        ├──────── q_proj ────────► Q [B,S,Hq,D]
        │
        ├──────── k_proj ────────► K [B,S,Hkv,D]
        │
        └──────── v_proj ────────► V [B,S,Hkv,D]
                                      │
                                    RoPE
                                      │
                    ┌─────────────────┴──────────────┐
                    │                                │
             Q transpose                     K/V transpose
                    │                                │
           [B,Hq,S,D]                    [B,Hkv,S,D]
                                                     │
                                              repeat KV heads
                                                     │
                                            [B,Hq,S,D]
                    │                                │
                    └──────────────┬─────────────────┘
                                   │
                              Q @ Kᵀ
                                   │
                         [B,Hq,S,S]
                                   │
                            × 1/sqrt(D)
                                   │
                             Causal Mask
                                   │
                               Softmax
                                   │
                            [B,Hq,S,S]
                                   │
                                @ V
                                   │
                            [B,Hq,S,D]
                                   │
                              transpose
                                   │
                            [B,S,Hq,D]
                                   │
                                reshape
                                   │
                         [B,S,Hq×D]
                                   │
                               o_proj
                                   │
                         [B,S,hidden_size]
"""
