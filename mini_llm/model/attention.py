from typing import Any

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension for RoPE."""
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_position_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key


def repeat_key_value(
    hidden_states: torch.Tensor,
    repeats: int,
) -> torch.Tensor:
    """Expand grouped K/V heads to the number of query heads."""
    if repeats == 1:
        return hidden_states

    batch_size, key_value_heads, sequence_length, head_dim = (
        hidden_states.shape
    )
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        key_value_heads,
        repeats,
        sequence_length,
        head_dim,
    )
    return hidden_states.reshape(
        batch_size,
        key_value_heads * repeats,
        sequence_length,
        head_dim,
    )


def build_causal_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an additive causal mask for a full-sequence forward pass."""
    batch_size, sequence_length = attention_mask.shape
    minimum = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        minimum,
        dtype=dtype,
        device=attention_mask.device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )
    padding_mask = attention_mask[:, None, None, :] == 0
    return causal_mask.masked_fill(padding_mask, minimum)


class RotaryEmbedding(nn.Module):
    """Qwen-style RoPE computed in float32 and returned in input dtype."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        rope_parameters = config.rope_parameters
        rope_theta = rope_parameters["rope_theta"]
        inverse_frequency = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inverse_frequency = self.inverse_frequency[None, :, None].expand(
            position_ids.shape[0],
            -1,
            1,
        )
        positions = position_ids[:, None, :].to(torch.float32)
        frequencies = torch.matmul(
            inverse_frequency.to(hidden_states.device),
            positions,
        ).transpose(1, 2)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embeddings.cos().to(hidden_states.dtype),
            embeddings.sin().to(hidden_states.dtype),
        )


class CausalSelfAttention(nn.Module):
    """Pure PyTorch grouped-query causal self-attention."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        )
        key = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        value = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        cosine, sine = position_embeddings
        query, key = apply_rotary_position_embedding(
            query,
            key,
            cosine,
            sine,
        )
        key = repeat_key_value(key, self.num_key_value_groups)
        value = repeat_key_value(value, self.num_key_value_groups)

        attention_weights = torch.matmul(
            query,
            key.transpose(2, 3),
        ) * self.scale
        attention_weights = attention_weights + attention_mask
        attention_weights = torch.softmax(
            attention_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            -1,
        )
        return self.o_proj(attention_output), attention_weights
