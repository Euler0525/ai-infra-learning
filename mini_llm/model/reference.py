from typing import Any

import torch
from torch import nn

from mini_llm.model.attention import (
    RotaryEmbedding,
    build_causal_attention_mask,
)
from mini_llm.model.layers import RMSNorm, TransformerBlock


class QwenDecoder(nn.Module):
    """Minimal full-sequence decoder composed from reference operators."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        batch_size, sequence_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.long,
                device=input_ids.device,
            )
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )
        causal_mask = build_causal_attention_mask(
            attention_mask,
            hidden_states.dtype,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                causal_mask,
            )
        return self.norm(hidden_states)


class TorchReferenceQwen(nn.Module):
    """Minimal Qwen causal LM for full-sequence correctness checks."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.model = QwenDecoder(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        return self.lm_head(hidden_states)
