from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mini_llm.config.model import ModelConfig
from mini_llm.layers.decoder_layer import DecoderLayer
from mini_llm.layers.rms_norm import RMSNorm

if TYPE_CHECKING:
    from mini_llm.engine.kv_cache import KVCache


class Qwen2p5Backbone(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        cache_position = 0
        cache_end = input_ids.shape[1]
        if kv_cache is not None:
            if input_ids.shape[0] != kv_cache.batch_size:
                raise ValueError("input batch size must match KV cache")
            cache_position = kv_cache.length
            cache_end = cache_position + input_ids.shape[1]
            if cache_end > kv_cache.max_model_len:
                raise ValueError("KV cache capacity exceeded")

        for index, layer in enumerate(self.layers):
            layer_cache = None
            if kv_cache is not None:
                layer_cache = (
                    kv_cache.keys[index],
                    kv_cache.values[index],
                )
            hidden_states = layer(
                hidden_states,
                positions,
                layer_cache,
                cache_position,
            )

        if kv_cache is not None:
            kv_cache.length = cache_end
        return self.norm(hidden_states)


class Qwen2p5ForCausalLM(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = Qwen2p5Backbone(config)
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_cache)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        if kv_cache.length != 0:
            raise ValueError("prefill requires an empty KV cache")
        positions = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).expand(input_ids.shape[0], -1)
        hidden_states = self(input_ids, positions, kv_cache)
        return self.compute_logits(hidden_states[:, -1])

    def decode(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode expects input_ids with shape (batch, 1)")
        positions = torch.full(
            (input_ids.shape[0], 1),
            kv_cache.length,
            device=input_ids.device,
            dtype=torch.long,
        )
        hidden_states = self(input_ids, positions, kv_cache)
        return self.compute_logits(hidden_states[:, -1])
