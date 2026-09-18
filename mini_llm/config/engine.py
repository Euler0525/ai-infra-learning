from __future__ import annotations

from dataclasses import dataclass

from mini_llm.config.model import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    max_model_len: int = 2048
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    block_size: int = 16
    gpu_memory_utilization: float = 0.8
    enforce_eager: bool = True
    world_size: int = 1

    def __post_init__(self) -> None:
        limits = (
            self.max_model_len,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.block_size,
            self.world_size,
        )
        if min(limits) <= 0:
            raise ValueError("engine limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

    def validate(self, model_config: ModelConfig) -> None:
        if self.max_model_len > model_config.max_position_embeddings:
            raise ValueError(
                "max_model_len must not exceed max_position_embeddings"
            )
        if model_config.num_attention_heads % self.world_size != 0:
            raise ValueError(
                "world_size must divide num_attention_heads"
            )
        if model_config.num_key_value_heads % self.world_size != 0:
            raise ValueError(
                "world_size must divide num_key_value_heads"
            )
