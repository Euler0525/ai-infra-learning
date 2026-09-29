from dataclasses import dataclass

import torch

from mini_llm.config.model import ModelConfig


@dataclass
class KVCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    length: int = 0

    @classmethod
    def allocate(
        cls,
        config: ModelConfig,
        batch_size: int,
        max_model_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> "KVCache":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < max_model_len <= config.max_position_embeddings:
            raise ValueError(
                "max_model_len must be in (0, max_position_embeddings]"
            )

        shape = (
            batch_size,
            config.num_key_value_heads,
            max_model_len,
            config.head_dim,
        )
        keys = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        values = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        return cls(keys=keys, values=values)

    @property
    def batch_size(self) -> int:
        return self.keys[0].shape[0]

    @property
    def max_model_len(self) -> int:
        return self.keys[0].shape[2]
