from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from mini_llm.config.settings import ModelSettings


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            max_position_embeddings=data["max_position_embeddings"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_theta=data["rope_theta"],
            tie_word_embeddings=data["tie_word_embeddings"],
            bos_token_id=data["bos_token_id"],
            eos_token_id=data["eos_token_id"],
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_pretrained(
        cls,
        settings: ModelSettings,
    ) -> "ModelConfig":
        model_path = Path(
            snapshot_download(
                repo_id=settings.model_id,
                revision=settings.revision,
                local_files_only=settings.local_files_only,
            )
        )
        return cls.from_json_file(model_path / "config.json")
