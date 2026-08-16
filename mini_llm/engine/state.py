from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DecodeState:
    past_key_values: Any
    attention_mask: torch.Tensor


@dataclass
class StepOutput:
    next_token_id: int
    state: DecodeState
    elapsed_ms: float
    input_shape: list[int]
    logits_shape: list[int]


@dataclass
class GenerationResult:
    prompt_tokens: int
    generated_token_ids: list[int]
    generated_text: str
    stop_reason: str
    prefill_calls: int
    decode_calls: int
    prefill_ms: float
    decode_ms: float

    @property
    def average_decode_ms(self) -> float:
        if self.decode_calls == 0:
            return 0.0
        return self.decode_ms / self.decode_calls

