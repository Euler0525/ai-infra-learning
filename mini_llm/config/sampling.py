from dataclasses import dataclass

from mini_llm.config.settings import DEFAULT_MAX_NEW_TOKENS


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ignore_eos: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
