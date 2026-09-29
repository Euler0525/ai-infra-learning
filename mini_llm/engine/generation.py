from dataclasses import dataclass
from typing import Protocol

import torch

from mini_llm.config.sampling import SamplingParams
from mini_llm.engine.kv_cache import KVCache


class SingleRequestModel(Protocol):
    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor: ...

    def decode(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class TokenGeneration:
    token_ids: list[int]
    stop_reason: str


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
) -> int:
    if temperature == 0:
        return logits.argmax(dim=-1).item()
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).item()


def generate_tokens(
    model: SingleRequestModel,
    input_ids: torch.Tensor,
    kv_cache: KVCache,
    sampling_params: SamplingParams,
    eos_token_id: int,
) -> TokenGeneration:
    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(sampling_params.seed)
    generated_token_ids: list[int] = []

    with torch.inference_mode():
        logits = model.prefill(input_ids, kv_cache)
        while input_ids.shape[1] + len(generated_token_ids) < kv_cache.max_model_len:
            token_id = sample_next_token(
                logits,
                sampling_params.temperature,
                generator,
            )
            generated_token_ids.append(token_id)

            if token_id == eos_token_id and not sampling_params.ignore_eos:
                return TokenGeneration(generated_token_ids, "eos")
            if len(generated_token_ids) == sampling_params.max_tokens:
                return TokenGeneration(generated_token_ids, "max_tokens")
            if input_ids.shape[1] + len(generated_token_ids) == kv_cache.max_model_len:
                return TokenGeneration(generated_token_ids, "max_model_len")

            next_input = torch.tensor(
                [[token_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            logits = model.decode(next_input, kv_cache)

    return TokenGeneration(generated_token_ids, "max_model_len")
