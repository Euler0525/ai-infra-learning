from dataclasses import dataclass

import torch

from mini_llm.config import SamplingParams
from mini_llm.engine.generation import generate_tokens, sample_next_token


@dataclass
class FakeCache:
    max_model_len: int
    length: int = 0


class ScriptedModel:
    def __init__(self, token_ids: list[int], vocab_size: int = 8):
        self.logits = []
        for token_id in token_ids:
            logits = torch.zeros(1, vocab_size)
            logits[0, token_id] = 10
            self.logits.append(logits)
        self.index = 0
        self.decode_calls = 0

    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: FakeCache,
    ) -> torch.Tensor:
        kv_cache.length = input_ids.shape[1]
        return self.logits[0]

    def decode(
        self,
        input_ids: torch.Tensor,
        kv_cache: FakeCache,
    ) -> torch.Tensor:
        self.decode_calls += 1
        self.index += 1
        kv_cache.length += 1
        return self.logits[self.index]


def run_script(
    token_ids: list[int],
    params: SamplingParams,
    max_model_len: int = 16,
) -> tuple[list[int], str, int]:
    model = ScriptedModel(token_ids)
    output = generate_tokens(
        model,
        torch.tensor([[4, 5]]),
        FakeCache(max_model_len),
        params,
        eos_token_id=2,
    )
    return output.token_ids, output.stop_reason, model.decode_calls


def test_greedy_is_reproducible_and_stops_at_max_tokens() -> None:
    params = SamplingParams(max_tokens=3)
    first = run_script([1, 3, 4], params)
    second = run_script([1, 3, 4], params)

    assert first == second == ([1, 3, 4], "max_tokens", 2)


def test_temperature_sampling_is_reproducible_with_a_fixed_seed() -> None:
    logits = torch.zeros(1, 8)
    first_generator = torch.Generator().manual_seed(42)
    second_generator = torch.Generator().manual_seed(42)

    first = [
        sample_next_token(logits, 1.0, first_generator)
        for _ in range(8)
    ]
    second = [
        sample_next_token(logits, 1.0, second_generator)
        for _ in range(8)
    ]

    assert first == second


def test_first_eos_stops_without_decode() -> None:
    actual = run_script([2], SamplingParams(max_tokens=4))

    assert actual == ([2], "eos", 0)


def test_ignore_eos_continues_until_max_tokens() -> None:
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    actual = run_script([2, 2, 2], params)

    assert actual == ([2, 2, 2], "max_tokens", 2)


def test_max_model_len_stops_before_an_extra_decode() -> None:
    params = SamplingParams(max_tokens=4)
    actual = run_script([1], params, max_model_len=3)

    assert actual == ([1], "max_model_len", 0)
