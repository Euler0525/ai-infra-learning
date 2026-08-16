from time import perf_counter
from typing import Any, Callable

import torch

from mini_llm.engine.state import DecodeState, GenerationResult, StepOutput
from mini_llm.loading import encode_prompt
from mini_llm.sampling import greedy_sample


def timed_device_call(
    call: Callable[[], Any],
    device: torch.device,
) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    output = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, (perf_counter() - start) * 1_000


def prefill(
    model: Any,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> StepOutput:
    """Process the complete prompt once and create the initial KV cache."""
    with torch.inference_mode():
        outputs, elapsed_ms = timed_device_call(
            lambda: model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                use_cache=True,
            ),
            token_ids.device,
        )

    return StepOutput(
        next_token_id=greedy_sample(outputs.logits[:, -1, :]),
        state=DecodeState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
        ),
        elapsed_ms=elapsed_ms,
        input_shape=list(token_ids.shape),
        logits_shape=list(outputs.logits.shape),
    )


def decode(
    model: Any,
    token_id: int,
    state: DecodeState,
) -> StepOutput:
    """Append one token, reuse the KV cache, and predict one new token."""
    device = state.attention_mask.device
    token_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.cat(
        [
            state.attention_mask,
            torch.ones(
                (state.attention_mask.shape[0], 1),
                dtype=state.attention_mask.dtype,
                device=device,
            ),
        ],
        dim=-1,
    )

    with torch.inference_mode():
        outputs, elapsed_ms = timed_device_call(
            lambda: model(
                input_ids=token_ids,
                attention_mask=attention_mask,
                past_key_values=state.past_key_values,
                use_cache=True,
            ),
            device,
        )

    return StepOutput(
        next_token_id=greedy_sample(outputs.logits[:, -1, :]),
        state=DecodeState(
            past_key_values=outputs.past_key_values,
            attention_mask=attention_mask,
        ),
        elapsed_ms=elapsed_ms,
        input_shape=list(token_ids.shape),
        logits_shape=list(outputs.logits.shape),
    )


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    device = str(next(model.parameters()).device)
    inputs = encode_prompt(tokenizer, prompt, device)
    prefill_output = prefill(
        model,
        inputs.input_ids,
        inputs.attention_mask,
    )
    generated_token_ids = [prefill_output.next_token_id]
    state = prefill_output.state
    decode_calls = 0
    decode_ms = 0.0
    stop_reason = None

    if prefill_output.next_token_id == tokenizer.eos_token_id:
        stop_reason = "eos"

    while stop_reason is None and len(generated_token_ids) < max_new_tokens:
        output = decode(model, generated_token_ids[-1], state)
        state = output.state
        generated_token_ids.append(output.next_token_id)
        decode_calls += 1
        decode_ms += output.elapsed_ms
        if output.next_token_id == tokenizer.eos_token_id:
            stop_reason = "eos"

    if stop_reason is None:
        stop_reason = "max_new_tokens"

    result = GenerationResult(
        prompt_tokens=inputs.input_ids.shape[-1],
        generated_token_ids=generated_token_ids,
        generated_text=tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        ),
        stop_reason=stop_reason,
        prefill_calls=1,
        decode_calls=decode_calls,
        prefill_ms=prefill_output.elapsed_ms,
        decode_ms=decode_ms,
    )
    verify_loop_boundaries(result, max_new_tokens)
    return result


def verify_loop_boundaries(
    result: GenerationResult,
    max_new_tokens: int,
) -> None:
    if result.prefill_calls != 1:
        raise AssertionError("Each request must prefill exactly once.")
    if result.decode_calls != len(result.generated_token_ids) - 1:
        raise AssertionError(
            "Each token after the first needs one decode call.")
    if len(result.generated_token_ids) > max_new_tokens:
        raise AssertionError("Generation exceeded max_new_tokens.")
    if result.stop_reason == "max_new_tokens":
        if len(result.generated_token_ids) != max_new_tokens:
            raise AssertionError("Length stop occurred at the wrong boundary.")
