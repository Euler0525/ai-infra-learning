import argparse

from mini_llm.config import (
    DEFAULT_MAX_NEW_TOKENS,
    REFERENCE_PROMPT,
    ModelSettings,
)
from mini_llm.engine.state import GenerationResult
from mini_llm.config import load_model, load_tokenizer, encode_prompt
from mini_llm.engine.prefill_decode import prefill, decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy generation with explicit prefill and decode.",
    )
    parser.add_argument("--prompt", default=REFERENCE_PROMPT)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    return parser.parse_args()


def run_qwen2p5() -> None:
    args = parse_args()
    settings = ModelSettings()
    tokenizer = load_tokenizer(settings)
    model = load_model(settings)

    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    device = str(next(model.parameters()).device)
    inputs = encode_prompt(tokenizer, args.prompt, device)
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

    while stop_reason is None and len(generated_token_ids) < args.max_new_tokens:
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

    print(f"prompt tokens: {result.prompt_tokens}")
    print(f"generated tokens: {len(result.generated_token_ids)}")
    print(f"prefill calls: {result.prefill_calls}")
    print(f"decode calls: {result.decode_calls}")
    print(f"prefill: {result.prefill_ms:.3f} ms")
    print(f"average decode: {result.average_decode_ms:.3f} ms")
    print(f"stop reason: {result.stop_reason}")
    print(f"\n{result.generated_text}")


if __name__ == "__main__":
    run_qwen2p5()
