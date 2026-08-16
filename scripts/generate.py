import argparse

from mini_llm.config import (
    DEFAULT_MAX_NEW_TOKENS,
    REFERENCE_PROMPT,
    ModelSettings,
)
from mini_llm.engine import generate
from mini_llm.loading import load_model, load_tokenizer


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


def main() -> None:
    args = parse_args()
    settings = ModelSettings()
    tokenizer = load_tokenizer(settings)
    model = load_model(settings)
    result = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
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
    main()

