from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_llm.config.settings import ModelSettings


def load_tokenizer(settings: ModelSettings) -> Any:
    return AutoTokenizer.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        local_files_only=settings.local_files_only,
    )


def load_model(
    settings: ModelSettings,
    attention_implementation: str | None = None,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        dtype=settings.dtype,
        attn_implementation=attention_implementation,
        local_files_only=settings.local_files_only,
    ).to(settings.device)
    model.eval()
    return model


def encode_prompt(tokenizer: Any, prompt: str, device: str) -> Any:
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(rendered_prompt, return_tensors="pt").to(device)
