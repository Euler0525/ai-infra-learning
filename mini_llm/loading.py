import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any

from mini_llm.config import ModelSettings


def require_device(settings: ModelSettings) -> None:
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No available NVIDIA GPU with CUDA.")


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
    require_device(settings)
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        dtype=settings.dtype,
        attn_implementation=attention_implementation,
        local_files_only=settings.local_files_only,
    ).to(device=settings.device)
    model.eval()
    return model


def encode_prompt(
    tokenizer: Any,
    prompt: str,
    device: str,
) -> Any:
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        rendered_prompt,
        return_tensors="pt",
    ).to(device=device)
