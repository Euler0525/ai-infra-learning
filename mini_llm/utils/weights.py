from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from mini_llm.models import Qwen2p5ForCausalLM


@dataclass(frozen=True)
class WeightLoadReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _weight_mapping(
    model: Qwen2p5ForCausalLM,
) -> dict[str, torch.nn.Parameter]:
    weights = {
        "model.embed_tokens.weight": model.model.embed_tokens.weight,
        "model.norm.weight": model.model.norm.weight,
    }
    for index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{index}"
        weights.update({
            f"{prefix}.input_layernorm.weight":
                layer.input_layernorm.weight,
            f"{prefix}.post_attention_layernorm.weight":
                layer.post_attention_layernorm.weight,
            f"{prefix}.self_attn.q_proj.weight":
                layer.self_attn.q_proj.weight,
            f"{prefix}.self_attn.q_proj.bias":
                layer.self_attn.q_proj.bias,
            f"{prefix}.self_attn.k_proj.weight":
                layer.self_attn.k_proj.weight,
            f"{prefix}.self_attn.k_proj.bias":
                layer.self_attn.k_proj.bias,
            f"{prefix}.self_attn.v_proj.weight":
                layer.self_attn.v_proj.weight,
            f"{prefix}.self_attn.v_proj.bias":
                layer.self_attn.v_proj.bias,
            f"{prefix}.self_attn.o_proj.weight":
                layer.self_attn.o_proj.weight,
            f"{prefix}.mlp.gate_proj.weight": layer.mlp.gate_proj.weight,
            f"{prefix}.mlp.up_proj.weight": layer.mlp.up_proj.weight,
            f"{prefix}.mlp.down_proj.weight": layer.mlp.down_proj.weight,
        })

    parameter_names = set(dict(model.named_parameters()))
    unmapped = sorted(parameter_names - weights.keys())
    if unmapped:
        raise RuntimeError(f"unmapped model parameters: {unmapped}")
    return weights


def load_safetensors_weights(
    model: Qwen2p5ForCausalLM,
    path: str | Path,
) -> WeightLoadReport:
    weights = _weight_mapping(model)
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = set(checkpoint.keys())
        missing_keys = tuple(sorted(weights.keys() - checkpoint_keys))
        unexpected_keys = tuple(sorted(checkpoint_keys - weights.keys()))
        report = WeightLoadReport(missing_keys, unexpected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "incompatible checkpoint: "
                f"missing_keys={list(missing_keys)}, "
                f"unexpected_keys={list(unexpected_keys)}"
            )

        for name, parameter in weights.items():
            actual_shape = tuple(checkpoint.get_slice(name).get_shape())
            expected_shape = tuple(parameter.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"shape mismatch for {name!r}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )

        with torch.no_grad():
            for name, parameter in weights.items():
                parameter.copy_(checkpoint.get_tensor(name))

    return report
