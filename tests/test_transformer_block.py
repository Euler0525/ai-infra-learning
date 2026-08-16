import pytest
import torch
from transformers import Qwen2ForCausalLM

from mini_llm.model import (
    TorchReferenceQwen,
    build_causal_attention_mask,
)
from mini_llm.sampling import greedy_sample


@pytest.mark.parametrize("sequence_length", [1, 4, 9])
def test_transformer_block_matches_reference_intermediates(
    tiny_qwen_pair: tuple[Qwen2ForCausalLM, TorchReferenceQwen],
    sequence_length: int,
) -> None:
    reference, manual = tiny_qwen_pair
    torch.manual_seed(sequence_length)
    hidden_states = torch.randn(
        1,
        sequence_length,
        reference.config.hidden_size,
    )
    position_ids = torch.arange(sequence_length).unsqueeze(0)
    attention_mask = build_causal_attention_mask(
        torch.ones(1, sequence_length, dtype=torch.long),
        hidden_states.dtype,
    )
    reference_positions = reference.model.rotary_emb(
        hidden_states,
        position_ids,
    )
    manual_positions = manual.model.rotary_emb(
        hidden_states,
        position_ids,
    )
    reference_layer = reference.model.layers[0]
    manual_layer = manual.model.layers[0]

    torch.testing.assert_close(
        manual_positions,
        reference_positions,
    )

    reference_normalized = reference_layer.input_layernorm(hidden_states)
    manual_normalized = manual_layer.input_layernorm(hidden_states)
    torch.testing.assert_close(
        manual_normalized,
        reference_normalized,
    )
    for projection_name in ("q_proj", "k_proj", "v_proj"):
        reference_projection = getattr(
            reference_layer.self_attn,
            projection_name,
        )(reference_normalized)
        manual_projection = getattr(
            manual_layer.self_attn,
            projection_name,
        )(manual_normalized)
        torch.testing.assert_close(
            manual_projection,
            reference_projection,
        )

    reference_attention, reference_weights = reference_layer.self_attn(
        reference_normalized,
        position_embeddings=reference_positions,
        attention_mask=attention_mask,
    )
    manual_attention, manual_weights = manual_layer.self_attn(
        manual_normalized,
        manual_positions,
        attention_mask,
    )
    torch.testing.assert_close(
        manual_attention,
        reference_attention,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        manual_weights,
        reference_weights,
        rtol=1e-5,
        atol=1e-6,
    )

    reference_post_attention = reference_layer.post_attention_layernorm(
        hidden_states + reference_attention,
    )
    manual_post_attention = manual_layer.post_attention_layernorm(
        hidden_states + manual_attention,
    )
    torch.testing.assert_close(
        manual_post_attention,
        reference_post_attention,
    )
    torch.testing.assert_close(
        manual_layer.mlp(manual_post_attention),
        reference_layer.mlp(reference_post_attention),
        rtol=1e-5,
        atol=1e-6,
    )

    reference_output = reference_layer(
        hidden_states,
        position_embeddings=reference_positions,
        attention_mask=attention_mask,
    )
    manual_output = manual_layer(
        hidden_states,
        manual_positions,
        attention_mask,
    )
    torch.testing.assert_close(
        manual_output,
        reference_output,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("sequence_length", [1, 4, 9])
def test_full_logits_and_greedy_token_match_reference(
    tiny_qwen_pair: tuple[Qwen2ForCausalLM, TorchReferenceQwen],
    sequence_length: int,
) -> None:
    reference, manual = tiny_qwen_pair
    torch.manual_seed(sequence_length)
    input_ids = torch.randint(
        1,
        reference.config.vocab_size,
        (1, sequence_length),
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        reference_logits = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        manual_logits = manual(input_ids, attention_mask)

    torch.testing.assert_close(
        manual_logits,
        reference_logits,
        rtol=1e-5,
        atol=1e-6,
    )
    assert greedy_sample(
        manual_logits[:, -1, :],
    ) == greedy_sample(reference_logits[:, -1, :])
