from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from mini_llm.config import ModelConfig, ModelSettings, REFERENCE_PROMPT
from mini_llm.engine import KVCache
from mini_llm.models import Qwen2p5ForCausalLM
from mini_llm.utils import load_safetensors_weights


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=56,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )


def make_hf_config(config: ModelConfig) -> Qwen2Config:
    hf_config = Qwen2Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
    )
    hf_config._attn_implementation = "eager"
    return hf_config


def test_model_shapes_and_tied_embeddings(small_config: ModelConfig) -> None:
    model = Qwen2p5ForCausalLM(small_config)
    layer_shapes: list[torch.Size] = []
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output: layer_shapes.append(output.shape)
        )
        for layer in model.model.layers
    ]
    input_ids = torch.randint(0, small_config.vocab_size, (2, 5))
    positions = torch.arange(5).expand(2, -1)

    hidden_states = model(input_ids, positions)
    logits = model.compute_logits(hidden_states)
    for hook in hooks:
        hook.remove()

    assert hidden_states.shape == (2, 5, small_config.hidden_size)
    assert logits.shape == (2, 5, small_config.vocab_size)
    assert layer_shapes == [hidden_states.shape] * \
        small_config.num_hidden_layers
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_random_model_matches_hugging_face(small_config: ModelConfig) -> None:
    torch.manual_seed(42)
    hf_model = HFQwen2ForCausalLM(make_hf_config(small_config)).eval()
    model = Qwen2p5ForCausalLM(small_config).eval()
    model.load_state_dict(hf_model.state_dict())
    input_ids = torch.randint(0, small_config.vocab_size, (2, 5))
    positions = torch.arange(5).expand(2, -1)

    with torch.no_grad():
        hidden_states = model(input_ids, positions)
        logits = model.compute_logits(hidden_states)
        expected_hidden = hf_model.model(
            input_ids=input_ids, position_ids=positions
        ).last_hidden_state
        expected_logits = hf_model.lm_head(expected_hidden)

    torch.testing.assert_close(
        hidden_states, expected_hidden, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(logits, expected_logits, rtol=1e-4, atol=1e-5)


def test_qwen2_5_architecture() -> None:
    config = ModelConfig.from_pretrained(ModelSettings())
    with torch.device("meta"):
        model = Qwen2p5ForCausalLM(config)

    assert len(model.model.layers) == 24
    assert model.model.embed_tokens.weight.shape == (151936, 896)
    assert model.model.layers[0].self_attn.q_proj.weight.shape == (896, 896)
    assert model.model.layers[0].self_attn.k_proj.weight.shape == (128, 896)
    assert model.model.layers[0].mlp.gate_proj.weight.shape == (4864, 896)
    assert model.lm_head.weight is model.model.embed_tokens.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_real_weights_match_hugging_face() -> None:
    settings = ModelSettings()
    config = ModelConfig.from_pretrained(settings)
    model_path = Path(snapshot_download(
        settings.model_id,
        revision=settings.revision,
        local_files_only=True,
    ))
    hf_model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("cuda").eval()
    model = Qwen2p5ForCausalLM(config).to(
        device="cuda", dtype=torch.bfloat16).eval()
    report = load_safetensors_weights(
        model,
        model_path / "model.safetensors",
    )
    assert report.missing_keys == ()
    assert report.unexpected_keys == ()
    assert model.lm_head.weight.data_ptr() == \
        model.model.embed_tokens.weight.data_ptr()
    assert all(
        layer.self_attn.rotary_emb.inv_freq.dtype == torch.float32
        for layer in model.model.layers
    )

    samples = {
        "model.embed_tokens.weight": model.model.embed_tokens.weight[:2],
        "model.layers.0.self_attn.q_proj.bias":
            model.model.layers[0].self_attn.q_proj.bias,
        "model.layers.12.mlp.up_proj.weight":
            model.model.layers[12].mlp.up_proj.weight[:2],
        "model.layers.23.post_attention_layernorm.weight":
            model.model.layers[23].post_attention_layernorm.weight,
        "model.norm.weight": model.model.norm.weight,
    }
    with safe_open(
        model_path / "model.safetensors",
        framework="pt",
        device="cpu",
    ) as checkpoint:
        for name, actual in samples.items():
            expected = checkpoint.get_slice(name)[:actual.shape[0]]
            torch.testing.assert_close(actual.cpu(), expected)

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id,
        revision=settings.revision,
        local_files_only=True,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": REFERENCE_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(
        rendered_prompt,
        return_tensors="pt",
    ).input_ids.to("cuda")
    positions = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)

    with torch.no_grad():
        logits = model.compute_logits(model(input_ids, positions))[:, -1]
        hf_outputs = hf_model(input_ids, use_cache=True)
        expected = hf_outputs.logits[:, -1]

        cache = KVCache.allocate(
            config,
            batch_size=1,
            max_model_len=input_ids.shape[1] + 32,
            device="cuda",
            dtype=torch.bfloat16,
        )
        cached_logits = model.prefill(input_ids, cache)
        torch.testing.assert_close(
            cached_logits,
            logits,
            rtol=5e-2,
            atol=5e-2,
        )

        generated_token_ids = []
        past_key_values = hf_outputs.past_key_values
        hf_logits = expected
        for step in range(32):
            actual_token = cached_logits.argmax(dim=-1)
            expected_token = hf_logits.argmax(dim=-1)
            assert actual_token.item() == expected_token.item()
            generated_token_ids.append(actual_token.item())
            if step == 31:
                break

            cached_logits = model.decode(actual_token.unsqueeze(1), cache)
            hf_outputs = hf_model(
                input_ids=expected_token.unsqueeze(1),
                past_key_values=past_key_values,
                use_cache=True,
            )
            hf_logits = hf_outputs.logits[:, -1]
            past_key_values = hf_outputs.past_key_values
            torch.testing.assert_close(
                cached_logits,
                hf_logits,
                rtol=5e-2,
                atol=6e-1,
            )

    actual_top5 = logits.topk(5).indices[0].tolist()
    expected_top5 = expected.topk(5).indices[0].tolist()
    difference = (logits.float() - expected.float()).abs()

    assert actual_top5[0] == expected_top5[0]
    assert set(actual_top5) == set(expected_top5)
    assert difference.mean() < 1e-1
    assert difference.max() < 6e-1
    assert len(generated_token_ids) == 32
