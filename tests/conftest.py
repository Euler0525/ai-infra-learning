import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from mini_llm.model import TorchReferenceQwen


@pytest.fixture()
def tiny_qwen_pair() -> tuple[
    Qwen2ForCausalLM,
    TorchReferenceQwen,
]:
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        tie_word_embeddings=True,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    reference = Qwen2ForCausalLM(config).eval()
    manual = TorchReferenceQwen(config).eval()
    manual.load_state_dict(reference.state_dict(), strict=True)
    return reference, manual
