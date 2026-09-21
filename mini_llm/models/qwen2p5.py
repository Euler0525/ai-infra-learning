import torch

from mini_llm.config.model import ModelConfig
from mini_llm.layers.decoder_layer import DecoderLayer
from mini_llm.layers.rms_norm import RMSNorm


class Qwen2p5Backbone(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.norm(hidden_states)


class Qwen2p5ForCausalLM(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = Qwen2p5Backbone(config)
        self.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
