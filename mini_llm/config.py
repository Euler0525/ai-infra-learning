import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "7ae557604adf67be50417f59c2c2f167def9a775"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    local_files_only: bool = True


REFERENCE_PROMPT = "What's for lunch today?"
DEFAULT_MAX_NEW_TOKENS = 1000
