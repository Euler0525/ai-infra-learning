import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from mini_llm.config.settings import ModelSettings


@dataclass(frozen=True)
class PreflightReport:
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_name: str
    model_path: str
    model_type: str
    tokenizer_class: str


def run_preflight(settings: ModelSettings | None = None) -> PreflightReport:
    settings = settings or ModelSettings()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")

    model_path = Path(
        snapshot_download(
            settings.model_id,
            revision=settings.revision,
            local_files_only=settings.local_files_only,
        )
    )
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    return PreflightReport(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_version=str(torch.version.cuda),
        gpu_name=torch.cuda.get_device_name(),
        model_path=str(model_path),
        model_type=config.model_type,
        tokenizer_class=type(tokenizer).__name__,
    )


def main() -> None:
    print(json.dumps(asdict(run_preflight()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
