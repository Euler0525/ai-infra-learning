import sys
import json
import torch

from importlib import import_module
from dataclasses import asdict, dataclass
from huggingface_hub import snapshot_download
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

from mini_llm.config import ModelSettings

PACKAGES = (
    "config",
    "layers",
    "models",
    "engine",
    "kernels",
    "utils",
)

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class PreflightReport:
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_name: str
    bf16_supported: bool
    model_path: str
    model_type: str
    tokenizer_class: str


def run_preflight(s: ModelSettings | None = None) -> PreflightReport:
    s = s or ModelSettings()

    if sys.version_info[:1] != (3, ):
        raise RuntimeError("Python 3.12 required")

    for p in PACKAGES:
        import_module(f"mini_llm.{p}")

    if s.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 required")

    path = Path(snapshot_download(
        s.model_id,
        revision=s.revision,
        local_files_only=s.local_files_only,
    ))

    if missing := [
        f for f in MODEL_FILES
        if not (path / f).is_file()
    ]:
        raise RuntimeError(f"Missing model files: {missing}")

    config = AutoConfig.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    report = PreflightReport(
        sys.version.split()[0],
        torch.__version__,
        str(torch.version.cuda),
        torch.cuda.get_device_name(),
        True,
        str(path),
        config.model_type,
        type(tokenizer).__name__,
    )
    return json.dumps(asdict(report), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    print(run_preflight())
