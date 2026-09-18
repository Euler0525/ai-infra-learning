from pathlib import Path

from mini_llm.preflight import run_preflight


def test_runtime_and_local_model_are_ready() -> None:
    report = run_preflight()

    assert report.model_type == "qwen2"
    assert report.tokenizer_class == "Qwen2Tokenizer"
    assert Path(report.model_path, "model.safetensors").is_file()
