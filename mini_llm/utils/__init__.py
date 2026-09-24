from time import perf_counter
from typing import Any, Callable

import torch

from mini_llm.utils.weights import (
    WeightLoadReport,
    load_safetensors_weights,
)


def timed_device_call(
    call: Callable[[], Any],
    device: torch.device,
) -> tuple[Any, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    output = call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, (perf_counter() - start) * 1000


__all__ = [
    "WeightLoadReport",
    "load_safetensors_weights",
    "timed_device_call",
]
