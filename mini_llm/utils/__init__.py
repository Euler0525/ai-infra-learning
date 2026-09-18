from time import perf_counter
from typing import Any, Callable

import torch


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
