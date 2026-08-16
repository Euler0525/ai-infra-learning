import torch


def greedy_sample(last_logits: torch.Tensor) -> int:
    """Select the highest-scoring token from one decoding position."""
    return last_logits.argmax(dim=-1).item()

