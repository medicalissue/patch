import torch
from torch import Tensor


def stack_trajectory(activations) -> Tensor:
    """
    Stack layer activations into trajectories.

    Args:
        activations: list of tensors [B, C, H, W]
    Returns:
        Tensor with shape [B, H, W, L, C]
    """
    if not activations:
        raise ValueError("activations list is empty")

    permuted = [act.permute(0, 2, 3, 1) for act in activations]
    embeddings = torch.stack(permuted, dim=3)
    return embeddings
