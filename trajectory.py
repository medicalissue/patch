"""
Trajectory Embedding Module for Few-shot Patch Detection

This module provides utilities for stacking multi-layer activations into trajectories.
Each spatial location across network layers forms a trajectory in feature space.
"""

import torch


def stack_trajectory(activations):
    """
    Stack ResNet layer activations into trajectories (100% GPU)

    Takes a list of activation tensors from different ResNet layers and
    stacks them along a temporal dimension to form trajectories. Each spatial
    location (h, w) across all layers forms a trajectory in feature space.

    Args:
        activations: List of [B, C, H, W] torch.Tensors on GPU
                    Each tensor represents activation from one layer
                    B = batch size
                    C = channel dimension
                    H, W = spatial dimensions

    Returns:
        embeddings: Tensor [B, H, W, L, C] on GPU
                   B = batch size
                   H, W = spatial dimensions
                   L = number of layers (temporal/trajectory axis)
                   C = channel dimension

    Example:
        >>> # Extract activations from 3 ResNet layers
        >>> act1 = torch.randn(4, 128, 7, 7)  # Layer 1: [B=4, C=128, H=7, W=7]
        >>> act2 = torch.randn(4, 128, 7, 7)  # Layer 2
        >>> act3 = torch.randn(4, 128, 7, 7)  # Layer 3
        >>> activations = [act1, act2, act3]
        >>> trajectories = stack_trajectory(activations)
        >>> print(trajectories.shape)
        torch.Size([4, 7, 7, 3, 128])  # [B, H, W, L, C]

        Each position (b, h, w) now has a trajectory of length L=3
        representing the evolution of features across network layers.
    """
    assert len(activations) > 0, "activations list is empty"

    # Convert all activations to [B, H, W, C] format
    permuted = [act.permute(0, 2, 3, 1) for act in activations]  # List of [B, H, W, C]

    # Stack along temporal axis (layer axis): [B, H, W, L, C]
    # This creates trajectories where L is the temporal dimension
    embeddings = torch.stack(permuted, dim=3)

    return embeddings  # GPU tensor
