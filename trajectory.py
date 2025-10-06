import torch

def stack_trajectory(activations):
    """
    ResNet의 모든 레이어 activation을 trajectory로 스택 (100% GPU)
    
    Args:
        activations: List of [B, C, H, W] torch.Tensors on GPU
                    각 텐서는 한 레이어의 activation
    
    Returns:
        embeddings: Tensor [B, H, W, L, C] on GPU
                   B = batch size
                   H, W = spatial dimensions
                   L = number of layers (시간 축)
                   C = channel dimension
    """
    assert len(activations) > 0, "activations list is empty"
    
    # 모든 activation을 [B, H, W, C] 형태로 변환
    permuted = [act.permute(0, 2, 3, 1) for act in activations]  # List of [B, H, W, C]
    
    # 시간 축(layer 축)으로 스택: [B, H, W, L, C]
    embeddings = torch.stack(permuted, dim=3)
    
    B, H, W, L, C = embeddings.shape
    
    return embeddings  # GPU tensor