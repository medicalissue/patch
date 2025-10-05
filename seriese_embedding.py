import torch

def takens_embedding_gpu(activations, m=3, tau=1):
    """
    GPU에서 Takens embedding 수행
    
    Args:
        activations: List of [B, C, H, W] torch.Tensors on GPU
        m: embedding dimension
        tau: time delay
    
    Returns:
        embeddings: Tensor [B, H, W, T, D] on GPU
    """
    assert len(activations) > 0
    assert m >= 1 and tau >= 1
    
    # Stack to [L, B, C, H, W]
    A = torch.stack(activations, dim=0)
    
    L, B, C, H, W = A.shape
    T = L - (m - 1) * tau
    D = m * C
    
    if T <= 0:
        device = A.device
        E = torch.zeros(B, H, W, 1, D, device=device, dtype=A.dtype)
        return E
    
    # Time-delayed slices
    delayed = [A[k * tau : k * tau + T] for k in range(m)]
    E = torch.cat(delayed, dim=2)  # [T, B, m*C, H, W]
    
    # Reshape to [B, H, W, T, D]
    E = E.permute(1, 3, 4, 0, 2).contiguous()
    
    return E  # GPU tensor