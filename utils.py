import torch

class TorchKDE:
    """PyTorch로 구현한 Kernel Density Estimation (완전히 GPU에서 동작)"""
    def __init__(self, bandwidth=0.5, device='cuda'):
        self.bandwidth = bandwidth
        self.device = device
        self.training_data = None
        
    def fit(self, X):
        """
        KDE 학습 (데이터 저장)
        Args:
            X: [N, D] tensor on GPU
        """
        self.training_data = X.clone()
        self.dim = X.shape[1]
        return self
    
    def score_samples(self, X):
        """
        Log-likelihood 계산
        Args:
            X: [M, D] or [T, D] tensor
        Returns:
            log_prob: [M] or [T] tensor
        """
        # Gaussian kernel: exp(-0.5 * ||x - x'||^2 / h^2)
        # log p(x) = log(1/N * sum_i K(x, x_i))
        
        # X: [M, D], training_data: [N, D]
        # distances: [M, N]
        distances = torch.cdist(X, self.training_data, p=2)
        
        # Kernel values
        h = self.bandwidth
        kernel_values = torch.exp(-0.5 * (distances / h) ** 2)
        
        # Normalization constant
        normalizer = (2 * torch.pi * h**2) ** (self.dim / 2)
        
        # Mean over training samples
        density = kernel_values.mean(dim=1) / normalizer
        
        # Log density
        log_prob = torch.log(density + 1e-10)
        
        return log_prob
    
class TorchPCA:
    """PyTorch로 구현한 PCA (완전히 GPU에서 동작)"""
    def __init__(self, n_components=32, device='cuda'):
        self.n_components = n_components
        self.device = device
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        
    def fit(self, X):
        """
        PCA 학습
        Args:
            X: [N, D] tensor on GPU
        """
        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Keep top n_components
        self.components_ = Vt[:self.n_components]  # [n_components, D]
        self.explained_variance_ = (S[:self.n_components] ** 2) / (X.shape[0] - 1)
        
        return self
    
    def transform(self, X):
        """
        데이터를 PCA space로 변환
        Args:
            X: [N, D] or [H, W, T, D] tensor
        Returns:
            Transformed tensor
        """
        original_shape = X.shape
        
        # Flatten if needed
        if len(original_shape) > 2:
            X_flat = X.reshape(-1, original_shape[-1])
        else:
            X_flat = X
        
        # Center and project
        X_centered = X_flat - self.mean_
        X_transformed = X_centered @ self.components_.T  # [N, n_components]
        
        # Reshape back if needed
        if len(original_shape) > 2:
            new_shape = list(original_shape[:-1]) + [self.n_components]
            X_transformed = X_transformed.reshape(new_shape)
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)