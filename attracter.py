from utils import TorchPCA, TorchKDE

class AttractorLearner:
    """완전히 GPU tensor만 사용하는 Attractor Learner"""
    def __init__(self, n_components=32, bandwidth=0.5, device='cuda'):
        self.n_components = n_components
        self.bandwidth = bandwidth
        self.device = device
        self.pca = TorchPCA(n_components=n_components, device=device)
        self.kde = TorchKDE(bandwidth=bandwidth, device=device)
        self.fitted = False
        
    def fit(self, clean_embeddings_gpu):
        """
        정상 이미지들의 embedding으로 attractor 학습
        Args:
            clean_embeddings_gpu: List of [H, W, T, D] tensors on GPU
        """
        all_points = []
        for emb in clean_embeddings_gpu:
            H, W, T, D = emb.shape
            points = emb.reshape(-1, D)  # [H*W*T, D]
            all_points.append(points)
        
        all_points = torch.cat(all_points, dim=0)  # [N_total, D]
        print(f"  Learning attractor from {len(all_points)} points (GPU tensor)...")
        
        # PCA로 차원 축소
        self.reduced = self.pca.fit_transform(all_points)  # [N_total, n_components]
        
        # KDE로 density 학습
        self.kde.fit(self.reduced)
        
        # 통계량 저장 (GPU)
        self.mean = self.reduced.mean(dim=0)
        
        # Covariance matrix
        centered = self.reduced - self.mean
        self.cov = (centered.T @ centered) / (self.reduced.shape[0] - 1)
        
        # Pseudo-inverse for Mahalanobis
        self.cov_inv = torch.linalg.pinv(self.cov + 1e-6 * torch.eye(len(self.cov), device=self.device))
        
        self.fitted = True
        
        # Explained variance
        explained_var_ratio = (self.pca.explained_variance_ / self.pca.explained_variance_.sum()).cpu().numpy()
        print(f"  Attractor learned: PCA explained variance = {explained_var_ratio.sum():.3f}")
        
        return self
    
    def transform(self, embedding_gpu):
        """
        Embedding을 PCA space로 변환
        Args:
            embedding_gpu: [H, W, T, D] tensor on GPU
        Returns:
            [H, W, T, n_components] tensor on GPU
        """
        return self.pca.transform(embedding_gpu)