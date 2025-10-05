import torch

class PatchDetector:
    """완전히 GPU tensor만 사용하는 Patch Detector"""
    def __init__(self, attractor_learner, device='cuda', chunk_size=100):
        self.attractor = attractor_learner
        self.device = device
        self.chunk_size = chunk_size
        
        # Reference data (already on GPU)
        self.ref_points_gpu = self.attractor.reduced
        self.mean_gpu = self.attractor.mean
        self.cov_inv_gpu = self.attractor.cov_inv
        
        print(f"  PatchDetector initialized on {device} (Pure GPU)")
        print(f"    Reference points: {self.ref_points_gpu.shape}")
        print(f"    Chunk size: {chunk_size}")
        
    def detect(self, test_embedding_gpu, threshold=2.5):
        """
        완전히 GPU에서 detection 수행
        
        Args:
            test_embedding_gpu: [H, W, T, D] tensor on GPU
            threshold: detection threshold
        
        Returns:
            anomaly_map, patch_mask, hausdorff_map, mahalanobis_map (all numpy for visualization)
        """
        # PCA transform (GPU)
        test_reduced = self.attractor.transform(test_embedding_gpu)  # [H, W, T, n_components]
        H, W, T, D = test_reduced.shape
        
        # Reshape to [N, T, D] where N = H*W
        trajectories = test_reduced.reshape(-1, T, D)
        N = trajectories.shape[0]
        
        # ===== 1. Hausdorff Distance (GPU) =====
        hausdorff_scores = torch.zeros(N, device=self.device)
        
        for start_idx in range(0, N, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, N)
            chunk = trajectories[start_idx:end_idx]
            
            chunk_h_scores = []
            for i in range(chunk.shape[0]):
                traj = chunk[i]
                distances = torch.cdist(traj, self.ref_points_gpu, p=2)
                h_score = distances.min(dim=1)[0].max()
                chunk_h_scores.append(h_score)
            
            hausdorff_scores[start_idx:end_idx] = torch.stack(chunk_h_scores)
        
        # ===== 2. Mahalanobis Distance (GPU) =====
        all_points = trajectories.reshape(-1, D)  # [N*T, D]
        diff = all_points - self.mean_gpu
        maha_dists = torch.sqrt(torch.sum(diff @ self.cov_inv_gpu * diff, dim=1))
        maha_dists = maha_dists.reshape(N, T)
        mahalanobis_scores = maha_dists.mean(dim=1)  # [N]
        
        # ===== 3. Log-likelihood (GPU) =====
        log_probs = torch.zeros(N, device=self.device)
        for i in range(N):
            log_prob = self.attractor.kde.score_samples(trajectories[i]).mean()
            log_probs[i] = log_prob
        
        # ===== 4. Combined Score (GPU) =====
        anomaly_scores = (
            0.4 * hausdorff_scores + 
            0.4 * mahalanobis_scores - 
            0.2 * log_probs
        )
        
        # Move to CPU only for visualization
        anomaly_map = anomaly_scores.cpu().numpy().reshape(H, W)
        hausdorff_map = hausdorff_scores.cpu().numpy().reshape(H, W)
        mahalanobis_map = mahalanobis_scores.cpu().numpy().reshape(H, W)
        
        # Threshold
        patch_mask = anomaly_map > threshold
        
        return anomaly_map, patch_mask, hausdorff_map, mahalanobis_map
