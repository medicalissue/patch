import torch

class PatchDetector:
    """Vector Field Consistency와 Spectral Analysis 기반 Detector (100% GPU)"""
    def __init__(self, attractor_learner, device='cuda'):
        self.attractor = attractor_learner
        self.device = device
        
        if not self.attractor.fitted:
            raise ValueError("AttractorLearner must be fitted before creating PatchDetector")
        
        print(f"  PatchDetector initialized on {device} (100% GPU)")
        print(f"    Using Vector Field + Spectral Analysis")
        
    def detect(self, test_embedding_gpu, threshold=2.5):
        """
        Vector field와 spectral analysis로 detection 수행 (모든 연산 GPU)
        
        Args:
            test_embedding_gpu: [H, W, L, D] tensor on GPU
            threshold: detection threshold
        
        Returns:
            anomaly_map_gpu: [H, W] tensor on GPU
            patch_mask_gpu: [H, W] boolean tensor on GPU
            vector_map_gpu: [H, W] tensor on GPU
            spectral_map_gpu: [H, W] tensor on GPU
        """
        H, W, L, D = test_embedding_gpu.shape
        trajectories = test_embedding_gpu.reshape(-1, L, D)  # [N, L, D] where N = H*W
        N = trajectories.shape[0]
        
        # ===== 1. Vector Field Consistency Score (GPU) =====
        vectors = trajectories[:, 1:] - trajectories[:, :-1]  # [N, L-1, D]
        magnitudes = torch.norm(vectors, dim=2)  # [N, L-1]
        
        # Magnitude deviation from reference
        magnitude_deviation = torch.abs(magnitudes - self.attractor.mean_magnitude) / (self.attractor.std_magnitude + 1e-8)
        magnitude_score = magnitude_deviation.mean(dim=1)  # [N]
        
        # Direction consistency
        if L > 2:
            v1 = vectors[:, :-1]  # [N, L-2, D]
            v2 = vectors[:, 1:]   # [N, L-2, D]
            
            v1_norm = v1 / (torch.norm(v1, dim=2, keepdim=True) + 1e-8)
            v2_norm = v2 / (torch.norm(v2, dim=2, keepdim=True) + 1e-8)
            
            cos_sim = (v1_norm * v2_norm).sum(dim=2)  # [N, L-2]
            direction_deviation = (1.0 - cos_sim)  # Lower is better (1 = smooth)
            direction_score = direction_deviation.mean(dim=1)  # [N]
            
            # Combine magnitude and direction
            vector_scores = 0.5 * magnitude_score + 0.5 * direction_score
        else:
            vector_scores = magnitude_score
        
        # ===== 2. Spectral Analysis Score (GPU) =====
        fft_result = torch.fft.fft(trajectories, dim=1)  # [N, L, D]
        power_spectrum = torch.abs(fft_result) ** 2  # [N, L, D]
        
        # Normalized spectral difference
        spectral_diff = torch.abs(power_spectrum - self.attractor.mean_spectrum) / (self.attractor.std_spectrum + 1e-8)
        spectral_scores = spectral_diff.mean(dim=[1, 2])  # [N]
        
        # ===== 3. Combined Score (GPU) =====
        # Z-score normalization for combining
        vector_mean = vector_scores.mean()
        vector_std = vector_scores.std()
        vector_scores_norm = (vector_scores - vector_mean) / (vector_std + 1e-8)
        
        spectral_mean = spectral_scores.mean()
        spectral_std = spectral_scores.std()
        spectral_scores_norm = (spectral_scores - spectral_mean) / (spectral_std + 1e-8)
        
        # Weighted combination (equal weights by default)
        anomaly_scores = 0.5 * vector_scores_norm + 0.5 * spectral_scores_norm
        
        # ===== 4. Reshape to spatial maps (GPU) =====
        anomaly_map_gpu = anomaly_scores.reshape(H, W)  # [H, W] on GPU
        vector_map_gpu = vector_scores.reshape(H, W)    # [H, W] on GPU
        spectral_map_gpu = spectral_scores.reshape(H, W)  # [H, W] on GPU
        
        # ===== 5. Thresholding (GPU) =====
        patch_mask_gpu = anomaly_map_gpu > threshold  # [H, W] boolean tensor on GPU
        
        return anomaly_map_gpu, patch_mask_gpu, vector_map_gpu, spectral_map_gpu