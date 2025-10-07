"""
PatchDetector for Few-shot Patch Detection

This module performs patch detection by comparing test trajectories against
learned normal characteristics from ImageNet (Few-shot base learning).

Detection is based on absolute deviation comparison with ImageNet statistics.

Key Metrics:
  - Spectral Score: frequency domain deviation
  - Wavelet Score: wavelet coefficient deviation
  - STFT Score: short-time Fourier transform deviation
  - Spectral Entropy Score: spectral entropy deviation
  - High-Frequency Ratio Score: high-frequency power ratio deviation
  - Spectral Skewness Score: spectral skewness deviation
  - Spatial Coherence: local anomaly clustering
"""

import torch


class PatchDetector:
    """
    Few-shot Patch Detector

    Detects patches by comparing trajectories against learned normal characteristics.
    All operations are performed on GPU.

    The detector uses absolute comparison with ImageNet-learned statistics,
    making it suitable for cross-domain detection without per-image normalization.

    Attributes:
        attractor: AttractorLearner with learned statistics
        device: Torch device (cuda or cpu)
    """

    def __init__(self, attractor_learner, device='cuda'):
        """
        Initialize PatchDetector

        Args:
            attractor_learner: Fitted AttractorLearner with normal statistics
            device: Torch device ('cuda' or 'cpu')

        Raises:
            ValueError: If attractor_learner is not fitted
        """
        self.attractor = attractor_learner
        self.device = device

        if not self.attractor.fitted:
            raise ValueError("AttractorLearner must be fitted before creating PatchDetector")

        print(f"  PatchDetector initialized on {device}")
        print(f"    Detection method: Few-shot with absolute deviation")
        print(f"    Metrics: Spectral + Wavelet + STFT + Spectral Entropy + HF Ratio + Spectral Skewness + Spatial")

    def detect(self, test_embedding_gpu, threshold):
        """
        Detect patches using Few-shot learned statistics

        Compares test trajectory characteristics against ImageNet-learned statistics
        using absolute comparison (no per-image normalization).

        Args:
            test_embedding_gpu: [H, W, L, D] tensor on GPU
                               H, W: spatial dimensions
                               L: trajectory length
                               D: feature dimension
            threshold: Detection threshold (from Few-shot adaptation phase)

        Returns:
            anomaly_map: [H, W] tensor on GPU - combined anomaly scores
            patch_mask: [H, W] boolean tensor on GPU - binary detection mask
            spectral_map: [H, W] tensor - spectral scores
            wavelet_map: [H, W] tensor - wavelet scores
            stft_map: [H, W] tensor - STFT scores
            entropy_map: [H, W] tensor - spectral entropy scores
            hf_ratio_map: [H, W] tensor - high-frequency ratio scores
            skewness_map: [H, W] tensor - spectral skewness scores
        """
        H, W, L, D = test_embedding_gpu.shape
        # Reshape to [N, L, D] where N = H*W spatial locations
        trajectories = test_embedding_gpu.reshape(-1, L, D)

        # ===== 1. Spectral Score =====
        fft_result = torch.fft.fft(trajectories, dim=1)  # [N, L, D]
        power_spectrum = torch.abs(fft_result) ** 2  # [N, L, D]

        # Absolute deviation from ImageNet spectral characteristics
        spectral_diff = torch.abs(power_spectrum - self.attractor.mean_spectrum.unsqueeze(0)) / (self.attractor.std_spectrum.unsqueeze(0) + 1e-8)
        spectral_scores = spectral_diff.mean(dim=[1, 2])  # [N]

        # ===== 2. Wavelet Score =====
        wavelet_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if L >= 2 and self.attractor.mean_wavelet is not None:
            half_L = L // 2
            detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2  # [N, L//2, D]
            wavelet_coeff = torch.abs(detail)  # [N, L//2, D]

            # Absolute deviation
            wavelet_diff = torch.abs(wavelet_coeff - self.attractor.mean_wavelet.unsqueeze(0)) / (self.attractor.std_wavelet.unsqueeze(0) + 1e-8)
            wavelet_scores = wavelet_diff.mean(dim=[1, 2])  # [N]

        # ===== 3. STFT Score =====
        stft_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if self.attractor.mean_stft is not None:
            window_size = max(2, L // 4)
            stft_powers = []
            for i in range(0, L - window_size + 1, window_size // 2):
                window = trajectories[:, i:i+window_size, :]  # [N, window_size, D]
                window_fft = torch.fft.fft(window, dim=1)  # [N, window_size, D]
                window_power = torch.abs(window_fft) ** 2  # [N, window_size, D]
                stft_powers.append(window_power.mean(dim=1))  # [N, D]
            if stft_powers:
                stft_power = torch.stack(stft_powers, dim=1)  # [N, num_windows, D]

                # Absolute deviation
                stft_diff = torch.abs(stft_power - self.attractor.mean_stft.unsqueeze(0)) / (self.attractor.std_stft.unsqueeze(0) + 1e-8)
                stft_scores = stft_diff.mean(dim=[1, 2])  # [N]

        # ===== 4. Spectral Entropy Score =====
        # Entropy of normalized power spectrum
        power_norm = power_spectrum / (power_spectrum.sum(dim=1, keepdim=True) + 1e-8)  # [N, L, D]
        entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum(dim=1)  # [N, D]

        # Absolute deviation
        entropy_diff = torch.abs(entropy - self.attractor.mean_spectral_entropy.unsqueeze(0)) / (self.attractor.std_spectral_entropy.unsqueeze(0) + 1e-8)
        entropy_scores = entropy_diff.mean(dim=1)  # [N]

        # ===== 5. High-Frequency Ratio Score =====
        half_L = L // 2
        high_freq_power = power_spectrum[:, half_L:, :].sum(dim=1)  # [N, D]
        total_power = power_spectrum.sum(dim=1) + 1e-8  # [N, D]
        hf_ratio = high_freq_power / total_power  # [N, D]

        # Absolute deviation
        hf_diff = torch.abs(hf_ratio - self.attractor.mean_hf_ratio.unsqueeze(0)) / (self.attractor.std_hf_ratio.unsqueeze(0) + 1e-8)
        hf_ratio_scores = hf_diff.mean(dim=1)  # [N]

        # ===== 6. Spectral Skewness Score =====
        mean_power = power_spectrum.mean(dim=1, keepdim=True)  # [N, 1, D]
        std_power = power_spectrum.std(dim=1, keepdim=True) + 1e-8  # [N, 1, D]
        skewness = ((power_spectrum - mean_power) / std_power) ** 3  # [N, L, D]
        skewness = skewness.mean(dim=1)  # [N, D]

        # Absolute deviation
        skewness_diff = torch.abs(skewness - self.attractor.mean_spectral_skewness.unsqueeze(0)) / (self.attractor.std_spectral_skewness.unsqueeze(0) + 1e-8)
        skewness_scores = skewness_diff.mean(dim=1)  # [N]

        # ===== 7. Combined Anomaly Score =====
        # Weighted combination (equal weights by default)
        num_scores = 6
        anomaly_scores = (spectral_scores + wavelet_scores + stft_scores + entropy_scores + hf_ratio_scores + skewness_scores) / num_scores

        # ===== 8. Spatial Coherence Enhancement =====
        # Reshape to spatial maps
        anomaly_map_2d = anomaly_scores.reshape(H, W)

        # Apply spatial smoothing to enhance coherent regions
        if H >= 3 and W >= 3:
            # Simple 3x3 average pooling for spatial coherence
            anomaly_map_padded = torch.nn.functional.pad(anomaly_map_2d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            kernel = torch.ones(1, 1, 3, 3, device=self.device) / 9.0
            anomaly_map_smooth = torch.nn.functional.conv2d(anomaly_map_padded, kernel)
            anomaly_map_final = anomaly_map_smooth[0, 0]  # [H, W]
        else:
            anomaly_map_final = anomaly_map_2d

        # ===== 9. Thresholding =====
        patch_mask = anomaly_map_final > threshold  # [H, W] boolean

        # ===== 10. Component Maps for Visualization =====
        spectral_map = spectral_scores.reshape(H, W)
        wavelet_map = wavelet_scores.reshape(H, W)
        stft_map = stft_scores.reshape(H, W)
        entropy_map = entropy_scores.reshape(H, W)
        hf_ratio_map = hf_ratio_scores.reshape(H, W)
        skewness_map = skewness_scores.reshape(H, W)

        return anomaly_map_final, patch_mask, spectral_map, wavelet_map, stft_map, entropy_map, hf_ratio_map, skewness_map
