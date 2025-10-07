"""
PatchDetector for Few-shot Patch Detection

This module performs patch detection by comparing test trajectories against
learned normal characteristics from ImageNet (Few-shot base learning).

Detection is based on absolute deviation comparison with ImageNet statistics.

Key Metrics:
  - Wavelet Score: wavelet coefficient deviation (Mahalanobis)
  - STFT Score: short-time Fourier transform deviation (Mahalanobis)
  - HHT/EMD Score: intrinsic mode functions deviation (Mahalanobis)
  - SST Score: synchrosqueezed STFT deviation (Mahalanobis)
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

    def __init__(self, attractor_learner, device='cuda', fusion_method='voting', voting_threshold=3, score_weights=None):
        """
        Initialize PatchDetector

        Args:
            attractor_learner: Fitted AttractorLearner with normal statistics
            device: Torch device ('cuda' or 'cpu')
            fusion_method: Score fusion method ('voting', 'weighted_voting', 'all', 'any')
            voting_threshold: Number of scores that must exceed threshold (for 'voting' method)
            score_weights: Weights for each score (for 'weighted_voting' method)

        Raises:
            ValueError: If attractor_learner is not fitted
        """
        self.attractor = attractor_learner
        self.device = device
        self.fusion_method = fusion_method
        self.voting_threshold = voting_threshold
        self.score_weights = score_weights if score_weights is not None else [1.0] * 4

        if not self.attractor.fitted:
            raise ValueError("AttractorLearner must be fitted before creating PatchDetector")

        print(f"  PatchDetector initialized on {device}")
        print(f"    Detection method: Per-score thresholds + {fusion_method}")
        print(f"    Metrics: Wavelet(M) + STFT(M) + HHT(M) + SST(M)")
        if fusion_method == 'voting':
            print(f"    Voting: {voting_threshold}/4 scores must be anomalous")

    def detect(self, test_embedding_gpu, thresholds):
        """
        Detect patches using Few-shot learned statistics with per-score thresholds

        Args:
            test_embedding_gpu: [H, W, L, D] tensor on GPU
                               H, W: spatial dimensions
                               L: trajectory length
                               D: feature dimension
            thresholds: dict with keys ['wavelet', 'stft', 'hht', 'sst']
                       Each value is a threshold for that score

        Returns:
            anomaly_map: [H, W] tensor on GPU - voting-based anomaly scores (number of votes)
            patch_mask: [H, W] boolean tensor on GPU - binary detection mask
            wavelet_map: [H, W] tensor - wavelet scores
            stft_map: [H, W] tensor - STFT scores
            hht_map: [H, W] tensor - HHT/EMD scores
            sst_map: [H, W] tensor - Synchrosqueezed STFT scores
            score_flags: dict - per-score binary flags for debugging
        """
        H, W, L, D = test_embedding_gpu.shape
        # Reshape to [N, L, D] where N = H*W spatial locations
        trajectories = test_embedding_gpu.reshape(-1, L, D)

        # ===== 1. Wavelet Score (Mahalanobis Distance) =====
        wavelet_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if L >= 2 and self.attractor.mean_wavelet is not None:
            half_L = L // 2
            detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2  # [N, L//2, D]
            wavelet_coeff = torch.abs(detail)  # [N, L//2, D]

            # Flatten for Mahalanobis distance
            wavelet_flat = wavelet_coeff.reshape(trajectories.shape[0], -1)  # [N, (L//2)*D]

            # Mahalanobis distance
            diff = wavelet_flat - self.attractor.mean_wavelet.unsqueeze(0)  # [N, (L//2)*D]
            mahal_sq = (diff @ self.attractor.cov_inv_wavelet * diff).sum(dim=1)  # [N]
            wavelet_scores = torch.sqrt(mahal_sq + 1e-8)  # [N]

        # ===== 2. STFT Score (Mahalanobis Distance) =====
        stft_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if self.attractor.mean_stft is not None:
            window_size = max(2, L // 4)
            stft_powers = []
            for i in range(0, L - window_size + 1, window_size // 2):
                window = trajectories[:, i:i+window_size, :]  # [N, window_size, D]
                window_fft = torch.fft.rfft(window, dim=1)  # [N, window_size//2+1, D]
                window_power = torch.abs(window_fft) ** 2  # [N, window_size//2+1, D]
                # Log PSD
                log_window_power = torch.log10(window_power + 1e-10)  # [N, window_size//2+1, D]
                stft_powers.append(log_window_power.mean(dim=1))  # [N, D]
            if stft_powers:
                stft_power = torch.stack(stft_powers, dim=1)  # [N, num_windows, D]

                # Flatten for Mahalanobis distance
                stft_flat = stft_power.reshape(trajectories.shape[0], -1)  # [N, num_windows*D]

                # Mahalanobis distance
                diff = stft_flat - self.attractor.mean_stft.unsqueeze(0)  # [N, num_windows*D]
                mahal_sq = (diff @ self.attractor.cov_inv_stft * diff).sum(dim=1)  # [N]
                stft_scores = torch.sqrt(mahal_sq + 1e-8)  # [N]

        # ===== 3. HHT/EMD Score (Mahalanobis Distance) =====
        hht_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if self.attractor.mean_hht is not None:
            # Compute IMFs (same as in attracter)
            N_traj = trajectories.shape[0]
            residual = trajectories.clone()
            imfs = []

            for _ in range(min(3, L // 2)):
                if L >= 3:
                    padded = torch.nn.functional.pad(residual, (0, 0, 1, 1), mode='replicate')
                    local_mean = (padded[:, :-2, :] + padded[:, 1:-1, :] + padded[:, 2:, :]) / 3.0
                    imf = residual - local_mean
                    imfs.append(torch.abs(imf).mean(dim=1))
                    residual = local_mean
                else:
                    break

            if imfs:
                hht_features = torch.stack(imfs, dim=1)  # [N, num_imfs, D]
                hht_flat = hht_features.reshape(trajectories.shape[0], -1)

                # Mahalanobis distance
                diff = hht_flat - self.attractor.mean_hht.unsqueeze(0)
                mahal_sq = (diff @ self.attractor.cov_inv_hht * diff).sum(dim=1)
                hht_scores = torch.sqrt(mahal_sq + 1e-8)

        # ===== 4. Synchrosqueezed STFT Score (Mahalanobis Distance) =====
        sst_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if self.attractor.mean_sst is not None:
            window_size = max(2, L // 4)
            sst_features = []

            for i in range(0, L - window_size + 1, window_size // 2):
                window = trajectories[:, i:i+window_size, :]
                window_fft = torch.fft.rfft(window, dim=1)

                if i > 0 and i + window_size <= L:
                    prev_window = trajectories[:, i-1:i-1+window_size, :]
                    prev_fft = torch.fft.rfft(prev_window, dim=1)

                    phase_curr = torch.angle(window_fft)
                    phase_prev = torch.angle(prev_fft)
                    inst_freq = torch.abs(phase_curr - phase_prev)

                    magnitude = torch.abs(window_fft)
                    sst_feature = (inst_freq * magnitude).sum(dim=1)
                else:
                    sst_feature = torch.abs(window_fft).mean(dim=1)

                sst_features.append(sst_feature)

            if sst_features:
                sst_power = torch.stack(sst_features, dim=1)  # [N, num_windows, D]
                sst_flat = sst_power.reshape(trajectories.shape[0], -1)

                # Mahalanobis distance
                diff = sst_flat - self.attractor.mean_sst.unsqueeze(0)
                mahal_sq = (diff @ self.attractor.cov_inv_sst * diff).sum(dim=1)
                sst_scores = torch.sqrt(mahal_sq + 1e-8)

        # ===== 5. Per-Score Thresholding =====
        # Compare each score against its own threshold
        flag_wavelet = wavelet_scores > thresholds['wavelet']  # [N]
        flag_stft = stft_scores > thresholds['stft']  # [N]
        flag_hht = hht_scores > thresholds['hht']  # [N]
        flag_sst = sst_scores > thresholds['sst']  # [N]

        # ===== 6. Score Fusion (Voting) =====
        if self.fusion_method == 'voting':
            # Count number of flags
            num_flags = (flag_wavelet.float() + flag_stft.float() +
                        flag_hht.float() + flag_sst.float())  # [N]
            # Voting: at least K scores must be anomalous
            anomaly_votes = num_flags  # [N]
            anomaly_flags = num_flags >= self.voting_threshold  # [N] boolean

        elif self.fusion_method == 'weighted_voting':
            # Weighted voting
            weighted_votes = (self.score_weights[0] * flag_wavelet.float() +
                            self.score_weights[1] * flag_stft.float() +
                            self.score_weights[2] * flag_hht.float() +
                            self.score_weights[3] * flag_sst.float())  # [N]
            total_weight = sum(self.score_weights)
            anomaly_votes = weighted_votes  # [N]
            anomaly_flags = weighted_votes >= (self.voting_threshold * total_weight / 4.0)  # [N] boolean

        elif self.fusion_method == 'all':
            # All scores must be anomalous (conservative)
            anomaly_flags = flag_wavelet & flag_stft & flag_hht & flag_sst
            anomaly_votes = (flag_wavelet.float() + flag_stft.float() +
                           flag_hht.float() + flag_sst.float())

        elif self.fusion_method == 'any':
            # Any score can be anomalous (sensitive)
            anomaly_flags = flag_wavelet | flag_stft | flag_hht | flag_sst
            anomaly_votes = (flag_wavelet.float() + flag_stft.float() +
                           flag_hht.float() + flag_sst.float())

        # ===== 7. Spatial Coherence Enhancement =====
        # Reshape votes to spatial map
        anomaly_map_2d = anomaly_votes.reshape(H, W)
        anomaly_flags_2d = anomaly_flags.reshape(H, W)

        # No spatial smoothing for integer vote counts
        anomaly_map_final = anomaly_map_2d  # [H, W]

        # Final mask
        patch_mask = anomaly_flags_2d  # [H, W] boolean

        # ===== 8. Component Maps for Visualization =====
        wavelet_map = wavelet_scores.reshape(H, W)
        stft_map = stft_scores.reshape(H, W)
        hht_map = hht_scores.reshape(H, W)
        sst_map = sst_scores.reshape(H, W)

        # Score flags for visualization
        score_flags = {
            'wavelet': flag_wavelet.reshape(H, W),
            'stft': flag_stft.reshape(H, W),
            'hht': flag_hht.reshape(H, W),
            'sst': flag_sst.reshape(H, W)
        }

        return anomaly_map_final, patch_mask, wavelet_map, stft_map, hht_map, sst_map, score_flags
