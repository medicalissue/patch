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
    Domain-Adapted Patch Detector

    Detects patches using domain-adapted Mahalanobis distance without thresholds.
    Adapts ImageNet statistics to domain distribution using CORAL-style alignment.
    All operations are performed on GPU without neural networks.

    Attributes:
        attractor: AttractorLearner with ImageNet statistics
        domain_stats: Domain-specific statistics for adaptation
        device: Torch device (cuda or cpu)
    """

    def __init__(self, attractor_learner, domain_stats, device='cuda', detection_cfg=None):
        """
        Initialize PatchDetector with domain adaptation

        Args:
            attractor_learner: Fitted AttractorLearner with ImageNet statistics
            domain_stats: Domain statistics from adapt_to_domain()
            device: Torch device ('cuda' or 'cpu')
            detection_cfg: Detection configuration (Hydra DictConfig or dict)

        Raises:
            ValueError: If attractor_learner is not fitted
        """
        self.attractor = attractor_learner
        self.domain_stats = domain_stats
        self.device = device
        self.detection_cfg = detection_cfg or {}

        if not self.attractor.fitted:
            raise ValueError("AttractorLearner must be fitted before creating PatchDetector")

        print(f"  PatchDetector initialized on {device}")
        print(f"    Detection method: Domain-adapted Mahalanobis distance")
        print(f"    Metrics: Wavelet + STFT + HHT + SST (no thresholds)")
        print(f"    Domain adaptation: CORAL-style statistical alignment")
        fusion_method = self._cfg_get('fusion_method', 'voting')
        print(f"    Fusion: {fusion_method}")
        threshold_method = self._cfg_get('score_threshold_method', 'percentile')
        if threshold_method == 'mean_std':
            multiplier = self._cfg_get('threshold_multiplier', 2.0)
            print(f"    Threshold: mean + {multiplier}*std")
        elif threshold_method == 'median_mad':
            multiplier = self._cfg_get('mad_multiplier', 3.0)
            print(f"    Threshold: median + {multiplier}*MAD")
        elif threshold_method == 'percentile':
            percentile = self._cfg_get('percentile', 90.0)
            print(f"    Threshold: top {percentile:.1f} percentile")
        else:
            print(f"    Threshold: custom method ({threshold_method})")

    def detect(self, test_embedding_gpu):
        """
        Detect patches using domain-adapted Mahalanobis distance with per-metric voting fusion

        Args:
            test_embedding_gpu: [H, W, L, D] tensor on GPU
                               H, W: spatial dimensions
                               L: trajectory length
                               D: feature dimension

        Returns:
            anomaly_map: [H, W] tensor on GPU - fused anomaly/vote scores
            patch_mask: [H, W] boolean tensor on GPU - binary detection mask (score-based threshold)
            wavelet_map: [H, W] tensor - wavelet scores
            stft_map: [H, W] tensor - STFT scores
            hht_map: [H, W] tensor - HHT/EMD scores
            sst_map: [H, W] tensor - Synchrosqueezed STFT scores
            thresholds: dict - per-metric scalar thresholds
            score_flags: dict - per-metric boolean maps after thresholding
        """
        H, W, L, D = test_embedding_gpu.shape
        trajectories = test_embedding_gpu.reshape(-1, L, D)

        # Helper function to compute domain-adapted Mahalanobis distance
        def compute_domain_adapted_score(features, metric_name):
            """Compute Mahalanobis distance to domain distribution"""
            if self.domain_stats.get(metric_name) is None:
                return torch.zeros(trajectories.shape[0], device=self.device)

            domain_mean = self.domain_stats[metric_name]['mean']
            domain_cov = self.domain_stats[metric_name]['cov']

            # Compute covariance inverse (cached for efficiency)
            cov_inv = torch.linalg.inv(domain_cov)

            # Mahalanobis distance to domain distribution
            diff = features - domain_mean.unsqueeze(0)
            mahal_sq = (diff @ cov_inv * diff).sum(dim=1)
            return torch.sqrt(mahal_sq + 1e-8)

        # ===== 1. Wavelet Score =====
        wavelet_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if L >= 2:
            half_L = L // 2
            detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2
            wavelet_coeff = torch.abs(detail)
            wavelet_flat = wavelet_coeff.reshape(trajectories.shape[0], -1)
            wavelet_scores = compute_domain_adapted_score(wavelet_flat, 'wavelet')

        # ===== 2. STFT Score =====
        stft_scores = torch.zeros(trajectories.shape[0], device=self.device)
        window_size = max(2, L // 4)
        stft_powers = []
        for i in range(0, L - window_size + 1, window_size // 2):
            window = trajectories[:, i:i+window_size, :]
            window_fft = torch.fft.rfft(window, dim=1)
            window_power = torch.abs(window_fft) ** 2
            log_window_power = torch.log10(window_power + 1e-10)
            stft_powers.append(log_window_power.mean(dim=1))
        if stft_powers:
            stft_power = torch.stack(stft_powers, dim=1)
            stft_flat = stft_power.reshape(trajectories.shape[0], -1)
            stft_scores = compute_domain_adapted_score(stft_flat, 'stft')

        # ===== 3. HHT/EMD Score =====
        hht_scores = torch.zeros(trajectories.shape[0], device=self.device)
        residual = trajectories
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
            hht_features = torch.stack(imfs, dim=1)
            hht_flat = hht_features.reshape(trajectories.shape[0], -1)
            hht_scores = compute_domain_adapted_score(hht_flat, 'hht')

        # ===== 4. Synchrosqueezed STFT Score =====
        sst_scores = torch.zeros(trajectories.shape[0], device=self.device)
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
            sst_power = torch.stack(sst_features, dim=1)
            sst_flat = sst_power.reshape(trajectories.shape[0], -1)
            sst_scores = compute_domain_adapted_score(sst_flat, 'sst')

        # ===== 5. Metric-wise thresholds =====
        metric_maps = {
            'wavelet': wavelet_scores.reshape(H, W),
            'stft': stft_scores.reshape(H, W),
            'hht': hht_scores.reshape(H, W),
            'sst': sst_scores.reshape(H, W)
        }

        thresholds = {}
        score_flags = {}

        for name, score_map in metric_maps.items():
            flat_scores = score_map.reshape(-1)
            threshold_tensor = self._compute_threshold(flat_scores)
            thresholds[name] = float(threshold_tensor.item())
            score_flags[name] = (flat_scores > threshold_tensor).reshape(H, W)

        # ===== 6. Fusion via voting =====
        metrics_order = ['wavelet', 'stft', 'hht', 'sst']
        flag_stack = torch.stack([score_flags[m] for m in metrics_order], dim=0)
        flag_float = flag_stack.float()

        fusion_method = self._cfg_get('fusion_method', 'voting')

        if fusion_method == 'voting':
            vote_map = flag_float.sum(dim=0)
            voting_threshold = float(self._cfg_get('voting_threshold', 3))
            patch_mask = vote_map >= voting_threshold
        elif fusion_method == 'weighted_voting':
            weights = self._cfg_get('score_weights', [1.0] * len(metrics_order))
            if len(weights) != len(metrics_order):
                raise ValueError(f"score_weights must have {len(metrics_order)} entries")
            weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device).view(-1, 1, 1)
            vote_map = (flag_float * weight_tensor).sum(dim=0)
            voting_threshold = float(self._cfg_get('voting_threshold', weight_tensor.sum().item()))
            patch_mask = vote_map >= voting_threshold
        elif fusion_method == 'all':
            vote_map = flag_float.sum(dim=0)
            patch_mask = flag_stack.all(dim=0)
        elif fusion_method == 'any':
            vote_map = flag_float.sum(dim=0)
            patch_mask = flag_stack.any(dim=0)
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        anomaly_map = vote_map

        return (
            anomaly_map,
            patch_mask,
            metric_maps['wavelet'],
            metric_maps['stft'],
            metric_maps['hht'],
            metric_maps['sst'],
            thresholds,
            score_flags
        )

    def _cfg_get(self, key, default):
        """Safely access detection configuration values"""
        cfg = self.detection_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _compute_threshold(self, scores):
        """Compute per-metric threshold based on configuration"""
        method = self._cfg_get('score_threshold_method', 'percentile')
        if method == 'mean_std':
            return self._threshold_mean_std(scores)
        if method == 'median_mad':
            return self._threshold_median_mad(scores)
        if method == 'percentile':
            return self._threshold_percentile(scores)
        raise ValueError(f"Unsupported score_threshold_method: {method}")

    def _threshold_mean_std(self, scores):
        """Mean + k*std threshold"""
        mean = scores.mean()
        std = scores.std(unbiased=False)
        multiplier = self._cfg_get('threshold_multiplier', 2.0)
        threshold = mean + multiplier * std
        return threshold

    def _threshold_median_mad(self, scores):
        """Median + c*MAD threshold (robust)"""
        median = scores.median()
        mad = (scores - median).abs().median()
        multiplier = self._cfg_get('mad_multiplier', 3.0)
        threshold = median + multiplier * mad
        return threshold

    def _threshold_percentile(self, scores):
        """Percentile-based threshold"""
        percentile = float(self._cfg_get('percentile', 90.0))
        percentile = min(max(percentile, 0.0), 100.0)
        quantile = percentile / 100.0
        threshold = torch.quantile(scores, quantile)
        return threshold
