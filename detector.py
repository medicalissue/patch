"""
PatchDetector for Model-based Patch Detection

This module performs patch detection using trained neural network models.
Detection is based on reconstruction error from the trained model.

High reconstruction error indicates anomalous trajectories (potential patches).
"""

import torch


class PatchDetector:
    """
    Model-based Patch Detector

    Detects patches using reconstruction error from trained neural network model.
    All operations are performed on GPU.

    Attributes:
        model_trainer: ModelTrainer with trained model
        device: Torch device (cuda or cpu)
        detection_cfg: Detection configuration
    """

    def __init__(self, model_trainer, device='cuda', detection_cfg=None):
        """
        Initialize PatchDetector

        Args:
            model_trainer: Trained ModelTrainer instance
            device: Torch device ('cuda' or 'cpu')
            detection_cfg: Detection configuration (Hydra DictConfig or dict)

        Raises:
            ValueError: If model_trainer is not fitted
        """
        self.model_trainer = model_trainer
        self.model = model_trainer.model
        self.model_type = model_trainer.model_type
        self.device = device
        self.detection_cfg = detection_cfg or {}

        if not self.model_trainer.fitted:
            raise ValueError("ModelTrainer must be fitted before creating PatchDetector")

        # Set model to evaluation mode
        self.model.eval()

        print(f"  PatchDetector initialized on {device}")
        print(f"    Model type: {self.model_type}")
        print(f"    Detection method: Reconstruction error")

        threshold_method = self._cfg_get('threshold_method', 'percentile')
        if threshold_method == 'mean_std':
            multiplier = self._cfg_get('threshold_multiplier', 3.0)
            print(f"    Threshold: mean + {multiplier}*std")
        elif threshold_method == 'median_mad':
            multiplier = self._cfg_get('mad_multiplier', 3.0)
            print(f"    Threshold: median + {multiplier}*MAD")
        elif threshold_method == 'percentile':
            percentile = self._cfg_get('percentile', 95.0)
            print(f"    Threshold: top {percentile:.1f} percentile")
        else:
            print(f"    Threshold: custom method ({threshold_method})")

    def detect(self, test_embedding_gpu):
        """
        Detect patches using model reconstruction error

        Args:
            test_embedding_gpu: [H, W, L, D] tensor on GPU
                               H, W: spatial dimensions
                               L: trajectory length
                               D: feature dimension

        Returns:
            anomaly_map: [H, W] tensor on GPU - reconstruction error scores
            patch_mask: [H, W] boolean tensor on GPU - binary detection mask
            threshold: float - computed threshold value
        """
        H, W, L, D = test_embedding_gpu.shape
        trajectories = test_embedding_gpu.reshape(-1, L, D)  # [H*W, L, D]

        with torch.no_grad():
            # Compute anomaly scores using model
            anomaly_scores = self.model.compute_anomaly_score(trajectories)  # [H*W]

        # Reshape to spatial map
        anomaly_map = anomaly_scores.reshape(H, W)  # [H, W]

        # Compute threshold from scores
        threshold_tensor = self._compute_threshold(anomaly_scores)
        threshold = float(threshold_tensor.item())

        # Create binary mask
        patch_mask = anomaly_map > threshold_tensor  # [H, W]

        return anomaly_map, patch_mask, threshold

    def _cfg_get(self, key, default):
        """Safely access detection configuration values"""
        cfg = self.detection_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _compute_threshold(self, scores):
        """
        Compute threshold based on configuration

        Args:
            scores: [N] tensor of anomaly scores

        Returns:
            threshold: scalar tensor
        """
        method = self._cfg_get('threshold_method', 'percentile')
        if method == 'mean_std':
            return self._threshold_mean_std(scores)
        elif method == 'median_mad':
            return self._threshold_median_mad(scores)
        elif method == 'percentile':
            return self._threshold_percentile(scores)
        else:
            raise ValueError(f"Unsupported threshold_method: {method}")

    def _threshold_mean_std(self, scores):
        """Mean + k*std threshold"""
        mean = scores.mean()
        std = scores.std(unbiased=False)
        multiplier = self._cfg_get('threshold_multiplier', 3.0)
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
        percentile = float(self._cfg_get('percentile', 95.0))
        percentile = min(max(percentile, 0.0), 100.0)
        quantile = percentile / 100.0
        threshold = torch.quantile(scores, quantile)
        return threshold
