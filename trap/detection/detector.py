import torch

from torch import Tensor

from trap.training.trainer import ModelTrainer


class PatchDetector:
    """Patch detector based on reconstruction error."""

    def __init__(self, model_trainer: ModelTrainer, device: str = "cuda", detection_cfg=None):
        if not model_trainer.fitted:
            raise ValueError("ModelTrainer must be fitted before creating PatchDetector")

        self.model_trainer = model_trainer
        self.model = model_trainer.model
        self.model_type = model_trainer.model_type
        self.device = device
        self.detection_cfg = detection_cfg or {}

        self.model.eval()

        print(f"  PatchDetector initialized on {device}")
        print(f"    Model type: {self.model_type}")
        print(f"    Detection method: Reconstruction error")

        threshold_method = self._cfg_get("threshold_method", "percentile")
        if threshold_method == "mean_std":
            multiplier = self._cfg_get("threshold_multiplier", 3.0)
            print(f"    Threshold: mean + {multiplier}*std")
        elif threshold_method == "median_mad":
            multiplier = self._cfg_get("mad_multiplier", 3.0)
            print(f"    Threshold: median + {multiplier}*MAD")
        elif threshold_method == "percentile":
            percentile = self._cfg_get("percentile", 95.0)
            print(f"    Threshold: top {percentile:.1f} percentile")
        else:
            print(f"    Threshold: custom method ({threshold_method})")

    def detect(self, test_embedding_gpu: Tensor):
        """
        Detect patches using model reconstruction error.

        Args:
            test_embedding_gpu: tensor of shape [H, W, L, D]
        """
        H, W, L, D = test_embedding_gpu.shape
        trajectories = test_embedding_gpu.reshape(-1, L, D)

        with torch.no_grad():
            anomaly_scores = self.model.compute_anomaly_score(trajectories)

        anomaly_map = anomaly_scores.reshape(H, W)
        threshold_tensor = self._compute_threshold(anomaly_scores)
        threshold = float(threshold_tensor.item())
        patch_mask = anomaly_map > threshold_tensor

        return anomaly_map, patch_mask, threshold

    def _cfg_get(self, key, default):
        cfg = self.detection_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _compute_threshold(self, scores: Tensor) -> Tensor:
        method = self._cfg_get("threshold_method", "percentile")
        if method == "mean_std":
            return self._threshold_mean_std(scores)
        if method == "median_mad":
            return self._threshold_median_mad(scores)
        if method == "percentile":
            return self._threshold_percentile(scores)
        raise ValueError(f"Unsupported threshold_method: {method}")

    def _threshold_mean_std(self, scores: Tensor) -> Tensor:
        mean = scores.mean()
        std = scores.std(unbiased=False)
        multiplier = self._cfg_get("threshold_multiplier", 3.0)
        return mean + multiplier * std

    def _threshold_median_mad(self, scores: Tensor) -> Tensor:
        median = scores.median()
        mad = (scores - median).abs().median()
        multiplier = self._cfg_get("mad_multiplier", 3.0)
        return median + multiplier * mad

    def _threshold_percentile(self, scores: Tensor) -> Tensor:
        percentile = float(self._cfg_get("percentile", 95.0))
        percentile = min(max(percentile, 0.0), 100.0)
        quantile = percentile / 100.0
        return torch.quantile(scores, quantile)
