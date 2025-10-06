"""
PatchDetector for Few-shot Patch Detection

This module performs patch detection by comparing test trajectories against
learned normal characteristics from ImageNet (Few-shot base learning).

Detection is based on absolute comparison with ImageNet statistics rather than
per-image normalization, enabling robust cross-domain detection.

Key Metrics:
  - Vector Field Score: magnitude and direction deviation
  - Spectral Score: frequency domain deviation
  - Curvature Score: trajectory smoothness deviation
  - Energy Score: kinetic and potential energy deviation (global)
  - Autocorrelation Score: temporal self-similarity deviation (global)
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
        print(f"    Detection method: Few-shot with absolute ImageNet comparison")
        print(f"    Metrics: Vector + Spectral + Curvature + Energy + Autocorr + Spatial")

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
            vector_map: [H, W] tensor - vector field scores
            spectral_map: [H, W] tensor - spectral scores
            curvature_map: [H, W] tensor - curvature scores
            energy_map: [H, W] tensor - energy scores (global)
            autocorr_map: [H, W] tensor - autocorrelation scores (global)
        """
        H, W, L, D = test_embedding_gpu.shape
        # Reshape to [N, L, D] where N = H*W spatial locations
        trajectories = test_embedding_gpu.reshape(-1, L, D)

        # ===== 1. Vector Field Score =====
        # Compute displacement vectors
        vectors = trajectories[:, 1:] - trajectories[:, :-1]  # [N, L-1, D]

        # Magnitude deviation from ImageNet reference (absolute comparison)
        magnitudes = torch.norm(vectors, dim=2)  # [N, L-1]
        magnitude_deviation = torch.abs(magnitudes - self.attractor.mean_magnitude) / (self.attractor.std_magnitude + 1e-8)
        magnitude_score = magnitude_deviation.mean(dim=1)  # [N]

        # Direction consistency
        if L > 2:
            v1 = vectors[:, :-1]  # [N, L-2, D]
            v2 = vectors[:, 1:]   # [N, L-2, D]

            # Normalize vectors
            v1_norm = v1 / (torch.norm(v1, dim=2, keepdim=True) + 1e-8)
            v2_norm = v2 / (torch.norm(v2, dim=2, keepdim=True) + 1e-8)

            # Cosine similarity
            cos_sim = (v1_norm * v2_norm).sum(dim=2)  # [N, L-2]

            # Deviation from ImageNet direction consistency
            direction_deviation = torch.abs(cos_sim - self.attractor.mean_direction)
            direction_score = direction_deviation.mean(dim=1)  # [N]

            # Combine magnitude and direction (weighted average)
            vector_scores = 0.5 * magnitude_score + 0.5 * direction_score
        else:
            vector_scores = magnitude_score

        # ===== 2. Spectral Score =====
        # FFT to analyze frequency components
        fft_result = torch.fft.fft(trajectories, dim=1)  # [N, L, D]
        power_spectrum = torch.abs(fft_result) ** 2  # [N, L, D]

        # Absolute deviation from ImageNet spectral characteristics
        spectral_diff = torch.abs(power_spectrum - self.attractor.mean_spectrum) / (self.attractor.std_spectrum + 1e-8)
        spectral_scores = spectral_diff.mean(dim=[1, 2])  # [N]

        # ===== 3. Curvature Score =====
        curvature_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if L > 2:
            # Second-order difference: acceleration
            acceleration = vectors[:, 1:] - vectors[:, :-1]  # [N, L-2, D]
            curvature = torch.norm(acceleration, dim=2)  # [N, L-2]

            # Absolute deviation from ImageNet curvature
            curvature_deviation = torch.abs(curvature - self.attractor.mean_curvature) / (self.attractor.std_curvature + 1e-8)
            curvature_scores = curvature_deviation.mean(dim=1)  # [N]

        # ===== 4. Energy Score (GLOBAL) =====
        # Kinetic energy: sum of squared velocities (per trajectory)
        kinetic_energy = (vectors ** 2).sum(dim=[1, 2])  # [N]
        kinetic_deviation = torch.abs(kinetic_energy - self.attractor.mean_kinetic_energy) / (self.attractor.std_kinetic_energy + 1e-8)

        # Potential energy: sum of squared accelerations (per trajectory)
        potential_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if L > 2:
            acceleration = vectors[:, 1:] - vectors[:, :-1]  # [N, L-2, D]
            potential_energy = (acceleration ** 2).sum(dim=[1, 2])  # [N]
            potential_deviation = torch.abs(potential_energy - self.attractor.mean_potential_energy) / (self.attractor.std_potential_energy + 1e-8)
            potential_scores = potential_deviation

        # Combined energy score
        energy_scores = (kinetic_deviation + potential_scores) / 2.0

        # ===== 5. Temporal Autocorrelation Score (GLOBAL) =====
        autocorr_scores = torch.zeros(trajectories.shape[0], device=self.device)
        if L > 1:
            # Center the trajectory
            traj_centered = trajectories - trajectories.mean(dim=1, keepdim=True)  # [N, L, D]

            # Autocorrelation at lag 1
            t1 = traj_centered[:, :-1, :]  # [N, L-1, D]
            t2 = traj_centered[:, 1:, :]   # [N, L-1, D]

            # Compute correlation per trajectory
            numerator = (t1 * t2).sum(dim=[1, 2])  # [N]
            denominator = torch.sqrt((t1 ** 2).sum(dim=[1, 2]) * (t2 ** 2).sum(dim=[1, 2])) + 1e-8
            autocorr = numerator / denominator  # [N]

            # Deviation from ImageNet autocorrelation
            autocorr_deviation = torch.abs(autocorr - self.attractor.mean_autocorr) / (self.attractor.std_autocorr + 1e-8)
            autocorr_scores = autocorr_deviation

        # ===== 6. Combined Anomaly Score =====
        # Use raw scores (already in absolute deviation units from ImageNet)
        # DO NOT normalize per-image to preserve absolute scale for threshold comparison

        # Weighted combination (equal weights by default)
        anomaly_scores = (vector_scores + spectral_scores + curvature_scores + energy_scores + autocorr_scores) / 5.0

        # ===== 7. Spatial Coherence Enhancement =====
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

        # ===== 8. Thresholding =====
        # Use threshold from Few-shot adaptation phase (absolute comparison)
        patch_mask = anomaly_map_final > threshold  # [H, W] boolean

        # ===== 9. Component Maps for Visualization =====
        vector_map = vector_scores.reshape(H, W)
        spectral_map = spectral_scores.reshape(H, W)
        curvature_map = curvature_scores.reshape(H, W)
        energy_map = energy_scores.reshape(H, W)
        autocorr_map = autocorr_scores.reshape(H, W)

        return anomaly_map_final, patch_mask, vector_map, spectral_map, curvature_map, energy_map, autocorr_map
