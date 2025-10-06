"""
AttractorLearner for Few-shot Patch Detection

This module learns normal trajectory characteristics from a few ImageNet samples.
The learned statistics serve as reference for detecting anomalies.

Key Statistics:
  - Vector Field: magnitude and direction consistency
  - Spectral: frequency domain characteristics
  - Curvature: trajectory smoothness
  - Energy: kinetic and potential energy (global)
  - Autocorrelation: temporal self-similarity (global)
"""

import torch


class AttractorLearner:
    """
    Few-shot Attractor Learner

    Learns normal trajectory characteristics from clean ImageNet images.
    All operations are performed on GPU for efficiency.

    Attributes:
        device: Torch device (cuda or cpu)
        fitted: Whether the learner has been fitted
        mean_magnitude: Mean vector magnitude in normal trajectories
        std_magnitude: Std of vector magnitude
        mean_direction: Mean direction consistency (cosine similarity)
        mean_spectrum: Mean spectral power
        std_spectrum: Std of spectral power
        mean_curvature: Mean trajectory curvature
        std_curvature: Std of trajectory curvature
        mean_kinetic_energy: Mean kinetic energy (global trajectory)
        std_kinetic_energy: Std of kinetic energy
        mean_potential_energy: Mean potential energy (global trajectory)
        std_potential_energy: Std of potential energy
        mean_autocorr: Mean temporal autocorrelation
        std_autocorr: Std of temporal autocorrelation
    """

    def __init__(self, device='cuda'):
        """
        Initialize AttractorLearner

        Args:
            device: Torch device ('cuda' or 'cpu')
        """
        self.device = device
        self.fitted = False

        # Vector field statistics
        self.mean_magnitude = None
        self.std_magnitude = None
        self.mean_direction = None

        # Spectral analysis statistics
        self.mean_spectrum = None
        self.std_spectrum = None

        # Curvature statistics
        self.mean_curvature = None
        self.std_curvature = None

        # Energy statistics (global)
        self.mean_kinetic_energy = None
        self.std_kinetic_energy = None
        self.mean_potential_energy = None
        self.std_potential_energy = None

        # Autocorrelation statistics (global)
        self.mean_autocorr = None
        self.std_autocorr = None

    def fit(self, clean_embeddings_gpu):
        """
        Learn normal trajectory characteristics from clean images

        This implements Phase 1 (Few-shot Base Learning):
        Extract statistical characteristics of normal trajectories from ImageNet.

        Args:
            clean_embeddings_gpu: List of [H, W, L, D] tensors on GPU
                                 H, W: spatial dimensions
                                 L: trajectory length (number of layers)
                                 D: feature dimension

        Returns:
            self: For method chaining
        """
        print(f"  [Phase 1: Few-shot Base Learning]")
        print(f"  Learning normal trajectory characteristics from {len(clean_embeddings_gpu)} ImageNet samples...")

        all_magnitudes = []
        all_directions = []
        all_spectrums = []
        all_curvatures = []
        all_kinetic_energies = []
        all_potential_energies = []
        all_autocorrs = []

        for emb in clean_embeddings_gpu:
            H, W, L, D = emb.shape
            # Reshape to [N, L, D] where N = H*W spatial locations
            trajectories = emb.reshape(-1, L, D)

            # ===== 1. Vector Field Statistics =====
            # Compute displacement vectors between consecutive time steps
            vectors = trajectories[:, 1:] - trajectories[:, :-1]  # [N, L-1, D]

            # Magnitude: ||v_t||
            magnitudes = torch.norm(vectors, dim=2)  # [N, L-1]
            all_magnitudes.append(magnitudes.reshape(-1))

            # Direction consistency: cosine similarity between consecutive vectors
            if L > 2:
                v1 = vectors[:, :-1]  # [N, L-2, D]
                v2 = vectors[:, 1:]   # [N, L-2, D]

                # Normalize vectors
                v1_norm = v1 / (torch.norm(v1, dim=2, keepdim=True) + 1e-8)
                v2_norm = v2 / (torch.norm(v2, dim=2, keepdim=True) + 1e-8)

                # Cosine similarity: v1 · v2
                cos_sim = (v1_norm * v2_norm).sum(dim=2)  # [N, L-2]
                all_directions.append(cos_sim.reshape(-1))

            # ===== 2. Spectral Statistics =====
            # FFT for each trajectory to analyze frequency components
            fft_result = torch.fft.fft(trajectories, dim=1)  # [N, L, D]
            power_spectrum = torch.abs(fft_result) ** 2  # [N, L, D]
            all_spectrums.append(power_spectrum.reshape(-1))

            # ===== 3. Curvature Statistics =====
            # Second-order difference: acceleration
            if L > 2:
                acceleration = vectors[:, 1:] - vectors[:, :-1]  # [N, L-2, D]
                curvature = torch.norm(acceleration, dim=2)  # [N, L-2]
                all_curvatures.append(curvature.reshape(-1))

            # ===== 4. Energy Statistics (GLOBAL) =====
            # Kinetic energy: sum of squared velocities (per trajectory)
            kinetic_energy = (vectors ** 2).sum(dim=[1, 2])  # [N] - sum over time and features
            all_kinetic_energies.append(kinetic_energy)

            # Potential energy: sum of squared accelerations (per trajectory)
            if L > 2:
                potential_energy = (acceleration ** 2).sum(dim=[1, 2])  # [N]
                all_potential_energies.append(potential_energy)

            # ===== 5. Temporal Autocorrelation (GLOBAL) =====
            # Measure temporal self-similarity across entire trajectory
            # For each trajectory, compute autocorrelation at lag 1
            if L > 1:
                # Center the trajectory
                traj_centered = trajectories - trajectories.mean(dim=1, keepdim=True)  # [N, L, D]

                # Autocorrelation at lag 1: corr between t and t+1
                t1 = traj_centered[:, :-1, :]  # [N, L-1, D]
                t2 = traj_centered[:, 1:, :]   # [N, L-1, D]

                # Compute correlation per trajectory
                numerator = (t1 * t2).sum(dim=[1, 2])  # [N]
                denominator = torch.sqrt((t1 ** 2).sum(dim=[1, 2]) * (t2 ** 2).sum(dim=[1, 2])) + 1e-8
                autocorr = numerator / denominator  # [N]
                all_autocorrs.append(autocorr)

        # Concatenate all statistics across images
        all_magnitudes = torch.cat(all_magnitudes, dim=0)
        all_spectrums = torch.cat(all_spectrums, dim=0)

        # Compute mean and std for each statistic
        self.mean_magnitude = all_magnitudes.mean()
        self.std_magnitude = all_magnitudes.std()

        self.mean_spectrum = all_spectrums.mean()
        self.std_spectrum = all_spectrums.std()

        if len(all_directions) > 0:
            all_directions = torch.cat(all_directions, dim=0)
            self.mean_direction = all_directions.mean()
        else:
            self.mean_direction = torch.tensor(1.0, device=self.device)

        if len(all_curvatures) > 0:
            all_curvatures = torch.cat(all_curvatures, dim=0)
            self.mean_curvature = all_curvatures.mean()
            self.std_curvature = all_curvatures.std()
        else:
            self.mean_curvature = torch.tensor(0.0, device=self.device)
            self.std_curvature = torch.tensor(1.0, device=self.device)

        # Energy statistics
        if len(all_kinetic_energies) > 0:
            all_kinetic_energies = torch.cat(all_kinetic_energies, dim=0)
            self.mean_kinetic_energy = all_kinetic_energies.mean()
            self.std_kinetic_energy = all_kinetic_energies.std()
        else:
            self.mean_kinetic_energy = torch.tensor(0.0, device=self.device)
            self.std_kinetic_energy = torch.tensor(1.0, device=self.device)

        if len(all_potential_energies) > 0:
            all_potential_energies = torch.cat(all_potential_energies, dim=0)
            self.mean_potential_energy = all_potential_energies.mean()
            self.std_potential_energy = all_potential_energies.std()
        else:
            self.mean_potential_energy = torch.tensor(0.0, device=self.device)
            self.std_potential_energy = torch.tensor(1.0, device=self.device)

        # Autocorrelation statistics
        if len(all_autocorrs) > 0:
            all_autocorrs = torch.cat(all_autocorrs, dim=0)
            self.mean_autocorr = all_autocorrs.mean()
            self.std_autocorr = all_autocorrs.std()
        else:
            self.mean_autocorr = torch.tensor(0.0, device=self.device)
            self.std_autocorr = torch.tensor(1.0, device=self.device)

        self.fitted = True

        # Print learned statistics
        print(f"  ✓ Normal trajectory characteristics learned:")
        print(f"    Vector Field:")
        print(f"      Magnitude:  mean={self.mean_magnitude.item():.4f}, std={self.std_magnitude.item():.4f}")
        print(f"      Direction:  mean_cos_sim={self.mean_direction.item():.4f}")
        print(f"    Spectral:")
        print(f"      Power:      mean={self.mean_spectrum.item():.4f}, std={self.std_spectrum.item():.4f}")
        print(f"    Curvature:")
        print(f"      Curvature:  mean={self.mean_curvature.item():.4f}, std={self.std_curvature.item():.4f}")
        print(f"    Energy (Global):")
        print(f"      Kinetic:    mean={self.mean_kinetic_energy.item():.4f}, std={self.std_kinetic_energy.item():.4f}")
        print(f"      Potential:  mean={self.mean_potential_energy.item():.4f}, std={self.std_potential_energy.item():.4f}")
        print(f"    Autocorrelation (Global):")
        print(f"      Lag-1:      mean={self.mean_autocorr.item():.4f}, std={self.std_autocorr.item():.4f}")

        return self
