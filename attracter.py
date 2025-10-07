"""
AttractorLearner for Few-shot Patch Detection

This module learns normal trajectory characteristics from a few ImageNet samples.
The learned statistics serve as reference for detecting anomalies.

Key Statistics:
  - Spectral: frequency domain characteristics
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
        mean_spectrum: Mean spectral power distribution
        std_spectrum: Std spectral power distribution (for z-score)
        mean_wavelet: Mean wavelet coefficients
        cov_inv_wavelet: Inverse covariance matrix for wavelet
        mean_stft: Mean STFT power
        cov_inv_stft: Inverse covariance matrix for STFT
        mean_spectral_entropy: Mean spectral entropy
        cov_inv_spectral_entropy: Inverse covariance matrix for spectral entropy
        mean_hf_ratio: Mean high-frequency ratio
        cov_inv_hf_ratio: Inverse covariance matrix for high-frequency ratio
        mean_spectral_skewness: Mean spectral skewness
        cov_inv_spectral_skewness: Inverse covariance matrix for spectral skewness
    """

    def __init__(self, device='cuda'):
        """
        Initialize AttractorLearner

        Args:
            device: Torch device ('cuda' or 'cpu')
        """
        self.device = device
        self.fitted = False

        # Wavelet analysis statistics (Mahalanobis)
        self.mean_wavelet = None
        self.cov_inv_wavelet = None

        # STFT statistics (Mahalanobis)
        self.mean_stft = None
        self.cov_inv_stft = None

        # HHT/EMD statistics (Mahalanobis)
        self.mean_hht = None
        self.cov_inv_hht = None

        # Synchrosqueezed STFT statistics (Mahalanobis)
        self.mean_sst = None
        self.cov_inv_sst = None


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

        all_wavelets = []
        all_stfts = []
        all_hhts = []
        all_ssts = []

        for emb in clean_embeddings_gpu:
            H, W, L, D = emb.shape
            # Reshape to [N, L, D] where N = H*W spatial locations
            trajectories = emb.reshape(-1, L, D)

            # ===== 1. Wavelet Statistics (Simple Haar-like wavelet) =====
            # Detail coefficients (high-frequency components)
            if L >= 2:
                half_L = L // 2
                detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2  # [N, L//2, D]
                wavelet_coeff = torch.abs(detail)  # [N, L//2, D]
                all_wavelets.append(wavelet_coeff)

            # ===== 2. STFT Statistics =====
            # Simple windowed RFFT (using half-window)
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
                all_stfts.append(stft_power)

            # ===== 3. HHT/EMD Statistics (Simplified EMD) =====
            # Simplified Empirical Mode Decomposition: extract intrinsic mode functions
            # We use iterative sifting to extract IMFs
            N = trajectories.shape[0]
            residual = trajectories.clone()  # [N, L, D]
            imfs = []

            for _ in range(min(3, L // 2)):  # Extract up to 3 IMFs
                # Simple sifting: subtract local mean (moving average)
                if L >= 3:
                    # Compute local mean using simple convolution
                    padded = torch.nn.functional.pad(residual, (0, 0, 1, 1), mode='replicate')  # [N, L+2, D]
                    local_mean = (padded[:, :-2, :] + padded[:, 1:-1, :] + padded[:, 2:, :]) / 3.0  # [N, L, D]
                    imf = residual - local_mean
                    imfs.append(torch.abs(imf).mean(dim=1))  # [N, D]
                    residual = local_mean
                else:
                    break

            if imfs:
                hht_features = torch.stack(imfs, dim=1)  # [N, num_imfs, D]
                all_hhts.append(hht_features)

            # ===== 4. Synchrosqueezed STFT (SST) =====
            # SST: Time-frequency reassignment for better localization
            # Simplified version: compute instantaneous frequency and reassign
            window_size = max(2, L // 4)
            sst_features = []

            for i in range(0, L - window_size + 1, window_size // 2):
                window = trajectories[:, i:i+window_size, :]  # [N, window_size, D]

                # Compute STFT
                window_fft = torch.fft.rfft(window, dim=1)  # [N, window_size//2+1, D]

                # Compute phase derivative (instantaneous frequency)
                if i > 0 and i + window_size <= L:
                    prev_window = trajectories[:, i-1:i-1+window_size, :]
                    prev_fft = torch.fft.rfft(prev_window, dim=1)

                    # Phase difference
                    phase_curr = torch.angle(window_fft)
                    phase_prev = torch.angle(prev_fft)
                    inst_freq = torch.abs(phase_curr - phase_prev)  # [N, window_size//2+1, D]

                    # Weighted by magnitude
                    magnitude = torch.abs(window_fft)
                    sst_feature = (inst_freq * magnitude).sum(dim=1)  # [N, D]
                else:
                    # Fallback: just use magnitude
                    sst_feature = torch.abs(window_fft).mean(dim=1)  # [N, D]

                sst_features.append(sst_feature)

            if sst_features:
                sst_power = torch.stack(sst_features, dim=1)  # [N, num_windows, D]
                all_ssts.append(sst_power)

        # Concatenate all statistics across images
        # Wavelet statistics
        if all_wavelets:
            all_wavelets = torch.cat(all_wavelets, dim=0)  # [N_total, L//2, D]
            N = all_wavelets.shape[0]
            all_wavelets_flat = all_wavelets.reshape(N, -1)  # [N_total, (L//2)*D]
            self.mean_wavelet = all_wavelets_flat.mean(dim=0)  # [(L//2)*D]

            centered = all_wavelets_flat - self.mean_wavelet.unsqueeze(0)
            cov = (centered.T @ centered) / (N - 1)
            cov += torch.eye(cov.shape[0], device=self.device) * 1e-6
            self.cov_inv_wavelet = torch.linalg.inv(cov)

        # STFT statistics
        if all_stfts:
            all_stfts = torch.cat(all_stfts, dim=0)  # [N_total, num_windows, D]
            N = all_stfts.shape[0]
            all_stfts_flat = all_stfts.reshape(N, -1)  # [N_total, num_windows*D]
            self.mean_stft = all_stfts_flat.mean(dim=0)  # [num_windows*D]

            centered = all_stfts_flat - self.mean_stft.unsqueeze(0)
            cov = (centered.T @ centered) / (N - 1)
            cov += torch.eye(cov.shape[0], device=self.device) * 1e-6
            self.cov_inv_stft = torch.linalg.inv(cov)

        # HHT/EMD statistics
        if all_hhts:
            all_hhts = torch.cat(all_hhts, dim=0)  # [N_total, num_imfs, D]
            N = all_hhts.shape[0]
            all_hhts_flat = all_hhts.reshape(N, -1)  # [N_total, num_imfs*D]
            self.mean_hht = all_hhts_flat.mean(dim=0)  # [num_imfs*D]

            centered = all_hhts_flat - self.mean_hht.unsqueeze(0)
            cov = (centered.T @ centered) / (N - 1)
            cov += torch.eye(cov.shape[0], device=self.device) * 1e-6
            self.cov_inv_hht = torch.linalg.inv(cov)

        # Synchrosqueezed STFT statistics
        if all_ssts:
            all_ssts = torch.cat(all_ssts, dim=0)  # [N_total, num_windows, D]
            N = all_ssts.shape[0]
            all_ssts_flat = all_ssts.reshape(N, -1)  # [N_total, num_windows*D]
            self.mean_sst = all_ssts_flat.mean(dim=0)  # [num_windows*D]

            centered = all_ssts_flat - self.mean_sst.unsqueeze(0)
            cov = (centered.T @ centered) / (N - 1)
            cov += torch.eye(cov.shape[0], device=self.device) * 1e-6
            self.cov_inv_sst = torch.linalg.inv(cov)

        self.fitted = True

        # Print learned statistics
        print(f"  ✓ Normal trajectory characteristics learned:")
        print(f"    Wavelet (Mahalanobis):")
        if self.mean_wavelet is not None:
            print(f"      Mean dim:   {self.mean_wavelet.shape[0]}")
            print(f"      Cov inv:    {self.cov_inv_wavelet.shape}")
        print(f"    STFT (Mahalanobis):")
        if self.mean_stft is not None:
            print(f"      Mean dim:   {self.mean_stft.shape[0]}")
            print(f"      Cov inv:    {self.cov_inv_stft.shape}")
        print(f"    HHT/EMD (Mahalanobis):")
        if self.mean_hht is not None:
            print(f"      Mean dim:   {self.mean_hht.shape[0]}")
            print(f"      Cov inv:    {self.cov_inv_hht.shape}")
        print(f"    SST (Mahalanobis):")
        if self.mean_sst is not None:
            print(f"      Mean dim:   {self.mean_sst.shape[0]}")
            print(f"      Cov inv:    {self.cov_inv_sst.shape}")

        return self
