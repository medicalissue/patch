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
        std_spectrum: Std of spectral power
        mean_wavelet: Mean wavelet coefficients
        std_wavelet: Std of wavelet coefficients
        mean_stft: Mean STFT power
        std_stft: Std of STFT power
        mean_spectral_entropy: Mean spectral entropy
        std_spectral_entropy: Std of spectral entropy
        mean_hf_ratio: Mean high-frequency ratio
        std_hf_ratio: Std of high-frequency ratio
        mean_spectral_skewness: Mean spectral skewness
        std_spectral_skewness: Std of spectral skewness
    """

    def __init__(self, device='cuda'):
        """
        Initialize AttractorLearner

        Args:
            device: Torch device ('cuda' or 'cpu')
        """
        self.device = device
        self.fitted = False

        # Spectral analysis statistics
        self.mean_spectrum = None
        self.std_spectrum = None

        # Wavelet analysis statistics
        self.mean_wavelet = None
        self.std_wavelet = None

        # STFT statistics
        self.mean_stft = None
        self.std_stft = None

        # Spectral entropy statistics
        self.mean_spectral_entropy = None
        self.std_spectral_entropy = None

        # High-frequency ratio statistics
        self.mean_hf_ratio = None
        self.std_hf_ratio = None

        # Spectral skewness statistics
        self.mean_spectral_skewness = None
        self.std_spectral_skewness = None

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

        all_spectrums = []
        all_wavelets = []
        all_stfts = []
        all_spectral_entropies = []
        all_hf_ratios = []
        all_spectral_skewnesses = []

        for emb in clean_embeddings_gpu:
            H, W, L, D = emb.shape
            # Reshape to [N, L, D] where N = H*W spatial locations
            trajectories = emb.reshape(-1, L, D)

            # ===== 1. Spectral Statistics =====
            # FFT for each trajectory to analyze frequency components
            fft_result = torch.fft.fft(trajectories, dim=1)  # [N, L, D]
            power_spectrum = torch.abs(fft_result) ** 2  # [N, L, D]
            all_spectrums.append(power_spectrum)

            # ===== 2. Wavelet Statistics (Simple Haar-like wavelet) =====
            # Detail coefficients (high-frequency components)
            if L >= 2:
                half_L = L // 2
                detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2  # [N, L//2, D]
                wavelet_coeff = torch.abs(detail)  # [N, L//2, D]
                all_wavelets.append(wavelet_coeff)

            # ===== 3. STFT Statistics =====
            # Simple windowed FFT (using half-window)
            window_size = max(2, L // 4)
            stft_powers = []
            for i in range(0, L - window_size + 1, window_size // 2):
                window = trajectories[:, i:i+window_size, :]  # [N, window_size, D]
                window_fft = torch.fft.fft(window, dim=1)  # [N, window_size, D]
                window_power = torch.abs(window_fft) ** 2  # [N, window_size, D]
                stft_powers.append(window_power.mean(dim=1))  # [N, D]
            if stft_powers:
                stft_power = torch.stack(stft_powers, dim=1)  # [N, num_windows, D]
                all_stfts.append(stft_power)

            # ===== 4. Spectral Entropy =====
            # Entropy of normalized power spectrum
            power_norm = power_spectrum / (power_spectrum.sum(dim=1, keepdim=True) + 1e-8)  # [N, L, D]
            entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum(dim=1)  # [N, D]
            all_spectral_entropies.append(entropy)

            # ===== 5. High-Frequency Ratio =====
            # Ratio of high-frequency power to total power
            half_L = L // 2
            high_freq_power = power_spectrum[:, half_L:, :].sum(dim=1)  # [N, D]
            total_power = power_spectrum.sum(dim=1) + 1e-8  # [N, D]
            hf_ratio = high_freq_power / total_power  # [N, D]
            all_hf_ratios.append(hf_ratio)

            # ===== 6. Spectral Skewness =====
            # Third moment of power spectrum
            mean_power = power_spectrum.mean(dim=1, keepdim=True)  # [N, 1, D]
            std_power = power_spectrum.std(dim=1, keepdim=True) + 1e-8  # [N, 1, D]
            skewness = ((power_spectrum - mean_power) / std_power) ** 3  # [N, L, D]
            skewness = skewness.mean(dim=1)  # [N, D]
            all_spectral_skewnesses.append(skewness)

        # Concatenate all statistics across images
        all_spectrums = torch.cat(all_spectrums, dim=0)  # [N_total, L, D]

        # Compute mean and std spectrum
        self.mean_spectrum = all_spectrums.mean(dim=0)  # [L, D]
        self.std_spectrum = all_spectrums.std(dim=0)    # [L, D]

        # Wavelet statistics
        if all_wavelets:
            all_wavelets = torch.cat(all_wavelets, dim=0)  # [N_total, L//2, D]
            self.mean_wavelet = all_wavelets.mean(dim=0)  # [L//2, D]
            self.std_wavelet = all_wavelets.std(dim=0)    # [L//2, D]

        # STFT statistics
        if all_stfts:
            all_stfts = torch.cat(all_stfts, dim=0)  # [N_total, num_windows, D]
            self.mean_stft = all_stfts.mean(dim=0)  # [num_windows, D]
            self.std_stft = all_stfts.std(dim=0)    # [num_windows, D]

        # Spectral entropy statistics
        all_spectral_entropies = torch.cat(all_spectral_entropies, dim=0)  # [N_total, D]
        self.mean_spectral_entropy = all_spectral_entropies.mean(dim=0)  # [D]
        self.std_spectral_entropy = all_spectral_entropies.std(dim=0)    # [D]

        # High-frequency ratio statistics
        all_hf_ratios = torch.cat(all_hf_ratios, dim=0)  # [N_total, D]
        self.mean_hf_ratio = all_hf_ratios.mean(dim=0)  # [D]
        self.std_hf_ratio = all_hf_ratios.std(dim=0)    # [D]

        # Spectral skewness statistics
        all_spectral_skewnesses = torch.cat(all_spectral_skewnesses, dim=0)  # [N_total, D]
        self.mean_spectral_skewness = all_spectral_skewnesses.mean(dim=0)  # [D]
        self.std_spectral_skewness = all_spectral_skewnesses.std(dim=0)    # [D]

        self.fitted = True

        # Print learned statistics
        print(f"  ✓ Normal trajectory characteristics learned:")
        print(f"    Spectral:")
        print(f"      Power:      mean={self.mean_spectrum.mean().item():.4f}, std={self.std_spectrum.mean().item():.4f}")
        print(f"    Wavelet:")
        if self.mean_wavelet is not None:
            print(f"      Coeff:      mean={self.mean_wavelet.mean().item():.4f}, std={self.std_wavelet.mean().item():.4f}")
        print(f"    STFT:")
        if self.mean_stft is not None:
            print(f"      Power:      mean={self.mean_stft.mean().item():.4f}, std={self.std_stft.mean().item():.4f}")
        print(f"    Spectral Entropy:")
        print(f"      Entropy:    mean={self.mean_spectral_entropy.mean().item():.4f}, std={self.std_spectral_entropy.mean().item():.4f}")
        print(f"    High-Freq Ratio:")
        print(f"      Ratio:      mean={self.mean_hf_ratio.mean().item():.4f}, std={self.std_hf_ratio.mean().item():.4f}")
        print(f"    Spectral Skewness:")
        print(f"      Skewness:   mean={self.mean_spectral_skewness.mean().item():.4f}, std={self.std_spectral_skewness.mean().item():.4f}")

        return self
