"""
AttractorLearner for Few-shot Patch Detection

This module learns normal trajectory characteristics from a few ImageNet samples.
The learned statistics serve as reference for detecting anomalies.

Key Statistics:
  - Spectral: frequency domain characteristics
"""

import torch
import pickle
from pathlib import Path
import hashlib
import json


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

        # Incremental statistics (Welford's algorithm)
        self.n_samples = 0
        self.M2_wavelet = None  # Sum of squared differences for covariance
        self.M2_stft = None
        self.M2_hht = None
        self.M2_sst = None


    def partial_fit(self, clean_embeddings_gpu):
        """
        Incrementally update normal trajectory characteristics from clean images

        This method uses batch-wise Welford's algorithm to compute mean and covariance
        without loading all data into memory at once. GPU-optimized for speed.

        Args:
            clean_embeddings_gpu: List of [H, W, L, D] tensors on GPU (mini-batch)
                                 H, W: spatial dimensions
                                 L: trajectory length (number of layers)
                                 D: feature dimension

        Returns:
            self: For method chaining
        """
        all_wavelets = []
        all_stfts = []
        all_hhts = []
        all_ssts = []

        # Compute statistics for all embeddings in batch (GPU parallel processing)
        for emb in clean_embeddings_gpu:
            H, W, L, D = emb.shape
            trajectories = emb.reshape(-1, L, D)

            # ===== 1. Wavelet Statistics =====
            if L >= 2:
                half_L = L // 2
                detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2
                wavelet_coeff = torch.abs(detail)
                all_wavelets.append(wavelet_coeff)

            # ===== 2. STFT Statistics =====
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
                all_stfts.append(stft_power)

            # ===== 3. HHT/EMD Statistics =====
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
                all_hhts.append(hht_features)

            # ===== 4. Synchrosqueezed STFT =====
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
                all_ssts.append(sst_power)

        # Batch update statistics (one call per metric type)
        if all_wavelets:
            all_wavelets = torch.cat(all_wavelets, dim=0)
            all_wavelets_flat = all_wavelets.reshape(all_wavelets.shape[0], -1)
            self._update_statistics('wavelet', all_wavelets_flat)
            del all_wavelets, all_wavelets_flat

        if all_stfts:
            all_stfts = torch.cat(all_stfts, dim=0)
            all_stfts_flat = all_stfts.reshape(all_stfts.shape[0], -1)
            self._update_statistics('stft', all_stfts_flat)
            del all_stfts, all_stfts_flat

        if all_hhts:
            all_hhts = torch.cat(all_hhts, dim=0)
            all_hhts_flat = all_hhts.reshape(all_hhts.shape[0], -1)
            self._update_statistics('hht', all_hhts_flat)
            del all_hhts, all_hhts_flat

        if all_ssts:
            all_ssts = torch.cat(all_ssts, dim=0)
            all_ssts_flat = all_ssts.reshape(all_ssts.shape[0], -1)
            self._update_statistics('sst', all_ssts_flat)
            del all_ssts, all_ssts_flat

        # Explicit memory cleanup
        torch.cuda.empty_cache()

        return self

    def _update_statistics(self, stat_type, batch_data):
        """
        Update mean and M2 using batch-wise Welford's algorithm (GPU-optimized)

        Args:
            stat_type: 'wavelet', 'stft', 'hht', or 'sst'
            batch_data: [N_batch, D] tensor
        """
        N_batch = batch_data.shape[0]

        # Get corresponding mean and M2
        if stat_type == 'wavelet':
            mean_attr = 'mean_wavelet'
            M2_attr = 'M2_wavelet'
        elif stat_type == 'stft':
            mean_attr = 'mean_stft'
            M2_attr = 'M2_stft'
        elif stat_type == 'hht':
            mean_attr = 'mean_hht'
            M2_attr = 'M2_hht'
        elif stat_type == 'sst':
            mean_attr = 'mean_sst'
            M2_attr = 'M2_sst'

        # Initialize if first batch
        if getattr(self, mean_attr) is None:
            setattr(self, mean_attr, torch.zeros(batch_data.shape[1], device=self.device))
            setattr(self, M2_attr, torch.zeros(batch_data.shape[1], device=self.device))

        old_mean = getattr(self, mean_attr)
        old_M2 = getattr(self, M2_attr)
        old_n = self.n_samples

        # Batch-wise update (GPU-accelerated)
        # Compute batch statistics
        batch_mean = batch_data.mean(dim=0)  # [D]
        batch_var = batch_data.var(dim=0, unbiased=False)  # [D]

        # Update count
        new_n = old_n + N_batch

        # Combine means
        delta = batch_mean - old_mean
        new_mean = old_mean + delta * (N_batch / new_n)

        # Combine M2 using parallel algorithm
        new_M2 = old_M2 + batch_var * N_batch + delta ** 2 * (old_n * N_batch / new_n)

        setattr(self, mean_attr, new_mean)
        setattr(self, M2_attr, new_M2)
        self.n_samples = new_n

    def finalize(self):
        """
        Finalize statistics by computing covariance inverse from accumulated M2

        Call this after all partial_fit calls are complete.

        Returns:
            self: For method chaining
        """
        print(f"  Finalizing statistics from {self.n_samples} samples...")

        # Compute covariance matrices from M2
        if self.M2_wavelet is not None and self.n_samples > 1:
            # For single-dimensional covariance, M2/(n-1) gives variance
            # For multi-dimensional, we need full covariance matrix
            # Here we use diagonal approximation for efficiency
            var_wavelet = self.M2_wavelet / (self.n_samples - 1)
            cov_wavelet = torch.diag(var_wavelet)
            cov_wavelet += torch.eye(cov_wavelet.shape[0], device=self.device) * 1e-6
            self.cov_inv_wavelet = torch.linalg.inv(cov_wavelet)

        if self.M2_stft is not None and self.n_samples > 1:
            var_stft = self.M2_stft / (self.n_samples - 1)
            cov_stft = torch.diag(var_stft)
            cov_stft += torch.eye(cov_stft.shape[0], device=self.device) * 1e-6
            self.cov_inv_stft = torch.linalg.inv(cov_stft)

        if self.M2_hht is not None and self.n_samples > 1:
            var_hht = self.M2_hht / (self.n_samples - 1)
            cov_hht = torch.diag(var_hht)
            cov_hht += torch.eye(cov_hht.shape[0], device=self.device) * 1e-6
            self.cov_inv_hht = torch.linalg.inv(cov_hht)

        if self.M2_sst is not None and self.n_samples > 1:
            var_sst = self.M2_sst / (self.n_samples - 1)
            cov_sst = torch.diag(var_sst)
            cov_sst += torch.eye(cov_sst.shape[0], device=self.device) * 1e-6
            self.cov_inv_sst = torch.linalg.inv(cov_sst)

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

    def save(self, cache_path):
        """
        Save attractor statistics to disk

        Args:
            cache_path: Path to save the attractor cache file
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'fitted': self.fitted,
            'n_samples': self.n_samples,
            'mean_wavelet': self.mean_wavelet,
            'cov_inv_wavelet': self.cov_inv_wavelet,
            'mean_stft': self.mean_stft,
            'cov_inv_stft': self.cov_inv_stft,
            'mean_hht': self.mean_hht,
            'cov_inv_hht': self.cov_inv_hht,
            'mean_sst': self.mean_sst,
            'cov_inv_sst': self.cov_inv_sst,
            'M2_wavelet': self.M2_wavelet,
            'M2_stft': self.M2_stft,
            'M2_hht': self.M2_hht,
            'M2_sst': self.M2_sst,
        }

        torch.save(state, cache_path)
        print(f"  ✓ Attractor saved to: {cache_path}")

    def load(self, cache_path):
        """
        Load attractor statistics from disk

        Args:
            cache_path: Path to the attractor cache file

        Returns:
            self: For method chaining

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Attractor cache not found: {cache_path}")

        state = torch.load(cache_path, map_location=self.device)

        self.fitted = state['fitted']
        self.n_samples = state['n_samples']
        self.mean_wavelet = state['mean_wavelet']
        self.cov_inv_wavelet = state['cov_inv_wavelet']
        self.mean_stft = state['mean_stft']
        self.cov_inv_stft = state['cov_inv_stft']
        self.mean_hht = state['mean_hht']
        self.cov_inv_hht = state['cov_inv_hht']
        self.mean_sst = state['mean_sst']
        self.cov_inv_sst = state['cov_inv_sst']
        self.M2_wavelet = state['M2_wavelet']
        self.M2_stft = state['M2_stft']
        self.M2_hht = state['M2_hht']
        self.M2_sst = state['M2_sst']

        print(f"  ✓ Attractor loaded from: {cache_path}")
        print(f"  ✓ Trained on {self.n_samples} samples")

        return self

    def adapt_to_domain(self, domain_embeddings_gpu):
        """
        Adapt ImageNet statistics to domain-specific distribution using statistical alignment

        This computes domain statistics and prepares adaptation parameters without neural networks.
        Uses CORAL-style covariance alignment for domain shift correction.

        Args:
            domain_embeddings_gpu: List of [H, W, L, D] tensors from domain clean images

        Returns:
            dict: Domain statistics for each metric type
        """
        print(f"\n  [Domain Adaptation]")
        print(f"  Computing domain statistics from {len(domain_embeddings_gpu)} clean images...")

        domain_stats = {}

        # Collect domain statistics (same feature extraction as ImageNet)
        all_wavelets = []
        all_stfts = []
        all_hhts = []
        all_ssts = []

        for emb in domain_embeddings_gpu:
            H, W, L, D = emb.shape
            trajectories = emb.reshape(-1, L, D)

            # Extract same features as ImageNet
            if L >= 2:
                half_L = L // 2
                detail = (trajectories[:, :half_L*2:2, :] - trajectories[:, 1:half_L*2:2, :]) / 2
                wavelet_coeff = torch.abs(detail)
                all_wavelets.append(wavelet_coeff)

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
                all_stfts.append(stft_power)

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
                all_hhts.append(hht_features)

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
                all_ssts.append(sst_power)

        # Compute domain statistics
        def compute_stats(data_list, name):
            if not data_list:
                return None
            data = torch.cat(data_list, dim=0)
            data_flat = data.reshape(data.shape[0], -1)

            mean_domain = data_flat.mean(dim=0)
            centered = data_flat - mean_domain
            cov_domain = (centered.T @ centered) / (data_flat.shape[0] - 1)
            cov_domain += torch.eye(cov_domain.shape[0], device=self.device) * 1e-6

            print(f"    {name}: domain mean shape {mean_domain.shape}")
            return {
                'mean': mean_domain,
                'cov': cov_domain
            }

        domain_stats['wavelet'] = compute_stats(all_wavelets, 'Wavelet')
        domain_stats['stft'] = compute_stats(all_stfts, 'STFT')
        domain_stats['hht'] = compute_stats(all_hhts, 'HHT/EMD')
        domain_stats['sst'] = compute_stats(all_ssts, 'SST')

        print(f"  ✓ Domain adaptation complete")

        return domain_stats

    @staticmethod
    def get_cache_filename(imagenet_path, num_samples, spatial_resolution, feature_dim):
        """
        Generate a unique cache filename based on configuration

        Args:
            imagenet_path: Path to ImageNet dataset
            num_samples: Number of samples used for training
            spatial_resolution: Spatial resolution of features
            feature_dim: Feature dimension

        Returns:
            str: Cache filename
        """
        # Create a hash of the configuration
        config_str = f"{imagenet_path}_{num_samples}_{spatial_resolution}_{feature_dim}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return f"attractor_s{num_samples}_r{spatial_resolution}_d{feature_dim}_{config_hash}.pt"
