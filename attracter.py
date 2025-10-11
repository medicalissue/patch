"""
ModelTrainer for Model-based Patch Detection

This module trains time-series anomaly detection models on clean images.
Three model types are supported:
1. Autoencoder - LSTM-based reconstruction
2. VAE - Variational autoencoder with probabilistic modeling
3. Transformer - Attention-based temporal modeling

Models are trained only on clean trajectories (Phase 1).
Optional LoRA-based domain adaptation (Phase 2) is supported.
"""

import torch
import torch.nn as nn
from pathlib import Path
import hashlib
from tqdm import tqdm

from models import create_model, apply_lora_to_model


class ModelTrainer:
    """
    Model-based Anomaly Detection Trainer

    Trains neural network models on clean trajectory data for anomaly detection.
    Supports saving/loading weights and optional LoRA-based domain adaptation.

    Attributes:
        model: Neural network model (Autoencoder/VAE/Transformer)
        device: Torch device (cuda or cpu)
        model_type: Type of model ('autoencoder', 'vae', 'transformer')
        optimizer: PyTorch optimizer for training
        fitted: Whether the model has been trained
    """

    def __init__(self, model_type, input_dim, device='cuda', model_cfg=None):
        """
        Initialize ModelTrainer

        Args:
            model_type: 'autoencoder', 'vae', or 'transformer'
            input_dim: Feature dimension (D)
            device: Torch device ('cuda' or 'cpu')
            model_cfg: Model configuration (Hydra DictConfig or dict)
        """
        self.device = device
        self.model_type = model_type.lower()
        self.model_cfg = model_cfg or {}
        self.fitted = False

        # Create model
        hidden_dim = self._cfg_get('hidden_dim', 128)
        latent_dim = self._cfg_get('latent_dim', 64)
        num_layers = self._cfg_get('num_layers', 2)
        num_heads = self._cfg_get('num_heads', 4)

        self.model = create_model(
            model_type=self.model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.model.to(device)

        # Create optimizer
        lr = self._cfg_get('learning_rate', 0.001)
        weight_decay = self._cfg_get('weight_decay', 0.0001)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"  ✓ ModelTrainer initialized")
        print(f"    Model type: {self.model_type}")
        print(f"    Input dim: {input_dim}")
        print(f"    Hidden dim: {hidden_dim}")
        if self.model_type in ['autoencoder', 'vae']:
            print(f"    Latent dim: {latent_dim}")
        print(f"    Num layers: {num_layers}")
        if self.model_type == 'transformer':
            print(f"    Num heads: {num_heads}")
        print(f"    Learning rate: {lr}")

    def _cfg_get(self, key, default):
        """Safely access configuration values"""
        cfg = self.model_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def train(self, train_embeddings_gpu, num_epochs=10, batch_size=128):
        """
        Train model on clean embeddings

        Args:
            train_embeddings_gpu: List of [H, W, L, D] tensors on GPU (clean images)
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            self: For method chaining
        """
        print(f"\n  [Phase 1: Model Training]")
        print(f"  Training {self.model_type} on {len(train_embeddings_gpu)} clean images...")
        print(f"  Epochs: {num_epochs}, Batch size: {batch_size}")

        # Prepare training data: flatten spatial dimensions
        all_trajectories = []
        for emb in train_embeddings_gpu:
            H, W, L, D = emb.shape
            trajectories = emb.reshape(-1, L, D)  # [H*W, L, D]
            all_trajectories.append(trajectories)

        # Concatenate all trajectories
        train_data = torch.cat(all_trajectories, dim=0)  # [N_total, L, D]
        print(f"    Total trajectories: {train_data.shape[0]}")

        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle data
            perm = torch.randperm(train_data.shape[0], device=self.device)
            shuffled_data = train_data[perm]

            # Mini-batch training
            for i in range(0, train_data.shape[0], batch_size):
                batch = shuffled_data[i:i+batch_size]

                # Forward pass
                self.optimizer.zero_grad()

                if self.model_type == 'vae':
                    reconstruction, mu, logvar = self.model(batch)

                    # VAE loss: reconstruction + KL divergence
                    recon_loss = nn.functional.mse_loss(reconstruction, batch)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.001 * kl_loss  # Weight KL lower
                else:
                    reconstruction, _ = self.model(batch)
                    loss = nn.functional.mse_loss(reconstruction, batch)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

        self.fitted = True
        print(f"  ✓ Model training completed")

        return self

    def save_weights(self, weights_path):
        """
        Save model weights to disk

        Args:
            weights_path: Path to save the model weights
        """
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'fitted': self.fitted
        }

        torch.save(state, weights_path)
        print(f"  ✓ Model weights saved to: {weights_path}")

    def load_weights(self, weights_path):
        """
        Load model weights from disk

        Args:
            weights_path: Path to the model weights file

        Returns:
            self: For method chaining

        Raises:
            FileNotFoundError: If weights file doesn't exist
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        state = torch.load(weights_path, map_location=self.device)

        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.model_type = state['model_type']
        self.fitted = state['fitted']

        print(f"  ✓ Model weights loaded from: {weights_path}")

        return self

    def adapt_with_lora(self, domain_embeddings_gpu, lora_cfg, num_epochs=5, batch_size=32):
        """
        Adapt model to domain using LoRA (Low-Rank Adaptation)

        This method freezes the base model and adds trainable LoRA layers
        for efficient domain adaptation.

        Args:
            domain_embeddings_gpu: List of [H, W, L, D] tensors from domain clean images
            lora_cfg: LoRA configuration (Hydra DictConfig or dict)
            num_epochs: Number of adaptation epochs
            batch_size: Batch size for adaptation

        Returns:
            self: For method chaining
        """
        print(f"\n  [Phase 2: LoRA Domain Adaptation]")
        print(f"  Adapting model to domain with {len(domain_embeddings_gpu)} clean images...")

        # Extract LoRA config
        if isinstance(lora_cfg, dict):
            rank = lora_cfg.get('rank', 8)
            alpha = lora_cfg.get('alpha', 16)
            target_modules = lora_cfg.get('target_modules', ['Linear'])
        else:
            rank = getattr(lora_cfg, 'rank', 8)
            alpha = getattr(lora_cfg, 'alpha', 16)
            target_modules = getattr(lora_cfg, 'target_modules', ['Linear'])

        print(f"    LoRA rank: {rank}")
        print(f"    LoRA alpha: {alpha}")
        print(f"    Target modules: {target_modules}")

        # Apply LoRA to model
        self.model, lora_params = apply_lora_to_model(
            self.model, rank, alpha, target_modules
        )

        # Create optimizer for LoRA parameters only
        if isinstance(lora_cfg, dict):
            lr = lora_cfg.get('learning_rate', 0.0001)
            weight_decay = lora_cfg.get('weight_decay', 0.0001)
        else:
            # Access from parent domain_adaptation config
            lr = 0.0001
            weight_decay = 0.0001

        lora_optimizer = torch.optim.Adam(
            lora_params,
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"    LoRA learning rate: {lr}")
        print(f"    Epochs: {num_epochs}, Batch size: {batch_size}")

        # Prepare domain data
        all_trajectories = []
        for emb in domain_embeddings_gpu:
            H, W, L, D = emb.shape
            trajectories = emb.reshape(-1, L, D)
            all_trajectories.append(trajectories)

        domain_data = torch.cat(all_trajectories, dim=0)
        print(f"    Total domain trajectories: {domain_data.shape[0]}")

        # LoRA training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle data
            perm = torch.randperm(domain_data.shape[0], device=self.device)
            shuffled_data = domain_data[perm]

            # Mini-batch training
            for i in range(0, domain_data.shape[0], batch_size):
                batch = shuffled_data[i:i+batch_size]

                # Forward pass
                lora_optimizer.zero_grad()

                if self.model_type == 'vae':
                    reconstruction, mu, logvar = self.model(batch)
                    recon_loss = nn.functional.mse_loss(reconstruction, batch)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + 0.001 * kl_loss
                else:
                    reconstruction, _ = self.model(batch)
                    loss = nn.functional.mse_loss(reconstruction, batch)

                # Backward pass (only LoRA parameters updated)
                loss.backward()
                lora_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

        print(f"  ✓ LoRA domain adaptation completed")

        return self

    def save_lora_weights(self, lora_weights_path):
        """
        Save only LoRA adaptation weights

        Args:
            lora_weights_path: Path to save LoRA weights
        """
        lora_weights_path = Path(lora_weights_path)
        lora_weights_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract LoRA parameters
        lora_state = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora'):
                lora_state[name] = module.lora.state_dict()

        torch.save(lora_state, lora_weights_path)
        print(f"  ✓ LoRA weights saved to: {lora_weights_path}")

    def load_lora_weights(self, lora_weights_path):
        """
        Load LoRA adaptation weights

        Args:
            lora_weights_path: Path to LoRA weights file

        Returns:
            self: For method chaining

        Raises:
            FileNotFoundError: If LoRA weights file doesn't exist
        """
        lora_weights_path = Path(lora_weights_path)
        if not lora_weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")

        lora_state = torch.load(lora_weights_path, map_location=self.device)

        # Load LoRA parameters
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora') and name in lora_state:
                module.lora.load_state_dict(lora_state[name])

        print(f"  ✓ LoRA weights loaded from: {lora_weights_path}")

        return self

    @staticmethod
    def get_weights_filename(model_type, imagenet_path, num_samples, spatial_resolution,
                           feature_dim, hidden_dim, latent_dim, num_layers):
        """
        Generate a unique weights filename based on configuration

        Args:
            model_type: Model type ('autoencoder', 'vae', 'transformer')
            imagenet_path: Path to ImageNet dataset
            num_samples: Number of samples used for training
            spatial_resolution: Spatial resolution of features
            feature_dim: Feature dimension
            hidden_dim: Hidden dimension
            latent_dim: Latent dimension
            num_layers: Number of layers

        Returns:
            str: Weights filename
        """
        # Create a hash of the configuration
        config_str = (f"{imagenet_path}_{num_samples}_{spatial_resolution}_{feature_dim}_"
                     f"{hidden_dim}_{latent_dim}_{num_layers}")
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return f"{model_type}_s{num_samples}_r{spatial_resolution}_d{feature_dim}_{config_hash}.pt"

    @staticmethod
    def get_lora_weights_filename(model_type, domain_path, num_samples, rank, alpha):
        """
        Generate a unique LoRA weights filename

        Args:
            model_type: Model type
            domain_path: Path to domain dataset
            num_samples: Number of domain samples
            rank: LoRA rank
            alpha: LoRA alpha

        Returns:
            str: LoRA weights filename
        """
        config_str = f"{model_type}_{domain_path}_{num_samples}_{rank}_{alpha}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return f"lora_{model_type}_r{rank}_a{alpha}_{config_hash}.pt"
