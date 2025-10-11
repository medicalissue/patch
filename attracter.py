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
from omegaconf import DictConfig, OmegaConf

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

    def __init__(self, model_type, input_dim, device='cuda', cfg=None, model_cfg=None):
        """
        Initialize ModelTrainer

        Args:
            model_type: 'autoencoder', 'vae', or 'transformer'
            input_dim: Feature dimension (D)
            device: Torch device ('cuda' or 'cpu')
            cfg: Full Hydra configuration (expects cfg.model for model settings)
            model_cfg: Model configuration (Hydra DictConfig or dict)
        """
        self.device = device
        self.model_type = model_type.lower()
        self.cfg = cfg
        if cfg is not None:
            self.model_cfg = cfg.model
        else:
            self.model_cfg = model_cfg or {}

        if isinstance(self.model_cfg, DictConfig):
            # Resolve to plain container for consistent access regardless of Hydra internals
            self.model_cfg = OmegaConf.to_container(self.model_cfg, resolve=True)

        self.fitted = False
        self.optimizer_type = str(self._cfg_get('optimizer', 'adamw')).lower()
        self.vae_beta_max = float(self._cfg_get('vae_beta_max', 0.001))
        warmup_steps = self._cfg_get('vae_beta_warmup_steps', 1000)
        self.vae_beta_warmup_steps = int(warmup_steps) if warmup_steps else 0
        if self.vae_beta_warmup_steps <= 0:
            self.vae_beta_warmup_steps = 1

        # Create model
        hidden_dim = self._cfg_get('hidden_dim', 128)
        latent_dim = self._cfg_get('latent_dim', 64)
        num_layers = self._cfg_get('num_layers', 2)
        num_heads = self._cfg_get('num_heads', 4)
        tcn_kernel_size = self._cfg_get('tcn_kernel_size', 3)
        tcn_dilation_base = self._cfg_get('tcn_dilation_base', 2)
        tcn_dropout = self._cfg_get('tcn_dropout', 0.1)

        self.model = create_model(
            model_type=self.model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilation_base=tcn_dilation_base,
            tcn_dropout=tcn_dropout,
        )
        self.model.to(device)

        # Create optimizer
        lr = self._cfg_get('learning_rate', 0.001)
        weight_decay = self._cfg_get('weight_decay', 0.0001)
        if self.optimizer_type != 'adamw':
            print(f"  ⚠ Optimizer '{self.optimizer_type}' specified but using AdamW")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"  ✓ ModelTrainer initialized")
        print(f"    Model type: {self.model_type}")
        print(f"    Input dim: {input_dim}")
        print(f"    Hidden dim: {hidden_dim}")
        if self.model_type in ['autoencoder', 'vae', 'tcn_autoencoder', 'tcn_vae']:
            print(f"    Latent dim: {latent_dim}")
        if self.model_type in ['vae', 'tcn_vae']:
            print(f"    VAE beta max: {self.vae_beta_max}")
            print(f"    VAE beta warmup steps: {self.vae_beta_warmup_steps}")
        if self.model_type in ['tcn_autoencoder', 'tcn_vae']:
            print(f"    TCN kernel size: {tcn_kernel_size}")
            print(f"    TCN dilation base: {tcn_dilation_base}")
            print(f"    TCN dropout: {tcn_dropout}")
        print(f"    Num layers: {num_layers}")
        if self.model_type == 'transformer':
            print(f"    Num heads: {num_heads}")
        print(f"    Learning rate: {lr}")
        print(f"    Optimizer: AdamW")

    def _cfg_get(self, key, default):
        """Safely access configuration values"""
        cfg = self.model_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _compute_beta(self, step):
        """Compute annealed beta weight for VAE KL term"""
        if self.model_type not in ['vae', 'tcn_vae']:
            return None
        if step is None:
            return self.vae_beta_max
        progress = min(1.0, max(step, 0) / float(self.vae_beta_warmup_steps))
        return self.vae_beta_max * progress

    def train_on_batch(self, embedding_batch, global_step=None):
        """
        Train model on a single batch of embeddings (memory-efficient)

        Args:
            embedding_batch: [B, H, W, L, D] tensor on GPU
            global_step: Optional 1-based global step for scheduling

        Returns:
            tuple: (loss_value, metrics dict)
        """
        _, _, _, L, D = embedding_batch.shape

        # Flatten spatial dimensions: [B*H*W, L, D]
        trajectories = embedding_batch.reshape(-1, L, D)

        # Forward pass
        self.optimizer.zero_grad()

        metrics = {}

        if self.model_type in ['vae', 'tcn_vae']:
            beta = self._compute_beta(global_step)
            reconstruction, mu, logvar = self.model(trajectories)

            # VAE loss: reconstruction + KL divergence
            recon_loss = nn.functional.mse_loss(reconstruction, trajectories)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            metrics.update({
                'recon_loss': recon_loss.detach().item(),
                'kl_loss': kl_loss.detach().item(),
                'beta': beta,
            })
        else:
            reconstruction, _ = self.model(trajectories)
            loss = nn.functional.mse_loss(reconstruction, trajectories)
            metrics['recon_loss'] = loss.detach().item()

        # Backward pass
        loss.backward()
        self.optimizer.step()

        metrics['loss'] = loss.detach().item()
        return metrics['loss'], metrics

    def train_streaming(self, dataloader, extractor, num_epochs=10, use_wandb=False,
                       wandb_config=None):
        """
        Train model in streaming fashion without accumulating all embeddings

        Args:
            dataloader: DataLoader yielding image batches
            extractor: ActivationExtractor for feature extraction
            num_epochs: Number of training epochs
            use_wandb: Whether to log to Weights & Biases
            wandb_config: Optional dict with wandb settings (project, entity, name)

        Returns:
            self: For method chaining
        """
        from trajectory import stack_trajectory
        from tqdm import tqdm

        print(f"\n  [Phase 1: Model Training (Streaming)]")
        print(f"  Training {self.model_type} in streaming mode...")
        print(f"  Epochs: {num_epochs}")

        if use_wandb:
            try:
                import wandb
                wandb_kwargs = {
                    'project': wandb_config.get('project', 'patch-detection') if wandb_config else 'patch-detection',
                    'name': wandb_config.get('name', f'train-{self.model_type}') if wandb_config else f'train-{self.model_type}',
                    'config': {
                        'model_type': self.model_type,
                        'num_epochs': num_epochs,
                        'hidden_dim': self._cfg_get('hidden_dim', 128),
                        'latent_dim': self._cfg_get('latent_dim', 64),
                    }
                }
                if wandb_config and wandb_config.get('entity'):
                    wandb_kwargs['entity'] = wandb_config['entity']
                wandb.init(**wandb_kwargs)
            except ImportError:
                print("  Warning: wandb not installed, skipping logging")
                use_wandb = False

        self.model.train()

        global_step = 0  # Global step counter for scheduling/logging

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            # Progress bar for batches
            pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False)

            for imgs, _ in pbar:
                imgs_gpu = imgs.to(self.device, non_blocking=True)

                # Extract activations
                with torch.no_grad():
                    activations = extractor(imgs_gpu)
                    embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, D]

                # Train on this batch
                step_id = global_step + 1
                loss, metrics = self.train_on_batch(embeddings_batch, global_step=step_id)

                epoch_loss += loss
                epoch_recon += metrics.get('recon_loss', 0.0)
                if 'kl_loss' in metrics:
                    epoch_kl += metrics['kl_loss']
                num_batches += 1
                global_step = step_id

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}'})

                # Log to wandb (per batch)
                if use_wandb:
                    log_data = {
                        "batch_loss": loss,
                        "batch_recon_loss": metrics.get('recon_loss', loss),
                        "global_step": global_step,
                        "epoch": epoch + 1,
                    }
                    if 'kl_loss' in metrics:
                        log_data["batch_kl_loss"] = metrics['kl_loss']
                        log_data["vae_beta"] = metrics.get('beta', self.vae_beta_max)
                    wandb.log(log_data)

                # CRITICAL: Delete to free GPU memory immediately
                del imgs_gpu, activations, embeddings_batch
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")

            # Log epoch average to wandb
            if use_wandb:
                log_data = {
                    "epoch": epoch + 1,
                    "epoch_loss": avg_loss,
                    "epoch_recon_loss": avg_recon,
                }
                if self.model_type in ['vae', 'tcn_vae'] and num_batches > 0:
                    log_data["epoch_kl_loss"] = epoch_kl / num_batches
                wandb.log(log_data)

        self.fitted = True
        print(f"  ✓ Model training completed")

        if use_wandb:
            wandb.finish()

        return self

    def train(self, train_embeddings_gpu, num_epochs=10, batch_size=128):
        """
        Train model on pre-collected embeddings (legacy method, kept for compatibility)

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
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            # Shuffle data
            perm = torch.randperm(train_data.shape[0], device=self.device)
            shuffled_data = train_data[perm]

            # Mini-batch training
            for i in range(0, train_data.shape[0], batch_size):
                batch = shuffled_data[i:i+batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                global_step += 1

                if self.model_type in ['vae', 'tcn_vae']:
                    reconstruction, mu, logvar = self.model(batch)

                    # VAE loss: reconstruction + KL divergence
                    recon_loss = nn.functional.mse_loss(reconstruction, batch)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    beta = self._compute_beta(global_step)
                    loss = recon_loss + beta * kl_loss  # Annealed beta
                    epoch_recon += recon_loss.detach().item()
                    epoch_kl += kl_loss.detach().item()
                else:
                    reconstruction, _ = self.model(batch)
                    loss = nn.functional.mse_loss(reconstruction, batch)
                    epoch_recon += loss.detach().item()

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
            if self.model_type in ['vae', 'tcn_vae'] and num_batches > 0:
                print(f"      Recon: {epoch_recon / num_batches:.6f}, KL: {epoch_kl / num_batches:.6f}")

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
            target_modules = lora_cfg.get('target_modules', ['Linear', 'Conv1d'])
            target_name_keywords = lora_cfg.get('target_name_keywords', ['decoder'])
            lora_optimizer_type = str(lora_cfg.get('optimizer', 'adamw')).lower()
            lr = lora_cfg.get('learning_rate', 0.0001)
            weight_decay = lora_cfg.get('weight_decay', 0.0001)
        else:
            rank = getattr(lora_cfg, 'rank', 8)
            alpha = getattr(lora_cfg, 'alpha', 16)
            target_modules = getattr(lora_cfg, 'target_modules', ['Linear', 'Conv1d'])
            target_name_keywords = getattr(lora_cfg, 'target_name_keywords', ['decoder'])
            lora_optimizer_type = getattr(lora_cfg, 'optimizer', 'adamw')
            lr = getattr(lora_cfg, 'learning_rate', 0.0001) if hasattr(lora_cfg, 'learning_rate') else 0.0001
            weight_decay = getattr(lora_cfg, 'weight_decay', 0.0001) if hasattr(lora_cfg, 'weight_decay') else 0.0001

        # Fallback to parent domain_adaptation defaults if still None
        parent_cfg = getattr(self.cfg, 'domain_adaptation', None) if hasattr(self, 'cfg') else None
        if (lr is None or weight_decay is None) and parent_cfg is not None:
            parent_lr = getattr(parent_cfg, 'learning_rate', None)
            parent_wd = getattr(parent_cfg, 'weight_decay', None)
            lr = lr if lr is not None else parent_lr if parent_lr is not None else 0.0001
            weight_decay = weight_decay if weight_decay is not None else parent_wd if parent_wd is not None else 0.0001

        lora_optimizer_type = str(lora_optimizer_type).lower()

        if 'Conv1d' not in target_modules and self.model_type in ['tcn_autoencoder', 'tcn_vae']:
            target_modules = list(target_modules) + ['Conv1d']

        print(f"    LoRA rank: {rank}")
        print(f"    LoRA alpha: {alpha}")
        print(f"    Target modules: {target_modules}")
        print(f"    Target name keywords: {target_name_keywords}")

        # Apply LoRA to model
        self.model, lora_params = apply_lora_to_model(
            self.model, rank, alpha, target_modules, target_name_keywords
        )

        # Create optimizer for LoRA parameters only
        if lora_optimizer_type != 'adamw':
            print(f"    ⚠ LoRA optimizer '{lora_optimizer_type}' specified but using AdamW")

        lora_optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"    LoRA learning rate: {lr}")
        print(f"    LoRA optimizer: AdamW")
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
        global_step = 0  # Track batches for beta scheduling
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            # Shuffle data
            perm = torch.randperm(domain_data.shape[0], device=self.device)
            shuffled_data = domain_data[perm]

            # Mini-batch training
            for i in range(0, domain_data.shape[0], batch_size):
                batch = shuffled_data[i:i+batch_size]

                # Forward pass
                lora_optimizer.zero_grad()
                global_step += 1

                if self.model_type in ['vae', 'tcn_vae']:
                    reconstruction, mu, logvar = self.model(batch)
                    recon_loss = nn.functional.mse_loss(reconstruction, batch)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    beta = self._compute_beta(global_step)
                    loss = recon_loss + beta * kl_loss
                    epoch_recon += recon_loss.detach().item()
                    epoch_kl += kl_loss.detach().item()
                else:
                    reconstruction, _ = self.model(batch)
                    loss = nn.functional.mse_loss(reconstruction, batch)
                    epoch_recon += loss.detach().item()

                # Backward pass (only LoRA parameters updated)
                loss.backward()
                lora_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
            if self.model_type in ['vae', 'tcn_vae'] and num_batches > 0:
                print(f"      Recon: {epoch_recon / num_batches:.6f}, KL: {epoch_kl / num_batches:.6f}")

        print(f"  ✓ LoRA domain adaptation completed")

        return self

    def adapt_with_lora_streaming(self, dataloader, extractor, lora_cfg, num_epochs=5,
                                 use_wandb=False, wandb_config=None):
        """
        Adapt model with LoRA in streaming fashion (memory-efficient)

        Args:
            dataloader: DataLoader yielding image batches
            extractor: ActivationExtractor for feature extraction
            lora_cfg: LoRA configuration (Hydra DictConfig or dict)
            num_epochs: Number of adaptation epochs
            use_wandb: Whether to log to Weights & Biases
            wandb_config: Optional dict with wandb settings (project, entity, name)

        Returns:
            self: For method chaining
        """
        from trajectory import stack_trajectory
        from tqdm import tqdm

        print(f"\n  [Phase 2: LoRA Domain Adaptation (Streaming)]")
        print(f"  Adapting model in streaming mode...")

        # Extract LoRA config
        if isinstance(lora_cfg, dict):
            rank = lora_cfg.get('rank', 8)
            alpha = lora_cfg.get('alpha', 16)
            target_modules = lora_cfg.get('target_modules', ['Linear', 'Conv1d'])
            lr = lora_cfg.get('learning_rate', 0.0001)
            weight_decay = lora_cfg.get('weight_decay', 0.0001)
            target_name_keywords = lora_cfg.get('target_name_keywords', ['decoder'])
            lora_optimizer_type = str(lora_cfg.get('optimizer', 'adamw')).lower()
        else:
            rank = getattr(lora_cfg, 'rank', 8)
            alpha = getattr(lora_cfg, 'alpha', 16)
            target_modules = getattr(lora_cfg, 'target_modules', ['Linear', 'Conv1d'])
            lr = getattr(lora_cfg, 'learning_rate', 0.0001) if hasattr(lora_cfg, 'learning_rate') else 0.0001
            weight_decay = getattr(lora_cfg, 'weight_decay', 0.0001) if hasattr(lora_cfg, 'weight_decay') else 0.0001
            target_name_keywords = getattr(lora_cfg, 'target_name_keywords', ['decoder'])
            lora_optimizer_type = getattr(lora_cfg, 'optimizer', 'adamw')

        lora_optimizer_type = str(lora_optimizer_type).lower()

        if 'Conv1d' not in target_modules and self.model_type in ['tcn_autoencoder', 'tcn_vae']:
            target_modules = list(target_modules) + ['Conv1d']

        print(f"    LoRA rank: {rank}")
        print(f"    LoRA alpha: {alpha}")
        print(f"    Target modules: {target_modules}")
        print(f"    Target name keywords: {target_name_keywords}")

        # Apply LoRA to model
        self.model, lora_params = apply_lora_to_model(
            self.model,
            rank,
            alpha,
            target_modules,
            target_name_keywords,
        )

        # Create optimizer for LoRA parameters only
        if lora_optimizer_type != 'adamw':
            print(f"    ⚠ LoRA optimizer '{lora_optimizer_type}' specified but using AdamW")

        lora_optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay
        )

        print(f"    LoRA learning rate: {lr}")
        print(f"    LoRA optimizer: AdamW")
        print(f"    Epochs: {num_epochs}")

        if use_wandb:
            try:
                import wandb
                wandb_kwargs = {
                    'project': wandb_config.get('project', 'patch-detection') if wandb_config else 'patch-detection',
                    'name': wandb_config.get('name', f'lora-{self.model_type}') if wandb_config else f'lora-{self.model_type}',
                    'reinit': True,
                    'config': {
                        'model_type': self.model_type,
                        'num_epochs': num_epochs,
                        'lora_rank': rank,
                        'lora_alpha': alpha,
                        'lora_lr': lr,
                    }
                }
                if wandb_config and wandb_config.get('entity'):
                    wandb_kwargs['entity'] = wandb_config['entity']
                wandb.init(**wandb_kwargs)
            except ImportError:
                print("  Warning: wandb not installed, skipping logging")
                use_wandb = False

        # LoRA training loop
        self.model.train()
        global_step = 0  # Track batches for wandb logging during adaptation
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            # Progress bar for batches
            pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False)

            for imgs, _ in pbar:
                imgs_gpu = imgs.to(self.device, non_blocking=True)

                # Extract activations
                with torch.no_grad():
                    activations = extractor(imgs_gpu)
                    embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, D]

                _, _, _, L, D = embeddings_batch.shape
                trajectories = embeddings_batch.reshape(-1, L, D)

                # Forward pass
                lora_optimizer.zero_grad()
                step_id = global_step + 1

                batch_recon_value = None
                if self.model_type in ['vae', 'tcn_vae']:
                    reconstruction, mu, logvar = self.model(trajectories)
                    recon_loss = nn.functional.mse_loss(reconstruction, trajectories)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    beta = self._compute_beta(step_id)
                    loss = recon_loss + beta * kl_loss
                    batch_recon_value = recon_loss.detach().item()
                    epoch_recon += recon_loss.detach().item()
                    epoch_kl += kl_loss.detach().item()
                else:
                    reconstruction, _ = self.model(trajectories)
                    loss = nn.functional.mse_loss(reconstruction, trajectories)
                    batch_recon_value = loss.detach().item()
                    epoch_recon += batch_recon_value

                # Backward pass
                loss.backward()
                lora_optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                num_batches += 1
                global_step = step_id

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})

                # Log per-batch loss to wandb
                if use_wandb:
                    log_data = {
                        "lora_batch_loss": loss_val,
                        "lora_batch_recon_loss": batch_recon_value,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    }
                if self.model_type in ['vae', 'tcn_vae']:
                    log_data["lora_batch_kl_loss"] = kl_loss.detach().item()
                    log_data["vae_beta"] = beta
                    wandb.log(log_data)

                # CRITICAL: Delete to free GPU memory
                del imgs_gpu, activations, embeddings_batch, trajectories, reconstruction, loss
                if self.model_type in ['vae', 'tcn_vae']:
                    del mu, logvar, recon_loss, kl_loss
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
            avg_recon = epoch_recon / num_batches if num_batches > 0 else 0.0
            if self.model_type in ['vae', 'tcn_vae'] and num_batches > 0:
                print(f"      Recon: {avg_recon:.6f}, KL: {epoch_kl / num_batches:.6f}")

            # Log to wandb
            if use_wandb:
                log_data = {
                    "epoch": epoch + 1,
                    "lora_loss": avg_loss,
                    "lora_epoch_recon_loss": avg_recon,
                }
                if self.model_type in ['vae', 'tcn_vae'] and num_batches > 0:
                    log_data["lora_epoch_kl_loss"] = epoch_kl / num_batches
                wandb.log(log_data)

        print(f"  ✓ LoRA domain adaptation completed")

        if use_wandb:
            wandb.finish()

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
