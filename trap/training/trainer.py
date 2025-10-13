import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from trap.features.trajectory import stack_trajectory
from trap.models.factory import apply_lora_to_model, create_model


class ModelTrainer:
    """Trainer for time-series anomaly detection models."""

    def __init__(self, model_type, input_dim, device: str = "cuda", cfg=None, model_cfg=None):
        self.device = device
        self.model_type = model_type.lower()
        self.cfg = cfg
        if cfg is not None:
            self.model_cfg = cfg.model
        else:
            self.model_cfg = model_cfg or {}

        if isinstance(self.model_cfg, DictConfig):
            self.model_cfg = OmegaConf.to_container(self.model_cfg, resolve=True)

        self.fitted = False
        self.optimizer_type = str(self._cfg_get("optimizer", "adamw")).lower()
        self.vae_beta_max = float(self._cfg_get("vae_beta_max", 0.001))
        warmup_steps = self._cfg_get("vae_beta_warmup_steps", 1000)
        self.vae_beta_warmup_steps = int(warmup_steps) if warmup_steps else 0
        if self.vae_beta_warmup_steps <= 0:
            self.vae_beta_warmup_steps = 1

        hidden_dim = self._cfg_get("hidden_dim", 128)
        latent_dim = self._cfg_get("latent_dim", 64)
        num_layers = self._cfg_get("num_layers", 2)
        num_heads = self._cfg_get("num_heads", 4)
        dropout = self._cfg_get("dropout", 0.1)
        tcn_kernel_size = self._cfg_get("tcn_kernel_size", 3)
        tcn_dilation_base = self._cfg_get("tcn_dilation_base", 2)
        tcn_dropout = self._cfg_get("tcn_dropout", 0.1)

        self.model = create_model(
            model_type=self.model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dilation_base=tcn_dilation_base,
            tcn_dropout=tcn_dropout,
        )
        self.model.to(device)

        lr = self._cfg_get("learning_rate", 0.001)
        weight_decay = self._cfg_get("weight_decay", 0.0001)
        if self.optimizer_type != "adamw":
            print(f"  ⚠ Optimizer '{self.optimizer_type}' specified but using AdamW")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        print(f"  ✓ ModelTrainer initialized")
        print(f"    Model type: {self.model_type}")
        print(f"    Input dim: {input_dim}")
        print(f"    Hidden dim: {hidden_dim}")
        if self.model_type in ["autoencoder", "vae", "tcn_autoencoder", "tcn_vae"]:
            print(f"    Latent dim: {latent_dim}")
        if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
            print(f"    VAE beta max: {self.vae_beta_max}")
            print(f"    VAE beta warmup steps: {self.vae_beta_warmup_steps}")
        if self.model_type in ["tcn_autoencoder", "tcn_vae"]:
            print(f"    TCN kernel size: {tcn_kernel_size}")
            print(f"    TCN dilation base: {tcn_dilation_base}")
            print(f"    TCN dropout: {tcn_dropout}")
        print(f"    Num layers: {num_layers}")
        if self.model_type == "transformer":
            print(f"    Num heads: {num_heads}")
        print(f"    Learning rate: {lr}")
        print(f"    Optimizer: AdamW")

    def _cfg_get(self, key, default):
        cfg = self.model_cfg
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _compute_beta(self, step: int):
        if self.model_type not in ["vae", "tcn_vae", "transformer_vae"]:
            return None
        if step is None:
            return self.vae_beta_max
        progress = min(1.0, max(step, 0) / float(self.vae_beta_warmup_steps))
        return self.vae_beta_max * progress

    def train_on_batch(self, embedding_batch: Tensor, global_step: int = None):
        _, _, _, L, D = embedding_batch.shape
        trajectories = embedding_batch.reshape(-1, L, D)

        self.optimizer.zero_grad()

        metrics: Dict[str, Any] = {}

        if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
            beta = self._compute_beta(global_step)
            reconstruction, mu, logvar = self.model(trajectories)

            recon_loss = nn.functional.mse_loss(reconstruction, trajectories)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            metrics.update(
                {
                    "recon_loss": recon_loss.detach().item(),
                    "kl_loss": kl_loss.detach().item(),
                    "beta": beta,
                }
            )
        else:
            reconstruction, _ = self.model(trajectories)
            loss = nn.functional.mse_loss(reconstruction, trajectories)
            metrics["recon_loss"] = loss.detach().item()

        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.detach().item()
        return metrics["loss"], metrics

    def train_streaming(
        self,
        dataloader: DataLoader,
        extractor,
        num_epochs: int = 10,
        use_wandb: bool = False,
        wandb_config: Dict[str, Any] | None = None,
    ):
        print(f"\n  [Phase 1: Model Training (Streaming)]")
        print(f"  Training {self.model_type} in streaming mode...")
        print(f"  Epochs: {num_epochs}")

        if use_wandb:
            try:
                import wandb

                wandb_kwargs = {
                    "project": wandb_config.get("project", "patch-detection") if wandb_config else "patch-detection",
                    "name": wandb_config.get("name", f"train-{self.model_type}")
                    if wandb_config
                    else f"train-{self.model_type}",
                    "config": {
                        "model_type": self.model_type,
                        "num_epochs": num_epochs,
                        "hidden_dim": self._cfg_get("hidden_dim", 128),
                        "latent_dim": self._cfg_get("latent_dim", 64),
                    },
                }
                if wandb_config and wandb_config.get("entity"):
                    wandb_kwargs["entity"] = wandb_config["entity"]
                wandb.init(**wandb_kwargs)
            except ImportError:
                print("  Warning: wandb not installed, skipping logging")
                use_wandb = False

        self.model.train()
        global_step = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{num_epochs}", leave=False)

            for imgs, _ in pbar:
                imgs_gpu = imgs.to(self.device, non_blocking=True)

                with torch.no_grad():
                    activations = extractor(imgs_gpu)
                    embeddings_batch = stack_trajectory(activations)

                step_id = global_step + 1
                loss, metrics = self.train_on_batch(embeddings_batch, global_step=step_id)

                epoch_loss += loss
                epoch_recon += metrics.get("recon_loss", 0.0)
                if "kl_loss" in metrics:
                    epoch_kl += metrics["kl_loss"]
                num_batches += 1
                global_step = step_id

                pbar.set_postfix({"loss": f"{loss:.4f}"})

                if use_wandb:
                    import wandb

                    log_data = {
                        "batch_loss": loss,
                        "batch_recon_loss": metrics.get("recon_loss", loss),
                        "global_step": global_step,
                        "epoch": epoch + 1,
                    }
                    if "kl_loss" in metrics:
                        log_data["batch_kl_loss"] = metrics["kl_loss"]
                        log_data["vae_beta"] = metrics.get("beta", self.vae_beta_max)
                    wandb.log(log_data)

                del imgs_gpu, activations, embeddings_batch
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / num_batches
            avg_recon = epoch_recon / num_batches
            print(f"    Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")

            if use_wandb:
                import wandb

                log_data = {
                    "epoch": epoch + 1,
                    "epoch_loss": avg_loss,
                    "epoch_recon_loss": avg_recon,
                }
                if self.model_type in ["vae", "tcn_vae", "transformer_vae"] and num_batches > 0:
                    log_data["epoch_kl_loss"] = epoch_kl / num_batches
                wandb.log(log_data)

        self.fitted = True
        print(f"  ✓ Model training completed")

        if use_wandb:
            import wandb

            wandb.finish()

        return self

    def train(self, train_embeddings_gpu: Iterable[Tensor], num_epochs: int = 10, batch_size: int = 128):
        print(f"\n  [Phase 1: Model Training]")
        print(f"  Training {self.model_type} on {len(train_embeddings_gpu)} clean images...")
        print(f"  Epochs: {num_epochs}, Batch size: {batch_size}")

        all_trajectories = []
        for emb in train_embeddings_gpu:
            H, W, L, D = emb.shape
            trajectories = emb.reshape(-1, L, D)
            all_trajectories.append(trajectories)

        train_data = torch.cat(all_trajectories, dim=0)
        print(f"    Total trajectories: {train_data.shape[0]}")

        self.model.train()
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            perm = torch.randperm(train_data.shape[0], device=self.device)
            shuffled_data = train_data[perm]

            for i in range(0, train_data.shape[0], batch_size):
                batch = shuffled_data[i:i + batch_size]

                self.optimizer.zero_grad()
                global_step += 1

                if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
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

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
            if self.model_type in ["vae", "tcn_vae", "transformer_vae"] and num_batches > 0:
                print(f"      Recon: {epoch_recon / num_batches:.6f}, KL: {epoch_kl / num_batches:.6f}")

        self.fitted = True
        print(f"  ✓ Model training completed")

        return self

    def save_weights(self, weights_path):
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_type": self.model_type,
            "fitted": self.fitted,
        }

        torch.save(state, weights_path)
        print(f"  ✓ Model weights saved to: {weights_path}")

    def load_weights(self, weights_path):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        state = torch.load(weights_path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.model_type = state["model_type"]
        self.fitted = state["fitted"]

        print(f"  ✓ Model weights loaded from: {weights_path}")

        return self

    def adapt_with_lora(self, domain_embeddings_gpu: Iterable[Tensor], lora_cfg, num_epochs: int = 5,
                        batch_size: int = 32):
        print(f"\n  [Phase 2: LoRA Domain Adaptation]")
        print(f"  Adapting model on {len(domain_embeddings_gpu)} clean images...")
        print(f"  Epochs: {num_epochs}, Batch size: {batch_size}")

        if isinstance(lora_cfg, dict):
            rank = lora_cfg.get("rank", 8)
            alpha = lora_cfg.get("alpha", 16)
            target_modules = lora_cfg.get("target_modules", ["Linear", "Conv1d"])
            lr = lora_cfg.get("learning_rate", 0.0001)
            weight_decay = lora_cfg.get("weight_decay", 0.0001)
            target_name_keywords = lora_cfg.get("target_name_keywords", ["decoder"])
            lora_optimizer_type = str(lora_cfg.get("optimizer", "adamw")).lower()
        else:
            rank = getattr(lora_cfg, "rank", 8)
            alpha = getattr(lora_cfg, "alpha", 16)
            target_modules = getattr(lora_cfg, "target_modules", ["Linear", "Conv1d"])
            lr = getattr(lora_cfg, "learning_rate", 0.0001) if hasattr(lora_cfg, "learning_rate") else 0.0001
            weight_decay = getattr(lora_cfg, "weight_decay", 0.0001) if hasattr(lora_cfg, "weight_decay") else 0.0001
            target_name_keywords = getattr(lora_cfg, "target_name_keywords", ["decoder"])
            lora_optimizer_type = str(getattr(lora_cfg, "optimizer", "adamw")).lower()

        if "Conv1d" not in target_modules and self.model_type in ["tcn_autoencoder", "tcn_vae"]:
            target_modules = list(target_modules) + ["Conv1d"]

        print(f"    LoRA rank: {rank}")
        print(f"    LoRA alpha: {alpha}")
        print(f"    Target modules: {target_modules}")
        print(f"    Target name keywords: {target_name_keywords}")

        self.model, lora_params = apply_lora_to_model(
            self.model,
            rank,
            alpha,
            target_modules,
            target_name_keywords,
        )

        if lora_optimizer_type != "adamw":
            print(f"    ⚠ LoRA optimizer '{lora_optimizer_type}' specified but using AdamW")

        lora_optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

        print(f"    LoRA learning rate: {lr}")
        print(f"    LoRA optimizer: AdamW")

        all_trajectories = []
        for emb in domain_embeddings_gpu:
            H, W, L, D = emb.shape
            trajectories = emb.reshape(-1, L, D)
            all_trajectories.append(trajectories)

        domain_data = torch.cat(all_trajectories, dim=0)
        print(f"    Total domain trajectories: {domain_data.shape[0]}")

        self.model.train()
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            perm = torch.randperm(domain_data.shape[0], device=self.device)
            shuffled_data = domain_data[perm]

            for i in range(0, domain_data.shape[0], batch_size):
                batch = shuffled_data[i:i + batch_size]

                lora_optimizer.zero_grad()
                global_step += 1

                if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
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

                loss.backward()
                lora_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
            if self.model_type in ["vae", "tcn_vae", "transformer_vae"] and num_batches > 0:
                print(f"      Recon: {epoch_recon / num_batches:.6f}, KL: {epoch_kl / num_batches:.6f}")

        print(f"  ✓ LoRA domain adaptation completed")

        return self

    def adapt_with_lora_streaming(
        self,
        dataloader: DataLoader,
        extractor,
        lora_cfg,
        num_epochs: int = 5,
        use_wandb: bool = False,
        wandb_config: Dict[str, Any] | None = None,
    ):
        print(f"\n  [Phase 2: LoRA Domain Adaptation (Streaming)]")
        print(f"  Adapting model in streaming mode...")

        if isinstance(lora_cfg, dict):
            rank = lora_cfg.get("rank", 8)
            alpha = lora_cfg.get("alpha", 16)
            target_modules = lora_cfg.get("target_modules", ["Linear", "Conv1d"])
            lr = lora_cfg.get("learning_rate", 0.0001)
            weight_decay = lora_cfg.get("weight_decay", 0.0001)
            target_name_keywords = lora_cfg.get("target_name_keywords", ["decoder"])
            lora_optimizer_type = str(lora_cfg.get("optimizer", "adamw")).lower()
        else:
            rank = getattr(lora_cfg, "rank", 8)
            alpha = getattr(lora_cfg, "alpha", 16)
            target_modules = getattr(lora_cfg, "target_modules", ["Linear", "Conv1d"])
            lr = getattr(lora_cfg, "learning_rate", 0.0001) if hasattr(lora_cfg, "learning_rate") else 0.0001
            weight_decay = getattr(lora_cfg, "weight_decay", 0.0001) if hasattr(lora_cfg, "weight_decay") else 0.0001
            target_name_keywords = getattr(lora_cfg, "target_name_keywords", ["decoder"])
            lora_optimizer_type = str(getattr(lora_cfg, "optimizer", "adamw")).lower()

        if "Conv1d" not in target_modules and self.model_type in ["tcn_autoencoder", "tcn_vae"]:
            target_modules = list(target_modules) + ["Conv1d"]

        print(f"    LoRA rank: {rank}")
        print(f"    LoRA alpha: {alpha}")
        print(f"    Target modules: {target_modules}")
        print(f"    Target name keywords: {target_name_keywords}")

        self.model, lora_params = apply_lora_to_model(
            self.model,
            rank,
            alpha,
            target_modules,
            target_name_keywords,
        )

        if lora_optimizer_type != "adamw":
            print(f"    ⚠ LoRA optimizer '{lora_optimizer_type}' specified but using AdamW")

        lora_optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

        print(f"    LoRA learning rate: {lr}")
        print(f"    LoRA optimizer: AdamW")
        print(f"    Epochs: {num_epochs}")

        if use_wandb:
            try:
                import wandb

                wandb_kwargs = {
                    "project": wandb_config.get("project", "patch-detection") if wandb_config else "patch-detection",
                    "name": wandb_config.get("name", f"lora-{self.model_type}")
                    if wandb_config
                    else f"lora-{self.model_type}",
                    "reinit": True,
                    "config": {
                        "model_type": self.model_type,
                        "num_epochs": num_epochs,
                        "lora_rank": rank,
                        "lora_alpha": alpha,
                        "lora_lr": lr,
                    },
                }
                if wandb_config and wandb_config.get("entity"):
                    wandb_kwargs["entity"] = wandb_config["entity"]
                wandb.init(**wandb_kwargs)
            except ImportError:
                print("  Warning: wandb not installed, skipping logging")
                use_wandb = False

        self.model.train()
        global_step = 0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            num_batches = 0

            pbar = tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{num_epochs}", leave=False)

            for imgs, _ in pbar:
                imgs_gpu = imgs.to(self.device, non_blocking=True)

                with torch.no_grad():
                    activations = extractor(imgs_gpu)
                    embeddings_batch = stack_trajectory(activations)

                _, _, _, L, D = embeddings_batch.shape
                trajectories = embeddings_batch.reshape(-1, L, D)

                lora_optimizer.zero_grad()
                step_id = global_step + 1

                batch_recon_value = None
                if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
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

                loss.backward()
                lora_optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                num_batches += 1
                global_step = step_id

                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                if use_wandb:
                    import wandb

                    log_data = {
                        "lora_batch_loss": loss_val,
                        "lora_batch_recon_loss": batch_recon_value,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    }
                    if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
                        log_data["lora_batch_kl_loss"] = kl_loss.detach().item()
                        log_data["vae_beta"] = beta
                    wandb.log(log_data)

                del imgs_gpu, activations, embeddings_batch, trajectories, reconstruction, loss
                if self.model_type in ["vae", "tcn_vae", "transformer_vae"]:
                    del mu, logvar, recon_loss, kl_loss
                torch.cuda.empty_cache()

            avg_loss = epoch_loss / num_batches
            print(f"    Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
            avg_recon = epoch_recon / num_batches if num_batches > 0 else 0.0
            if self.model_type in ["vae", "tcn_vae", "transformer_vae"] and num_batches > 0:
                print(f"      Recon: {avg_recon:.6f}, KL: {epoch_kl / num_batches:.6f}")

            if use_wandb:
                import wandb

                log_data = {
                    "epoch": epoch + 1,
                    "lora_loss": avg_loss,
                    "lora_epoch_recon_loss": avg_recon,
                }
                if self.model_type in ["vae", "tcn_vae", "transformer_vae"] and num_batches > 0:
                    log_data["lora_epoch_kl_loss"] = epoch_kl / num_batches
                wandb.log(log_data)

        print(f"  ✓ LoRA domain adaptation completed")

        if use_wandb:
            import wandb

            wandb.finish()

        return self

    def save_lora_weights(self, lora_weights_path):
        lora_weights_path = Path(lora_weights_path)
        lora_weights_path.parent.mkdir(parents=True, exist_ok=True)

        lora_state = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "lora"):
                lora_state[name] = module.lora.state_dict()

        torch.save(lora_state, lora_weights_path)
        print(f"  ✓ LoRA weights saved to: {lora_weights_path}")

    def load_lora_weights(self, lora_weights_path):
        lora_weights_path = Path(lora_weights_path)
        if not lora_weights_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {lora_weights_path}")

        lora_state = torch.load(lora_weights_path, map_location=self.device)

        for name, module in self.model.named_modules():
            if hasattr(module, "lora") and name in lora_state:
                module.lora.load_state_dict(lora_state[name])

        print(f"  ✓ LoRA weights loaded from: {lora_weights_path}")

        return self

    @staticmethod
    def get_weights_filename(model_type, imagenet_path, num_samples, spatial_resolution,
                             feature_dim, hidden_dim, latent_dim, num_layers):
        config_str = (
            f"{imagenet_path}_{num_samples}_{spatial_resolution}_{feature_dim}_"
            f"{hidden_dim}_{latent_dim}_{num_layers}"
        )
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{model_type}_s{num_samples}_r{spatial_resolution}_d{feature_dim}_{config_hash}.pt"

    @staticmethod
    def get_lora_weights_filename(model_type, domain_path, num_samples, rank, alpha):
        config_str = f"{model_type}_{domain_path}_{num_samples}_{rank}_{alpha}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"lora_{model_type}_r{rank}_a{alpha}_{config_hash}.pt"
