"""
Time-series Anomaly Detection Models for Patch Detection

This module implements three neural network models for time-series anomaly detection:
1. Autoencoder - Reconstruction-based
2. VAE - Variational Autoencoder with probabilistic modeling
3. Transformer - Attention-based temporal modeling

All models are trained on clean images only (Phase 1) and can be optionally
fine-tuned with LoRA for domain adaptation (Phase 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeSeriesAutoencoder(nn.Module):
    """
    Autoencoder for time-series trajectory anomaly detection

    Architecture:
        - Encoder: LSTM layers that compress trajectory to latent representation
        - Decoder: LSTM layers that reconstruct trajectory from latent

    Anomaly detection:
        - High reconstruction error indicates anomaly
        - Trained only on clean trajectories (Phase 1)
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_layers=2):
        """
        Initialize Autoencoder

        Args:
            input_dim: Feature dimension (D)
            hidden_dim: Hidden dimension for LSTM
            latent_dim: Latent representation dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: trajectory -> latent
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> trajectory
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encode trajectory to latent representation

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            latent: [N, latent_dim] latent representation
        """
        # LSTM encoder
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)  # [N, L, hidden_dim]

        # Use last hidden state as sequence representation
        latent = self.encoder_fc(h_n[-1])  # [N, latent_dim]

        return latent

    def decode(self, z, seq_len):
        """
        Decode latent to trajectory

        Args:
            z: [N, latent_dim] latent representation
            seq_len: Length of sequence to generate

        Returns:
            reconstruction: [N, L, D] reconstructed trajectory
        """
        N = z.shape[0]

        # Project latent to hidden dimension
        hidden = self.decoder_fc(z)  # [N, hidden_dim]

        # Repeat for each time step
        hidden_seq = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [N, L, hidden_dim]

        # LSTM decoder
        lstm_out, _ = self.decoder_lstm(hidden_seq)  # [N, L, hidden_dim]

        # Project to output dimension
        reconstruction = self.decoder_output(lstm_out)  # [N, L, D]

        return reconstruction

    def forward(self, x):
        """
        Forward pass: encode then decode

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            reconstruction: [N, L, D] reconstructed trajectory
            latent: [N, latent_dim] latent representation
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent, x.shape[1])

        return reconstruction, latent

    def compute_anomaly_score(self, x):
        """
        Compute anomaly score based on reconstruction error

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            anomaly_scores: [N] scalar anomaly score per trajectory
        """
        reconstruction, _ = self.forward(x)

        # MSE reconstruction error per trajectory
        mse = F.mse_loss(reconstruction, x, reduction='none')  # [N, L, D]
        anomaly_scores = mse.mean(dim=[1, 2])  # [N]

        return anomaly_scores


class TimeSeriesVAE(nn.Module):
    """
    Variational Autoencoder for time-series trajectory anomaly detection

    Architecture:
        - Encoder: LSTM -> mu, logvar (probabilistic latent)
        - Decoder: Sample from latent -> LSTM -> reconstruction

    Anomaly detection:
        - High reconstruction error + KL divergence indicates anomaly
        - Probabilistic modeling of normal trajectories
    """

    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_layers=2):
        """
        Initialize VAE

        Args:
            input_dim: Feature dimension (D)
            hidden_dim: Hidden dimension for LSTM
            latent_dim: Latent representation dimension
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder: trajectory -> mu, logvar
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent -> trajectory
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=False
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        Encode trajectory to latent distribution

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            mu: [N, latent_dim] mean of latent distribution
            logvar: [N, latent_dim] log variance of latent distribution
        """
        # LSTM encoder
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)  # [N, L, hidden_dim]

        # Use last hidden state
        hidden = h_n[-1]  # [N, hidden_dim]

        # Compute mu and logvar
        mu = self.fc_mu(hidden)  # [N, latent_dim]
        logvar = self.fc_logvar(hidden)  # [N, latent_dim]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon

        Args:
            mu: [N, latent_dim] mean
            logvar: [N, latent_dim] log variance

        Returns:
            z: [N, latent_dim] sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, seq_len):
        """
        Decode latent to trajectory

        Args:
            z: [N, latent_dim] latent representation
            seq_len: Length of sequence to generate

        Returns:
            reconstruction: [N, L, D] reconstructed trajectory
        """
        N = z.shape[0]

        # Project latent to hidden dimension
        hidden = self.decoder_fc(z)  # [N, hidden_dim]

        # Repeat for each time step
        hidden_seq = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [N, L, hidden_dim]

        # LSTM decoder
        lstm_out, _ = self.decoder_lstm(hidden_seq)  # [N, L, hidden_dim]

        # Project to output dimension
        reconstruction = self.decoder_output(lstm_out)  # [N, L, D]

        return reconstruction

    def forward(self, x):
        """
        Forward pass: encode -> reparameterize -> decode

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            reconstruction: [N, L, D] reconstructed trajectory
            mu: [N, latent_dim] latent mean
            logvar: [N, latent_dim] latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.shape[1])

        return reconstruction, mu, logvar

    def compute_anomaly_score(self, x):
        """
        Compute anomaly score based on reconstruction + KL divergence

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            anomaly_scores: [N] scalar anomaly score per trajectory
        """
        reconstruction, mu, logvar = self.forward(x)

        # Reconstruction error
        recon_loss = F.mse_loss(reconstruction, x, reduction='none')  # [N, L, D]
        recon_loss = recon_loss.mean(dim=[1, 2])  # [N]

        # KL divergence to standard normal
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [N]

        # Combined anomaly score
        anomaly_scores = recon_loss + 0.001 * kl_loss  # Weight KL lower

        return anomaly_scores


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based autoencoder for time-series anomaly detection

    Architecture:
        - Encoder: Multi-head self-attention to capture temporal dependencies
        - Decoder: Cross-attention to reconstruct trajectory

    Anomaly detection:
        - High reconstruction error indicates anomaly
        - Attention mechanism captures long-range dependencies
    """

    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        """
        Initialize Transformer Autoencoder

        Args:
            input_dim: Feature dimension (D)
            hidden_dim: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        Forward pass: encode then decode

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            reconstruction: [N, L, D] reconstructed trajectory
            memory: [N, L, hidden_dim] encoded representation
        """
        N, L, D = x.shape

        # Project input to hidden dimension
        x_proj = self.input_projection(x)  # [N, L, hidden_dim]

        # Add positional encoding
        x_pos = self.pos_encoder(x_proj)  # [N, L, hidden_dim]

        # Transformer encoder
        memory = self.transformer_encoder(x_pos)  # [N, L, hidden_dim]

        # Transformer decoder (use memory as both target and memory)
        decoded = self.transformer_decoder(x_pos, memory)  # [N, L, hidden_dim]

        # Project back to input dimension
        reconstruction = self.output_projection(decoded)  # [N, L, D]

        return reconstruction, memory

    def compute_anomaly_score(self, x):
        """
        Compute anomaly score based on reconstruction error

        Args:
            x: [N, L, D] trajectory tensor

        Returns:
            anomaly_scores: [N] scalar anomaly score per trajectory
        """
        reconstruction, _ = self.forward(x)

        # MSE reconstruction error per trajectory
        mse = F.mse_loss(reconstruction, x, reduction='none')  # [N, L, D]
        anomaly_scores = mse.mean(dim=[1, 2])  # [N]

        return anomaly_scores


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [N, L, D] tensor

        Returns:
            [N, L, D] tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient fine-tuning

    Instead of fine-tuning full weight matrix W, LoRA decomposes adaptation as:
        W' = W + BA
    where B: [out_dim, rank], A: [rank, in_dim]

    This dramatically reduces trainable parameters for domain adaptation.
    """

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        """
        Initialize LoRA layer

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank of decomposition (lower = fewer parameters)
            alpha: Scaling factor for LoRA weights
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x, original_output):
        """
        Apply LoRA adaptation

        Args:
            x: Input tensor [*, in_features]
            original_output: Output from original layer [*, out_features]

        Returns:
            Adapted output [*, out_features]
        """
        # Compute low-rank adaptation: (x @ A^T) @ B^T
        lora_output = (x @ self.lora_A.T) @ self.lora_B.T

        # Add to original output with scaling
        return original_output + self.scaling * lora_output


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['Linear']):
    """
    Apply LoRA to specified modules in a model

    Args:
        model: PyTorch model to adapt
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        target_modules: List of module types to adapt (e.g., ['Linear', 'LSTM'])

    Returns:
        model: Model with LoRA layers attached
        lora_params: List of LoRA parameters for training
    """
    lora_params = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'Linear' in target_modules:
            # Create LoRA layer
            lora = LoRALayer(module.in_features, module.out_features, rank, alpha)

            # Attach LoRA to module
            setattr(module, 'lora', lora)

            # Freeze original weights
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

            # Collect LoRA parameters
            lora_params.extend(lora.parameters())

            # Monkey-patch forward to include LoRA
            original_forward = module.forward
            def forward_with_lora(x, original_forward=original_forward, lora=lora):
                original_output = original_forward(x)
                return lora(x, original_output)
            module.forward = forward_with_lora

    print(f"  ✓ Applied LoRA: {len(lora_params)} trainable parameter tensors")

    return model, lora_params


def create_model(model_type, input_dim, hidden_dim=128, latent_dim=64, num_layers=2, num_heads=4):
    """
    Factory function to create time-series anomaly detection model

    Args:
        model_type: 'autoencoder', 'vae', or 'transformer'
        input_dim: Feature dimension (D)
        hidden_dim: Hidden dimension
        latent_dim: Latent dimension (for autoencoder/vae)
        num_layers: Number of layers
        num_heads: Number of attention heads (for transformer)

    Returns:
        model: PyTorch model

    Raises:
        ValueError: If model_type is not supported
    """
    model_type = model_type.lower()

    if model_type == 'autoencoder':
        return TimeSeriesAutoencoder(input_dim, hidden_dim, latent_dim, num_layers)
    elif model_type == 'vae':
        return TimeSeriesVAE(input_dim, hidden_dim, latent_dim, num_layers)
    elif model_type == 'transformer':
        return TimeSeriesTransformer(input_dim, hidden_dim, num_heads, num_layers)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose from 'autoencoder', 'vae', 'transformer'")
