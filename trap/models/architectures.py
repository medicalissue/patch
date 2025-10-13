import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TimeSeriesAutoencoder",
    "TimeSeriesVAE",
    "TemporalConvAutoencoder",
    "TemporalConvVAE",
    "TimeSeriesTransformer",
    "TimeSeriesTransformerVAE",
]


class TimeSeriesAutoencoder(nn.Module):
    """LSTM autoencoder for trajectory reconstruction."""

    def __init__(self, input_dim, hidden_dim: int = 128, latent_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])
        return latent

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden = self.decoder_fc(z)
        hidden_seq = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.decoder_lstm(hidden_seq)
        reconstruction = self.decoder_output(lstm_out)
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        reconstruction = self.decode(latent, x.shape[1])
        return reconstruction, latent

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction="none")
        anomaly_scores = mse.mean(dim=[1, 2])
        return anomaly_scores


class TimeSeriesVAE(nn.Module):
    """LSTM variational autoencoder for trajectory modelling."""

    def __init__(self, input_dim, hidden_dim: int = 128, latent_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, (h_n, _) = self.encoder_lstm(x)
        hidden = h_n[-1]
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden = self.decoder_fc(z)
        hidden_seq = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.decoder_lstm(hidden_seq)
        reconstruction = self.decoder_output(lstm_out)
        return reconstruction

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.shape[1])
        return reconstruction, mu, logvar

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(reconstruction, x, reduction="none")
        recon_loss = recon_loss.mean(dim=[1, 2])
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return recon_loss + 0.001 * kl_loss


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block with dilation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = self.residual(x) if self.residual is not None else x
        return out + res


class TemporalConvAutoencoder(nn.Module):
    """Temporal convolutional autoencoder."""

    def __init__(
        self,
        input_dim,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        encoder_blocks = []
        in_channels = input_dim
        dilation = 1
        for _ in range(num_layers):
            encoder_blocks.append(
                TemporalConvBlock(in_channels, hidden_dim, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
            in_channels = hidden_dim
            dilation *= dilation_base
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=1, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_output = nn.Conv1d(hidden_dim, input_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = x.permute(0, 2, 1)
        for block in self.encoder_blocks:
            out = block(out)
        pooled = out.mean(dim=-1)
        latent = self.encoder_fc(pooled)
        return latent

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden = self.decoder_fc(z)
        out = hidden.unsqueeze(-1).repeat(1, 1, seq_len)
        for block in self.decoder_blocks:
            out = block(out)
        reconstruction = self.decoder_output(out)
        return reconstruction.permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        reconstruction = self.decode(latent, x.shape[1])
        return reconstruction, latent

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction="none")
        return mse.mean(dim=[1, 2])


class TemporalConvVAE(nn.Module):
    """Temporal convolutional variational autoencoder."""

    def __init__(
        self,
        input_dim,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        encoder_blocks = []
        in_channels = input_dim
        dilation = 1
        for _ in range(num_layers):
            encoder_blocks.append(
                TemporalConvBlock(in_channels, hidden_dim, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )
            in_channels = hidden_dim
            dilation *= dilation_base
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=1, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_output = nn.Conv1d(hidden_dim, input_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

    def encode(self, x: torch.Tensor):
        out = x.permute(0, 2, 1)
        for block in self.encoder_blocks:
            out = block(out)
        pooled = out.mean(dim=-1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        hidden = self.decoder_fc(z)
        out = hidden.unsqueeze(-1).repeat(1, 1, seq_len)
        for block in self.decoder_blocks:
            out = block(out)
        reconstruction = self.decoder_output(out)
        return reconstruction.permute(0, 2, 1)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.shape[1])
        return reconstruction, mu, logvar

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, mu, logvar = self.forward(x)

        recon_loss = F.mse_loss(reconstruction, x, reduction="none")
        recon_loss = recon_loss.mean(dim=[1, 2])

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return recon_loss + 0.001 * kl_loss


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer-based sequence autoencoder."""

    def __init__(self, input_dim, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        N, L, _ = x.shape
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        memory = self.transformer_encoder(x_pos)
        decoded = self.transformer_decoder(x_pos, memory)
        reconstruction = self.output_projection(decoded)
        return reconstruction, memory

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction="none")
        anomaly_scores = mse.mean(dim=[1, 2])
        return anomaly_scores


class TimeSeriesTransformerVAE(nn.Module):
    """Transformer-based variational autoencoder for trajectory modelling."""

    def __init__(
        self,
        input_dim,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Encoder
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # VAE latent projection
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent distribution parameters."""
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        memory = self.transformer_encoder(x_pos)
        # Use mean pooling over sequence
        pooled = memory.mean(dim=1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent vector to sequence."""
        # Project latent to hidden and repeat for sequence
        hidden = self.latent_projection(z)
        # Create memory as repeated latent representation
        memory = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # Apply positional encoding to memory
        memory = self.pos_encoder(memory)
        # Decode
        decoded = self.transformer_decoder(memory, memory)
        reconstruction = self.output_projection(decoded)
        return reconstruction

    def forward(self, x: torch.Tensor):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x.shape[1])
        return reconstruction, mu, logvar

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score combining reconstruction and KL divergence."""
        reconstruction, mu, logvar = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction="none")
        recon_loss = recon_loss.mean(dim=[1, 2])

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return recon_loss + 0.001 * kl_loss
