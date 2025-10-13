from trap.models.architectures import (
    TemporalConvAutoencoder,
    TemporalConvVAE,
    TimeSeriesAutoencoder,
    TimeSeriesTransformer,
    TimeSeriesTransformerVAE,
    TimeSeriesVAE,
)
from trap.models.lora import apply_lora_to_model

__all__ = ["create_model", "apply_lora_to_model"]


def create_model(
    model_type,
    input_dim,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    tcn_kernel_size: int = 3,
    tcn_dilation_base: int = 2,
    tcn_dropout: float = 0.1,
):
    """
    Factory function to create time-series anomaly detection models.

    Args:
        model_type: autoencoder, vae, tcn_autoencoder, tcn_vae, transformer, transformer_vae
        input_dim: feature dimension
        hidden_dim: hidden dimension
        latent_dim: latent dimension (autoencoder/vae)
        num_layers: number of layers
        num_heads: attention heads (transformer)
        dropout: dropout rate (transformer)
        tcn_kernel_size: kernel size for TCN variants
        tcn_dilation_base: dilation factor per block
        tcn_dropout: dropout used in TCN blocks
    """
    model_type = model_type.lower()

    if model_type == "autoencoder":
        return TimeSeriesAutoencoder(input_dim, hidden_dim, latent_dim, num_layers)
    if model_type == "vae":
        return TimeSeriesVAE(input_dim, hidden_dim, latent_dim, num_layers)
    if model_type == "tcn_autoencoder":
        return TemporalConvAutoencoder(
            input_dim,
            hidden_dim,
            latent_dim,
            num_layers,
            kernel_size=tcn_kernel_size,
            dilation_base=tcn_dilation_base,
            dropout=tcn_dropout,
        )
    if model_type == "tcn_vae":
        return TemporalConvVAE(
            input_dim,
            hidden_dim,
            latent_dim,
            num_layers,
            kernel_size=tcn_kernel_size,
            dilation_base=tcn_dilation_base,
            dropout=tcn_dropout,
        )
    if model_type == "transformer":
        return TimeSeriesTransformer(input_dim, hidden_dim, num_heads, num_layers, dropout)
    if model_type == "transformer_vae":
        return TimeSeriesTransformerVAE(input_dim, hidden_dim, latent_dim, num_heads, num_layers, dropout)

    raise ValueError(
        f"Unsupported model_type: {model_type}. Choose from "
        "'autoencoder', 'vae', 'tcn_autoencoder', 'tcn_vae', 'transformer', 'transformer_vae'"
    )
