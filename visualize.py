"""
Visualization Module for Model-based Patch Detection

This module provides visualization functions for displaying detection results.
Supports model-based reconstruction error approach with simple, clean visualization.
"""

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def denormalize_image(img_tensor):
    """
    Reverse ImageNet normalization

    Args:
        img_tensor: [C, H, W] tensor with ImageNet normalization

    Returns:
        [H, W, C] numpy array with values in [0, 1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def visualize_results(image, anomaly_map_gpu, patch_mask_gpu,
                     image_name="", threshold=0.0, detection_pixel_threshold=0,
                     threshold_method="percentile", threshold_formula="",
                     model_type="autoencoder"):
    """
    Visualize model-based detection results with reconstruction error

    Creates a clean visualization showing:
    - Original image
    - Reconstruction error heatmap
    - Detection overlay
    - Detection metrics

    This visualization supports the model-based detection approach:
      Phase 1: Train model on clean ImageNet images
      Phase 2 (Optional): Domain adaptation with LoRA
      Phase 3: Detection using reconstruction error

    Args:
        image: [C, H, W] tensor - original image
        anomaly_map_gpu: [H, W] tensor on GPU - reconstruction error map
        patch_mask_gpu: [H, W] boolean tensor on GPU - detection mask
        image_name: str - image filename
        threshold: float - computed threshold value
        detection_pixel_threshold: int - minimum pixels for positive detection
        threshold_method: str - threshold calculation method ('mean_std', 'median_mad', 'percentile')
        threshold_formula: str - formula description for threshold calculation
        model_type: str - model type ('autoencoder', 'vae', 'transformer')

    Returns:
        matplotlib figure object
    """

    # Convert GPU tensors to numpy for visualization
    anomaly_map = anomaly_map_gpu.cpu().numpy()
    patch_mask = patch_mask_gpu.cpu().numpy()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    img_display = denormalize_image(image)

    # Row 1: Original, Anomaly Map, Detection Overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_display)
    ax1.set_title(f'Original Image\n{image_name}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im1 = ax2.imshow(anomaly_map, cmap='hot', interpolation='bilinear')
    title_text = f'Reconstruction Error Map\nMax: {anomaly_map.max():.3f} | Threshold: {threshold:.3f}'
    ax2.set_title(title_text, fontsize=11, fontweight='bold')
    ax2.axis('off')

    cbar1 = plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_display)

    if patch_mask.sum() > 0:
        mask_upsampled = F.interpolate(
            torch.tensor(patch_mask).float().unsqueeze(0).unsqueeze(0),
            size=image.shape[1:], mode='nearest'
        )[0, 0].numpy()

        overlay = torch.zeros((*mask_upsampled.shape, 4))
        overlay[mask_upsampled > 0.5] = torch.tensor([1, 0, 0, 0.5])  # Red overlay
        ax3.imshow(overlay.numpy())
        ax3.contour(mask_upsampled, colors='red', linewidths=2, levels=[0.5])

    detected_pixels = patch_mask.sum()
    detection_rate = detected_pixels / patch_mask.size * 100
    ax3.set_title(f'Detection Overlay\nDetected: {detected_pixels}/{patch_mask.size} ({detection_rate:.1f}%)',
                 fontsize=11, fontweight='bold')
    ax3.axis('off')

    # Row 2: Statistics distribution
    ax4 = fig.add_subplot(gs[1, 0])
    scores_flat = anomaly_map.flatten()
    ax4.hist(scores_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    ax4.set_xlabel('Reconstruction Error', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Score Distribution', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Row 2: Binary mask visualization
    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.imshow(patch_mask, cmap='Reds', interpolation='nearest')
    ax5.set_title(f'Binary Detection Mask\nAnomalous: {detected_pixels} pixels', fontsize=11, fontweight='bold')
    ax5.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=10)

    # Row 2: Metrics panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    spatial_res = f"{anomaly_map.shape[0]}×{anomaly_map.shape[1]}"

    if detected_pixels > detection_pixel_threshold:
        status = "🔴 ANOMALY DETECTED"
        status_color = 'red'
        bg_color = 'mistyrose'
    else:
        status = "🟢 CLEAN"
        status_color = 'green'
        bg_color = 'lightgreen'

    # Model type display name
    model_display = {
        'autoencoder': 'Autoencoder (LSTM)',
        'vae': 'VAE (Variational)',
        'transformer': 'Transformer (Attention)'
    }.get(model_type.lower(), model_type)

    metrics_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL-BASED DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━

Model: {model_display}

Detection Method:
  Reconstruction Error

Spatial Resolution:
  {spatial_res}

Threshold Method:
  {threshold_method}

Threshold Formula:
  {threshold_formula}

Threshold Value:
  {threshold:.4f}

Score Statistics:
  Max:  {anomaly_map.max():.4f}
  Mean: {anomaly_map.mean():.4f}
  Std:  {anomaly_map.std():.4f}

Detection Results:
  Pixels: {detected_pixels} / {patch_mask.size}
  Rate:   {detection_rate:.2f}%
  Status: {status}

━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax6.text(0.5, 0.5, metrics_text,
            fontsize=10,
            verticalalignment='center',
            horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1.2',
                     facecolor=bg_color,
                     alpha=0.9,
                     edgecolor=status_color,
                     linewidth=3))

    return fig
