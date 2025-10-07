"""
Visualization Module for Few-shot Patch Detection

This module provides visualization functions for displaying detection results.
Supports Few-shot detection approach with multiple component visualizations.
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
                     spectral_map_gpu, wavelet_map_gpu, stft_map_gpu,
                     entropy_map_gpu, hf_ratio_map_gpu, skewness_map_gpu,
                     image_name="", threshold=None, detection_pixel_threshold=0):
    """
    Visualize Few-shot detection results

    Creates a comprehensive visualization showing:
    - Original image
    - Combined anomaly score map
    - Detection overlay
    - Component score maps (spectral, wavelet, STFT, entropy, HF ratio, skewness)
    - Detection metrics and statistics

    This visualization supports the Few-shot detection approach:
      Phase 1: Base learning from ImageNet (learned statistics)
      Phase 2: Adaptive threshold from domain clean images
      Phase 3: Detection on test images

    Args:
        image: [C, H, W] tensor - original image
        anomaly_map_gpu: [H, W] tensor on GPU - combined anomaly scores
        patch_mask_gpu: [H, W] boolean tensor on GPU - detection mask
        spectral_map_gpu: [H, W] tensor on GPU - spectral scores
        wavelet_map_gpu: [H, W] tensor on GPU - wavelet scores
        stft_map_gpu: [H, W] tensor on GPU - STFT scores
        entropy_map_gpu: [H, W] tensor on GPU - spectral entropy scores
        hf_ratio_map_gpu: [H, W] tensor on GPU - high-frequency ratio scores
        skewness_map_gpu: [H, W] tensor on GPU - spectral skewness scores
        image_name: str - image filename
        threshold: float - detection threshold (from Phase 2 adaptation)
        detection_pixel_threshold: int - minimum pixels for positive detection

    Returns:
        matplotlib figure object
    """

    # Convert GPU tensors to numpy for visualization
    anomaly_map = anomaly_map_gpu.cpu().numpy()
    patch_mask = patch_mask_gpu.cpu().numpy()
    spectral_map = spectral_map_gpu.cpu().numpy()
    wavelet_map = wavelet_map_gpu.cpu().numpy()
    stft_map = stft_map_gpu.cpu().numpy()
    entropy_map = entropy_map_gpu.cpu().numpy()
    hf_ratio_map = hf_ratio_map_gpu.cpu().numpy()
    skewness_map = skewness_map_gpu.cpu().numpy()

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    img_display = denormalize_image(image)
    
    # Row 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_display)
    ax1.set_title(f'Original Image\n{image_name}', fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im1 = ax2.imshow(anomaly_map, cmap='hot', interpolation='bilinear')
    
    threshold_text = f'Threshold: {threshold:.2f}' if threshold is not None else ''
    title_text = f'Anomaly Score Map\nMax: {anomaly_map.max():.2f}, Mean: {anomaly_map.mean():.2f}'
    if threshold is not None:
        title_text += f'\n{threshold_text}'
    ax2.set_title(title_text, fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    cbar1 = plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=9)
    
    if threshold is not None and threshold <= anomaly_map.max():
        cbar1.ax.axhline(y=threshold, color='cyan', linestyle='--', linewidth=2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_display)
    
    if patch_mask.sum() > 0:
        mask_upsampled = F.interpolate(
            torch.tensor(patch_mask).float().unsqueeze(0).unsqueeze(0),
            size=image.shape[1:], mode='nearest'
        )[0, 0].numpy()
        
        overlay = torch.zeros((*mask_upsampled.shape, 4))
        overlay[mask_upsampled > 0.5] = torch.tensor([0, 1, 1, 0.4])
        ax3.imshow(overlay.numpy())
        ax3.contour(mask_upsampled, colors='cyan', linewidths=2, levels=[0.5])
    
    detected_pixels = patch_mask.sum()
    detection_rate = detected_pixels / patch_mask.size * 100
    ax3.set_title(f'Detection Overlay\nDetected: {detected_pixels}/{patch_mask.size} ({detection_rate:.1f}%)', 
                 fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Component scores (Spectral, Wavelet, STFT)
    ax4 = fig.add_subplot(gs[1, 0])
    im2 = ax4.imshow(spectral_map, cmap='plasma', interpolation='bilinear')
    ax4.set_title(f'Spectral Score\nMax: {spectral_map.max():.2f}, Mean: {spectral_map.mean():.2f}',
                 fontsize=10, fontweight='bold')
    ax4.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    im3 = ax5.imshow(wavelet_map, cmap='viridis', interpolation='bilinear')
    ax5.set_title(f'Wavelet Score\nMax: {wavelet_map.max():.2f}, Mean: {wavelet_map.mean():.2f}',
                 fontsize=10, fontweight='bold')
    ax5.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=9)

    ax6 = fig.add_subplot(gs[1, 2])
    im4 = ax6.imshow(stft_map, cmap='coolwarm', interpolation='bilinear')
    ax6.set_title(f'STFT Score\nMax: {stft_map.max():.2f}, Mean: {stft_map.mean():.2f}',
                 fontsize=10, fontweight='bold')
    ax6.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax6, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(labelsize=9)

    # Row 3: More component scores (Entropy, HF Ratio, Skewness)
    ax7 = fig.add_subplot(gs[2, 0])
    im5 = ax7.imshow(entropy_map, cmap='inferno', interpolation='bilinear')
    ax7.set_title(f'Spectral Entropy Score\nMax: {entropy_map.max():.2f}, Mean: {entropy_map.mean():.2f}',
                 fontsize=10, fontweight='bold')
    ax7.axis('off')
    cbar5 = plt.colorbar(im5, ax=ax7, fraction=0.046, pad=0.04)
    cbar5.ax.tick_params(labelsize=9)

    ax8 = fig.add_subplot(gs[2, 1])
    im6 = ax8.imshow(hf_ratio_map, cmap='magma', interpolation='bilinear')
    ax8.set_title(f'High-Freq Ratio Score\nMax: {hf_ratio_map.max():.2f}, Mean: {hf_ratio_map.mean():.2f}',
                 fontsize=10, fontweight='bold')
    ax8.axis('off')
    cbar6 = plt.colorbar(im6, ax=ax8, fraction=0.046, pad=0.04)
    cbar6.ax.tick_params(labelsize=9)

    ax9 = fig.add_subplot(gs[2, 2])
    im7 = ax9.imshow(skewness_map, cmap='cividis', interpolation='bilinear')
    ax9.set_title(f'Spectral Skewness Score\nMax: {skewness_map.max():.2f}, Mean: {skewness_map.mean():.2f}',
                 fontsize=10, fontweight='bold')
    ax9.axis('off')
    cbar7 = plt.colorbar(im7, ax=ax9, fraction=0.046, pad=0.04)
    cbar7.ax.tick_params(labelsize=9)

    # Row 4: Metrics and Few-shot explanation
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')

    spatial_res = f"{anomaly_map.shape[0]}×{anomaly_map.shape[1]}"
    threshold_str = f"{threshold:.3f}" if threshold is not None else "N/A"

    if detected_pixels > detection_pixel_threshold:
        status = "🔴 ANOMALY DETECTED"
        status_color = 'red'
        bg_color = 'wheat'
    else:
        status = "🟢 CLEAN"
        status_color = 'green'
        bg_color = 'lightgreen'

    metrics_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FEW-SHOT PATCH DETECTION RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detection Method: Few-shot Learning with Absolute Deviation
  Phase 1: Base Learning      → Learn normal trajectory characteristics from ImageNet
  Phase 2: Threshold Adaptation → Set adaptive threshold using domain-specific clean images
  Phase 3: Testing            → Detect patches using absolute deviation comparison

Spatial Resolution: {spatial_res} ({anomaly_map.size} pixels)

Anomaly Scores:
  Max:  {anomaly_map.max():.4f}  │  Mean: {anomaly_map.mean():.4f}  │  Std: {anomaly_map.std():.4f}

Adaptive Threshold: {threshold_str} (from Phase 2)

Detection Results:
  Detected Pixels: {detected_pixels} / {patch_mask.size} ({detection_rate:.2f}%)
  Status: {status}

Component Scores (Max - Absolute Deviation):
  Spectral: {spectral_map.max():.3f}  │  Wavelet: {wavelet_map.max():.3f}  │  STFT: {stft_map.max():.3f}
  Entropy: {entropy_map.max():.3f}  │  HF Ratio: {hf_ratio_map.max():.3f}  │  Skewness: {skewness_map.max():.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax10.text(0.5, 0.5, metrics_text,
            fontsize=9,
            verticalalignment='center',
            horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1.5',
                     facecolor=bg_color,
                     alpha=0.9,
                     edgecolor=status_color,
                     linewidth=3))

    return fig