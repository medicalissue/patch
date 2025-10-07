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
                     hht_map_gpu, cqt_map_gpu, sst_map_gpu,
                     score_flags, image_name="", thresholds=None, detection_pixel_threshold=0,
                     threshold_method="mean_std", threshold_formula="",
                     fusion_method="voting", voting_threshold=4):
    """
    Visualize Few-shot detection results with per-score thresholds and voting

    Creates a comprehensive visualization showing:
    - Original image
    - Vote count map (number of scores that flagged each pixel as anomalous)
    - Detection overlay
    - Component score maps with thresholds and flagged pixel counts
    - Detection metrics and per-score statistics

    This visualization supports the Few-shot detection approach:
      Phase 1: Base learning from ImageNet (learned statistics)
      Phase 2: Per-score adaptive thresholds from domain clean images
      Phase 3: Detection using voting fusion

    Args:
        image: [C, H, W] tensor - original image
        anomaly_map_gpu: [H, W] tensor on GPU - vote count map
        patch_mask_gpu: [H, W] boolean tensor on GPU - detection mask
        spectral_map_gpu: [H, W] tensor on GPU - spectral scores
        wavelet_map_gpu: [H, W] tensor on GPU - wavelet scores
        stft_map_gpu: [H, W] tensor on GPU - STFT scores
        hht_map_gpu: [H, W] tensor on GPU - HHT/EMD scores
        cqt_map_gpu: [H, W] tensor on GPU - CQT scores
        sst_map_gpu: [H, W] tensor on GPU - Synchrosqueezed STFT scores
        score_flags: dict - per-score binary flags [H, W] showing which pixels were flagged
        image_name: str - image filename
        thresholds: dict - per-score thresholds {'spectral': t1, ..., 'sst': t6}
        detection_pixel_threshold: int - minimum pixels for positive detection
        threshold_method: str - threshold calculation method ('mean_std', 'median_mad', 'percentile')
        threshold_formula: str - formula description for threshold calculation
        fusion_method: str - score fusion method ('voting', 'weighted_voting', 'all', 'any')
        voting_threshold: int - number of scores that must be anomalous (for 'voting' method, default 4/6)

    Returns:
        matplotlib figure object
    """

    # Convert GPU tensors to numpy for visualization
    anomaly_map = anomaly_map_gpu.cpu().numpy()
    patch_mask = patch_mask_gpu.cpu().numpy()
    spectral_map = spectral_map_gpu.cpu().numpy()
    wavelet_map = wavelet_map_gpu.cpu().numpy()
    stft_map = stft_map_gpu.cpu().numpy()
    hht_map = hht_map_gpu.cpu().numpy()
    cqt_map = cqt_map_gpu.cpu().numpy()
    sst_map = sst_map_gpu.cpu().numpy()

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

    title_text = f'Vote Count Map'
    if fusion_method == 'voting':
        title_text += f' ({voting_threshold}/6 votes needed)'
    title_text += f'\nMax: {anomaly_map.max():.1f}, Mean: {anomaly_map.mean():.1f}'
    ax2.set_title(title_text, fontsize=10, fontweight='bold')
    ax2.axis('off')

    cbar1 = plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=9)
    
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
    
    # Row 2 & 3: Component scores (6 scores: Spectral, Wavelet, STFT, HHT, CQT, SST)
    # Get thresholds (handle both dict and None)
    spectral_th = thresholds.get('spectral', 0) if thresholds else 0
    wavelet_th = thresholds.get('wavelet', 0) if thresholds else 0
    stft_th = thresholds.get('stft', 0) if thresholds else 0
    hht_th = thresholds.get('hht', 0) if thresholds else 0
    cqt_th = thresholds.get('cqt', 0) if thresholds else 0
    sst_th = thresholds.get('sst', 0) if thresholds else 0

    # Count flagged pixels
    spectral_flags = score_flags['spectral'].cpu().numpy()
    wavelet_flags = score_flags['wavelet'].cpu().numpy()
    stft_flags = score_flags['stft'].cpu().numpy()
    hht_flags = score_flags['hht'].cpu().numpy()
    cqt_flags = score_flags['cqt'].cpu().numpy()
    sst_flags = score_flags['sst'].cpu().numpy()

    # Row 2: First 3 scores
    ax4 = fig.add_subplot(gs[1, 0])
    im2 = ax4.imshow(spectral_map, cmap='plasma', interpolation='bilinear')
    ax4.set_title(f'Spectral (Z)\nMax: {spectral_map.max():.2f} | Th: {spectral_th:.2f} | Flagged: {spectral_flags.sum()}',
                 fontsize=9, fontweight='bold')
    ax4.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    im3 = ax5.imshow(wavelet_map, cmap='viridis', interpolation='bilinear')
    ax5.set_title(f'Wavelet (M)\nMax: {wavelet_map.max():.2f} | Th: {wavelet_th:.2f} | Flagged: {wavelet_flags.sum()}',
                 fontsize=9, fontweight='bold')
    ax5.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=9)

    ax6 = fig.add_subplot(gs[1, 2])
    im4 = ax6.imshow(stft_map, cmap='coolwarm', interpolation='bilinear')
    ax6.set_title(f'STFT (M)\nMax: {stft_map.max():.2f} | Th: {stft_th:.2f} | Flagged: {stft_flags.sum()}',
                 fontsize=9, fontweight='bold')
    ax6.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax6, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(labelsize=9)

    # Row 3: Last 3 scores
    ax7 = fig.add_subplot(gs[2, 0])
    im5 = ax7.imshow(hht_map, cmap='inferno', interpolation='bilinear')
    ax7.set_title(f'HHT/EMD (M)\nMax: {hht_map.max():.2f} | Th: {hht_th:.2f} | Flagged: {hht_flags.sum()}',
                 fontsize=9, fontweight='bold')
    ax7.axis('off')
    cbar5 = plt.colorbar(im5, ax=ax7, fraction=0.046, pad=0.04)
    cbar5.ax.tick_params(labelsize=9)

    ax8 = fig.add_subplot(gs[2, 1])
    im6 = ax8.imshow(cqt_map, cmap='magma', interpolation='bilinear')
    ax8.set_title(f'CQT (M)\nMax: {cqt_map.max():.2f} | Th: {cqt_th:.2f} | Flagged: {cqt_flags.sum()}',
                 fontsize=9, fontweight='bold')
    ax8.axis('off')
    cbar6 = plt.colorbar(im6, ax=ax8, fraction=0.046, pad=0.04)
    cbar6.ax.tick_params(labelsize=9)

    ax9 = fig.add_subplot(gs[2, 2])
    im7 = ax9.imshow(sst_map, cmap='cividis', interpolation='bilinear')
    ax9.set_title(f'SST (M)\nMax: {sst_map.max():.2f} | Th: {sst_th:.2f} | Flagged: {sst_flags.sum()}',
                 fontsize=9, fontweight='bold')
    ax9.axis('off')
    cbar7 = plt.colorbar(im7, ax=ax9, fraction=0.046, pad=0.04)
    cbar7.ax.tick_params(labelsize=9)

    # Row 4: Metrics and Few-shot explanation
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')

    spatial_res = f"{anomaly_map.shape[0]}×{anomaly_map.shape[1]}"

    if detected_pixels > detection_pixel_threshold:
        status = "🔴 ANOMALY DETECTED"
        status_color = 'red'
        bg_color = 'wheat'
    else:
        status = "🟢 CLEAN"
        status_color = 'green'
        bg_color = 'lightgreen'

    # Format thresholds
    threshold_lines = []
    for name in ['spectral', 'wavelet', 'stft', 'hht', 'cqt', 'sst']:
        th_val = thresholds.get(name, 0) if thresholds else 0
        threshold_lines.append(f"  {name:12s}: {th_val:.3f}")

    metrics_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FEW-SHOT PATCH DETECTION RESULTS (Per-Score Thresholds + Voting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Detection Method: Few-shot Learning (Z-score + Mahalanobis) with Voting Fusion
  Phase 1: Base Learning      → Learn normal trajectory characteristics from ImageNet
  Phase 2: Threshold Adaptation → Set per-score adaptive thresholds using domain clean images
  Phase 3: Testing            → Detect patches using per-score thresholds + {fusion_method}

Spatial Resolution: {spatial_res} ({anomaly_map.size} pixels)

Vote Count Map:
  Max Votes:  {anomaly_map.max():.1f}/6  │  Mean: {anomaly_map.mean():.1f}  │  Std: {anomaly_map.std():.1f}

Fusion Method: {fusion_method}
  {'Voting Threshold: ' + str(voting_threshold) + '/6 scores must be anomalous' if fusion_method == 'voting' else ''}

Per-Score Adaptive Thresholds ({threshold_method}):
  Method:  {threshold_method}
  Formula: {threshold_formula}
{chr(10).join(threshold_lines)}

Detection Results:
  Detected Pixels: {detected_pixels} / {patch_mask.size} ({detection_rate:.2f}%)
  Status: {status}

Component Scores (Max / Flagged Pixels):
  Spectral (Z):  {spectral_map.max():.3f} / {spectral_flags.sum()}  │  Wavelet (M):  {wavelet_map.max():.3f} / {wavelet_flags.sum()}
  STFT (M):      {stft_map.max():.3f} / {stft_flags.sum()}           │  HHT/EMD (M):  {hht_map.max():.3f} / {hht_flags.sum()}
  CQT (M):       {cqt_map.max():.3f} / {cqt_flags.sum()}            │  SST (M):      {sst_map.max():.3f} / {sst_flags.sum()}

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