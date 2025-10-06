import torch
import matplotlib as plt

from config import Config

def denormalize_image(img_tensor):
    """ImageNet normalize 역변환"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

def visualize_results(image, anomaly_map_gpu, patch_mask_gpu, 
                     vector_map_gpu, spectral_map_gpu, image_name="", threshold=None):
    """결과 시각화 (GPU tensor를 받아서 필요할 때만 CPU로 이동)"""
    
    # GPU tensor를 numpy로 변환 (visualization을 위해서만)
    anomaly_map = anomaly_map_gpu.cpu().numpy()
    patch_mask = patch_mask_gpu.cpu().numpy()
    vector_map = vector_map_gpu.cpu().numpy()
    spectral_map = spectral_map_gpu.cpu().numpy()
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
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
    
    # Row 2
    ax4 = fig.add_subplot(gs[1, 0])
    im2 = ax4.imshow(vector_map, cmap='viridis', interpolation='bilinear')
    ax4.set_title(f'Vector Field Score\nMax: {vector_map.max():.2f}, Mean: {vector_map.mean():.2f}', 
                 fontsize=10, fontweight='bold')
    ax4.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=9)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im3 = ax5.imshow(spectral_map, cmap='plasma', interpolation='bilinear')
    ax5.set_title(f'Spectral Analysis Score\nMax: {spectral_map.max():.2f}, Mean: {spectral_map.mean():.2f}', 
                 fontsize=10, fontweight='bold')
    ax5.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax5, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=9)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    spatial_res = f"{anomaly_map.shape[0]}×{anomaly_map.shape[1]}"
    threshold_str = f"{threshold:.3f}" if threshold is not None else "N/A"
    
    if detected_pixels > Config.DETECTION_PIXEL_THRESHOLD:
        status = "🔴 ANOMALY DETECTED"
        status_color = 'red'
    else:
        status = "🟢 CLEAN"
        status_color = 'green'
    
    metrics_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━
  DETECTION METRICS
━━━━━━━━━━━━━━━━━━━━━━━

Spatial Resolution:
  {spatial_res} ({anomaly_map.size} pixels)

Anomaly Scores:
  Max:  {anomaly_map.max():.4f}
  Mean: {anomaly_map.mean():.4f}
  Std:  {anomaly_map.std():.4f}
  
Threshold: {threshold_str}

Detection Results:
  Pixels: {detected_pixels} / {patch_mask.size}
  Rate:   {detection_rate:.2f}%

Component Scores:
  Vector Field (max):  {vector_map.max():.3f}
  Spectral (max):      {spectral_map.max():.3f}

Status: {status}
━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax6.text(0.05, 0.95, metrics_text, 
            fontsize=10, 
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', 
                     facecolor='wheat' if detected_pixels > Config.DETECTION_PIXEL_THRESHOLD else 'lightgreen', 
                     alpha=0.8,
                     edgecolor=status_color,
                     linewidth=2))
    
    return fig