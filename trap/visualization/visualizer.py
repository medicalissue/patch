"""Visualization utilities for TRAP detection results."""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


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

def _connected_components(mask):
    """
    Find 4-connected components in a binary mask.

    Args:
        mask: [H, W] boolean numpy array

    Returns:
        labels: [H, W] int32 array with 0 for background and positive integers per component
        num_labels: number of detected components
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    stack = []

    for i in range(h):
        for j in range(w):
            if not mask[i, j] or labels[i, j] != 0:
                continue
            label += 1
            stack.append((i, j))
            while stack:
                x, y = stack.pop()
                if labels[x, y] != 0:
                    continue
                labels[x, y] = label
                for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and labels[nx, ny] == 0:
                        stack.append((nx, ny))

    return labels, label


def _build_component_colors(num_labels):
    """
    Create a color mapping for components.

    Args:
        num_labels: number of connected components

    Returns:
        dict mapping label -> RGBA tuple
    """
    if num_labels <= 0:
        return {}

    cmap = plt.cm.get_cmap('tab20', num_labels)
    colors = {}
    for idx in range(num_labels):
        rgb = cmap(idx)[:3]
        colors[idx + 1] = (*rgb, 0.5)
    return colors


def visualize_results(image, anomaly_map_gpu, patch_mask_gpu,
                     image_name="", threshold=0.0, detection_pixel_threshold=0,
                     threshold_method="percentile",
                     model_type="autoencoder", trajectories=None,
                     gt_grid=None, pred_grid=None, spatial_resolution=14, grid_metrics=None):
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
        model_type: str - model type ('autoencoder', 'vae', 'tcn_autoencoder', 'tcn_vae', 'transformer')
        trajectories: Optional torch.Tensor [H, W, L, D] - trajectories used for t-SNE visualization

    Returns:
        matplotlib figure object
    """

    # Convert GPU tensors to numpy for visualization
    anomaly_map = anomaly_map_gpu.cpu().numpy()
    patch_mask = patch_mask_gpu.cpu().numpy()

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, height_ratios=[1.0, 1.0, 1.0, 0.5])

    img_display = denormalize_image(image)

    component_labels, num_components = _connected_components(patch_mask.astype(bool))
    component_colors = _build_component_colors(num_components)

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
        labels_tensor = torch.from_numpy(component_labels).float().unsqueeze(0).unsqueeze(0)
        upsampled_labels = F.interpolate(
            labels_tensor,
            size=image.shape[1:],
            mode='nearest'
        )[0, 0].numpy().round().astype(np.int32)

        overlay = np.zeros((*upsampled_labels.shape, 4), dtype=np.float32)
        contour_levels = []
        for label_idx, color in component_colors.items():
            mask = upsampled_labels == label_idx
            if not np.any(mask):
                continue
            overlay[mask] = np.array(color, dtype=np.float32)
            contour_levels.append(label_idx)

        ax3.imshow(overlay)
        for label_idx in contour_levels:
            ax3.contour(upsampled_labels == label_idx, colors=[component_colors[label_idx][:3]],
                        linewidths=2, levels=[0.5])

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

    # Row 2: t-SNE visualization
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_title('t-SNE of Trajectories', fontsize=11, fontweight='bold')
    ax7.axis('off')

    legend_handles = []
    if trajectories is not None:
        if torch.is_tensor(trajectories):
            traj_cpu = trajectories.detach().cpu().numpy()
        else:
            traj_cpu = np.asarray(trajectories)
        H, W, L, D = traj_cpu.shape
        flattened = traj_cpu.reshape(H * W, L * D).astype(np.float32)

        # Ensure perplexity is valid
        n_samples = flattened.shape[0]
        if n_samples > 2:
            perplexity = max(5, min(30, n_samples // 3))
            perplexity = min(perplexity, n_samples - 1)
            if perplexity < 2:
                perplexity = n_samples - 1 if n_samples - 1 >= 1 else 1

            tsne = TSNE(
                n_components=2,
                init='pca',
                learning_rate='auto',
                perplexity=perplexity
            )
            embedding = tsne.fit_transform(flattened)

            labels_flat = component_labels.reshape(-1)
            colors = []
            sizes = []
            for lbl in labels_flat:
                if lbl == 0:
                    colors.append((0.7, 0.7, 0.7, 0.6))
                    sizes.append(20)
                else:
                    rgba = component_colors.get(lbl, (1.0, 0.0, 0.0, 0.8))
                    colors.append((*rgba[:3], 0.9))
                    sizes.append(55)

            ax7.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors,
                s=sizes,
                edgecolors='k',
                linewidths=0.3,
                alpha=0.85
            )
            ax7.set_axis_on()
            ax7.set_xticks([])
            ax7.set_yticks([])

            # Build legend entries for detected components
            if component_colors:
                legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=(0.7, 0.7, 0.7),
                                             markeredgecolor='k',
                                             markersize=8, label='Background'))
                for label_idx, color in component_colors.items():
                    legend_handles.append(Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=color[:3],
                        markeredgecolor='k',
                        markersize=8,
                        label=f'Detection {label_idx}'
                    ))
            else:
                legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=(0.7, 0.7, 0.7),
                                             markeredgecolor='k',
                                             markersize=8, label='Background'))

            ax7.legend(handles=legend_handles, fontsize=9, loc='best', framealpha=0.8)
        else:
            ax7.text(0.5, 0.5, "t-SNE requires at least 3 trajectories",
                     ha='center', va='center', fontsize=10)

    # Row 3: Grid visualization
    if gt_grid is not None and pred_grid is not None:
        # GT Grid
        ax8 = fig.add_subplot(gs[2, 0])
        ax8.imshow(img_display)
        cell_h = img_display.shape[0] / spatial_resolution
        cell_w = img_display.shape[1] / spatial_resolution

        for i in range(spatial_resolution):
            for j in range(spatial_resolution):
                if gt_grid[i, j]:
                    rect = plt.Rectangle((j * cell_w, i * cell_h), cell_w, cell_h,
                                        fill=True, facecolor='lime', edgecolor='green',
                                        linewidth=1.5, alpha=0.4)
                    ax8.add_patch(rect)

        gt_count = int(gt_grid.sum())
        ax8.set_title(f'Ground Truth Grid\n{gt_count} cells ({spatial_resolution}×{spatial_resolution})',
                     fontsize=11, fontweight='bold')
        ax8.axis('off')

        # Predicted Grid
        ax9 = fig.add_subplot(gs[2, 1])
        ax9.imshow(img_display)

        for i in range(spatial_resolution):
            for j in range(spatial_resolution):
                if pred_grid[i, j]:
                    rect = plt.Rectangle((j * cell_w, i * cell_h), cell_w, cell_h,
                                        fill=True, facecolor='orange', edgecolor='red',
                                        linewidth=1.5, alpha=0.4)
                    ax9.add_patch(rect)

        pred_count = int(pred_grid.sum())
        ax9.set_title(f'Predicted Grid\n{pred_count} cells ({spatial_resolution}×{spatial_resolution})',
                     fontsize=11, fontweight='bold')
        ax9.axis('off')

        # Grid Comparison (TP/FP/FN)
        ax10 = fig.add_subplot(gs[2, 2])
        ax10.imshow(img_display, alpha=0.3)

        for i in range(spatial_resolution):
            for j in range(spatial_resolution):
                gt_val = gt_grid[i, j]
                pred_val = pred_grid[i, j]

                if gt_val and pred_val:
                    # True Positive - Green
                    color = 'lime'
                elif not gt_val and pred_val:
                    # False Positive - Red
                    color = 'red'
                elif gt_val and not pred_val:
                    # False Negative - Orange
                    color = 'orange'
                else:
                    # True Negative - skip
                    continue

                rect = plt.Rectangle((j * cell_w, i * cell_h), cell_w, cell_h,
                                    fill=True, facecolor=color, edgecolor='black',
                                    linewidth=1, alpha=0.6)
                ax10.add_patch(rect)

        ax10.set_title(f'Grid Comparison\nTP(green) FP(red) FN(orange)',
                      fontsize=11, fontweight='bold')
        ax10.axis('off')

    # Row 4: Metrics panel - split into 3 columns
    spatial_res = f"{anomaly_map.shape[0]}×{anomaly_map.shape[1]}"

    if detected_pixels > detection_pixel_threshold:
        status = "🔴 DETECTED"
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
        'tcn_autoencoder': 'TCN Autoencoder',
        'tcn_vae': 'TCN VAE',
        'transformer': 'Transformer (Attention)'
    }.get(model_type.lower(), model_type)

    # Column 1: Model Info
    ax_col1 = fig.add_subplot(gs[3, 0])
    ax_col1.axis('off')
    col1_text = f"""MODEL INFO
Model: {model_display}
Method: Reconstruction Error
Resolution: {spatial_res}
Threshold: {threshold_method}
Value: {threshold:.4f}"""
    ax_col1.text(0.5, 0.5, col1_text,
                fontsize=9,
                verticalalignment='center',
                horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8',
                          facecolor='lightblue',
                          alpha=0.8,
                          edgecolor='steelblue',
                          linewidth=2))

    # Column 2: Detection Results
    ax_col2 = fig.add_subplot(gs[3, 1])
    ax_col2.axis('off')
    col2_text = f"""DETECTION
Status: {status}
Pixels: {detected_pixels}/{patch_mask.size}
Rate: {detection_rate:.2f}%
Max Score: {anomaly_map.max():.4f}
Mean: {anomaly_map.mean():.4f}"""
    ax_col2.text(0.5, 0.5, col2_text,
                fontsize=9,
                verticalalignment='center',
                horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8',
                          facecolor=bg_color,
                          alpha=0.9,
                          edgecolor=status_color,
                          linewidth=2))

    # Column 3: Grid Metrics
    ax_col3 = fig.add_subplot(gs[3, 2])
    ax_col3.axis('off')
    if grid_metrics is not None:
        col3_text = f"""GRID METRICS ({spatial_resolution}×{spatial_resolution})
TP:{grid_metrics['tp']:4d} FP:{grid_metrics['fp']:4d}
FN:{grid_metrics['fn']:4d} TN:{grid_metrics['tn']:4d}
Acc: {grid_metrics['accuracy']*100:.1f}%
Prec: {grid_metrics['precision']*100:.1f}%
Rec: {grid_metrics['recall']*100:.1f}%
F1: {grid_metrics['f1']*100:.1f}%"""
        grid_bg_color = 'lightyellow'
        grid_edge_color = 'orange'
    else:
        col3_text = """GRID METRICS
No ground truth
available"""
        grid_bg_color = 'lightgray'
        grid_edge_color = 'gray'

    ax_col3.text(0.5, 0.5, col3_text,
                fontsize=9,
                verticalalignment='center',
                horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8',
                          facecolor=grid_bg_color,
                          alpha=0.8,
                          edgecolor=grid_edge_color,
                          linewidth=2))

    return fig
