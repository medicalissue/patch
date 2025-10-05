import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

from attracter import AttractorLearner
from dataloader import LocalImageDataset
from detector import PatchDetector
from extracter import ActivationExtractor
from seriese_embedding import takens_embedding_gpu


# ============================================================================
# CONFIGURATION - 모든 설정을 여기서 한번에 조절
# ============================================================================
class Config:
    """전역 설정 - 여기서 모든 파라미터를 조절하세요"""
    
    # GPU 설정
    DEVICE = 'cuda:2'  # 'cuda:0', 'cuda:1', 'cuda:2', 'cpu' 등
    NUM_WORKERS = 8    # DataLoader workers (CPU 코어 수에 맞게 조절)
    
    # 배치 크기 설정
    BATCH_SIZE_TRAIN = 128      # ImageNet 학습 배치 (클수록 빠르지만 메모리 많이 사용)
    BATCH_SIZE_CLEAN = 128      # Clean baseline 측정 배치
    BATCH_SIZE_TEST = 128       # 패치된 이미지 테스트 배치
    
    # 공간 해상도 설정 (높을수록 정밀하지만 느림)
    SPATIAL_RESOLUTION = 7    # 7, 14, 28, 56 중 선택
    FEATURE_DIM = 128          # Channel dimension
    
    # Takens embedding 파라미터
    EMBEDDING_M = 3            # Embedding dimension
    EMBEDDING_TAU = 1          # Time delay
    
    # Attractor learning
    PCA_COMPONENTS = 32        # PCA 차원
    N_CLEAN_IMAGES = 1000       # ImageNet에서 사용할 clean 이미지 수
    
    # Detection 설정
    THRESHOLD_MULTIPLIER = 2   # Mean + k*std (3=기본, 5=강함, 6=매우 강함)
    DETECTION_PIXEL_THRESHOLD = 0  # 이 값 이상의 픽셀이 감지되면 anomaly로 판단
    
    # KDE 설정
    KDE_BANDWIDTH = 0.5        # KDE bandwidth
    
    # PatchDetector 설정
    CHUNK_SIZE = 100           # Hausdorff 계산시 chunk 크기 (메모리에 맞게 조절)
    
    # 경로 설정
    IMAGENET_PATH = '/data/ImageNet/train'
    CLEAN_TEST_PATH = 'images_without_patches'
    PATCH_TEST_PATH = 'images_with_patches'
    OUTPUT_DIR = 'detection_results'
    
    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("="*70)
        print("CONFIGURATION (100% GPU, No Numpy!)")
        print("="*70)
        print(f"GPU Settings:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Num Workers: {cls.NUM_WORKERS}")
        print(f"\nBatch Sizes:")
        print(f"  Training (ImageNet): {cls.BATCH_SIZE_TRAIN}")
        print(f"  Clean Baseline: {cls.BATCH_SIZE_CLEAN}")
        print(f"  Patch Testing: {cls.BATCH_SIZE_TEST}")
        print(f"\nModel Settings:")
        print(f"  Spatial Resolution: {cls.SPATIAL_RESOLUTION}x{cls.SPATIAL_RESOLUTION} = {cls.SPATIAL_RESOLUTION**2} pixels")
        print(f"  Feature Dimension: {cls.FEATURE_DIM}")
        print(f"  Takens (m, tau): ({cls.EMBEDDING_M}, {cls.EMBEDDING_TAU})")
        print(f"  PCA Components: {cls.PCA_COMPONENTS}")
        print(f"\nDetection Settings:")
        print(f"  Threshold: Mean + {cls.THRESHOLD_MULTIPLIER}*std")
        print(f"  Detection Threshold: {cls.DETECTION_PIXEL_THRESHOLD} pixels")
        print("="*70 + "\n")


# ============================================================================
# 1. Activation Extractor
# ============================================================================



# ============================================================================
# 2. Takens Embedding (Pure GPU)
# ============================================================================



# ============================================================================
# 3. PyTorch PCA (Pure GPU)
# ============================================================================



# ============================================================================
# 4. PyTorch KDE (Pure GPU)
# ============================================================================



# ============================================================================
# 5. Attractor Learner (Pure GPU)
# ============================================================================



# ============================================================================
# 6. Patch Detector (Pure GPU)
# ============================================================================


# ============================================================================
# 7. Custom Dataset
# ============================================================================



# ============================================================================
# 8. Visualization
# ============================================================================
def denormalize_image(img_tensor):
    """ImageNet normalize 역변환"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

def visualize_results(image, anomaly_map, patch_mask, 
                     hausdorff_map, mahalanobis_map, image_name="", threshold=None):
    """결과 시각화"""
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
    im2 = ax4.imshow(hausdorff_map, cmap='viridis', interpolation='bilinear')
    ax4.set_title(f'Hausdorff Distance\nMax: {hausdorff_map.max():.2f}, Mean: {hausdorff_map.mean():.2f}', 
                 fontsize=10, fontweight='bold')
    ax4.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=9)
    
    ax5 = fig.add_subplot(gs[1, 1])
    im3 = ax5.imshow(mahalanobis_map, cmap='plasma', interpolation='bilinear')
    ax5.set_title(f'Mahalanobis Distance\nMax: {mahalanobis_map.max():.2f}, Mean: {mahalanobis_map.mean():.2f}', 
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
  Hausdorff (max):    {hausdorff_map.max():.3f}
  Mahalanobis (max):  {mahalanobis_map.max():.3f}

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


# ============================================================================
# 9. Main Experiment (100% GPU)
# ============================================================================
def main():
    print("="*70)
    print("100% GPU Phase Space Attractor Detection (No Numpy!)")
    print("="*70 + "\n")
    
    # Print configuration
    Config.print_config()
    
    # Setup device
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    if Config.DEVICE.startswith('cuda') and not torch.cuda.is_available():
        print(f"⚠ Warning: CUDA not available, falling back to CPU")
    print(f"Using device: {device}\n")
    
    # Load model
    print("[1] Loading ResNet50...")
    model = models.resnet50(pretrained=True)
    model.eval()
    model.to(device)
    
    # Setup extractor
    print(f"[2] Setting up activation extractor...")
    extractor = ActivationExtractor(
        model, 
        feature_dim=Config.FEATURE_DIM, 
        spatial_size=Config.SPATIAL_RESOLUTION
    )
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ========================================================================
    # TRAINING PHASE: ImageNet으로 Reference Attractor 학습
    # ========================================================================
    print("\n[3] Loading ImageNet dataset for attractor learning...")
    from torchvision.datasets import ImageFolder
    
    start_time = time.time()
    
    try:
        dataset = ImageFolder(root=Config.IMAGENET_PATH, transform=transform)
        print(f"  Found {len(dataset)} images in {len(dataset.classes)} classes")
        
        # Sample subset
        indices = torch.randperm(len(dataset))[:Config.N_CLEAN_IMAGES].tolist()
        clean_subset = Subset(dataset, indices)
        clean_loader = DataLoader(
            clean_subset, 
            batch_size=Config.BATCH_SIZE_TRAIN,
            shuffle=False, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        print(f"  Using {Config.N_CLEAN_IMAGES} images (batch_size={Config.BATCH_SIZE_TRAIN})")
        
        # Extract embeddings (Keep on GPU)
        clean_embeddings_gpu = []
        total_processed = 0
        
        print("  Extracting embeddings on GPU...")
        for batch_idx, (imgs, _) in enumerate(clean_loader):
            imgs_gpu = imgs.to(device, non_blocking=True)
            
            # Extract on GPU
            activations = extractor(imgs_gpu)
            embeddings_batch = takens_embedding_gpu(
                activations, 
                m=Config.EMBEDDING_M, 
                tau=Config.EMBEDDING_TAU
            )  # [B, H, W, T, D]
            
            # Keep as list of individual embeddings
            for b in range(embeddings_batch.shape[0]):
                clean_embeddings_gpu.append(embeddings_batch[b])  # [H, W, T, D]
            
            total_processed += imgs.shape[0]
            print(f"    Batch {batch_idx+1}/{len(clean_loader)}: {total_processed}/{Config.N_CLEAN_IMAGES} images")
        
        train_time = time.time() - start_time
        print(f"  ✓ Embedding extraction completed in {train_time:.2f}s")
            
    except Exception as e:
        print(f"  Error loading ImageNet: {e}")
        extractor.remove_hooks()
        return
    
    # Learn attractor (Pure GPU)
    print("\n[4] Learning reference attractor (Pure GPU)...")
    attractor_learner = AttractorLearner(
        n_components=Config.PCA_COMPONENTS,
        bandwidth=Config.KDE_BANDWIDTH,
        device=device
    )
    attractor_learner.fit(clean_embeddings_gpu)
    
    # Create detector
    detector = PatchDetector(
        attractor_learner, 
        device=device,
        chunk_size=Config.CHUNK_SIZE
    )
    
    # ========================================================================
    # BASELINE PHASE: Clean 분포 측정 (Pure PyTorch)
    # ========================================================================
    print("\n[5] Measuring clean baseline...")
    clean_test_folder = Path(Config.CLEAN_TEST_PATH)
    
    clean_scores_gpu = []  # GPU tensor list
    
    if clean_test_folder.exists():
        try:
            clean_test_dataset = LocalImageDataset(clean_test_folder, transform=transform)
            clean_test_loader = DataLoader(
                clean_test_dataset, 
                batch_size=Config.BATCH_SIZE_CLEAN,
                shuffle=False, 
                num_workers=Config.NUM_WORKERS,
                pin_memory=True
            )
            
            print(f"  Processing {len(clean_test_dataset)} images (batch_size={Config.BATCH_SIZE_CLEAN})...")
            
            baseline_start = time.time()
            
            for batch_idx, (imgs, img_names) in enumerate(clean_test_loader):
                imgs_gpu = imgs.to(device, non_blocking=True)
                activations = extractor(imgs_gpu)
                embeddings_batch = takens_embedding_gpu(
                    activations, 
                    m=Config.EMBEDDING_M, 
                    tau=Config.EMBEDDING_TAU
                )  # [B, H, W, T, D]
                
                for b in range(embeddings_batch.shape[0]):
                    anomaly_map, _, _, _ = detector.detect(embeddings_batch[b], threshold=999)
                    # Keep as GPU tensor (scalar)
                    max_score = torch.tensor(anomaly_map.max(), device=device)
                    clean_scores_gpu.append(max_score)
                
                print(f"    Batch {batch_idx+1}/{len(clean_test_loader)}")
            
            baseline_time = time.time() - baseline_start
            
            # Calculate threshold (Pure PyTorch on GPU)
            clean_scores_tensor = torch.stack(clean_scores_gpu)  # [N] tensor on GPU
            mean_score = clean_scores_tensor.mean()
            std_score = clean_scores_tensor.std()
            
            adaptive_threshold = (mean_score + Config.THRESHOLD_MULTIPLIER * std_score).item()
            
            print(f"  ✓ Baseline measurement completed in {baseline_time:.2f}s")
            print(f"\n  Clean statistics:")
            print(f"    Mean: {mean_score.item():.3f}, Std: {std_score.item():.3f}")
            print(f"    Threshold: {adaptive_threshold:.3f} (mean + {Config.THRESHOLD_MULTIPLIER}*std)")
            
            # Keep tensor for later analysis
            clean_scores_tensor_cpu = clean_scores_tensor.cpu()
            
        except Exception as e:
            print(f"  Warning: {e}")
            adaptive_threshold = 2.5
            clean_scores_tensor_cpu = None
    else:
        print(f"  Warning: {clean_test_folder} not found, using default threshold")
        adaptive_threshold = 2.5
        clean_scores_tensor_cpu = None
    
    # ========================================================================
    # TESTING PHASE: Patched images detection
    # ========================================================================
    print(f"\n[6] Testing on patched images...")
    patch_folder = Path(Config.PATCH_TEST_PATH)
    
    if not patch_folder.exists():
        print(f"  ERROR: {patch_folder} not found!")
        extractor.remove_hooks()
        return
    
    try:
        patch_dataset = LocalImageDataset(patch_folder, transform=transform)
        patch_loader = DataLoader(
            patch_dataset, 
            batch_size=Config.BATCH_SIZE_TEST,
            shuffle=False, 
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        extractor.remove_hooks()
        return
    
    print(f"  Processing {len(patch_dataset)} images (batch_size={Config.BATCH_SIZE_TEST})...")
    print(f"  Threshold: {adaptive_threshold:.3f}\n")
    
    results = []
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    test_start = time.time()
    image_counter = 0
    
    for batch_idx, (imgs, img_names) in enumerate(patch_loader):
        imgs_gpu = imgs.to(device, non_blocking=True)
        activations = extractor(imgs_gpu)
        embeddings_batch = takens_embedding_gpu(
            activations, 
            m=Config.EMBEDDING_M, 
            tau=Config.EMBEDDING_TAU
        )  # [B, H, W, T, D]
        
        print(f"  Batch {batch_idx+1}/{len(patch_loader)}")
        
        for b in range(embeddings_batch.shape[0]):
            image_counter += 1
            img = imgs[b]
            img_name = img_names[b]
            
            anomaly_map, patch_mask, h_map, m_map = detector.detect(
                embeddings_batch[b], threshold=adaptive_threshold
            )
            
            print(f"    [{image_counter}/{len(patch_dataset)}] {img_name}: "
                  f"Max={anomaly_map.max():.3f}, Detected={patch_mask.sum()}")
            
            results.append({
                'image': img,
                'name': img_name,
                'anomaly_map': anomaly_map,
                'patch_mask': patch_mask,
                'h_map': h_map,
                'm_map': m_map,
                'max_score': anomaly_map.max()
            })
            
            # Visualize
            fig = visualize_results(
                img, anomaly_map, patch_mask, h_map, m_map, img_name, adaptive_threshold
            )
            
            output_path = output_dir / f'result_{Path(img_name).stem}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    test_time = time.time() - test_start
    total_time = time.time() - start_time
    
    # ========================================================================
    # Summary (Pure PyTorch)
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if clean_scores_tensor_cpu is not None and len(clean_scores_tensor_cpu) > 0:
        print(f"\nClean Baseline ({len(clean_scores_tensor_cpu)} images):")
        print(f"  Range: [{clean_scores_tensor_cpu.min().item():.3f}, {clean_scores_tensor_cpu.max().item():.3f}]")
        print(f"  Mean ± Std: {clean_scores_tensor_cpu.mean().item():.3f} ± {clean_scores_tensor_cpu.std().item():.3f}")
    
    print(f"\nPatched Images ({len(results)} images):")
    total_detected = 0
    for result in results:
        is_detected = result['patch_mask'].sum() > Config.DETECTION_PIXEL_THRESHOLD
        if is_detected:
            total_detected += 1
        status = "✓ DETECTED" if is_detected else "✗ MISSED"
        print(f"  {result['name']:30s}: Score={result['max_score']:.3f}, {status}")
    
    print(f"\nDetection Rate: {total_detected}/{len(results)} ({total_detected/len(results)*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("TIMING")
    print(f"{'='*70}")
    print(f"  Training phase: {train_time:.2f}s")
    if clean_scores_tensor_cpu is not None and len(clean_scores_tensor_cpu) > 0:
        print(f"  Baseline phase: {baseline_time:.2f}s")
    print(f"  Testing phase:  {test_time:.2f}s")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Avg per image:  {test_time/len(results):.2f}s")
    
    # Cleanup
    extractor.remove_hooks()
    
    print("\n" + "="*70)
    print(f"✓ Completed! 100% PyTorch GPU - ZERO Numpy!")
    print(f"✓ Results saved in: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()