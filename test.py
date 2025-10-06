import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

from config import Config
from attracter import AttractorLearner
from dataloader import LocalImageDataset
from detector import PatchDetector
from extracter import ActivationExtractor
from trajectory import stack_trajectory
from visualize import visualize_results

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
            activations = extractor(imgs_gpu)  # List of [B, C, H, W]

            # 모든 레이어를 시간 축으로 스택
            embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]
            
            # Keep as list of individual embeddings
            for b in range(embeddings_batch.shape[0]):
                clean_embeddings_gpu.append(embeddings_batch[b])  # [H, W, L, C]
            
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
    attractor_learner = AttractorLearner(device=device)
    attractor_learner.fit(clean_embeddings_gpu)
    
    # Create detector
    detector = PatchDetector(attractor_learner, device=device)
    
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
                embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]
                
                for b in range(embeddings_batch.shape[0]):
                    anomaly_map, _, _, _ = detector.detect(embeddings_batch[b], threshold=999)
                    # Keep as GPU tensor (scalar)
                    max_score = anomaly_map.max()
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
        embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]
        
        print(f"  Batch {batch_idx+1}/{len(patch_loader)}")
        
        for b in range(embeddings_batch.shape[0]):
            image_counter += 1
            img = imgs[b]
            img_name = img_names[b]
            
            anomaly_map, patch_mask, v_map, s_map = detector.detect(
                embeddings_batch[b], threshold=adaptive_threshold
            )
            
            print(f"    [{image_counter}/{len(patch_dataset)}] {img_name}: "
                f"Max={anomaly_map.max().item():.3f}, Detected={patch_mask.sum().item()}")

            results.append({
                'image': img,
                'name': img_name,
                'anomaly_map': anomaly_map,
                'patch_mask': patch_mask,
                'v_map': v_map,
                's_map': s_map,
                'max_score': anomaly_map.max()
            })
            
            # Visualize
            fig = visualize_results(
                img, anomaly_map, patch_mask, v_map, s_map, img_name, adaptive_threshold
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
        is_detected = result['patch_mask'].sum().item() > Config.DETECTION_PIXEL_THRESHOLD
        if is_detected:
            total_detected += 1
        status = "✓ DETECTED" if is_detected else "✗ MISSED"
        print(f"  {result['name']:30s}: Score={result['max_score'].item():.3f}, {status}")
    
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