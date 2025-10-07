"""
Few-shot Image Patch Detection System

This system detects adversarial patches in images using a Few-shot learning approach
with three phases:

Phase 1: Setup - Load model and extractor
Phase 2: Few-shot Base Learning - Learn normal trajectory characteristics from ImageNet
Phase 3: Few-shot Threshold Adaptation - Set adaptive threshold using domain clean images
Phase 4: Testing - Detect patches in test images

Configuration is managed via Hydra for flexibility and reproducibility.
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from attracter import AttractorLearner
from dataloader import LocalImageDataset
from detector import PatchDetector
from extracter import ActivationExtractor
from trajectory import stack_trajectory
from visualize import visualize_results


def print_phase_header(phase_num: int, phase_name: str):
    """Print formatted phase header"""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {phase_name}")
    print("=" * 80)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for Few-shot patch detection

    Args:
        cfg: Hydra configuration object
    """
    # =========================================================================
    # INITIAL SETUP
    # =========================================================================
    print("=" * 80)
    print("FEW-SHOT IMAGE PATCH DETECTION SYSTEM")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    start_time_total = time.time()

    # =========================================================================
    # PHASE 1: SETUP
    # =========================================================================
    print_phase_header(1, "SETUP")

    # Device configuration
    device_str = f"cuda:{cfg.device.cuda_id}"
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print(f"⚠ Warning: CUDA not available, falling back to CPU")

    print(f"Device: {device}")
    print(f"Workers: {cfg.device.num_workers}")

    # Load ResNet50 model
    print("\nLoading ResNet50 model...")
    model = models.resnet50(pretrained=True)
    model.eval()
    model.to(device)
    print("✓ Model loaded successfully")

    # Setup activation extractor
    print(f"\nSetting up activation extractor...")
    print(f"  Spatial resolution: {cfg.model.spatial_resolution}x{cfg.model.spatial_resolution}")
    print(f"  Feature dimension: {cfg.model.feature_dim}")

    extractor = ActivationExtractor(
        model,
        feature_dim=cfg.model.feature_dim,
        spatial_size=cfg.model.spatial_resolution
    )
    print("✓ Extractor initialized")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # =========================================================================
    # PHASE 2: FEW-SHOT BASE LEARNING
    # =========================================================================
    print_phase_header(2, "FEW-SHOT BASE LEARNING")
    print("Learning normal trajectory characteristics from ImageNet samples")
    print(f"ImageNet path: {cfg.data.imagenet.path}")
    print(f"Number of samples: {cfg.data.imagenet.num_samples}")
    print(f"Batch size: {cfg.data.imagenet.batch_size}")

    phase2_start = time.time()

    try:
        from torchvision.datasets import ImageFolder

        # Load ImageNet dataset
        imagenet_dataset = ImageFolder(root=cfg.data.imagenet.path, transform=transform)
        print(f"\n✓ Found {len(imagenet_dataset)} images in {len(imagenet_dataset.classes)} classes")

        # Sample subset
        num_samples = cfg.data.imagenet.num_samples if cfg.data.imagenet.num_samples > 0 else len(imagenet_dataset)
        num_samples = min(num_samples, len(imagenet_dataset))

        indices = torch.randperm(len(imagenet_dataset))[:num_samples].tolist()
        imagenet_subset = Subset(imagenet_dataset, indices)

        imagenet_loader = DataLoader(
            imagenet_subset,
            batch_size=cfg.data.imagenet.batch_size,
            shuffle=False,
            num_workers=cfg.device.num_workers,
            pin_memory=True
        )

        print(f"✓ Using {num_samples} samples for base learning")

        # Extract embeddings from ImageNet (keep on GPU)
        print("\nExtracting trajectory embeddings...")
        clean_embeddings_gpu = []
        total_processed = 0

        for batch_idx, (imgs, _) in enumerate(imagenet_loader):
            imgs_gpu = imgs.to(device, non_blocking=True)

            # Extract activations
            activations = extractor(imgs_gpu)  # List of [B, C, H, W]

            # Stack all layers into trajectories
            embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]

            # Keep as list of individual embeddings
            for b in range(embeddings_batch.shape[0]):
                clean_embeddings_gpu.append(embeddings_batch[b])  # [H, W, L, C]

            total_processed += imgs.shape[0]

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(imagenet_loader):
                print(f"  Processed: {total_processed}/{num_samples} images "
                      f"({100*total_processed/num_samples:.1f}%)")

        phase2_time = time.time() - phase2_start
        print(f"\n✓ Embedding extraction completed in {phase2_time:.2f}s")
        print(f"  Extracted {len(clean_embeddings_gpu)} trajectory embeddings")

    except FileNotFoundError as e:
        print(f"\n✗ Error: ImageNet path not found: {cfg.data.imagenet.path}")
        print(f"  Please update the path in configs/config.yaml")
        extractor.remove_hooks()
        return
    except Exception as e:
        print(f"\n✗ Error loading ImageNet: {e}")
        extractor.remove_hooks()
        return

    # Learn attractor from ImageNet trajectories
    print("\nLearning normal trajectory characteristics...")
    attractor_learner = AttractorLearner(device=device)
    attractor_learner.fit(clean_embeddings_gpu)

    # Create detector (will be updated after Phase 3)
    detector = None
    print("\n✓ Phase 2 completed successfully")

    # =========================================================================
    # PHASE 3: FEW-SHOT THRESHOLD ADAPTATION
    # =========================================================================
    print_phase_header(3, "FEW-SHOT THRESHOLD ADAPTATION")
    print("Setting adaptive threshold using domain-specific clean images")
    print(f"Clean images path: {cfg.data.domain.clean_path}")
    print(f"Number of samples: {cfg.data.domain.num_samples} (-1 = use all)")
    print(f"Batch size: {cfg.data.domain.batch_size}")

    phase3_start = time.time()

    clean_test_folder = Path(cfg.data.domain.clean_path)

    # Collect scores for each of the 4 metrics separately
    clean_scores_wavelet = []
    clean_scores_stft = []
    clean_scores_hht = []
    clean_scores_sst = []

    if clean_test_folder.exists():
        try:
            clean_test_dataset = LocalImageDataset(clean_test_folder, transform=transform)

            # Sample subset if specified
            if cfg.data.domain.num_samples > 0 and cfg.data.domain.num_samples < len(clean_test_dataset):
                indices = list(range(cfg.data.domain.num_samples))
                clean_test_dataset.image_paths = [clean_test_dataset.image_paths[i] for i in indices]

            clean_test_loader = DataLoader(
                clean_test_dataset,
                batch_size=cfg.data.domain.batch_size,
                shuffle=False,
                num_workers=cfg.device.num_workers,
                pin_memory=True
            )

            print(f"\n✓ Processing {len(clean_test_dataset)} clean images...")

            # Create temporary detector for score calculation (no fusion needed yet)
            temp_detector = PatchDetector(
                attractor_learner,
                device=device,
                fusion_method='voting',
                voting_threshold=1
            )

            # Dummy thresholds (we just want to collect scores)
            dummy_thresholds = {
                'wavelet': 999.0,
                'stft': 999.0,
                'hht': 999.0,
                'sst': 999.0
            }

            for batch_idx, (imgs, img_names) in enumerate(clean_test_loader):
                imgs_gpu = imgs.to(device, non_blocking=True)

                # Extract trajectories
                activations = extractor(imgs_gpu)
                embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]

                # Collect scores for each metric
                for b in range(embeddings_batch.shape[0]):
                    anomaly_map, patch_mask, w_map, st_map, hht_map, sst_map, score_flags = temp_detector.detect(
                        embeddings_batch[b], dummy_thresholds
                    )

                    # Collect max scores for each metric (GPU tensors)
                    clean_scores_wavelet.append(w_map.max())
                    clean_scores_stft.append(st_map.max())
                    clean_scores_hht.append(hht_map.max())
                    clean_scores_sst.append(sst_map.max())

                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(clean_test_loader):
                    processed = min((batch_idx + 1) * cfg.data.domain.batch_size, len(clean_test_dataset))
                    print(f"  Processed: {processed}/{len(clean_test_dataset)} images "
                          f"({100*processed/len(clean_test_dataset):.1f}%)")

            # Stack scores into tensors (Pure PyTorch on GPU)
            scores_dict = {
                'wavelet': torch.stack(clean_scores_wavelet),
                'stft': torch.stack(clean_scores_stft),
                'hht': torch.stack(clean_scores_hht),
                'sst': torch.stack(clean_scores_sst)
            }

            # Compute per-score thresholds based on selected method
            threshold_method = cfg.detection.score_threshold_method
            adaptive_thresholds = {}

            # Helper function to compute threshold for a single score
            def compute_threshold(scores_tensor, method):
                if method == 'mean_std':
                    mean_score = scores_tensor.mean()
                    std_score = scores_tensor.std()
                    return (mean_score + cfg.detection.threshold_multiplier * std_score).item()

                elif method == 'median_mad':
                    median_score = scores_tensor.median()
                    mad = (scores_tensor - median_score).abs().median()
                    return (median_score + cfg.detection.mad_multiplier * mad).item()

                elif method == 'percentile':
                    sorted_scores = torch.sort(scores_tensor)[0]
                    percentile_idx = int(len(sorted_scores) * cfg.detection.percentile / 100.0)
                    percentile_idx = min(percentile_idx, len(sorted_scores) - 1)
                    return sorted_scores[percentile_idx].item()

                else:
                    raise ValueError(f"Unknown threshold method: {method}")

            # Compute threshold for each score
            for score_name, scores_tensor in scores_dict.items():
                adaptive_thresholds[score_name] = compute_threshold(scores_tensor, threshold_method)

            # Formula for display
            if threshold_method == 'mean_std':
                formula = f"mean + {cfg.detection.threshold_multiplier} * std"
            elif threshold_method == 'median_mad':
                formula = f"median + {cfg.detection.mad_multiplier} * MAD"
            elif threshold_method == 'percentile':
                formula = f"{cfg.detection.percentile}th percentile"

            phase3_time = time.time() - phase3_start

            print(f"\n✓ Threshold adaptation completed in {phase3_time:.2f}s")
            print(f"\nPer-Score Adaptive Thresholds:")
            print(f"  Method:  {threshold_method}")
            print(f"  Formula: {formula}")
            print(f"\n  Score Thresholds:")
            for score_name, threshold_value in adaptive_thresholds.items():
                score_tensor = scores_dict[score_name]
                print(f"    {score_name:12s}: {threshold_value:.4f} "
                      f"(range: [{score_tensor.min().item():.4f}, {score_tensor.max().item():.4f}])")

        except FileNotFoundError:
            print(f"\n⚠ Warning: Clean test folder not found: {clean_test_folder}")
            print(f"  Using default thresholds")
            adaptive_thresholds = {
                'wavelet': 2.5, 'stft': 2.5,
                'hht': 2.5, 'sst': 2.5
            }
            threshold_method = 'mean_std'
            formula = 'default'
            phase3_time = 0

        except Exception as e:
            print(f"\n⚠ Warning: Error processing clean images: {e}")
            print(f"  Using default thresholds")
            adaptive_thresholds = {
                'wavelet': 2.5, 'stft': 2.5,
                'hht': 2.5, 'sst': 2.5
            }
            threshold_method = 'mean_std'
            formula = 'default'
            phase3_time = 0
    else:
        print(f"\n⚠ Warning: Clean test folder not found: {clean_test_folder}")
        print(f"  Using default thresholds")
        adaptive_thresholds = {
            'wavelet': 2.5, 'stft': 2.5,
            'hht': 2.5, 'sst': 2.5
        }
        threshold_method = 'mean_std'
        formula = 'default'
        phase3_time = 0

    # Create detector with fusion parameters
    detector = PatchDetector(
        attractor_learner,
        device=device,
        fusion_method=cfg.detection.fusion_method,
        voting_threshold=cfg.detection.voting_threshold,
        score_weights=cfg.detection.score_weights
    )

    print("\n✓ Phase 3 completed successfully")

    # =========================================================================
    # PHASE 4: TESTING
    # =========================================================================
    print_phase_header(4, "TESTING")
    print("Detecting patches in test images")
    print(f"Test images path: {cfg.data.test.patch_path}")
    print(f"Batch size: {cfg.data.test.batch_size}")
    print(f"Fusion method: {cfg.detection.fusion_method}")
    if cfg.detection.fusion_method == 'voting':
        print(f"Voting threshold: {cfg.detection.voting_threshold}/6 scores")
    print(f"Output directory: {cfg.output.dir}")

    phase4_start = time.time()

    patch_folder = Path(cfg.data.test.patch_path)

    if not patch_folder.exists():
        print(f"\n✗ Error: Test folder not found: {patch_folder}")
        print(f"  Please update the path in configs/config.yaml")
        extractor.remove_hooks()
        return

    try:
        patch_dataset = LocalImageDataset(patch_folder, transform=transform)
        patch_loader = DataLoader(
            patch_dataset,
            batch_size=cfg.data.test.batch_size,
            shuffle=False,
            num_workers=cfg.device.num_workers,
            pin_memory=True
        )

        print(f"\n✓ Found {len(patch_dataset)} test images")

        # Create output directory
        output_dir = Path(cfg.output.dir)
        output_dir.mkdir(exist_ok=True)
        print(f"✓ Output directory: {output_dir}")

        # Process test images
        print(f"\nProcessing test images...")
        results = []
        image_counter = 0

        for batch_idx, (imgs, img_names) in enumerate(patch_loader):
            imgs_gpu = imgs.to(device, non_blocking=True)

            # Extract trajectories
            activations = extractor(imgs_gpu)
            embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]

            for b in range(embeddings_batch.shape[0]):
                image_counter += 1
                img = imgs[b]
                img_name = img_names[b]

                # Detect patches with per-score thresholds
                anomaly_map, patch_mask, w_map, st_map, hht_map, sst_map, score_flags = detector.detect(
                    embeddings_batch[b], adaptive_thresholds
                )

                detected_pixels = patch_mask.sum().item()
                max_votes = anomaly_map.max().item()

                is_detected = detected_pixels > cfg.detection.detection_pixel_threshold
                status = "✓ DETECTED" if is_detected else "✗ CLEAN"

                print(f"  [{image_counter}/{len(patch_dataset)}] {img_name:30s} | "
                      f"Votes: {max_votes:.1f} | Pixels: {detected_pixels:4d} | {status}")

                # Store results
                results.append({
                    'image': img,
                    'name': img_name,
                    'anomaly_map': anomaly_map,
                    'patch_mask': patch_mask,
                    'w_map': w_map,
                    'st_map': st_map,
                    'hht_map': hht_map,
                    'sst_map': sst_map,
                    'score_flags': score_flags,
                    'max_votes': max_votes,
                    'detected_pixels': detected_pixels,
                    'is_detected': is_detected
                })

                # Visualize and save
                if cfg.output.save_visualizations:
                    fig = visualize_results(
                        img, anomaly_map, patch_mask, w_map, st_map, hht_map, sst_map,
                        score_flags, img_name, adaptive_thresholds, cfg.detection.detection_pixel_threshold,
                        threshold_method, formula, cfg.detection.fusion_method, cfg.detection.voting_threshold
                    )

                    output_path = output_dir / f'result_{Path(img_name).stem}.png'
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

        phase4_time = time.time() - phase4_start

        print(f"\n✓ Testing completed in {phase4_time:.2f}s")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        extractor.remove_hooks()
        return

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time_total

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Detection statistics
    total_detected = sum(1 for r in results if r['is_detected'])
    detection_rate = 100 * total_detected / len(results) if len(results) > 0 else 0

    print(f"\nDetection Results ({len(results)} images):")
    print(f"  Detected: {total_detected}")
    print(f"  Clean:    {len(results) - total_detected}")
    print(f"  Rate:     {detection_rate:.1f}%")

    print(f"\nVote Statistics:")
    if len(results) > 0:
        votes = [r['max_votes'] for r in results]
        print(f"  Min:  {min(votes):.2f}")
        print(f"  Max:  {max(votes):.2f}")
        print(f"  Mean: {sum(votes)/len(votes):.2f}")

    # Timing breakdown
    print(f"\n{'=' * 80}")
    print("TIMING")
    print(f"{'=' * 80}")
    print(f"  Phase 1 (Setup):              {0:.2f}s")
    print(f"  Phase 2 (Base Learning):      {phase2_time:.2f}s")
    print(f"  Phase 3 (Threshold Adapt):    {phase3_time:.2f}s")
    print(f"  Phase 4 (Testing):            {phase4_time:.2f}s")
    print(f"  {'─' * 78}")
    print(f"  Total:                        {total_time:.2f}s")

    if len(results) > 0:
        print(f"  Avg per test image:           {phase4_time/len(results):.3f}s")

    # Cleanup
    extractor.remove_hooks()

    print("\n" + "=" * 80)
    print("✓ FEW-SHOT PATCH DETECTION COMPLETED")
    print(f"✓ Results saved to: {output_dir}/")
    print("=" * 80)
    print("\nTo run with different settings, use command-line overrides:")
    print("  python test.py data.imagenet.num_samples=500")
    print("  python test.py detection.threshold_multiplier=3.0")
    print("  python test.py model.spatial_resolution=14")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
