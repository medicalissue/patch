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
from tqdm import tqdm

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

    # Check for cached attractor
    attractor_learner = AttractorLearner(device=device)
    use_cache = cfg.data.attractor.use_cache
    force_recompute = cfg.data.attractor.force_recompute
    cache_dir = Path(cfg.data.attractor.cache_dir)

    # Generate cache filename based on configuration
    cache_filename = AttractorLearner.get_cache_filename(
        cfg.data.imagenet.path,
        cfg.data.imagenet.num_samples,
        cfg.model.spatial_resolution,
        cfg.model.feature_dim
    )
    cache_path = cache_dir / cache_filename

    # Try to load cached attractor
    attractor_loaded = False
    if use_cache and not force_recompute and cache_path.exists():
        try:
            print(f"\n✓ Found cached attractor: {cache_path}")
            print(f"  Loading cached attractor...")
            attractor_learner.load(cache_path)
            attractor_loaded = True
            phase2_time = time.time() - phase2_start
            print(f"\n✓ Attractor loaded in {phase2_time:.2f}s")
        except Exception as e:
            print(f"\n⚠ Warning: Failed to load cache: {e}")
            print(f"  Will recompute attractor...")
            attractor_loaded = False

    if not attractor_loaded:
        print(f"\n  Computing attractor from scratch...")

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

            # Extract embeddings and incrementally update attractor
            print("\nExtracting trajectory embeddings and updating attractor...")

            for imgs, _ in tqdm(imagenet_loader, desc="  Learning attractor", total=len(imagenet_loader)):
                imgs_gpu = imgs.to(device, non_blocking=True)

                # Extract activations
                activations = extractor(imgs_gpu)  # List of [B, C, H, W]

                # Stack all layers into trajectories
                embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]

                # Convert batch to list of individual embeddings
                batch_embeddings = [embeddings_batch[b] for b in range(embeddings_batch.shape[0])]

                # Incrementally update attractor with this batch
                attractor_learner.partial_fit(batch_embeddings)

            total_processed = num_samples

            phase2_time = time.time() - phase2_start
            print(f"\n✓ Embedding extraction completed in {phase2_time:.2f}s")
            print(f"  Processed {total_processed} images incrementally")

        except FileNotFoundError as e:
            print(f"\n✗ Error: ImageNet path not found: {cfg.data.imagenet.path}")
            print(f"  Please update the path in configs/config.yaml")
            extractor.remove_hooks()
            return
        except Exception as e:
            print(f"\n✗ Error loading ImageNet: {e}")
            extractor.remove_hooks()
            return

        # Finalize attractor statistics
        print("\nFinalizing normal trajectory characteristics...")
        attractor_learner.finalize()

        # Save attractor to cache
        if use_cache:
            print(f"\nSaving attractor to cache...")
            attractor_learner.save(cache_path)

    # Create detector (will be updated after Phase 3)
    detector = None
    print("\n✓ Phase 2 completed successfully")

    # =========================================================================
    # PHASE 3: DOMAIN ADAPTATION
    # =========================================================================
    print_phase_header(3, "DOMAIN ADAPTATION")
    print("Adapting ImageNet statistics to domain distribution")
    print(f"Clean images path: {cfg.data.domain.clean_path}")
    print(f"Number of samples: {cfg.data.domain.num_samples} (-1 = use all)")
    print(f"Batch size: {cfg.data.domain.batch_size}")

    phase3_start = time.time()

    clean_test_folder = Path(cfg.data.domain.clean_path)
    domain_stats = None

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

            print(f"\n✓ Processing {len(clean_test_dataset)} clean images for domain adaptation...")

            # Extract domain embeddings
            domain_embeddings = []
            for batch_idx, (imgs, img_names) in enumerate(tqdm(clean_test_loader, desc="  Extracting domain embeddings")):
                imgs_gpu = imgs.to(device, non_blocking=True)
                activations = extractor(imgs_gpu)
                embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]

                for b in range(embeddings_batch.shape[0]):
                    domain_embeddings.append(embeddings_batch[b])

            # Compute domain statistics
            domain_stats = attractor_learner.adapt_to_domain(domain_embeddings)

            phase3_time = time.time() - phase3_start
            print(f"\n✓ Domain adaptation completed in {phase3_time:.2f}s")

        except FileNotFoundError:
            print(f"\n⚠ Warning: Clean test folder not found: {clean_test_folder}")
            print(f"  Skipping domain adaptation")
            domain_stats = None
            phase3_time = 0

        except Exception as e:
            print(f"\n⚠ Warning: Error during domain adaptation: {e}")
            print(f"  Skipping domain adaptation")
            domain_stats = None
            phase3_time = 0
    else:
        print(f"\n⚠ Warning: Clean test folder not found: {clean_test_folder}")
        print(f"  Skipping domain adaptation")
        domain_stats = None
        phase3_time = 0

    # Create detector with domain adaptation
    if domain_stats is not None:
        detector = PatchDetector(
            attractor_learner,
            domain_stats=domain_stats,
            device=device,
            detection_cfg=cfg.detection
        )
    else:
        print(f"\n⚠ Error: Domain adaptation required but failed")
        print(f"  Please provide valid domain clean images")
        extractor.remove_hooks()
        return

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

        visualization_dir = output_dir / "visualize_results"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Visualization directory: {visualization_dir}")

        # Process test images
        print(f"\nProcessing test images...")
        results = []
        image_counter = 0

        for batch_idx, (imgs, img_names) in enumerate(
                tqdm(patch_loader, desc="  Detecting patches"), start=1):
            batch_visualized = False
            imgs_gpu = imgs.to(device, non_blocking=True)

            # Extract trajectories
            activations = extractor(imgs_gpu)
            embeddings_batch = stack_trajectory(activations)  # [B, H, W, L, C]

            for b in range(embeddings_batch.shape[0]):
                image_counter += 1
                img = imgs[b]
                img_name = img_names[b]

                # Detect patches with domain-adapted Mahalanobis distance and per-metric voting
                (anomaly_map,
                 patch_mask,
                 w_map,
                 st_map,
                 hht_map,
                 sst_map,
                 thresholds,
                 score_flags) = detector.detect(embeddings_batch[b])

                detected_pixels = patch_mask.sum().item()
                max_score = anomaly_map.max().item()

                is_detected = detected_pixels > cfg.detection.detection_pixel_threshold
                status = "✓ DETECTED" if is_detected else "✗ CLEAN"

                print(f"  [{image_counter}/{len(patch_dataset)}] {img_name:30s} | "
                      f"Max fusion: {max_score:.3f} | Pixels: {detected_pixels:4d} | {status}")

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
                    'max_score': max_score,
                    'detected_pixels': detected_pixels,
                    'is_detected': is_detected,
                    'thresholds': thresholds
                })

                # Visualize and save (first item per batch with configured thresholds)
                if cfg.output.save_visualizations:
                    if not batch_visualized:
                        threshold_method_name = str(cfg.detection.score_threshold_method)
                        if threshold_method_name == "mean_std":
                            threshold_formula = f"Threshold = mean + {cfg.detection.threshold_multiplier}*std"
                        elif threshold_method_name == "median_mad":
                            threshold_formula = f"Threshold = median + {cfg.detection.mad_multiplier}*MAD"
                        else:
                            threshold_formula = f"Threshold = top {cfg.detection.percentile}% percentile"

                        fig = visualize_results(
                            img,
                            anomaly_map,
                            patch_mask,
                            w_map,
                            st_map,
                            hht_map,
                            sst_map,
                            score_flags=score_flags,
                            image_name=img_name,
                            thresholds=thresholds,
                            detection_pixel_threshold=cfg.detection.detection_pixel_threshold,
                            threshold_method=threshold_method_name,
                            threshold_formula=threshold_formula,
                            fusion_method=cfg.detection.fusion_method,
                            voting_threshold=cfg.detection.voting_threshold
                        )

                        viz_filename = f"batch{batch_idx:04d}_{Path(img_name).stem}.png"
                        fig.savefig(visualization_dir / viz_filename, bbox_inches='tight')
                        plt.close(fig)
                        batch_visualized = True

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

    print(f"\nScore Statistics:")
    if len(results) > 0:
        scores = [r['max_score'] for r in results]
        print(f"  Min:  {min(scores):.3f}")
        print(f"  Max:  {max(scores):.3f}")
        print(f"  Mean: {sum(scores)/len(scores):.3f}")

    # Timing breakdown
    print(f"\n{'=' * 80}")
    print("TIMING")
    print(f"{'=' * 80}")
    print(f"  Phase 1 (Setup):              {0:.2f}s")
    print(f"  Phase 2 (Base Learning):      {phase2_time:.2f}s")
    print(f"  Phase 3 (Domain Adapt):       {phase3_time:.2f}s")
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
    print("  python test.py data.domain.num_samples=100")
    print("  python test.py model.spatial_resolution=14")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
