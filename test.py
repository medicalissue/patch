"""
Model-based Image Patch Detection System

This system detects adversarial patches in images using neural network models
trained on clean trajectories with three phases:

Phase 1: Model Training - Train anomaly detection model on clean ImageNet images
Phase 2 (Optional): Domain Adaptation - LoRA-based fine-tuning on domain clean images
Phase 3: Testing - Detect patches in test images using reconstruction error

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

from attracter import ModelTrainer
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
    Main function for model-based patch detection

    Args:
        cfg: Hydra configuration object
    """
    # =========================================================================
    # INITIAL SETUP
    # =========================================================================
    print("=" * 80)
    print("MODEL-BASED IMAGE PATCH DETECTION SYSTEM")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    start_time_total = time.time()

    # =========================================================================
    # PHASE 0: SETUP
    # =========================================================================
    print_phase_header(0, "SETUP")

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

    trajectory_cfg = getattr(cfg.model, 'trajectory', None)
    normalize_steps = True
    normalization_eps = 1e-6
    depth_scaling = None

    if trajectory_cfg is not None:
        normalize_steps = getattr(trajectory_cfg, 'normalize_steps', True)
        normalization_eps = getattr(trajectory_cfg, 'normalization_eps', 1e-6)
        depth_scaling_cfg = getattr(trajectory_cfg, 'depth_scaling', None)
        if depth_scaling_cfg is not None and getattr(depth_scaling_cfg, 'enabled', False):
            depth_scaling = (
                getattr(depth_scaling_cfg, 'start', 1.0),
                getattr(depth_scaling_cfg, 'end', 1.0),
            )

    extractor = ActivationExtractor(
        model,
        feature_dim=cfg.model.feature_dim,
        spatial_size=cfg.model.spatial_resolution,
        normalize_steps=normalize_steps,
        normalization_eps=normalization_eps,
        depth_scaling=depth_scaling,
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
    # PHASE 1: MODEL TRAINING
    # =========================================================================
    print_phase_header(1, "MODEL TRAINING")
    print("Training anomaly detection model on clean ImageNet images")
    print(f"Model type: {cfg.model.type}")
    print(f"ImageNet path: {cfg.data.imagenet.path}")
    print(f"Number of samples: {cfg.data.imagenet.num_samples}")
    print(f"Batch size: {cfg.data.imagenet.batch_size}")
    print(f"Training epochs: {cfg.data.imagenet.num_epochs}")

    phase1_start = time.time()

    # Create model trainer
    model_trainer = ModelTrainer(
        model_type=cfg.model.type,
        input_dim=cfg.model.feature_dim,
        device=device,
        cfg=cfg,
    )

    # Check for saved weights
    use_saved_weights = cfg.model.phase1.load_weights
    save_weights = cfg.model.phase1.save_weights
    weights_dir = Path(cfg.model.phase1.weights_dir)

    # Generate weights filename
    weights_filename = ModelTrainer.get_weights_filename(
        cfg.model.type,
        cfg.data.imagenet.path,
        cfg.data.imagenet.num_samples,
        cfg.model.spatial_resolution,
        cfg.model.feature_dim,
        cfg.model.hidden_dim,
        cfg.model.latent_dim,
        cfg.model.num_layers
    )
    weights_path = weights_dir / weights_filename

    # Try to load saved weights
    model_loaded = False
    if use_saved_weights and weights_path.exists():
        try:
            print(f"\n✓ Found saved model weights: {weights_path}")
            print(f"  Loading model weights...")
            model_trainer.load_weights(weights_path)
            model_loaded = True
            phase1_time = time.time() - phase1_start
            print(f"\n✓ Model loaded in {phase1_time:.2f}s")
        except Exception as e:
            print(f"\n⚠ Warning: Failed to load weights: {e}")
            print(f"  Will train model from scratch...")
            model_loaded = False

    if not model_loaded:
        print(f"\n  Training model from scratch...")

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

            print(f"✓ Using {num_samples} samples for training")

            # Train model in streaming mode (no memory accumulation!)
            print("\nTraining model in streaming mode (memory-efficient)...")
            use_wandb = cfg.experiment.use_wandb if hasattr(cfg, 'experiment') else False
            wandb_config = None
            if use_wandb and hasattr(cfg, 'experiment'):
                wandb_config = {
                    'project': cfg.experiment.wandb_project,
                    'entity': cfg.experiment.wandb_entity,
                    'name': f'train-{cfg.model.type}'
                }
            model_trainer.train_streaming(
                imagenet_loader,
                extractor,
                num_epochs=cfg.data.imagenet.num_epochs,
                use_wandb=use_wandb,
                wandb_config=wandb_config
            )

            phase1_time = time.time() - phase1_start
            print(f"\n✓ Model training completed in {phase1_time:.2f}s")

            # Save model weights
            if save_weights:
                print(f"\nSaving model weights...")
                model_trainer.save_weights(weights_path)

        except FileNotFoundError as e:
            print(f"\n✗ Error: ImageNet path not found: {cfg.data.imagenet.path}")
            print(f"  Please update the path in configs/config.yaml")
            extractor.remove_hooks()
            return
        except Exception as e:
            print(f"\n✗ Error during model training: {e}")
            import traceback
            traceback.print_exc()
            extractor.remove_hooks()
            return

    print("\n✓ Phase 1 completed successfully")

    # =========================================================================
    # PHASE 2: DOMAIN ADAPTATION (OPTIONAL)
    # =========================================================================
    if cfg.domain_adaptation.enabled:
        print_phase_header(2, "DOMAIN ADAPTATION WITH LoRA")
        print("Fine-tuning model on domain-specific clean images using LoRA")
        print(f"Clean images path: {cfg.data.domain.clean_path}")
        print(f"Number of samples: {cfg.data.domain.num_samples} (-1 = use all)")
        print(f"Batch size: {cfg.data.domain.batch_size}")
        print(f"Adaptation epochs: {cfg.data.domain.num_epochs}")

        phase2_start = time.time()

        clean_domain_folder = Path(cfg.data.domain.clean_path)

        if clean_domain_folder.exists():
            try:
                clean_domain_dataset = LocalImageDataset(clean_domain_folder, transform=transform)

                # Sample subset if specified
                if cfg.data.domain.num_samples > 0 and cfg.data.domain.num_samples < len(clean_domain_dataset):
                    indices = list(range(cfg.data.domain.num_samples))
                    clean_domain_dataset.image_paths = [clean_domain_dataset.image_paths[i] for i in indices]

                clean_domain_loader = DataLoader(
                    clean_domain_dataset,
                    batch_size=cfg.data.domain.batch_size,
                    shuffle=False,
                    num_workers=cfg.device.num_workers,
                    pin_memory=True
                )

                print(f"\n✓ Processing {len(clean_domain_dataset)} clean domain images...")

                # Check for saved LoRA weights
                use_saved_lora = cfg.domain_adaptation.phase2.load_weights
                save_lora = cfg.domain_adaptation.phase2.save_weights
                lora_weights_dir = Path(cfg.domain_adaptation.phase2.weights_dir)

                # Generate LoRA weights filename
                lora_weights_filename = ModelTrainer.get_lora_weights_filename(
                    cfg.model.type,
                    cfg.data.domain.clean_path,
                    cfg.data.domain.num_samples,
                    cfg.domain_adaptation.lora.rank,
                    cfg.domain_adaptation.lora.alpha
                )
                lora_weights_path = lora_weights_dir / lora_weights_filename

                # Try to load saved LoRA weights
                lora_loaded = False
                if use_saved_lora and lora_weights_path.exists():
                    try:
                        print(f"\n✓ Found saved LoRA weights: {lora_weights_path}")
                        print(f"  Loading LoRA weights...")
                        model_trainer.load_lora_weights(lora_weights_path)
                        lora_loaded = True
                        phase2_time = time.time() - phase2_start
                        print(f"\n✓ LoRA weights loaded in {phase2_time:.2f}s")
                    except Exception as e:
                        print(f"\n⚠ Warning: Failed to load LoRA weights: {e}")
                        print(f"  Will train LoRA from scratch...")
                        lora_loaded = False

                if not lora_loaded:
                    # Apply LoRA and train in streaming mode (no memory accumulation!)
                    print("\nTraining LoRA in streaming mode (memory-efficient)...")
                    use_wandb = cfg.experiment.use_wandb if hasattr(cfg, 'experiment') else False
                    wandb_config = None
                    if use_wandb and hasattr(cfg, 'experiment'):
                        wandb_config = {
                            'project': cfg.experiment.wandb_project,
                            'entity': cfg.experiment.wandb_entity,
                            'name': f'lora-{cfg.model.type}'
                        }
                    model_trainer.adapt_with_lora_streaming(
                        clean_domain_loader,
                        extractor,
                        lora_cfg=cfg.domain_adaptation.lora,
                        num_epochs=cfg.data.domain.num_epochs,
                        use_wandb=use_wandb,
                        wandb_config=wandb_config
                    )

                    phase2_time = time.time() - phase2_start
                    print(f"\n✓ Domain adaptation completed in {phase2_time:.2f}s")

                    # Save LoRA weights
                    if save_lora:
                        print(f"\nSaving LoRA weights...")
                        model_trainer.save_lora_weights(lora_weights_path)

            except FileNotFoundError:
                print(f"\n⚠ Warning: Clean domain folder not found: {clean_domain_folder}")
                print(f"  Skipping domain adaptation")
                phase2_time = 0

            except Exception as e:
                print(f"\n⚠ Warning: Error during domain adaptation: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Skipping domain adaptation")
                phase2_time = 0
        else:
            print(f"\n⚠ Warning: Clean domain folder not found: {clean_domain_folder}")
            print(f"  Skipping domain adaptation")
            phase2_time = 0

        print("\n✓ Phase 2 completed successfully")
    else:
        print_phase_header(2, "DOMAIN ADAPTATION (SKIPPED)")
        print("Domain adaptation is disabled in configuration")
        phase2_time = 0

    # Create detector
    detector = PatchDetector(
        model_trainer,
        device=device,
        detection_cfg=cfg.detection
    )

    # =========================================================================
    # PHASE 3: TESTING
    # =========================================================================
    print_phase_header(3, "TESTING")
    print("Detecting patches in test images using reconstruction error")
    print(f"Test images path: {cfg.data.test.patch_path}")
    print(f"Batch size: {cfg.data.test.batch_size}")
    print(f"Output directory: {cfg.output.dir}")

    phase3_start = time.time()

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

                # Detect patches using model reconstruction error
                anomaly_map, patch_mask, threshold = detector.detect(embeddings_batch[b])

                detected_pixels = patch_mask.sum().item()
                max_score = anomaly_map.max().item()

                is_detected = detected_pixels > cfg.detection.detection_pixel_threshold
                status = "✓ DETECTED" if is_detected else "✗ CLEAN"

                print(f"  [{image_counter}/{len(patch_dataset)}] {img_name:30s} | "
                      f"Max score: {max_score:.3f} | Threshold: {threshold:.3f} | "
                      f"Pixels: {detected_pixels:4d} | {status}")

                # Store results
                results.append({
                    'image': img,
                    'name': img_name,
                    'anomaly_map': anomaly_map,
                    'patch_mask': patch_mask,
                    'max_score': max_score,
                    'threshold': threshold,
                    'detected_pixels': detected_pixels,
                    'is_detected': is_detected
                })

                # Visualize and save (first item per batch)
                if cfg.output.save_visualizations:
                    if not batch_visualized:
                        threshold_method_name = str(cfg.detection.threshold_method)
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
                            image_name=img_name,
                            threshold=threshold,
                            detection_pixel_threshold=cfg.detection.detection_pixel_threshold,
                            threshold_method=threshold_method_name,
                            threshold_formula=threshold_formula,
                            model_type=cfg.model.type
                        )

                        viz_filename = f"batch{batch_idx:04d}_{Path(img_name).stem}.png"
                        fig.savefig(visualization_dir / viz_filename, bbox_inches='tight')
                        plt.close(fig)
                        batch_visualized = True

        phase3_time = time.time() - phase3_start

        print(f"\n✓ Testing completed in {phase3_time:.2f}s")

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
    print(f"  Phase 0 (Setup):              {0:.2f}s")
    print(f"  Phase 1 (Model Training):     {phase1_time:.2f}s")
    print(f"  Phase 2 (Domain Adaptation):  {phase2_time:.2f}s")
    print(f"  Phase 3 (Testing):            {phase3_time:.2f}s")
    print(f"  {'─' * 78}")
    print(f"  Total:                        {total_time:.2f}s")

    if len(results) > 0:
        print(f"  Avg per test image:           {phase3_time/len(results):.3f}s")

    # Cleanup
    extractor.remove_hooks()

    print("\n" + "=" * 80)
    print("✓ MODEL-BASED PATCH DETECTION COMPLETED")
    print(f"✓ Results saved to: {output_dir}/")
    print("=" * 80)
    print("\nTo run with different settings, use command-line overrides:")
    print(f"  python test.py model.type=vae")
    print(f"  python test.py model.type=transformer")
    print(f"  python test.py domain_adaptation.enabled=true")
    print(f"  python test.py data.imagenet.num_epochs=20")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
