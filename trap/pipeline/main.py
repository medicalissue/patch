"""
Model-based Image Patch Detection System

This system detects adversarial patches in images using neural network models
trained on clean trajectories with three phases:

Phase 1: Model Training - Train anomaly detection model on clean ImageNet images
Phase 2 (Optional): Domain Adaptation - LoRA-based fine-tuning on domain clean images
Phase 3: Testing - Detect patches in test images using reconstruction error

Configuration is managed via Hydra for flexibility and reproducibility.
"""

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from tqdm import tqdm

from trap.data import LocalImageDataset
from trap.detection import PatchDetector
from trap.evaluation import compute_grid_metrics, compute_ground_truth_grid
from trap.features import ActivationExtractor, stack_trajectory
from trap.training import ModelTrainer
from trap.visualization import visualize_results


def load_backbone_model(backbone_name, pretrained=True):
    """
    Load backbone model for feature extraction

    Args:
        backbone_name: Name of the backbone (e.g., 'resnet50', 'convnext_tiny')
        pretrained: Whether to load pretrained weights

    Returns:
        model: Loaded backbone model
    """
    backbone_name = backbone_name.lower()

    # ResNet family
    if backbone_name == 'resnet18':
        model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'resnet34':
        model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    elif backbone_name == 'resnet101':
        model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
    elif backbone_name == 'resnet152':
        model = models.resnet152(weights='IMAGENET1K_V2' if pretrained else None)

    # ConvNeXt family
    elif backbone_name == 'convnext_tiny':
        model = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'convnext_small':
        model = models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'convnext_base':
        model = models.convnext_base(weights='IMAGENET1K_V1' if pretrained else None)

    # MobileNet V3 family
    elif backbone_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None)

    # EfficientNet family
    elif backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)

    # Vision Transformer
    elif backbone_name == 'vit_b_16':
        model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'vit_b_32':
        model = models.vit_b_32(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'vit_l_16':
        model = models.vit_l_16(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'vit_l_32':
        model = models.vit_l_32(weights='IMAGENET1K_V1' if pretrained else None)
    elif backbone_name == 'vit_h_14':
        model = models.vit_h_14(weights='IMAGENET1K_V1' if pretrained else None)

    # DeiT family
    elif backbone_name == 'deit_tiny_patch16_224':
        model = models.deit_tiny_patch16_224(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'deit_small_patch16_224':
        model = models.deit_small_patch16_224(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'deit_base_patch16_224':
        model = models.deit_base_patch16_224(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'deit_base_patch16_384':
        model = models.deit_base_patch16_384(weights='DEFAULT' if pretrained else None)

    # Swin Transformer family
    elif backbone_name == 'swin_t':
        model = models.swin_t(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'swin_s':
        model = models.swin_s(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'swin_b':
        model = models.swin_b(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'swin_v2_t':
        model = models.swin_v2_t(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'swin_v2_s':
        model = models.swin_v2_s(weights='DEFAULT' if pretrained else None)
    elif backbone_name == 'swin_v2_b':
        model = models.swin_v2_b(weights='DEFAULT' if pretrained else None)

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}. "
                        f"Supported: resnet18/34/50/101/152, convnext_tiny/small/base, "
                        f"mobilenet_v3_small/large, efficientnet_b0/b4, "
                        f"vit_b_16/b_32/l_16/l_32/h_14, deit_tiny/small/base, "
                        f"swin_t/s/b/v2_t/v2_s/v2_b")

    return model


class Conv2dWithReflectionPadding(nn.Module):
    """
    Wraps a Conv2d to apply reflection padding instead of zero padding.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if isinstance(conv.padding, tuple):
            pad_h, pad_w = conv.padding
        else:
            pad_h = pad_w = conv.padding

        self.pad_h = pad_h
        self.pad_w = pad_w

        # Clone the original conv without padding
        self.conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=0,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None
        )
        self.conv.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.conv.bias.data.copy_(conv.bias.data)

        # Preserve gradients requirement
        self.conv.weight.requires_grad = conv.weight.requires_grad
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = conv.bias.requires_grad

    def forward(self, x):
        if self.pad_h or self.pad_w:
            padding = (self.pad_w, self.pad_w, self.pad_h, self.pad_h)
            x = F.pad(x, padding, mode='reflect')
        return self.conv(x)


def convert_conv_layers_to_reflection(module: nn.Module):
    """
    Recursively replace Conv2d layers with reflection-padding equivalents.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and any(child.padding):
            wrapped = Conv2dWithReflectionPadding(child)
            module._modules[name] = wrapped
        else:
            convert_conv_layers_to_reflection(child)


def _normalize_polygon(points):
    normalized = []
    for pt in points:
        if isinstance(pt, dict):
            x = float(pt.get('x', 0.0))
            y = float(pt.get('y', 0.0))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            x = float(pt[0])
            y = float(pt[1])
        else:
            continue
        normalized.append((x, y))
    return normalized


def _extract_polygons(entry):
    polygons = []
    if not isinstance(entry, dict):
        return polygons

    if 'patch_corners' in entry:
        poly = _normalize_polygon(entry['patch_corners'])
        if len(poly) >= 3:
            polygons.append(poly)

    if 'patches' in entry and isinstance(entry['patches'], list):
        for patch in entry['patches']:
            corners = patch.get('patch_corners') or patch.get('corners') or patch.get('points')
            poly = _normalize_polygon(corners or [])
            if len(poly) >= 3:
                polygons.append(poly)

    if 'polygons' in entry and isinstance(entry['polygons'], list):
        for poly_entry in entry['polygons']:
            poly = _normalize_polygon(poly_entry)
            if len(poly) >= 3:
                polygons.append(poly)

    return polygons


def load_patch_metadata(metadata_path, eval_mode="silent"):
    """
    Load patch metadata from various formats.
    Supports:
    - Legacy format: {"images": [...]}
    - Generator.py format: {"test": {"clean": [...], "patched": [...]}}
    - Evaluation format: {"test_mixed": [...], "test_mixed_gaussian": [...]}

    Args:
        metadata_path: Path to metadata JSON file
        eval_mode: Evaluation mode - "silent", "gaussian", "shot", "impulse"
                   Determines which noise variant to load for generator.py datasets
    """
    metadata_path = Path(metadata_path)
    with metadata_path.open('r') as f:
        data = json.load(f)

    images = []

    if isinstance(data, dict):
        # Check if it's generator.py format
        if 'test' in data and isinstance(data['test'], dict):
            # Generator.py dataset format
            test_data = data['test']

            if eval_mode == "silent":
                # Load clean and patched without noise
                images.extend(test_data.get('clean', []))
                images.extend(test_data.get('patched', []))
            else:
                # Load noisy versions
                images.extend(test_data.get(f'clean_{eval_mode}', []))
                images.extend(test_data.get(f'patched_{eval_mode}', []))

            print(f"  Loaded generator.py format metadata (mode: {eval_mode})")

        # Check if it's evaluation format
        elif f'test_mixed' in data:
            # Evaluation format (mixed clean + patched)
            if eval_mode == "silent":
                images = data.get('test_mixed', [])
            else:
                images = data.get(f'test_mixed_{eval_mode}', [])

            print(f"  Loaded evaluation format metadata (mode: {eval_mode})")

        # Legacy format
        else:
            images = data.get('images') or data.get('data') or []
            print(f"  Loaded legacy format metadata")

    elif isinstance(data, list):
        images = data
        print(f"  Loaded list format metadata")

    metadata = {}
    for entry in images:
        polygons = _extract_polygons(entry)
        keys = set()
        for key_name in ('filename', 'original_filename', 'image'):
            key_val = entry.get(key_name)
            if key_val:
                keys.add(key_val)
        if not keys and 'id' in entry:
            keys.add(str(entry['id']))

        if not keys:
            continue

        for key in keys:
            existing = metadata.setdefault(key, [])
            existing.extend(polygons)

    return metadata


def create_polygon_mask(polygons, height, width):
    if not polygons:
        return np.zeros((height, width), dtype=bool)

    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        if len(poly) < 3:
            continue
        clamped = []
        for x, y in poly:
            cx = max(0.0, min(float(x), width - 1))
            cy = max(0.0, min(float(y), height - 1))
            clamped.append((cx, cy))
        if len(clamped) >= 3:
            draw.polygon(clamped, outline=1, fill=1)

    return np.array(mask_img, dtype=bool)


def print_phase_header(phase_num: int, phase_name: str):
    """Print formatted phase header"""
    print("\n" + "=" * 80)
    print(f"PHASE {phase_num}: {phase_name}")
    print("=" * 80)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
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

    # Load backbone model
    backbone_name = getattr(cfg.model, 'backbone', 'resnet50')
    print(f"\nLoading backbone model: {backbone_name}...")
    try:
        model = load_backbone_model(backbone_name, pretrained=True)
        convert_conv_layers_to_reflection(model)
        print("✓ Converted convolutions to reflection padding")
        model.eval()
        model.to(device)
        print(f"✓ Model loaded successfully: {backbone_name}")
    except ValueError as e:
        print(f"\n✗ Error loading backbone: {e}")
        return

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

    # Image preprocessing - MUST match patchgen.py exactly!
    # If images are already 224x224 (from patchgen.py), we need different preprocessing
    # Check if we're loading pre-processed images
    test_path = Path(cfg.data.test.patch_path)

    # Determine if images are already 224x224 (from patchgen.py)
    # Check metadata for image size info
    metadata_path_for_check = None
    if hasattr(cfg, 'evaluation') and hasattr(cfg.evaluation, 'patch_metadata_path'):
        metadata_path_for_check = Path(cfg.evaluation.patch_metadata_path)
        if not metadata_path_for_check.is_absolute():
            metadata_path_for_check = test_path / "patch_metadata.json"

    images_are_preprocessed = False
    if metadata_path_for_check and metadata_path_for_check.exists():
        try:
            with open(metadata_path_for_check) as f_check:
                meta_check = json.load(f_check)
                # If patch_shape exists in config, images are from patchgen.py
                if 'config' in meta_check and 'patch_shape' in meta_check['config']:
                    images_are_preprocessed = True
                    print(f"✓ Detected preprocessed images from patchgen.py")
        except:
            pass

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if images_are_preprocessed:
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("  Train transform: Resize→Crop→Normalize")
        print("  Eval transform: Direct ToTensor (images already 224x224)")
    else:
        eval_transform = train_transform
        print("  Train/Eval transform: Resize→Crop→Normalize")

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

    # Generate weights directory structure: model_weights/{backbone}/{model_config}/
    backbone_name = getattr(cfg.model, 'backbone', 'resnet50')
    model_config_str = (
        f"{cfg.model.type}_"
        f"res{cfg.model.spatial_resolution}_"
        f"feat{cfg.model.feature_dim}_"
        f"hid{cfg.model.hidden_dim}_"
        f"lat{cfg.model.latent_dim}_"
        f"L{cfg.model.num_layers}"
    )

    weights_subdir = weights_dir / backbone_name / model_config_str
    weights_subdir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_subdir / "model.pt"

    print(f"  Weights directory: {weights_subdir}")
    print(f"  Backbone: {backbone_name}")

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
            imagenet_dataset = ImageFolder(root=cfg.data.imagenet.path, transform=train_transform)
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
                clean_domain_dataset = LocalImageDataset(clean_domain_folder, transform=train_transform)

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

                # Generate LoRA weights directory structure: lora_weights/{backbone}/{model_config}/
                lora_config_str = (
                    f"{cfg.model.type}_"
                    f"rank{cfg.domain_adaptation.lora.rank}_"
                    f"alpha{cfg.domain_adaptation.lora.alpha}"
                )

                lora_weights_subdir = lora_weights_dir / backbone_name / model_config_str / lora_config_str
                lora_weights_subdir.mkdir(parents=True, exist_ok=True)
                lora_weights_path = lora_weights_subdir / "lora.pt"

                print(f"  LoRA weights directory: {lora_weights_subdir}")
                print(f"  Backbone: {backbone_name}")

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

    # Get evaluation mode to auto-select the correct subdirectory
    evaluation_cfg = getattr(cfg, 'evaluation', None)
    def _eval_cfg_get_early(key, default=None):
        if evaluation_cfg is None:
            return default
        if isinstance(evaluation_cfg, dict):
            return evaluation_cfg.get(key, default)
        return getattr(evaluation_cfg, key, default)

    eval_mode = _eval_cfg_get_early('mode', 'silent')
    if eval_mode not in ['silent', 'gaussian', 'shot', 'impulse']:
        print(f"\n⚠ Warning: Invalid evaluation mode '{eval_mode}', using 'silent'")
        eval_mode = 'silent'

    patch_folder = Path(cfg.data.test.patch_path)

    # Auto-detect and select correct subdirectory based on evaluation mode
    # Check if path is a directory containing subdirectories (generator.py structure)
    if patch_folder.exists() and patch_folder.is_dir():
        # Check for evaluation structure (evaluation/test_mixed, evaluation/test_mixed_gaussian, etc.)
        if eval_mode == "silent":
            candidate_dirs = [
                patch_folder / "test_mixed",  # Evaluation format
                patch_folder / "test",  # Generator format (will need further selection)
                patch_folder  # Already pointing to the right place
            ]
        else:
            candidate_dirs = [
                patch_folder / f"test_mixed_{eval_mode}",  # Evaluation format
                patch_folder / "test",  # Generator format (will need further selection)
                patch_folder  # Already pointing to the right place
            ]

        # Try to find the right directory
        for candidate in candidate_dirs:
            if candidate.exists() and candidate.is_dir():
                # Check if it has images
                has_images = any(
                    f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
                    for f in candidate.iterdir()
                    if f.is_file()
                )
                if has_images:
                    if candidate != patch_folder:
                        print(f"✓ Auto-selected subdirectory for mode '{eval_mode}': {candidate.name}")
                        patch_folder = candidate
                    break

    if not patch_folder.exists():
        print(f"\n✗ Error: Test folder not found: {patch_folder}")
        print(f"  Please update the path in configs/config.yaml")
        print(f"  Current evaluation mode: {eval_mode}")
        extractor.remove_hooks()
        return

    patch_eval_enabled = False
    patch_metadata_map = {}
    missing_metadata = set()
    eval_pixel_tp = eval_pixel_fp = eval_pixel_fn = eval_pixel_tn = 0
    eval_images = 0
    eval_images_with_patch = 0
    eval_images_detected = 0
    eval_images_missed = 0
    eval_images_false_alarm = 0
    gt_mask_cache = {}

    try:
        patch_dataset = LocalImageDataset(patch_folder, transform=eval_transform)

        # Apply num_samples limit if specified
        num_samples = getattr(cfg.data.test, 'num_samples', -1)
        if num_samples > 0 and num_samples < len(patch_dataset):
            print(f"✓ Found {len(patch_dataset)} test images, using first {num_samples}")
            patch_dataset = Subset(patch_dataset, list(range(num_samples)))
        else:
            print(f"\n✓ Found {len(patch_dataset)} test images")

        patch_loader = DataLoader(
            patch_dataset,
            batch_size=cfg.data.test.batch_size,
            shuffle=False,
            num_workers=cfg.device.num_workers,
            pin_memory=True
        )

        # Create output directory
        output_dir = Path(cfg.output.dir)
        output_dir.mkdir(exist_ok=True)
        print(f"✓ Output directory: {output_dir}")

        visualization_dir = output_dir / "visualize_results"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Visualization directory: {visualization_dir}")

        evaluation_cfg = getattr(cfg, 'evaluation', None)

        def _eval_cfg_get(key, default=None):
            if evaluation_cfg is None:
                return default
            if isinstance(evaluation_cfg, dict):
                return evaluation_cfg.get(key, default)
            return getattr(evaluation_cfg, key, default)

        patch_eval_enabled = bool(_eval_cfg_get('enable_patch_metrics', False))
        patch_metadata_map = {}
        missing_metadata = set()
        eval_pixel_tp = eval_pixel_fp = eval_pixel_fn = eval_pixel_tn = 0
        eval_images = 0
        eval_images_with_patch = 0
        eval_images_detected = 0
        eval_images_missed = 0
        eval_images_false_alarm = 0
        gt_mask_cache = {}

        # Grid-level metrics accumulation
        grid_tp_total = 0
        grid_tn_total = 0
        grid_fp_total = 0
        grid_fn_total = 0
        grid_images_evaluated = 0

        if patch_eval_enabled:
            metadata_path_value = _eval_cfg_get('patch_metadata_path', '')
            metadata_path = Path(metadata_path_value)
            if not metadata_path.is_absolute():
                metadata_path = Path(metadata_path_value)

            # Auto-detect metadata file if path doesn't exist
            if not metadata_path.exists():
                # Try to find metadata in parent directory of patch_folder
                parent_dir = Path(cfg.data.test.patch_path)
                if (parent_dir / "evaluation").exists():
                    parent_dir = parent_dir / "evaluation"

                candidate_metadata = [
                    parent_dir / "evaluation_metadata.json",
                    parent_dir / "dataset_metadata.json",
                    parent_dir.parent / "evaluation_metadata.json",
                    parent_dir.parent / "dataset_metadata.json",
                ]

                for candidate in candidate_metadata:
                    if candidate.exists():
                        print(f"✓ Auto-detected metadata: {candidate}")
                        metadata_path = candidate
                        break

            print(f"✓ Evaluation mode: {eval_mode}")

            if not metadata_path.exists():
                print(f"\n⚠ Warning: Patch metadata not found: {metadata_path}. Disabling patch metrics.")
                print(f"  Tried to auto-detect metadata file but couldn't find it.")
                patch_eval_enabled = False
            else:
                try:
                    patch_metadata_map = load_patch_metadata(metadata_path, eval_mode=eval_mode)
                    print(f"✓ Loaded patch metadata for {len(patch_metadata_map)} images")
                except Exception as meta_exc:
                    print(f"\n⚠ Warning: Failed to load patch metadata: {meta_exc}. Disabling patch metrics.")
                    patch_eval_enabled = False

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

                if patch_eval_enabled:
                    polygons = None
                    for key in (img_name, Path(img_name).name):
                        if key in patch_metadata_map:
                            polygons = patch_metadata_map[key]
                            break

                    if polygons is None:
                        missing_metadata.add(img_name)
                    else:
                        height, width = img.shape[1], img.shape[2]
                        with torch.no_grad():
                            pred_mask_tensor = patch_mask.unsqueeze(0).unsqueeze(0).float()
                            upsampled = F.interpolate(
                                pred_mask_tensor,
                                size=(height, width),
                                mode='nearest'
                            )
                        pred_mask_np = (upsampled[0, 0].detach().cpu().numpy() >= 0.5)

                        cache_key = (img_name, height, width)
                        if cache_key not in gt_mask_cache:
                            gt_mask_cache[cache_key] = create_polygon_mask(polygons, height, width)
                        gt_mask_np = gt_mask_cache[cache_key]

                        pred_bool = pred_mask_np.astype(bool)
                        gt_bool = gt_mask_np.astype(bool)

                        eval_pixel_tp += int(np.logical_and(pred_bool, gt_bool).sum())
                        eval_pixel_fp += int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
                        eval_pixel_fn += int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
                        eval_pixel_tn += int(np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).sum())

                        intersect = bool(np.logical_and(pred_bool, gt_bool).any())
                        pred_any = bool(pred_bool.any())
                        gt_any = bool(gt_bool.any())

                        eval_images += 1
                        if gt_any:
                            eval_images_with_patch += 1
                            if intersect:
                                eval_images_detected += 1
                            else:
                                eval_images_missed += 1
                                if pred_any:
                                    eval_images_false_alarm += 1
                        else:
                            if pred_any:
                                eval_images_false_alarm += 1

                # Visualize and save (first item per batch)
                if cfg.output.save_visualizations:
                    if not batch_visualized:
                        threshold_method_name = str(cfg.detection.threshold_method)

                        # Get patch corners from metadata if available
                        patch_corners_for_viz = None
                        gt_grid_for_viz = None
                        grid_metrics_for_viz = None

                        if patch_metadata_map:
                            for key in (img_name, Path(img_name).name):
                                if key in patch_metadata_map:
                                    polygons = patch_metadata_map[key]
                                    if polygons and len(polygons) > 0:
                                        # Use first polygon as patch corners
                                        corners = polygons[0]
                                        if len(corners) >= 4:
                                            patch_corners_for_viz = [
                                                {'x': corners[0][0], 'y': corners[0][1]},
                                                {'x': corners[1][0], 'y': corners[1][1]},
                                                {'x': corners[2][0], 'y': corners[2][1]},
                                                {'x': corners[3][0], 'y': corners[3][1]}
                                            ]

                                            # Compute ground truth grid
                                            gt_grid_for_viz = compute_ground_truth_grid(
                                                patch_corners_for_viz,
                                                cfg.model.spatial_resolution,
                                                image_size=224
                                            )
                                    break

                        # Compute predicted grid from patch_mask (convert tensor to numpy first)
                        patch_mask_np = patch_mask.cpu().numpy() if torch.is_tensor(patch_mask) else patch_mask
                        pred_grid_for_viz = (patch_mask_np > 0).astype(bool)

                        # Compute grid metrics if we have GT
                        if gt_grid_for_viz is not None:
                            grid_metrics_for_viz = compute_grid_metrics(gt_grid_for_viz, pred_grid_for_viz)

                            # Accumulate grid metrics
                            grid_tp_total += grid_metrics_for_viz['tp']
                            grid_tn_total += grid_metrics_for_viz['tn']
                            grid_fp_total += grid_metrics_for_viz['fp']
                            grid_fn_total += grid_metrics_for_viz['fn']
                            grid_images_evaluated += 1

                        fig = visualize_results(
                            img,
                            anomaly_map,
                            patch_mask,
                            image_name=img_name,
                            threshold=threshold,
                            detection_pixel_threshold=cfg.detection.detection_pixel_threshold,
                            threshold_method=threshold_method_name,
                            model_type=cfg.model.type,
                            trajectories=embeddings_batch[b],
                            gt_grid=gt_grid_for_viz,
                            pred_grid=pred_grid_for_viz,
                            spatial_resolution=cfg.model.spatial_resolution,
                            grid_metrics=grid_metrics_for_viz
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

    if patch_eval_enabled:
        evaluated_pixels = eval_pixel_tp + eval_pixel_fp + eval_pixel_fn + eval_pixel_tn
        if evaluated_pixels > 0:
            accuracy = (eval_pixel_tp + eval_pixel_tn) / evaluated_pixels
            precision = eval_pixel_tp / (eval_pixel_tp + eval_pixel_fp) if (eval_pixel_tp + eval_pixel_fp) > 0 else 0.0
            recall = eval_pixel_tp / (eval_pixel_tp + eval_pixel_fn) if (eval_pixel_tp + eval_pixel_fn) > 0 else 0.0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            print(f"\nPatch Localization Metrics (pixel-level):")
            print(f"  Evaluated images:          {eval_images}")
            print(f"  Accuracy:                  {accuracy * 100:.2f}%")
            print(f"  Precision:                 {precision * 100:.2f}%")
            print(f"  Recall:                    {recall * 100:.2f}%")
            print(f"  F1 Score:                  {f1_score * 100:.2f}%")
            print(f"  Images with patch:         {eval_images_with_patch}")
            print(f"    Detected (overlap >0):   {eval_images_detected}")
            print(f"    Missed patches:          {eval_images_missed}")
            print(f"  False-alarm images:        {eval_images_false_alarm}")

        # Grid-level metrics
        if grid_images_evaluated > 0:
            grid_total_cells = grid_tp_total + grid_tn_total + grid_fp_total + grid_fn_total
            grid_accuracy = (grid_tp_total + grid_tn_total) / grid_total_cells if grid_total_cells > 0 else 0.0
            grid_precision = grid_tp_total / (grid_tp_total + grid_fp_total) if (grid_tp_total + grid_fp_total) > 0 else 0.0
            grid_recall = grid_tp_total / (grid_tp_total + grid_fn_total) if (grid_tp_total + grid_fn_total) > 0 else 0.0
            grid_f1 = (2 * grid_precision * grid_recall / (grid_precision + grid_recall)) if (grid_precision + grid_recall) > 0 else 0.0

            print(f"\nGrid-Level Metrics ({cfg.model.spatial_resolution}×{cfg.model.spatial_resolution} cells):")
            print(f"  Evaluated images:          {grid_images_evaluated}")
            print(f"  Total cells:               {grid_total_cells}")
            print(f"  Confusion Matrix:")
            print(f"    TP: {grid_tp_total:6d}  |  FP: {grid_fp_total:6d}")
            print(f"    FN: {grid_fn_total:6d}  |  TN: {grid_tn_total:6d}")
            print(f"  Performance Metrics:")
            print(f"    Accuracy:                {grid_accuracy * 100:.2f}%")
            print(f"    Precision:               {grid_precision * 100:.2f}%")
            print(f"    Recall:                  {grid_recall * 100:.2f}%")
            print(f"    F1 Score:                {grid_f1 * 100:.2f}%")

        if missing_metadata:
            missing_list = sorted(list(missing_metadata))
            preview = ', '.join(missing_list[:5])
            more = '' if len(missing_list) <= 5 else f", ... (+{len(missing_list) - 5} more)"
            print(f"\n⚠ Warning: Missing patch metadata for {len(missing_list)} images: {preview}{more}")

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
    print(f"  python -m trap.pipeline.main model.type=vae")
    print(f"  python -m trap.pipeline.main model.type=transformer")
    print(f"  python -m trap.pipeline.main domain_adaptation.enabled=true")
    print(f"  python -m trap.pipeline.main data.imagenet.num_epochs=20")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
