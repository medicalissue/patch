# Quick Start Guide

## Installation

```bash
# Clone the repository (if needed)
cd /path/to/patch

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. **Prepare your data directories**:
   ```
   /data/ImageNet/train/          # ImageNet training data
   images_without_patches/         # Clean images for threshold adaptation
   images_with_patches/            # Test images with patches
   ```

2. **Update configuration** (if paths differ):
   Edit `configs/config.yaml`:
   ```yaml
   data:
     imagenet:
       path: /your/path/to/ImageNet/train
     domain:
       clean_path: your_clean_images_folder
     test:
       patch_path: your_test_images_folder
   ```

## Run Detection

### Basic Run
```bash
python test.py
```

### Quick Test (with fewer samples)
```bash
python test.py data.imagenet.num_samples=100 data.domain.num_samples=10
```

### High Sensitivity
```bash
python test.py detection.threshold_multiplier=1.5
```

### High Precision (fewer false positives)
```bash
python test.py detection.threshold_multiplier=3.0
```

### High Resolution
```bash
python test.py model.spatial_resolution=14
```

## Expected Output

```
================================================================================
FEW-SHOT IMAGE PATCH DETECTION SYSTEM
================================================================================

Configuration:
device:
  cuda_id: 2
  num_workers: 8
...

================================================================================
PHASE 1: SETUP
================================================================================
Device: cuda:2
Workers: 8

Loading ResNet50 model...
✓ Model loaded successfully
...

================================================================================
PHASE 2: FEW-SHOT BASE LEARNING
================================================================================
Learning normal trajectory characteristics from ImageNet samples
...
✓ Normal trajectory characteristics learned:
  Vector Field:
    Magnitude:  mean=X.XXXX, std=X.XXXX
    Direction:  mean_cos_sim=X.XXXX
  Spectral:
    Power:      mean=X.XXXX, std=X.XXXX
  Curvature:
    Curvature:  mean=X.XXXX, std=X.XXXX

================================================================================
PHASE 3: FEW-SHOT THRESHOLD ADAPTATION
================================================================================
Setting adaptive threshold using domain-specific clean images
...
Adaptive threshold: X.XXXX
  Formula: mean + 2.0 * std

================================================================================
PHASE 4: TESTING
================================================================================
Detecting patches in test images
...
  [1/10] image1.jpg                   | Score: 2.345 | Pixels:  120 | ✓ DETECTED
  [2/10] image2.jpg                   | Score: 1.234 | Pixels:    5 | ✗ CLEAN
...

================================================================================
SUMMARY
================================================================================

Detection Results (10 images):
  Detected: 7
  Clean:    3
  Rate:     70.0%

Score Statistics:
  Min:  0.5432
  Max:  3.2109
  Mean: 1.8765

================================================================================
TIMING
================================================================================
  Phase 1 (Setup):              0.00s
  Phase 2 (Base Learning):      45.23s
  Phase 3 (Threshold Adapt):    12.34s
  Phase 4 (Testing):            8.91s
  ──────────────────────────────────────────────────────────────────────────────
  Total:                        66.48s
  Avg per test image:           0.891s

================================================================================
✓ FEW-SHOT PATCH DETECTION COMPLETED
✓ Results saved to: detection_results/
================================================================================
```

## View Results

Results are saved in `detection_results/` directory:
- `result_image1.png` - Visualization with all component maps
- `result_image2.png` - ...

Each visualization includes:
- Original image
- Anomaly score map
- Detection overlay
- Vector field score map
- Spectral score map
- Curvature score map
- Detection metrics and Few-shot explanation

## Troubleshooting

### "ImageNet path not found"
```bash
# Update the path in configs/config.yaml
vim configs/config.yaml
# Or use override
python test.py data.imagenet.path=/your/path/to/ImageNet/train
```

### "CUDA out of memory"
```bash
# Reduce batch sizes
python test.py data.imagenet.batch_size=32 data.test.batch_size=8
```

### "Too slow"
```bash
# Use lower resolution
python test.py model.spatial_resolution=7

# Or increase workers
python test.py device.num_workers=16
```

### "Too many false positives"
```bash
# Increase threshold multiplier
python test.py detection.threshold_multiplier=3.0
```

### "Missing detections"
```bash
# Decrease threshold multiplier
python test.py detection.threshold_multiplier=1.5

# Or use more ImageNet samples
python test.py data.imagenet.num_samples=2000
```

## Parameter Tuning Guide

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `threshold_multiplier` | More sensitive | More specific |
| `spatial_resolution` | Faster | More detailed |
| `imagenet.num_samples` | Faster | Better baseline |
| `feature_dim` | Faster | More information |
| `domain.num_samples` | Faster | Better adaptation |

## Common Use Cases

### Quick Experiment
```bash
python test.py \
  data.imagenet.num_samples=200 \
  data.domain.num_samples=20 \
  model.spatial_resolution=7
```

### High Quality Detection
```bash
python test.py \
  data.imagenet.num_samples=5000 \
  data.domain.num_samples=100 \
  model.spatial_resolution=14 \
  detection.threshold_multiplier=2.5
```

### Production Deployment
```bash
python test.py \
  data.imagenet.num_samples=1000 \
  data.domain.num_samples=50 \
  model.spatial_resolution=7 \
  detection.threshold_multiplier=2.0 \
  output.save_visualizations=false
```

## Next Steps

1. ✅ Review results in `detection_results/`
2. ✅ Tune parameters based on performance
3. ✅ Adapt to your specific domain
4. ✅ See [README.md](README.md) for detailed documentation
5. ✅ See [CHANGES.md](CHANGES.md) for what changed

## Help

For more information:
- **README.md**: Comprehensive system documentation
- **CHANGES.md**: Detailed refactoring summary
- **configs/config.yaml**: All configuration options

Need help? Check the configuration comments in `configs/config.yaml` - each parameter is documented!
