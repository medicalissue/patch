# Few-shot Image Patch Detection System

A sophisticated image patch detection system using Few-shot learning approach with trajectory analysis in deep neural network feature spaces.

## 🎯 Overview

This system detects adversarial patches in images using a **3-phase Few-shot learning** approach:

1. **Phase 1: Setup** - Load ResNet50 model and activation extractor
2. **Phase 2: Few-shot Base Learning** - Learn normal trajectory characteristics from ImageNet
3. **Phase 3: Few-shot Threshold Adaptation** - Set adaptive threshold using domain-specific clean images
4. **Phase 4: Testing** - Detect patches in test images

### Key Features

- ✅ **Few-shot Learning**: Learns from small samples (1000 ImageNet images by default)
- ✅ **Absolute Comparison**: Detection based on ImageNet statistics, not per-image normalization
- ✅ **Multi-metric Detection**: Vector field, spectral analysis, and curvature scores
- ✅ **Hydra Configuration**: Flexible configuration management with command-line overrides
- ✅ **100% GPU**: All computations performed on GPU for maximum efficiency
- ✅ **Comprehensive Visualization**: Detailed detection results with component breakdowns

## 📋 Requirements

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended)

## 🏗️ Architecture

### Detection Pipeline

```
Input Image → ResNet50 → Multi-layer Activations → Trajectory Embedding
                                                           ↓
                                    Vector Field + Spectral + Curvature Analysis
                                                           ↓
                                              Anomaly Score → Detection
```

### Key Components

- **AttractorLearner** ([attracter.py](attracter.py)): Learns normal trajectory characteristics
- **PatchDetector** ([detector.py](detector.py)): Detects anomalies using learned statistics
- **ActivationExtractor** ([extracter.py](extracter.py)): Extracts ResNet activations
- **Visualization** ([visualize.py](visualize.py)): Creates comprehensive detection visualizations

## 🚀 Usage

### Basic Usage

```bash
python test.py
```

### Configuration

All settings are in [configs/config.yaml](configs/config.yaml):

```yaml
device:
  cuda_id: 2  # GPU device
  num_workers: 8

data:
  imagenet:
    path: /data/ImageNet/train
    num_samples: 1000  # -1 for all
    batch_size: 128

  domain:
    clean_path: images_without_patches
    num_samples: 50  # -1 for all
    batch_size: 32

  test:
    patch_path: images_with_patches
    batch_size: 32

model:
  spatial_resolution: 7  # 7, 14, 28, 56
  feature_dim: 128

detection:
  threshold_multiplier: 2.0  # mean + k*std
  detection_pixel_threshold: 0

output:
  dir: detection_results
  save_visualizations: true
```

### Command-line Overrides

Hydra allows easy parameter overrides:

```bash
# Use 500 ImageNet samples
python test.py data.imagenet.num_samples=500

# Stricter threshold
python test.py detection.threshold_multiplier=3.0

# Higher spatial resolution
python test.py model.spatial_resolution=14

# Multiple overrides
python test.py data.imagenet.num_samples=500 detection.threshold_multiplier=3.0
```

## 📊 Detection Metrics

The system computes three complementary metrics:

### 1. Vector Field Score
- Measures trajectory magnitude and direction consistency
- Detects changes in feature space dynamics

### 2. Spectral Score
- Analyzes frequency domain characteristics
- Identifies unusual spectral patterns

### 3. Curvature Score
- Measures trajectory smoothness
- Detects abrupt changes in feature evolution

All metrics are combined with spatial coherence enhancement for robust detection.

## 📁 Directory Structure

```
patch/
├── configs/
│   └── config.yaml          # Hydra configuration
├── attracter.py             # Few-shot base learning
├── detector.py              # Patch detection
├── extracter.py             # Activation extraction
├── seriese_embedding.py     # Trajectory embedding
├── visualize.py             # Result visualization
├── dataloader.py            # Image loading
├── trajectory.py            # Trajectory stacking
├── test.py                  # Main entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔬 Detection Methodology

### Phase 2: Few-shot Base Learning

Learns statistical characteristics of normal trajectories:
- **Vector Field**: Mean/std of magnitude, mean direction consistency
- **Spectral**: Mean/std of power spectrum
- **Curvature**: Mean/std of trajectory curvature

These statistics are learned from ~1000 ImageNet samples.

### Phase 3: Threshold Adaptation

Sets adaptive threshold using domain clean images:
```
threshold = mean_score + k * std_score
```

This adapts the ImageNet-learned baseline to the target domain.

### Phase 4: Testing

Compares test images against learned statistics using **absolute comparison**:
- No per-image normalization
- Direct comparison with ImageNet baseline
- Robust cross-domain detection

## 📈 Output

Detection results are saved in the output directory (default: `detection_results/`):

- **Visualization images**: `result_<image_name>.png`
- Each visualization includes:
  - Original image
  - Anomaly score map
  - Detection overlay
  - Component scores (vector field, spectral, curvature)
  - Few-shot detection metrics and statistics

## 🎛️ Tuning Parameters

### For Higher Sensitivity
```bash
python test.py detection.threshold_multiplier=1.5
```

### For Lower False Positives
```bash
python test.py detection.threshold_multiplier=3.0
```

### For Higher Resolution
```bash
python test.py model.spatial_resolution=14
```

### For More Training Data
```bash
python test.py data.imagenet.num_samples=5000
```

## 🐛 Troubleshooting

### ImageNet path not found
Update the path in `configs/config.yaml`:
```yaml
data:
  imagenet:
    path: /path/to/your/ImageNet/train
```

### CUDA out of memory
Reduce batch sizes:
```bash
python test.py data.imagenet.batch_size=64 data.test.batch_size=16
```

### Slow processing
Increase workers or reduce spatial resolution:
```bash
python test.py device.num_workers=16 model.spatial_resolution=7
```

## 📝 Citation

If you use this code, please cite:

```
Few-shot Image Patch Detection System
Using Trajectory Analysis in Deep Feature Spaces
```

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- ResNet50 from torchvision
- Hydra for configuration management
- PyTorch team for the framework
