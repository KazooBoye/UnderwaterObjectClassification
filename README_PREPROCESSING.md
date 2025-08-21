# Underwater Object Classification - Dataset Preprocessing Pipeline

This repository contains a comprehensive preprocessing pipeline for the underwater aquarium dataset that addresses class imbalance, multi-scale objects, and complex scene challenges.

## Features

### Core Preprocessing Strategies
- **Class Imbalance Mitigation**: Smart oversampling with 25:1 ratio balancing
- **Multi-Scale Augmentation**: Size-aware transformations preserving object relationships
- **Advanced Augmentations**: Underwater-specific effects, geometric transformations
- **Mosaic Augmentation**: Complex multi-object scene generation
- **Quality Control**: Automated validation and corruption detection
- **Comprehensive Analytics**: Detailed statistics and training recommendations

### Key Benefits
- **Balanced Dataset**: Equal representation across all 7 classes
- **Enhanced Minority Classes**: 10-25x augmentation for rare classes (starfish, stingray, puffin)
- **Preserved Data Quality**: Intelligent augmentation that maintains object integrity
- **Training Ready**: Generates YAML configs and class weights for immediate use
- **Detailed Analytics**: Complete preprocessing reports and statistics

## Requirements

### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.21.0` - Numerical operations
- `opencv-python>=4.5.0` - Image processing
- `matplotlib>=3.5.0` - Visualization
- `pandas>=1.3.0` - Data analysis
- `scikit-learn>=1.0.0` - ML utilities
- `albumentations>=1.3.0` - Advanced augmentations
- `tqdm>=4.62.0` - Progress bars
- `PyYAML>=6.0.0` - Configuration files
- `Pillow>=8.3.0` - Image handling

## Quick Start

### Method 1: Simple Runner Script
```bash
# Install dependencies and run preprocessing
python run_preprocessing.py
```

### Method 2: Direct Script Usage
```bash
# Basic usage
python preprocess_dataset.py \
    --input ./aquarium_pretrain \
    --output ./preprocessed_dataset \
    --suffix balanced_v1

# With custom configuration
python preprocess_dataset.py \
    --input ./aquarium_pretrain \
    --output ./preprocessed_dataset \
    --config preprocessing_config.json \
    --suffix custom_v1
```

## Dataset Challenges Addressed

### 1. Severe Class Imbalance
**Problem**: Fish (59%) vs Starfish (2.3%) = 25:1 ratio
**Solution**: 
- Oversample minority classes: Starfish 25x, Stingray 14x, Puffin 11x
- Smart undersampling of majority class (Fish)
- Class-specific augmentation intensity

### 2. Multi-Scale Object Detection
**Problem**: Object sizes range from 0.6% to 6% normalized area
**Solution**:
- Size-aware augmentation pipelines
- Scale-preserving transformations  
- Multi-resolution training preparation

### 3. Complex Multi-Object Scenes
**Problem**: Average 7.4 objects per image, up to 56 objects, 33% multi-class images
**Solution**:
- Mosaic augmentation for complex scene simulation
- Copy-paste techniques for minority classes
- Crowd-aware bounding box handling

### 4. Shape Variability
**Problem**: High aspect ratio variance (Sharks 2.12, Fish 1.64±1.11)
**Solution**:
- Shape-preserving augmentations
- Aspect ratio constraints
- Class-specific transformation strategies

## Configuration Options

### Basic Configuration (`preprocessing_config.json`)
```json
{
  "preprocessing": {
    "output_suffix": "balanced_v1",
    "max_samples_per_class": 2000,
    "heavy_augmentation_count": 10,
    "medium_augmentation_count": 5,
    "light_augmentation_count": 2,
    "random_seed": 42
  }
}
```

### Class-Specific Settings
Each class has tailored augmentation strategies:
- **Fish**: Light augmentation (majority class)
- **Jellyfish/Penguin/Shark**: Medium augmentation
- **Puffin/Stingray/Starfish**: Heavy augmentation (minorities)

## Output Structure

```
preprocessed_dataset/
├── train_balanced_v1/
│   ├── images/           # Balanced training images
│   └── labels/           # YOLO format labels
├── val_balanced_v1/
│   ├── images/           # Validation images (original)
│   └── labels/           # Validation labels
├── test_balanced_v1/
│   ├── images/           # Test images (original)
│   └── labels/           # Test labels
├── statistics/
│   ├── preprocessing_stats.json    # Detailed statistics
│   ├── preprocessing_stats.pkl     # Python objects
│   └── preprocessing_report.txt    # Human-readable report
├── configs/
├── data_balanced_v1.yaml          # Training configuration
└── README_preprocessed.txt         # Output documentation
```

## Expected Results

### Training Set Transformation
**Before Preprocessing:**
- Fish: 1,961 samples (59.0%)
- Jellyfish: 385 samples (11.6%)
- Penguin: 330 samples (9.9%)
- Shark: 259 samples (7.8%)
- Puffin: 175 samples (5.3%)
- Stingray: 136 samples (4.1%)
- Starfish: 78 samples (2.3%)

**After Preprocessing:**
- All classes: ~1,500-2,000 samples each
- Balanced distribution: ~14.3% per class
- Total dataset size: 3-5x larger due to augmentation

### Quality Improvements
- **Eliminated class bias**: No single class dominates
- **Enhanced minority representation**: Rare classes now well-represented
- **Improved robustness**: Diverse augmentations increase generalization
- **Training efficiency**: Balanced batches improve convergence

## Training Recommendations

### Recommended Model Configuration
```yaml
# Loss Function
loss_function: "focal_loss"
focal_alpha: 0.25
focal_gamma: 2.0

# Class Weights (auto-generated)
class_weights: [0.5, 1.2, 1.4, 1.8, 3.2, 4.1, 7.2]

# Training Parameters
batch_size: 16-32
learning_rate: 1e-4
optimizer: "AdamW"
scheduler: "cosine_annealing"
epochs: 100-150
```

### Evaluation Metrics Priority
1. **Per-class F1 scores** (most important)
2. **Balanced accuracy** 
3. **mAP@0.5** and **mAP@0.5:0.95**
4. **Recall for minority classes** (Starfish, Stingray, Puffin)
5. **Confusion matrix analysis**

**Avoid using overall accuracy** - misleading due to original imbalance

## Monitoring and Validation

### Quality Checks Included
- Corrupted file detection
- Annotation validation
- Duplicate image removal  
- Bounding box sanity checks
- Size and aspect ratio validation

### Generated Reports
- **preprocessing_stats.json**: Complete numerical analysis
- **preprocessing_report.txt**: Human-readable summary
- **Training recommendations**: Optimal hyperparameters
- **Class weight calculations**: For balanced loss functions

## Advanced Usage

### Custom Augmentation Pipeline
```python
from preprocess_dataset import UnderwaterDatasetPreprocessor

# Custom configuration
config = {
    'output_suffix': 'custom_heavy',
    'max_samples_per_class': 3000,
    'heavy_augmentation_count': 15,
    'enable_underwater_effects': True
}

# Initialize preprocessor
preprocessor = UnderwaterDatasetPreprocessor(
    dataset_path="./aquarium_pretrain",
    output_path="./custom_output", 
    config=config
)

# Run preprocessing
preprocessor.run_preprocessing()
```

### Batch Processing Multiple Configurations
```bash
# Process with different augmentation intensities
for intensity in light medium heavy; do
    python preprocess_dataset.py \
        --input ./aquarium_pretrain \
        --output ./preprocessed_${intensity} \
        --suffix ${intensity}_v1
done
```

## Troubleshooting

### Common Issues

**1. Memory Issues**
```bash
# Reduce batch processing size
export OPENCV_IO_MAX_IMAGE_PIXELS=1000000000
```

**2. Missing Dependencies**
```bash
# Install with conda instead of pip
conda install -c conda-forge opencv albumentations
```

**3. Slow Processing**
```bash
# Reduce augmentation counts in config
"heavy_augmentation_count": 5  # instead of 10
```

### Verification Steps
1. Check output folder structure
2. Verify balanced class distribution in statistics
3. Examine sample augmented images
4. Validate YAML configuration file

## Support

### Generated Documentation
After preprocessing, check:
- `statistics/preprocessing_report.txt` - Detailed analysis
- `data_balanced_v1.yaml` - Ready-to-use training config
- Sample images in output folders

### Performance Expectations
- **Processing Time**: ~10-30 minutes for full dataset
- **Storage Requirements**: 3-5x original dataset size
- **Memory Usage**: ~2-4GB during processing

## Expected Training Improvements

Using this preprocessing pipeline, you should expect:

1. **Balanced Performance**: Equal performance across all classes
2. **Improved Minority Class Recall**: Starfish, Stingray, Puffin detection significantly improved
3. **Faster Convergence**: Balanced batches lead to stable training
4. **Better Generalization**: Diverse augmentations improve test performance
5. **Reduced Overfitting**: Larger, more diverse training set

The preprocessing pipeline transforms a severely imbalanced dataset into a training-ready, balanced dataset optimized for underwater object detection challenges.

---

*This preprocessing pipeline is specifically designed for the underwater aquarium dataset but can be adapted for other imbalanced object detection datasets with similar challenges.*
