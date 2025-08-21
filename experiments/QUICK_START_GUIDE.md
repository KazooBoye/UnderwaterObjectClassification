# Underwater Object Detection - Experiments Quick Start Guide

## Overview

This guide will help you run the complete experimental comparison of three object detection approaches for underwater aquarium images:

1. **YOLOv8** (One-stage detector)
2. **Faster R-CNN with FPN** (Two-stage detector)  
3. **DETR** (Transformer-based detector)

## Prerequisites

### 1. Environment Setup

```bash
# Navigate to experiments directory
cd experiments/

# Install all dependencies
pip install -r requirements.txt

# Verify TensorFlow GPU setup
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

### 2. Dataset Preparation

Ensure your preprocessed dataset is available:

```bash
# Check if preprocessed dataset exists
ls -la ../preprocessed_dataset/

# If not available, run preprocessing first
cd ..
python run_preprocessing.py
cd experiments/
```

## Quick Start Commands

### Run All Three Approaches (Recommended)

```bash
# Train all approaches with default settings
python run_all_experiments.py --approach all

# Or use the executable script
./run_all_experiments.py --approach all
```

### Train Individual Approaches

```bash
# Train only YOLOv8
python run_all_experiments.py --approach yolov8

# Train only Faster R-CNN
python run_all_experiments.py --approach faster_rcnn

# Train only DETR
python run_all_experiments.py --approach detr
```

### Advanced Usage

```bash
# Custom dataset path
python run_all_experiments.py --dataset /path/to/your/dataset

# Custom experiment directory
python run_all_experiments.py --experiment_dir ./my_experiments

# Skip training and run only comparative evaluation
python run_all_experiments.py --skip_training
```

## Expected Output Structure

After running experiments, you'll get the following structure:

```
experiments_results/
└── experiment_20240121_143022/
    ├── approach_yolov8/
    │   ├── training_results.json
    │   ├── training_history.json
    │   ├── yolov8_best_model.h5
    │   └── logs/
    ├── approach_faster_rcnn/
    │   ├── training_results.json
    │   ├── training_history.json
    │   ├── faster_rcnn_best_model.h5
    │   └── logs/
    ├── approach_detr/
    │   ├── training_results.json
    │   ├── training_history.json
    │   ├── detr_best_model.h5
    │   └── logs/
    ├── comparative_analysis_report.md
    ├── comparative_plots/
    │   ├── map_comparison.png
    │   ├── per_class_performance.png
    │   ├── minority_class_performance.png
    │   ├── training_curves_comparison.png
    │   └── efficiency_comparison.png
    ├── all_training_results.json
    └── logs/
        └── training_20240121_143022.log
```

## Key Results Files

### 1. Individual Approach Results
- `training_results.json` - Training metrics, test performance, timing
- `training_history.json` - Epoch-by-epoch training curves
- `*_best_model.h5` - Trained model weights

### 2. Comparative Analysis
- `comparative_analysis_report.md` - Comprehensive comparison report
- `comparative_plots/` - Visualization charts comparing all approaches
- `all_training_results.json` - Summary of all experimental results

## Configuration Options

### Default Training Parameters

```yaml
epochs: 100
batch_size: 16
learning_rate: 0.001
image_size: 640
num_classes: 7
early_stopping_patience: 15
reduce_lr_patience: 10
```

### Approach-Specific Settings

**YOLOv8:**
- confidence_threshold: 0.25
- nms_threshold: 0.45
- anchor_sizes: [16, 32, 64]

**Faster R-CNN:**
- backbone: resnet50
- fpn_channels: 256
- rpn_batch_size: 256

**DETR:**
- hidden_dim: 256
- num_heads: 8
- num_encoder_layers: 6

## Estimated Training Times

On a modern GPU (e.g., RTX 3080):

- **YOLOv8**: ~2-3 hours
- **Faster R-CNN**: ~4-5 hours  
- **DETR**: ~6-8 hours

**Total for all approaches: ~12-16 hours**

## Monitoring Progress

### Real-time Monitoring

```bash
# Watch training logs
tail -f experiments_results/experiment_*/logs/training_*.log

# Monitor GPU usage
watch nvidia-smi
```

### TensorBoard (if enabled)

```bash
# Launch TensorBoard
tensorboard --logdir experiments_results/

# Open browser to http://localhost:6006
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce batch size in configs
   # Or use mixed precision training
   ```

2. **Dataset Not Found**
   ```bash
   # Check dataset path
   ls -la ../preprocessed_dataset/
   
   # Run preprocessing if needed
   python ../run_preprocessing.py
   ```

3. **GPU Not Detected**
   ```bash
   # Check CUDA installation
   nvidia-smi
   
   # Verify TensorFlow GPU
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

### Getting Help

1. Check the detailed logs in `experiments_results/*/logs/`
2. Review the error messages in the console output
3. Ensure all dependencies are correctly installed
4. Verify dataset format and availability

## Next Steps

After running experiments:

1. **Review Results**: Check `comparative_analysis_report.md`
2. **Analyze Performance**: Examine plots in `comparative_plots/`
3. **Model Selection**: Choose best approach based on your requirements
4. **Fine-tuning**: Adjust configurations and re-run specific approaches
5. **Deployment**: Use the best model for your underwater detection application

## Performance Metrics

The experiments will evaluate:

- **mAP@0.5** - Mean Average Precision at IoU 0.5
- **mAP@0.75** - Mean Average Precision at IoU 0.75  
- **Per-class AP** - Average Precision for each of 7 classes
- **Minority class performance** - Special focus on Starfish, Stingray, Jellyfish
- **Training time** - Time to convergence
- **Inference speed** - FPS on test images
- **Model size** - Storage requirements

## Citation

If you use this experimental framework in your research, please cite:

```
Underwater Object Detection: A Comparative Study of One-Stage, Two-Stage, and Transformer-Based Approaches
[Your Name], 2024
```
