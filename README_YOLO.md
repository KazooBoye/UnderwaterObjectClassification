# YOLO Underwater Object Detection Training

This project implements a YOLOv8-style object detection model using TensorFlow for underwater multi-class object detection.

## 🎯 Features

- **YOLOv8 Architecture**: CSPDarknet backbone with FPN neck
- **TensorFlow Implementation**: Native TF 2.x with CUDA GPU acceleration
- **Multi-Scale Detection**: Handles objects from small starfish to large sharks
- **Class Imbalance Handling**: Focal loss for severely imbalanced classes
- **Edge-Safe Augmentation**: Preserves bounding box integrity
- **Automatic Hardware Detection**: GPU with CPU fallback

## 🏗️ Architecture

```
Input (640x640x3)
    ↓
CSPDarknet Backbone
    ├── P3 (80x80) - Small objects
    ├── P4 (40x40) - Medium objects  
    └── P5 (20x20) - Large objects
    ↓
Feature Pyramid Network (FPN)
    ↓
YOLO Detection Heads
    ├── Bounding Box Regression
    ├── Objectness Prediction
    └── Classification (7 classes)
```

## 📊 Dataset Classes

1. **Fish** (59% of dataset)
2. **Jellyfish** (12.1%)
3. **Penguin** (9.8%)
4. **Puffin** (6.5%)
5. **Shark** (5.6%)
6. **Stingray** (4.7%)
7. **Starfish** (2.3%) - Most challenging due to scarcity

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install requirements
pip install -r yolo_requirements.txt

# Or use the runner script to auto-install
python run_yolo_training.py --install-deps
```

### 2. Check System Requirements

```bash
# Verify everything is ready
python run_yolo_training.py --check-only
```

Expected output:
```
✓ TensorFlow 2.13.0
✓ OpenCV 4.8.0
✓ Found 1 GPU(s)
  GPU 0: /physical_device:GPU:0
✓ Dataset structure valid: aquarium_pretrain
  Training images: 1848
  Validation images: 264
✓ All checks passed! System is ready for training.
```

### 3. Start Training

```bash
# Basic training with defaults
python run_yolo_training.py

# Custom configuration
python run_yolo_training.py --epochs 50 --batch-size 8 --output my_training

# For systems with limited memory
python run_yolo_training.py --batch-size 4 --epochs 200
```

## ⚙️ Configuration

### Training Parameters

```python
class YOLOConfig:
    input_size = 640           # Input image size
    num_classes = 7           # Number of object classes
    batch_size = 16           # Training batch size
    epochs = 100              # Training epochs
    learning_rate = 0.001     # Initial learning rate
    
    # Loss weights (tuned for class imbalance)
    box_loss_weight = 0.05    # Bounding box regression
    cls_loss_weight = 0.5     # Classification loss
    obj_loss_weight = 1.0     # Objectness loss
```

### Hardware Recommendations

| Hardware | Batch Size | Expected Training Time |
|----------|------------|----------------------|
| RTX 4090 | 16-32 | 8-12 hours |
| RTX 3080 | 8-16 | 12-18 hours |
| RTX 2080 | 4-8 | 18-24 hours |
| CPU Only | 2-4 | 2-3 days |

## 📁 File Structure

```
├── yolo_tensorflow.py          # Main YOLO implementation
├── yolo_label_encoder.py       # Multi-scale label encoding
├── run_yolo_training.py        # Training runner script
├── yolo_requirements.txt       # Dependencies
└── README_YOLO.md             # This file

Training Output:
├── yolo_training_results/
│   ├── best_model.h5          # Best model weights
│   ├── config.json            # Training configuration
│   ├── checkpoint_epoch_*.h5  # Periodic checkpoints
│   └── training_log.txt       # Training progress
```

## 🔬 Technical Details

### Label Encoding Process

1. **Multi-Scale Targets**: Converts YOLO format to grid targets for 3 scales
2. **Anchor Assignment**: Finds best anchor for each ground truth box
3. **Grid Mapping**: Maps normalized coordinates to grid cells
4. **One-Hot Encoding**: Converts class IDs to probability vectors

### Loss Function

```python
Total Loss = λ₁ × Box Loss + λ₂ × Object Loss + λ₃ × Class Loss

Where:
- Box Loss: IoU/MSE loss for bounding box regression
- Object Loss: Binary cross-entropy for objectness
- Class Loss: Focal loss for classification (handles imbalance)
```

### Data Augmentation

- **Geometric**: Rotation (±30°), scaling (±20%), translation
- **Photometric**: Brightness, contrast, saturation adjustment
- **Edge Protection**: Ensures augmented boxes remain ≥50% visible
- **Mosaic**: Combines 4 images for better small object detection

## 🎯 Training Process

### Phase 1: Initialization (Epochs 1-5)
- Warmup learning rate schedule
- Basic augmentations only
- Focus on stable gradient flow

### Phase 2: Main Training (Epochs 6-80)
- Full augmentation pipeline
- Cosine learning rate decay
- Multi-scale training

### Phase 3: Fine-tuning (Epochs 81-100)
- Reduced learning rate
- Minimal augmentation
- Model convergence

## 📈 Monitoring Training

### Loss Components

```bash
# Training progress example
Epoch 25/100
Batch 50: Loss = 2.1543
  Box Loss: 0.3245
  Obj Loss: 1.2156
  Cls Loss: 0.6142
Train Loss: 2.0123, Val Loss: 2.1876
✓ Saved best model
```

### Key Metrics to Watch

1. **Total Loss**: Should decrease steadily
2. **Box Loss**: Bounding box accuracy (target: <0.5)
3. **Object Loss**: Object detection confidence
4. **Class Loss**: Classification accuracy (challenging due to imbalance)

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Reduce batch size
python run_yolo_training.py --batch-size 4
```

**Slow Training on CPU**
```bash
# Use smaller model or fewer epochs
python run_yolo_training.py --batch-size 2 --epochs 50
```

**Dataset Not Found**
```bash
# Check dataset path
python run_yolo_training.py --data /path/to/your/dataset
```

**Loss Not Decreasing**
```bash
# Check data loading and augmentation
# Verify label format and encoding
# Try different learning rate
```

## 🎨 Custom Datasets

To use your own dataset:

1. **Format**: Ensure YOLO format with `data.yaml`
2. **Structure**: Follow train/valid/test split
3. **Classes**: Update `num_classes` in config
4. **Anchors**: May need recomputation for different object scales

## 🏆 Expected Results

After 100 epochs with the underwater dataset:

- **mAP@0.5**: ~0.65-0.75 (varies by class)
- **Fish Detection**: High accuracy (>90%)
- **Starfish Detection**: Lower accuracy (~40%) due to scarcity
- **Inference Speed**: ~30-50 FPS on RTX 3080

## 🔮 Next Steps

1. **Model Export**: Convert to TensorFlow Lite for mobile deployment
2. **Inference Pipeline**: Create detection script for new images
3. **Model Ensemble**: Combine multiple models for better accuracy
4. **Data Collection**: Add more starfish samples to balance dataset

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify system requirements with `--check-only`
3. Review training logs in output directory
4. Ensure dataset follows YOLO format exactly
