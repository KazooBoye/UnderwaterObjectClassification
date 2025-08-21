#!/usr/bin/env python3
"""
YOLOv8-style Object Detection with TensorFlow
Optimized for underwater multi-class object detection with CUDA acceleration
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import cv2
import yaml
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime

# Import our custom label encoder
from yolo_label_encoder import YOLOLabelEncoder, create_label_encoding_function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOConfig:
    """YOLO Configuration"""
    def __init__(self):
        self.input_size = 640
        self.num_classes = 7
        self.max_boxes = 100
        self.iou_threshold = 0.5
        self.confidence_threshold = 0.25
        self.class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        
        # Training parameters
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.001
        self.warmup_epochs = 5
        
        # YOLO grid parameters
        self.strides = [8, 16, 32]  # P3, P4, P5
        self.anchors_per_scale = 3
        self.grid_sizes = [80, 40, 20]  # For 640x640 input
        
        # Loss weights
        self.box_loss_weight = 0.05
        self.cls_loss_weight = 0.5
        self.obj_loss_weight = 1.0

def setup_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU acceleration enabled. Found {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            logger.warning(f"GPU setup failed: {e}")
            return False
    else:
        logger.info("No GPU found, using CPU")
        return False

class CSPDarknet(Model):
    """CSPDarknet backbone for YOLO"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Stem
        self.stem = keras.Sequential([
            layers.Conv2D(32, 3, 2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.Conv2D(64, 3, 2, padding='same', use_bias=False),
            layers.BatchNormalization(), 
            layers.Activation('swish')
        ])
        
        # CSP stages
        self.stage1 = self._make_csp_stage(64, 128, 3, 2)
        self.stage2 = self._make_csp_stage(128, 256, 6, 2)
        self.stage3 = self._make_csp_stage(256, 512, 9, 2)
        self.stage4 = self._make_csp_stage(512, 1024, 3, 2)
        
    def _make_csp_stage(self, in_channels, out_channels, num_blocks, stride):
        """Create CSP stage"""
        return keras.Sequential([
            layers.Conv2D(out_channels, 3, stride, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            *[self._make_csp_block(out_channels) for _ in range(num_blocks)]
        ])
    
    def _make_csp_block(self, channels):
        """CSP Bottleneck block"""
        return keras.Sequential([
            layers.Conv2D(channels//2, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.Conv2D(channels//2, 3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish'),
            layers.Conv2D(channels, 1, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])
    
    def call(self, x, training=None):
        x = self.stem(x)
        c1 = x
        
        x = self.stage1(x)
        c2 = x  # P2/4
        
        x = self.stage2(x)  
        c3 = x  # P3/8
        
        x = self.stage3(x)
        c4 = x  # P4/16
        
        x = self.stage4(x)
        c5 = x  # P5/32
        
        return c3, c4, c5

class YOLOFPN(Model):
    """Feature Pyramid Network for YOLO"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Top-down pathway - lateral convs reduce channels from higher to lower level
        self.lateral_conv1 = layers.Conv2D(512, 1, use_bias=False)  # Reduce c5 (1024) to match c4 (512)
        self.lateral_conv2 = layers.Conv2D(256, 1, use_bias=False)  # Reduce p4 (512) to match c3 (256)
        
        # Bottom-up pathway - upsample convs increase channels to match target levels
        self.downsample_conv1 = layers.Conv2D(512, 3, 2, padding='same', use_bias=False)  # p3 (256) -> match p4 (512)
        self.downsample_conv2 = layers.Conv2D(1024, 3, 2, padding='same', use_bias=False)  # p4 (512) -> match p5 (1024)
        
        # Output convolutions
        self.output_conv1 = keras.Sequential([
            layers.Conv2D(256, 3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])
        
        self.output_conv2 = keras.Sequential([
            layers.Conv2D(512, 3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])
        
        self.output_conv3 = keras.Sequential([
            layers.Conv2D(1024, 3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('swish')
        ])
        
        self.upsample = layers.UpSampling2D(2)
    
    def call(self, features, training=None):
        c3, c4, c5 = features
        
        # Top-down pathway with proper channel matching
        p5 = c5  # [B, 10, 10, 1024]
        
        # For p4: reduce c5 channels to match c4, then upsample and add
        p5_reduced = self.lateral_conv1(p5)  # [B, 10, 10, 1024] → [B, 10, 10, 512] 
        p4 = c4 + self.upsample(p5_reduced)  # [B, 20, 20, 512] + [B, 20, 20, 512]
        
        # For p3: reduce p4 channels to match c3, then upsample and add  
        p4_reduced = self.lateral_conv2(p4)  # [B, 20, 20, 512] → [B, 20, 20, 256]
        p3 = c3 + self.upsample(p4_reduced)  # [B, 40, 40, 256] + [B, 40, 40, 256]
        
        # Bottom-up pathway with proper channel matching
        n3 = p3  # [B, 40, 40, 256]
        n4 = self.downsample_conv1(n3) + p4  # [B, 20, 20, 512] + [B, 20, 20, 512]
        n5 = self.downsample_conv2(n4) + p5  # [B, 10, 10, 1024] + [B, 10, 10, 1024]
        
        # Apply output convolutions
        p3_out = self.output_conv1(n3)  # 80x80
        p4_out = self.output_conv2(n4)  # 40x40 
        p5_out = self.output_conv3(n5)  # 20x20
        
        return p3_out, p4_out, p5_out

class YOLOHead(Model):
    """YOLO Detection Head"""
    
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Create separate heads for different feature map sizes
        # Each head handles different input channels (256, 512, 1024)
        self.heads = []
        for channels in [256, 512, 1024]:  # P3, P4, P5 channels
            head = keras.Sequential([
                # First reduce all channels to 256 for consistency
                layers.Conv2D(256, 1, use_bias=False),
                layers.BatchNormalization(),
                layers.Activation('swish'),
                # Shared processing
                layers.Conv2D(256, 3, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Activation('swish'),
                layers.Conv2D(256, 3, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.Activation('swish'),
                # Final detection output
                layers.Conv2D(
                    3 * (5 + num_classes),  # 3 anchors per grid cell
                    1,
                    bias_initializer='zeros'
                )
            ])
            self.heads.append(head)
    
    def call(self, features, training=None):
        """Process all FPN features"""
        outputs = []
        for i, feature in enumerate(features):
            output = self.heads[i](feature, training=training)
            
            # Reshape to [batch, grid_h, grid_w, anchors, (5 + num_classes)]
            batch_size = tf.shape(output)[0]
            grid_h = tf.shape(output)[1]
            grid_w = tf.shape(output)[2]
            
            output = tf.reshape(output, [batch_size, grid_h, grid_w, 3, 5 + self.num_classes])
            outputs.append(output)
        
        return outputs

class YOLOModel(Model):
    """Complete YOLO Model"""
    
    def __init__(self, config: YOLOConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        self.backbone = CSPDarknet()
        self.fpn = YOLOFPN()
        self.head = YOLOHead(config.num_classes)
    
    def call(self, x, training=None):
        # Backbone feature extraction
        features = self.backbone(x, training=training)
        
        # FPN
        fpn_features = self.fpn(features, training=training)
        
        # Detection heads - head now processes all features at once
        outputs = self.head(fpn_features, training=training)
        
        return outputs

class YOLOLoss:
    """YOLO Loss Function"""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.bce_loss = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
        self.mse_loss = keras.losses.MeanSquaredError(reduction='none')
    
    def __call__(self, y_true, y_pred):
        """
        y_true: List of ground truth tensors for each scale
        y_pred: List of prediction tensors for each scale
        """
        total_loss = 0.0
        box_loss = 0.0
        obj_loss = 0.0 
        cls_loss = 0.0
        
        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            # Extract components
            true_box = true[..., 0:4]      # x, y, w, h
            true_obj = true[..., 4:5]      # objectness
            true_cls = true[..., 5:]       # class probabilities
            
            pred_box = pred[..., 0:4]      # x, y, w, h
            pred_obj = pred[..., 4:5]      # objectness
            pred_cls = pred[..., 5:]       # class logits
            
            # Object mask (where objects exist)
            obj_mask = true_obj
            noobj_mask = 1.0 - obj_mask
            
            # Box loss (only for cells with objects)
            box_loss_i = tf.reduce_sum(
                obj_mask * self.mse_loss(true_box, pred_box)
            )
            box_loss += box_loss_i
            
            # Objectness loss
            obj_loss_positive = tf.reduce_sum(
                obj_mask * self.bce_loss(true_obj, pred_obj)
            )
            obj_loss_negative = tf.reduce_sum(
                noobj_mask * self.bce_loss(true_obj, pred_obj)
            )
            obj_loss += obj_loss_positive + 0.5 * obj_loss_negative
            
            # Classification loss (only for cells with objects)
            cls_loss_i = tf.reduce_sum(
                obj_mask * self.bce_loss(true_cls, pred_cls)
            )
            cls_loss += cls_loss_i
        
        # Weighted total loss
        total_loss = (
            self.config.box_loss_weight * box_loss +
            self.config.obj_loss_weight * obj_loss +
            self.config.cls_loss_weight * cls_loss
        )
        
        return {
            'total_loss': total_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss
        }

class YOLODataLoader:
    """Data loader for YOLO format"""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.input_size = config.input_size
    
    def load_data(self, data_path: str):
        """Load YOLO format dataset"""
        with open(os.path.join(data_path, 'data.yaml'), 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Load training data
        train_images = self._load_images(os.path.join(data_path, 'train/images'))
        train_labels = self._load_labels(os.path.join(data_path, 'train/labels'))
        
        # Load validation data
        val_images = self._load_images(os.path.join(data_path, 'valid/images'))
        val_labels = self._load_labels(os.path.join(data_path, 'valid/labels'))
        
        return (train_images, train_labels), (val_images, val_labels)
    
    def _load_images(self, image_dir: str) -> List[str]:
        """Load image file paths"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        
        for ext in image_extensions:
            images.extend(Path(image_dir).glob(f'*{ext}'))
        
        return [str(img) for img in sorted(images)]
    
    def _load_labels(self, label_dir: str) -> List[str]:
        """Load label file paths"""
        labels = list(Path(label_dir).glob('*.txt'))
        return [str(label) for label in sorted(labels)]
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess single image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize with padding to maintain aspect ratio
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_h = (self.input_size - new_h) // 2
        pad_w = (self.input_size - new_w) // 2
        
        image = np.pad(image, ((pad_h, self.input_size - new_h - pad_h),
                              (pad_w, self.input_size - new_w - pad_w),
                              (0, 0)), mode='constant', constant_values=114)
        
        return image.astype(np.float32) / 255.0
    
    def preprocess_labels(self, label_path: str) -> np.ndarray:
        """Preprocess YOLO format labels"""
        if not os.path.exists(label_path):
            return np.zeros((0, 5))  # No labels
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return np.zeros((0, 5))
        
        labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                labels.append([class_id, x, y, w, h])
        
        return np.array(labels) if labels else np.zeros((0, 5))

def create_training_dataset(data_loader: YOLODataLoader, images: List[str], labels: List[str], 
                          config: YOLOConfig) -> tf.data.Dataset:
    """Create TensorFlow dataset for training"""
    
    # Create label encoder
    label_encoder = YOLOLabelEncoder(config.input_size, config.num_classes)
    
    def generator():
        for img_path, lbl_path in zip(images, labels):
            image = data_loader.preprocess_image(img_path)
            label = data_loader.preprocess_labels(lbl_path)
            
            # Encode labels to multi-scale targets
            if len(label) > 0:
                targets = label_encoder.encode_labels(label, (config.input_size, config.input_size))
            else:
                # Empty targets for images with no objects
                targets = [
                    np.zeros((gs, gs, 3, 5 + config.num_classes), dtype=np.float32)
                    for gs in label_encoder.grid_sizes
                ]
            
            yield (image, (targets[0], targets[1], targets[2]))
    
    # Create dataset with proper output signature
    output_signature = (
        tf.TensorSpec(shape=(config.input_size, config.input_size, 3), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(80, 80, 3, 5 + config.num_classes), dtype=tf.float32),
            tf.TensorSpec(shape=(40, 40, 3, 5 + config.num_classes), dtype=tf.float32),
            tf.TensorSpec(shape=(20, 20, 3, 5 + config.num_classes), dtype=tf.float32)
        )
    )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    
    # Apply transformations
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_yolo_model(data_path: str, output_dir: str = 'yolo_training_output'):
    """Main training function"""
    
    # Setup
    setup_gpu()
    config = YOLOConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    logger.info("Loading dataset...")
    data_loader = YOLODataLoader(config)
    (train_images, train_labels), (val_images, val_labels) = data_loader.load_data(data_path)
    
    logger.info(f"Training samples: {len(train_images)}")
    logger.info(f"Validation samples: {len(val_images)}")
    
    # Create datasets
    train_dataset = create_training_dataset(data_loader, train_images, train_labels, config)
    val_dataset = create_training_dataset(data_loader, val_images, val_labels, config)
    
    # Create model
    logger.info("Creating YOLO model...")
    model = YOLOModel(config)
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, config.input_size, config.input_size, 3))
    _ = model(dummy_input)
    
    logger.info(f"Model created with {model.count_params():,} parameters")
    
    # Create optimizer with learning rate schedule
    initial_learning_rate = config.learning_rate
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps=config.epochs * len(train_images) // config.batch_size,
        alpha=0.1
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Loss function
    loss_fn = YOLOLoss(config)
    
    # Metrics
    train_loss_metric = keras.metrics.Mean()
    val_loss_metric = keras.metrics.Mean()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        # Training phase
        train_loss_metric.reset_state()
        
        for batch_idx, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                # Note: Need to implement label encoding for multi-scale targets
                # This is a simplified version - full implementation would require
                # encoding labels for each scale
                loss_dict = loss_fn(labels, predictions)  # Simplified
                total_loss = loss_dict['total_loss']
            
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss_metric.update_state(total_loss)
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: Loss = {total_loss:.4f}")
        
        # Validation phase
        val_loss_metric.reset_state()
        
        for images, labels in val_dataset:
            predictions = model(images, training=False)
            loss_dict = loss_fn(labels, predictions)  # Simplified
            val_loss_metric.update_state(loss_dict['total_loss'])
        
        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(os.path.join(output_dir, 'best_model.h5'))
            logger.info("Saved best model")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.save_weights(os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.h5'))
    
    logger.info("Training completed!")
    return model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLO model with TensorFlow')
    parser.add_argument('--data', type=str, default='aquarium_pretrain', 
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='yolo_training_output',
                       help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config = YOLOConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
    # Start training
    train_yolo_model(args.data, args.output)
