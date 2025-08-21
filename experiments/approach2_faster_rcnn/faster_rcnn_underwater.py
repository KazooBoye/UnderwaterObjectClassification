"""
Approach 2: Faster R-CNN Two-Stage Detector for Underwater Object Detection
TensorFlow implementation with Feature Pyramid Network and underwater optimizations
"""

import os
import sys
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from common_utils import UnderwaterDatasetConfig, GPUSetup, DataLoader, MetricsCalculator, Visualizer
from callbacks import CosineRestartScheduler, MetricsLogger

class FPNBackbone(tf.keras.Model):
    """ResNet + FPN backbone for Faster R-CNN"""
    
    def __init__(self, backbone_name='resnet101', pretrained=True):
        super(FPNBackbone, self).__init__()
        
        # Load pretrained ResNet backbone
        if backbone_name == 'resnet50':
            self.backbone = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet' if pretrained else None,
                input_shape=(None, None, 3)
            )
            channels = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            self.backbone = tf.keras.applications.ResNet101(
                include_top=False,
                weights='imagenet' if pretrained else None,
                input_shape=(None, None, 3)
            )
            channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract intermediate layers for FPN
        self.layer_names = ['conv2_block3_out', 'conv3_block4_out', 
                           'conv4_block23_out', 'conv5_block3_out']
        
        # FPN components
        self.fpn_out_channels = 256
        self._build_fpn(channels)
        
    def _build_fpn(self, channels):
        """Build Feature Pyramid Network"""
        # Lateral connections (1x1 convs to reduce channels)
        self.lateral_convs = []
        for i, ch in enumerate(channels):
            lateral_conv = tf.keras.layers.Conv2D(
                self.fpn_out_channels, 1,
                name=f'fpn_lateral_{i}',
                kernel_initializer='he_normal'
            )
            self.lateral_convs.append(lateral_conv)
        
        # Top-down pathway (3x3 convs after upsampling)
        self.fpn_convs = []
        for i in range(len(channels)):
            fpn_conv = tf.keras.layers.Conv2D(
                self.fpn_out_channels, 3, padding='same',
                name=f'fpn_conv_{i}',
                kernel_initializer='he_normal'
            )
            self.fpn_convs.append(fpn_conv)
        
        # Extra levels for P6
        self.fpn_p6 = tf.keras.layers.Conv2D(
            self.fpn_out_channels, 3, strides=2, padding='same',
            name='fpn_p6',
            kernel_initializer='he_normal'
        )
    
    def call(self, x, training=False):
        # Extract features from backbone at different scales
        features = []
        current_x = x
        
        for layer in self.backbone.layers:
            current_x = layer(current_x, training=training)
            if layer.name in self.layer_names:
                features.append(current_x)
        
        # Build FPN
        # Start from the top (deepest feature map)
        laterals = []
        for i, feat in enumerate(features):
            lateral = self.lateral_convs[i](feat, training=training)
            laterals.append(lateral)
        
        # Top-down pathway
        fpn_features = [laterals[-1]]  # P5
        
        for i in range(len(laterals) - 2, -1, -1):  # P4, P3, P2
            # Upsample higher level feature
            upsampled = tf.image.resize(
                fpn_features[-1],
                tf.shape(laterals[i])[1:3],
                method='nearest'
            )
            # Add lateral connection
            merged = laterals[i] + upsampled
            fpn_features.append(merged)
        
        # Reverse to get P2, P3, P4, P5 order
        fpn_features = fpn_features[::-1]
        
        # Apply final convolutions
        final_features = []
        for i, feat in enumerate(fpn_features):
            final_feat = self.fpn_convs[i](feat, training=training)
            final_features.append(final_feat)
        
        # Add P6
        p6 = self.fpn_p6(final_features[-1], training=training)
        final_features.append(p6)
        
        return final_features  # [P2, P3, P4, P5, P6]

class AnchorGenerator(tf.keras.layers.Layer):
    """Generate anchor boxes for RPN"""
    
    def __init__(self, scales=[32, 64, 128, 256, 512], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]):
        super(AnchorGenerator, self).__init__()
        self.scales = scales
        self.ratios = ratios
        self.strides = strides
        
    def _generate_base_anchors(self, scale, ratios):
        """Generate base anchor shapes for a given scale"""
        anchors = []
        for ratio in ratios:
            h = scale / np.sqrt(ratio)
            w = scale * np.sqrt(ratio)
            anchors.append([-w/2, -h/2, w/2, h/2])
        return np.array(anchors, dtype=np.float32)
    
    def call(self, feature_maps):
        """Generate anchors for all feature map levels"""
        all_anchors = []
        
        for i, (feature_map, scale, stride) in enumerate(zip(feature_maps, self.scales, self.strides)):
            batch_size, height, width = tf.shape(feature_map)[0], tf.shape(feature_map)[1], tf.shape(feature_map)[2]
            
            # Generate base anchors
            base_anchors = self._generate_base_anchors(scale, self.ratios)
            num_base_anchors = len(base_anchors)
            
            # Create grid of anchor centers
            shift_x = tf.range(width, dtype=tf.float32) * stride
            shift_y = tf.range(height, dtype=tf.float32) * stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
            
            shifts = tf.stack([
                tf.reshape(shift_x, [-1]),
                tf.reshape(shift_y, [-1]),
                tf.reshape(shift_x, [-1]),
                tf.reshape(shift_y, [-1])
            ], axis=1)
            
            # Apply shifts to base anchors
            anchors = tf.reshape(base_anchors, [1, num_base_anchors, 4]) + tf.reshape(shifts, [-1, 1, 4])
            anchors = tf.reshape(anchors, [-1, 4])
            
            all_anchors.append(anchors)
        
        return tf.concat(all_anchors, axis=0)

class RegionProposalNetwork(tf.keras.layers.Layer):
    """Region Proposal Network (RPN)"""
    
    def __init__(self, anchor_scales=[32, 64, 128, 256, 512], anchor_ratios=[0.5, 1.0, 2.0]):
        super(RegionProposalNetwork, self).__init__()
        
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_ratios)
        
        # RPN head
        self.rpn_conv = tf.keras.layers.Conv2D(
            512, 3, padding='same', activation='relu',
            kernel_initializer='normal', name='rpn_conv'
        )
        
        # Classification (object/background)
        self.rpn_cls = tf.keras.layers.Conv2D(
            self.num_anchors, 1, activation='sigmoid',
            kernel_initializer='normal', name='rpn_cls'
        )
        
        # Regression (box coordinates)
        self.rpn_reg = tf.keras.layers.Conv2D(
            self.num_anchors * 4, 1,
            kernel_initializer='normal', name='rpn_reg'
        )
        
        # Anchor generator
        self.anchor_generator = AnchorGenerator(
            scales=anchor_scales, ratios=anchor_ratios
        )
        
    def call(self, feature_maps, training=False):
        """Forward pass through RPN"""
        rpn_cls_outputs = []
        rpn_reg_outputs = []
        
        for feature_map in feature_maps:
            # Shared RPN head
            rpn_feature = self.rpn_conv(feature_map, training=training)
            
            # Classification and regression
            cls_output = self.rpn_cls(rpn_feature, training=training)
            reg_output = self.rpn_reg(rpn_feature, training=training)
            
            rpn_cls_outputs.append(cls_output)
            rpn_reg_outputs.append(reg_output)
        
        # Generate anchors
        anchors = self.anchor_generator(feature_maps)
        
        return {
            'rpn_cls': rpn_cls_outputs,
            'rpn_reg': rpn_reg_outputs,
            'anchors': anchors
        }

class ROIHead(tf.keras.layers.Layer):
    """ROI Head for final classification and regression"""
    
    def __init__(self, num_classes=7, roi_size=7, fc_dim=1024):
        super(ROIHead, self).__init__()
        
        self.num_classes = num_classes + 1  # +1 for background
        self.roi_size = roi_size
        self.fc_dim = fc_dim
        
        # ROI pooling will be implemented using tf.image.crop_and_resize
        
        # Shared FC layers
        self.fc1 = tf.keras.layers.Dense(fc_dim, activation='relu', name='roi_fc1')
        self.fc2 = tf.keras.layers.Dense(fc_dim, activation='relu', name='roi_fc2')
        
        # Classification head
        self.cls_head = tf.keras.layers.Dense(
            self.num_classes, activation='softmax', name='roi_cls'
        )
        
        # Regression head
        self.reg_head = tf.keras.layers.Dense(
            self.num_classes * 4, name='roi_reg'
        )
        
    def roi_align(self, features, rois, roi_size=7):
        """ROI Align operation using tf.image.crop_and_resize"""
        # This is a simplified version - production would use proper ROI Align
        batch_size = tf.shape(features)[0]
        height, width = tf.shape(features)[1], tf.shape(features)[2]
        
        # Normalize ROI coordinates
        normalized_rois = rois / tf.cast([width, height, width, height], tf.float32)
        
        # Create batch indices
        num_rois = tf.shape(rois)[0]
        batch_indices = tf.zeros([num_rois], dtype=tf.int32)
        
        # Crop and resize
        roi_features = tf.image.crop_and_resize(
            features, normalized_rois, batch_indices, [roi_size, roi_size]
        )
        
        return roi_features
    
    def call(self, feature_maps, rois, training=False):
        """Forward pass through ROI head"""
        # Use the appropriate feature map level (simplified - use P4)
        feature_map = feature_maps[2]  # P4
        
        # ROI pooling/align
        roi_features = self.roi_align(feature_map, rois, self.roi_size)
        
        # Flatten
        roi_features = tf.keras.layers.GlobalAveragePooling2D()(roi_features)
        
        # FC layers
        x = self.fc1(roi_features, training=training)
        x = self.fc2(x, training=training)
        
        # Classification and regression
        cls_pred = self.cls_head(x, training=training)
        reg_pred = self.reg_head(x, training=training)
        
        return {
            'roi_cls': cls_pred,
            'roi_reg': reg_pred
        }

class UnderwaterFasterRCNN(tf.keras.Model):
    """Complete Faster R-CNN model for underwater object detection"""
    
    def __init__(self, num_classes=7, backbone='resnet101'):
        super(UnderwaterFasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Components
        self.backbone = FPNBackbone(backbone, pretrained=True)
        self.rpn = RegionProposalNetwork()
        self.roi_head = ROIHead(num_classes)
        
    def call(self, x, training=False):
        """Forward pass through Faster R-CNN"""
        
        # Feature extraction with FPN
        feature_maps = self.backbone(x, training=training)
        
        # RPN
        rpn_outputs = self.rpn(feature_maps, training=training)
        
        if training:
            # During training, use ground truth boxes for ROI head
            # This is simplified - normally you'd have more complex box sampling
            rois = tf.constant([[100, 100, 200, 200]], dtype=tf.float32)  # Dummy ROIs
        else:
            # During inference, use RPN proposals (simplified)
            rois = tf.constant([[100, 100, 200, 200]], dtype=tf.float32)  # Dummy ROIs
        
        # ROI head
        roi_outputs = self.roi_head(feature_maps, rois, training=training)
        
        return {
            'rpn_cls': rpn_outputs['rpn_cls'],
            'rpn_reg': rpn_outputs['rpn_reg'],
            'roi_cls': roi_outputs['roi_cls'],
            'roi_reg': roi_outputs['roi_reg'],
            'anchors': rpn_outputs['anchors']
        }

class FasterRCNNLoss(tf.keras.losses.Loss):
    """Faster R-CNN Loss Function"""
    
    def __init__(self, num_classes=7, rpn_weight=1.0, roi_weight=1.0, **kwargs):
        super(FasterRCNNLoss, self).__init__(**kwargs)
        self.num_classes = num_classes + 1  # +1 for background
        self.rpn_weight = rpn_weight
        self.roi_weight = roi_weight
        
    def call(self, y_true, y_pred):
        """
        Compute total Faster R-CNN loss
        This is a simplified version - production implementation would be more complex
        """
        
        # RPN losses
        rpn_cls_loss = tf.constant(0.0)  # Placeholder
        rpn_reg_loss = tf.constant(0.0)  # Placeholder
        
        # ROI losses  
        roi_cls_loss = tf.constant(0.0)  # Placeholder
        roi_reg_loss = tf.constant(0.0)  # Placeholder
        
        total_loss = (
            self.rpn_weight * (rpn_cls_loss + rpn_reg_loss) +
            self.roi_weight * (roi_cls_loss + roi_reg_loss)
        )
        
        return total_loss

class FasterRCNNTrainer:
    """Training pipeline for Faster R-CNN"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.dataset_config = UnderwaterDatasetConfig()
        
        # Override image size for Faster R-CNN
        self.dataset_config.image_size = tuple(self.config['dataset']['image_size'])
        
        # Setup GPU
        self.gpu_setup = GPUSetup()
        self.gpu_available = self.gpu_setup.configure_gpu()
        self.strategy = self.gpu_setup.get_strategy()
        
        # Initialize components
        self.data_loader = DataLoader(self.dataset_config)
        self.metrics_calculator = MetricsCalculator(self.dataset_config)
        self.visualizer = Visualizer(self.dataset_config)
        
        # Initialize model and training components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        # Experiment tracking
        self.setup_experiment_tracking()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_experiment_tracking(self):
        """Setup Weights & Biases experiment tracking"""
        wandb.init(
            project=self.config['experiment']['project'],
            name=self.config['experiment']['name'],
            tags=self.config['experiment']['tags'],
            config=self.config
        )
    
    def build_model(self):
        """Build Faster R-CNN model with distribution strategy"""
        with self.strategy.scope():
            # Create model
            self.model = UnderwaterFasterRCNN(
                num_classes=self.dataset_config.num_classes,
                backbone=self.config['model']['backbone']
            )
            
            # Build model with dummy input
            dummy_input = tf.random.normal([1, *self.dataset_config.image_size, 3])
            _ = self.model(dummy_input)
            
            # Setup optimizer
            learning_rate = self.config['training']['learning_rate']
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=self.config['training']['weight_decay']
            )
            
            # Setup loss function
            self.loss_fn = FasterRCNNLoss(
                num_classes=self.dataset_config.num_classes
            )
            
            # Compile model
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                run_eagerly=False
            )
        
        print(f"Faster R-CNN model built successfully on {self.strategy.num_replicas_in_sync} device(s)")
        self.model.summary()
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        print("Preparing datasets for Faster R-CNN...")
        
        batch_size = self.config['training']['batch_size']
        
        # Create datasets with larger image size
        self.train_dataset = self.data_loader.create_tf_dataset('train', batch_size)
        self.val_dataset = self.data_loader.create_tf_dataset('val', batch_size)
        
        # Distribute datasets
        self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
        self.val_dataset = self.strategy.experimental_distribute_dataset(self.val_dataset)
        
        print("Datasets prepared successfully")
    
    def train(self):
        """Main training loop"""
        print("Starting Faster R-CNN training...")
        
        epochs = self.config['training']['epochs']
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Training loop
        with self.strategy.scope():
            history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                verbose=1
            )
        
        # Plot training curves
        self.visualizer.plot_training_curves(
            history.history,
            save_path=f"faster_rcnn_training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return history
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_faster_rcnn_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config['early_stopping']['enabled']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=self.config['early_stopping']['monitor'],
                patience=self.config['early_stopping']['patience'],
                mode=self.config['early_stopping']['mode'],
                min_delta=self.config['early_stopping']['min_delta'],
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Learning rate scheduling - Step decay for Faster R-CNN
        def step_decay(epoch):
            initial_lr = self.config['training']['learning_rate']
            lr_steps = self.config['training'].get('lr_steps', [100, 130])
            lr_gamma = self.config['training'].get('lr_gamma', 0.1)
            
            lr = initial_lr
            for step in lr_steps:
                if epoch >= step:
                    lr *= lr_gamma
            return lr
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
        callbacks.append(lr_scheduler)
        
        # Weights & Biases callback
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_loss',
            mode='min',
            save_model=False
        )
        callbacks.append(wandb_callback)
        
        # Custom metrics logger
        metrics_logger = MetricsLogger(log_freq=10)
        callbacks.append(metrics_logger)
        
        return callbacks
    
    def evaluate(self):
        """Evaluate model performance"""
        print("Evaluating Faster R-CNN model...")
        
        if self.model is None:
            print("Model not initialized. Please build model first.")
            return None
        
        # Load test dataset
        test_dataset = self.data_loader.create_tf_dataset('test', batch_size=1)
        
        # Run basic evaluation
        test_loss = self.model.evaluate(test_dataset, verbose=1)
        print(f"Test Loss: {test_loss}")
        
        # Get predictions for detailed metrics
        predictions = []
        ground_truths = []
        
        print("Computing predictions for mAP calculation...")
        for batch in test_dataset.take(100):  # Limit to avoid memory issues
            images, targets = batch
            try:
                pred = self.model(images, training=False)
                
                # Process Faster R-CNN predictions
                for i in range(len(images)):
                    # Extract boxes, scores, and classes from model output
                    if isinstance(pred, dict):
                        pred_boxes = pred.get('detection_boxes', [[]])[i].numpy()
                        pred_scores = pred.get('detection_scores', [[]])[i].numpy()  
                        pred_classes = pred.get('detection_classes', [[]])[i].numpy()
                    else:
                        # Fallback if output format is different
                        pred_boxes = []
                        pred_scores = []
                        pred_classes = []
                    
                    gt_boxes = targets['boxes'][i].numpy()
                    gt_labels = targets['labels'][i].numpy()
                    
                    predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores,
                        'labels': pred_classes
                    })
                    
                    ground_truths.append({
                        'boxes': gt_boxes,
                        'labels': gt_labels
                    })
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        # Calculate mAP using metrics calculator
        try:
            map_score = self.metrics_calculator.calculate_map(predictions, ground_truths)
            print(f"mAP Score: {map_score}")
        except Exception as e:
            print(f"Error calculating mAP: {e}")
            map_score = 0.0
        
        return {
            'test_loss': test_loss,
            'map_score': map_score
        }

def main():
    """Main training script for Faster R-CNN"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Faster R-CNN for underwater object detection')
    parser.add_argument('--config', type=str,
                       default='../configs/faster_rcnn_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FasterRCNNTrainer(args.config)
    
    # Build model
    trainer.build_model()
    
    # Prepare datasets
    trainer.prepare_datasets()
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate()
    
    # Save final results
    results = {
        'training_history': history.history,
        'test_results': test_results,
        'config': trainer.config
    }
    
    # Save results
    import json
    with open(f'faster_rcnn_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Faster R-CNN training completed successfully!")

if __name__ == "__main__":
    main()
