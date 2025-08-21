"""
Shared utilities for underwater object detection experiments
Handles data loading, preprocessing, and evaluation for all three approaches
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnderwaterDatasetConfig:
    """Configuration class for underwater dataset"""
    
    def __init__(self, config_path: str = None):
        self.dataset_root = "/Users/caoducanh/Desktop/Coding/UnderwaterObjectClassification/preprocessed_dataset"
        self.classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        self.num_classes = 7
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
        
        # Class weights from preprocessing analysis
        self.class_weights = {
            0: 0.258,   # fish
            1: 0.992,   # jellyfish
            2: 1.334,   # penguin
            3: 2.423,   # puffin
            4: 1.944,   # shark
            5: 5.932,   # starfish
            6: 3.740    # stingray
        }
        
        # Image settings
        self.image_size = (640, 640)  # Can be overridden per approach
        self.channels = 3
        
        # Training settings
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.epochs = 150
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                setattr(self, key, value)
    
    def get_data_yaml_path(self):
        """Get path to data.yaml file"""
        return os.path.join(self.dataset_root, "data_balanced_v1.yaml")

class GPUSetup:
    """Setup GPU acceleration for TensorFlow"""
    
    @staticmethod
    def configure_gpu():
        """Configure GPU for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth to avoid OOM errors
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Use mixed precision for better performance
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                logger.info(f"GPU setup completed. Found {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    logger.info(f"GPU {i}: {gpu}")
                
                return True
            except RuntimeError as e:
                logger.error(f"GPU setup failed: {e}")
                return False
        else:
            logger.warning("No GPU found. Using CPU.")
            return False
    
    @staticmethod
    def get_strategy():
        """Get distribution strategy for multi-GPU training"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            logger.info(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
        else:
            strategy = tf.distribute.get_strategy()  # Default strategy
            logger.info("Using default strategy (single GPU or CPU)")
        
        return strategy

class DataLoader:
    """Data loading utilities for YOLO format datasets"""
    
    def __init__(self, config: UnderwaterDatasetConfig):
        self.config = config
        self.train_path = os.path.join(config.dataset_root, "train_balanced_v1")
        self.val_path = os.path.join(config.dataset_root, "val_balanced_v1")
        self.test_path = os.path.join(config.dataset_root, "test_balanced_v1")
    
    def load_yolo_annotations(self, labels_dir: str, images_dir: str) -> List[Dict]:
        """Load YOLO format annotations"""
        annotations = []
        
        for label_file in Path(labels_dir).glob("*.txt"):
            image_file = Path(images_dir) / f"{label_file.stem}.jpg"
            
            if not image_file.exists():
                continue
            
            # Read image dimensions
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            height, width = image.shape[:2]
            
            # Parse annotations
            boxes = []
            labels = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        
                        # Convert to absolute coordinates
                        x_center *= width
                        y_center *= height
                        w *= width
                        h *= height
                        
                        # Convert to x1, y1, x2, y2
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
            
            annotations.append({
                'image_path': str(image_file),
                'image_id': label_file.stem,
                'width': width,
                'height': height,
                'boxes': boxes,
                'labels': labels
            })
        
        return annotations
    
    def create_tf_dataset(self, split: str = 'train', batch_size: int = None) -> tf.data.Dataset:
        """Create TensorFlow dataset"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if split == 'train':
            images_dir = os.path.join(self.train_path, 'images')
            labels_dir = os.path.join(self.train_path, 'labels')
        elif split == 'val':
            images_dir = os.path.join(self.val_path, 'images')
            labels_dir = os.path.join(self.val_path, 'labels')
        elif split == 'test':
            images_dir = os.path.join(self.test_path, 'images')
            labels_dir = os.path.join(self.test_path, 'labels')
        else:
            raise ValueError(f"Invalid split: {split}")
        
        annotations = self.load_yolo_annotations(labels_dir, images_dir)
        
        def generator():
            for ann in annotations:
                # Load and preprocess image
                image = cv2.imread(ann['image_path'])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.config.image_size)
                image = image.astype(np.float32) / 255.0
                
                # Prepare targets
                boxes = np.array(ann['boxes'], dtype=np.float32)
                labels = np.array(ann['labels'], dtype=np.int32)
                
                # Resize boxes to match resized image
                scale_x = self.config.image_size[0] / ann['width']
                scale_y = self.config.image_size[1] / ann['height']
                
                if len(boxes) > 0:
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                
                yield image, {
                    'boxes': boxes,
                    'labels': labels,
                    'image_id': ann['image_id']
                }
        
        # Create dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.config.image_size, 3), dtype=tf.float32),
                {
                    'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                    'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    'image_id': tf.TensorSpec(shape=(), dtype=tf.string)
                }
            )
        )
        
        if split == 'train':
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

class MetricsCalculator:
    """Calculate evaluation metrics for object detection"""
    
    def __init__(self, config: UnderwaterDatasetConfig):
        self.config = config
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_map(self, predictions: List[Dict], ground_truths: List[Dict], 
                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate mAP for object detection"""
        # This is a simplified mAP calculation
        # For production, consider using pycocotools or similar
        
        class_aps = {}
        
        for class_id in range(self.config.num_classes):
            class_name = self.config.id_to_class[class_id]
            
            # Collect all predictions and ground truths for this class
            all_predictions = []
            all_ground_truths = []
            
            for pred, gt in zip(predictions, ground_truths):
                # Filter by class
                pred_class = [p for p in pred if p['class_id'] == class_id]
                gt_class = [g for g in gt if g['class_id'] == class_id]
                
                all_predictions.extend(pred_class)
                all_ground_truths.extend(gt_class)
            
            if len(all_ground_truths) == 0:
                class_aps[class_name] = 0.0
                continue
            
            # Sort predictions by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision and recall
            true_positives = np.zeros(len(all_predictions))
            false_positives = np.zeros(len(all_predictions))
            
            for i, pred in enumerate(all_predictions):
                # Find best matching ground truth
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt in enumerate(all_ground_truths):
                    if gt.get('matched', False):
                        continue
                    
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_threshold:
                    true_positives[i] = 1
                    all_ground_truths[best_gt_idx]['matched'] = True
                else:
                    false_positives[i] = 1
            
            # Calculate precision and recall curves
            tp_cumsum = np.cumsum(true_positives)
            fp_cumsum = np.cumsum(false_positives)
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / len(all_ground_truths)
            
            # Calculate AP using 11-point interpolation
            ap = self._calculate_ap_11_point(precision, recall)
            class_aps[class_name] = ap
        
        # Calculate mAP
        map_score = np.mean(list(class_aps.values()))
        
        return {
            'mAP': map_score,
            'class_aps': class_aps
        }
    
    def _calculate_ap_11_point(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """Calculate AP using 11-point interpolation"""
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        return ap

class Visualizer:
    """Visualization utilities for experiments"""
    
    def __init__(self, config: UnderwaterDatasetConfig):
        self.config = config
        self.colors = plt.cm.Set3(np.linspace(0, 1, self.config.num_classes))
    
    def plot_training_curves(self, history: Dict, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        if 'loss' in history:
            axes[0, 0].plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # mAP curves
        if 'mAP' in history:
            axes[0, 1].plot(history['mAP'], label='mAP@0.5')
            if 'val_mAP' in history:
                axes[0, 1].plot(history['val_mAP'], label='Validation mAP@0.5')
            axes[0, 1].set_title('mAP Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].legend()
        
        # Learning rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
        
        # Additional metrics
        if 'precision' in history:
            axes[1, 1].plot(history['precision'], label='Precision')
            if 'recall' in history:
                axes[1, 1].plot(history['recall'], label='Recall')
            axes[1, 1].set_title('Precision/Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.config.classes,
                   yticklabels=self.config.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, image: np.ndarray, predictions: List[Dict],
                            ground_truths: List[Dict] = None, save_path: str = None):
        """Visualize predictions on image"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Show image
        ax.imshow(image)
        
        # Draw ground truth boxes in green
        if ground_truths:
            for gt in ground_truths:
                bbox = gt['bbox']
                class_id = gt['class_id']
                class_name = self.config.id_to_class[class_id]
                
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                   linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
                ax.text(bbox[0], bbox[1] - 5, f'GT: {class_name}', 
                       color='green', fontsize=8, fontweight='bold')
        
        # Draw prediction boxes in red
        for pred in predictions:
            bbox = pred['bbox']
            class_id = pred['class_id']
            confidence = pred['confidence']
            class_name = self.config.id_to_class[class_id]
            
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                               linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[3] + 5, f'{class_name}: {confidence:.2f}', 
                   color='red', fontsize=8, fontweight='bold')
        
        ax.set_title('Predictions vs Ground Truth')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Initialize GPU setup when module is imported
if __name__ == "__main__":
    gpu_setup = GPUSetup()
    gpu_available = gpu_setup.configure_gpu()
    print(f"GPU setup: {'Success' if gpu_available else 'Failed'}")
