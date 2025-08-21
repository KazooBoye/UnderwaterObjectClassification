#!/usr/bin/env python3
"""
YOLO Label Encoding Utilities
Converts YOLO format labels to multi-scale grid targets for training
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple
import math

class YOLOLabelEncoder:
    """Encode YOLO labels for multi-scale training"""
    
    def __init__(self, input_size: int = 640, num_classes: int = 7):
        self.input_size = input_size
        self.num_classes = num_classes
        
        # YOLO scales (strides)
        self.strides = [8, 16, 32]
        self.grid_sizes = [input_size // stride for stride in self.strides]  # [80, 40, 20]
        
        # Anchors for each scale (width, height) normalized to grid cell
        self.anchors = [
            [[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],    # Small objects (P3/8)
            [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],  # Medium objects (P4/16)  
            [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]  # Large objects (P5/32)
        ]
    
    def encode_labels(self, labels: np.ndarray, image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Encode YOLO format labels to grid targets
        
        Args:
            labels: Array of shape (N, 5) with [class_id, x_center, y_center, width, height]
                   All coordinates are normalized (0-1)
            image_shape: (height, width) of original image
            
        Returns:
            List of target tensors for each scale [P3, P4, P5]
            Each tensor shape: (grid_h, grid_w, num_anchors, 5 + num_classes)
        """
        targets = []
        
        # Process each scale
        for scale_idx, (grid_size, stride, scale_anchors) in enumerate(
            zip(self.grid_sizes, self.strides, self.anchors)
        ):
            # Initialize target tensor
            target = np.zeros((grid_size, grid_size, 3, 5 + self.num_classes), dtype=np.float32)
            
            # Process each ground truth box
            for label in labels:
                if len(label) < 5:
                    continue
                    
                class_id = int(label[0])
                x_center, y_center, width, height = label[1:5]
                
                # Convert to grid coordinates
                grid_x = x_center * grid_size
                grid_y = y_center * grid_size
                
                # Grid cell indices
                grid_i = int(grid_x)
                grid_j = int(grid_y)
                
                # Skip if outside grid
                if grid_i >= grid_size or grid_j >= grid_size:
                    continue
                
                # Relative position within grid cell
                x_offset = grid_x - grid_i
                y_offset = grid_y - grid_j
                
                # Convert width/height to grid scale
                grid_width = width * grid_size
                grid_height = height * grid_size
                
                # Find best anchor based on IoU
                best_anchor_idx = self._find_best_anchor(
                    grid_width, grid_height, scale_anchors
                )
                
                # Check if this cell/anchor is already occupied
                if target[grid_j, grid_i, best_anchor_idx, 4] == 0:  # Not occupied
                    # Box coordinates (relative to grid cell)
                    target[grid_j, grid_i, best_anchor_idx, 0] = x_offset
                    target[grid_j, grid_i, best_anchor_idx, 1] = y_offset
                    target[grid_j, grid_i, best_anchor_idx, 2] = grid_width
                    target[grid_j, grid_i, best_anchor_idx, 3] = grid_height
                    
                    # Objectness
                    target[grid_j, grid_i, best_anchor_idx, 4] = 1.0
                    
                    # Class (one-hot encoding)
                    target[grid_j, grid_i, best_anchor_idx, 5 + class_id] = 1.0
            
            targets.append(target)
        
        return targets
    
    def _find_best_anchor(self, box_width: float, box_height: float, 
                         anchors: List[List[float]]) -> int:
        """Find the anchor with highest IoU with the ground truth box"""
        best_iou = 0
        best_anchor_idx = 0
        
        for idx, (anchor_w, anchor_h) in enumerate(anchors):
            # Calculate IoU between box and anchor
            iou = self._calculate_iou_wh(box_width, box_height, anchor_w, anchor_h)
            
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = idx
        
        return best_anchor_idx
    
    def _calculate_iou_wh(self, w1: float, h1: float, w2: float, h2: float) -> float:
        """Calculate IoU between two boxes given only width and height"""
        # Intersection
        intersection_w = min(w1, w2)
        intersection_h = min(h1, h2)
        intersection = intersection_w * intersection_h
        
        # Union
        union = w1 * h1 + w2 * h2 - intersection
        
        # IoU
        if union <= 0:
            return 0
        return intersection / union

def create_label_encoding_function(encoder: YOLOLabelEncoder):
    """Create TensorFlow function for label encoding"""
    
    @tf.function
    def encode_batch_labels(labels_batch):
        """Encode a batch of labels using tf.py_function"""
        
        def encode_single_label(labels):
            # Convert tensor to numpy
            labels_np = labels.numpy()
            
            # Encode labels
            encoded = encoder.encode_labels(labels_np, (640, 640))
            
            # Convert back to tensors
            return [tf.constant(target, dtype=tf.float32) for target in encoded]
        
        # Apply to each sample in batch
        encoded_batch = tf.map_fn(
            lambda x: tf.py_function(
                encode_single_label, 
                [x], 
                [tf.float32, tf.float32, tf.float32]
            ),
            labels_batch,
            dtype=[tf.float32, tf.float32, tf.float32]
        )
        
        return encoded_batch
    
    return encode_batch_labels

# Example usage function
def test_label_encoder():
    """Test the label encoder with sample data"""
    encoder = YOLOLabelEncoder()
    
    # Sample labels: [class_id, x_center, y_center, width, height]
    sample_labels = np.array([
        [0, 0.5, 0.3, 0.2, 0.4],    # Fish
        [1, 0.8, 0.7, 0.1, 0.1],    # Jellyfish
        [4, 0.2, 0.6, 0.3, 0.2]     # Shark
    ])
    
    # Encode labels
    targets = encoder.encode_labels(sample_labels, (640, 640))
    
    print("Label encoding test:")
    for i, target in enumerate(targets):
        print(f"Scale {i}: {target.shape}")
        # Count non-zero objectness targets
        obj_count = np.sum(target[:, :, :, 4] > 0)
        print(f"  Objects assigned: {obj_count}")
    
    return targets

if __name__ == '__main__':
    # Run test
    test_targets = test_label_encoder()
    
    print("\nLabel encoder test completed successfully!")
    print("The encoder is ready for use in YOLO training.")
