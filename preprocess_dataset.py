#!/usr/bin/env python3
"""
Comprehensive Dataset Preprocessing Script for Underwater Object Classification
Implements all advanced strategies to handle class imbalance, multi-scale objects, and complex scenes.
"""

import os
import json
import shutil
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any
import yaml
import pickle

class UnderwaterDatasetPreprocessor:
    def __init__(self, dataset_path: str, output_path: str, config: Dict[str, Any]):
        """
        Initialize the preprocessor with dataset paths and configuration.
        
        Args:
            dataset_path: Path to original dataset
            output_path: Path for preprocessed output
            config: Configuration dictionary
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.config = config
        
        # Class names and mappings
        self.class_names = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Statistics tracking
        self.original_stats = {}
        self.processed_stats = {}
        
        # Create output directories
        self.setup_output_structure()
        
    def setup_output_structure(self):
        """Create the output directory structure."""
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (self.output_path / f"{split}_{self.config['output_suffix']}" / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create additional directories
        (self.output_path / "statistics").mkdir(parents=True, exist_ok=True)
        (self.output_path / "configs").mkdir(parents=True, exist_ok=True)
        
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the original dataset and gather statistics."""
        print("Analyzing original dataset...")
        
        splits_data = {}
        overall_stats = {
            'class_counts': Counter(),
            'bbox_areas': [],
            'aspect_ratios': [],
            'objects_per_image': [],
            'classes_per_image': []
        }
        
        for split in ['train', 'valid', 'test']:
            split_path = self.dataset_path / split
            if not split_path.exists():
                continue
                
            labels_path = split_path / 'labels'
            images_path = split_path / 'images'
            
            split_stats = {
                'files': 0,
                'class_counts': Counter(),
                'bbox_data': []
            }
            
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                split_stats['files'] = len(label_files)
                
                for label_file in tqdm(label_files, desc=f"Analyzing {split}"):
                    objects_in_image = []
                    classes_in_image = set()
                    
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:])
                                
                                area = width * height
                                aspect_ratio = width / height if height > 0 else 0
                                
                                split_stats['class_counts'][class_id] += 1
                                overall_stats['class_counts'][class_id] += 1
                                overall_stats['bbox_areas'].append(area)
                                overall_stats['aspect_ratios'].append(aspect_ratio)
                                
                                objects_in_image.append(class_id)
                                classes_in_image.add(class_id)
                                
                                split_stats['bbox_data'].append({
                                    'class_id': class_id,
                                    'area': area,
                                    'aspect_ratio': aspect_ratio,
                                    'width': width,
                                    'height': height
                                })
                    
                    overall_stats['objects_per_image'].append(len(objects_in_image))
                    overall_stats['classes_per_image'].append(len(classes_in_image))
            
            splits_data[split] = split_stats
        
        self.original_stats = {
            'splits': splits_data,
            'overall': overall_stats
        }
        
        return self.original_stats
    
    def calculate_class_weights(self) -> Dict[int, float]:
        """Calculate optimal class weights for loss function."""
        class_counts = self.original_stats['overall']['class_counts']
        total_samples = sum(class_counts.values())
        
        # Inverse frequency weighting with smoothing
        class_weights = {}
        for class_id in range(len(self.class_names)):
            count = class_counts.get(class_id, 1)
            weight = total_samples / (len(self.class_names) * count)
            class_weights[class_id] = min(weight, 10.0)  # Cap extreme weights
        
        return class_weights
    
    def create_augmentation_pipeline(self, class_id: int, intensity: str = 'medium') -> A.Compose:
        """Create class-specific augmentation pipeline with edge-safe transformations."""
        
        # Safe augmentations that don't risk edge objects
        safe_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.15, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ]
        
        # Edge-aware geometric transformations (reduced limits to preserve edge objects)
        if intensity == 'light':
            geometric_transforms = [
                A.Rotate(limit=15, p=0.4),  # Reduced rotation
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10, p=0.3),  # Minimal shifts
            ]
        elif intensity == 'medium':
            geometric_transforms = [
                A.Rotate(limit=25, p=0.5),  # Moderate rotation
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.4),  # Conservative shifts
                A.ElasticTransform(alpha=0.5, sigma=25, alpha_affine=25, p=0.2),  # Gentle elastic
            ]
        else:  # heavy - but still edge-conscious
            geometric_transforms = [
                A.Rotate(limit=30, p=0.6),  # Still limited rotation
                A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=25, p=0.5),  # Controlled shifts
                A.ElasticTransform(alpha=1, sigma=35, alpha_affine=35, p=0.3),  # Moderate elastic
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),  # Minimal distortion
                A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.2),  # Gentle grid distortion
            ]
        
        # Combine safe and geometric transforms
        all_transforms = safe_transforms + geometric_transforms
        
        # Add minority class specific augmentations
        if self.class_names[class_id] in ['starfish', 'stingray', 'puffin']:
            # Extra augmentations for rare classes but still edge-safe
            all_transforms.extend([
                A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=0.2),  # Small dropouts
                A.ToGray(p=0.05),  # Occasional grayscale
                A.Blur(blur_limit=3, p=0.2),  # Light blur
            ])
        
        return A.Compose(
            all_transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.5,  # Increased from 0.3 to 0.5 for better edge protection
                min_area=0.0002  # Minimum 0.02% of image area to keep tiny objects
            )
        )
    
    def oversample_minority_classes(self, split: str = 'train') -> List[Tuple[str, str]]:
        """Create oversampled dataset with balanced class distribution."""
        print(f"Oversampling minority classes for {split} split...")
        
        split_path = self.dataset_path / split
        labels_path = split_path / 'labels'
        images_path = split_path / 'images'
        
        # Group files by class
        class_files = defaultdict(list)
        
        for label_file in tqdm(labels_path.glob('*.txt')):
            image_file = images_path / (label_file.stem + '.jpg')
            if not image_file.exists():
                continue
            
            # Find dominant class in image
            class_counts = Counter()
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
            
            if class_counts:
                dominant_class = class_counts.most_common(1)[0][0]
                class_files[dominant_class].append((str(image_file), str(label_file)))
        
        # Calculate oversampling factors
        max_samples = max(len(files) for files in class_files.values())
        target_samples = min(max_samples, self.config['max_samples_per_class'])
        
        oversampled_files = []
        
        for class_id in range(len(self.class_names)):
            files = class_files[class_id]
            if not files:
                continue
                
            current_count = len(files)
            oversample_factor = max(1, target_samples // current_count)
            
            # Add original files
            oversampled_files.extend(files)
            
            # Add oversampled files
            additional_needed = target_samples - current_count
            if additional_needed > 0:
                additional_files = random.choices(files, k=additional_needed)
                oversampled_files.extend(additional_files)
            
            print(f"  {self.class_names[class_id]}: {current_count} → {len([f for f in oversampled_files if self.get_dominant_class(f[1]) == class_id])} samples")
        
        return oversampled_files
    
    def get_dominant_class(self, label_path: str) -> int:
        """Get the dominant class ID from a label file."""
        class_counts = Counter()
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
        
        if class_counts:
            return class_counts.most_common(1)[0][0]
        return 0
    
    def validate_edge_objects(self, bboxes: List, image_shape: Tuple[int, int], margin: float = 0.05) -> List:
        """
        Validate that objects near edges are properly handled.
        
        Args:
            bboxes: List of bounding boxes in YOLO format
            image_shape: (height, width) of image
            margin: Safety margin from edge (default 5% of image size)
        
        Returns:
            List of valid bounding boxes with edge safety checks
        """
        valid_bboxes = []
        
        for bbox, class_id in bboxes:
            x_center, y_center, width, height = bbox
            
            # Calculate actual bbox boundaries
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2
            
            # Check if bbox is mostly within image boundaries with margin
            if (x_min >= -margin and x_max <= 1 + margin and 
                y_min >= -margin and y_max <= 1 + margin):
                
                # Clamp bbox to image boundaries if slightly outside
                x_center = max(width/2, min(1 - width/2, x_center))
                y_center = max(height/2, min(1 - height/2, y_center))
                
                # Ensure minimum size after clamping
                if width >= 0.01 and height >= 0.01:  # At least 1% of image
                    valid_bboxes.append(([x_center, y_center, width, height], class_id))
        
        return valid_bboxes

    def apply_augmentation(self, image_path: str, label_path: str, class_id: int, intensity: str) -> Tuple[np.ndarray, List]:
        """Apply augmentation to image and bounding boxes."""
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load bounding boxes
        bboxes = []
        class_labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(cls_id)
        
        if not bboxes:
            return image, []
        
        # Apply augmentation
        transform = self.create_augmentation_pipeline(class_id, intensity)
        
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            # Additional edge safety validation
            if transformed['bboxes']:
                augmented_bboxes = list(zip(transformed['bboxes'], transformed['class_labels']))
                safe_bboxes = self.validate_edge_objects(augmented_bboxes, image.shape[:2])
                return transformed['image'], safe_bboxes
            else:
                return transformed['image'], []
                
        except Exception as e:
            print(f"Augmentation failed for {image_path}: {e}")
            # Return original if augmentation fails
            original_bboxes = list(zip(bboxes, class_labels))
            return image, original_bboxes
    
    def create_mosaic(self, image_paths: List[str], label_paths: List[str]) -> Tuple[np.ndarray, List]:
        """Create mosaic augmentation from 4 images."""
        
        if len(image_paths) != 4:
            raise ValueError("Mosaic requires exactly 4 images")
        
        # Load images and labels
        images = []
        all_bboxes = []
        all_labels = []
        
        for img_path, lbl_path in zip(image_paths, label_paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            img = cv2.resize(img, (512, 512))
            images.append(img)
            
            # Load bboxes
            bboxes = []
            labels = []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        bboxes.append([x_center, y_center, width, height])
                        labels.append(cls_id)
            
            all_bboxes.append(bboxes)
            all_labels.append(labels)
        
        # Create mosaic
        mosaic = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        # Place images in quadrants
        mosaic[0:512, 0:512] = images[0]      # Top-left
        mosaic[0:512, 512:1024] = images[1]   # Top-right
        mosaic[512:1024, 0:512] = images[2]   # Bottom-left
        mosaic[512:1024, 512:1024] = images[3] # Bottom-right
        
        # Adjust bounding boxes for mosaic
        final_bboxes = []
        final_labels = []
        
        offsets = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]  # x_offset, y_offset
        
        for i, (bboxes, labels, (x_off, y_off)) in enumerate(zip(all_bboxes, all_labels, offsets)):
            for bbox, label in zip(bboxes, labels):
                x_center, y_center, width, height = bbox
                
                # Scale and offset for mosaic
                new_x = (x_center * 0.5) + x_off
                new_y = (y_center * 0.5) + y_off
                new_w = width * 0.5
                new_h = height * 0.5
                
                # Check if bbox is still valid
                if new_w > 0.01 and new_h > 0.01:  # Minimum size threshold
                    final_bboxes.append([new_x, new_y, new_w, new_h])
                    final_labels.append(label)
        
        return mosaic, list(zip(final_bboxes, final_labels))
    
    def process_split(self, split: str, file_list: List[Tuple[str, str]]):
        """Process a dataset split with augmentation and balancing."""
        
        print(f"Processing {split} split...")
        
        output_images_dir = self.output_path / f"{split}_{self.config['output_suffix']}" / "images"
        output_labels_dir = self.output_path / f"{split}_{self.config['output_suffix']}" / "labels"
        
        # Determine augmentation intensity by class
        class_counts = Counter()
        for _, label_path in file_list:
            dominant_class = self.get_dominant_class(label_path)
            class_counts[dominant_class] += 1
        
        max_count = max(class_counts.values()) if class_counts else 1
        
        file_counter = 0
        processed_stats = Counter()
        
        for i, (image_path, label_path) in enumerate(tqdm(file_list, desc=f"Processing {split}")):
            
            # Determine processing strategy
            dominant_class = self.get_dominant_class(label_path)
            class_count = class_counts[dominant_class]
            
            # Determine augmentation intensity
            if class_count < max_count * 0.1:  # < 10% of max class
                intensity = 'heavy'
                num_augmentations = self.config['heavy_augmentation_count']
            elif class_count < max_count * 0.3:  # < 30% of max class
                intensity = 'medium'
                num_augmentations = self.config['medium_augmentation_count']
            else:
                intensity = 'light'
                num_augmentations = self.config['light_augmentation_count']
            
            # Process original image
            self.copy_file_pair(image_path, label_path, output_images_dir, output_labels_dir, file_counter)
            processed_stats[dominant_class] += 1
            file_counter += 1
            
            # Apply augmentations
            for aug_idx in range(num_augmentations):
                try:
                    aug_image, aug_annotations = self.apply_augmentation(
                        image_path, label_path, dominant_class, intensity
                    )
                    
                    # Save augmented files
                    aug_image_name = f"{Path(image_path).stem}_aug_{aug_idx}.jpg"
                    aug_label_name = f"{Path(label_path).stem}_aug_{aug_idx}.txt"
                    
                    # Save image
                    aug_image_path = output_images_dir / aug_image_name
                    cv2.imwrite(str(aug_image_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                    
                    # Save labels
                    aug_label_path = output_labels_dir / aug_label_name
                    with open(aug_label_path, 'w') as f:
                        for bbox, cls_id in aug_annotations:
                            x_center, y_center, width, height = bbox
                            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    processed_stats[dominant_class] += 1
                    file_counter += 1
                    
                except Exception as e:
                    print(f"Failed to process augmentation {aug_idx} for {image_path}: {e}")
            
            # Create mosaic augmentations for training split
            if split == 'train' and i % 4 == 3 and i >= 3:
                try:
                    # Use last 4 processed images for mosaic
                    mosaic_images = [file_list[i-3][0], file_list[i-2][0], file_list[i-1][0], image_path]
                    mosaic_labels = [file_list[i-3][1], file_list[i-2][1], file_list[i-1][1], label_path]
                    
                    mosaic_image, mosaic_annotations = self.create_mosaic(mosaic_images, mosaic_labels)
                    
                    # Save mosaic
                    mosaic_image_name = f"mosaic_{file_counter}.jpg"
                    mosaic_label_name = f"mosaic_{file_counter}.txt"
                    
                    mosaic_image_path = output_images_dir / mosaic_image_name
                    cv2.imwrite(str(mosaic_image_path), cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR))
                    
                    mosaic_label_path = output_labels_dir / mosaic_label_name
                    with open(mosaic_label_path, 'w') as f:
                        for bbox, cls_id in mosaic_annotations:
                            x_center, y_center, width, height = bbox
                            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    file_counter += 1
                    
                except Exception as e:
                    print(f"Failed to create mosaic at iteration {i}: {e}")
        
        print(f"  Processed {file_counter} files for {split}")
        return processed_stats
    
    def copy_file_pair(self, image_path: str, label_path: str, 
                      output_images_dir: Path, output_labels_dir: Path, counter: int):
        """Copy image and label file pair to output directory."""
        
        image_ext = Path(image_path).suffix
        new_image_name = f"{counter:06d}{image_ext}"
        new_label_name = f"{counter:06d}.txt"
        
        shutil.copy2(image_path, output_images_dir / new_image_name)
        shutil.copy2(label_path, output_labels_dir / new_label_name)
    
    def create_data_yaml(self):
        """Create data.yaml file for training frameworks."""
        
        data_config = {
            'path': str(self.output_path.absolute()),
            'train': f'train_{self.config["output_suffix"]}/images',
            'val': f'val_{self.config["output_suffix"]}/images',
            'test': f'test_{self.config["output_suffix"]}/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        output_file = self.output_path / f'data_{self.config["output_suffix"]}.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Created data configuration: {output_file}")
    
    def save_statistics(self):
        """Save preprocessing statistics and analysis."""
        
        stats = {
            'original': self.original_stats,
            'processed': self.processed_stats,
            'config': self.config,
            'class_names': self.class_names,
            'class_weights': self.calculate_class_weights()
        }
        
        # Save as JSON
        stats_file = self.output_path / 'statistics' / 'preprocessing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Save as pickle for Python objects
        pickle_file = self.output_path / 'statistics' / 'preprocessing_stats.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(stats, f)
        
        print(f"Saved statistics: {stats_file}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a human-readable summary report."""
        
        report_lines = [
            "=" * 80,
            "UNDERWATER DATASET PREPROCESSING REPORT",
            "=" * 80,
            "",
            "ORIGINAL DATASET STATISTICS:",
            "-" * 40,
        ]
        
        # Original class distribution
        original_counts = self.original_stats['overall']['class_counts']
        total_original = sum(original_counts.values())
        
        for class_id, class_name in enumerate(self.class_names):
            count = original_counts.get(class_id, 0)
            percentage = (count / total_original * 100) if total_original > 0 else 0
            report_lines.append(f"{class_name:>12}: {count:>6} samples ({percentage:>5.1f}%)")
        
        report_lines.extend([
            "",
            f"Total objects: {total_original}",
            f"Average objects per image: {np.mean(self.original_stats['overall']['objects_per_image']):.1f}",
            f"Images with multiple classes: {sum(1 for x in self.original_stats['overall']['classes_per_image'] if x > 1)} "
            f"({sum(1 for x in self.original_stats['overall']['classes_per_image'] if x > 1) / len(self.original_stats['overall']['classes_per_image']) * 100:.1f}%)",
            "",
            "PREPROCESSING CONFIGURATION:",
            "-" * 40,
            f"Output suffix: {self.config['output_suffix']}",
            f"Max samples per class: {self.config['max_samples_per_class']}",
            f"Heavy augmentation count: {self.config['heavy_augmentation_count']}",
            f"Medium augmentation count: {self.config['medium_augmentation_count']}",
            f"Light augmentation count: {self.config['light_augmentation_count']}",
            "",
            "RECOMMENDED TRAINING PARAMETERS:",
            "-" * 40,
        ])
        
        # Add class weights
        class_weights = self.calculate_class_weights()
        report_lines.append("Class weights for loss function:")
        for class_id, class_name in enumerate(self.class_names):
            weight = class_weights.get(class_id, 1.0)
            report_lines.append(f"  {class_name:>12}: {weight:.3f}")
        
        report_lines.extend([
            "",
            "Recommended loss function: Focal Loss with α=0.25, γ=2.0",
            "Recommended optimizer: AdamW with cosine annealing",
            "Recommended batch size: 16-32 (adjust based on GPU memory)",
            "Recommended learning rate: 1e-4 to 1e-3",
            "",
            "=" * 80
        ])
        
        # Save report
        report_file = self.output_path / 'statistics' / 'preprocessing_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Created summary report: {report_file}")
        
        # Print key statistics
        print("\nPREPROCESSING SUMMARY:")
        print(f"Original total objects: {total_original}")
        print(f"Class imbalance ratio (max:min): {max(original_counts.values())//min(original_counts.values()) if min(original_counts.values()) > 0 else 'inf'}")
        print(f"Output directory: {self.output_path}")
    
    def run_preprocessing(self):
        """Execute the complete preprocessing pipeline."""
        
        print("Starting Underwater Dataset Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Analyze original dataset
        self.analyze_dataset()
        
        # Step 2: Process training split with oversampling
        if (self.dataset_path / 'train').exists():
            train_files = self.oversample_minority_classes('train')
            self.processed_stats['train'] = self.process_split('train', train_files)
        
        # Step 3: Process validation split (no oversampling)
        if (self.dataset_path / 'valid').exists():
            valid_files = []
            labels_path = self.dataset_path / 'valid' / 'labels'
            images_path = self.dataset_path / 'valid' / 'images'
            
            for label_file in labels_path.glob('*.txt'):
                image_file = images_path / (label_file.stem + '.jpg')
                if image_file.exists():
                    valid_files.append((str(image_file), str(label_file)))
            
            self.processed_stats['val'] = self.process_split('val', valid_files)
        
        # Step 4: Process test split (no oversampling)
        if (self.dataset_path / 'test').exists():
            test_files = []
            labels_path = self.dataset_path / 'test' / 'labels'
            images_path = self.dataset_path / 'test' / 'images'
            
            for label_file in labels_path.glob('*.txt'):
                image_file = images_path / (label_file.stem + '.jpg')
                if image_file.exists():
                    test_files.append((str(image_file), str(label_file)))
            
            self.processed_stats['test'] = self.process_split('test', test_files)
        
        # Step 5: Create configuration files
        self.create_data_yaml()
        
        # Step 6: Save statistics and reports
        self.save_statistics()
        
        print("Preprocessing pipeline completed successfully!")
        print(f"Output saved to: {self.output_path}")


def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description="Preprocess Underwater Object Classification Dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to original dataset")
    parser.add_argument("--output", "-o", required=True, help="Path for preprocessed output")
    parser.add_argument("--config", "-c", help="Path to configuration file (JSON)")
    parser.add_argument("--suffix", "-s", default="balanced", help="Output suffix for processed data")
    
    args = parser.parse_args()
    
    # Default configuration
    default_config = {
        'output_suffix': args.suffix,
        'max_samples_per_class': 2000,
        'heavy_augmentation_count': 8,
        'medium_augmentation_count': 4,
        'light_augmentation_count': 1,
        'random_seed': 42
    }
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            default_config.update(custom_config)
    
    # Set random seeds for reproducibility
    random.seed(default_config['random_seed'])
    np.random.seed(default_config['random_seed'])
    
    # Initialize and run preprocessor
    preprocessor = UnderwaterDatasetPreprocessor(
        dataset_path=args.input,
        output_path=args.output,
        config=default_config
    )
    
    preprocessor.run_preprocessing()


if __name__ == "__main__":
    main()
